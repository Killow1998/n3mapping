#include "n3mapping/pcl_compat.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
namespace test {

class PointCloudMatcherPBT : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_correspondence_distance = 2.0;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.3;
        config_.gicp_num_neighbors = 20;
        config_.num_threads = 4;

        matcher_ = std::make_unique<PointCloudMatcher>(config_);
    }

    void TearDown() override { matcher_.reset(); }

    Keyframe::PointCloudT::Ptr generateRandomCloud(std::mt19937& rng, size_t min_points = 500, size_t max_points = 2000, double size = 10.0)
    {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        size_t num_points = num_dist(rng);

        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);

        std::uniform_real_distribution<float> pos_dist(-size / 2, size / 2);

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = pos_dist(rng);
            cloud->points[i].y = pos_dist(rng);
            cloud->points[i].z = pos_dist(rng);
            cloud->points[i].intensity = 1.0f;
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Keyframe::PointCloudT::Ptr generateRandomStructuredCloud(std::mt19937& rng,
                                                             size_t min_points = 800,
                                                             size_t max_points = 1500,
                                                             double size = 10.0)
    {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        size_t num_points = num_dist(rng);

        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);

        std::uniform_real_distribution<float> pos_dist(-size / 2, size / 2);
        std::normal_distribution<float> noise_dist(0.0f, 0.05f);

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = pos_dist(rng);
            cloud->points[i].y = pos_dist(rng);

            float base_z = 0.5f * std::sin(cloud->points[i].x * 0.5f) * std::cos(cloud->points[i].y * 0.5f);
            cloud->points[i].z = base_z + noise_dist(rng);
            cloud->points[i].intensity = 1.0f;
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Eigen::Isometry3d generateRandomTransform(std::mt19937& rng, double max_trans = 1.0, double max_rot = 0.3)
    {
        std::uniform_real_distribution<double> trans_dist(-max_trans, max_trans);
        std::uniform_real_distribution<double> rot_dist(-max_rot, max_rot);

        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.translation() = Eigen::Vector3d(trans_dist(rng), trans_dist(rng), trans_dist(rng));

        Eigen::AngleAxisd roll(rot_dist(rng), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(rot_dist(rng), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(rot_dist(rng), Eigen::Vector3d::UnitZ());

        transform.rotate(yaw * pitch * roll);
        return transform;
    }

    Keyframe::PointCloudT::Ptr transformCloud(const Keyframe::PointCloudT::Ptr& cloud, const Eigen::Isometry3d& transform)
    {
        auto transformed = pcl::make_shared<Keyframe::PointCloudT>();
        transformed->points.resize(cloud->size());

        for (size_t i = 0; i < cloud->size(); ++i) {
            Eigen::Vector3d pt(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            Eigen::Vector3d pt_transformed = transform * pt;

            transformed->points[i].x = static_cast<float>(pt_transformed.x());
            transformed->points[i].y = static_cast<float>(pt_transformed.y());
            transformed->points[i].z = static_cast<float>(pt_transformed.z());
            transformed->points[i].intensity = cloud->points[i].intensity;
        }

        transformed->width = cloud->size();
        transformed->height = 1;
        transformed->is_dense = true;
        return transformed;
    }

    bool areInverseTransforms(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2, double trans_tol = 1.0, double rot_tol = 0.5)
    {
        Eigen::Isometry3d composed = T1 * T2;

        double trans_error = composed.translation().norm();

        Eigen::Matrix3d R = composed.rotation();
        double rot_error = std::acos(std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0)));

        return trans_error < trans_tol && rot_error < rot_tol;
    }

    Config config_;
    std::unique_ptr<PointCloudMatcher> matcher_;
};

TEST_F(PointCloudMatcherPBT, RegistrationSymmetryProperty)
{
    constexpr int NUM_ITERATIONS = 100;
    int successful_tests = 0;
    int failed_symmetry = 0;
    int failed_registration = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
      std::mt19937 rng(iter * 12345);

      auto base_cloud = generateRandomStructuredCloud(rng);

      Eigen::Isometry3d T_true = generateRandomTransform(rng, 0.2, 0.1);

      auto transformed_cloud = transformCloud(base_cloud, T_true);

      auto kf_A =
          Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), base_cloud);
      auto kf_B = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(),
                                   transformed_cloud);

      auto result_AB = matcher_->align(kf_A, kf_B);
      auto result_BA = matcher_->align(kf_B, kf_A);

      if (!result_AB.converged || !result_BA.converged) {
        failed_registration++;
        continue;
        }

        if (areInverseTransforms(result_AB.T_target_source, result_BA.T_target_source)) {
            successful_tests++;
        } else {
            failed_symmetry++;

            Eigen::Isometry3d composed = result_AB.T_target_source * result_BA.T_target_source;
            std::cout << "Iteration " << iter << " failed symmetry check:" << std::endl;
            std::cout << "  T_AB translation: " << result_AB.T_target_source.translation().transpose() << std::endl;
            std::cout << "  T_BA translation: " << result_BA.T_target_source.translation().transpose() << std::endl;
            std::cout << "  Composed translation error: " << composed.translation().norm() << std::endl;
        }
    }

    std::cout << "\n=== Property Test Summary ===" << std::endl;
    std::cout << "Total iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Successful symmetry tests: " << successful_tests << std::endl;
    std::cout << "Failed symmetry: " << failed_symmetry << std::endl;
    std::cout << "Failed registration: " << failed_registration << std::endl;

    int total_successful_registrations = successful_tests + failed_symmetry;
    if (total_successful_registrations > 0) {
        double symmetry_rate = static_cast<double>(successful_tests) / total_successful_registrations;
        std::cout << "Symmetry rate: " << (symmetry_rate * 100) << "%" << std::endl;
        EXPECT_GE(symmetry_rate, 0.3) << "Symmetry property should hold for at least 30% of converged registrations";
    }

    EXPECT_GT(successful_tests, 0) << "At least some tests should pass the symmetry property";
}

}
}
