/**
 * @file test_point_cloud_matcher_pbt.cpp
 * @brief Property-Based Tests for PointCloudMatcher
 *
 * Feature: n3mapping-backend
 * Property 2: 点云配准对称性
 * Validates: Requirements 3.1, 3.2
 *
 * 对于任意两个点云 A 和 B，align(A, B) 的逆变换应近似等于 align(B, A) 的结果
 * （在数值误差范围内）。
 */

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <rclcpp/rclcpp.hpp>

#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
namespace test {

/**
 * @brief Property-Based Test fixture for PointCloudMatcher
 */
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

    /**
     * @brief 生成随机点云
     * @param rng 随机数生成器
     * @param num_points 点数范围
     * @param size 空间大小
     * @return 随机点云
     */
    Keyframe::PointCloudT::Ptr generateRandomCloud(std::mt19937& rng, size_t min_points = 500, size_t max_points = 2000, double size = 10.0)
    {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        size_t num_points = num_dist(rng);

        auto cloud = std::make_shared<Keyframe::PointCloudT>();
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

    /**
     * @brief 生成随机结构化点云 (平面 + 噪声)
     * 结构化点云更容易配准成功
     */
    Keyframe::PointCloudT::Ptr generateRandomStructuredCloud(std::mt19937& rng,
                                                             size_t min_points = 800,
                                                             size_t max_points = 1500,
                                                             double size = 10.0)
    {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        size_t num_points = num_dist(rng);

        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);

        std::uniform_real_distribution<float> pos_dist(-size / 2, size / 2);
        std::normal_distribution<float> noise_dist(0.0f, 0.05f);

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = pos_dist(rng);
            cloud->points[i].y = pos_dist(rng);
            // 创建波浪形表面以增加结构
            float base_z = 0.5f * std::sin(cloud->points[i].x * 0.5f) * std::cos(cloud->points[i].y * 0.5f);
            cloud->points[i].z = base_z + noise_dist(rng);
            cloud->points[i].intensity = 1.0f;
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    /**
     * @brief 生成随机变换
     * @param rng 随机数生成器
     * @param max_trans 最大平移
     * @param max_rot 最大旋转 (弧度)
     * @return 随机变换
     */
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

    /**
     * @brief 变换点云
     */
    Keyframe::PointCloudT::Ptr transformCloud(const Keyframe::PointCloudT::Ptr& cloud, const Eigen::Isometry3d& transform)
    {
        auto transformed = std::make_shared<Keyframe::PointCloudT>();
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

    /**
     * @brief 检查两个变换是否互为逆
     * T1 * T2 应该接近单位矩阵
     */
    bool areInverseTransforms(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2, double trans_tol = 1.0, double rot_tol = 0.5)
    {
        Eigen::Isometry3d composed = T1 * T2;

        // 平移误差
        double trans_error = composed.translation().norm();

        // 旋转误差
        Eigen::Matrix3d R = composed.rotation();
        double rot_error = std::acos(std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0)));

        return trans_error < trans_tol && rot_error < rot_tol;
    }

    Config config_;
    std::unique_ptr<PointCloudMatcher> matcher_;
};

/**
 * @brief Property 2: 点云配准对称性
 *
 * Feature: n3mapping-backend, Property 2: 点云配准对称性
 * Validates: Requirements 3.1, 3.2
 *
 * 对于任意两个点云 A 和 B:
 * - T_AB = align(A, B) 表示将 B 配准到 A 的变换
 * - T_BA = align(B, A) 表示将 A 配准到 B 的变换
 * - 则 T_AB ≈ T_BA^(-1)，即 T_AB * T_BA ≈ I
 */
TEST_F(PointCloudMatcherPBT, RegistrationSymmetryProperty)
{
    constexpr int NUM_ITERATIONS = 100;
    int successful_tests = 0;
    int failed_symmetry = 0;
    int failed_registration = 0;

    // 使用不同的随机种子进行多次测试
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::mt19937 rng(iter * 12345); // 可重复的随机种子

        // 生成基础点云
        auto base_cloud = generateRandomStructuredCloud(rng);

        // 生成随机变换
        Eigen::Isometry3d T_true = generateRandomTransform(rng, 0.2, 0.1);

        // 创建变换后的点云
        auto transformed_cloud = transformCloud(base_cloud, T_true);

        // 创建关键帧
        auto kf_A = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), base_cloud);
        auto kf_B = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), transformed_cloud);

        // 执行双向配准
        auto result_AB = matcher_->align(kf_A, kf_B); // B -> A
        auto result_BA = matcher_->align(kf_B, kf_A); // A -> B

        // 检查配准是否成功
        if (!result_AB.converged || !result_BA.converged) {
            failed_registration++;
            continue;
        }

        // 验证对称性: T_AB * T_BA ≈ I
        // T_AB 将 B 变换到 A，T_BA 将 A 变换到 B
        // 所以 T_AB * T_BA 应该接近单位矩阵
        if (areInverseTransforms(result_AB.T_target_source, result_BA.T_target_source)) {
            successful_tests++;
        } else {
            failed_symmetry++;

            // 输出失败案例的详细信息
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

    // 对于收敛的双向配准，至少 30% 应满足近似互逆。
    int total_successful_registrations = successful_tests + failed_symmetry;
    if (total_successful_registrations > 0) {
        double symmetry_rate = static_cast<double>(successful_tests) / total_successful_registrations;
        std::cout << "Symmetry rate: " << (symmetry_rate * 100) << "%" << std::endl;
        EXPECT_GE(symmetry_rate, 0.3) << "Symmetry property should hold for at least 30% of converged registrations";
    }

    // 至少应该有一些成功的测试
    EXPECT_GT(successful_tests, 0) << "At least some tests should pass the symmetry property";
}

} // namespace test
} // namespace n3mapping

int
main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
