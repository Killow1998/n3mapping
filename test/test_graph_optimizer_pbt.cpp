#include <gtest/gtest.h>
#include "n3mapping/graph_optimizer.h"
#include <random>
#include <cmath>

namespace n3mapping {
namespace test {

class GraphOptimizerPBT : public ::testing::Test {
protected:
    void SetUp() override {
      config_.optimization_iterations = 20;
      config_.prior_noise_position = 0.01;
      config_.prior_noise_rotation = 0.01;
      config_.odom_noise_position = 0.1;
      config_.odom_noise_rotation = 0.1;
      config_.loop_noise_position = 0.1;
      config_.loop_noise_rotation = 0.1;
    }

    Eigen::Isometry3d generateRandomPose(std::mt19937& rng) {
        std::uniform_real_distribution<double> pos_dist(-10.0, 10.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(
            pos_dist(rng),
            pos_dist(rng),
            pos_dist(rng)
        );

        Eigen::AngleAxisd yaw(angle_dist(rng), Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitch(angle_dist(rng) * 0.3, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd roll(angle_dist(rng) * 0.3, Eigen::Vector3d::UnitX());

        pose.rotate(yaw * pitch * roll);
        return pose;
    }

    Eigen::Isometry3d generateRandomRelativePose(std::mt19937& rng) {
        std::uniform_real_distribution<double> pos_dist(-2.0, 2.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI / 4, M_PI / 4);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() =
            Eigen::Vector3d(pos_dist(rng), pos_dist(rng), pos_dist(rng) * 0.1);

        Eigen::AngleAxisd yaw(angle_dist(rng), Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitch(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd roll(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitX());

        pose.rotate(yaw * pitch * roll);
        return pose;
    }

    Eigen::Matrix<double, 6, 6> createInformationMatrix(double pos_weight = 1.0,
                                                         double rot_weight = 1.0) {
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
        info.block<3, 3>(0, 0) *= pos_weight;
        info.block<3, 3>(3, 3) *= rot_weight;
        return info;
    }

    Eigen::Isometry3d computeRelativePose(const Eigen::Isometry3d& from,
                                          const Eigen::Isometry3d& to) {
        return from.inverse() * to;
    }

    bool posesNear(const Eigen::Isometry3d& p1, const Eigen::Isometry3d& p2,
                   double pos_tol, double rot_tol) {
        double pos_diff = (p1.translation() - p2.translation()).norm();
        Eigen::Quaterniond q1(p1.rotation());
        Eigen::Quaterniond q2(p2.rotation());
        double rot_diff = q1.angularDistance(q2);
        return pos_diff < pos_tol && rot_diff < rot_tol;
    }

    Config config_;
};

TEST_F(GraphOptimizerPBT, PoseConsistencyProperty) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr double POS_TOLERANCE = 0.15;
    constexpr double ROT_TOLERANCE = 0.1;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_nodes_dist(3, 15);

    int total_tests = 0;
    int passed_tests = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {

        int num_nodes = num_nodes_dist(rng);

        GraphOptimizer optimizer(config_);

        Eigen::Isometry3d first_pose = generateRandomPose(rng);
        optimizer.addPriorFactor(0, first_pose);

        std::vector<Eigen::Isometry3d> measurements;

        for (int i = 0; i < num_nodes - 1; ++i) {
            Eigen::Isometry3d relative_pose = generateRandomRelativePose(rng);
            measurements.push_back(relative_pose);

            EdgeInfo edge;
            edge.from_id = i;
            edge.to_id = i + 1;
            edge.measurement = relative_pose;
            edge.information = createInformationMatrix(100.0, 100.0);
            edge.type = EdgeType::ODOMETRY;

            optimizer.addOdometryEdge(edge);
        }

        optimizer.incrementalOptimize();

        auto optimized_poses = optimizer.getOptimizedPoses();

        bool all_consistent = true;
        for (int i = 0; i < num_nodes - 1; ++i) {
            total_tests++;

            Eigen::Isometry3d optimized_relative = computeRelativePose(
                optimized_poses[i],
                optimized_poses[i + 1]
            );

            if (posesNear(optimized_relative, measurements[i], POS_TOLERANCE, ROT_TOLERANCE)) {
                passed_tests++;
            } else {
                all_consistent = false;

                double pos_error = (optimized_relative.translation() - measurements[i].translation()).norm();
                Eigen::Quaterniond q_opt(optimized_relative.rotation());
                Eigen::Quaterniond q_meas(measurements[i].rotation());
                double rot_error = q_opt.angularDistance(q_meas);

                if (iter < 5) {
                  std::cerr << "Iteration " << iter << ", Edge " << i << "->"
                            << (i + 1) << ": pos_error=" << pos_error
                            << ", rot_error=" << rot_error << std::endl;
                }
            }
        }

        EXPECT_TRUE(all_consistent) << "Iteration " << iter << " failed";
    }

    double success_rate = static_cast<double>(passed_tests) / total_tests * 100.0;
    std::cout << "Property Test Results:" << std::endl;
    std::cout << "  Total edge tests: " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Success rate: " << success_rate << "%" << std::endl;

    EXPECT_GE(success_rate, 95.0) << "Success rate too low";
}

TEST_F(GraphOptimizerPBT, PoseConsistencyWithNoise) {
    constexpr int NUM_ITERATIONS = 50;
    constexpr double POS_TOLERANCE = 0.2;
    constexpr double ROT_TOLERANCE = 0.15;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_nodes_dist(5, 10);
    std::normal_distribution<double> noise_dist(0.0, 0.05);

    int passed_iterations = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int num_nodes = num_nodes_dist(rng);
        GraphOptimizer optimizer(config_);

        Eigen::Isometry3d first_pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, first_pose);

        std::vector<Eigen::Isometry3d> true_measurements;

        for (int i = 0; i < num_nodes - 1; ++i) {
            Eigen::Isometry3d true_relative = generateRandomRelativePose(rng);
            true_measurements.push_back(true_relative);

            Eigen::Isometry3d noisy_relative = true_relative;
            noisy_relative.translation() += Eigen::Vector3d(
                noise_dist(rng),
                noise_dist(rng),
                noise_dist(rng) * 0.5
            );

            EdgeInfo edge;
            edge.from_id = i;
            edge.to_id = i + 1;
            edge.measurement = noisy_relative;
            edge.information = createInformationMatrix(50.0, 50.0);
            edge.type = EdgeType::ODOMETRY;

            optimizer.addOdometryEdge(edge);
        }

        optimizer.incrementalOptimize();
        auto optimized_poses = optimizer.getOptimizedPoses();

        bool iteration_passed = true;
        for (int i = 0; i < num_nodes - 1; ++i) {
            Eigen::Isometry3d optimized_relative = computeRelativePose(
                optimized_poses[i],
                optimized_poses[i + 1]
            );

            if (!posesNear(optimized_relative, true_measurements[i], POS_TOLERANCE, ROT_TOLERANCE)) {
                iteration_passed = false;
                break;
            }
        }

        if (iteration_passed) {
            passed_iterations++;
        }
    }

    double success_rate = static_cast<double>(passed_iterations) / NUM_ITERATIONS * 100.0;
    std::cout << "Noisy Property Test Results:" << std::endl;
    std::cout << "  Iterations passed: " << passed_iterations << "/" << NUM_ITERATIONS << std::endl;
    std::cout << "  Success rate: " << success_rate << "%" << std::endl;

    EXPECT_GE(success_rate, 80.0) << "Success rate too low with noise";
}

}
}
