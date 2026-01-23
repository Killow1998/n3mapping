#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "n3mapping/graph_optimizer.h"
#include <random>
#include <cmath>

namespace n3mapping {
namespace test {

/**
 * @brief 属性测试：图优化位姿一致性
 * 
 * Feature: n3mapping-backend
 * Property 3: 图优化位姿一致性
 * Validates: Requirements 5.1, 5.3
 * 
 * Property: For any 因子图，添加里程计边后优化，相邻关键帧之间的相对位姿
 *          应与里程计测量值一致（在优化容差范围内）
 */
class GraphOptimizerPBT : public ::testing::Test {
protected:
    void SetUp() override {
        config_.optimization_iterations = 20;  // 更多迭代以确保收敛
        config_.prior_noise_position = 0.01;
        config_.prior_noise_rotation = 0.01;
        config_.odom_noise_position = 0.1;
        config_.odom_noise_rotation = 0.1;
        config_.loop_noise_position = 0.1;
        config_.loop_noise_rotation = 0.1;
    }

    // 生成随机位姿
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

    // 生成随机相对位姿（较小的移动）
    Eigen::Isometry3d generateRandomRelativePose(std::mt19937& rng) {
        std::uniform_real_distribution<double> pos_dist(-2.0, 2.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI / 4, M_PI / 4);
        
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(
            pos_dist(rng),
            pos_dist(rng),
            pos_dist(rng) * 0.1  // z 方向变化较小
        );
        
        Eigen::AngleAxisd yaw(angle_dist(rng), Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitch(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd roll(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitX());
        
        pose.rotate(yaw * pitch * roll);
        return pose;
    }

    // 创建信息矩阵
    Eigen::Matrix<double, 6, 6> createInformationMatrix(double pos_weight = 1.0, 
                                                         double rot_weight = 1.0) {
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
        info.block<3, 3>(0, 0) *= pos_weight;
        info.block<3, 3>(3, 3) *= rot_weight;
        return info;
    }

    // 计算相对位姿
    Eigen::Isometry3d computeRelativePose(const Eigen::Isometry3d& from, 
                                          const Eigen::Isometry3d& to) {
        return from.inverse() * to;
    }

    // 比较两个位姿是否接近
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

/**
 * @brief Property 3: 图优化位姿一致性
 * 
 * 测试策略：
 * 1. 生成随机的关键帧轨迹（随机数量的节点）
 * 2. 为每对相邻节点生成随机的相对位姿测量
 * 3. 构建因子图并优化
 * 4. 验证优化后相邻节点的相对位姿与测量值一致
 */
TEST_F(GraphOptimizerPBT, PoseConsistencyProperty) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr double POS_TOLERANCE = 0.15;   // 15cm 容差
    constexpr double ROT_TOLERANCE = 0.1;    // ~5.7度 容差
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_nodes_dist(3, 15);
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 1. 生成随机数量的节点
        int num_nodes = num_nodes_dist(rng);
        
        // 2. 创建优化器
        GraphOptimizer optimizer(config_);
        
        // 3. 添加先验因子（第一个节点）
        Eigen::Isometry3d first_pose = generateRandomPose(rng);
        optimizer.addPriorFactor(0, first_pose);
        
        // 4. 存储测量值
        std::vector<Eigen::Isometry3d> measurements;
        
        // 5. 添加里程计边
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
        
        // 6. 执行优化
        optimizer.incrementalOptimize();
        
        // 7. 获取优化后的位姿
        auto optimized_poses = optimizer.getOptimizedPoses();
        
        // 8. 验证相邻节点的相对位姿
        bool all_consistent = true;
        for (int i = 0; i < num_nodes - 1; ++i) {
            total_tests++;
            
            // 计算优化后的相对位姿
            Eigen::Isometry3d optimized_relative = computeRelativePose(
                optimized_poses[i], 
                optimized_poses[i + 1]
            );
            
            // 与测量值比较
            if (posesNear(optimized_relative, measurements[i], POS_TOLERANCE, ROT_TOLERANCE)) {
                passed_tests++;
            } else {
                all_consistent = false;
                
                // 输出调试信息（仅在失败时）
                double pos_error = (optimized_relative.translation() - measurements[i].translation()).norm();
                Eigen::Quaterniond q_opt(optimized_relative.rotation());
                Eigen::Quaterniond q_meas(measurements[i].rotation());
                double rot_error = q_opt.angularDistance(q_meas);
                
                if (iter < 5) {  // 只输出前几次失败
                    std::cerr << "Iteration " << iter << ", Edge " << i << "->" << (i+1) 
                              << ": pos_error=" << pos_error 
                              << ", rot_error=" << rot_error << std::endl;
                }
            }
        }
        
        // 每条边都应该一致
        EXPECT_TRUE(all_consistent) << "Iteration " << iter << " failed";
    }
    
    // 统计信息
    double success_rate = static_cast<double>(passed_tests) / total_tests * 100.0;
    std::cout << "Property Test Results:" << std::endl;
    std::cout << "  Total edge tests: " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Success rate: " << success_rate << "%" << std::endl;
    
    // 至少95%的边应该满足一致性
    EXPECT_GE(success_rate, 95.0) << "Success rate too low";
}

/**
 * @brief Property 3 变体：带噪声的位姿一致性
 * 
 * 测试在有噪声的情况下，优化器是否仍能保持合理的一致性
 */
TEST_F(GraphOptimizerPBT, PoseConsistencyWithNoise) {
    constexpr int NUM_ITERATIONS = 50;
    constexpr double POS_TOLERANCE = 0.2;    // 更宽松的容差
    constexpr double ROT_TOLERANCE = 0.15;
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_nodes_dist(5, 10);
    std::normal_distribution<double> noise_dist(0.0, 0.05);  // 5cm 标准差
    
    int passed_iterations = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int num_nodes = num_nodes_dist(rng);
        GraphOptimizer optimizer(config_);
        
        // 添加先验
        Eigen::Isometry3d first_pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, first_pose);
        
        // 存储真实的相对位姿
        std::vector<Eigen::Isometry3d> true_measurements;
        
        // 添加带噪声的里程计边
        for (int i = 0; i < num_nodes - 1; ++i) {
            Eigen::Isometry3d true_relative = generateRandomRelativePose(rng);
            true_measurements.push_back(true_relative);
            
            // 添加噪声
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
        
        // 验证：优化后的相对位姿应该接近带噪声的测量值
        bool iteration_passed = true;
        for (int i = 0; i < num_nodes - 1; ++i) {
            Eigen::Isometry3d optimized_relative = computeRelativePose(
                optimized_poses[i], 
                optimized_poses[i + 1]
            );
            
            // 与真实测量值比较（允许更大误差）
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
    
    // 至少80%的迭代应该通过（噪声情况下）
    EXPECT_GE(success_rate, 80.0) << "Success rate too low with noise";
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
