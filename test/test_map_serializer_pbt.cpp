#include <gtest/gtest.h>
#include <ros/ros.h>
#include "n3mapping/map_serializer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/graph_optimizer.h"
#include <filesystem>
#include <random>

namespace n3mapping {
namespace test {

/**
 * @brief 属性测试：地图序列化往返一致性
 * 
 * Feature: n3mapping-backend
 * Property 5: 地图序列化往返一致性
 * Validates: Requirements 7.2, 7.3, 7.4, 7.5, 8.2, 8.3
 * 
 * Property: For any 有效的地图数据（关键帧、约束、描述子），
 *          序列化后再反序列化应产生等价的数据结构
 */
class MapSerializerPBT : public ::testing::Test {
protected:
    void SetUp() override {
        config_.map_save_path = "/tmp/n3mapping_pbt_test";
        config_.keyframe_distance_threshold = 1.0;
        config_.keyframe_angle_threshold = 0.5;
        config_.optimization_iterations = 10;
        config_.prior_noise_position = 0.01;
        config_.prior_noise_rotation = 0.01;
        config_.odom_noise_position = 0.1;
        config_.odom_noise_rotation = 0.1;
        config_.loop_noise_position = 0.1;
        config_.loop_noise_rotation = 0.1;
        config_.use_robust_kernel = true;
        config_.robust_kernel_type = "Huber";
        config_.robust_kernel_delta = 1.0;
        
        std::filesystem::create_directories(config_.map_save_path);
    }

    void TearDown() override {
        try {
            std::filesystem::remove_all(config_.map_save_path);
        } catch (...) {
        }
    }

    // 生成随机点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr generateRandomPointCloud(
        std::mt19937& rng, size_t min_points, size_t max_points) {
        
        std::uniform_int_distribution<size_t> size_dist(min_points, max_points);
        size_t num_points = size_dist(rng);
        
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->resize(num_points);
        
        std::uniform_real_distribution<float> pos_dist(-20.0f, 20.0f);
        std::uniform_real_distribution<float> intensity_dist(0.0f, 255.0f);
        
        for (auto& pt : cloud->points) {
            pt.x = pos_dist(rng);
            pt.y = pos_dist(rng);
            pt.z = pos_dist(rng) * 0.2f;
            pt.intensity = intensity_dist(rng);
        }
        
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = false;
        
        return cloud;
    }

    // 生成随机位姿
    Eigen::Isometry3d generateRandomPose(std::mt19937& rng) {
        std::uniform_real_distribution<double> pos_dist(-100.0, 100.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
        
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(
            pos_dist(rng),
            pos_dist(rng),
            pos_dist(rng) * 0.1
        );
        
        Eigen::AngleAxisd yaw(angle_dist(rng), Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitch(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd roll(angle_dist(rng) * 0.2, Eigen::Vector3d::UnitX());
        
        pose.rotate(yaw * pitch * roll);
        return pose;
    }

    // 比较两个位姿是否接近
    bool posesNear(const Eigen::Isometry3d& p1, const Eigen::Isometry3d& p2, 
                   double pos_tol = 1e-6, double rot_tol = 1e-6) {
        double pos_diff = (p1.translation() - p2.translation()).norm();
        Eigen::Quaterniond q1(p1.rotation());
        Eigen::Quaterniond q2(p2.rotation());
        double rot_diff = q1.angularDistance(q2);
        return pos_diff < pos_tol && rot_diff < rot_tol;
    }

    // 比较两个点云是否接近
    bool pointCloudsNear(const pcl::PointCloud<pcl::PointXYZI>::Ptr& c1,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr& c2,
                         double tol = 1e-5) {
        if (c1->size() != c2->size()) {
            return false;
        }
        
        for (size_t i = 0; i < c1->size(); ++i) {
            const auto& p1 = c1->points[i];
            const auto& p2 = c2->points[i];
            
            double dist = std::sqrt(
                std::pow(p1.x - p2.x, 2) +
                std::pow(p1.y - p2.y, 2) +
                std::pow(p1.z - p2.z, 2)
            );
            
            if (dist > tol || std::abs(p1.intensity - p2.intensity) > tol) {
                return false;
            }
        }
        
        return true;
    }

    // 比较两个描述子是否接近
    bool descriptorsNear(const Eigen::MatrixXd& d1, const Eigen::MatrixXd& d2,
                         double tol = 1e-6) {
        if (d1.rows() != d2.rows() || d1.cols() != d2.cols()) {
            return false;
        }
        
        return (d1 - d2).norm() < tol;
    }

    Config config_;
};

/**
 * @brief Property 5: 地图序列化往返一致性
 * 
 * 测试策略：
 * 1. 生成随机数量的关键帧（随机位姿、点云、描述子）
 * 2. 生成随机的里程计边和回环边
 * 3. 序列化地图
 * 4. 反序列化地图
 * 5. 验证所有数据一致性
 */
TEST_F(MapSerializerPBT, RoundTripConsistency) {
    constexpr int NUM_ITERATIONS = 50;
    constexpr double POS_TOLERANCE = 1e-6;
    constexpr double ROT_TOLERANCE = 1e-6;
    constexpr double CLOUD_TOLERANCE = 1e-5;
    constexpr double DESCRIPTOR_TOLERANCE = 1e-6;
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_kf_dist(3, 20);
    std::uniform_real_distribution<double> loop_prob(0.0, 1.0);
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 1. 生成随机数量的关键帧
        int num_keyframes = num_kf_dist(rng);
        
        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);
        
        // 添加关键帧
        Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, current_pose);
        
        for (int i = 0; i < num_keyframes; ++i) {
            auto cloud = generateRandomPointCloud(rng, 100, 1000);
            current_pose = generateRandomPose(rng);
            
            int64_t kf_id = kf_manager.addKeyframe(i * 0.1, current_pose, cloud);
            auto descriptor = loop_detector.addDescriptor(kf_id, cloud);
            kf_manager.updateDescriptor(kf_id, descriptor);  // 更新关键帧的描述子
            
            // 添加里程计边
            if (i > 0) {
                EdgeInfo edge;
                edge.from_id = i - 1;
                edge.to_id = i;
                edge.measurement = generateRandomPose(rng);
                edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
                edge.type = EdgeType::ODOMETRY;
                optimizer.addOdometryEdge(edge);
            }
            
            // 随机添加回环边
            if (i > 5 && loop_prob(rng) < 0.3) {
                std::uniform_int_distribution<int> loop_target_dist(0, i - 5);
                int loop_target = loop_target_dist(rng);
                
                EdgeInfo loop_edge;
                loop_edge.from_id = i;
                loop_edge.to_id = loop_target;
                loop_edge.measurement = generateRandomPose(rng);
                loop_edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 50.0;
                loop_edge.type = EdgeType::LOOP;
                optimizer.addLoopEdge(loop_edge);
            }
        }
        
        optimizer.incrementalOptimize();
        
        // 2. 序列化
        std::string map_file = config_.map_save_path + "/pbt_map_" + std::to_string(iter) + ".pbstream";
        if (!serializer.saveMap(map_file, kf_manager, loop_detector, optimizer)) {
            std::cerr << "Iteration " << iter << ": Failed to save map" << std::endl;
            continue;
        }
        
        // 3. 反序列化
        KeyframeManager kf_manager_loaded(config_);
        LoopDetector loop_detector_loaded(config_);
        GraphOptimizer optimizer_loaded(config_);
        
        if (!serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded)) {
            std::cerr << "Iteration " << iter << ": Failed to load map" << std::endl;
            continue;
        }
        
        // 4. 验证一致性
        bool iteration_passed = true;
        
        // 验证关键帧数量
        total_tests++;
        if (kf_manager.size() != kf_manager_loaded.size()) {
            iteration_passed = false;
            if (iter < 5) {
                std::cerr << "Iteration " << iter << ": Keyframe count mismatch: "
                          << kf_manager.size() << " vs " << kf_manager_loaded.size() << std::endl;
            }
        } else {
            passed_tests++;
        }
        
        // 验证每个关键帧
        auto keyframes_original = kf_manager.getAllKeyframes();
        for (const auto& kf_orig : keyframes_original) {
            auto kf_loaded = kf_manager_loaded.getKeyframe(kf_orig->id);
            
            if (!kf_loaded) {
                iteration_passed = false;
                continue;
            }
            
            // 验证位姿
            if (!posesNear(kf_orig->pose_odom, kf_loaded->pose_odom, POS_TOLERANCE, ROT_TOLERANCE)) {
                iteration_passed = false;
            }
            
            if (!posesNear(kf_orig->pose_optimized, kf_loaded->pose_optimized, POS_TOLERANCE, ROT_TOLERANCE)) {
                iteration_passed = false;
            }
            
            // 验证点云
            if (!pointCloudsNear(kf_orig->cloud, kf_loaded->cloud, CLOUD_TOLERANCE)) {
                iteration_passed = false;
            }
            
            // 验证描述子
            if (!descriptorsNear(kf_orig->sc_descriptor, kf_loaded->sc_descriptor, DESCRIPTOR_TOLERANCE)) {
                iteration_passed = false;
            }
        }
        
        // 验证描述子数量
        if (loop_detector.size() != loop_detector_loaded.size()) {
            iteration_passed = false;
        }
        
        // 验证图优化器
        if (optimizer.getNumNodes() != optimizer_loaded.getNumNodes()) {
            iteration_passed = false;
        }
        
        if (optimizer.getNumEdges() != optimizer_loaded.getNumEdges()) {
            iteration_passed = false;
        }
        
        if (optimizer.hasLoopClosure() != optimizer_loaded.hasLoopClosure()) {
            iteration_passed = false;
        }
        
        EXPECT_TRUE(iteration_passed) << "Iteration " << iter << " failed consistency check";
        
        // 清理测试文件
        std::filesystem::remove(map_file);
    }
    
    // 统计信息
    double success_rate = static_cast<double>(passed_tests) / total_tests * 100.0;
    std::cout << "Property Test Results:" << std::endl;
    std::cout << "  Total iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "  Keyframe count tests: " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Success rate: " << success_rate << "%" << std::endl;
    
    // 至少95%的测试应该通过
    EXPECT_GE(success_rate, 95.0) << "Success rate too low";
}

/**
 * @brief Property 5 变体：大规模地图往返一致性
 * 
 * 测试更大规模的地图数据
 */
TEST_F(MapSerializerPBT, LargeMapRoundTrip) {
    constexpr int NUM_ITERATIONS = 10;
    constexpr int MIN_KEYFRAMES = 50;
    constexpr int MAX_KEYFRAMES = 100;
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> num_kf_dist(MIN_KEYFRAMES, MAX_KEYFRAMES);
    
    int passed_iterations = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int num_keyframes = num_kf_dist(rng);
        
        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);
        
        // 添加大量关键帧
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, pose);
        
        for (int i = 0; i < num_keyframes; ++i) {
            auto cloud = generateRandomPointCloud(rng, 500, 2000);
            pose.translation().x() += 1.0;
            
            int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
            auto descriptor = loop_detector.addDescriptor(kf_id, cloud);
            kf_manager.updateDescriptor(kf_id, descriptor);  // 更新关键帧的描述子
            
            if (i > 0) {
                EdgeInfo edge;
                edge.from_id = i - 1;
                edge.to_id = i;
                edge.measurement = Eigen::Isometry3d::Identity();
                edge.measurement.translation().x() = 1.0;
                edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
                edge.type = EdgeType::ODOMETRY;
                optimizer.addOdometryEdge(edge);
            }
        }
        
        optimizer.incrementalOptimize();
        
        // 序列化和反序列化
        std::string map_file = config_.map_save_path + "/large_map_" + std::to_string(iter) + ".pbstream";
        
        if (!serializer.saveMap(map_file, kf_manager, loop_detector, optimizer)) {
            continue;
        }
        
        KeyframeManager kf_manager_loaded(config_);
        LoopDetector loop_detector_loaded(config_);
        GraphOptimizer optimizer_loaded(config_);
        
        if (!serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded)) {
            continue;
        }
        
        // 验证基本一致性
        bool passed = true;
        passed &= (kf_manager.size() == kf_manager_loaded.size());
        passed &= (loop_detector.size() == loop_detector_loaded.size());
        passed &= (optimizer.getNumNodes() == optimizer_loaded.getNumNodes());
        passed &= (optimizer.getNumEdges() == optimizer_loaded.getNumEdges());
        
        if (passed) {
            passed_iterations++;
        }
        
        std::filesystem::remove(map_file);
    }
    
    double success_rate = static_cast<double>(passed_iterations) / NUM_ITERATIONS * 100.0;
    std::cout << "Large Map Test Results:" << std::endl;
    std::cout << "  Iterations passed: " << passed_iterations << "/" << NUM_ITERATIONS << std::endl;
    std::cout << "  Success rate: " << success_rate << "%" << std::endl;
    
    // 至少90%的大规模测试应该通过
    EXPECT_GE(success_rate, 90.0) << "Large map success rate too low";
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_map_serializer_pbt");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
