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
 * @brief MapSerializer 单元测试
 * 
 * 测试地图序列化和反序列化功能
 * Requirements: 7.2, 7.3, 8.2, 8.3
 */
class MapSerializerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化配置
        config_.map_save_path = "/tmp/n3mapping_test";
        config_.keyframe_distance_threshold = 1.0;
        config_.keyframe_angle_threshold = 0.5;
        config_.optimization_iterations = 10;
        config_.prior_noise_position = 0.01;
        config_.prior_noise_rotation = 0.01;
        config_.odom_noise_position = 0.1;
        config_.odom_noise_rotation = 0.1;
        config_.loop_noise_position = 0.1;
        config_.loop_noise_rotation = 0.1;
        
        // 创建测试目录
        std::filesystem::create_directories(config_.map_save_path);
    }

    void TearDown() override {
        // 清理测试文件
        try {
            std::filesystem::remove_all(config_.map_save_path);
        } catch (...) {
            // 忽略清理错误
        }
    }

    // 生成随机点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr generateRandomPointCloud(size_t num_points) {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->resize(num_points);
        
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> intensity_dist(0.0f, 255.0f);
        
        for (auto& pt : cloud->points) {
            pt.x = dist(rng);
            pt.y = dist(rng);
            pt.z = dist(rng) * 0.1f;  // z 方向变化较小
            pt.intensity = intensity_dist(rng);
        }
        
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = false;
        
        return cloud;
    }

    // 生成随机位姿
    Eigen::Isometry3d generateRandomPose() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> pos_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
        
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(pos_dist(rng), pos_dist(rng), 0.0);
        
        Eigen::AngleAxisd yaw(angle_dist(rng), Eigen::Vector3d::UnitZ());
        pose.rotate(yaw);
        
        return pose;
    }

    Config config_;
};

/**
 * @brief 测试基本的序列化和反序列化
 */
TEST_F(MapSerializerTest, BasicSaveAndLoad) {
    // 创建组件
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    
    // 添加一些关键帧
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    optimizer.addPriorFactor(0, pose);
    
    for (int i = 0; i < 5; ++i) {
        auto cloud = generateRandomPointCloud(1000);
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
    
    // 保存地图
    std::string map_file = config_.map_save_path + "/test_map.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    ASSERT_TRUE(std::filesystem::exists(map_file));
    
    // 加载地图
    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    
    // 验证关键帧数量
    EXPECT_EQ(kf_manager.size(), kf_manager_loaded.size());
    EXPECT_EQ(5, kf_manager_loaded.size());
    
    // 验证描述子数量
    EXPECT_EQ(loop_detector.size(), loop_detector_loaded.size());
    
    // 验证节点数量
    EXPECT_EQ(optimizer.getNumNodes(), optimizer_loaded.getNumNodes());
    
    // 验证边数量
    EXPECT_EQ(optimizer.getNumEdges(), optimizer_loaded.getNumEdges());
}

/**
 * @brief 测试关键帧数据的完整性
 */
TEST_F(MapSerializerTest, KeyframeDataIntegrity) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    
    // 添加关键帧
    auto cloud = generateRandomPointCloud(500);
    Eigen::Isometry3d pose = generateRandomPose();
    double timestamp = 123.456;
    
    int64_t kf_id = kf_manager.addKeyframe(timestamp, pose, cloud);
    auto descriptor = loop_detector.addDescriptor(kf_id, cloud);
    kf_manager.updateDescriptor(kf_id, descriptor);  // 更新关键帧的描述子
    optimizer.addPriorFactor(kf_id, pose);
    
    // 保存和加载
    std::string map_file = config_.map_save_path + "/integrity_test.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    
    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    
    // 验证关键帧数据
    auto kf_original = kf_manager.getKeyframe(kf_id);
    auto kf_loaded = kf_manager_loaded.getKeyframe(kf_id);
    
    ASSERT_NE(kf_original, nullptr);
    ASSERT_NE(kf_loaded, nullptr);
    
    // 验证 ID 和时间戳
    EXPECT_EQ(kf_original->id, kf_loaded->id);
    EXPECT_DOUBLE_EQ(kf_original->timestamp, kf_loaded->timestamp);
    
    // 验证位姿
    EXPECT_TRUE(kf_original->pose_odom.isApprox(kf_loaded->pose_odom, 1e-6));
    EXPECT_TRUE(kf_original->pose_optimized.isApprox(kf_loaded->pose_optimized, 1e-6));
    
    // 验证点云大小
    EXPECT_EQ(kf_original->cloud->size(), kf_loaded->cloud->size());
    
    // 验证描述子
    EXPECT_EQ(kf_original->sc_descriptor.rows(), kf_loaded->sc_descriptor.rows());
    EXPECT_EQ(kf_original->sc_descriptor.cols(), kf_loaded->sc_descriptor.cols());
}

/**
 * @brief 测试约束边的序列化
 */
TEST_F(MapSerializerTest, EdgeSerialization) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    
    // 添加节点和边
    for (int i = 0; i < 3; ++i) {
        auto cloud = generateRandomPointCloud(100);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = i * 1.0;
        
        int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
        auto descriptor = loop_detector.addDescriptor(kf_id, cloud);
        kf_manager.updateDescriptor(kf_id, descriptor);  // 更新关键帧的描述子
        
        if (i == 0) {
            optimizer.addPriorFactor(kf_id, pose);
        } else {
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
    
    // 添加回环边
    EdgeInfo loop_edge;
    loop_edge.from_id = 2;
    loop_edge.to_id = 0;
    loop_edge.measurement = Eigen::Isometry3d::Identity();
    loop_edge.measurement.translation().x() = -2.0;
    loop_edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 50.0;
    loop_edge.type = EdgeType::LOOP;
    optimizer.addLoopEdge(loop_edge);
    
    optimizer.incrementalOptimize();
    
    // 保存和加载
    std::string map_file = config_.map_save_path + "/edge_test.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    
    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    
    // 验证边数量
    auto edges_original = optimizer.getEdges();
    auto edges_loaded = optimizer_loaded.getEdges();
    
    EXPECT_EQ(edges_original.size(), edges_loaded.size());
    
    // 验证回环边存在
    bool has_loop_original = optimizer.hasLoopClosure();
    bool has_loop_loaded = optimizer_loaded.hasLoopClosure();
    EXPECT_EQ(has_loop_original, has_loop_loaded);
    EXPECT_TRUE(has_loop_loaded);
}

/**
 * @brief 测试空地图的处理
 */
TEST_F(MapSerializerTest, EmptyMap) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    
    // 保存空地图
    std::string map_file = config_.map_save_path + "/empty_map.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    
    // 加载空地图
    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    
    // 验证为空
    EXPECT_EQ(0, kf_manager_loaded.size());
    EXPECT_EQ(0, loop_detector_loaded.size());
    EXPECT_EQ(0, optimizer_loaded.getNumNodes());
}

/**
 * @brief 测试文件不存在的情况
 */
TEST_F(MapSerializerTest, FileNotFound) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    
    std::string non_existent_file = config_.map_save_path + "/non_existent.pbstream";
    EXPECT_FALSE(serializer.loadMap(non_existent_file, kf_manager, loop_detector, optimizer));
}

/**
 * @brief 测试全局地图保存
 */
TEST_F(MapSerializerTest, SaveGlobalMap) {
    KeyframeManager kf_manager(config_);
    MapSerializer serializer(config_);
    
    // 添加关键帧
    for (int i = 0; i < 3; ++i) {
        auto cloud = generateRandomPointCloud(500);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = i * 2.0;
        
        kf_manager.addKeyframe(i * 0.1, pose, cloud);
    }
    
    // 保存全局地图
    std::string pcd_file = config_.map_save_path + "/global_map.pcd";
    ASSERT_TRUE(serializer.saveGlobalMap(pcd_file, kf_manager, 0.1));
    ASSERT_TRUE(std::filesystem::exists(pcd_file));
    
    // 验证文件大小 > 0
    auto file_size = std::filesystem::file_size(pcd_file);
    EXPECT_GT(file_size, 0);
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_map_serializer");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
