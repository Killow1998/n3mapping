#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "n3mapping/map_serializer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/n3map_nav_resource_reader.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3map.pb.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include "n3mapping/pcl_compat.h"
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
        config_.global_map_publish_hz = 1.0;
        
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
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
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

    n3mapping::KeyframeProto* addValidKeyframeProto(n3mapping::N3Map* map_proto,
                                                    int64_t id,
                                                    double timestamp = 1.0) {
        auto* kf = map_proto->add_keyframes();
        kf->set_id(id);
        kf->set_timestamp(timestamp);
        kf->mutable_pose_odom()->set_qw(1.0);
        kf->mutable_pose_optimized()->set_qw(1.0);
        auto* cloud = kf->mutable_cloud();
        cloud->set_num_points(1);
        cloud->add_points(1.0f + static_cast<float>(id));
        cloud->add_points(0.0f);
        cloud->add_points(0.0f);
        cloud->add_points(1.0f);
        return kf;
    }

    void addUpperTriangularInformation(n3mapping::InformationMatrix* info) {
        for (int r = 0; r < 6; ++r) {
            for (int c = r; c < 6; ++c) {
                info->add_values(r == c ? 1.0 : 0.0);
            }
        }
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

TEST_F(MapSerializerTest, KeyframeManagerSwapWithExchangesState) {
    KeyframeManager first(config_);
    KeyframeManager second(config_);
    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t first_id = first.addKeyframe(1.0, pose, generateRandomPointCloud(8));
    Eigen::Isometry3d second_pose = Eigen::Isometry3d::Identity();
    second_pose.translation().x() = 5.0;
    const int64_t second_id = second.addKeyframe(2.0, second_pose, generateRandomPointCloud(8));

    first.swapWith(second);

    EXPECT_EQ(first.size(), 1U);
    EXPECT_EQ(second.size(), 1U);
    EXPECT_NE(first.getKeyframe(second_id), nullptr);
    EXPECT_NE(second.getKeyframe(first_id), nullptr);
    EXPECT_NEAR(first.getKeyframe(second_id)->pose_optimized.translation().x(), 5.0, 1e-9);
}

TEST_F(MapSerializerTest, LoopDetectorSwapWithExchangesDescriptorsAndRhpd) {
    LoopDetector first(config_);
    LoopDetector second(config_);
    auto first_cloud = generateRandomPointCloud(64);
    auto second_cloud = generateRandomPointCloud(64);
    first.addDescriptor(1, first_cloud);
    first.addRHPD(1, first_cloud);
    second.addDescriptor(2, second_cloud);
    second.addRHPD(2, second_cloud);

    first.swapWith(second);

    EXPECT_EQ(first.size(), 1U);
    EXPECT_EQ(second.size(), 1U);
    EXPECT_EQ(first.getDescriptor(2).size(), first.getDescriptorDimensions().first * first.getDescriptorDimensions().second);
    EXPECT_EQ(second.getDescriptor(1).size(), second.getDescriptorDimensions().first * second.getDescriptorDimensions().second);
    Eigen::VectorXd rhpd;
    EXPECT_TRUE(first.getRHPDManager().get(2, &rhpd));
    EXPECT_TRUE(second.getRHPDManager().get(1, &rhpd));
}

TEST_F(MapSerializerTest, GraphOptimizerSwapWithExchangesGraphState) {
    GraphOptimizer first(config_);
    GraphOptimizer second(config_);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    first.addPriorFactor(1, pose);
    pose.translation().x() = 3.0;
    second.addPriorFactor(2, pose);

    first.swapWith(second);

    EXPECT_TRUE(first.hasNode(2));
    EXPECT_FALSE(first.hasNode(1));
    EXPECT_TRUE(second.hasNode(1));
    EXPECT_FALSE(second.hasNode(2));
    EXPECT_NEAR(first.getOptimizedPose(2).translation().x(), 3.0, 1e-9);
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

TEST_F(MapSerializerTest, EmptyMapIsRejected) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    std::string map_file = config_.map_save_path + "/empty_map.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));

    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    EXPECT_EQ(kf_manager_loaded.size(), 0U);
    EXPECT_EQ(loop_detector_loaded.size(), 0U);
    EXPECT_EQ(optimizer_loaded.getNumNodes(), 0U);
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

TEST_F(MapSerializerTest, MalformedCloudRejectsKeyframe) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(42);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(2);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/malformed_cloud.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
    EXPECT_EQ(loaded_optimizer.getNumNodes(), 0);
}

TEST_F(MapSerializerTest, StrictLoadRejectsMalformedKeyframeEvenWithValidRemaining) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);
    addValidKeyframeProto(&map_proto, 0);

    auto* malformed = map_proto.add_keyframes();
    malformed->set_id(1);
    malformed->set_timestamp(2.0);
    malformed->mutable_pose_odom()->set_qw(1.0);
    malformed->mutable_pose_optimized()->set_qw(1.0);
    malformed->mutable_cloud()->set_num_points(2);
    malformed->mutable_cloud()->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/strict_malformed_with_valid.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, SalvageLoadSkipsMalformedKeyframeAndLoadsValidRemaining) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);
    addValidKeyframeProto(&map_proto, 0);

    auto* malformed = map_proto.add_keyframes();
    malformed->set_id(1);
    malformed->set_timestamp(2.0);
    malformed->mutable_pose_odom()->set_qw(1.0);
    malformed->mutable_pose_optimized()->set_qw(1.0);
    malformed->mutable_cloud()->set_num_points(2);
    malformed->mutable_cloud()->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/salvage_malformed_with_valid.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    PbstreamLoadOptions options;
    options.policy = PbstreamLoadPolicy::SALVAGE;
    ASSERT_TRUE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer, options));
    EXPECT_EQ(loaded_kfs.size(), 1U);
    EXPECT_NE(loaded_kfs.getKeyframe(0), nullptr);
    EXPECT_EQ(loaded_optimizer.getNumNodes(), 1U);
}

TEST_F(MapSerializerTest, StrictLoadRejectsMissingEdgeEndpoint) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);

    addValidKeyframeProto(&map_proto, 0);

    auto* edge = map_proto.add_edges();
    edge->set_from_id(0);
    edge->set_to_id(99);
    edge->mutable_measurement()->set_qw(1.0);
    addUpperTriangularInformation(edge->mutable_information());
    edge->set_type(n3mapping::EdgeProto::ODOMETRY);

    const std::string map_file = config_.map_save_path + "/missing_edge_endpoint.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
    EXPECT_EQ(loaded_optimizer.getNumNodes(), 0U);
}

TEST_F(MapSerializerTest, ReadN3MapProtoRejectsOversizedFile) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    addValidKeyframeProto(&map_proto, 0);

    const std::string map_file = config_.map_save_path + "/resource_file_limit.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    PbstreamResourceLimits limits = defaultPbstreamResourceLimits();
    limits.max_file_bytes = 1;
    n3mapping::N3Map parsed;
    std::string error;
    EXPECT_FALSE(readN3MapProtoFromFile(map_file, limits, &parsed, &error));
    EXPECT_NE(error.find("file size"), std::string::npos);
}

TEST_F(MapSerializerTest, ReadN3MapProtoRejectsTooManyKeyframes) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    addValidKeyframeProto(&map_proto, 0);
    addValidKeyframeProto(&map_proto, 1);

    const std::string map_file = config_.map_save_path + "/resource_keyframe_limit.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    PbstreamResourceLimits limits = defaultPbstreamResourceLimits();
    limits.max_keyframes = 1;
    n3mapping::N3Map parsed;
    std::string error;
    EXPECT_FALSE(readN3MapProtoFromFile(map_file, limits, &parsed, &error));
    EXPECT_NE(error.find("too many keyframes"), std::string::npos);
}

TEST_F(MapSerializerTest, ReadN3MapProtoCountsEncodedPointValuesForLimits) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    auto* keyframe = addValidKeyframeProto(&map_proto, 0);
    keyframe->mutable_cloud()->set_num_points(0);
    keyframe->mutable_cloud()->add_points(2.0f);
    keyframe->mutable_cloud()->add_points(0.0f);
    keyframe->mutable_cloud()->add_points(0.0f);
    keyframe->mutable_cloud()->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/resource_encoded_point_limit.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    PbstreamResourceLimits limits = defaultPbstreamResourceLimits();
    limits.max_total_points = 0;
    n3mapping::N3Map parsed;
    std::string error;
    EXPECT_FALSE(readN3MapProtoFromFile(map_file, limits, &parsed, &error));
    EXPECT_NE(error.find("invalid pbstream resource limits"), std::string::npos);

    limits.max_total_points = 1;
    EXPECT_FALSE(readN3MapProtoFromFile(map_file, limits, &parsed, &error));
    EXPECT_NE(error.find("too many total points"), std::string::npos);
}

TEST_F(MapSerializerTest, StrictLoadRejectsNonPositiveDefiniteInformationMatrix) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);
    addValidKeyframeProto(&map_proto, 0);
    addValidKeyframeProto(&map_proto, 1);

    auto* edge = map_proto.add_edges();
    edge->set_from_id(0);
    edge->set_to_id(1);
    edge->mutable_measurement()->set_qw(1.0);
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            edge->mutable_information()->add_values((r == c && r != 5) ? 1.0 : 0.0);
        }
    }
    edge->set_type(n3mapping::EdgeProto::ODOMETRY);

    const std::string map_file = config_.map_save_path + "/non_spd_information.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
}

TEST_F(MapSerializerTest, SalvageLoadSkipsMissingEdgeEndpoint) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);
    addValidKeyframeProto(&map_proto, 0);

    auto* edge = map_proto.add_edges();
    edge->set_from_id(0);
    edge->set_to_id(99);
    edge->mutable_measurement()->set_qw(1.0);
    addUpperTriangularInformation(edge->mutable_information());
    edge->set_type(n3mapping::EdgeProto::ODOMETRY);

    const std::string map_file = config_.map_save_path + "/salvage_missing_edge_endpoint.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    PbstreamLoadOptions options;
    options.policy = PbstreamLoadPolicy::SALVAGE;
    ASSERT_TRUE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer, options));
    EXPECT_EQ(loaded_kfs.size(), 1U);
    EXPECT_EQ(loaded_optimizer.getNumNodes(), 1U);
    EXPECT_EQ(loaded_optimizer.getNumEdges(), 0U);
}

TEST_F(MapSerializerTest, NonFiniteKeyframeTimestampRejectsMapOnLoad) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(0);
    kf->set_timestamp(std::numeric_limits<double>::infinity());
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/nonfinite_timestamp_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, DuplicateKeyframeIdsRejectMapOnLoad) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);

    for (int i = 0; i < 2; ++i) {
        auto* kf = map_proto.add_keyframes();
        kf->set_id(0);
        kf->set_timestamp(1.0 + static_cast<double>(i));
        kf->mutable_pose_odom()->set_qw(1.0);
        kf->mutable_pose_optimized()->set_qw(1.0);
        auto* cloud = kf->mutable_cloud();
        cloud->set_num_points(1);
        cloud->add_points(1.0f + static_cast<float>(i));
        cloud->add_points(0.0f);
        cloud->add_points(0.0f);
        cloud->add_points(1.0f);
    }

    const std::string map_file = config_.map_save_path + "/duplicate_keyframe_ids_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, DuplicateKeyframeIdsRejectMapInSalvageMode) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);
    addValidKeyframeProto(&map_proto, 0, 1.0);
    addValidKeyframeProto(&map_proto, 0, 2.0);

    const std::string map_file = config_.map_save_path + "/duplicate_keyframe_ids_salvage.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    PbstreamLoadOptions options;
    options.policy = PbstreamLoadPolicy::SALVAGE;
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer, options));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, NegativeKeyframeIdRejectsMapOnLoad) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(-1);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    const std::string map_file = config_.map_save_path + "/negative_keyframe_id_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, AtomicSaveFailureDoesNotClobberExistingMap) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(0.0, pose, generateRandomPointCloud(64));
    kf_manager.updateDescriptor(kf_id, loop_detector.addDescriptor(kf_id, kf_manager.getKeyframe(kf_id)->cloud));
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/atomic.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    const auto original_size = std::filesystem::file_size(map_file);
    ASSERT_GT(original_size, 0U);

    const std::filesystem::path tmp_path = map_file + ".tmp";
    std::filesystem::create_directories(tmp_path);
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_TRUE(std::filesystem::exists(map_file));
    EXPECT_EQ(std::filesystem::file_size(map_file), original_size);

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    EXPECT_TRUE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 1U);
    std::filesystem::remove_all(tmp_path);
}

TEST_F(MapSerializerTest, SaveMapRejectsEmptyKeyframeCloud) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    auto empty_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, empty_cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/empty_cloud_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsNegativeKeyframeId) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    auto keyframe = std::make_shared<Keyframe>();
    keyframe->id = -1;
    keyframe->timestamp = 1.0;
    keyframe->pose_odom = Eigen::Isometry3d::Identity();
    keyframe->pose_optimized = Eigen::Isometry3d::Identity();
    keyframe->cloud = generateRandomPointCloud(16);
    kf_manager.loadKeyframes({keyframe});

    const std::string map_file = config_.map_save_path + "/negative_keyframe_id_save.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsAllNonFiniteKeyframeCloud) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (int i = 0; i < 4; ++i) {
        pcl::PointXYZI point;
        point.x = std::numeric_limits<float>::quiet_NaN();
        point.y = std::numeric_limits<float>::infinity();
        point.z = 0.0f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/nonfinite_cloud_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsMixedNonFiniteKeyframeCloud) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointXYZI finite;
    finite.x = 1.0f;
    finite.y = 0.0f;
    finite.z = 0.0f;
    finite.intensity = 1.0f;
    cloud->push_back(finite);
    pcl::PointXYZI nonfinite = finite;
    nonfinite.x = std::numeric_limits<float>::quiet_NaN();
    cloud->push_back(nonfinite);
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/mixed_nonfinite_cloud_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsNonFiniteKeyframePose) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, generateRandomPointCloud(16));
    auto kf = kf_manager.getKeyframe(kf_id);
    ASSERT_NE(kf, nullptr);
    kf->pose_optimized.translation().x() = std::numeric_limits<double>::quiet_NaN();
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/nonfinite_keyframe_pose_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsNonFiniteKeyframeTimestamp) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(
        std::numeric_limits<double>::infinity(), pose, generateRandomPointCloud(16));
    optimizer.addPriorFactor(kf_id, pose);

    const std::string map_file = config_.map_save_path + "/nonfinite_keyframe_timestamp_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapDoesNotPersistRolledBackNonPositiveDefiniteEdge) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf0 = kf_manager.addKeyframe(1.0, pose, generateRandomPointCloud(16));
    Eigen::Isometry3d pose1 = pose;
    pose1.translation().x() = 1.0;
    const int64_t kf1 = kf_manager.addKeyframe(2.0, pose1, generateRandomPointCloud(16));
    optimizer.addPriorFactor(kf0, pose);

    EdgeInfo edge;
    edge.from_id = kf0;
    edge.to_id = kf1;
    edge.measurement = pose.inverse() * pose1;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information(5, 5) = 0.0;
    edge.type = EdgeType::ODOMETRY;
    optimizer.addOdometryEdge(edge);
    optimizer.incrementalOptimize();
    EXPECT_EQ(optimizer.getNumEdges(), 0U);

    const std::string map_file = config_.map_save_path + "/non_spd_save_rejected.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    N3Map map_proto;
    std::ifstream ifs(map_file, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    ASSERT_TRUE(map_proto.ParseFromIstream(&ifs));
    EXPECT_EQ(map_proto.edges_size(), 0);
}

TEST_F(MapSerializerTest, SaveMapRejectsEdgeWithMissingEndpoint) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, generateRandomPointCloud(16));
    optimizer.addPriorFactor(kf_id, pose);

    EdgeInfo edge;
    edge.from_id = kf_id;
    edge.to_id = 99;
    edge.measurement = Eigen::Isometry3d::Identity();
    edge.measurement.translation().x() = 1.0;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.type = EdgeType::ODOMETRY;
    optimizer.addOdometryEdge(edge);
    optimizer.incrementalOptimize();

    const std::string map_file = config_.map_save_path + "/missing_save_edge_endpoint_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapDoesNotPersistRolledBackFailedGraphEdge) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    Eigen::Isometry3d pose0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
    pose1.translation().x() = 1.0;

    const int64_t kf0 = kf_manager.addKeyframe(1.0, pose0, generateRandomPointCloud(64));
    const int64_t kf1 = kf_manager.addKeyframe(2.0, pose1, generateRandomPointCloud(64));
    optimizer.addPriorFactor(kf0, pose0);
    optimizer.addOdometryEdge({kf0, kf1, pose0.inverse() * pose1,
                               Eigen::Matrix<double, 6, 6>::Identity(), EdgeType::ODOMETRY});
    optimizer.incrementalOptimize();
    ASSERT_EQ(optimizer.getNumEdges(), 1U);

    EdgeInfo bad_loop;
    bad_loop.from_id = kf1;
    bad_loop.to_id = 999;
    bad_loop.measurement = Eigen::Isometry3d::Identity();
    bad_loop.information = Eigen::Matrix<double, 6, 6>::Identity();
    bad_loop.type = EdgeType::LOOP;
    optimizer.addLoopEdge(bad_loop);
    EXPECT_FALSE(optimizer.incrementalOptimize());

    ASSERT_EQ(optimizer.getNumEdges(), 1U);
    ASSERT_FALSE(optimizer.hasNode(999));

    const std::string map_file = config_.map_save_path + "/rolled_back_edge_not_saved.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    N3Map map_proto;
    std::ifstream ifs(map_file, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    ASSERT_TRUE(map_proto.ParseFromIstream(&ifs));
    ASSERT_EQ(map_proto.edges_size(), 1);
    EXPECT_EQ(map_proto.edges(0).to_id(), kf1);
    for (const auto& edge_proto : map_proto.edges()) {
        EXPECT_NE(edge_proto.to_id(), 999);
        EXPECT_NE(edge_proto.from_id(), 999);
    }
}

TEST_F(MapSerializerTest, SaveMapRejectsNonFiniteDenseTrajectoryPose) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, generateRandomPointCloud(16));
    optimizer.addPriorFactor(kf_id, pose);

    std::vector<core::DenseTrajectoryPose> dense(1);
    dense[0].seq = 0;
    dense[0].timestamp = 1.0;
    dense[0].pose_world_lidar = Eigen::Isometry3d::Identity();
    dense[0].pose_world_lidar.translation().z() = std::numeric_limits<double>::quiet_NaN();

    const std::string map_file = config_.map_save_path + "/nonfinite_dense_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer, dense));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, SaveMapRejectsNonEmptyDenseTrajectoryWithNoneSource) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = kf_manager.addKeyframe(1.0, pose, generateRandomPointCloud(16));
    optimizer.addPriorFactor(kf_id, pose);

    std::vector<core::DenseTrajectoryPose> dense(1);
    dense[0].seq = 0;
    dense[0].timestamp = 1.0;
    dense[0].pose_world_lidar = Eigen::Isometry3d::Identity();
    core::DenseTrajectoryMetadata metadata;
    metadata.source = "none";
    metadata.degraded = true;

    const std::string map_file = config_.map_save_path + "/dense_none_source_rejected.pbstream";
    EXPECT_FALSE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer, dense, metadata));
    EXPECT_FALSE(std::filesystem::exists(map_file));
}

TEST_F(MapSerializerTest, LoadMapRejectsDenseTrajectoryWithNoneSource) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);
    map_proto.mutable_metadata()->set_dense_trajectory_source("none");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(true);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(0);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(1.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/dense_none_source_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, RoundTripDenseOptimizedTrajectory) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    Eigen::Isometry3d pose0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
    pose1.translation().x() = 1.5;

    const int64_t kf0 = kf_manager.addKeyframe(1.0, pose0, generateRandomPointCloud(120));
    const int64_t kf1 = kf_manager.addKeyframe(2.0, pose1, generateRandomPointCloud(120));
    kf_manager.updateDescriptor(kf0, loop_detector.addDescriptor(kf0, kf_manager.getKeyframe(kf0)->cloud));
    kf_manager.updateDescriptor(kf1, loop_detector.addDescriptor(kf1, kf_manager.getKeyframe(kf1)->cloud));
    optimizer.addPriorFactor(kf0, pose0);
    EdgeInfo edge;
    edge.from_id = kf0;
    edge.to_id = kf1;
    edge.measurement = pose0.inverse() * pose1;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.type = EdgeType::ODOMETRY;
    optimizer.addOdometryEdge(edge);

    std::vector<core::DenseTrajectoryPose> dense;
    for (int i = 0; i < 4; ++i) {
        core::DenseTrajectoryPose sample;
        sample.seq = static_cast<uint64_t>(i);
        sample.timestamp = 10.0 + static_cast<double>(i) * 0.1;
        sample.pose_world_lidar = Eigen::Isometry3d::Identity();
        sample.pose_world_lidar.translation().x() = static_cast<double>(i) * 0.25;
        dense.push_back(sample);
    }

    const std::string map_file = config_.map_save_path + "/dense_roundtrip.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer, dense));

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense;
    ASSERT_TRUE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer, &loaded_dense));

    ASSERT_EQ(loaded_dense.size(), dense.size());
    for (std::size_t i = 0; i < dense.size(); ++i) {
        EXPECT_EQ(loaded_dense[i].seq, dense[i].seq);
        EXPECT_DOUBLE_EQ(loaded_dense[i].timestamp, dense[i].timestamp);
        EXPECT_TRUE(loaded_dense[i].pose_world_lidar.isApprox(dense[i].pose_world_lidar, 1e-9));
    }
}

TEST_F(MapSerializerTest, LoadMapRejectsMalformedDenseTrajectoryPose) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(0);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(1.0);
    dense->mutable_pose_world_lidar()->set_qw(std::numeric_limits<double>::quiet_NaN());

    const std::string map_file = config_.map_save_path + "/malformed_dense_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense;
    core::DenseTrajectoryPose sentinel_dense;
    sentinel_dense.seq = 99;
    sentinel_dense.timestamp = 42.0;
    loaded_dense.push_back(sentinel_dense);
    core::DenseTrajectoryMetadata loaded_metadata;
    loaded_metadata.source = "existing";
    loaded_metadata.degraded = false;
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer,
                                    &loaded_dense, &loaded_metadata));
    EXPECT_EQ(loaded_kfs.size(), 0U);
    ASSERT_EQ(loaded_dense.size(), 1U);
    EXPECT_EQ(loaded_dense.front().seq, sentinel_dense.seq);
    EXPECT_DOUBLE_EQ(loaded_dense.front().timestamp, sentinel_dense.timestamp);
    EXPECT_EQ(loaded_metadata.source, "existing");
    EXPECT_FALSE(loaded_metadata.degraded);
}

TEST_F(MapSerializerTest, LoadMapWithoutDenseOutputRejectsMalformedDenseTrajectoryPose) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(0);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(std::numeric_limits<double>::quiet_NaN());
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/malformed_dense_no_output_load.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 0U);
}

TEST_F(MapSerializerTest, LoadFailureDoesNotClearExistingMap) {
    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t existing_id = loaded_kfs.addKeyframe(10.0, pose, generateRandomPointCloud(64));
    auto descriptor = loaded_loops.addDescriptor(existing_id, loaded_kfs.getKeyframe(existing_id)->cloud);
    loaded_kfs.updateDescriptor(existing_id, descriptor);
    loaded_optimizer.addPriorFactor(existing_id, pose);
    ASSERT_EQ(loaded_kfs.size(), 1U);
    ASSERT_EQ(loaded_loops.size(), 1U);
    ASSERT_GT(loaded_optimizer.getNumNodes(), 0U);

    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(1.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(1.0);
    dense->mutable_pose_world_lidar()->set_qw(std::numeric_limits<double>::quiet_NaN());

    const std::string map_file = config_.map_save_path + "/failed_load_keeps_existing_map.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    std::vector<core::DenseTrajectoryPose> dense_output;
    core::DenseTrajectoryPose sentinel_dense;
    sentinel_dense.seq = 123;
    sentinel_dense.timestamp = 456.0;
    dense_output.push_back(sentinel_dense);
    core::DenseTrajectoryMetadata dense_metadata;
    dense_metadata.source = "existing";
    dense_metadata.degraded = false;

    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer,
                                    &dense_output, &dense_metadata));
    EXPECT_EQ(loaded_kfs.size(), 1U);
    ASSERT_NE(loaded_kfs.getKeyframe(existing_id), nullptr);
    EXPECT_EQ(loaded_loops.size(), 1U);
    EXPECT_GT(loaded_optimizer.getNumNodes(), 0U);
    ASSERT_EQ(dense_output.size(), 1U);
    EXPECT_EQ(dense_output.front().seq, sentinel_dense.seq);
    EXPECT_EQ(dense_metadata.source, "existing");
    EXPECT_FALSE(dense_metadata.degraded);
}

TEST_F(MapSerializerTest, StrictKeyframeLoadFailureDoesNotClearExistingMap) {
    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t existing_id = loaded_kfs.addKeyframe(10.0, pose, generateRandomPointCloud(64));
    auto descriptor = loaded_loops.addDescriptor(existing_id, loaded_kfs.getKeyframe(existing_id)->cloud);
    loaded_kfs.updateDescriptor(existing_id, descriptor);
    loaded_optimizer.addPriorFactor(existing_id, pose);
    ASSERT_EQ(loaded_kfs.size(), 1U);
    ASSERT_EQ(loaded_loops.size(), 1U);
    ASSERT_GT(loaded_optimizer.getNumNodes(), 0U);

    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);
    addValidKeyframeProto(&map_proto, 0);
    auto* malformed = map_proto.add_keyframes();
    malformed->set_id(1);
    malformed->set_timestamp(std::numeric_limits<double>::quiet_NaN());
    malformed->mutable_pose_odom()->set_qw(1.0);
    malformed->mutable_pose_optimized()->set_qw(1.0);
    malformed->mutable_cloud()->set_num_points(0);

    const std::string map_file = config_.map_save_path + "/strict_keyframe_failure_keeps_existing.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    EXPECT_FALSE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer));
    EXPECT_EQ(loaded_kfs.size(), 1U);
    EXPECT_NE(loaded_kfs.getKeyframe(existing_id), nullptr);
    EXPECT_EQ(loaded_loops.size(), 1U);
    EXPECT_GT(loaded_optimizer.getNumNodes(), 0U);
}

TEST_F(MapSerializerTest, OldPbstreamFallsBackDenseTrajectoryFromKeyframePoses) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.2.0");
    map_proto.mutable_metadata()->set_num_keyframes(2);

    for (int i = 0; i < 2; ++i) {
        auto* kf = map_proto.add_keyframes();
        kf->set_id(i);
        kf->set_timestamp(100.0 + i);
        kf->mutable_pose_odom()->set_qw(1.0);
        kf->mutable_pose_optimized()->set_tx(static_cast<double>(i));
        kf->mutable_pose_optimized()->set_qw(1.0);
        auto* cloud = kf->mutable_cloud();
        cloud->set_num_points(64);
        for (int p = 0; p < 64; ++p) {
            cloud->add_points(static_cast<float>(p % 8) * 0.1f + 1.0f);
            cloud->add_points(static_cast<float>(p / 8) * 0.1f);
            cloud->add_points(0.0f);
            cloud->add_points(1.0f);
        }
    }

    const std::string map_file = config_.map_save_path + "/old_no_dense.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager loaded_kfs(config_);
    LoopDetector loaded_loops(config_);
    GraphOptimizer loaded_optimizer(config_);
    MapSerializer serializer(config_);
    std::vector<core::DenseTrajectoryPose> dense;
    core::DenseTrajectoryMetadata dense_metadata;
    ASSERT_TRUE(serializer.loadMap(map_file, loaded_kfs, loaded_loops, loaded_optimizer, &dense, &dense_metadata));

    ASSERT_EQ(dense.size(), 2U);
    EXPECT_EQ(dense[0].seq, 0U);
    EXPECT_EQ(dense[1].seq, 1U);
    EXPECT_DOUBLE_EQ(dense[0].timestamp, 100.0);
    EXPECT_DOUBLE_EQ(dense[1].timestamp, 101.0);
    EXPECT_NEAR(dense[1].pose_world_lidar.translation().x(), 1.0, 1e-9);
    EXPECT_EQ(dense_metadata.source, "keyframe_fallback");
    EXPECT_TRUE(dense_metadata.degraded);

    const std::string resaved_file = config_.map_save_path + "/old_no_dense_resaved.pbstream";
    ASSERT_TRUE(serializer.saveMap(resaved_file, loaded_kfs, loaded_loops, loaded_optimizer, dense, dense_metadata));
    n3mapping::N3Map resaved;
    {
        std::ifstream ifs(resaved_file, std::ios::binary);
        ASSERT_TRUE(resaved.ParseFromIstream(&ifs));
    }
    EXPECT_EQ(resaved.metadata().dense_trajectory_source(), "keyframe_fallback");
    EXPECT_TRUE(resaved.metadata().dense_trajectory_degraded());
}

TEST_F(MapSerializerTest, NavResourceReaderRequiresNativeDenseByDefault) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.2.0");
    map_proto.mutable_metadata()->set_map_frame("map");
    map_proto.mutable_metadata()->set_body_frame("body");

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_tx(2.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(9.0f);

    const std::string map_file = config_.map_save_path + "/reader_no_dense.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "pbstream_missing_dense_trajectory");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsMissingKeyframes) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);
    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(1.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_zero_keyframes.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "pbstream_missing_keyframes");
}

TEST_F(MapSerializerTest, NavResourceReaderDoesNotTreatDegradedDenseAsNative) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_map_frame("map");
    map_proto.mutable_metadata()->set_body_frame("body");
    map_proto.mutable_metadata()->set_dense_trajectory_source("keyframe_fallback");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(true);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_tx(2.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(9.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_tx(2.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_degraded_dense.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_TRUE(readN3NavResource(map_file, &resource, &error)) << error;
    EXPECT_EQ(resource.dense_trajectory_source, "keyframe_fallback");
    EXPECT_TRUE(resource.dense_trajectory_degraded);
    EXPECT_FALSE(resource.has_native_dense_trajectory);
    EXPECT_TRUE(resource.dense_trajectory_from_keyframe_fallback);
    ASSERT_EQ(resource.dense_optimized_trajectory.size(), 1U);
    EXPECT_NEAR(resource.dense_optimized_trajectory.front().pose_world_lidar.translation().x(), 2.0, 1e-9);
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsDenseTrajectoryWithNoneSource) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("none");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(true);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_dense_none_source.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "invalid dense trajectory source");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsMissingKeyframeCloud) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_missing_cloud.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "missing keyframe cloud");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsEmptyKeyframeCloud) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    kf->mutable_cloud()->set_num_points(0);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_empty_cloud.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "empty keyframe cloud");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsDuplicateKeyframeIds) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    for (int i = 0; i < 2; ++i) {
        auto* kf = map_proto.add_keyframes();
        kf->set_id(7);
        kf->set_timestamp(3.0 + static_cast<double>(i));
        kf->mutable_pose_odom()->set_qw(1.0);
        kf->mutable_pose_optimized()->set_qw(1.0);
        auto* cloud = kf->mutable_cloud();
        cloud->set_num_points(1);
        cloud->add_points(1.0f + static_cast<float>(i));
        cloud->add_points(0.0f);
        cloud->add_points(0.0f);
        cloud->add_points(1.0f);
    }

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_duplicate_keyframe_id.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "duplicate keyframe id");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsNonFiniteKeyframeTimestamp) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(std::numeric_limits<double>::quiet_NaN());
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_nonfinite_keyframe_timestamp.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "non-finite keyframe timestamp");
}

TEST_F(MapSerializerTest, NavResourceReaderRejectsNegativeKeyframeId) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(-1);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(1.0f);

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    const std::string map_file = config_.map_save_path + "/reader_negative_keyframe_id.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    ASSERT_FALSE(readN3NavResource(map_file, &resource, &error));
    EXPECT_EQ(error, "invalid keyframe id");
}

TEST_F(MapSerializerTest, NavResourceReaderSkipsUnusedDescriptors) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.3.0");
    map_proto.mutable_metadata()->set_dense_trajectory_source("native");
    map_proto.mutable_metadata()->set_dense_trajectory_degraded(false);

    auto* kf = addValidKeyframeProto(&map_proto, 7, 3.0);
    kf->mutable_pose_optimized()->set_tx(2.0);
    auto* sc = kf->mutable_sc_descriptor();
    sc->set_rows(2);
    sc->set_cols(2);
    sc->add_values(1.0);
    sc->add_values(2.0);
    sc->add_values(3.0);
    sc->add_values(4.0);
    auto* rhpd = kf->mutable_rhpd_descriptor();
    rhpd->set_dim(RHPD_DIM);
    for (int i = 0; i < RHPD_DIM; ++i) {
        rhpd->add_values(static_cast<double>(i) * 0.001);
    }

    auto* dense = map_proto.add_dense_optimized_trajectory();
    dense->set_seq(0);
    dense->set_timestamp(3.0);
    dense->mutable_pose_world_lidar()->set_tx(2.0);
    dense->mutable_pose_world_lidar()->set_qw(1.0);

    std::vector<ParsedKeyframeProto> parsed_keyframes;
    std::unordered_set<int64_t> keyframe_ids;
    std::string error;
    PbstreamKeyframeParseOptions parse_options;
    parse_options.policy = PbstreamLoadPolicy::STRICT;
    parse_options.expected_rhpd_dim = RHPD_DIM;
    parse_options.parse_descriptors = false;
    ASSERT_TRUE(parseKeyframesFromProto(map_proto, parse_options, &parsed_keyframes, &keyframe_ids, &error))
        << error;
    ASSERT_EQ(parsed_keyframes.size(), 1U);
    EXPECT_EQ(parsed_keyframes.front().sc_descriptor.size(), 0);
    EXPECT_EQ(parsed_keyframes.front().rhpd_descriptor.size(), 0);

    const std::string map_file = config_.map_save_path + "/reader_skips_descriptors.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    ASSERT_TRUE(readN3NavResource(map_file, &resource, &error)) << error;
    ASSERT_EQ(resource.keyframes.size(), 1U);
    EXPECT_EQ(resource.keyframes.front().id, 7);
    ASSERT_EQ(resource.keyframes.front().cloud->size(), 1U);
    EXPECT_NEAR(resource.keyframes.front().pose_optimized.translation().x(), 2.0, 1e-9);
    EXPECT_TRUE(resource.has_native_dense_trajectory);
}

TEST_F(MapSerializerTest, NavResourceReaderExplicitFallbackDoesNotNeedGlobalMapPcd) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.2.0");
    map_proto.mutable_metadata()->set_map_frame("map");
    map_proto.mutable_metadata()->set_body_frame("body");
    map_proto.mutable_metadata()->set_nav_cloud_filter_applied(true);
    map_proto.mutable_metadata()->set_nav_cloud_filter_policy("rear_sector_width_deg=45");
    map_proto.mutable_metadata()->set_descriptors_recomputed_from_filtered_cloud(true);
    map_proto.mutable_metadata()->set_nav_filter_raw_points(10);
    map_proto.mutable_metadata()->set_nav_filter_kept_points(8);
    map_proto.mutable_metadata()->set_nav_filter_removed_points(2);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(7);
    kf->set_timestamp(3.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_tx(2.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(1);
    cloud->add_points(1.0f);
    cloud->add_points(0.0f);
    cloud->add_points(0.0f);
    cloud->add_points(9.0f);

    const std::string map_file = config_.map_save_path + "/reader_no_dense_fallback.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    N3NavResource resource;
    std::string error;
    N3NavReaderOptions options;
    options.allow_keyframe_fallback = true;
    ASSERT_TRUE(readN3NavResource(map_file, options, &resource, &error)) << error;
    ASSERT_EQ(resource.keyframes.size(), 1U);
    ASSERT_EQ(resource.dense_optimized_trajectory.size(), 1U);
    EXPECT_FALSE(resource.has_native_dense_trajectory);
    EXPECT_TRUE(resource.dense_trajectory_from_keyframe_fallback);
    EXPECT_EQ(resource.dense_trajectory_source, "keyframe_fallback");
    EXPECT_TRUE(resource.dense_trajectory_degraded);
    EXPECT_EQ(resource.keyframes.front().id, 7);
    EXPECT_EQ(resource.map_frame, "map");
    EXPECT_EQ(resource.body_frame, "body");
    EXPECT_TRUE(resource.nav_cloud_filter_applied);
    EXPECT_EQ(resource.nav_cloud_filter_policy, "rear_sector_width_deg=45");
    EXPECT_TRUE(resource.descriptors_recomputed_from_filtered_cloud);
    EXPECT_EQ(resource.nav_filter_raw_points, 10U);
    EXPECT_EQ(resource.nav_filter_kept_points, 8U);
    EXPECT_EQ(resource.nav_filter_removed_points, 2U);
    EXPECT_NE(resource.optimized_poses.find(7), resource.optimized_poses.end());
    EXPECT_NEAR(resource.dense_optimized_trajectory.front().pose_world_lidar.translation().x(), 2.0, 1e-9);
}

TEST_F(MapSerializerTest, MalformedRhpdDescriptorIsRejectedAndRebuilt) {
    n3mapping::N3Map map_proto;
    map_proto.mutable_metadata()->set_version("2.2.0");
    map_proto.mutable_metadata()->set_num_keyframes(1);

    auto* kf = map_proto.add_keyframes();
    kf->set_id(0);
    kf->set_timestamp(0.0);
    kf->mutable_pose_odom()->set_qw(1.0);
    kf->mutable_pose_optimized()->set_qw(1.0);
    auto* cloud = kf->mutable_cloud();
    cloud->set_num_points(64);
    for (int i = 0; i < 64; ++i) {
        const float x = static_cast<float>(i % 8) * 0.5f + 1.0f;
        const float y = static_cast<float>(i / 8) * 0.25f;
        const float z = static_cast<float>((i % 4) - 1) * 0.2f;
        cloud->add_points(x);
        cloud->add_points(y);
        cloud->add_points(z);
        cloud->add_points(1.0f);
    }

    auto* rhpd = kf->mutable_rhpd_descriptor();
    rhpd->set_dim(RHPD_DIM);
    for (int i = 0; i < RHPD_DIM - 3; ++i) {
        rhpd->add_values(123.0);
    }

    std::string map_file = config_.map_save_path + "/malformed_rhpd.pbstream";
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    MapSerializer serializer(config_);

    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));
    auto loaded_kf = kf_manager_loaded.getKeyframe(0);
    ASSERT_NE(loaded_kf, nullptr);
    EXPECT_EQ(loaded_kf->rhpd_descriptor.size(), RHPD_DIM);
    EXPECT_LT(loaded_kf->rhpd_descriptor.maxCoeff(), 10.0) << "malformed serialized values were accepted";
}

TEST_F(MapSerializerTest, MatchingRhpdSchemaLoadsSaveTimeRecomputedDescriptorWithoutRebuild) {
    Config cfg = config_;
    cfg.rhpd_submap_voxel_size = 0.0;
    KeyframeManager kf_manager(cfg);
    LoopDetector loop_detector(cfg);
    GraphOptimizer optimizer(cfg);
    MapSerializer serializer(cfg);

    auto cloud = generateRandomPointCloud(500);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    int64_t kf_id = kf_manager.addKeyframe(0.1, pose, cloud);
    auto sc = loop_detector.addDescriptor(kf_id, cloud);
    kf_manager.updateDescriptor(kf_id, sc);
    optimizer.addPriorFactor(kf_id, pose);

    Eigen::VectorXd custom = Eigen::VectorXd::LinSpaced(RHPD_DIM, 0.001, 0.999);
    auto kf = kf_manager.getKeyframe(kf_id);
    ASSERT_NE(kf, nullptr);
    kf->rhpd_descriptor = custom;
    loop_detector.loadRHPDDescriptors({{kf_id, custom}});
    auto expected_cloud = kf_manager.buildCausalSubmapInRootFrame(kf_id, cfg.rhpd_submap_kf_radius, kf_id);
    Eigen::VectorXd expected_saved = loop_detector.computeRHPD(expected_cloud);
    ASSERT_EQ(expected_saved.size(), RHPD_DIM);
    ASSERT_FALSE(expected_saved.isZero());

    std::string map_file = cfg.map_save_path + "/rhpd_schema_match.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    KeyframeManager kf_manager_loaded(cfg);
    LoopDetector loop_detector_loaded(cfg);
    GraphOptimizer optimizer_loaded(cfg);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));

    Eigen::VectorXd loaded;
    ASSERT_TRUE(loop_detector_loaded.getRHPDManager().get(kf_id, &loaded));
    EXPECT_EQ(loaded.size(), RHPD_DIM);
    EXPECT_NEAR((loaded - expected_saved).norm(), 0.0, 1e-9);
    EXPECT_GT((loaded - custom).norm(), 1.0);
}

TEST_F(MapSerializerTest, RhpdSchemaMismatchRebuildsSerializedDescriptor) {
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);

    auto cloud = generateRandomPointCloud(500);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    int64_t kf_id = kf_manager.addKeyframe(0.1, pose, cloud);
    auto sc = loop_detector.addDescriptor(kf_id, cloud);
    kf_manager.updateDescriptor(kf_id, sc);
    optimizer.addPriorFactor(kf_id, pose);

    Eigen::VectorXd custom = Eigen::VectorXd::Constant(RHPD_DIM, 123.0);
    auto kf = kf_manager.getKeyframe(kf_id);
    ASSERT_NE(kf, nullptr);
    kf->rhpd_descriptor = custom;
    loop_detector.loadRHPDDescriptors({{kf_id, custom}});

    std::string map_file = config_.map_save_path + "/rhpd_schema_mismatch.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    n3mapping::N3Map map_proto;
    {
        std::ifstream ifs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.ParseFromIstream(&ifs));
    }
    map_proto.mutable_metadata()->set_rhpd_schema("mismatch");
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager kf_manager_loaded(config_);
    LoopDetector loop_detector_loaded(config_);
    GraphOptimizer optimizer_loaded(config_);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));

    Eigen::VectorXd loaded;
    ASSERT_TRUE(loop_detector_loaded.getRHPDManager().get(kf_id, &loaded));
    EXPECT_EQ(loaded.size(), RHPD_DIM);
    EXPECT_GT((loaded - custom).norm(), 1.0);
    EXPECT_LT(loaded.maxCoeff(), 10.0);
}

TEST_F(MapSerializerTest, CausalRhpdRebuildMatchesOnlineDescriptors) {
    Config cfg = config_;
    cfg.rhpd_submap_kf_radius = 2;
    cfg.rhpd_submap_voxel_size = 0.0;

    KeyframeManager kf_manager(cfg);
    LoopDetector loop_detector(cfg);
    GraphOptimizer optimizer(cfg);
    MapSerializer serializer(cfg);

    std::vector<std::pair<int64_t, Eigen::VectorXd>> online_rhpd;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    for (int i = 0; i < 6; ++i) {
        auto cloud = generateRandomPointCloud(240);
        pose.translation().x() = static_cast<double>(i);
        int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
        auto sc = loop_detector.addDescriptor(kf_id, cloud);
        kf_manager.updateDescriptor(kf_id, sc);
        if (i == 0) {
            optimizer.addPriorFactor(kf_id, pose);
        }

        auto rhpd_cloud = kf_manager.buildCausalSubmapInRootFrame(kf_id, cfg.rhpd_submap_kf_radius, kf_id);
        auto rhpd = loop_detector.addRHPD(kf_id, rhpd_cloud);
        auto kf = kf_manager.getKeyframe(kf_id);
        ASSERT_NE(kf, nullptr);
        kf->rhpd_descriptor = rhpd;
        online_rhpd.emplace_back(kf_id, rhpd);
    }

    std::string map_file = cfg.map_save_path + "/causal_rhpd_rebuild.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    n3mapping::N3Map map_proto;
    {
        std::ifstream ifs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.ParseFromIstream(&ifs));
    }
    map_proto.mutable_metadata()->set_rhpd_schema("force-rebuild");
    {
        std::ofstream ofs(map_file, std::ios::binary);
        ASSERT_TRUE(map_proto.SerializeToOstream(&ofs));
    }

    KeyframeManager kf_manager_loaded(cfg);
    LoopDetector loop_detector_loaded(cfg);
    GraphOptimizer optimizer_loaded(cfg);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));

    for (const auto& [kf_id, expected] : online_rhpd) {
        Eigen::VectorXd loaded;
        ASSERT_TRUE(loop_detector_loaded.getRHPDManager().get(kf_id, &loaded));
        ASSERT_EQ(loaded.size(), RHPD_DIM);
        EXPECT_NEAR((loaded - expected).norm(), 0.0, 1e-9) << "kf_id=" << kf_id;
    }
}

TEST_F(MapSerializerTest, SaveMapRecomputesCausalRhpdUsingCurrentOptimizedPoses) {
    Config cfg = config_;
    cfg.rhpd_submap_kf_radius = 2;
    cfg.rhpd_submap_voxel_size = 0.0;

    KeyframeManager kf_manager(cfg);
    LoopDetector loop_detector(cfg);
    GraphOptimizer optimizer(cfg);
    MapSerializer serializer(cfg);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    for (int i = 0; i < 5; ++i) {
        auto cloud = generateRandomPointCloud(420);
        pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = static_cast<double>(i) * 2.0;
        int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
        auto sc = loop_detector.addDescriptor(kf_id, cloud);
        kf_manager.updateDescriptor(kf_id, sc);
        if (i == 0) optimizer.addPriorFactor(kf_id, pose);

        auto rhpd_cloud = kf_manager.buildCausalSubmapInRootFrame(kf_id, cfg.rhpd_submap_kf_radius, kf_id);
        auto rhpd = loop_detector.addRHPD(kf_id, rhpd_cloud);
        auto kf = kf_manager.getKeyframe(kf_id);
        ASSERT_NE(kf, nullptr);
        kf->rhpd_descriptor = rhpd;
    }

    Eigen::VectorXd old_online;
    ASSERT_TRUE(loop_detector.getRHPDManager().get(4, &old_online));
    ASSERT_EQ(old_online.size(), RHPD_DIM);

    std::map<int64_t, Eigen::Isometry3d> optimized_poses;
    for (const auto& kf : kf_manager.getAllKeyframes()) {
        ASSERT_NE(kf, nullptr);
        optimized_poses[kf->id] = kf->pose_optimized;
    }
    optimized_poses[2].translation().y() += 8.0;
    optimized_poses[2].rotate(Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()));
    kf_manager.updateOptimizedPoses(optimized_poses);

    auto expected_cloud = kf_manager.buildCausalSubmapInRootFrame(4, cfg.rhpd_submap_kf_radius, 4);
    Eigen::VectorXd expected_saved = loop_detector.computeRHPD(expected_cloud);
    ASSERT_EQ(expected_saved.size(), RHPD_DIM);
    ASSERT_FALSE(expected_saved.isZero());
    ASSERT_GT((expected_saved - old_online).norm(), 1e-3);

    std::string map_file = cfg.map_save_path + "/save_recomputes_rhpd.pbstream";
    ASSERT_TRUE(serializer.saveMap(map_file, kf_manager, loop_detector, optimizer));

    Eigen::VectorXd manager_after_save;
    ASSERT_TRUE(loop_detector.getRHPDManager().get(4, &manager_after_save));
    EXPECT_NEAR((manager_after_save - old_online).norm(), 0.0, 1e-9)
        << "saveMap should not mutate runtime RHPDManager state";

    KeyframeManager kf_manager_loaded(cfg);
    LoopDetector loop_detector_loaded(cfg);
    GraphOptimizer optimizer_loaded(cfg);
    ASSERT_TRUE(serializer.loadMap(map_file, kf_manager_loaded, loop_detector_loaded, optimizer_loaded));

    Eigen::VectorXd loaded;
    ASSERT_TRUE(loop_detector_loaded.getRHPDManager().get(4, &loaded));
    ASSERT_EQ(loaded.size(), RHPD_DIM);
    EXPECT_LT((loaded - expected_saved).norm(), (loaded - old_online).norm());
    EXPECT_NEAR((loaded - expected_saved).norm(), 0.0, 1e-9);
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
    rclcpp::init(argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
