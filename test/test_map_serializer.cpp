#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "n3mapping/map_serializer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/graph_optimizer.h"
#include "n3map.pb.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <pcl/memory.h>
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
