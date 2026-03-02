#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <random>
#include <ros/ros.h>

namespace n3mapping {
namespace test {

/**
 * @brief MappingResuming 单元测试
 *
 * Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
 */
class MappingResumingTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // 初始化配置
        config_.map_save_path = "/tmp/n3mapping_extension_test";
        config_.keyframe_distance_threshold = 1.0;
        config_.keyframe_angle_threshold = 0.5;
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.5;
        config_.sc_dist_threshold = 0.3;
        config_.sc_num_exclude_recent = 3;
        config_.sc_num_candidates = 5;
        config_.reloc_num_candidates = 5;
        config_.reloc_sc_dist_threshold = 0.5;
        config_.reloc_search_radius = 20.0;
        config_.reloc_max_track_failures = 5;
        config_.odom_noise_position = 0.1;
        config_.odom_noise_rotation = 0.1;
        config_.loop_noise_position = 0.1;
        config_.loop_noise_rotation = 0.1;
        config_.prior_noise_position = 0.01;
        config_.prior_noise_rotation = 0.01;

        // 创建测试目录
        std::filesystem::create_directories(config_.map_save_path);
    }

    void TearDown() override
    {
        // 清理测试文件
        try {
            std::filesystem::remove_all(config_.map_save_path);
        } catch (...) {
            // 忽略清理错误
        }
    }

    // 生成走廊点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr generateCorridorCloud(const Eigen::Isometry3d& pose, double corridor_width = 3.0)
    {

        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

        // 生成走廊两侧的墙壁点
        for (float x = -5.0f; x <= 5.0f; x += 0.3f) {
            for (float z = 0.0f; z <= 2.0f; z += 0.4f) {
                pcl::PointXYZI pt_left, pt_right;
                pt_left.x = x + noise(rng);
                pt_left.y = -corridor_width / 2.0f + noise(rng);
                pt_left.z = z + noise(rng);
                pt_left.intensity = 50.0f;
                cloud->push_back(pt_left);

                pt_right.x = x + noise(rng);
                pt_right.y = corridor_width / 2.0f + noise(rng);
                pt_right.z = z + noise(rng);
                pt_right.intensity = 50.0f;
                cloud->push_back(pt_right);
            }
        }

        // 变换到世界坐标系
        auto transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *transformed, transform);

        transformed->width = transformed->size();
        transformed->height = 1;
        transformed->is_dense = true;

        return transformed;
    }

    // 创建并保存测试地图
    std::string createTestMap(int num_keyframes = 5)
    {
        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, pose);

        for (int i = 0; i < num_keyframes; ++i) {
            auto cloud = generateCorridorCloud(pose);
            int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
            loop_detector.addDescriptor(kf_id, cloud);

            if (i > 0) {
                EdgeInfo edge;
                edge.from_id = i - 1;
                edge.to_id = i;
                edge.measurement = Eigen::Isometry3d::Identity();
                edge.measurement.translation().x() = 2.0;
                edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
                edge.type = EdgeType::ODOMETRY;
                optimizer.addOdometryEdge(edge);
            }

            pose.translation().x() += 2.0;
        }

        optimizer.incrementalOptimize();

        std::string map_file = config_.map_save_path + "/test_map.pbstream";
        serializer.saveMap(map_file, kf_manager, loop_detector, optimizer);

        return map_file;
    }

    Config config_;
};

/**
 * @brief 测试初始状态
 */
TEST_F(MappingResumingTest, InitialState)
{
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    EXPECT_EQ(extension.getState(), MappingResumingState::NOT_INITIALIZED);
    EXPECT_EQ(extension.getOriginalKeyframeCount(), 0);
    EXPECT_EQ(extension.getNewKeyframeCount(), 0);
    EXPECT_EQ(extension.getCrossLoopCount(), 0);
}

/**
 * @brief 测试加载地图
 * Requirements: 12.1
 */
TEST_F(MappingResumingTest, LoadExistingMap)
{
    // 创建测试地图
    std::string map_file = createTestMap(5);

    // 创建新的组件
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    // 加载地图
    ASSERT_TRUE(extension.loadExistingMap(map_file));

    EXPECT_EQ(extension.getState(), MappingResumingState::MAP_LOADED);
    EXPECT_EQ(extension.getOriginalKeyframeCount(), 5);
    EXPECT_EQ(kf_manager.size(), 5);
}

/**
 * @brief 测试加载不存在的地图
 */
TEST_F(MappingResumingTest, LoadNonExistentMap)
{
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    EXPECT_FALSE(extension.loadExistingMap("/non/existent/path.pbstream"));
    EXPECT_EQ(extension.getState(), MappingResumingState::NOT_INITIALIZED);
}

/**
 * @brief 测试关键帧 ID 连续性
 * Requirements: 12.3
 */
TEST_F(MappingResumingTest, KeyframeIdContinuity)
{
    // 创建测试地图
    std::string map_file = createTestMap(5);

    // 创建新的组件
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    // 加载地图
    ASSERT_TRUE(extension.loadExistingMap(map_file));

    // 获取原始地图最大 ID
    int64_t max_original_id = -1;
    for (const auto& kf : kf_manager.getAllKeyframes()) {
        if (kf && kf->id > max_original_id) {
            max_original_id = kf->id;
        }
    }

    // 手动设置重定位状态以便添加新关键帧
    relocalization.setMapToOdomTransform(Eigen::Isometry3d::Identity());

    // 模拟添加新关键帧
    Eigen::Isometry3d new_pose = Eigen::Isometry3d::Identity();
    new_pose.translation().x() = 12.0; // 在原始地图末端之后

    auto cloud = generateCorridorCloud(new_pose);

    // 由于 processNewKeyframe 需要 RELOCALIZED 状态，我们直接测试 KeyframeManager
    int64_t new_id = kf_manager.addKeyframe(0.5, new_pose, cloud);

    // 验证新 ID > 原始最大 ID
    EXPECT_GT(new_id, max_original_id);
}

/**
 * @brief 测试重置功能
 */
TEST_F(MappingResumingTest, Reset)
{
    std::string map_file = createTestMap(3);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    // 加载地图
    ASSERT_TRUE(extension.loadExistingMap(map_file));
    EXPECT_EQ(extension.getState(), MappingResumingState::MAP_LOADED);

    // 重置
    extension.reset();

    EXPECT_EQ(extension.getState(), MappingResumingState::NOT_INITIALIZED);
    EXPECT_EQ(extension.getOriginalKeyframeCount(), 0);
    EXPECT_EQ(extension.getCrossLoopCount(), 0);
}

/**
 * @brief 测试 isFromOriginalMap
 */
TEST_F(MappingResumingTest, IsFromOriginalMap)
{
    std::string map_file = createTestMap(5);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    ASSERT_TRUE(extension.loadExistingMap(map_file));

    // 原始地图关键帧 ID 为 0-4
    EXPECT_TRUE(extension.isFromOriginalMap(0));
    EXPECT_TRUE(extension.isFromOriginalMap(2));
    EXPECT_TRUE(extension.isFromOriginalMap(4));

    // ID > 4 不是原始地图
    EXPECT_FALSE(extension.isFromOriginalMap(5));
    EXPECT_FALSE(extension.isFromOriginalMap(10));
}

/**
 * @brief 测试保存扩展地图
 * Requirements: 12.6, 12.7
 */
TEST_F(MappingResumingTest, SaveExtendedMap)
{
    std::string map_file = createTestMap(3);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    ASSERT_TRUE(extension.loadExistingMap(map_file));

    // 保存扩展地图
    std::string extended_map_file = config_.map_save_path + "/extended_map.pbstream";
    ASSERT_TRUE(extension.saveExtendedMap(extended_map_file));
    ASSERT_TRUE(std::filesystem::exists(extended_map_file));

    // 验证保存的地图可以加载
    KeyframeManager kf_manager2(config_);
    LoopDetector loop_detector2(config_);
    GraphOptimizer optimizer2(config_);

    ASSERT_TRUE(serializer.loadMap(extended_map_file, kf_manager2, loop_detector2, optimizer2));
    EXPECT_EQ(kf_manager2.size(), 3);
}

} // namespace test
} // namespace n3mapping

int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_map_extension_module");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
