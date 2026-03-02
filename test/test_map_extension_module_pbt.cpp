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
 * @brief MappingResuming 属性测试
 *
 * Feature: n3mapping-backend
 * Property 9: 地图续建关键帧 ID 连续性
 *
 * *For any* 地图续建操作，新添加的关键帧 ID 应从已加载地图的最大 ID + 1 开始连续递增。
 *
 * **Validates: Requirements 12.3, 12.7**
 */
class MappingResumingPBTTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        config_.map_save_path = "/tmp/n3mapping_extension_pbt_test";
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

        std::filesystem::create_directories(config_.map_save_path);
        rng_.seed(std::random_device{}());
    }

    void TearDown() override
    {
        try {
            std::filesystem::remove_all(config_.map_save_path);
        } catch (...) {
        }
    }

    // 生成随机点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr generateRandomCloud(size_t num_points = 500)
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->resize(num_points);

        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> intensity_dist(0.0f, 255.0f);

        for (auto& pt : cloud->points) {
            pt.x = dist(rng_);
            pt.y = dist(rng_);
            pt.z = dist(rng_) * 0.1f;
            pt.intensity = intensity_dist(rng_);
        }

        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = false;

        return cloud;
    }

    // 创建随机大小的测试地图
    std::pair<std::string, int64_t> createRandomTestMap()
    {
        std::uniform_int_distribution<int> size_dist(3, 10);
        int num_keyframes = size_dist(rng_);

        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        optimizer.addPriorFactor(0, pose);

        int64_t max_id = -1;

        for (int i = 0; i < num_keyframes; ++i) {
            auto cloud = generateRandomCloud();
            int64_t kf_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
            loop_detector.addDescriptor(kf_id, cloud);

            if (kf_id > max_id) max_id = kf_id;

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

        std::string map_file = config_.map_save_path + "/random_map_" + std::to_string(rng_()) + ".pbstream";
        serializer.saveMap(map_file, kf_manager, loop_detector, optimizer);

        return { map_file, max_id };
    }

    Config config_;
    std::mt19937 rng_;
};

/**
 * @brief Property 9: 地图续建关键帧 ID 连续性
 *
 * 测试新添加的关键帧 ID 从已加载地图的最大 ID + 1 开始
 *
 * **Validates: Requirements 12.3, 12.7**
 */
TEST_F(MappingResumingPBTTest, Property9_KeyframeIdContinuity)
{
    constexpr int NUM_ITERATIONS = 20;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 创建随机大小的测试地图
        auto [map_file, original_max_id] = createRandomTestMap();

        // 创建新的组件
        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);
        WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

        MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

        // 加载地图
        ASSERT_TRUE(extension.loadExistingMap(map_file)) << "Iteration " << iter << ": Failed to load map";

        // 验证原始地图关键帧 ID
        int64_t loaded_max_id = -1;
        for (const auto& kf : kf_manager.getAllKeyframes()) {
            if (kf && kf->id > loaded_max_id) {
                loaded_max_id = kf->id;
            }
        }

        EXPECT_EQ(loaded_max_id, original_max_id) << "Iteration " << iter << ": Max ID mismatch after loading";

        // 添加新关键帧
        std::uniform_int_distribution<int> new_kf_dist(1, 5);
        int num_new_keyframes = new_kf_dist(rng_);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = (original_max_id + 1) * 2.0;

        std::vector<int64_t> new_ids;
        for (int i = 0; i < num_new_keyframes; ++i) {
            auto cloud = generateRandomCloud();
            int64_t new_id = kf_manager.addKeyframe(i * 0.1, pose, cloud);
            new_ids.push_back(new_id);
            pose.translation().x() += 2.0;
        }

        // 验证新 ID 从 original_max_id + 1 开始连续递增
        for (size_t i = 0; i < new_ids.size(); ++i) {
            int64_t expected_id = original_max_id + 1 + static_cast<int64_t>(i);
            EXPECT_EQ(new_ids[i], expected_id) << "Iteration " << iter << ", new keyframe " << i << ": Expected ID " << expected_id << ", got "
                                               << new_ids[i];
        }

        // 清理
        std::filesystem::remove(map_file);
    }
}

/**
 * @brief 测试原始地图完整性保持
 *
 * 验证加载地图后，原始关键帧数据保持不变
 *
 * **Validates: Requirements 12.7**
 */
TEST_F(MappingResumingPBTTest, OriginalMapIntegrity)
{
    constexpr int NUM_ITERATIONS = 15;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto [map_file, original_max_id] = createRandomTestMap();

        // 第一次加载获取原始数据
        KeyframeManager kf_manager1(config_);
        LoopDetector loop_detector1(config_);
        GraphOptimizer optimizer1(config_);
        MapSerializer serializer1(config_);

        ASSERT_TRUE(serializer1.loadMap(map_file, kf_manager1, loop_detector1, optimizer1));

        size_t original_count = kf_manager1.size();
        size_t original_edges = optimizer1.getNumEdges();

        // 第二次加载并添加新关键帧
        KeyframeManager kf_manager2(config_);
        LoopDetector loop_detector2(config_);
        PointCloudMatcher matcher2(config_);
        GraphOptimizer optimizer2(config_);
        MapSerializer serializer2(config_);
        WorldLocalizing relocalization2(config_, kf_manager2, loop_detector2, matcher2);

        MappingResuming extension(config_, kf_manager2, loop_detector2, matcher2, optimizer2, serializer2, relocalization2);

        ASSERT_TRUE(extension.loadExistingMap(map_file));

        // 添加一些新关键帧
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = 100.0;

        for (int i = 0; i < 3; ++i) {
            auto cloud = generateRandomCloud();
            kf_manager2.addKeyframe(i * 0.1, pose, cloud);
            pose.translation().x() += 2.0;
        }

        // 保存扩展地图
        std::string extended_file = config_.map_save_path + "/extended_" + std::to_string(iter) + ".pbstream";
        ASSERT_TRUE(extension.saveExtendedMap(extended_file));

        // 加载扩展地图并验证原始数据
        KeyframeManager kf_manager3(config_);
        LoopDetector loop_detector3(config_);
        GraphOptimizer optimizer3(config_);

        ASSERT_TRUE(serializer2.loadMap(extended_file, kf_manager3, loop_detector3, optimizer3));

        // 验证原始关键帧仍然存在
        for (int64_t id = 0; id <= original_max_id; ++id) {
            auto kf = kf_manager3.getKeyframe(id);
            EXPECT_NE(kf, nullptr) << "Iteration " << iter << ": Original keyframe " << id << " missing";
        }

        // 验证总数增加
        EXPECT_EQ(kf_manager3.size(), original_count + 3) << "Iteration " << iter << ": Keyframe count mismatch";

        // 清理
        std::filesystem::remove(map_file);
        std::filesystem::remove(extended_file);
    }
}

/**
 * @brief 测试 isFromOriginalMap 的正确性
 */
TEST_F(MappingResumingPBTTest, IsFromOriginalMapCorrectness)
{
    constexpr int NUM_ITERATIONS = 20;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto [map_file, original_max_id] = createRandomTestMap();

        KeyframeManager kf_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        GraphOptimizer optimizer(config_);
        MapSerializer serializer(config_);
        WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

        MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

        ASSERT_TRUE(extension.loadExistingMap(map_file));

        // 验证所有原始 ID 返回 true
        for (int64_t id = 0; id <= original_max_id; ++id) {
            EXPECT_TRUE(extension.isFromOriginalMap(id)) << "Iteration " << iter << ": ID " << id << " should be from original map";
        }

        // 验证新 ID 返回 false
        std::uniform_int_distribution<int64_t> new_id_dist(original_max_id + 1, original_max_id + 100);
        for (int i = 0; i < 10; ++i) {
            int64_t new_id = new_id_dist(rng_);
            EXPECT_FALSE(extension.isFromOriginalMap(new_id))
              << "Iteration " << iter << ": ID " << new_id << " should NOT be from original map";
        }

        std::filesystem::remove(map_file);
    }
}

} // namespace test
} // namespace n3mapping

int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_map_extension_module_pbt");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
