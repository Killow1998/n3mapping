#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <random>
#include <rclcpp/rclcpp.hpp>

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

    void configureSingleFrameRelocalization()
    {
        config_.sc_num_exclude_recent = 0;
        config_.rhpd_enabled = false;
        config_.reloc_temporal_window_size = 1;
        config_.reloc_lock_min_winner_streak = 1;
        config_.reloc_lock_min_converged_updates = 1;
        config_.reloc_lock_log_likelihood_threshold = -1e308;
        config_.reloc_min_confidence = 0.0;
        config_.reloc_min_inlier_ratio = 0.0;
        config_.reloc_static_agg_enable = false;
        config_.reloc_free_space_enable = false;
    }

    bool loadForExtensionTest(const std::string& map_file,
                              KeyframeManager& keyframe_manager,
                              LoopDetector& loop_detector,
                              GraphOptimizer& optimizer,
                              MapSerializer& serializer,
                              MappingResuming& extension)
    {
        return serializer.loadMap(
                   map_file, keyframe_manager, loop_detector, optimizer) &&
               extension.initializeFromLoadedMap();
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
    EXPECT_EQ(extension.getCurrentSessionId(), kInvalidMapSessionId);
}

/**
 * @brief 测试加载地图
 * Requirements: 12.1
 */
TEST_F(MappingResumingTest, InitializeFromSerializedMap)
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
    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));

    EXPECT_EQ(extension.getState(), MappingResumingState::MAP_LOADED);
    EXPECT_EQ(extension.getOriginalKeyframeCount(), 5);
    EXPECT_EQ(kf_manager.size(), 5);
}

/**
 * @brief 测试加载不存在的地图
 */
TEST_F(MappingResumingTest, SerializedLoadFailureKeepsUninitialized)
{
    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher, optimizer, serializer, relocalization);

    EXPECT_FALSE(loadForExtensionTest("/non/existent/path.pbstream", kf_manager,
                                      loop_detector, optimizer, serializer,
                                      extension));
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
    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));

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
    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));
    EXPECT_EQ(extension.getState(), MappingResumingState::MAP_LOADED);

    // 重置
    extension.reset();

    EXPECT_EQ(extension.getState(), MappingResumingState::NOT_INITIALIZED);
    EXPECT_EQ(extension.getOriginalKeyframeCount(), 0);
    EXPECT_EQ(extension.getCrossLoopCount(), 0);
    EXPECT_EQ(extension.getCurrentSessionId(), kInvalidMapSessionId);
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

    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));

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
    config_.submap_shadow_enable = true;
    std::string map_file = createTestMap(3);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    SubmapBuilder submap_builder(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);

    MappingResuming extension(config_, kf_manager, loop_detector, matcher,
                              optimizer, serializer, relocalization,
                              &submap_builder);

    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));
    for (const auto& keyframe : kf_manager.getAllKeyframes()) {
        ASSERT_TRUE(submap_builder.appendKeyframe(keyframe));
    }

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

    std::ifstream trial_input(
        config_.map_save_path + "/submap_graph_trial.jsonl");
    ASSERT_TRUE(trial_input.is_open());
    std::string trial_record;
    ASSERT_TRUE(static_cast<bool>(std::getline(trial_input, trial_record)));
    EXPECT_NE(trial_record.find(
                  "\"runtime_source\":\"mapping_resuming\""),
              std::string::npos);
    EXPECT_NE(trial_record.find("\"context\":\"save_extended_map\""),
              std::string::npos);
    EXPECT_NE(trial_record.find("\"valid\":true"), std::string::npos);
    EXPECT_NE(trial_record.find("\"solved\":true"), std::string::npos);
    EXPECT_NE(trial_record.find(
                  "\"optimized_keyframe_reference_count\":3"),
              std::string::npos);
}

TEST_F(MappingResumingTest, CommitsSessionAnchorThenRawOdometry)
{
    configureSingleFrameRelocalization();
    config_.loop_noise_position = 0.2;
    config_.loop_noise_rotation = 0.3;
    config_.loaded_map_tracking_noise_position = 0.2;
    config_.loaded_map_tracking_noise_rotation = 0.3;
    config_.odom_noise_position = 0.1;
    config_.odom_noise_rotation = 0.05;
    const std::string map_file = createTestMap(1);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);
    MappingResuming extension(config_, kf_manager, loop_detector, matcher,
                              optimizer, serializer, relocalization);

    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));
    auto anchor = kf_manager.getKeyframe(0);
    ASSERT_NE(anchor, nullptr);
    ASSERT_TRUE(extension.performInitialRelocalization(
        anchor->cloud, Eigen::Isometry3d::Identity(), "lio_odom"));
    ASSERT_EQ(relocalization.getLastMatchedKeyframeId(), 0);
    EXPECT_EQ(extension.getCurrentSessionId(), 1u);
    const Eigen::Isometry3d anchor_pose_before = anchor->pose_optimized;

    Eigen::Isometry3d first_odom = Eigen::Isometry3d::Identity();
    first_odom.translation().x() = 2.0;
    const Eigen::Isometry3d first_pose_in_map =
        relocalization.getMapToOdomTransform() * first_odom;
    const int64_t first_id = extension.processNewKeyframe(
        10.0, first_odom, anchor->cloud);
    ASSERT_EQ(first_id, 1);
    EXPECT_EQ(extension.getState(), MappingResumingState::EXTENDING);
    EXPECT_TRUE(optimizer.hasNode(first_id));
    auto first_keyframe = kf_manager.getKeyframe(first_id);
    ASSERT_NE(first_keyframe, nullptr);
    EXPECT_EQ(first_keyframe->session_id, 1u);
    EXPECT_TRUE(first_keyframe->pose_odom.isApprox(first_odom, 1e-9));
    EXPECT_TRUE(first_keyframe->pose_optimized.isApprox(
        optimizer.getOptimizedPose(first_id), 1e-9));
    const auto sessions = kf_manager.getSessions();
    ASSERT_EQ(sessions.size(), 2u);
    EXPECT_EQ(sessions[1].id, 1u);
    EXPECT_EQ(sessions[1].source_frame_id, "lio_odom");
    EXPECT_DOUBLE_EQ(sessions[1].start_timestamp, 10.0);
    EXPECT_FALSE(sessions[1].loaded);

    const auto& first_edges = optimizer.getEdges();
    const auto anchor_edge = std::find_if(
        first_edges.begin(), first_edges.end(), [first_id](const EdgeInfo& edge) {
            return edge.type == EdgeType::SESSION_ANCHOR &&
                   edge.from_id == 0 && edge.to_id == first_id;
        });
    ASSERT_NE(anchor_edge, first_edges.end());
    EXPECT_TRUE(anchor_edge->measurement.isApprox(
        anchor_pose_before.inverse() * first_pose_in_map, 1e-9));
    EXPECT_NEAR(anchor_edge->information(0, 0), 25.0, 1e-9);
    EXPECT_NEAR(anchor_edge->information(3, 3), 1.0 / 0.09, 1e-9);

    // The explicit anchor is the only cross-session constraint on the first
    // new keyframe. An immediate legacy loop would duplicate the same evidence
    // and could overpower it with an ICP Hessian.
    EXPECT_EQ(extension.detectCrossLoops(first_id), 0);
    EXPECT_EQ(optimizer.getEdges().size(), 1u);

    // Deliberately perturb the stored optimized pose. The next base edge must
    // still come from the two raw session odometry poses.
    first_keyframe->pose_optimized.translation().y() += 50.0;

    Eigen::Isometry3d second_odom = first_odom;
    second_odom.translation().x() = 4.0;
    const int64_t second_id = extension.processNewKeyframe(
        11.0, second_odom, anchor->cloud);
    ASSERT_EQ(second_id, 2);
    const auto second_keyframe = kf_manager.getKeyframe(second_id);
    ASSERT_NE(second_keyframe, nullptr);
    EXPECT_EQ(second_keyframe->session_id, 1u);
    EXPECT_TRUE(second_keyframe->pose_odom.isApprox(second_odom, 1e-9));

    const auto& all_edges = optimizer.getEdges();
    const auto odom_edge = std::find_if(
        all_edges.begin(), all_edges.end(),
        [first_id, second_id](const EdgeInfo& edge) {
            return edge.type == EdgeType::ODOMETRY &&
                   edge.from_id == first_id && edge.to_id == second_id;
        });
    ASSERT_NE(odom_edge, all_edges.end());
    EXPECT_TRUE(odom_edge->measurement.isApprox(
        first_odom.inverse() * second_odom, 1e-9));
    EXPECT_NEAR(odom_edge->measurement.translation().y(), 0.0, 1e-9);
}

TEST_F(MappingResumingTest, TrackedKeyframeAtomicallyPinsSessionDrift)
{
    configureSingleFrameRelocalization();
    config_.submap_shadow_enable = true;
    config_.submap_max_keyframes = 2;
    config_.use_robust_kernel = true;
    config_.robust_kernel_type = "Cauchy";
    config_.robust_kernel_delta = 1.0;
    config_.odom_noise_position = 0.01;
    config_.odom_noise_rotation = 0.001;
    config_.loop_noise_position = 0.05;
    config_.loop_noise_rotation = 0.5;
    config_.loaded_map_tracking_noise_position = 0.05;
    config_.loaded_map_tracking_noise_rotation = 0.01;
    const std::string map_file = createTestMap(1);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    SubmapBuilder submap_builder(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);
    MappingResuming extension(config_, kf_manager, loop_detector, matcher,
                              optimizer, serializer, relocalization,
                              &submap_builder);

    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));
    auto loaded = kf_manager.getKeyframe(0);
    ASSERT_NE(loaded, nullptr);
    ASSERT_TRUE(extension.performInitialRelocalization(
        loaded->cloud, Eigen::Isometry3d::Identity()));

    Eigen::Isometry3d first_odom = Eigen::Isometry3d::Identity();
    first_odom.translation().x() = 2.0;
    Eigen::Isometry3d first_tracked = Eigen::Isometry3d::Identity();
    first_tracked.translation().x() = 2.0;
    ASSERT_EQ(extension.processNewKeyframe(
                  10.0, first_odom, loaded->cloud, 0, first_tracked),
              1);
    const auto submaps_after_first = submap_builder.getSubmaps();
    ASSERT_EQ(submaps_after_first.size(), 1u);
    const Eigen::Isometry3d first_origin_before_tracking =
        submaps_after_first.front().T_map_submap;

    Eigen::Isometry3d second_odom = first_odom;
    second_odom.translation().x() = 4.0;
    second_odom.rotate(
        Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
    Eigen::Isometry3d second_tracked = Eigen::Isometry3d::Identity();
    second_tracked.translation().x() = 10.0;
    ASSERT_EQ(extension.processNewKeyframe(
                  11.0, second_odom, loaded->cloud, 0, second_tracked),
              2);

    EXPECT_NEAR(optimizer.getOptimizedPose(2).translation().x(), 10.0, 0.2);
    EXPECT_NEAR(Eigen::AngleAxisd(
                    optimizer.getOptimizedPose(2).rotation()).angle(),
                0.0, 1.0 * M_PI / 180.0);
    EXPECT_EQ(extension.getCrossLoopCount(), 1u);
    const auto edges = optimizer.getEdges();
    const auto odometry = std::find_if(
        edges.begin(), edges.end(), [](const EdgeInfo& edge) {
            return edge.type == EdgeType::ODOMETRY &&
                   edge.from_id == 1 && edge.to_id == 2;
        });
    const auto tracking_loop = std::find_if(
        edges.begin(), edges.end(), [](const EdgeInfo& edge) {
            return edge.type == EdgeType::LOOP &&
                   edge.from_id == 0 && edge.to_id == 2;
        });
    ASSERT_NE(odometry, edges.end());
    ASSERT_NE(tracking_loop, edges.end());
    EXPECT_NEAR(odometry->measurement.translation().x(), 2.0, 1e-9);
    EXPECT_NEAR(tracking_loop->measurement.translation().x(), 10.0, 1e-9);
    EXPECT_NEAR(tracking_loop->information(3, 3), 10000.0, 1e-9);

    const auto first_keyframe = kf_manager.getKeyframe(1);
    ASSERT_NE(first_keyframe, nullptr);
    const auto refreshed_submaps = submap_builder.getSubmaps();
    ASSERT_EQ(refreshed_submaps.size(), 1u);
    EXPECT_EQ(refreshed_submaps.front().keyframe_ids,
              (std::vector<int64_t>{1, 2}));
    EXPECT_TRUE(refreshed_submaps.front().closed);
    EXPECT_GT((refreshed_submaps.front().T_map_submap.translation() -
               first_origin_before_tracking.translation()).norm(),
              1e-5);
    EXPECT_TRUE(refreshed_submaps.front().T_map_submap.isApprox(
        first_keyframe->pose_optimized, 1e-9));
    const auto projection = submap_builder.evaluatePoseProjection(
        kf_manager.getAllKeyframes());
    ASSERT_TRUE(projection.valid) << projection.failure_reason;
    EXPECT_EQ(projection.projected_keyframe_count, 2u);
    EXPECT_EQ(projection.unassigned_keyframe_count, 1u);
}

TEST_F(MappingResumingTest, OptimizeFailureLeavesNoGhostAndCanRetrySameId)
{
    configureSingleFrameRelocalization();
    const std::string map_file = createTestMap(1);

    KeyframeManager kf_manager(config_);
    LoopDetector loop_detector(config_);
    PointCloudMatcher matcher(config_);
    GraphOptimizer optimizer(config_);
    MapSerializer serializer(config_);
    WorldLocalizing relocalization(config_, kf_manager, loop_detector, matcher);
    MappingResuming extension(config_, kf_manager, loop_detector, matcher,
                              optimizer, serializer, relocalization);

    ASSERT_TRUE(loadForExtensionTest(map_file, kf_manager, loop_detector,
                                     optimizer, serializer, extension));
    auto anchor = kf_manager.getKeyframe(0);
    ASSERT_NE(anchor, nullptr);
    ASSERT_TRUE(extension.performInitialRelocalization(
        anchor->cloud, Eigen::Isometry3d::Identity()));

    const size_t keyframes_before = kf_manager.size();
    const size_t sc_before = loop_detector.size();
    const size_t rhpd_before = loop_detector.getRHPDManager().size();
    const size_t edges_before = optimizer.getNumEdges();
    const int64_t next_id_before = kf_manager.getNextKeyframeId();

    // This pending factor deterministically makes the combined iSAM2 update
    // fail; GraphOptimizer then clears every pending factor in that update.
    EdgeInfo invalid_loop;
    invalid_loop.from_id = 0;
    invalid_loop.to_id = 999;
    invalid_loop.measurement = Eigen::Isometry3d::Identity();
    invalid_loop.information =
        Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    invalid_loop.type = EdgeType::LOOP;
    optimizer.addLoopEdge(invalid_loop);

    Eigen::Isometry3d new_odom = Eigen::Isometry3d::Identity();
    new_odom.translation().x() = 2.0;
    EXPECT_EQ(extension.processNewKeyframe(10.0, new_odom, anchor->cloud), -1);
    EXPECT_EQ(extension.getState(), MappingResumingState::RELOCALIZED);
    EXPECT_EQ(extension.getNewKeyframeCount(), 0u);
    EXPECT_EQ(kf_manager.size(), keyframes_before);
    EXPECT_EQ(kf_manager.getNextKeyframeId(), next_id_before);
    EXPECT_EQ(loop_detector.size(), sc_before);
    EXPECT_EQ(loop_detector.getRHPDManager().size(), rhpd_before);
    EXPECT_EQ(optimizer.getNumEdges(), edges_before);
    EXPECT_FALSE(optimizer.hasNode(next_id_before));
    EXPECT_FALSE(optimizer.hasGlobalConstraint());
    EXPECT_EQ(kf_manager.getSessions().size(), 1u);

    // The first-frame state was not advanced, and rollback restored the id.
    EXPECT_EQ(extension.processNewKeyframe(11.0, new_odom, anchor->cloud),
              next_id_before);
    EXPECT_EQ(extension.getState(), MappingResumingState::EXTENDING);
    EXPECT_TRUE(optimizer.hasNode(next_id_before));
    EXPECT_EQ(kf_manager.getSessions().size(), 2u);
}

} // namespace test
} // namespace n3mapping

int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
