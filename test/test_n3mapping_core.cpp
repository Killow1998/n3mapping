#include <filesystem>
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <pcl/memory.h>

#include "n3mapping/core/n3mapping_core.h"

namespace n3mapping {
namespace test {

namespace {

core::LioFrame makeFrame(int64_t stamp_nsec,
                         const Eigen::Isometry3d& pose,
                         std::size_t num_points = 120)
{
    auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    cloud->reserve(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        pcl::PointXYZI point;
        point.x = static_cast<float>(i % 12) * 0.1f;
        point.y = static_cast<float>(i / 12) * 0.1f;
        point.z = static_cast<float>((i % 5) * 0.05f);
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;

    core::LioFrame frame;
    frame.stamp.nsec = stamp_nsec;
    frame.T_world_lidar = pose;
    frame.undistorted_cloud = cloud;
    frame.pose_valid = true;
    return frame;
}

Config makeCoreTestConfig()
{
    Config config;
    config.keyframe_distance_threshold = 0.5;
    config.keyframe_angle_threshold = 0.2;
    config.rhpd_submap_kf_radius = 1;
    config.rhpd_submap_voxel_size = 0.0;
    config.sc_num_exclude_recent = 0;
    config.loop_kf_gap = 1;
    return config;
}

}  // namespace

TEST(N3MappingCoreTest, ConstructAndRejectInvalidMappingFrame)
{
    N3MappingCore core(makeCoreTestConfig());

    core::LioFrame frame;
    frame.pose_valid = false;

    const auto output = core.processMappingFrame(frame);
    EXPECT_FALSE(output.success);
    EXPECT_FALSE(output.accepted_keyframe);
    EXPECT_EQ(output.keyframe_id, -1);
    EXPECT_TRUE(core.getAllKeyframes().empty());
}

TEST(N3MappingCoreTest, ProcessMappingFrameAcceptsFirstKeyframe)
{
    N3MappingCore core(makeCoreTestConfig());

    const auto output = core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity()));

    EXPECT_TRUE(output.success);
    EXPECT_TRUE(output.accepted_keyframe);
    EXPECT_EQ(output.keyframe_id, 0);
    ASSERT_NE(output.cloud_body, nullptr);
    ASSERT_NE(output.cloud_world, nullptr);
    EXPECT_EQ(output.cloud_body->size(), 120U);
    EXPECT_EQ(output.cloud_world->size(), 120U);

    const auto keyframes = core.getAllKeyframes();
    ASSERT_EQ(keyframes.size(), 1U);
    ASSERT_NE(keyframes.front(), nullptr);
    EXPECT_EQ(keyframes.front()->id, 0);
    EXPECT_EQ(keyframes.front()->rhpd_descriptor.size(), RHPD_DIM);

    const auto optimized = core.getOptimizedPoses();
    EXPECT_EQ(optimized.size(), 1U);
    EXPECT_NE(optimized.find(0), optimized.end());
}

TEST(N3MappingCoreTest, MappingFrameBelowKeyframeThresholdStillProducesPoseOutput)
{
    N3MappingCore core(makeCoreTestConfig());
    ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity())).accepted_keyframe);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = 0.1;
    const auto output = core.processMappingFrame(makeFrame(1100000000, pose));

    EXPECT_TRUE(output.success);
    EXPECT_FALSE(output.accepted_keyframe);
    EXPECT_EQ(output.keyframe_id, -1);
    EXPECT_EQ(core.getAllKeyframes().size(), 1U);
    EXPECT_NEAR(output.T_world_lidar.translation().x(), 0.1, 1e-9);
    const auto dense = core.getDenseOptimizedTrajectory();
    ASSERT_EQ(dense.size(), 2U);
    EXPECT_NEAR(dense.back().pose_world_lidar.translation().x(), 0.1, 1e-9);
}

TEST(N3MappingCoreTest, LocalizationDoesNotAppendDenseOptimizedTrajectory)
{
    Config config = makeCoreTestConfig();
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "n3mapping_core_localization_dense";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path map_path = dir / "map.pbstream";

    {
        N3MappingCore core(config);
        ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity())).accepted_keyframe);
        ASSERT_TRUE(core.saveMap(map_path.string()));
    }

    N3MappingCore localization_core(config);
    ASSERT_TRUE(localization_core.loadMap(map_path.string()));
    const auto loaded_dense = localization_core.getDenseOptimizedTrajectory();
    ASSERT_FALSE(loaded_dense.empty());

    const auto output = localization_core.processLocalizationFrame(
        makeFrame(2000000000, Eigen::Isometry3d::Identity()));
    (void)output;
    EXPECT_EQ(localization_core.getDenseOptimizedTrajectory().size(), loaded_dense.size());

    std::filesystem::remove_all(dir);
}

TEST(N3MappingCoreTest, BuildGlobalMapAccumulatesAcceptedKeyframes)
{
    Config config = makeCoreTestConfig();
    config.global_map_voxel_size = 0.0;
    N3MappingCore core(config);

    ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity(), 10)).accepted_keyframe);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = 1.0;
    ASSERT_TRUE(core.processMappingFrame(makeFrame(2000000000, pose, 10)).accepted_keyframe);

    auto global_map = core.buildGlobalMap();
    ASSERT_NE(global_map, nullptr);
    EXPECT_EQ(global_map->size(), 20U);

    bool has_translated_points = false;
    for (const auto& point : global_map->points) {
        if (point.x > 1.5f) {
            has_translated_points = true;
            break;
        }
    }
    EXPECT_TRUE(has_translated_points);
}

TEST(N3MappingCoreTest, PendingLoopClosureProcessingIsCoreOwned)
{
    N3MappingCore core(makeCoreTestConfig());

    ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity())).accepted_keyframe);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = 1.0;
    ASSERT_TRUE(core.processMappingFrame(makeFrame(2000000000, pose)).accepted_keyframe);

    const auto result = core.processPendingLoopClosures();
    EXPECT_TRUE(result.optimized);
    EXPECT_EQ(result.edge_count, 1U);
    EXPECT_EQ(result.pose_update_count, 2U);
    EXPECT_TRUE(std::isfinite(result.loop_residual_translation_before));
    EXPECT_TRUE(std::isfinite(result.loop_residual_translation_after));
    EXPECT_TRUE(std::isfinite(result.loop_residual_rotation_before));
    EXPECT_TRUE(std::isfinite(result.loop_residual_rotation_after));
    EXPECT_TRUE(std::isfinite(result.mean_pose_update_translation));
    EXPECT_TRUE(std::isfinite(result.max_pose_update_translation));
    EXPECT_TRUE(std::isfinite(result.mean_pose_update_rotation));
    EXPECT_TRUE(std::isfinite(result.max_pose_update_rotation));
    ASSERT_EQ(result.accepted_loops.size(), 1U);
    EXPECT_EQ(result.accepted_loops.front().query_id, 1);
    EXPECT_EQ(result.accepted_loops.front().match_id, 0);
    EXPECT_TRUE(result.accepted_loops.front().isValid());
}

TEST(N3MappingCoreTest, DenseTrajectorySavedAfterLoopClosureMatchesFinalKeyframePose)
{
    Config config = makeCoreTestConfig();
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "n3mapping_core_dense_loop_optimized";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path map_path = dir / "dense_loop.pbstream";

    N3MappingCore core(config);
    ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity())).accepted_keyframe);

    Eigen::Isometry3d intermediate_pose = Eigen::Isometry3d::Identity();
    intermediate_pose.translation().x() = 0.1;
    ASSERT_FALSE(core.processMappingFrame(makeFrame(1500000000, intermediate_pose)).accepted_keyframe);

    Eigen::Isometry3d second_pose = Eigen::Isometry3d::Identity();
    second_pose.translation().x() = 1.0;
    ASSERT_TRUE(core.processMappingFrame(makeFrame(2000000000, second_pose)).accepted_keyframe);
    ASSERT_EQ(core.getDenseOptimizedTrajectory().size(), 3U);

    const auto result = core.processPendingLoopClosures();
    ASSERT_TRUE(result.optimized);

    const auto final_kf = core.getKeyframe(1);
    ASSERT_NE(final_kf, nullptr);
    ASSERT_TRUE(core.saveMap(map_path.string()));

    N3MappingCore loaded(config);
    ASSERT_TRUE(loaded.loadMap(map_path.string()));
    const auto dense = loaded.getDenseOptimizedTrajectory();
    ASSERT_EQ(dense.size(), 3U);

    const auto dense_at_keyframe = std::find_if(
        dense.begin(), dense.end(), [](const core::DenseTrajectoryPose& pose) {
            return std::abs(pose.timestamp - 2.0) < 1e-9;
        });
    ASSERT_NE(dense_at_keyframe, dense.end());
    EXPECT_TRUE(dense_at_keyframe->pose_world_lidar.isApprox(final_kf->pose_optimized, 1e-6));

    std::filesystem::remove_all(dir);
}

TEST(N3MappingCoreTest, SaveLoadSmoke)
{
    Config config = makeCoreTestConfig();
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "n3mapping_core_save_load_smoke";
    std::filesystem::create_directories(dir);
    const std::filesystem::path map_path = dir / "core_smoke.pbstream";

    {
        N3MappingCore core(config);
        ASSERT_TRUE(core.processMappingFrame(makeFrame(1000000000, Eigen::Isometry3d::Identity())).accepted_keyframe);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = 1.0;
        ASSERT_TRUE(core.processMappingFrame(makeFrame(2000000000, pose)).accepted_keyframe);
        ASSERT_TRUE(core.saveMap(map_path.string()));
    }

    N3MappingCore loaded(config);
    ASSERT_TRUE(loaded.loadMap(map_path.string()));
    EXPECT_TRUE(loaded.mapLoaded());
    EXPECT_EQ(loaded.getAllKeyframes().size(), 2U);

    std::filesystem::remove(map_path);
}

}  // namespace test
}  // namespace n3mapping
