#include <filesystem>

#include <gtest/gtest.h>

#include "n3mapping/core/n3mapping_core.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (float x = -4.0f; x <= 4.0f; x += 0.25f) {
        for (float z = -0.5f; z <= 2.0f; z += 0.25f) {
            pcl::PointXYZI left;
            left.x = x;
            left.y = -2.0f;
            left.z = z;
            left.intensity = 1.0f;
            cloud->push_back(left);

            pcl::PointXYZI right = left;
            right.y = 2.0f;
            cloud->push_back(right);
        }
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(N3MappingCoreTest, ProcessesExternalLioFrameAndSavesMap) {
    Config config;
    config.map_save_path = "/tmp/n3mapping_core_test";
    config.keyframe_distance_threshold = 0.5;
    config.rhpd_submap_voxel_size = 0.0;
    std::filesystem::remove_all(config.map_save_path);
    std::filesystem::create_directories(config.map_save_path);

    core::N3MappingCore mapping_core(config);

    core::LioFrame invalid;
    invalid.pose_valid = false;
    EXPECT_FALSE(mapping_core.processLioFrame(invalid).accepted_keyframe);

    core::LioFrame frame;
    frame.pose_valid = true;
    frame.stamp.nsec = 100000000;
    frame.T_world_lidar = Eigen::Isometry3d::Identity();
    frame.undistorted_cloud = makeCloud();

    const auto out0 = mapping_core.processLioFrame(frame);
    EXPECT_TRUE(out0.accepted_keyframe);
    EXPECT_EQ(out0.keyframe_id, 0);

    const auto out1 = mapping_core.processLioFrame(frame);
    EXPECT_FALSE(out1.accepted_keyframe);

    const std::string map_path = config.map_save_path + "/core_map.pbstream";
    ASSERT_TRUE(mapping_core.saveMap(map_path));
    ASSERT_TRUE(std::filesystem::exists(map_path));

    core::N3MappingCore loaded_core(config);
    EXPECT_TRUE(loaded_core.loadMap(map_path));

    std::filesystem::remove_all(config.map_save_path);
}

TEST(N3MappingCoreTest, RelocalizeRejectsInvalidOrUnloadedInput) {
    Config config;
    core::N3MappingCore mapping_core(config);

    core::LioFrame invalid;
    invalid.pose_valid = false;
    EXPECT_FALSE(mapping_core.relocalize(invalid).success);

    core::LioFrame empty;
    empty.pose_valid = true;
    empty.T_world_lidar = Eigen::Isometry3d::Identity();
    empty.undistorted_cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    EXPECT_FALSE(mapping_core.relocalize(empty).success);

    core::LioFrame valid_without_map;
    valid_without_map.pose_valid = true;
    valid_without_map.T_world_lidar = Eigen::Isometry3d::Identity();
    valid_without_map.undistorted_cloud = makeCloud();
    EXPECT_FALSE(mapping_core.relocalize(valid_without_map).success);
}

}  // namespace test
}  // namespace n3mapping
