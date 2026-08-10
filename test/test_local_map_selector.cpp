#include "n3mapping/local_map_selector.h"

#include <map>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

Keyframe::Ptr makeKeyframe(int64_t id, double x, double y = 0.0,
                           double z = 0.0) {
  auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
  cloud->push_back(pcl::PointXYZI());
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(x, y, z);
  return Keyframe::create(id, static_cast<double>(id), pose, cloud);
}

TEST(KeyframeSpatialIndexTest, RadiusSearchIsDeterministicAndRevisionAware) {
  KeyframeSpatialIndex index;
  auto left = makeKeyframe(0, -1.0);
  auto right = makeKeyframe(1, 1.0);
  auto far = makeKeyframe(2, 10.0);
  std::vector<Keyframe::Ptr> keyframes = {right, far, left};

  const auto first =
      index.radiusSearch(keyframes, {1, 1, 1}, Eigen::Vector3d::Zero(), 2.0, 3);
  ASSERT_EQ(first.matches.size(), 2u);
  EXPECT_EQ(first.matches[0].keyframe_id, 0);
  EXPECT_EQ(first.matches[1].keyframe_id, 1);
  EXPECT_TRUE(first.index_rebuilt);

  const auto second =
      index.radiusSearch(keyframes, {1, 1, 1}, Eigen::Vector3d::Zero(), 2.0, 3);
  EXPECT_FALSE(second.index_rebuilt);

  far->pose_optimized.translation().x() = 0.25;
  const auto after_pose_revision =
      index.radiusSearch(keyframes, {1, 1, 2}, Eigen::Vector3d::Zero(), 2.0, 3);
  ASSERT_EQ(after_pose_revision.matches.size(), 3u);
  EXPECT_EQ(after_pose_revision.matches[0].keyframe_id, 2);
  EXPECT_TRUE(after_pose_revision.index_rebuilt);
}

TEST(LocalMapSelectorTest, UnionsSpatialAnchorAndRecentTail) {
  Config config;
  KeyframeManager manager(config);
  for (int id = 0; id < 4; ++id) {
    const double x =
        id < 2 ? static_cast<double>(id) : 8.0 + static_cast<double>(id);
    const auto keyframe = makeKeyframe(id, x);
    manager.addKeyframe(static_cast<double>(id), keyframe->pose_optimized,
                        keyframe->cloud);
  }
  manager.getKeyframe(0)->is_from_loaded_map = true;

  LocalMapSelector selector(manager);
  LocalMapSelectionRequest request;
  request.predicted_pose.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
  request.anchor_id = 1;
  request.spatial_radius = 1.1;
  request.max_spatial_keyframes = 2;
  request.recent_tail_count = 2;
  request.map_revision = manager.revision();

  const LocalMapSelection selection = selector.select(request);

  EXPECT_EQ(selection.reason, "spatial_anchor_recent");
  EXPECT_EQ(selection.keyframe_ids, (std::vector<int64_t>{0, 1, 3, 2}));
  EXPECT_EQ(selection.selected_points, 4u);
  EXPECT_EQ(selection.loaded_keyframes, 1u);
  EXPECT_EQ(selection.current_keyframes, 3u);
  EXPECT_TRUE(selection.requested_revision_matched);
}

} // namespace
} // namespace n3mapping
