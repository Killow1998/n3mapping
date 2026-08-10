#include "n3mapping/submap_builder.h"

#include <gtest/gtest.h>

#include <limits>

namespace n3mapping {
namespace {

Keyframe::Ptr makeKeyframe(int64_t id, MapSessionId session_id, double x,
                           std::size_t point_count = 2) {
  auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
  for (std::size_t i = 0; i < point_count; ++i) {
    pcl::PointXYZI point;
    point.x = static_cast<float>(i);
    point.y = static_cast<float>(id);
    point.z = 0.0f;
    point.intensity = static_cast<float>(10 + i);
    cloud->push_back(point);
  }
  Eigen::Isometry3d session_pose = Eigen::Isometry3d::Identity();
  session_pose.translation().x() = x;
  Eigen::Isometry3d map_pose = session_pose;
  map_pose.translation().y() = static_cast<double>(session_id) * 10.0;
  return Keyframe::create(id, static_cast<double>(id), session_pose, map_pose,
                          cloud, session_id);
}

std::vector<MapSessionInfo> makeSessions() {
  MapSessionInfo session0;
  session0.id = 0;
  MapSessionInfo session1;
  session1.id = 1;
  return {session0, session1};
}

TEST(SubmapBuilderTest, DisabledBuilderDoesNotChangeState) {
  SubmapBuilder builder(SubmapBuilderOptions{});
  EXPECT_FALSE(builder.appendKeyframe(makeKeyframe(0, 0, 0.0)));
  EXPECT_TRUE(builder.getSubmaps().empty());
}

TEST(SubmapBuilderTest, RejectsEmptyAndNonFiniteClouds) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.cloud_max_bytes = 4 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  auto empty = makeKeyframe(0, 0, 0.0);
  empty->cloud->clear();
  EXPECT_FALSE(builder.appendKeyframe(empty));

  auto non_finite = makeKeyframe(1, 0, 0.0);
  non_finite->cloud->front().x =
      std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(builder.appendKeyframe(non_finite));
  EXPECT_TRUE(builder.getSubmaps().empty());
}

TEST(SubmapBuilderTest, GroupsDeterministicallyAndKeepsOneBoundedCloud) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 3 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(0, 0, 5.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(1, 0, 7.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(2, 0, 9.0)));

  const auto submaps = builder.getSubmaps();
  ASSERT_EQ(submaps.size(), 2u);
  EXPECT_EQ(submaps[0].keyframe_ids, (std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(submaps[0].closed);
  EXPECT_EQ(submaps[0].cloud_point_count, 3u);
  EXPECT_TRUE(submaps[0].cloud_truncated);
  EXPECT_EQ(submaps[0].materializedCloudBytes(),
            3u * sizeof(pcl::PointXYZI));
  EXPECT_EQ(submaps[0].registration_cloud,
            submaps[0].visualization_cloud);
  ASSERT_EQ(submaps[0].registration_cloud->size(), 3u);
  EXPECT_FLOAT_EQ(submaps[0].registration_cloud->points[2].x, 2.0f);
  EXPECT_EQ(submaps[0].descriptor_keyframe_id, 0);
  EXPECT_EQ(submaps[0].content_revision, 3u);

  EXPECT_EQ(submaps[1].keyframe_ids, (std::vector<int64_t>{2}));
  EXPECT_FALSE(submaps[1].closed);
  EXPECT_EQ(submaps[1].content_revision, 1u);
}

TEST(SubmapBuilderTest, SessionBoundaryClosesWithoutCrossSessionMembership) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 10;
  options.cloud_max_bytes = 32 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(0, 0, 0.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(1, 1, 0.0)));
  const auto before_more_input = builder.getSubmaps();
  ASSERT_EQ(before_more_input.size(), 2u);
  EXPECT_TRUE(before_more_input[0].closed);
  EXPECT_EQ(before_more_input[0].session_id, 0u);
  EXPECT_EQ(before_more_input[1].session_id, 1u);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(2, 1, 1.0)));
  const auto after_more_input = builder.getSubmaps();
  EXPECT_EQ(after_more_input[0].keyframe_ids,
            before_more_input[0].keyframe_ids);
  EXPECT_EQ(after_more_input[0].content_revision,
            before_more_input[0].content_revision);
  EXPECT_EQ(after_more_input[1].keyframe_ids,
            (std::vector<int64_t>{1, 2}));
}

TEST(SubmapBuilderTest, LoadMaterializesAndFailureIsTransactional) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 4 * sizeof(pcl::PointXYZI);
  SubmapBuilder source(options);
  const std::vector<Keyframe::Ptr> keyframes = {
      makeKeyframe(0, 0, 0.0), makeKeyframe(1, 0, 1.0)};
  ASSERT_TRUE(source.appendKeyframe(keyframes[0]));
  ASSERT_TRUE(source.appendKeyframe(keyframes[1]));

  const auto serialized = source.getSubmaps();
  SubmapBuilder loaded(options);
  ASSERT_TRUE(loaded.loadSubmaps(serialized, keyframes, makeSessions()));
  const auto round_trip = loaded.getSubmaps();
  ASSERT_EQ(round_trip.size(), 1u);
  EXPECT_EQ(round_trip[0].keyframe_ids,
            serialized[0].keyframe_ids);
  EXPECT_EQ(round_trip[0].cloud_point_count,
            serialized[0].cloud_point_count);
  EXPECT_EQ(round_trip[0].registration_cloud,
            round_trip[0].visualization_cloud);

  auto invalid = serialized;
  invalid[0].id = kInvalidSubmapId - 1u;
  EXPECT_FALSE(loaded.loadSubmaps(invalid, keyframes, makeSessions()));
  const auto after_failed_load = loaded.getSubmaps();
  ASSERT_EQ(after_failed_load.size(), 1u);
  EXPECT_EQ(after_failed_load[0].id, serialized[0].id);
  EXPECT_EQ(after_failed_load[0].cloud_point_count,
            serialized[0].cloud_point_count);
}

}  // namespace
}  // namespace n3mapping
