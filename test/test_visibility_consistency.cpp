#include "n3mapping/visibility_consistency.h"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

pcl::PointCloud<pcl::PointXYZI> makeFourRayCloud(double range) {
  pcl::PointCloud<pcl::PointXYZI> cloud;
  const std::array<Eigen::Vector3d, 4> points = {
      Eigen::Vector3d(range, 0.0, 0.0),
      Eigen::Vector3d(0.0, range, 0.0),
      Eigen::Vector3d(-range, 0.0, 0.0),
      Eigen::Vector3d(0.0, -range, 0.0),
  };
  for (const auto &value : points) {
    pcl::PointXYZI point;
    point.x = static_cast<float>(value.x());
    point.y = static_cast<float>(value.y());
    point.z = static_cast<float>(value.z());
    cloud.push_back(point);
  }
  return cloud;
}

TEST(VisibilityConsistencyTest, ExactPredictionExplainsEveryObservedRay) {
  const auto query = makeFourRayCloud(5.0);
  const auto result = evaluateVisibilityConsistency(
      query, query, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.observed_bins, 4u);
  EXPECT_EQ(result.consistent_bins, 4u);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 1.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_ratio, 0.0);
  EXPECT_NEAR(result.evidence_log_odds, std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.median_abs_range_error_m, 0.0);
}

TEST(VisibilityConsistencyTest,
     PredictedForegroundContradictsObservedFreeSpace) {
  const auto query = makeFourRayCloud(5.0);
  const auto foreground_map = makeFourRayCloud(2.0);
  const auto result = evaluateVisibilityConsistency(
      foreground_map, query, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.common_bins, 4u);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 0.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_ratio, 1.0);
  EXPECT_NEAR(result.evidence_log_odds, -std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.median_abs_range_error_m, 3.0);
}

TEST(VisibilityConsistencyTest,
     SurfaceInFrontOfAMatchingReturnIsOcclusionNotContradiction) {
  // What a place mapped twice looks like: the map holds the wall the query
  // measures, and also nearer returns the other pass saw from elsewhere.
  const auto query = makeFourRayCloud(5.0);
  auto doubled_map = makeFourRayCloud(5.0);
  const auto nearer = makeFourRayCloud(2.0);
  doubled_map += nearer;

  const auto nearest_wins = evaluateVisibilityConsistency(
      doubled_map, query, Eigen::Isometry3d::Identity());
  ASSERT_TRUE(nearest_wins.valid);
  EXPECT_DOUBLE_EQ(nearest_wins.consistency_ratio, 0.0);
  EXPECT_DOUBLE_EQ(nearest_wins.foreground_conflict_ratio, 1.0);

  VisibilityConsistencyOptions options;
  options.occlusion_aware = true;
  const auto occlusion_aware = evaluateVisibilityConsistency(
      doubled_map, query, Eigen::Isometry3d::Identity(), options);
  ASSERT_TRUE(occlusion_aware.valid);
  EXPECT_DOUBLE_EQ(occlusion_aware.consistency_ratio, 1.0);
  EXPECT_DOUBLE_EQ(occlusion_aware.foreground_conflict_ratio, 0.0);
  EXPECT_GT(occlusion_aware.evidence_log_odds, nearest_wins.evidence_log_odds);
}

TEST(VisibilityConsistencyTest,
     OcclusionAwarenessStillContradictsAWallThatIsNotThere) {
  // Nothing at the measured range: every accumulated return falls short, so the
  // bearing is a contradiction whichever return is scored against.
  const auto query = makeFourRayCloud(5.0);
  const auto foreground_map = makeFourRayCloud(2.0);

  VisibilityConsistencyOptions options;
  options.occlusion_aware = true;
  const auto result = evaluateVisibilityConsistency(
      foreground_map, query, Eigen::Isometry3d::Identity(), options);

  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 0.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_ratio, 1.0);
}

TEST(VisibilityConsistencyTest, SymmetricObservationProvidesEqualPoseEvidence) {
  const auto symmetric = makeFourRayCloud(5.0);
  Eigen::Isometry3d quarter_turn = Eigen::Isometry3d::Identity();
  quarter_turn.rotate(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()));

  const auto identity = evaluateVisibilityConsistency(
      symmetric, symmetric, Eigen::Isometry3d::Identity());
  const auto rotated =
      evaluateVisibilityConsistency(symmetric, symmetric, quarter_turn);

  ASSERT_TRUE(identity.valid);
  ASSERT_TRUE(rotated.valid);
  EXPECT_DOUBLE_EQ(identity.consistency_ratio, rotated.consistency_ratio);
  EXPECT_DOUBLE_EQ(identity.evidence_log_odds, rotated.evidence_log_odds);
}

} // namespace
} // namespace n3mapping
