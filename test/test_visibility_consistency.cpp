#include "n3mapping/visibility_consistency.h"
#include "../src/visibility_angular_bins.h"

#include <array>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

int referenceAngularBin(double y, double x, double resolution, bool azimuth) {
  double degrees = std::atan2(y, x) * 180.0 / M_PI + (azimuth ? 180.0 : 90.0);
  if (azimuth) {
    if (degrees >= 360.0) degrees -= 360.0;
    if (degrees < 0.0) degrees += 360.0;
  }
  const int count = std::max(1, static_cast<int>(std::ceil((azimuth ? 360.0 : 180.0) / resolution)));
  return std::clamp(static_cast<int>(std::floor(degrees / resolution)), 0, count - 1);
}

TEST(VisibilityConsistencyTest, AngularLookupPreservesBinsAcrossQuadrantsAndScales) {
  std::mt19937_64 random(20261017);
  std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
  for (const double resolution : {0.001, 0.1, 0.5, 1.0, 1.234567, 2.0, 4.0}) {
    for (const bool azimuth : {false, true}) {
      const int count = std::ceil((azimuth ? 360.0 : 180.0) / resolution);
      const visibility_detail::AngularBinLookup bins(
          azimuth ? 180.0 : 90.0, resolution, count, azimuth);
      for (int i = 0; i < 10000; ++i) {
        const double scale = i % 3 == 0 ? 1e-300 : (i % 3 == 1 ? 1.0 : 1e300);
        const double y = coordinate(random) * scale;
        const double x = (azimuth ? coordinate(random) : std::abs(coordinate(random))) * scale;
        ASSERT_EQ(bins(y, x), referenceAngularBin(y, x, resolution, azimuth));
      }
      for (const double x : {-1.0, -0.0, 0.0, 1.0}) {
        if (!azimuth && x < 0.0) continue;
        for (const double y : {-1.0, -0.0, 0.0, 1.0})
          EXPECT_EQ(bins(y, x), referenceAngularBin(y, x, resolution, azimuth));
      }
    }
  }
}

TEST(VisibilityConsistencyTest, AngularLookupKeepsOriginalRoundingAtEveryBinBoundary) {
  for (const double resolution : {0.001, 0.5, 1.234567, 4.0}) {
    for (const bool azimuth : {false, true}) {
      const int count = std::ceil((azimuth ? 360.0 : 180.0) / resolution);
      const double offset = azimuth ? 180.0 : 90.0;
      const visibility_detail::AngularBinLookup bins(offset, resolution, count, azimuth);
      for (int i = 0; i < count; ++i) {
        const double boundary = (i * resolution - offset) * M_PI / 180.0;
        for (const double delta : {-1e-12, 0.0, 1e-12}) {
          const double y = std::sin(boundary + delta);
          const double x = std::cos(boundary + delta);
          ASSERT_EQ(bins(y, x), referenceAngularBin(y, x, resolution, azimuth));
        }
      }
    }
  }
}

TEST(VisibilityConsistencyTest, ElevationLookupRetainsHypotAtBoundariesAndExtremeScales) {
  std::mt19937_64 random(3588);
  std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
  for (const double resolution : {0.001, 1.0, 1.234567, 4.0}) {
    const int count = std::ceil(180.0 / resolution);
    const visibility_detail::AngularBinLookup bins(90.0, resolution, count, false);
    for (const double scale : {1e-300, 1e-162, 1e-154, 1.0, 1e150, 1e300}) {
      for (int i = 0; i < 10000; ++i) {
        const double x = coordinate(random) * scale;
        const double y = coordinate(random) * scale;
        const double z = coordinate(random) * scale;
        ASSERT_EQ(bins.elevation(z, x, y),
                  referenceAngularBin(z, std::hypot(x, y), resolution, false));
      }
    }
    for (int i = 0; i < count; ++i) {
      const double boundary = (i * resolution - 90.0) * M_PI / 180.0;
      for (const double delta : {-1e-12, 0.0, 1e-12}) {
        const double z = std::sin(boundary + delta);
        const double x = std::cos(boundary + delta) * std::cos(0.713);
        const double y = std::cos(boundary + delta) * std::sin(0.713);
        ASSERT_EQ(bins.elevation(z, x, y),
                  referenceAngularBin(z, std::hypot(x, y), resolution, false));
      }
    }
  }
}

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
  EXPECT_EQ(result.known_bins, 4u);
  EXPECT_EQ(result.unknown_bins, 0u);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 1.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_ratio, 0.0);
  EXPECT_NEAR(result.evidence_log_odds, std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.known_fraction, 1.0);
  EXPECT_DOUBLE_EQ(result.consistent_given_known, 1.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_given_known, 0.0);
  EXPECT_NEAR(result.evidence_log_odds_given_known, std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.median_abs_range_error_m, 0.0);
}

TEST(VisibilityConsistencyTest, PreparedObservationPreservesOcclusionAndInvalidResults) {
  auto query = makeFourRayCloud(5.0);
  auto map = query;
  map += makeFourRayCloud(2.0);
  query.push_back(pcl::PointXYZI{});
  query.back().x = std::numeric_limits<float>::quiet_NaN();
  for (bool occlusion_aware : {false, true}) {
    VisibilityConsistencyOptions options;
    options.occlusion_aware = occlusion_aware;
    const auto prepared = prepareVisibilityObservation(query, options);
    for (double translation : {0.0, 1.0, 100.0}) {
      auto pose = Eigen::Isometry3d::Identity();
      pose.translation().x() = translation;
      const auto direct = evaluateVisibilityConsistency(map, query, pose, options);
      const auto reused = evaluatePreparedVisibilityConsistency(map, prepared, pose);
      EXPECT_EQ(reused.valid, direct.valid);
      EXPECT_EQ(reused.observed_bins, direct.observed_bins);
      EXPECT_EQ(reused.predicted_bins, direct.predicted_bins);
      EXPECT_EQ(reused.common_bins, direct.common_bins);
      EXPECT_EQ(reused.consistent_bins, direct.consistent_bins);
      EXPECT_EQ(reused.foreground_conflict_bins, direct.foreground_conflict_bins);
      EXPECT_DOUBLE_EQ(reused.consistency_ratio, direct.consistency_ratio);
      EXPECT_DOUBLE_EQ(reused.foreground_conflict_ratio, direct.foreground_conflict_ratio);
      if (direct.valid) {
        EXPECT_DOUBLE_EQ(reused.evidence_log_odds, direct.evidence_log_odds);
      }
    }
    auto invalid_pose = Eigen::Isometry3d::Identity();
    invalid_pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(evaluatePreparedVisibilityConsistency(map, prepared, invalid_pose).valid);
  }
  EXPECT_FALSE(evaluatePreparedVisibilityConsistency(map,
      prepareVisibilityObservation({}), Eigen::Isometry3d::Identity()).valid);
  VisibilityConsistencyOptions invalid;
  invalid.range_tolerance_m = 0.0;
  EXPECT_TRUE(prepareVisibilityObservation(query, invalid).ranges.empty());
}

TEST(VisibilityConsistencyTest, MalformedPreparedObservationIsRejectedBeforeIndexing) {
  const auto query = makeFourRayCloud(5.0);
  VisibilityConsistencyOptions options;
  options.occlusion_aware = true;
  auto prepared = prepareVisibilityObservation(query, options);
  prepared.ranges.pop_back();
  EXPECT_FALSE(evaluatePreparedVisibilityConsistency(query, prepared,
      Eigen::Isometry3d::Identity()).valid);
  prepared = prepareVisibilityObservation(query, options);
  for (double resolution : {0.0, -1.0, 1e-100,
       std::numeric_limits<double>::quiet_NaN()}) {
    prepared.resolution_deg = resolution;
    EXPECT_FALSE(evaluatePreparedVisibilityConsistency(query, prepared,
        Eigen::Isometry3d::Identity()).valid);
  }
}

TEST(VisibilityConsistencyTest,
     PredictedForegroundContradictsObservedFreeSpace) {
  const auto query = makeFourRayCloud(5.0);
  const auto foreground_map = makeFourRayCloud(2.0);
  const auto result = evaluateVisibilityConsistency(
      foreground_map, query, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.common_bins, 4u);
  EXPECT_EQ(result.known_bins, 4u);
  EXPECT_EQ(result.unknown_bins, 0u);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 0.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_ratio, 1.0);
  EXPECT_NEAR(result.evidence_log_odds, -std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.known_fraction, 1.0);
  EXPECT_DOUBLE_EQ(result.consistent_given_known, 0.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_given_known, 1.0);
  EXPECT_NEAR(result.evidence_log_odds_given_known, -std::log(9.0), 1e-12);
  EXPECT_DOUBLE_EQ(result.median_abs_range_error_m, 3.0);
}

TEST(VisibilityConsistencyTest,
     KnownNormalizationSeparatesAgreementFromMissingCoverage) {
  const auto query = makeFourRayCloud(5.0);
  auto partial_map = makeFourRayCloud(5.0);
  partial_map.erase(partial_map.begin() + 2, partial_map.end());

  const auto result = evaluateVisibilityConsistency(
      partial_map, query, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.observed_bins, 4u);
  EXPECT_EQ(result.known_bins, 2u);
  EXPECT_EQ(result.unknown_bins, 2u);
  EXPECT_DOUBLE_EQ(result.known_fraction, 0.5);
  EXPECT_DOUBLE_EQ(result.consistency_ratio, 0.5);
  EXPECT_DOUBLE_EQ(result.consistent_given_known, 1.0);
  EXPECT_DOUBLE_EQ(result.foreground_conflict_given_known, 0.0);
  EXPECT_NEAR(result.evidence_log_odds, 0.0, 1e-12);
  EXPECT_NEAR(result.evidence_log_odds_given_known, std::log(5.0), 1e-12);
}

TEST(VisibilityConsistencyTest, NoKnownBinsRemainUndefinedShadowEvidence) {
  const auto query = makeFourRayCloud(5.0);
  pcl::PointCloud<pcl::PointXYZI> map;
  pcl::PointXYZI outside_range;
  outside_range.x = 100.0f;
  map.push_back(outside_range);

  const auto result = evaluateVisibilityConsistency(
      map, query, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.known_bins, 0u);
  EXPECT_EQ(result.unknown_bins, 4u);
  EXPECT_DOUBLE_EQ(result.known_fraction, 0.0);
  EXPECT_TRUE(std::isnan(result.consistent_given_known));
  EXPECT_TRUE(std::isnan(result.foreground_conflict_given_known));
  EXPECT_TRUE(std::isnan(result.evidence_log_odds_given_known));
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
