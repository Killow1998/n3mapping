#include "n3mapping/visibility_consistency.h"
#include "../src/visibility_angular_bins.h"

#include <array>
#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

namespace n3mapping {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr planarWitnessRoom(bool walls = true) {
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  const auto add = [&](double x, double y, double z) {
    pcl::PointXYZI point;
    point.x = x; point.y = y; point.z = z;
    cloud->push_back(point);
  };
  for (int x = -40; x <= 40; ++x)
    for (int y = -30; y <= 30; ++y) {
      add(x * 0.2, y * 0.2, -0.8);
      add(x * 0.2, y * 0.2, 2.2);
    }
  if (walls)
    for (int z = -3; z <= 10; ++z) {
      for (int x = -40; x <= 40; ++x) {
        add(x * 0.2, -6.0, z * 0.2);
        add(x * 0.2, 6.0, z * 0.2);
      }
      for (int y = -30; y <= 30; ++y) {
        add(-8.0, y * 0.2, z * 0.2);
        add(8.0, y * 0.2, z * 0.2);
      }
    }
  return cloud;
}

TEST(VisibilityConsistencyTest, MatchingFloorCannotHideContradictoryWalls) {
  const auto query = planarWitnessRoom();
  auto map = *query;
  for (auto &point : map)
    if (point.z > -0.7 && point.z < 2.1) {
      point.x *= 1.4;
      point.y *= 1.4;
    }
  auto observation = prepareVisibilityObservation(*query);
  prepareVisibilityPlanarSupport(query, Eigen::Vector3d::UnitZ(), &observation);
  const auto result = evaluatePreparedVisibilityConsistency(
      map, observation, Eigen::Isometry3d::Identity());
  ASSERT_TRUE(result.valid);
  ASSERT_GT(result.consistency_ratio, 0.5); // The old pooled witness passes.
  EXPECT_LT(result.planar_min_support, 0.5);
}

TEST(VisibilityConsistencyTest, PlanarWitnessHandlesTiltAndMissingReturns) {
  const auto map = planarWitnessRoom();
  for (double tilt : {0.0, 0.35, 0.8}) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = (Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()) *
                     Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    auto query = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    // Missing query returns do not turn the remaining measured surfaces into
    // contradiction. Use a deterministic dropout independent of surface type.
    for (std::size_t i = 0; i < map->size(); ++i) {
      if (i % 4 == 0) continue;
      const auto q = pose.inverse() * (*map)[i].getVector3fMap().cast<double>();
      pcl::PointXYZI point;
      point.x = q.x(); point.y = q.y(); point.z = q.z();
      query->push_back(point);
    }
    auto observation = prepareVisibilityObservation(*query);
    prepareVisibilityPlanarSupport(query, pose.rotation().transpose() * Eigen::Vector3d::UnitZ(),
                                  &observation);
    const auto result = evaluatePreparedVisibilityConsistency(*map, observation, pose);
    EXPECT_GT(result.planar_min_support, 0.5) << "sensor tilt=" << tilt;
  }
}

TEST(VisibilityConsistencyTest, OtherDirectionsCannotHideUnsupportedTranslation) {
  const auto query = planarWitnessRoom();
  auto map = *query;
  for (auto &point : map)
    if (std::abs(point.x) > 7.9 && point.z > -0.7 && point.z < 2.1)
      point.x *= 1.4;
  auto observation = prepareVisibilityObservation(*query);
  prepareVisibilityPlanarSupport(query, Eigen::Vector3d::UnitZ(), &observation);
  const auto result = evaluatePreparedVisibilityConsistency(
      map, observation, Eigen::Isometry3d::Identity());
  ASSERT_GT(result.consistency_ratio, 0.8);
  EXPECT_LT(result.planar_min_support, 0.5);
}

TEST(VisibilityConsistencyTest, FloorAloneDoesNotEstablishHorizontalLocation) {
  const auto floor = planarWitnessRoom(false);
  for (double tilt : {0.0, 0.35, 0.8}) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
    auto query = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::transformPointCloud(*floor, *query, pose.inverse().matrix());
    auto observation = prepareVisibilityObservation(*query);
    prepareVisibilityPlanarSupport(query, pose.rotation().transpose() * Eigen::Vector3d::UnitZ(),
                                  &observation);
    const auto result = evaluatePreparedVisibilityConsistency(*floor, observation, pose);
    ASSERT_GT(result.consistency_ratio, 0.9);
    EXPECT_FALSE(std::isfinite(result.planar_min_support)) << "sensor tilt=" << tilt;
  }
}

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
      const std::vector<std::uint32_t> subset{0, 2, 4, 6};
      pcl::PointCloud<pcl::PointXYZI> cropped;
      for (auto index : subset) cropped.push_back(map[index]);
      const auto copied = evaluatePreparedVisibilityConsistency(cropped, prepared, pose);
      const auto indexed = evaluatePreparedVisibilityConsistency(map, prepared, pose, &subset);
      EXPECT_EQ(indexed.valid, copied.valid);
      EXPECT_EQ(indexed.predicted_bins, copied.predicted_bins);
      EXPECT_EQ(indexed.common_bins, copied.common_bins);
      EXPECT_EQ(indexed.consistent_bins, copied.consistent_bins);
      EXPECT_EQ(indexed.foreground_conflict_bins, copied.foreground_conflict_bins);
      if (copied.valid) {
        EXPECT_DOUBLE_EQ(indexed.evidence_log_odds, copied.evidence_log_odds);
        EXPECT_DOUBLE_EQ(indexed.median_abs_range_error_m, copied.median_abs_range_error_m);
        EXPECT_DOUBLE_EQ(indexed.p90_abs_range_error_m, copied.p90_abs_range_error_m);
      }
      const std::vector<std::uint32_t> invalid_index{static_cast<std::uint32_t>(map.size())};
      EXPECT_FALSE(evaluatePreparedVisibilityConsistency(map, prepared, pose, &invalid_index).valid);

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
     ExperimentalRangeMatchingCanHideForegroundContradictions) {
  // Characterize the experimental mode's limitation: matching the background
  // erases the nearer return's contradiction. This does not certify visibility
  // through an opaque foreground surface; the Product profile leaves it off.
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
