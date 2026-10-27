#include <atomic>
#include <future>
#include <sstream>
#include <thread>

#include <gtest/gtest.h>
#include <json/json.h>

#include "n3mapping/core/realtime_output.h"

namespace n3mapping::core {
namespace {

OdometrySource source(int64_t stamp) {
  return {stamp, "continuous", "body", Eigen::Isometry3d::Identity(), true};
}

BackendOutput trusted(double x = 0.0) {
  BackendOutput output;
  output.success = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  output.T_world_lidar.translation().x() = x;
  return output;
}

Json::Value json(const std::string& text) {
  Json::Value result;
  std::istringstream stream(text);
  stream >> result;
  return result;
}

TEST(RealtimeOutput, CloudFirstEstimateReportsAppliedAnchorNotNewBackendCorrection) {
  RealtimeOutput live(CoreRunMode::LOCALIZATION, "map", 4);
  auto backend = trusted();
  ASSERT_TRUE(live.acceptBackend(source(90000000000), 90.0, backend).live_authorized);
  const auto cloud = live.observeCloud(source(100000000000), 100000000000,
                                       100000000001, "body");
  ASSERT_TRUE(cloud);
  const auto cloud_status = json(cloud->status_json);
  EXPECT_DOUBLE_EQ(cloud_status["stamp"].asDouble(), 0.0);
  EXPECT_DOUBLE_EQ(cloud_status["correction_stamp"].asDouble(), 0.0);
  backend = trusted(10.0);
  ASSERT_TRUE(live.acceptBackend(source(99000000000), 99.0, backend).live_authorized);
  const auto estimate = live.observeOdometry(source(100000000000), 101000000000, true);
  ASSERT_TRUE(estimate.pose);
  const auto status = json(estimate.status_json);
  EXPECT_EQ(status["state"], "tracking");
  EXPECT_DOUBLE_EQ(status["stamp"].asDouble(), 100.0);
  EXPECT_DOUBLE_EQ(status["observation_stamp"].asDouble(), 99.0);
  EXPECT_DOUBLE_EQ(status["correction_stamp"].asDouble(), 90.0);
  EXPECT_DOUBLE_EQ(estimate.pose->pose.translation().x(), 0.0);
  EXPECT_EQ(status["projection"]["basis_revision"],
            cloud_status["projection"]["basis_revision"]);
}

TEST(RealtimeOutput, FrozenProjectionCarriesExactGeometryAndIntegerAssociation) {
  RealtimeOutput live(CoreRunMode::LOCALIZATION, "map", 8);
  constexpr int64_t stamp = 1700000000000000001LL;
  auto raw = source(stamp);
  raw.pose.translation() = Eigen::Vector3d(2.0, 3.0, 4.0);
  auto backend = trusted();
  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
  correction.linear() = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
  correction.translation() = Eigen::Vector3d(5.0, 0.0, 1.0);
  backend.T_world_lidar = correction * raw.pose;
  auto anchor = raw;
  anchor.stamp_nsec = stamp - 10;
  ASSERT_TRUE(live.acceptBackend(anchor, 1.0, backend).live_authorized);
  const auto odom = live.observeOdometry(raw, stamp + 2, true);
  ASSERT_TRUE(odom.pose);
  EXPECT_TRUE(odom.pose->map_from_source.matrix().isApprox(correction.matrix(), 1e-12));
  EXPECT_TRUE(odom.pose->source_from_body.matrix().isApprox(raw.pose.matrix(), 1e-12));
  backend.T_world_lidar = raw.pose;
  anchor.stamp_nsec = stamp - 1;
  live.acceptBackend(anchor, 2.0, backend);
  const auto cloud = live.observeCloud(raw, stamp + 2, stamp + 1, "body");
  ASSERT_TRUE(cloud);
  EXPECT_TRUE(cloud->projection.pose.matrix().isApprox(odom.pose->pose.matrix(), 1e-12));
  auto cloud_status = json(cloud->status_json);
  auto odom_status = json(odom.status_json);
  const auto& metadata = cloud_status["projection"];
  EXPECT_EQ(metadata["odom_stamp_nsec"].asUInt64(), stamp);
  EXPECT_EQ(metadata["cloud_stamp_nsec"].asUInt64(), stamp + 1);
  EXPECT_EQ(metadata["basis_revision"].asUInt64(), 1U);
  EXPECT_TRUE(metadata["available"].asBool());
  EXPECT_DOUBLE_EQ(metadata["map_from_source"]["translation"][0].asDouble(), 5.0);
  EXPECT_DOUBLE_EQ(metadata["source_from_body"]["translation"][1].asDouble(), 3.0);
  EXPECT_DOUBLE_EQ(metadata["map_from_source"]["quaternion_xyzw"][0].asDouble(),
                   Eigen::Quaterniond(correction.linear()).x());
  cloud_status.removeMember("projection");
  odom_status.removeMember("projection");
  EXPECT_EQ(cloud_status, odom_status);
}

TEST(RealtimeOutput, OnlyTrustedCorrectionRestoresUnavailableGeneration) {
  RealtimeOutput live(CoreRunMode::LOCALIZATION, "map", 8);
  auto backend = trusted();
  live.acceptBackend(source(90), 90e-9, backend);
  backend.relocalization_state = RelocalizationState::RECENTLY_LOST;
  backend.pose_source = PoseSource::ODOM_PREDICTED;
  EXPECT_FALSE(live.acceptBackend(source(99), 99e-9, backend).live_authorized);
  const auto ready = live.observeOdometry(source(100), 100, true);
  ASSERT_TRUE(ready.pose);
  EXPECT_EQ(ready.pose->generation, 0U);
  EXPECT_EQ(ready.pose->basis_revision, 1U);
  EXPECT_EQ(ready.pose->correction_stamp_nsec, 90);
  const auto cloud = live.observeCloud(source(100), 100, 100, "body");
  ASSERT_TRUE(cloud);
  const auto invalid = live.observeOdometry(source(101), 101, false);
  EXPECT_FALSE(invalid.pose);
  const auto status = json(invalid.status_json);
  EXPECT_EQ(status["reason"], "invalid_twist_or_body_frame");
  EXPECT_DOUBLE_EQ(status["stamp"].asDouble(), 0.0);
  EXPECT_FALSE(status["projection"]["available"].asBool());
  EXPECT_FALSE(status["projection"].isMember("map_from_source"));
  EXPECT_EQ(status["projection"]["generation"].asUInt64(), 1U);
  EXPECT_FALSE(live.isCurrent(*cloud));
  backend.relocalization_state = RelocalizationState::RECENTLY_LOST;
  backend.pose_source = PoseSource::ODOM_PREDICTED;
  EXPECT_FALSE(live.acceptBackend(source(102), 102e-9, backend).live_authorized);
  EXPECT_FALSE(live.observeCloud(source(103), 103, 103, "body"));
  EXPECT_FALSE(live.observeOdometry(source(103), 103, true).pose);
  backend = trusted();
  ASSERT_TRUE(live.acceptBackend(source(104), 104e-9, backend).live_authorized);
  const auto recovered = live.observeOdometry(source(105), 105, true);
  ASSERT_TRUE(recovered.pose);
  EXPECT_EQ(recovered.pose->generation, 1U);
  EXPECT_EQ(recovered.pose->basis_revision, 2U);
  EXPECT_EQ(json(recovered.status_json)["projection"]["session_id"],
            status["projection"]["session_id"]);
}

TEST(RealtimeOutput, ModesRetainTheirExistingAuthorityAndDivergenceRules) {
  for (auto mode : {CoreRunMode::MAPPING, CoreRunMode::LOCALIZATION,
                    CoreRunMode::MAP_EXTENSION}) {
    SCOPED_TRACE(coreRunModeName(mode));
    RealtimeOutput live(mode, "map", 8);
    BackendOutput backend;
    backend.success = true;
    auto committed = live.acceptBackend(source(90), 90e-9, backend);
    EXPECT_EQ(committed.live_authorized, mode != CoreRunMode::LOCALIZATION);
    backend = trusted(10.0);
    backend.success = false;
    backend.accepted_keyframe = true;
    ASSERT_TRUE(live.acceptBackend(source(95), 95e-9, backend).live_authorized);
    const auto first = live.observeOdometry(source(100), 100, true);
    ASSERT_TRUE(first.pose);
    backend = BackendOutput{};
    EXPECT_FALSE(live.acceptBackend(source(101), 101e-9, backend).live_authorized);
    const auto retained = live.observeOdometry(source(102), 102, true);
    ASSERT_TRUE(retained.pose);
    EXPECT_EQ(retained.pose->basis_revision, first.pose->basis_revision);
    EXPECT_DOUBLE_EQ(retained.pose->pose.translation().x(), 10.0);
    EXPECT_DOUBLE_EQ(json(retained.status_json)["observation_stamp"].asDouble(), 101e-9);
    backend.mapping_block_reason = MappingBlockReason::OdometryDiverged;
    live.acceptBackend(source(103), 103e-9, backend);
    backend = trusted();
    live.acceptBackend(source(104), 104e-9, backend);
    const auto diverged = live.observeOdometry(source(105), 105, false);
    EXPECT_FALSE(diverged.pose);
    EXPECT_EQ(json(diverged.status_json)["reason"], "odometry_diverged");
    EXPECT_FALSE(live.observeCloud(source(105), 105, 105, "body"));
  }
}

TEST(RealtimeOutput, ExtensionPredictionDoesNotAdvertiseAnotherGlobalLock) {
  RealtimeOutput live(CoreRunMode::MAP_EXTENSION, "map", 8);
  BackendOutput backend;
  live.acceptBackend(source(1000000000), 1.0, backend);
  EXPECT_EQ(json(live.observeOdometry(source(1000000001), 1000000001, true).status_json)["state"],
            "initializing");
  backend = trusted();
  backend.relocalization_locked = true;
  ASSERT_TRUE(live.acceptBackend(source(2000000000), 2.0, backend).plan.publish_legacy_lock);
  int64_t stamp = 3000000000LL;
  for (const auto state : {RelocalizationState::RECENTLY_LOST, RelocalizationState::LOST}) {
    backend.relocalization_locked = false;
    backend.accepted_keyframe = true;
    backend.relocalization_state = state;
    backend.pose_source = PoseSource::ODOM_PREDICTED;
    const auto publication = live.acceptBackend(source(stamp), stamp*1e-9, backend);
    EXPECT_TRUE(publication.live_authorized);
    EXPECT_FALSE(publication.plan.publish_legacy_lock);
    EXPECT_FALSE(publication.plan.publish_authoritative_pose);
    const auto emission = live.observeOdometry(source(stamp+1), stamp+1, true);
    ASSERT_TRUE(emission.pose);
    EXPECT_EQ(json(emission.status_json)["state"],
              state == RelocalizationState::LOST ? "lost" : "degraded");
    EXPECT_EQ(json(emission.status_json)["reason"], "map_extension_odometry_only");
    stamp += 1000000000LL;
  }
  backend = trusted();
  const auto recovered = live.acceptBackend(source(stamp), stamp*1e-9, backend);
  EXPECT_FALSE(recovered.plan.publish_legacy_lock);
  EXPECT_EQ(json(live.observeOdometry(source(stamp+1), stamp+1, true).status_json)["state"],
            "tracking");
}

TEST(RealtimeOutput, DroppedDuplicatesAndExpiredReservationsDoNotRevokeValidInput) {
  RealtimeOutput live(CoreRunMode::LOCALIZATION, "map", 1);
  auto backend = trusted();
  live.acceptBackend(source(90), 90e-9, backend);
  ASSERT_TRUE(live.observeCloud(source(100), 100, 100, "body"));
  const auto retained = live.observeCloud(source(101), 101, 101, "body");
  ASSERT_TRUE(retained);
  const auto expired = live.observeOdometry(source(100), 102, true);
  EXPECT_FALSE(expired.pose);
  EXPECT_EQ(expired.reason, "projection_expired");
  EXPECT_FALSE(json(expired.status_json).isMember("projection"));
  EXPECT_TRUE(live.isCurrent(*retained));
  ASSERT_TRUE(live.observeOdometry(source(101), 102, true).pose);
  EXPECT_TRUE(live.observeOdometry(source(101), 102, true).status_json.empty());
  backend = trusted();
  live.acceptBackend(source(110), 110e-9, backend);
  EXPECT_TRUE(live.observeOdometry(source(105), 120, true).status_json.empty());
  EXPECT_TRUE(live.observeOdometry(source(120), 120, true).pose);
  const auto regressed = live.observeOdometry(source(119), 121, true);
  EXPECT_FALSE(regressed.pose);
  EXPECT_FALSE(json(regressed.status_json)["projection"]["available"].asBool());
  EXPECT_EQ(regressed.reason, "input_time_regressed");
  EXPECT_FALSE(live.isCurrent(*retained));
}

TEST(RealtimeOutput, BackendCommitAndLiveSnapshotNeverMixObservations) {
  RealtimeOutput live(CoreRunMode::LOCALIZATION, "map", 2);
  auto commit = [&](int64_t stamp) {
    auto backend = trusted(static_cast<double>(stamp));
    backend.tracking_performance.available = true;
    backend.tracking_performance.tracking_total_ms = stamp;
    live.acceptBackend(source(stamp), stamp * 1e-9, backend);
  };
  commit(1);
  std::atomic<bool> start{false};
  auto backend = std::async(std::launch::async, [&] {
    while (!start.load()) std::this_thread::yield();
    for (int64_t stamp = 2; stamp <= 4000; ++stamp) commit(stamp);
  });
  start = true;
  for (int64_t stamp = 10000; stamp < 14000; ++stamp) {
    const auto emission = live.observeOdometry(source(stamp), stamp, true);
    ASSERT_TRUE(emission.pose);
    const auto status = json(emission.status_json);
    EXPECT_DOUBLE_EQ(status["observation_stamp"].asDouble(),
                     status["correction_stamp"].asDouble());
    EXPECT_DOUBLE_EQ(status["tracking_performance"]["total_ms"].asDouble(),
                     emission.pose->pose.translation().x());
  }
  backend.get();
}

}  // namespace
}  // namespace n3mapping::core
