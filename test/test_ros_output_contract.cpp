#include <gtest/gtest.h>
#include <json/json.h>
#include <limits>
#include <sstream>

#ifdef N3MAPPING_ROS2_OUTPUT_TEST
#include <nav_msgs/msg/odometry.hpp>
using Odometry = nav_msgs::msg::Odometry;
#else
#include <nav_msgs/Odometry.h>
using Odometry = nav_msgs::Odometry;
#endif

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/ros_output_contract.h"

namespace n3mapping {
namespace {

Odometry sample() {
    Odometry odom;
    odom.child_frame_id = "base_link";
    odom.pose.pose.orientation.w = std::sqrt(0.5);
    odom.pose.pose.orientation.z = std::sqrt(0.5);
    odom.twist.twist.linear.x = 1.0;
    odom.twist.twist.angular.z = 0.3;
    odom.twist.covariance[0] = 4.0;
    odom.twist.covariance[7] = 1.0;
    odom.twist.covariance[5] = 0.2;
    odom.twist.covariance[30] = 0.2;
    return odom;
}

TEST(RosOutputContract, StandardBodyVelocityDoesNotFollowGlobalPoseCorrections) {
    auto input = sample();
    Odometry output;
    output.child_frame_id = input.child_frame_id;
    output.pose.pose.position.x = 100.0;
    output.pose.pose.orientation.z = 1.0;
    ASSERT_TRUE(forwardOdometryTwist(input, InputLinearVelocityFrame::CHILD, &output));
    EXPECT_DOUBLE_EQ(output.twist.twist.linear.x, 1.0);
    EXPECT_DOUBLE_EQ(output.twist.twist.angular.z, 0.3);
    EXPECT_EQ(output.twist.covariance, input.twist.covariance);
}

TEST(RosOutputContract, ParentLinearVelocityAndCovarianceBecomeBodyRelative) {
    auto input = sample();
    Odometry output;
    output.child_frame_id = input.child_frame_id;
    ASSERT_TRUE(forwardOdometryTwist(input, InputLinearVelocityFrame::PARENT, &output));
    EXPECT_NEAR(output.twist.twist.linear.x, 0.0, 1e-12);
    EXPECT_NEAR(output.twist.twist.linear.y, -1.0, 1e-12);
    EXPECT_DOUBLE_EQ(output.twist.twist.angular.z, 0.3);
    EXPECT_NEAR(output.twist.covariance[0], 1.0, 1e-12);
    EXPECT_NEAR(output.twist.covariance[7], 4.0, 1e-12);
    EXPECT_NEAR(output.twist.covariance[11], -0.2, 1e-12);
    EXPECT_NEAR(output.twist.covariance[31], -0.2, 1e-12);
}

TEST(RosOutputContract, InvalidSamplesNeverProduceSuccessfulZeroVelocity) {
    auto input = sample();
    Odometry output;
    output.child_frame_id = "another_body";
    EXPECT_FALSE(forwardOdometryTwist(input, InputLinearVelocityFrame::CHILD, &output));
    output.child_frame_id = input.child_frame_id;
    input.twist.twist.linear.x = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(forwardOdometryTwist(input, InputLinearVelocityFrame::CHILD, &output));
    input = sample();
    input.twist.covariance[3] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(forwardOdometryTwist(input, InputLinearVelocityFrame::CHILD, &output));
    input = sample();
    input.pose.pose.orientation.w = input.pose.pose.orientation.z = 0.0;
    EXPECT_FALSE(forwardOdometryTwist(input, InputLinearVelocityFrame::PARENT, &output));
    EXPECT_THROW(parseInputLinearVelocityFrame("world_guess"), std::invalid_argument);
}

Json::Value status(const core::BackendOutput& output, double stamp = 2.0) {
    std::istringstream stream(localizationStatusJson(
        CoreRunMode::LOCALIZATION, output, stamp, "map\"frame"));
    Json::Value result;
    stream >> result;
    return result;
}

TEST(RosOutputContract, OnlyAuthoritativeFiniteLocalizationIsTracking) {
    core::BackendOutput output;
    EXPECT_EQ(status(output)["state"], "localizing");
    output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
    output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    auto json = status(output);
    EXPECT_EQ(json["state"], "tracking");
    EXPECT_TRUE(json["stamp"].isNumeric());
    EXPECT_EQ(json["frame_id"], "map\"frame");
    output.relocalization_state = RelocalizationState::RECENTLY_LOST;
    output.pose_source = PoseSource::ODOM_PREDICTED;
    EXPECT_EQ(status(output)["state"], "degraded");
    output.relocalization_state = RelocalizationState::LOST;
    EXPECT_EQ(status(output)["state"], "lost");
    output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
    output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    output.T_world_lidar.translation().x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(status(output)["state"], "error");
    EXPECT_EQ(status(output, 0.0)["state"], "error");
}

TEST(RosOutputContract, MappingWaitAndErrorsAreNotLocalizationProgress) {
    core::BackendOutput output;
    const auto mapping = [&]() {
        std::istringstream stream(localizationStatusJson(CoreRunMode::MAPPING, output, 2.0, "map"));
        Json::Value json;
        stream >> json;
        return json;
    };
    output.mapping_block_reason = core::MappingBlockReason::StaticStartGuard;
    EXPECT_EQ(mapping()["mode"], "mapping");
    EXPECT_EQ(mapping()["state"], "initializing");
    EXPECT_EQ(mapping()["reason"], "waiting_for_static_start_guard");
    output.mapping_block_reason = core::MappingBlockReason::OdometryDiverged;
    EXPECT_EQ(mapping()["state"], "error");
    EXPECT_EQ(mapping()["reason"], "odometry_diverged");
    output.mapping_block_reason = core::MappingBlockReason::None;
    output.success = true;
    EXPECT_EQ(mapping()["state"], "tracking");
}

TEST(RosOutputContract, RealtimeStampNeverRefreshesObservationOrCorrectionTime) {
    core::BackendOutput output;
    output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
    output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    RealtimeLocalizationStatus realtime;
    realtime.observation_stamp = 1.2;
    realtime.correction_stamp = 1.0;
    realtime.estimate_available = true;
    Json::Value json;
    std::istringstream stream(localizationStatusJson(
        CoreRunMode::LOCALIZATION, output, 62.0, "map", &realtime));
    stream >> json;
    EXPECT_EQ(json["state"], "tracking");
    EXPECT_DOUBLE_EQ(json["stamp"].asDouble(), 62.0);
    EXPECT_DOUBLE_EQ(json["observation_stamp"].asDouble(), 1.2);
    EXPECT_DOUBLE_EQ(json["correction_stamp"].asDouble(), 1.0);
    realtime.estimate_available = false;
    realtime.input_reason = "invalid_input_pose";
    std::istringstream invalid(localizationStatusJson(
        CoreRunMode::LOCALIZATION, output, 62.1, "map", &realtime));
    invalid >> json;
    EXPECT_EQ(json["state"], "error");
    EXPECT_EQ(json["reason"], "invalid_input_pose");
}

TEST(RosOutputContract, RelocalizationCostsAreOptionalFiniteAndDoNotChangeState) {
    core::BackendOutput output;
    ASSERT_FALSE(status(output).isMember("relocalization_performance"));
    auto &cost = output.relocalization_performance;
    cost.available = true;
    cost.total_ms = 12.5;
    cost.target_builds = 2;
    cost.effective_cache_max_bytes = 128u * 1024 * 1024;
    cost.align_ms = std::numeric_limits<double>::infinity();
    const auto json = status(output);
    EXPECT_EQ(json["state"], "localizing");
    const auto &performance = json["relocalization_performance"];
    EXPECT_DOUBLE_EQ(performance["total_ms"].asDouble(), 12.5);
    EXPECT_EQ(performance["target_builds"].asUInt64(), 2u);
    EXPECT_EQ(performance["effective_cache_max_bytes"].asUInt64(), 128u * 1024 * 1024);
    EXPECT_FALSE(performance.isMember("align_ms"));
    cost.available = false;
    EXPECT_FALSE(status(output).isMember("relocalization_performance"));
}

TEST(RosOutputContract, TrackingCostsDoNotRequireRelocalizationOrChangePoseAuthority) {
    core::BackendOutput output;
    ASSERT_FALSE(status(output).isMember("tracking_performance"));
    output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
    output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    auto &cost = output.tracking_performance;
    cost.available = true;
    cost.tracking_total_ms = 400.0;
    cost.lock_wait_ms = 2.0;
    cost.source_prepare_ms = 37.0;
    cost.registration_ms = 360.0;
    cost.localization_target_cache_hit = true;
    const auto json = status(output);
    EXPECT_EQ(json["state"], "tracking");
    EXPECT_FALSE(json.isMember("relocalization_performance"));
    const auto &performance = json["tracking_performance"];
    EXPECT_DOUBLE_EQ(performance["total_ms"].asDouble(), 400.0);
    EXPECT_DOUBLE_EQ(performance["lock_wait_ms"].asDouble(), 2.0);
    EXPECT_DOUBLE_EQ(performance["source_prepare_ms"].asDouble(), 37.0);
    EXPECT_TRUE(performance["target_cache_hit"].asBool());
    EXPECT_FALSE(performance.isMember("retry_registration_ms"));
    EXPECT_FALSE(performance["strict_loaded_map"].asBool());
    cost.strict_loaded_map = true;
    cost.loaded_map_target_cache_miss = true;
    EXPECT_FALSE(status(output)["tracking_performance"]["target_cache_hit"].asBool());
    EXPECT_TRUE(status(output)["tracking_performance"]["target_cache_miss"].asBool());
}

}  // namespace
}  // namespace n3mapping
