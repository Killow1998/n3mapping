#include "n3mapping_node.h"
#include "n3mapping/pcl_compat.h"
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

void N3MappingNode::startLiveOutput()
{
    live_odom_worker_->start();
    live_cloud_worker_->start();
}

void N3MappingNode::initializeLiveSubscriptions()
{
    live_odom_worker_ = std::make_unique<CallbackWorker>(*this);
    live_cloud_worker_ = std::make_unique<CallbackWorker>(*this);
    live_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        config_.odom_topic, rclcpp::QoS(1),
        std::bind(&N3MappingNode::liveOdomCallback, this, std::placeholders::_1),
        live_odom_worker_->options());
    const int queue_size = std::max(1, config_.sync_queue_size);
    auto qos = rmw_qos_profile_default;
    qos.depth = static_cast<std::size_t>(queue_size);
    const auto options = live_cloud_worker_->options();
    live_cloud_sub_.subscribe(this, config_.cloud_topic, qos, options);
    live_paired_odom_sub_.subscribe(this, config_.odom_topic, qos, options);
    if (product_profile_v1_) {
        ExactSyncPolicy policy(static_cast<uint32_t>(queue_size));
        live_exact_sync_ = std::make_unique<message_filters::Synchronizer<ExactSyncPolicy>>(
            static_cast<const ExactSyncPolicy&>(policy), live_cloud_sub_, live_paired_odom_sub_);
        live_exact_sync_->registerCallback(std::bind(&N3MappingNode::liveCloudCallback,
            this, std::placeholders::_1, std::placeholders::_2));
    } else {
        ApproxSyncPolicy policy(static_cast<uint32_t>(queue_size));
        policy.setMaxIntervalDuration(rclcpp::Duration::from_seconds(config_.sync_time_tolerance));
        live_approx_sync_ = std::make_unique<message_filters::Synchronizer<ApproxSyncPolicy>>(
            static_cast<const ApproxSyncPolicy&>(policy), live_cloud_sub_, live_paired_odom_sub_);
        live_approx_sync_->registerCallback(std::bind(&N3MappingNode::liveCloudCallback,
            this, std::placeholders::_1, std::placeholders::_2));
    }
}

void N3MappingNode::stopLiveOutput()
{
    if (live_odom_worker_) live_odom_worker_->stop();
    if (live_cloud_worker_) live_cloud_worker_->stop();
}

int64_t N3MappingNode::liveOutputNow()
{
    const auto probe_time = authority_probe_now_nsec_.load();
    return probe_time > 0 ? probe_time : this->get_clock()->now().nanoseconds();
}

void N3MappingNode::liveOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
    if (shutdown_called_) return;
    const bool trace = performance_debug_logger_ && performance_debug_logger_->ready();
    using Clock = std::chrono::steady_clock;
    const auto started = trace ? Clock::now() : Clock::time_point{};
    const double processing_time = trace ? std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count() : 0.0;
    nav_msgs::msg::Odometry message;
    message.header = odom->header;
    message.header.frame_id = config_.world_frame;
    message.child_frame_id = config_.body_frame;
    const bool publishable = forwardOdometryTwist(*odom, input_linear_velocity_frame_, &message);
    const auto emission = realtime_output_->observeOdometry(
        toCoreOdometrySource(*odom), liveOutputNow(), publishable);
    const auto projected = trace ? Clock::now() : Clock::time_point{};
    if (emission.pose) {
        message.pose.pose = makePose(emission.pose->pose);
        odom_pub_->publish(message);
        publishLiveTransform(emission.pose->pose, odom->header);
    }
    const auto pose_published = trace ? Clock::now() : Clock::time_point{};
    if (!emission.status_json.empty()) {
        std_msgs::msg::String status;
        status.data = emission.status_json;
        localization_status_pub_->publish(status);
    }
    if (trace) {
        const auto& d = emission.diagnostic;
        RuntimePerformanceLiveEvent event{
            d.input_stamp_nsec, d.correction_stamp_nsec, d.now_nsec,
            d.generation, emission.reason, emission.pose.has_value()};
        event.processing_time = processing_time;
        event.projection_ms = std::chrono::duration<double, std::milli>(projected - started).count();
        event.pose_publish_ms = std::chrono::duration<double, std::milli>(pose_published - projected).count();
        event.status_publish_ms = std::chrono::duration<double, std::milli>(Clock::now() - pose_published).count();
        performance_debug_logger_->append(event);
    }
}

void N3MappingNode::liveCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                       const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
{
    if (shutdown_called_ || cloud->header.frame_id != odom->child_frame_id ||
        odom->child_frame_id != config_.body_frame) return;
    cloud_body_pub_->publish(*cloud);
    const auto emission = realtime_output_->observeCloud(
        toCoreOdometrySource(*odom), liveOutputNow(),
        toCoreTimeStamp(cloud->header.stamp).nsec, cloud->header.frame_id);
    if (!emission) return;
    std_msgs::msg::String association;
    association.data = emission->status_json;
    localization_status_pub_->publish(association);
    if (cloud_world_pub_->get_subscription_count() == 0) return;
    const auto message = transformLiveCloud(*cloud, emission->projection.pose);
    if (shutdown_called_ || !realtime_output_->isCurrent(*emission)) return;
    cloud_world_pub_->publish(message);
}

sensor_msgs::msg::PointCloud2 N3MappingNode::transformLiveCloud(
    const sensor_msgs::msg::PointCloud2& cloud, const Eigen::Isometry3d& pose)
{
    PointCloud body, world;
    pcl::fromROSMsg(cloud, body);
    pcl::transformPointCloud(body, world, pose.matrix());
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(world, message);
    message.header = cloud.header;
    message.header.frame_id = config_.world_frame;
    return message;
}

RelocalizationPublicationPlan N3MappingNode::publishRelocalizationOutput(
  const std_msgs::msg::Header& header, const nav_msgs::msg::Odometry& source,
  core::BackendOutput& output)
{
    const auto committed = realtime_output_->acceptBackend(
        toCoreOdometrySource(source), rclcpp::Time(header.stamp).seconds(), output);
    const auto& publication = committed.plan;

    if (run_mode_ == CoreRunMode::MAP_EXTENSION &&
        output.relocalization_locked && !output.accepted_keyframe) {
        RCLCPP_INFO(
          this->get_logger(),
          "Initial relocalization successful for map extension");
    }
    if (publication.inconsistent_lock_event) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Suppressed inconsistent relocalization lock event: "
          "state=%s pose_source=%s",
          relocalizationStateName(output.relocalization_state),
          poseSourceName(output.pose_source));
    }
    if (publication.invalid_usable_pose_suppressed) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Suppressed non-finite or non-rigid localization pose");
    }
    if (publication.publish_legacy_lock) {
        publishLegacyRelocalizationLock(publication.lock_epoch);
    }
    if (publication.publish_status) {
        publishRelocalizationStatus(
          header, output, publication.lock_epoch);
    }
    if (publication.publish_authoritative_pose) {
        publishAuthoritativeRelocalizationPose(
          header, output.T_world_lidar, publication.lock_epoch);
    }
    return publication;
}

geometry_msgs::msg::Pose N3MappingNode::makePose(const Eigen::Isometry3d& pose) const
{
    geometry_msgs::msg::Pose out;
    out.position.x = pose.translation().x();
    out.position.y = pose.translation().y();
    out.position.z = pose.translation().z();
    const Eigen::Quaterniond q(pose.rotation());
    out.orientation.w = q.w();
    out.orientation.x = q.x();
    out.orientation.y = q.y();
    out.orientation.z = q.z();
    return out;
}

void N3MappingNode::publishLiveTransform(const Eigen::Isometry3d& pose,
                          const std_msgs::msg::Header& source_header)
{
    const rclcpp::Time stamp(source_header.stamp);
    if (stamp.nanoseconds() <= 0 || (last_tf_stamp_ && stamp <= *last_tf_stamp_)) return;
    geometry_msgs::msg::TransformStamped tf;
    tf.header = source_header;
    tf.header.frame_id = config_.world_frame;
    tf.child_frame_id = config_.body_frame;
    tf.transform.translation.x = pose.translation().x();
    tf.transform.translation.y = pose.translation().y();
    tf.transform.translation.z = pose.translation().z();
    tf.transform.rotation = makePose(pose).orientation;
    tf_broadcaster_->sendTransform(tf);
    last_tf_stamp_ = stamp;
}

void N3MappingNode::publishLegacyRelocalizationLock(std::uint64_t lock_epoch)
{
    if (relocalization_lock_pub_) {
        std_msgs::msg::UInt32 msg;
        msg.data = static_cast<std::uint32_t>(lock_epoch);
        relocalization_lock_pub_->publish(msg);
    }
}

void N3MappingNode::publishAuthoritativeRelocalizationPose(
  const std_msgs::msg::Header& header, const Eigen::Isometry3d& pose,
  std::uint64_t lock_epoch)
{
    if (relocalization_pose_pub_) {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = config_.world_frame;
        pose_msg.pose = makePose(pose);
        relocalization_pose_pub_->publish(pose_msg);
    }

    Eigen::Quaterniond q(pose.rotation());
    const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                  1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    RCLCPP_INFO(this->get_logger(),
                "Relocalization lock event #%llu @ %.3f pose=(%.3f, %.3f, %.3f) yaw=%.3f",
                static_cast<unsigned long long>(lock_epoch),
                rclcpp::Time(header.stamp).seconds(),
                pose.translation().x(),
                pose.translation().y(),
                pose.translation().z(),
                yaw);
}

void N3MappingNode::publishRelocalizationStatus(
  const std_msgs::msg::Header& header, const core::BackendOutput& output,
  std::uint64_t lock_epoch)
{
    if (!relocalization_status_pub_) {
        return;
    }

    msg::RelocalizationStatus status;
    status.header = header;
    status.header.frame_id = config_.world_frame;
    status.state = static_cast<std::uint8_t>(output.relocalization_state);
    status.pose_source = static_cast<std::uint8_t>(output.pose_source);
    status.pose.orientation.w = 1.0;
    if (hasUsableGlobalRelocalizationPose(
          output.relocalization_state, output.pose_source)) {
        status.pose = makePose(output.T_world_lidar);
    }
    status.lock_epoch = lock_epoch;
    status.seed_keyframe_id = output.relocalization_seed_keyframe_id;
    status.support_keyframe_id =
      output.relocalization_support_keyframe_id;
    status.decision = output.relocalization_decision;
    relocalization_status_pub_->publish(status);
}

}  // namespace n3mapping
