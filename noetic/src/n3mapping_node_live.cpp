#include "n3mapping_node.h"
#include "n3mapping_noetic/conversions.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

void N3MappingNoeticNode::initializeLiveSubscriptions()
{
    const int sync_queue_size = std::max(1, config_.sync_queue_size);
    auto live_options = ros::SubscribeOptions::create<nav_msgs::Odometry>(
        config_.odom_topic, 1,
        boost::bind(&N3MappingNoeticNode::liveOdomCallback, this, _1),
        ros::VoidPtr(), &live_odom_queue_);
    live_odom_sub_ = nh_.subscribe(live_options);
    ros::NodeHandle live_nh(nh_);
    // Full cloud conversion/publication must not occupy the raw odom queue.
    live_nh.setCallbackQueue(&live_cloud_queue_);
    live_cloud_sub_.subscribe(live_nh, config_.cloud_topic, sync_queue_size);
    live_paired_odom_sub_.subscribe(live_nh, config_.odom_topic, sync_queue_size);
    if (product_profile_v1_) {
        live_exact_sync_ = std::make_unique<ExactSynchronizer>(
            ExactSyncPolicy(sync_queue_size), live_cloud_sub_, live_paired_odom_sub_);
        live_exact_sync_->registerCallback(boost::bind(
            &N3MappingNoeticNode::liveCloudCallback, this, _1, _2));
    } else {
        ApproxSyncPolicy policy(sync_queue_size);
        policy.setMaxIntervalDuration(ros::Duration(config_.sync_time_tolerance));
        live_approx_sync_ = std::make_unique<ApproxSynchronizer>(
            static_cast<const ApproxSyncPolicy&>(policy), live_cloud_sub_, live_paired_odom_sub_);
        live_approx_sync_->registerCallback(boost::bind(
            &N3MappingNoeticNode::liveCloudCallback, this, _1, _2));
    }
    live_odom_spinner_ = std::make_unique<ros::AsyncSpinner>(1, &live_odom_queue_);
    live_cloud_spinner_ = std::make_unique<ros::AsyncSpinner>(1, &live_cloud_queue_);
}

void N3MappingNoeticNode::startLiveOutput()
{
    live_odom_spinner_->start();
    live_cloud_spinner_->start();
}

void N3MappingNoeticNode::stopLiveOutput()
{
    if (live_odom_spinner_) live_odom_spinner_->stop();
    if (live_cloud_spinner_) live_cloud_spinner_->stop();
    live_odom_sub_.shutdown();
    live_cloud_sub_.unsubscribe();
    live_paired_odom_sub_.unsubscribe();
}

void N3MappingNoeticNode::liveOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    const auto received = std::chrono::steady_clock::now();
    const double receive_gap_s = last_live_odom_receive_
        ? std::chrono::duration<double>(received - *last_live_odom_receive_).count()
        : 0.0;
    last_live_odom_receive_ = received;
    nav_msgs::Odometry odometry;
    odometry.header = odom_msg->header;
    odometry.header.frame_id = config_.world_frame;
    odometry.child_frame_id = config_.body_frame;
    const bool twist_publishable = forwardOdometryTwist(
        *odom_msg, input_linear_velocity_frame_, &odometry);
    const auto source = toCoreOdometrySource(*odom_msg);
    const auto emission = realtime_output_->observeOdometry(
        source, static_cast<int64_t>(ros::Time::now().toNSec()), twist_publishable);
    const auto& diagnostic = emission.diagnostic;
    const auto& reason = emission.reason;
    if (emission.pose) {
        odometry.pose.pose = makePose(emission.pose->pose);
        odom_pub_.publish(odometry);
        publishLiveTransform(emission.pose->pose, odom_msg->header);
    }
    if (!emission.status_json.empty()) {
        std_msgs::String status;
        status.data = emission.status_json;
        localization_status_pub_.publish(status);
    }
    if (last_logged_live_reason_ != reason ||
        receive_gap_s > 0.5) {
        LOG(INFO) << std::fixed << std::setprecision(9)
                  << "realtime_odometry reason=" << reason
                  << " input_stamp_s=" << diagnostic.input_stamp_nsec * 1e-9
                  << " correction_stamp_s=" << diagnostic.correction_stamp_nsec * 1e-9
                  << " source_age_s=" << (diagnostic.now_nsec - diagnostic.input_stamp_nsec) * 1e-9
                  << " correction_age_s=" << (diagnostic.now_nsec - diagnostic.correction_stamp_nsec) * 1e-9
                  << " input_receive_gap_s=" << receive_gap_s
                  << " source_frame=" << odom_msg->header.frame_id
                  << " child_frame=" << odom_msg->child_frame_id;
        last_logged_live_reason_ = reason;
    }
}

void N3MappingNoeticNode::publishLiveTransform(const Eigen::Isometry3d& pose,
                          const std_msgs::Header& header)
{
    const ros::Time& stamp = header.stamp;
    if (stamp.isZero()) {
        ROS_WARN_THROTTLE(2.0, "Skip map->%s TF with zero live source stamp",
                          config_.body_frame.c_str());
        return;
    }
    if (!last_tf_stamp_.isZero() && stamp <= last_tf_stamp_) {
        ROS_WARN_THROTTLE(2.0,
                          "Skip non-increasing live map->%s TF stamp: current=%.6f last=%.6f",
                          config_.body_frame.c_str(), stamp.toSec(),
                          last_tf_stamp_.toSec());
        return;
    }
    geometry_msgs::TransformStamped tf;
    tf.header.frame_id = config_.world_frame;
    tf.header.stamp = stamp;
    tf.child_frame_id = config_.body_frame;
    const auto transform = makePose(pose);
    tf.transform.translation.x = transform.position.x;
    tf.transform.translation.y = transform.position.y;
    tf.transform.translation.z = transform.position.z;
    tf.transform.rotation = transform.orientation;
    tf_broadcaster_.sendTransform(tf);
    last_tf_stamp_ = stamp;
}

void N3MappingNoeticNode::liveCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud,
                       const nav_msgs::OdometryConstPtr& odom)
{
    if (cloud->header.frame_id != odom->child_frame_id ||
        odom->child_frame_id != config_.body_frame) return;
    cloud_body_pub_.publish(cloud);
    const auto projected = realtime_output_->observeCloud(
        toCoreOdometrySource(*odom), ros::Time::now().toNSec(),
        cloud->header.stamp.toNSec(), cloud->header.frame_id);
    if (!projected) return;
    std_msgs::String association;
    association.data = projected->status_json;
    localization_status_pub_.publish(association);
    if (cloud_world_pub_.getNumSubscribers() == 0) return;
    pcl::PointCloud<pcl::PointXYZI> body, world;
    pcl::fromROSMsg(*cloud, body);
    pcl::transformPointCloud(body, world, projected->projection.pose.matrix());
    sensor_msgs::PointCloud2 message;
    pcl::toROSMsg(world, message);
    message.header = cloud->header;
    message.header.frame_id = config_.world_frame;
    if (!realtime_output_->isCurrent(*projected)) return;
    cloud_world_pub_.publish(message);
}

void N3MappingNoeticNode::publishLegacyRelocalizationLock(std::uint64_t lock_epoch)
{
    std_msgs::UInt32 legacy_msg;
    legacy_msg.data = static_cast<std::uint32_t>(lock_epoch);
    relocalization_lock_pub_.publish(legacy_msg);
}

void N3MappingNoeticNode::publishAuthoritativeRelocalizationPose(
    const std_msgs::Header& header, const Eigen::Isometry3d& pose,
    std::uint64_t lock_epoch)
{
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = header;
    pose_msg.header.frame_id = config_.world_frame;
    pose_msg.pose = makePose(pose);
    relocalization_pose_pub_.publish(pose_msg);

    const Eigen::Quaterniond q(pose.rotation());
    const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                  1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    ROS_INFO("Relocalization lock #%llu pose=(%.3f, %.3f, %.3f) yaw=%.3f",
             static_cast<unsigned long long>(lock_epoch),
             pose.translation().x(),
             pose.translation().y(),
             pose.translation().z(),
             yaw);
}

void N3MappingNoeticNode::publishRelocalizationStatus(const std_msgs::Header& header,
                                 const core::BackendOutput& output,
                                 std::uint64_t lock_epoch)
{
    RelocalizationStatus status;
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
    relocalization_status_pub_.publish(status);
}

geometry_msgs::Pose N3MappingNoeticNode::makePose(const Eigen::Isometry3d& pose) const
{
    geometry_msgs::Pose out;
    out.position = toPoint(pose);
    const Eigen::Quaterniond q(pose.rotation());
    out.orientation.w = q.w();
    out.orientation.x = q.x();
    out.orientation.y = q.y();
    out.orientation.z = q.z();
    return out;
}

}  // namespace n3mapping
