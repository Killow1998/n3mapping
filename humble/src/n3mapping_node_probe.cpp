#include "n3mapping_node.h"
#include <pcl_conversions/pcl_conversions.h>
#include <stdexcept>

namespace n3mapping {

void N3MappingNode::initializeAuthorityProbe()
{
    if (*authority_probe_scenario_ == "localization_authority_sequence" ||
        *authority_probe_scenario_ == "invalid_pose_suppression") {
        run_mode_ = CoreRunMode::LOCALIZATION;
    } else if (
      *authority_probe_scenario_ ==
      "map_extension_legacy_compatibility") {
        run_mode_ = CoreRunMode::MAP_EXTENSION;
    } else {
        throw std::runtime_error(
          "unknown Product V1 authority probe scenario: " +
          *authority_probe_scenario_);
    }

    realtime_output_ = std::make_unique<core::RealtimeOutput>(
        run_mode_, config_.world_frame, std::max(1, config_.sync_queue_size));
    tf_broadcaster_ =
      std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      config_.output_odom_topic, 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
      config_.output_path_topic, 10);
    cloud_body_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        config_.output_cloud_body_topic, 10);
    cloud_world_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        config_.output_cloud_world_topic, 10);
    relocalization_lock_pub_ =
      this->create_publisher<std_msgs::msg::UInt32>(
        "/n3mapping/relocalization_lock", 10);

    rclcpp::QoS marker_qos(10);
    marker_qos.transient_local();
    loop_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/n3mapping/loop_closure_markers", marker_qos);

    rclcpp::QoS status_qos(1);
    status_qos.reliable();
    status_qos.transient_local();
    relocalization_status_pub_ =
      this->create_publisher<msg::RelocalizationStatus>(
        "/n3mapping/relocalization_status", status_qos);

    rclcpp::QoS pose_qos(1);
    pose_qos.reliable();
    pose_qos.durability_volatile();
    relocalization_pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/n3mapping/relocalization_pose", pose_qos);

    const auto status_actual =
      relocalization_status_pub_->get_actual_qos().get_rmw_qos_profile();
    const auto pose_actual =
      relocalization_pose_pub_->get_actual_qos().get_rmw_qos_profile();
    const auto lock_actual =
      relocalization_lock_pub_->get_actual_qos().get_rmw_qos_profile();
    const auto odom_actual =
      odom_pub_->get_actual_qos().get_rmw_qos_profile();
    const auto cloud_actual =
      cloud_world_pub_->get_actual_qos().get_rmw_qos_profile();
    std::cout
      << "N3MAPPING_AUTHORITY_PROBE_QOS "
      << "{\"relocalization_status\":" << status_actual.depth
      << ",\"authoritative_pose\":" << pose_actual.depth
      << ",\"legacy_lock\":" << lock_actual.depth
      << ",\"global_odometry\":" << odom_actual.depth
      << ",\"global_world_cloud\":" << cloud_actual.depth
      << "}" << std::endl;

    initializeLiveSubscriptions();
    authority_probe_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(250),
      std::bind(&N3MappingNode::authorityProbeTick, this));
    RCLCPP_INFO(
      this->get_logger(),
      "Product V1 active-runtime authority probe ready: %s",
      authority_probe_scenario_->c_str());
}

core::BackendOutput N3MappingNode::makeAuthorityProbeOutput(
  const std::string& label, int event_index)
{
    core::BackendOutput output;
    output.relocalization_decision = label;
    output.relocalization_seed_keyframe_id = 1000 + event_index;
    output.relocalization_support_keyframe_id = 2000 + event_index;
    output.T_world_lidar.translation() =
      Eigen::Vector3d(10.0 * event_index, event_index, 0.1 * event_index);
    output.T_world_lidar.linear() =
      Eigen::AngleAxisd(
        0.1 * event_index, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    if (label == "provisional") {
        output.relocalization_state =
          RelocalizationState::PROVISIONAL;
    } else if (label == "first_full_lock" ||
               label == "steady_full_lock" ||
               label == "recovered_full_lock" ||
               label == "invalid_full_pose" ||
               label == "map_extension_lock") {
        output.success = true;
        output.relocalization_state =
          RelocalizationState::FULL_6DOF_LOCKED;
        output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
        output.relocalization_locked =
          label != "steady_full_lock";
    } else if (label == "recently_lost") {
        output.success = true;
        output.relocalization_state =
          RelocalizationState::RECENTLY_LOST;
        output.pose_source = PoseSource::ODOM_PREDICTED;
    } else if (label == "lost") {
        output.relocalization_state = RelocalizationState::LOST;
    }

    if (label == "invalid_full_pose") {
        output.T_world_lidar.linear()(0, 0) = 2.0;
    }
    return output;
}

void N3MappingNode::authorityProbeTick()
{
    ++authority_probe_tick_;
    if (authority_probe_tick_ < 8) {
        return;
    }

    std::vector<std::string> labels;
    if (*authority_probe_scenario_ ==
        "localization_authority_sequence") {
        labels = {
          "searching",
          "provisional",
          "first_full_lock",
          "steady_full_lock",
          "recently_lost",
          "lost",
          "recovered_full_lock",
        };
    } else if (
      *authority_probe_scenario_ == "invalid_pose_suppression") {
        labels = {"invalid_full_pose"};
    } else {
        labels = {"map_extension_lock"};
    }

    const int elapsed_ticks = authority_probe_tick_ - 8;
    if (elapsed_ticks % 2 != 0) {
        return;
    }
    const int event_index = elapsed_ticks / 2;
    if (event_index >= static_cast<int>(labels.size())) {
        if (event_index >= static_cast<int>(labels.size()) + 2) {
            std::cout << "N3MAPPING_AUTHORITY_PROBE_DONE "
                      << *authority_probe_scenario_ << std::endl;
            g_shutdown_requested = true;
        }
        return;
    }

    const std::string& label = labels[static_cast<std::size_t>(event_index)];
    std_msgs::msg::Header header;
    header.frame_id = config_.body_frame;
    header.stamp.sec = 100 + event_index;
    header.stamp.nanosec = 0;
    auto output = makeAuthorityProbeOutput(label, event_index + 1);
    std::cout << "N3MAPPING_AUTHORITY_PROBE_EVENT "
              << *authority_probe_scenario_ << " " << label << " "
              << header.stamp.sec << std::endl;
    auto probe_odom = std::make_shared<nav_msgs::msg::Odometry>();
    probe_odom->header = header;
    probe_odom->header.frame_id = "continuous";
    probe_odom->child_frame_id = config_.body_frame;
    probe_odom->pose.pose.orientation.w = 1.0;
    // The same production callbacks use a consistent synthetic clock. The
    // source epoch stays stable for the verifier's existing event buckets.
    authority_probe_now_nsec_ = toCoreTimeStamp(header.stamp).nsec;
    const auto publication = publishRelocalizationOutput(header, *probe_odom, output);
    liveOdomCallback(probe_odom);
    if (publication.publish_global_pose) {
        publishPath(header, &output.T_world_lidar);
    }
    PointCloud body;
    body.push_back(pcl::PointXYZI{1.0F, 0.0F, 0.0F, 1.0F});
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(body, *cloud);
    cloud->header = header;
    liveCloudCallback(cloud, probe_odom);
}

}  // namespace n3mapping
