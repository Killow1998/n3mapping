#include "n3mapping/ros2/mode_handlers.h"

#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>

namespace n3mapping {

MappingModeHandler::MappingModeHandler(const Config& config,
                                       KeyframeManager& keyframe_manager,
                                       LoopDetector& loop_detector,
                                       GraphOptimizer& graph_optimizer,
                                       std::mutex& loop_queue_mutex,
                                       std::vector<int64_t>& loop_detection_queue,
                                       ModePublishCallbacks publish,
                                       std::function<void()> on_keyframe_added,
                                       rclcpp::Logger logger)
  : processor_(config, keyframe_manager, loop_detector, graph_optimizer)
  , loop_queue_mutex_(loop_queue_mutex)
  , loop_detection_queue_(loop_detection_queue)
  , publish_(std::move(publish))
  , on_keyframe_added_(std::move(on_keyframe_added))
  , logger_(std::move(logger))
{
}

void
MappingModeHandler::process(double timestamp, const Eigen::Isometry3d& pose_odom, PointCloud::Ptr cloud, const std_msgs::msg::Header& header)
{
    auto result = processor_.process(timestamp, pose_odom, cloud);
    if (!result.accepted_keyframe) {
        publish_.publish_odometry(pose_odom, header);
        publish_.publish_path(header, &pose_odom);
        publish_.publish_point_clouds(cloud, pose_odom, header);
        return;
    }

    publish_.log_optimization_result("mapping_incremental", timestamp, &result.publish_pose);

    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        loop_detection_queue_.push_back(result.keyframe_id);
    }

    Eigen::Isometry3d publish_pose = result.publish_pose;
    publish_.publish_odometry(publish_pose, header);
    publish_.publish_path(header, &publish_pose);
    publish_.publish_point_clouds(cloud, publish_pose, header);

    if (on_keyframe_added_) {
        on_keyframe_added_();
    }
}

LocalizationModeHandler::LocalizationModeHandler(WorldLocalizing& world_localizing, ModePublishCallbacks publish)
  : processor_(world_localizing)
  , publish_(std::move(publish))
{
}

void
LocalizationModeHandler::process(bool map_loaded,
                                 const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud,
                                 const std_msgs::msg::Header& header)
{
    auto result = processor_.process(map_loaded, pose_odom, cloud);
    if (!result.map_loaded) {
        LOG_EVERY_N(WARNING, 10) << "Map not loaded, cannot perform localization";
        publish_.publish_odometry(result.publish_pose, header);
        publish_.publish_path(header, &result.publish_pose);
        publish_.publish_point_clouds(cloud, result.publish_pose, header);
        return;
    }

    if (result.success) {
        if (result.relocalization_locked && publish_.publish_relocalization_lock) {
            publish_.publish_relocalization_lock(header, result.publish_pose);
        }
        publish_.publish_odometry(result.publish_pose, header);
        publish_.publish_path(header, &result.publish_pose);
        publish_.publish_point_clouds(cloud, result.publish_pose, header);
    } else {
        publish_.publish_odometry(result.publish_pose, header);
        publish_.publish_path(header, &result.publish_pose);
        publish_.publish_point_clouds(cloud, result.publish_pose, header);
    }
}

MapResumingModeHandler::MapResumingModeHandler(const Config& config,
                                               KeyframeManager& keyframe_manager,
                                               GraphOptimizer& graph_optimizer,
                                               WorldLocalizing& world_localizing,
                                               MappingResuming& mapping_resuming,
                                               ModePublishCallbacks publish,
                                               std::function<void()> on_keyframe_added,
                                               rclcpp::Logger logger,
                                               rclcpp::Clock::SharedPtr clock)
  : processor_(keyframe_manager, graph_optimizer, world_localizing, mapping_resuming)
  , publish_(std::move(publish))
  , on_keyframe_added_(std::move(on_keyframe_added))
  , logger_(std::move(logger))
  , clock_(std::move(clock))
{
    (void)config;
}

void
MapResumingModeHandler::process(bool map_loaded,
                                double timestamp,
                                const Eigen::Isometry3d& pose_odom,
                                PointCloud::Ptr cloud,
                                const std_msgs::msg::Header& header)
{
    auto result = processor_.process(map_loaded, timestamp, pose_odom, cloud);
    if (!result.map_loaded) {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "Map not loaded for extension");
        return;
    }

    if (result.initial_relocalization_attempted) {
        if (result.initial_relocalization_success) {
            RCLCPP_INFO(logger_, "Initial relocalization successful for map extension");
        }
        return;
    }

    if (!result.should_publish) {
        return;
    }

    if (result.cross_loops > 0) {
        RCLCPP_INFO(logger_, "Detected %d cross-loops for keyframe %ld", result.cross_loops, result.keyframe_id);
    }
    if (result.accepted_keyframe) {
        if (on_keyframe_added_) {
            on_keyframe_added_();
        }
    }

    if (result.log_optimization) {
        publish_.log_optimization_result("map_extension_incremental", timestamp, &result.publish_pose);
    }

    publish_.publish_odometry(result.publish_pose, header);
    publish_.publish_path(header, &result.publish_pose);
    publish_.publish_point_clouds(cloud, result.publish_pose, header);
}

} // namespace n3mapping
