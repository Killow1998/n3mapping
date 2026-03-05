// ModeHandlers: per-mode data processing — mapping, localization, map extension.
#pragma once

#include <functional>
#include <mutex>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {

using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

struct ModePublishCallbacks {
    std::function<void(const Eigen::Isometry3d&, const std_msgs::Header&)> publish_odometry;
    std::function<void(const std_msgs::Header&, const Eigen::Isometry3d*)> publish_path;
    std::function<void(PointCloud::Ptr, const Eigen::Isometry3d&, const std_msgs::Header&)> publish_point_clouds;
    std::function<void(const std::string&, double, const Eigen::Isometry3d*)> log_optimization_result;
};

class MappingModeHandler {
public:
    MappingModeHandler(const Config& config, KeyframeManager& keyframe_manager,
                       LoopDetector& loop_detector, GraphOptimizer& graph_optimizer,
                       std::mutex& loop_queue_mutex, std::vector<int64_t>& loop_detection_queue,
                       ModePublishCallbacks publish, std::function<void()> on_keyframe_added);

    void process(double timestamp, const Eigen::Isometry3d& pose_odom,
                 PointCloud::Ptr cloud, const std_msgs::Header& header);

private:
    const Config& config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    GraphOptimizer& graph_optimizer_;
    std::mutex& loop_queue_mutex_;
    std::vector<int64_t>& loop_detection_queue_;
    ModePublishCallbacks publish_;
    std::function<void()> on_keyframe_added_;
};

class LocalizationModeHandler {
public:
    LocalizationModeHandler(WorldLocalizing& world_localizing, ModePublishCallbacks publish);
    void process(bool map_loaded, const Eigen::Isometry3d& pose_odom,
                 PointCloud::Ptr cloud, const std_msgs::Header& header);

private:
    WorldLocalizing& world_localizing_;
    ModePublishCallbacks publish_;
};

class MapResumingModeHandler {
public:
    MapResumingModeHandler(const Config& config, KeyframeManager& keyframe_manager,
                           GraphOptimizer& graph_optimizer, WorldLocalizing& world_localizing,
                           MappingResuming& mapping_resuming, ModePublishCallbacks publish,
                           std::function<void()> on_keyframe_added);

    void process(bool map_loaded, double timestamp, const Eigen::Isometry3d& pose_odom,
                 PointCloud::Ptr cloud, const std_msgs::Header& header);

private:
    const Config& config_;
    KeyframeManager& keyframe_manager_;
    GraphOptimizer& graph_optimizer_;
    WorldLocalizing& world_localizing_;
    MappingResuming& mapping_resuming_;
    ModePublishCallbacks publish_;
    std::function<void()> on_keyframe_added_;
};

} // namespace n3mapping
