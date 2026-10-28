#include "n3mapping_node.h"
#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

void N3MappingNode::globalMapTimerCallback()
{
    if (!global_map_pub_ || global_map_pub_->get_subscription_count() == 0) {
        return;
    }
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        snapshot = snapshotMapKeyframes(n3mapping_core_.get());
    }
    publishGlobalMap(snapshot);
}

void N3MappingNode::resetGlobalMapCache()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    map_artifacts_->clear();
    global_map_last_published_revision_.reset();
}

void N3MappingNode::markGlobalMapFullRebuildRequired()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    map_artifacts_->invalidate();
    global_map_last_published_revision_.reset();
}

bool N3MappingNode::refreshGlobalMapMessage(const MapKeyframeSnapshot& snapshot,
                             sensor_msgs::msg::PointCloud2* out_msg)
{
    if (!out_msg) {
        return false;
    }

    std::lock_guard<std::mutex> lock(global_map_mutex_);
    bool changed = false;
    map_artifacts_->withGlobalCloud(snapshot, [&](const auto& global) {
        const auto& cloud = global.cloud;
        if (!cloud || cloud->empty()) {
            return;
        }

        const auto revision = global.revision;
        if (global_map_last_published_revision_ == revision) {
            return;
        }
        pcl::toROSMsg(*cloud, *out_msg);
        out_msg->header.frame_id = config_.world_frame;
        out_msg->header.stamp = this->get_clock()->now();
        global_map_last_published_revision_ = revision;
        changed = true;
    });
    return changed;
}

void N3MappingNode::publishGlobalMap(const MapKeyframeSnapshot& snapshot)
{
    if (!n3mapping_core_) {
        return;
    }
    sensor_msgs::msg::PointCloud2 msg;
    if (!refreshGlobalMapMessage(snapshot, &msg)) {
        return;
    }
    global_map_pub_->publish(msg);
}

void N3MappingNode::saveMapOnShutdown()
{
    std::string error;
    std::string warning;
    MapKeyframeSnapshot snapshot;
    bool saved = false;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        saved = map_artifacts_->save(n3mapping_core_.get(), &error, &snapshot);
    }
    if (saved && config_.save_global_map_on_shutdown) {
        map_artifacts_->saveDebug(snapshot, &warning);
    }
    if (saved) {
        RCLCPP_INFO(this->get_logger(), "Map snapshot saved under: %s", config_.map_save_path.c_str());
        if (!warning.empty()) {
            RCLCPP_WARN(this->get_logger(), "Map snapshot debug artifact warning: %s", warning.c_str());
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to save map snapshot: %s", error.c_str());
    }
}

void N3MappingNode::handleSaveMap(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    std::string error;
    std::string warning;
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        response->success = map_artifacts_->save(n3mapping_core_.get(), &error, &snapshot);
    }
    if (response->success && config_.save_global_map_on_shutdown) {
        map_artifacts_->saveDebug(snapshot, &warning);
    }
    response->message = response->success ? ("saved:" + config_.map_save_path) : (error.empty() ? "save_failed" : error);
    if (response->success && !warning.empty()) {
        response->message += " warning:" + warning;
    }
    if (response->success) {
        RCLCPP_INFO(this->get_logger(), "save_map service finished: %s", response->message.c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "save_map service failed: %s", response->message.c_str());
    }
}

void N3MappingNode::loadMap()
{
    if (config_.map_path.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Map path is empty!");
        return;
    }

    if (!n3mapping_core_->loadMap(config_.map_path)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load map: %s", config_.map_path.c_str());
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded map with %zu keyframes", n3mapping_core_->getAllKeyframes().size());
    map_loaded_ = true;
    resetGlobalMapCache();
}

}  // namespace n3mapping
