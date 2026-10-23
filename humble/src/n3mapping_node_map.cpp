#include "n3mapping_node.h"
#include "n3mapping/pcl_compat.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

void N3MappingNode::loadAndPublishGlobalMap()
{
    std::filesystem::path pbstream_path(config_.map_path);
    std::filesystem::path global_map_path = pbstream_path.parent_path() / "global_map.pcd";

    if (!std::filesystem::exists(global_map_path)) {
        RCLCPP_WARN(this->get_logger(), "Global map PCD not found: %s, building from loaded keyframes", global_map_path.c_str());
        publishGlobalMap();
        return;
    }

    auto global_map = pcl::make_shared<PointCloud>();
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(global_map_path.string(), *global_map) == -1) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load global map: %s", global_map_path.c_str());
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Loaded global map: %s (%zu points)", global_map_path.c_str(), global_map->size());

    // 转换为 ROS 消息并发布
    sensor_msgs::msg::PointCloud2 global_map_msg;
    pcl::toROSMsg(*global_map, global_map_msg);
    global_map_msg.header.frame_id = config_.world_frame;
    global_map_msg.header.stamp = this->get_clock()->now();
    {
        std::lock_guard<std::mutex> lock(global_map_mutex_);
        global_map_msg_cache_ = global_map_msg;
        global_map_msg_revision_ = map_artifacts_->revision();
        global_map_last_published_revision_ = global_map_msg_revision_;
    }
    global_map_pub_->publish(global_map_msg);

    RCLCPP_INFO(this->get_logger(), "Published global map on /n3mapping/global_map");
}

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
    global_map_msg_cache_.reset();
    global_map_last_published_revision_.reset();
}

void N3MappingNode::markGlobalMapFullRebuildRequired()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    map_artifacts_->invalidate();
    global_map_msg_cache_.reset();
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
        if (!global_map_msg_cache_ || global_map_msg_revision_ != revision) {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*cloud, msg);
            msg.header.frame_id = config_.world_frame;
            msg.header.stamp = this->get_clock()->now();
            global_map_msg_cache_ = msg;
            global_map_msg_revision_ = revision;
        }

        if (!global_map_msg_cache_) {
            return;
        }
        if (global_map_last_published_revision_ &&
            *global_map_last_published_revision_ == global_map_msg_revision_) {
            return;
        }

        *out_msg = *global_map_msg_cache_;
        global_map_last_published_revision_ = global_map_msg_revision_;
        changed = true;
    });
    return changed;
}

void N3MappingNode::publishGlobalMap()
{
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        snapshot = snapshotMapKeyframes(n3mapping_core_.get());
    }
    publishGlobalMap(snapshot);
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

}  // namespace n3mapping
