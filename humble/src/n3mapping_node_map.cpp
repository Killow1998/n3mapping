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
        global_map_msg_revision_ = global_map_cache_.revision();
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
    std::vector<Keyframe::Ptr> keyframes;
    KeyframeMapRevision map_revision;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        keyframes = snapshotKeyframesForGlobalMapLocked(&map_revision);
    }
    publishGlobalMap(keyframes, map_revision);
}

std::vector<Keyframe::Ptr> N3MappingNode::snapshotKeyframesForGlobalMapLocked(
    KeyframeMapRevision* map_revision) const
{
    std::vector<Keyframe::Ptr> snapshots;
    const auto keyframes = getKeyframesForPublishing();
    if (map_revision && n3mapping_core_) {
        *map_revision = n3mapping_core_->mapRevision();
    }
    snapshots.reserve(keyframes.size());
    for (const auto& keyframe : keyframes) {
        if (!keyframe) {
            continue;
        }
        auto snapshot = std::make_shared<Keyframe>();
        snapshot->id = keyframe->id;
        snapshot->timestamp = keyframe->timestamp;
        snapshot->pose_odom = keyframe->pose_odom;
        snapshot->pose_optimized = keyframe->pose_optimized;
        snapshot->cloud = keyframe->cloud;
        snapshot->is_from_loaded_map = keyframe->is_from_loaded_map;
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void N3MappingNode::resetGlobalMapCache()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    global_map_cache_.clear();
    global_map_msg_cache_.reset();
    global_map_last_published_revision_.reset();
}

void N3MappingNode::markGlobalMapFullRebuildRequired()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    global_map_cache_.markFullRebuildRequired();
    global_map_msg_cache_.reset();
    global_map_last_published_revision_.reset();
}

bool N3MappingNode::refreshGlobalMapMessage(const std::vector<Keyframe::Ptr>& keyframes,
                             const KeyframeMapRevision& map_revision,
                             sensor_msgs::msg::PointCloud2* out_msg)
{
    if (!out_msg) {
        return false;
    }

    std::lock_guard<std::mutex> lock(global_map_mutex_);
    const auto cloud = global_map_cache_.update(keyframes, map_revision);
    if (!cloud || cloud->empty()) {
        return false;
    }

    const auto revision = global_map_cache_.revision();
    if (!global_map_msg_cache_ || global_map_msg_revision_ != revision) {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = config_.world_frame;
        msg.header.stamp = this->get_clock()->now();
        global_map_msg_cache_ = msg;
        global_map_msg_revision_ = revision;
    }

    if (!global_map_msg_cache_) {
        return false;
    }
    if (global_map_last_published_revision_ &&
        *global_map_last_published_revision_ == global_map_msg_revision_) {
        return false;
    }

    *out_msg = *global_map_msg_cache_;
    global_map_last_published_revision_ = global_map_msg_revision_;
    return true;
}

void N3MappingNode::publishGlobalMap()
{
    std::vector<Keyframe::Ptr> keyframes;
    KeyframeMapRevision map_revision;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        keyframes = snapshotKeyframesForGlobalMapLocked(&map_revision);
    }
    publishGlobalMap(keyframes, map_revision);
}

void N3MappingNode::publishGlobalMap(const std::vector<Keyframe::Ptr>& keyframes,
                      const KeyframeMapRevision& map_revision)
{
    if (!n3mapping_core_) {
        return;
    }
    sensor_msgs::msg::PointCloud2 msg;
    if (!refreshGlobalMapMessage(keyframes, map_revision, &msg)) {
        return;
    }
    global_map_pub_->publish(msg);
}

void N3MappingNode::saveMapOnShutdown()
{
    std::string error;
    std::string warning;
    std::vector<Keyframe::Ptr> global_map_keyframes;
    KeyframeMapRevision global_map_revision;
    bool saved = false;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        saved = saveMapPbstreamAndSnapshotLocked(
            &error, &global_map_keyframes, &global_map_revision);
    }
    if (saved && config_.save_global_map_on_shutdown) {
        saveDebugGlobalMap(
            global_map_keyframes, global_map_revision, &warning);
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
    std::vector<Keyframe::Ptr> global_map_keyframes;
    KeyframeMapRevision global_map_revision;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!coreRunModeSavesMap(run_mode_)) {
            response->success = false;
            response->message = std::string("save_disabled_in_mode:") + coreRunModeName(run_mode_);
            RCLCPP_WARN(this->get_logger(), "save_map service rejected: %s", response->message.c_str());
            return;
        }
        response->success = saveMapPbstreamAndSnapshotLocked(
            &error, &global_map_keyframes, &global_map_revision);
    }
    if (response->success && config_.save_global_map_on_shutdown) {
        saveDebugGlobalMap(
            global_map_keyframes, global_map_revision, &warning);
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

bool N3MappingNode::saveMapPbstreamAndSnapshotLocked(std::string* error,
                                      std::vector<Keyframe::Ptr>* global_map_keyframes,
                                      KeyframeMapRevision* map_revision)
{
    if (!n3mapping_core_) {
        if (error) *error = "core_unavailable";
        return false;
    }
    if (n3mapping_core_->getAllKeyframes().empty()) {
        if (error) *error = "no_keyframes";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(config_.map_save_path, ec);
    if (ec) {
        if (error) *error = "create_map_directory_failed";
        return false;
    }

    const std::string map_file = config_.map_save_path + "/n3map.pbstream";
    if (!n3mapping_core_->saveMap(map_file)) {
        if (error) *error = "save_pbstream_failed";
        return false;
    }
    RCLCPP_INFO(this->get_logger(), "Map pbstream saved: %s", map_file.c_str());

    if (config_.save_global_map_on_shutdown && global_map_keyframes) {
        *global_map_keyframes =
            snapshotKeyframesForGlobalMapLocked(map_revision);
    }
    return true;
}

void N3MappingNode::saveDebugGlobalMap(const std::vector<Keyframe::Ptr>& keyframes,
                        const KeyframeMapRevision& map_revision,
                        std::string* warning)
{
    if (warning) {
        warning->clear();
    }
    if (keyframes.empty()) {
        if (warning) *warning = "save_global_map_empty";
        return;
    }
    std::lock_guard<std::mutex> global_map_lock(global_map_mutex_);
    const auto cloud = global_map_cache_.update(keyframes, map_revision);
    if (!cloud || cloud->empty()) {
        if (warning) *warning = "save_global_map_empty";
        return;
    }

    const std::string global_map_file = config_.map_save_path + "/global_map.pcd";
    if (pcl::io::savePCDFileBinary(global_map_file, *cloud) == -1) {
        if (warning) *warning = "save_global_map_failed";
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Debug global map saved: %s (%zu points)", global_map_file.c_str(), cloud->size());
}

}  // namespace n3mapping
