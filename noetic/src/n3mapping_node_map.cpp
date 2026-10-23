#include "n3mapping_node.h"
#include <pcl_conversions/pcl_conversions.h>

namespace n3mapping {

void N3MappingNoeticNode::globalMapTimerCallback(const ros::TimerEvent&)
{
    if (global_map_pub_.getNumSubscribers() == 0) {
        return;
    }
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        snapshot = snapshotMapKeyframes(core_.get());
    }
    publishGlobalMap(snapshot);
}

bool N3MappingNoeticNode::handleSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
{
    std::string error;
    std::string warning;
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        res.success = map_artifacts_->save(core_.get(), &error, &snapshot);
    }
    if (res.success && config_.save_global_map_on_shutdown) {
        map_artifacts_->saveDebug(snapshot, &warning);
    }
    res.message = res.success ? ("saved:" + config_.map_save_path) : (error.empty() ? "save_failed" : error);
    if (res.success && !warning.empty()) {
        res.message += " warning:" + warning;
    }
    if (res.success) {
        ROS_INFO("save_map service finished: %s", res.message.c_str());
    } else {
        ROS_ERROR("save_map service failed: %s", res.message.c_str());
    }
    return true;
}

void N3MappingNoeticNode::resetGlobalMapCache()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    map_artifacts_->clear();
    global_map_msg_cache_.reset();
    global_map_last_published_revision_.reset();
}

void N3MappingNoeticNode::markGlobalMapFullRebuildRequired()
{
    std::lock_guard<std::mutex> lock(global_map_mutex_);
    map_artifacts_->invalidate();
    global_map_msg_cache_.reset();
    global_map_last_published_revision_.reset();
}

bool N3MappingNoeticNode::refreshGlobalMapMessage(const MapKeyframeSnapshot& snapshot,
                             sensor_msgs::PointCloud2* out_msg)
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
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*cloud, msg);
            msg.header.frame_id = config_.world_frame;
            msg.header.stamp = ros::Time::now();
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

void N3MappingNoeticNode::publishGlobalMap()
{
    MapKeyframeSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        snapshot = snapshotMapKeyframes(core_.get());
    }
    publishGlobalMap(snapshot);
}

void N3MappingNoeticNode::publishGlobalMap(const MapKeyframeSnapshot& snapshot)
{
    sensor_msgs::PointCloud2 msg;
    if (!refreshGlobalMapMessage(snapshot, &msg)) {
        return;
    }
    global_map_pub_.publish(msg);
}

void N3MappingNoeticNode::saveMapOnShutdown()
{
    std::string error;
    std::string warning;
    MapKeyframeSnapshot snapshot;
    bool saved = false;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        saved = map_artifacts_->save(core_.get(), &error, &snapshot);
    }
    if (saved && config_.save_global_map_on_shutdown) {
        map_artifacts_->saveDebug(snapshot, &warning);
    }
    if (saved) {
        ROS_INFO("Map snapshot saved under: %s", config_.map_save_path.c_str());
        if (!warning.empty()) {
            ROS_WARN("Map snapshot debug artifact warning: %s", warning.c_str());
        }
    } else {
        ROS_ERROR("Failed to save map snapshot: %s", error.c_str());
    }
}

void N3MappingNoeticNode::loadMap()
{
    if (config_.map_path.empty()) {
        ROS_ERROR("Map path is empty");
        return;
    }
    if (!core_->loadMap(config_.map_path)) {
        ROS_ERROR("Failed to load map: %s", config_.map_path.c_str());
        return;
    }
    ROS_INFO("Loaded map with %zu keyframes", core_->getAllKeyframes().size());
    resetGlobalMapCache();
}

}  // namespace n3mapping
