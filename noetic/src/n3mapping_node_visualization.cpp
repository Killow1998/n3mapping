#include "n3mapping_node.h"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace n3mapping {

void N3MappingNoeticNode::publishPath(const std_msgs::Header& header, const Eigen::Isometry3d* current_pose)
{
    nav_msgs::Path path_msg;
    path_msg.header = header;
    path_msg.header.frame_id = config_.world_frame;

    if (run_mode_ == CoreRunMode::LOCALIZATION) {
        if (current_pose) {
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose = makePose(*current_pose);
            localization_path_.push_back(pose);
        }
        path_msg.poses = localization_path_;
    } else {
        std::vector<geometry_msgs::Point> loaded_points;
        std::vector<geometry_msgs::Point> new_points;
        for (const auto& kf : core_->getAllKeyframes()) {
            if (!kf) continue;
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose = makePose(kf->pose_optimized);
            path_msg.poses.push_back(pose);
            auto& points = kf->is_from_loaded_map ? loaded_points : new_points;
            points.push_back(pose.pose.position);
        }
        if (current_pose) {
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose = makePose(*current_pose);
            path_msg.poses.push_back(pose);
            new_points.push_back(pose.pose.position);
        }
        publishPathMarkers(loaded_points, new_points, header);
    }

    path_pub_.publish(path_msg);
}

void N3MappingNoeticNode::publishLoopMarkers(const std::vector<VerifiedLoop>& loops)
{
    visualization_msgs::MarkerArray markers;
    for (const auto& loop : loops) {
        auto query = core_->getKeyframe(loop.query_id);
        auto match = core_->getKeyframe(loop.match_id);
        if (!query || !match) continue;

        visualization_msgs::Marker marker;
        marker.header.frame_id = config_.world_frame;
        marker.header.stamp = ros::Time::now();
        marker.ns = "loop_closure";
        marker.id = static_cast<int>(loop.query_id);
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.08;
        marker.color.r = 1.0f;
        marker.color.g = 0.2f;
        marker.color.b = 0.1f;
        marker.color.a = 0.9f;
        marker.points.push_back(toPoint(query->pose_optimized));
        marker.points.push_back(toPoint(match->pose_optimized));
        markers.markers.push_back(marker);
    }
    if (!markers.markers.empty()) {
        loop_marker_pub_.publish(markers);
    }
}

void N3MappingNoeticNode::publishPathMarkers(const std::vector<geometry_msgs::Point>& loaded_points,
                        const std::vector<geometry_msgs::Point>& new_points,
                        const std_msgs::Header& header)
{
    visualization_msgs::MarkerArray markers;
    addPathMarker("path_loaded", 0, loaded_points, 0.2f, 0.4f, 1.0f, header, &markers);
    addPathMarker("path_new", 1, new_points, 0.1f, 1.0f, 0.2f, header, &markers);
    if (!markers.markers.empty()) {
        loop_marker_pub_.publish(markers);
    }
}

void N3MappingNoeticNode::addPathMarker(const std::string& ns,
                   int id,
                   const std::vector<geometry_msgs::Point>& points,
                   float r,
                   float g,
                   float b,
                   const std_msgs::Header& header,
                   visualization_msgs::MarkerArray* markers)
{
    if (points.size() < 2) {
        return;
    }
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.header.frame_id = config_.world_frame;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.15;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.9f;
    marker.points = points;
    markers->markers.push_back(marker);
}

geometry_msgs::Point N3MappingNoeticNode::toPoint(const Eigen::Isometry3d& pose) const
{
    geometry_msgs::Point point;
    point.x = pose.translation().x();
    point.y = pose.translation().y();
    point.z = pose.translation().z();
    return point;
}

void N3MappingNoeticNode::initializeOptimizationLogging()
{
    if (!ensureMapDirectory(config_.map_save_path)) {
        ROS_WARN("Failed to create optimization log directory: %s", config_.map_save_path.c_str());
    }
    optimization_log_path_ = config_.map_save_path + "/optimization.log";
    std::ofstream file(optimization_log_path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        ROS_WARN("Failed to open optimization log: %s", optimization_log_path_.c_str());
    }
}

void N3MappingNoeticNode::logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose)
{
    auto keyframes = core_ ? core_->getAllKeyframes() : std::vector<Keyframe::Ptr>{};
    std::vector<Keyframe::Ptr> valid_keyframes;
    valid_keyframes.reserve(keyframes.size());
    for (const auto& kf : keyframes) {
        if (kf) {
            valid_keyframes.push_back(kf);
        }
    }
    std::sort(valid_keyframes.begin(), valid_keyframes.end(), [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) {
        return a->id < b->id;
    });

    const std::string latest_id = valid_keyframes.empty() ? "none" : std::to_string(valid_keyframes.back()->id);
    std::ostringstream summary;
    summary << "[OPTIMIZATION] context=" << context << " time=" << timestamp
            << " keyframes=" << valid_keyframes.size()
            << " latest_id=" << latest_id;
    appendOptimizationLogLine(summary.str());

    if (current_pose) {
        const Eigen::Quaterniond q(current_pose->rotation());
        std::ostringstream pose_line;
        pose_line << "[OPTIMIZATION] current t=" << current_pose->translation().x() << ","
                  << current_pose->translation().y() << "," << current_pose->translation().z()
                  << " q=" << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
        appendOptimizationLogLine(pose_line.str());
    }
}

void N3MappingNoeticNode::appendOptimizationLogLine(const std::string& line)
{
    std::lock_guard<std::mutex> lock(optimization_log_mutex_);
    std::ofstream file(optimization_log_path_, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        return;
    }
    file << line << '\n';
}

}  // namespace n3mapping
