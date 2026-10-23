#include "n3mapping_node.h"
#include <fstream>

namespace n3mapping {

void N3MappingNode::publishPath(const std_msgs::msg::Header& header, const Eigen::Isometry3d* current_pose)
{
    nav_msgs::msg::Path mapping_path;
    // Serialize localization history directly, without copying every pose.
    auto& path_msg = run_mode_ == CoreRunMode::LOCALIZATION
      ? localization_path_ : mapping_path;
    path_msg.header = header;
    path_msg.header.frame_id = config_.world_frame;

    if (run_mode_ == CoreRunMode::LOCALIZATION) {
        // 定位模式: 只累积发布实时定位轨迹
        if (current_pose) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = header;
            pose.header.frame_id = config_.world_frame;
            pose.pose.position.x = current_pose->translation().x();
            pose.pose.position.y = current_pose->translation().y();
            pose.pose.position.z = current_pose->translation().z();
            Eigen::Quaterniond q(current_pose->rotation());
            pose.pose.orientation.w = q.w();
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            path_msg.poses.push_back(pose);
        }
    } else {
        // 建图/续建模式: 按 version 分段构建彩色轨迹 marker
        // 同时保留 nav_msgs/Path 发布全部轨迹（用于数据记录/导航）
        std::vector<geometry_msgs::msg::Point> loaded_points; // 旧地图 (蓝色)
        std::vector<geometry_msgs::msg::Point> new_points;    // 新建帧 (绿色)

        for (const auto& kf : getKeyframesForPublishing()) {
            if (!kf) continue;
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;

            const auto& p = kf->pose_optimized;
            pose.pose.position.x = p.translation().x();
            pose.pose.position.y = p.translation().y();
            pose.pose.position.z = p.translation().z();

            Eigen::Quaterniond q(p.rotation());
            pose.pose.orientation.w = q.w();
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();

            path_msg.poses.push_back(pose);

            geometry_msgs::msg::Point pt;
            pt.x = p.translation().x();
            pt.y = p.translation().y();
            pt.z = p.translation().z();
            if (kf->is_from_loaded_map) {
                loaded_points.push_back(pt);
            } else {
                new_points.push_back(pt);
            }
        }

        if (current_pose) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = current_pose->translation().x();
            pose.pose.position.y = current_pose->translation().y();
            pose.pose.position.z = current_pose->translation().z();
            Eigen::Quaterniond q(current_pose->rotation());
            pose.pose.orientation.w = q.w();
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            path_msg.poses.push_back(pose);

            geometry_msgs::msg::Point pt;
            pt.x = current_pose->translation().x();
            pt.y = current_pose->translation().y();
            pt.z = current_pose->translation().z();
            new_points.push_back(pt);
        }

        // 发布彩色轨迹 marker 到 loop_closure_markers 话题
        publishPathMarkers(loaded_points, new_points, header);
    }

    path_pub_->publish(path_msg);
}

void N3MappingNode::publishPathMarkers(const std::vector<geometry_msgs::msg::Point>& loaded_points,
                        const std::vector<geometry_msgs::msg::Point>& new_points,
                        const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::MarkerArray markers;

    // 旧地图轨迹 (蓝色)
    if (loaded_points.size() >= 2) {
        visualization_msgs::msg::Marker loaded_marker;
        loaded_marker.header.frame_id = config_.world_frame;
        loaded_marker.header.stamp = header.stamp;
        loaded_marker.ns = "path_loaded";
        loaded_marker.id = 0;
        loaded_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        loaded_marker.action = visualization_msgs::msg::Marker::ADD;
        loaded_marker.scale.x = 0.15;
        loaded_marker.color.r = 0.2f;
        loaded_marker.color.g = 0.4f;
        loaded_marker.color.b = 1.0f;
        loaded_marker.color.a = 0.9f;
        loaded_marker.points = loaded_points;
        markers.markers.push_back(loaded_marker);
    }

    // 新建轨迹 (绿色)
    if (new_points.size() >= 2) {
        visualization_msgs::msg::Marker new_marker;
        new_marker.header.frame_id = config_.world_frame;
        new_marker.header.stamp = header.stamp;
        new_marker.ns = "path_new";
        new_marker.id = 0;
        new_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        new_marker.action = visualization_msgs::msg::Marker::ADD;
        new_marker.scale.x = 0.15;
        new_marker.color.r = 0.1f;
        new_marker.color.g = 1.0f;
        new_marker.color.b = 0.2f;
        new_marker.color.a = 0.9f;
        new_marker.points = new_points;
        markers.markers.push_back(new_marker);
    }

    if (!markers.markers.empty()) {
        loop_marker_pub_->publish(markers);
    }
}

void N3MappingNode::logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose)
{
    auto keyframes = getKeyframesForPublishing();
    std::vector<Keyframe::Ptr> valid_keyframes;
    valid_keyframes.reserve(keyframes.size());
    for (const auto& kf : keyframes) {
        if (kf) {
            valid_keyframes.push_back(kf);
        }
    }
    std::sort(valid_keyframes.begin(), valid_keyframes.end(), [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) { return a->id < b->id; });

    std::optional<int64_t> latest_id;
    if (!valid_keyframes.empty()) {
        latest_id = valid_keyframes.back()->id;
    }

    std::ostringstream summary;
    summary << "[OPTIMIZATION] context=" << context << " time=" << timestamp
            << " keyframes=" << valid_keyframes.size()
            << " latest_id=" << (latest_id ? std::to_string(*latest_id) : std::string("none"));
    appendOptimizationLogLine(summary.str());

    if (current_pose) {
        Eigen::Quaterniond q(current_pose->rotation());
        std::ostringstream pose_line;
        pose_line << "[OPTIMIZATION] current t=" << current_pose->translation().x() << ","
                  << current_pose->translation().y() << "," << current_pose->translation().z()
                  << " q=" << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
        appendOptimizationLogLine(pose_line.str());
    }
}

void N3MappingNode::appendOptimizationLogLine(const std::string& line)
{
    std::lock_guard<std::mutex> lock(optimization_log_mutex_);
    std::ofstream file(optimization_log_path_, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        return;
    }
    file << line << '\n';
}

void N3MappingNode::publishLoopMarkers(const std::vector<VerifiedLoop>& loops)
{
    for (const auto& loop : loops) {
        const auto key = std::make_pair(loop.match_id, loop.query_id);
        const auto it = std::find(loop_marker_history_.begin(), loop_marker_history_.end(), key);
        if (it == loop_marker_history_.end()) {
            loop_marker_history_.push_back(key);
        }
    }

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    int marker_id = 0;
    for (const auto& [match_id, query_id] : loop_marker_history_) {
        auto match_kf = getKeyframeForPublishing(match_id);
        auto query_kf = getKeyframeForPublishing(query_id);
        if (!match_kf || !query_kf) {
            continue;
        }

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = config_.world_frame;
        marker.header.stamp = this->now();
        marker.ns = "loop_closure";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.color.r = 1.0f;
        marker.color.g = 0.2f;
        marker.color.b = 0.2f;
        marker.color.a = 0.9f;

        geometry_msgs::msg::Point p_match;
        p_match.x = match_kf->pose_optimized.translation().x();
        p_match.y = match_kf->pose_optimized.translation().y();
        p_match.z = match_kf->pose_optimized.translation().z() + 0.5;

        geometry_msgs::msg::Point p_query;
        p_query.x = query_kf->pose_optimized.translation().x();
        p_query.y = query_kf->pose_optimized.translation().y();
        p_query.z = query_kf->pose_optimized.translation().z() + 0.5;

        marker.points.push_back(p_match);
        marker.points.push_back(p_query);

        markers.markers.push_back(marker);
    }

    loop_marker_pub_->publish(markers);
}

std::vector<Keyframe::Ptr> N3MappingNode::getKeyframesForPublishing() const
{
    if (n3mapping_core_) {
        return n3mapping_core_->getAllKeyframes();
    }
    return {};
}

Keyframe::Ptr N3MappingNode::getKeyframeForPublishing(int64_t id) const
{
    if (n3mapping_core_) {
        return n3mapping_core_->getKeyframe(id);
    }
    return nullptr;
}

}  // namespace n3mapping
