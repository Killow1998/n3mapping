// ModeHandlers: per-mode data processing — mapping, localization, map extension.
#include "n3mapping/mode_handlers.h"

#include <glog/logging.h>

namespace n3mapping {

MappingModeHandler::MappingModeHandler(const Config& config,
                                       KeyframeManager& keyframe_manager,
                                       LoopDetector& loop_detector,
                                       GraphOptimizer& graph_optimizer,
                                       std::mutex& loop_queue_mutex,
                                       std::vector<int64_t>& loop_detection_queue,
                                       ModePublishCallbacks publish,
                                       std::function<void()> on_keyframe_added)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , loop_detector_(loop_detector)
    , graph_optimizer_(graph_optimizer)
    , loop_queue_mutex_(loop_queue_mutex)
    , loop_detection_queue_(loop_detection_queue)
    , publish_(std::move(publish))
    , on_keyframe_added_(std::move(on_keyframe_added)) {}

void MappingModeHandler::process(double timestamp, const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud, const std_msgs::Header& header) {
    if (!keyframe_manager_.shouldAddKeyframe(pose_odom)) {
        publish_.publish_odometry(pose_odom, header);
        publish_.publish_path(header, &pose_odom);
        publish_.publish_point_clouds(cloud, pose_odom, header);
        return;
    }

    int64_t kf_id = keyframe_manager_.addKeyframe(timestamp, pose_odom, cloud);
    loop_detector_.addDescriptor(kf_id, cloud);
    auto kf = keyframe_manager_.getKeyframe(kf_id);
    if (kf) kf->rhpd_descriptor = loop_detector_.addRHPD(kf_id, cloud);

    if (kf_id == 0) {
        graph_optimizer_.addPriorFactor(kf_id, pose_odom);
    } else {
        auto prev_kf = keyframe_manager_.getKeyframe(kf_id - 1);
        if (prev_kf) {
            EdgeInfo edge;
            edge.from_id = kf_id - 1;
            edge.to_id = kf_id;
            edge.measurement = prev_kf->pose_odom.inverse() * pose_odom;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity();
            edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
            edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
            edge.type = EdgeType::ODOMETRY;
            graph_optimizer_.addOdometryEdge(edge);
        }
    }

    graph_optimizer_.incrementalOptimize();

    auto optimized_poses = graph_optimizer_.getOptimizedPoses();
    keyframe_manager_.updateOptimizedPoses(optimized_poses);

    Eigen::Isometry3d optimized_pose = pose_odom;
    if (graph_optimizer_.hasNode(kf_id)) {
        try {
            optimized_pose = graph_optimizer_.getOptimizedPose(kf_id);
        } catch (const std::exception& e) {
            ROS_WARN("Failed to get optimized pose: %s", e.what());
        }
    }

    publish_.log_optimization_result("mapping_incremental", timestamp, &optimized_pose);

    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        loop_detection_queue_.push_back(kf_id);
    }

    publish_.publish_odometry(optimized_pose, header);
    publish_.publish_path(header, &optimized_pose);
    publish_.publish_point_clouds(cloud, optimized_pose, header);

    if (on_keyframe_added_) on_keyframe_added_();
}

LocalizationModeHandler::LocalizationModeHandler(WorldLocalizing& world_localizing, ModePublishCallbacks publish)
    : world_localizing_(world_localizing), publish_(std::move(publish)) {}

void LocalizationModeHandler::process(bool map_loaded,
                                      const Eigen::Isometry3d& pose_odom,
                                      PointCloud::Ptr cloud,
                                      const std_msgs::Header& header) {
    if (!map_loaded) {
        LOG_EVERY_N(WARNING, 10) << "Map not loaded, cannot perform localization";
        // Still publish odom-frame pose so TF tree stays valid
        publish_.publish_odometry(pose_odom, header);
        publish_.publish_path(header, &pose_odom);
        publish_.publish_point_clouds(cloud, pose_odom, header);
        return;
    }

    Eigen::Isometry3d pose_map;
    bool success = false;

    if (world_localizing_.isRelocalized()) {
        auto result = world_localizing_.trackLocalization(cloud, pose_odom);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
        }
    }
    if (!world_localizing_.isRelocalized() || !success) {
        auto result = world_localizing_.relocalize(cloud, pose_odom);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
        }
    }

    if (success) {
        publish_.publish_odometry(pose_map, header);
        publish_.publish_path(header, &pose_map);
        publish_.publish_point_clouds(cloud, pose_map, header);
    } else {
        // Relocalization not yet successful — publish odom pose as fallback
        // so that TF tree (map->body) is always valid and RViz does not error.
        Eigen::Isometry3d fallback = world_localizing_.getMapToOdomTransform() * pose_odom;
        publish_.publish_odometry(fallback, header);
        publish_.publish_path(header, &fallback);
        publish_.publish_point_clouds(cloud, fallback, header);
    }
}

MapResumingModeHandler::MapResumingModeHandler(const Config& config,
                                               KeyframeManager& keyframe_manager,
                                               GraphOptimizer& graph_optimizer,
                                               WorldLocalizing& world_localizing,
                                               MappingResuming& mapping_resuming,
                                               ModePublishCallbacks publish,
                                               std::function<void()> on_keyframe_added)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , graph_optimizer_(graph_optimizer)
    , world_localizing_(world_localizing)
    , mapping_resuming_(mapping_resuming)
    , publish_(std::move(publish))
    , on_keyframe_added_(std::move(on_keyframe_added)) {}

void MapResumingModeHandler::process(bool map_loaded, double timestamp,
                                     const Eigen::Isometry3d& pose_odom,
                                     PointCloud::Ptr cloud,
                                     const std_msgs::Header& header) {
    if (!map_loaded) {
        ROS_WARN_THROTTLE(5.0, "Map not loaded for extension");
        return;
    }

    auto state = mapping_resuming_.getState();

    if (state == MappingResumingState::MAP_LOADED) {
        if (mapping_resuming_.performInitialRelocalization(cloud, pose_odom))
            ROS_INFO("Initial relocalization successful for map extension");
        return;
    }

    if (state != MappingResumingState::RELOCALIZED && state != MappingResumingState::EXTENDING)
        return;

    if (!keyframe_manager_.shouldAddKeyframe(pose_odom)) {
        auto T_map_odom = world_localizing_.getMapToOdomTransform();
        Eigen::Isometry3d pose_map = T_map_odom * pose_odom;
        publish_.publish_odometry(pose_map, header);
        publish_.publish_path(header, &pose_map);
        publish_.publish_point_clouds(cloud, pose_map, header);
        return;
    }

    auto T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_map = T_map_odom * pose_odom;

    int64_t kf_id = mapping_resuming_.processNewKeyframe(timestamp, pose_odom, cloud);
    if (kf_id >= 0) {
        int cross_loops = mapping_resuming_.detectCrossLoops(kf_id);
        if (cross_loops > 0)
            ROS_INFO("Detected %d cross-loops for keyframe %ld", cross_loops, kf_id);

        graph_optimizer_.incrementalOptimize();

        auto optimized_poses = graph_optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);

        if (graph_optimizer_.hasNode(kf_id)) {
            try {
                pose_map = graph_optimizer_.getOptimizedPose(kf_id);
            } catch (const std::exception& e) {
                ROS_WARN("Failed to get optimized pose: %s", e.what());
            }
        }

        publish_.log_optimization_result("map_extension_incremental", timestamp, &pose_map);

        if (on_keyframe_added_) on_keyframe_added_();
    }

    publish_.publish_odometry(pose_map, header);
    publish_.publish_path(header, &pose_map);
    publish_.publish_point_clouds(cloud, pose_map, header);
}

} // namespace n3mapping
