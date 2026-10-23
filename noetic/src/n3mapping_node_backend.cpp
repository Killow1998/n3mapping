#include "n3mapping_node.h"
#include "n3mapping_noetic/conversions.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <glog/logging.h>

namespace n3mapping {

void N3MappingNoeticNode::syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                  const nav_msgs::OdometryConstPtr& odom_msg)
{
    if (run_mode_ == CoreRunMode::LOCALIZATION) {
        pending_localization_frame_.submit(
            static_cast<int64_t>(cloud_msg->header.stamp.toNSec()),
            SynchronizedFrame{cloud_msg, odom_msg});
        return;
    }
    processSynchronizedFrame(cloud_msg, odom_msg);
}

void N3MappingNoeticNode::processSynchronizedFrame(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                              const nav_msgs::OdometryConstPtr& odom_msg)
{
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        const double frame_start_source_age_s =
            (ros::Time::now() - odom_msg->header.stamp).toSec();
        const auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);

        const auto backend_started = std::chrono::steady_clock::now();
        auto output = core_->processFrame(run_mode_, frame);
        const double backend_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - backend_started).count();
        const auto committed = realtime_output_->acceptBackend(
            {static_cast<int64_t>(odom_msg->header.stamp.toNSec()),
             frame.source_frame_id, odom_msg->child_frame_id,
             frame.T_world_lidar, frame.pose_valid},
            cloud_msg->header.stamp.toSec(), output);
        const auto& publication = committed.plan;
        const bool backend_authorized = committed.live_authorized;
        const double backend_source_age_s =
            (ros::Time::now() - odom_msg->header.stamp).toSec();
        const auto backend_completed = std::chrono::steady_clock::now();
        const bool has_previous_backend_authorized =
            previous_backend_authorized_update_.has_value();
        const double previous_backend_authorized_source_age_s =
            has_previous_backend_authorized
                ? (ros::Time::now() - previous_backend_authorized_stamp_).toSec()
                : -1.0;
        const double backend_authorized_update_gap_ms =
            has_previous_backend_authorized
                ? std::chrono::duration<double, std::milli>(
                      backend_completed - *previous_backend_authorized_update_).count()
                : -1.0;
        if (last_logged_backend_authorized_ != backend_authorized ||
            backend_seconds > 0.5) {
            LOG(INFO) << std::fixed << std::setprecision(9)
                      << "realtime_odometry_backend authorized=" << backend_authorized
                      << " process_frame_ms=" << backend_seconds * 1000.0
                      << " paired_stamp_s=" << odom_msg->header.stamp.toSec()
                      << " source_age_s=" << backend_source_age_s
                      << " frame_start_source_age_s=" << frame_start_source_age_s
                      << " has_previous_backend_authorized=" << has_previous_backend_authorized
                      << " previous_backend_authorized_source_age_s=" << previous_backend_authorized_source_age_s
                      << " backend_authorized_update_gap_ms=" << backend_authorized_update_gap_ms
                      << " state=" << relocalizationStateName(output.relocalization_state)
                      << " pose_source=" << poseSourceName(output.pose_source)
                      << " decision=" << output.relocalization_decision;
            last_logged_backend_authorized_ = backend_authorized;
        }
        if (backend_authorized) {
            previous_backend_authorized_stamp_ = odom_msg->header.stamp;
            previous_backend_authorized_update_ = backend_completed;
        } else {
            previous_backend_authorized_update_.reset();
        }
        if (run_mode_ == CoreRunMode::MAP_EXTENSION && output.relocalization_locked && !output.accepted_keyframe) {
            ROS_INFO_THROTTLE(2.0, "Initial relocalization successful for map extension");
        }
        if (publication.inconsistent_lock_event) {
            ROS_ERROR("Suppressed inconsistent relocalization lock event: state=%s pose_source=%s",
                      relocalizationStateName(output.relocalization_state),
                      poseSourceName(output.pose_source));
        }
        if (publication.invalid_usable_pose_suppressed) {
            ROS_ERROR("Suppressed non-finite or non-rigid localization pose");
        }
        if (publication.publish_legacy_lock) {
            publishLegacyRelocalizationLock(publication.lock_epoch);
        }
        if (publication.publish_status) {
            publishRelocalizationStatus(
                cloud_msg->header, output, publication.lock_epoch);
        }
        if (publication.publish_authoritative_pose) {
            publishAuthoritativeRelocalizationPose(
                cloud_msg->header, output.T_world_lidar,
                publication.lock_epoch);
        }
        if (!output.success && !output.accepted_keyframe && run_mode_ != CoreRunMode::LOCALIZATION) {
            ++frame_count_;
            return;
        }

        if (publication.publish_global_pose) {
            publishPath(cloud_msg->header, &output.T_world_lidar);
        }
        if (output.accepted_keyframe) {
            ++keyframe_count_;
            if (run_mode_ == CoreRunMode::MAPPING || run_mode_ == CoreRunMode::MAP_EXTENSION) {
                const char* context = run_mode_ == CoreRunMode::MAP_EXTENSION ? "map_extension_incremental" : "mapping_incremental";
                logOptimizationResult(context, cloud_msg->header.stamp.toSec(), &output.T_world_lidar);
            }
        }
        ++frame_count_;
    }
}

void N3MappingNoeticNode::denseOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    if (!core_ || run_mode_ != CoreRunMode::MAPPING) {
        return;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    core_->recordDenseTrajectoryPose(
        run_mode_, odom_msg->header.stamp.toSec(), odometryPoseToIsometry(*odom_msg));
}

void N3MappingNoeticNode::loopDetectionTimerCallback(const ros::TimerEvent&)
{
    if (!coreRunModeProcessesLoopClosures(run_mode_)) {
        return;
    }

    bool needs_global_map_rebuild = false;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        const auto result = core_->processPendingLoopClosuresAsync();
        if (!result.accepted_loops.empty()) {
            publishLoopMarkers(result.accepted_loops);
        }
        if (result.optimized) {
            loop_count_ += result.edge_count;
            logOptimizationResult("loop_closure", ros::Time::now().toSec(), nullptr);
            std::ostringstream oss;
            oss << "[OPTIMIZATION] loop impact edges=" << result.edge_count
                << " accepted=" << result.accepted_loops.size()
                << " loop_residual_t=" << result.loop_residual_translation_before
                << "->" << result.loop_residual_translation_after
                << " loop_residual_r=" << result.loop_residual_rotation_before
                << "->" << result.loop_residual_rotation_after
                << " pose_update_mean_max_t=" << result.mean_pose_update_translation
                << "/" << result.max_pose_update_translation
                << " pose_update_mean_max_r=" << result.mean_pose_update_rotation
                << "/" << result.max_pose_update_rotation
                << " pose_update_count=" << result.pose_update_count;
            appendOptimizationLogLine(oss.str());

            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = config_.world_frame;
            publishPath(header, nullptr);
            needs_global_map_rebuild = true;
        }
    }
    if (needs_global_map_rebuild) {
        markGlobalMapFullRebuildRequired();
    }
}

}  // namespace n3mapping
