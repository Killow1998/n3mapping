#include "n3mapping_node.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace n3mapping {

void N3MappingNode::processPendingLocalizationFrame()
{
    auto pending = pending_localization_frame_.take();
    if (pending && !shutdown_called_) {
        processSynchronizedFrame(pending->cloud, pending->odom);
    }
}

void N3MappingNode::syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                  const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
{
    if (run_mode_ == CoreRunMode::LOCALIZATION) {
        pending_localization_frame_.submit(toCoreTimeStamp(cloud->header.stamp).nsec,
                                          SynchronizedFrame{cloud, odom});
    } else {
        processSynchronizedFrame(cloud, odom);
    }
}

void N3MappingNode::processSynchronizedFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
{
    using PerformanceClock = std::chrono::steady_clock;
    const bool performance_enabled =
      performance_debug_logger_ && performance_debug_logger_->ready();
    const auto callback_started = performance_enabled
      ? PerformanceClock::now()
      : PerformanceClock::time_point{};
    const auto elapsed_ms = [&](PerformanceClock::time_point started) {
        return std::chrono::duration<double, std::milli>(
          PerformanceClock::now() - started).count();
    };

    RuntimePerformanceDebugEvent performance_event;
    if (performance_enabled) {
        performance_event.mode = coreRunModeName(run_mode_);
        performance_event.processing_time =
          std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        performance_event.sensor_timestamp =
          rclcpp::Time(cloud_msg->header.stamp).seconds();
        performance_event.input_points =
          static_cast<std::size_t>(cloud_msg->width) * cloud_msg->height;
    }

    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        const auto locked_started = performance_enabled
          ? PerformanceClock::now()
          : PerformanceClock::time_point{};
        if (performance_enabled) {
            performance_event.callback_lock_wait_ms =
              std::chrono::duration<double, std::milli>(
                locked_started - callback_started).count();
            performance_event.frame_index = ++performance_frame_index_;
            if (last_performance_callback_started_) {
                performance_event.callback_interarrival_ms =
                  std::chrono::duration<double, std::milli>(
                    callback_started - *last_performance_callback_started_)
                    .count();
            }
            last_performance_callback_started_ = callback_started;
            if (last_performance_sensor_timestamp_) {
                performance_event.sensor_delta_ms =
                  (performance_event.sensor_timestamp -
                   *last_performance_sensor_timestamp_) * 1000.0;
            }
            last_performance_sensor_timestamp_ =
              performance_event.sensor_timestamp;
        }

        const auto conversion_started = performance_enabled
          ? PerformanceClock::now()
          : PerformanceClock::time_point{};
        auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);
        if (performance_enabled) {
            performance_event.ros_conversion_ms =
              elapsed_ms(conversion_started);
        }
        const double timestamp = rclcpp::Time(cloud_msg->header.stamp).seconds();
        const auto core_started = PerformanceClock::now();
        auto output = n3mapping_core_->processFrame(run_mode_, frame);
        runtime_events_.backend(output, frame.stamp.nsec, elapsed_ms(core_started));
        if (performance_enabled) {
            performance_event.core_frame_ms = elapsed_ms(core_started);
            populateFramePerformance(performance_event, output);
        }
        // Retain the human-readable divergence context alongside shared events.
        if (const auto& sanity = n3mapping_core_->odometrySanity();
            sanity.diverged && !odometry_divergence_reported_) {
            odometry_divergence_reported_ = true;
            RCLCPP_ERROR(get_logger(),
                "[ODOM-SANITY] front-end odometry diverged at t=%.3f "
                "(speed=%.1f m/s, rate=%.0f deg/s, %d consecutive). "
                "Mapping stopped; the saved map covers only what came before.",
                sanity.timestamp, sanity.speed_mps, sanity.angular_rate_dps,
                sanity.consecutive);
        }
        const auto& header = cloud_msg->header;
        const auto authority_started = performance_enabled
          ? PerformanceClock::now()
          : PerformanceClock::time_point{};
        const auto publication =
          publishRelocalizationOutput(header, *odom_msg, output);
        if (performance_enabled) {
            performance_event.authority_publish_ms =
              elapsed_ms(authority_started);
            // Live odometry/cloud publication is measured outside this
            // serial backend callback; these publication counters stay false.
        }
        if (!output.success && !output.accepted_keyframe && run_mode_ != CoreRunMode::LOCALIZATION) {
            ++frame_count_;
            performance_event.callback_skipped = true;
        } else {
            if (publication.publish_global_pose) {
                const auto pose_publish_started = performance_enabled
                  ? PerformanceClock::now()
                  : PerformanceClock::time_point{};
                publishPath(header, &output.T_world_lidar);
                if (performance_enabled) {
                    performance_event.odometry_path_publish_ms =
                      elapsed_ms(pose_publish_started);
                }
            }

            if (output.accepted_keyframe && (run_mode_ == CoreRunMode::MAPPING || run_mode_ == CoreRunMode::MAP_EXTENSION)) {
                ++keyframe_count_;
                const char* context = run_mode_ == CoreRunMode::MAP_EXTENSION ? "map_extension_incremental" : "mapping_incremental";
                logOptimizationResult(context, timestamp, &output.T_world_lidar);
            }

            ++frame_count_;
        }
        if (performance_enabled) {
            performance_event.callback_locked_ms =
              elapsed_ms(locked_started);
        }
    }

    if (performance_enabled) {
        performance_event.callback_total_ms =
          elapsed_ms(callback_started);
        if (!performance_debug_logger_->append(performance_event) &&
            !performance_log_failure_reported_.exchange(true)) {
            RCLCPP_WARN(
              this->get_logger(),
              "Runtime performance JSONL write failed: %s",
              performance_debug_logger_->path().c_str());
        }
    }
}

void N3MappingNode::denseOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
{
    if (!n3mapping_core_ || run_mode_ != CoreRunMode::MAPPING) {
        return;
    }

    const auto source = toCoreOdometrySource(*odom_msg);
    if (!source.pose_valid) return;
    std::lock_guard<std::mutex> lock(data_mutex_);
    const double timestamp = rclcpp::Time(odom_msg->header.stamp).seconds();
    n3mapping_core_->recordDenseTrajectoryPose(run_mode_, timestamp, source.pose);
}

void N3MappingNode::loopDetectionTimerCallback()
{
    if (!coreRunModeProcessesLoopClosures(run_mode_)) {
        return;
    }

    using PerformanceClock = std::chrono::steady_clock;
    const bool performance_enabled =
      performance_debug_logger_ && performance_debug_logger_->ready();
    const auto callback_started = performance_enabled
      ? PerformanceClock::now()
      : PerformanceClock::time_point{};
    const auto elapsed_ms = [&](PerformanceClock::time_point started) {
        return std::chrono::duration<double, std::milli>(
          PerformanceClock::now() - started).count();
    };
    RuntimePerformanceLoopEvent performance_event;
    if (performance_enabled) {
        performance_event.mode = coreRunModeName(run_mode_);
        performance_event.processing_time =
          std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        performance_event.cycle_index = ++performance_loop_cycle_index_;
    }

    bool needs_global_map_rebuild = false;
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        if (performance_enabled) {
            performance_event.lock_wait_ms = elapsed_ms(callback_started);
        }
        const auto core_started = performance_enabled
          ? PerformanceClock::now()
          : PerformanceClock::time_point{};
        const auto result = n3mapping_core_->processPendingLoopClosuresAsync();
        runtime_events_.loop(result);
        if (performance_enabled) {
            performance_event.core_ms = elapsed_ms(core_started);
            populateLoopPerformance(performance_event, result);
        }
        const auto publish_started = performance_enabled
          ? PerformanceClock::now()
          : PerformanceClock::time_point{};
        if (!result.accepted_loops.empty()) {
            publishLoopMarkers(result.accepted_loops);
        }
        if (result.optimized) {
            loop_count_ += result.edge_count;
            logOptimizationResult("loop_closure", this->now().seconds(), nullptr);
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

            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = config_.world_frame;
            publishPath(header, nullptr);
            needs_global_map_rebuild = true;
        }
        if (performance_enabled) {
            performance_event.publish_ms = elapsed_ms(publish_started);
        }
    }
    if (needs_global_map_rebuild) {
        markGlobalMapFullRebuildRequired();
    }
    if (performance_enabled) {
        performance_event.total_ms = elapsed_ms(callback_started);
        if (!performance_debug_logger_->append(performance_event) &&
            !performance_log_failure_reported_.exchange(true)) {
            RCLCPP_WARN(
              this->get_logger(),
              "Runtime performance JSONL write failed: %s",
              performance_debug_logger_->path().c_str());
        }
    }
}

}  // namespace n3mapping
