#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/Point.h>
#include <glog/logging.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt32.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <n3mapping/RelocalizationStatus.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/latest_frame_slot.h"
#include "n3mapping/core/realtime_odometry.h"
#include "n3mapping/odometry_pose_validation.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping/product_build_identity.h"
#include "n3mapping/relocalization_output_authority.h"
#include "n3mapping/ros_output_contract.h"
#include "n3mapping_noetic/config_noetic.h"
#include "n3mapping_noetic/conversions.h"

namespace n3mapping {

class N3MappingNoeticNode {
  public:
    N3MappingNoeticNode()
      : nh_()
      , private_nh_("~")
    {
        std::string input_linear_velocity_frame = "child";
        private_nh_.param("input_linear_velocity_frame", input_linear_velocity_frame,
                          input_linear_velocity_frame);
        input_linear_velocity_frame_ =
            parseInputLinearVelocityFrame(input_linear_velocity_frame);
        private_nh_.param("output_status_topic", output_status_topic_,
                          std::string("/localization/status"));
        private_nh_.param(
            "product_profile_v1", product_profile_v1_, product_profile_v1_);
        if (product_profile_v1_) {
            std::string map_path;
            std::string atlas_path;
            private_nh_.param("map_path", map_path, map_path);
            private_nh_.param("reloc_atlas_path", atlas_path, atlas_path);
            if (map_path.empty() || atlas_path.empty()) {
                throw std::runtime_error(
                    "product_profile_v1 requires non-empty map_path and "
                    "reloc_atlas_path");
            }
            config_ = makeProductLocalizationConfig(map_path, atlas_path);
        } else {
            loadConfigFromNoetic(private_nh_, &config_);
        }
        global_map_cache_.setVoxelSize(config_.global_map_voxel_size);
        ROS_INFO("%s", config_.toString().c_str());

        run_mode_ = parseCoreRunMode(config_.mode);
        core_ = std::make_unique<N3MappingCore>(config_);
        core_->setExternalDenseTrajectoryRecordingEnabled(
            run_mode_ == CoreRunMode::MAPPING);
        if (coreRunModeSavesMap(run_mode_)) {
            initializeOptimizationLogging();
        }

        initializeRosInterfaces();

        if (coreRunModeLoadsMap(run_mode_)) {
            loadMap();
            if (core_->mapLoaded()) {
                publishGlobalMap();
            }
        }

        ROS_INFO("N3Mapping Noetic node initialized. Mode: %s", config_.mode.c_str());
    }

    ~N3MappingNoeticNode() { shutdown(); }

    void spin()
    {
        live_odom_spinner_->start();
        if (run_mode_ != CoreRunMode::LOCALIZATION) {
            ros::spin();
            return;
        }
        while (ros::ok()) {
            // Drain received pairs before spending time on geometric tracking.
            // All callbacks and backend work retain their single-thread owner.
            ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.01));
            auto pending = pending_localization_frame_.take();
            if (ros::ok() && pending) {
                processSynchronizedFrame(pending->cloud, pending->odom);
            }
        }
    }

  private:
    using ApproxSyncPolicy =
        message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using ExactSyncPolicy =
        message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using ApproxSynchronizer =
        message_filters::Synchronizer<ApproxSyncPolicy>;
    using ExactSynchronizer =
        message_filters::Synchronizer<ExactSyncPolicy>;

    void initializeRosInterfaces()
    {
        const int sync_queue_size = std::max(1, config_.sync_queue_size);
        cloud_sub_.subscribe(nh_, config_.cloud_topic, static_cast<uint32_t>(sync_queue_size));
        odom_sub_.subscribe(nh_, config_.odom_topic, static_cast<uint32_t>(sync_queue_size));
        if (product_profile_v1_) {
            ExactSyncPolicy sync_policy(static_cast<uint32_t>(sync_queue_size));
            exact_sync_ = std::make_unique<ExactSynchronizer>(
                static_cast<const ExactSyncPolicy&>(sync_policy),
                cloud_sub_, odom_sub_);
            exact_sync_->registerCallback(boost::bind(
                &N3MappingNoeticNode::syncCallback, this, _1, _2));
        } else {
            ApproxSyncPolicy sync_policy(
                static_cast<uint32_t>(sync_queue_size));
            sync_policy.setMaxIntervalDuration(
                ros::Duration(config_.sync_time_tolerance));
            approx_sync_ = std::make_unique<ApproxSynchronizer>(
                static_cast<const ApproxSyncPolicy&>(sync_policy),
                cloud_sub_, odom_sub_);
            approx_sync_->registerCallback(boost::bind(
                &N3MappingNoeticNode::syncCallback, this, _1, _2));
        }
        if (run_mode_ == CoreRunMode::MAPPING) {
            dense_odom_sub_ = nh_.subscribe(
                config_.odom_topic, static_cast<uint32_t>(sync_queue_size),
                &N3MappingNoeticNode::denseOdomCallback, this);
        }

        odom_pub_ = nh_.advertise<nav_msgs::Odometry>(config_.output_odom_topic, 10);
        auto live_options = ros::SubscribeOptions::create<nav_msgs::Odometry>(
            config_.odom_topic, 1,
            boost::bind(&N3MappingNoeticNode::liveOdomCallback, this, _1),
            ros::VoidPtr(), &live_odom_queue_);
        live_odom_sub_ = nh_.subscribe(live_options);
        ros::NodeHandle live_nh(nh_);
        live_nh.setCallbackQueue(&live_odom_queue_);
        live_cloud_sub_.subscribe(live_nh, config_.cloud_topic, sync_queue_size);
        live_paired_odom_sub_.subscribe(live_nh, config_.odom_topic, sync_queue_size);
        if (product_profile_v1_) {
            live_exact_sync_ = std::make_unique<ExactSynchronizer>(
                ExactSyncPolicy(sync_queue_size), live_cloud_sub_, live_paired_odom_sub_);
            live_exact_sync_->registerCallback(boost::bind(
                &N3MappingNoeticNode::liveCloudCallback, this, _1, _2));
        } else {
            ApproxSyncPolicy policy(sync_queue_size);
            policy.setMaxIntervalDuration(ros::Duration(config_.sync_time_tolerance));
            live_approx_sync_ = std::make_unique<ApproxSynchronizer>(
                static_cast<const ApproxSyncPolicy&>(policy), live_cloud_sub_, live_paired_odom_sub_);
            live_approx_sync_->registerCallback(boost::bind(
                &N3MappingNoeticNode::liveCloudCallback, this, _1, _2));
        }
        live_odom_spinner_ = std::make_unique<ros::AsyncSpinner>(1, &live_odom_queue_);
        localization_status_pub_ = nh_.advertise<std_msgs::String>(output_status_topic_, 1, false);
        path_pub_ = nh_.advertise<nav_msgs::Path>(config_.output_path_topic, 10);
        cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_body_topic, 10);
        cloud_world_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_world_topic, 10);
        loop_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/n3mapping/loop_closure_markers", 10, true);
        global_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/n3mapping/global_map", 1, true);
        relocalization_lock_pub_ = nh_.advertise<std_msgs::UInt32>("/n3mapping/relocalization_lock", 10);
        relocalization_status_pub_ =
            nh_.advertise<RelocalizationStatus>("/n3mapping/relocalization_status", 1, true);
        relocalization_pose_pub_ =
            nh_.advertise<geometry_msgs::PoseStamped>("/n3mapping/relocalization_pose", 1, false);
        if (coreRunModeSavesMap(run_mode_)) {
            save_map_srv_ = nh_.advertiseService(
                "/n3mapping/save_map",
                &N3MappingNoeticNode::handleSaveMap, this);
        }

        loop_timer_ = nh_.createTimer(ros::Duration(0.1), &N3MappingNoeticNode::loopTimerCallback, this);
        const double global_map_hz = std::max(0.1, config_.global_map_publish_hz);
        global_map_timer_ =
            nh_.createTimer(ros::Duration(1.0 / global_map_hz), &N3MappingNoeticNode::globalMapTimerCallback, this);
    }

    void loadMap()
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

    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
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

    void processSynchronizedFrame(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
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
            const bool localization_mode = run_mode_ == CoreRunMode::LOCALIZATION;
            const auto output_mode =
                localization_mode
                    ? RelocalizationOutputMode::LOCALIZATION
                    : (run_mode_ == CoreRunMode::MAP_EXTENSION
                           ? RelocalizationOutputMode::MAP_EXTENSION
                           : RelocalizationOutputMode::OTHER);
            const auto publication =
                relocalization_output_authority_.process(output_mode, output);
            const bool live_authorized = publication.publish_global_pose &&
                (localization_mode ? hasAuthoritativeRelocalizationInitializationPose(
                    output.relocalization_state, output.pose_source) :
                    (output.success || output.accepted_keyframe));
            std::optional<core::MapOdometryCorrection> correction;
            {
                // Install correction and its observation as one small snapshot.
                std::lock_guard<std::mutex> status_lock(status_mutex_);
                correction = realtime_odometry_.updateCorrection(
                    static_cast<int64_t>(odom_msg->header.stamp.toNSec()),
                    odom_msg->header.frame_id, odom_msg->child_frame_id,
                    frame.T_world_lidar, output.T_world_lidar,
                    live_authorized && frame.pose_valid);
                last_backend_status_ = output;
                last_backend_status_.cloud_body.reset();
                last_backend_status_.cloud_world.reset();
                observation_stamp_ = cloud_msg->header.stamp.toSec();
            }
            if (correction) {
                publishCorrection(*correction);
            }
            const bool backend_authorized = live_authorized && frame.pose_valid;
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

    void liveOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        const auto received = std::chrono::steady_clock::now();
        const double receive_gap_s = last_live_odom_receive_
            ? std::chrono::duration<double>(received - *last_live_odom_receive_).count()
            : 0.0;
        last_live_odom_receive_ = received;
        Eigen::Isometry3d raw_pose;
        const auto& pose = odom_msg->pose.pose;
        const bool pose_valid = tryMakeRigidOdometryPose(
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z,
            pose.orientation.w, &raw_pose);
        core::RealtimeOdometryDiagnostic diagnostic;
        core::BackendOutput backend_status;
        RealtimeLocalizationStatus realtime;
        std::optional<core::RealtimeOdometryPose> output;
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            output = realtime_odometry_.project(
                static_cast<int64_t>(odom_msg->header.stamp.toNSec()),
                static_cast<int64_t>(ros::Time::now().toNSec()),
                odom_msg->header.frame_id, odom_msg->child_frame_id,
                raw_pose, pose_valid, &diagnostic);
            backend_status = last_backend_status_;
            realtime.observation_stamp = observation_stamp_;
        }
        const char* reason = core::realtimeOdometryReasonName(diagnostic.reason);
        bool estimate_available = false;
        if (output) {
            estimate_available = publishOdometry(output->pose, odom_msg->header, *odom_msg);
            if (!estimate_available) {
                reason = "invalid_twist_or_body_frame";
            }
        }
        if (diagnostic.reason != core::RealtimeOdometryReason::DuplicateInput &&
            diagnostic.reason != core::RealtimeOdometryReason::InputBeforeCorrection) {
            realtime.correction_stamp = diagnostic.correction_stamp_nsec * 1e-9;
            realtime.estimate_available = estimate_available;
            realtime.input_reason = reason;
            std_msgs::String status;
            status.data = localizationStatusJson(run_mode_, backend_status,
                odom_msg->header.stamp.toSec(), config_.world_frame, &realtime);
            localization_status_pub_.publish(status);
        }
        if (last_logged_live_reason_ != reason ||
            receive_gap_s > 0.5) {
            LOG(INFO) << std::fixed << std::setprecision(9)
                      << "realtime_odometry reason=" << reason
                      << " input_stamp_s=" << diagnostic.input_stamp_nsec * 1e-9
                      << " correction_stamp_s=" << diagnostic.correction_stamp_nsec * 1e-9
                      << " source_age_s=" << (diagnostic.now_nsec - diagnostic.input_stamp_nsec) * 1e-9
                      << " correction_age_s=" << (diagnostic.now_nsec - diagnostic.correction_stamp_nsec) * 1e-9
                      << " input_receive_gap_s=" << receive_gap_s
                      << " source_frame=" << odom_msg->header.frame_id
                      << " child_frame=" << odom_msg->child_frame_id;
            last_logged_live_reason_ = reason;
        }
    }

    void denseOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        if (!core_ || run_mode_ != CoreRunMode::MAPPING) {
            return;
        }

        std::lock_guard<std::mutex> lock(data_mutex_);
        core_->recordDenseTrajectoryPose(
            run_mode_, odom_msg->header.stamp.toSec(), odometryPoseToIsometry(*odom_msg));
    }

    void loopTimerCallback(const ros::TimerEvent&)
    {
        if (!coreRunModeProcessesLoopClosures(run_mode_)) {
            return;
        }

        bool needs_global_map_rebuild = false;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            const auto result = core_->processPendingLoopClosures();
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

    void globalMapTimerCallback(const ros::TimerEvent&)
    {
        if (global_map_pub_.getNumSubscribers() == 0) {
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

    bool handleSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        std::string error;
        std::string warning;
        std::vector<Keyframe::Ptr> global_map_keyframes;
        KeyframeMapRevision global_map_revision;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (!coreRunModeSavesMap(run_mode_)) {
                res.success = false;
                res.message = std::string("save_disabled_in_mode:") + coreRunModeName(run_mode_);
                ROS_WARN("save_map service rejected: %s", res.message.c_str());
                return true;
            }
            res.success = saveMapPbstreamAndSnapshotLocked(
                &error, &global_map_keyframes, &global_map_revision);
        }
        if (res.success && config_.save_global_map_on_shutdown) {
            saveDebugGlobalMap(
                global_map_keyframes, global_map_revision, &warning);
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

    bool saveMapPbstreamAndSnapshotLocked(std::string* error,
                                          std::vector<Keyframe::Ptr>* global_map_keyframes,
                                          KeyframeMapRevision* map_revision)
    {
        if (!core_) {
            if (error) *error = "core_unavailable";
            return false;
        }
        if (core_->getAllKeyframes().empty()) {
            if (error) *error = "no_keyframes";
            return false;
        }
        if (!ensureDirectoryExists(config_.map_save_path)) {
            if (error) *error = "create_map_directory_failed";
            return false;
        }

        const std::string map_file = config_.map_save_path + "/n3map.pbstream";
        if (!core_->saveMap(map_file)) {
            if (error) *error = "save_pbstream_failed";
            return false;
        }
        ROS_INFO("Map pbstream saved: %s", map_file.c_str());

        if (config_.save_global_map_on_shutdown && global_map_keyframes) {
            *global_map_keyframes =
                snapshotKeyframesForGlobalMapLocked(map_revision);
        }
        return true;
    }

    void saveDebugGlobalMap(const std::vector<Keyframe::Ptr>& keyframes,
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
        ROS_INFO("Debug global map saved: %s (%zu points)", global_map_file.c_str(), cloud->size());
    }

    bool publishOdometry(const Eigen::Isometry3d& pose, const std_msgs::Header& header,
                         const nav_msgs::Odometry& input)
    {
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = config_.world_frame;
        odom_msg.child_frame_id = config_.body_frame;
        odom_msg.pose.pose = makePose(pose);
        if (!forwardOdometryTwist(input, input_linear_velocity_frame_, &odom_msg)) {
            return false;
        }
        odom_pub_.publish(odom_msg);
        return true;
    }

    void publishCorrection(const core::MapOdometryCorrection& correction)
    {
        ros::Time stamp;
        stamp.fromNSec(correction.stamp_nsec);
        if (!last_tf_stamp_.isZero() && stamp <= last_tf_stamp_) {
            return;
        }
        geometry_msgs::TransformStamped tf;
        tf.header.frame_id = config_.world_frame;
        tf.header.stamp = stamp;
        tf.child_frame_id = correction.odom_frame;
        const auto pose = makePose(correction.map_to_odom);
        tf.transform.translation.x = pose.position.x;
        tf.transform.translation.y = pose.position.y;
        tf.transform.translation.z = pose.position.z;
        tf.transform.rotation = pose.orientation;
        // The upstream odometry owns odom -> body. Never give body two TF parents.
        tf_broadcaster_.sendTransform(tf);
        last_tf_stamp_ = stamp;
    }

    void publishPath(const std_msgs::Header& header, const Eigen::Isometry3d* current_pose)
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

    void liveCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud,
                           const nav_msgs::OdometryConstPtr& odom)
    {
        if (cloud->header.frame_id != odom->child_frame_id ||
            odom->child_frame_id != config_.body_frame) return;
        cloud_body_pub_.publish(cloud);
        if (cloud_world_pub_.getNumSubscribers() == 0) return;
        const auto& p = odom->pose.pose;
        Eigen::Isometry3d raw;
        if (!tryMakeRigidOdometryPose(p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w, &raw)) return;
        const auto projected = realtime_odometry_.projectScan(
            odom->header.stamp.toNSec(), ros::Time::now().toNSec(),
            odom->header.frame_id, odom->child_frame_id, raw);
        if (!projected) return;
        pcl::PointCloud<pcl::PointXYZI> body, world;
        pcl::fromROSMsg(*cloud, body);
        pcl::transformPointCloud(body, world, projected->pose.matrix());
        sensor_msgs::PointCloud2 message;
        pcl::toROSMsg(world, message);
        message.header = cloud->header;
        message.header.frame_id = config_.world_frame;
        cloud_world_pub_.publish(message);
    }

    void initializeOptimizationLogging()
    {
        if (!ensureDirectoryExists(config_.map_save_path)) {
            ROS_WARN("Failed to create optimization log directory: %s", config_.map_save_path.c_str());
        }
        optimization_log_path_ = config_.map_save_path + "/optimization.log";
        std::ofstream file(optimization_log_path_, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            ROS_WARN("Failed to open optimization log: %s", optimization_log_path_.c_str());
        }
    }

    static bool ensureDirectoryExists(const std::string& path)
    {
        if (path.empty()) {
            return false;
        }

        std::string current;
        std::size_t start = 0;
        if (path[0] == '/') {
            current = "/";
            start = 1;
        }

        while (start <= path.size()) {
            const std::size_t end = path.find('/', start);
            const std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!part.empty()) {
                if (!current.empty() && current.back() != '/') {
                    current += "/";
                }
                current += part;
                if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        return true;
    }

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose)
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

    void appendOptimizationLogLine(const std::string& line)
    {
        std::lock_guard<std::mutex> lock(optimization_log_mutex_);
        std::ofstream file(optimization_log_path_, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            return;
        }
        file << line << '\n';
    }

    std::vector<Keyframe::Ptr> snapshotKeyframesForGlobalMapLocked(
        KeyframeMapRevision* map_revision = nullptr) const
    {
        std::vector<Keyframe::Ptr> snapshots;
        const auto keyframes = core_ ? core_->getAllKeyframes() : std::vector<Keyframe::Ptr>{};
        if (map_revision && core_) {
            *map_revision = core_->mapRevision();
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
            snapshots.push_back(snapshot);
        }
        return snapshots;
    }

    void resetGlobalMapCache()
    {
        std::lock_guard<std::mutex> lock(global_map_mutex_);
        global_map_cache_.clear();
        global_map_msg_cache_.reset();
        global_map_last_published_revision_.reset();
    }

    void markGlobalMapFullRebuildRequired()
    {
        std::lock_guard<std::mutex> lock(global_map_mutex_);
        global_map_cache_.markFullRebuildRequired();
        global_map_msg_cache_.reset();
        global_map_last_published_revision_.reset();
    }

    bool refreshGlobalMapMessage(const std::vector<Keyframe::Ptr>& keyframes,
                                 const KeyframeMapRevision& map_revision,
                                 sensor_msgs::PointCloud2* out_msg)
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
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*cloud, msg);
            msg.header.frame_id = config_.world_frame;
            msg.header.stamp = ros::Time::now();
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

    void publishGlobalMap()
    {
        std::vector<Keyframe::Ptr> keyframes;
        KeyframeMapRevision map_revision;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            keyframes = snapshotKeyframesForGlobalMapLocked(&map_revision);
        }
        publishGlobalMap(keyframes, map_revision);
    }

    void publishGlobalMap(const std::vector<Keyframe::Ptr>& keyframes,
                          const KeyframeMapRevision& map_revision)
    {
        sensor_msgs::PointCloud2 msg;
        if (!refreshGlobalMapMessage(keyframes, map_revision, &msg)) {
            return;
        }
        global_map_pub_.publish(msg);
    }

    void publishLegacyRelocalizationLock(std::uint64_t lock_epoch)
    {
        std_msgs::UInt32 legacy_msg;
        legacy_msg.data = static_cast<std::uint32_t>(lock_epoch);
        relocalization_lock_pub_.publish(legacy_msg);
    }

    void publishAuthoritativeRelocalizationPose(
        const std_msgs::Header& header, const Eigen::Isometry3d& pose,
        std::uint64_t lock_epoch)
    {
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = config_.world_frame;
        pose_msg.pose = makePose(pose);
        relocalization_pose_pub_.publish(pose_msg);

        const Eigen::Quaterniond q(pose.rotation());
        const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
        ROS_INFO("Relocalization lock #%llu pose=(%.3f, %.3f, %.3f) yaw=%.3f",
                 static_cast<unsigned long long>(lock_epoch),
                 pose.translation().x(),
                 pose.translation().y(),
                 pose.translation().z(),
                 yaw);
    }

    void publishRelocalizationStatus(const std_msgs::Header& header,
                                     const core::BackendOutput& output,
                                     std::uint64_t lock_epoch)
    {
        RelocalizationStatus status;
        status.header = header;
        status.header.frame_id = config_.world_frame;
        status.state = static_cast<std::uint8_t>(output.relocalization_state);
        status.pose_source = static_cast<std::uint8_t>(output.pose_source);
        status.pose.orientation.w = 1.0;
        if (hasUsableGlobalRelocalizationPose(
                output.relocalization_state, output.pose_source)) {
            status.pose = makePose(output.T_world_lidar);
        }
        status.lock_epoch = lock_epoch;
        status.seed_keyframe_id = output.relocalization_seed_keyframe_id;
        status.support_keyframe_id =
            output.relocalization_support_keyframe_id;
        status.decision = output.relocalization_decision;
        relocalization_status_pub_.publish(status);
    }

    void publishLoopMarkers(const std::vector<VerifiedLoop>& loops)
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

    void publishPathMarkers(const std::vector<geometry_msgs::Point>& loaded_points,
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

    void addPathMarker(const std::string& ns,
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

    geometry_msgs::Pose makePose(const Eigen::Isometry3d& pose) const
    {
        geometry_msgs::Pose out;
        out.position = toPoint(pose);
        const Eigen::Quaterniond q(pose.rotation());
        out.orientation.w = q.w();
        out.orientation.x = q.x();
        out.orientation.y = q.y();
        out.orientation.z = q.z();
        return out;
    }

    geometry_msgs::Point toPoint(const Eigen::Isometry3d& pose) const
    {
        geometry_msgs::Point point;
        point.x = pose.translation().x();
        point.y = pose.translation().y();
        point.z = pose.translation().z();
        return point;
    }

    void shutdown()
    {
        if (shutdown_called_) return;
        shutdown_called_ = true;
        if (live_odom_spinner_) live_odom_spinner_->stop();
        live_odom_sub_.shutdown();
        if (coreRunModeSavesMap(run_mode_)) {
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
                ROS_INFO("Map snapshot saved under: %s", config_.map_save_path.c_str());
                if (!warning.empty()) {
                    ROS_WARN("Map snapshot debug artifact warning: %s", warning.c_str());
                }
            } else {
                ROS_ERROR("Failed to save map snapshot: %s", error.c_str());
            }
        }
        ROS_INFO("N3Mapping statistics: frames=%zu keyframes=%zu loops=%zu",
                 frame_count_,
                 keyframe_count_,
                 loop_count_);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    Config config_;
    InputLinearVelocityFrame input_linear_velocity_frame_ = InputLinearVelocityFrame::CHILD;
    std::string output_status_topic_;
    ros::Publisher localization_status_pub_;
    CoreRunMode run_mode_ = CoreRunMode::MAPPING;
    bool product_profile_v1_ = false;
    std::unique_ptr<N3MappingCore> core_;
    std::mutex data_mutex_;
    core::RealtimeOdometry realtime_odometry_;
    std::optional<bool> last_logged_backend_authorized_;
    std::mutex status_mutex_;
    core::BackendOutput last_backend_status_;
    double observation_stamp_ = 0.0;
    ros::Time previous_backend_authorized_stamp_;
    std::optional<std::chrono::steady_clock::time_point>
        previous_backend_authorized_update_;
    std::string last_logged_live_reason_;
    std::optional<std::chrono::steady_clock::time_point> last_live_odom_receive_;
    ros::CallbackQueue live_odom_queue_;
    ros::Subscriber live_odom_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> live_cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> live_paired_odom_sub_;
    std::unique_ptr<ApproxSynchronizer> live_approx_sync_;
    std::unique_ptr<ExactSynchronizer> live_exact_sync_;
    std::unique_ptr<ros::AsyncSpinner> live_odom_spinner_;
    struct SynchronizedFrame {
        sensor_msgs::PointCloud2ConstPtr cloud;
        nav_msgs::OdometryConstPtr odom;
    };
    core::LatestFrameSlot<SynchronizedFrame> pending_localization_frame_;
    std::mutex global_map_mutex_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::unique_ptr<ApproxSynchronizer> approx_sync_;
    std::unique_ptr<ExactSynchronizer> exact_sync_;
    ros::Subscriber dense_odom_sub_;

    ros::Publisher odom_pub_;
    ros::Publisher path_pub_;
    ros::Publisher cloud_body_pub_;
    ros::Publisher cloud_world_pub_;
    ros::Publisher loop_marker_pub_;
    ros::Publisher global_map_pub_;
    ros::Publisher relocalization_lock_pub_;
    ros::Publisher relocalization_status_pub_;
    ros::Publisher relocalization_pose_pub_;
    ros::ServiceServer save_map_srv_;
    ros::Timer loop_timer_;
    ros::Timer global_map_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    GlobalMapCache global_map_cache_;
    std::optional<sensor_msgs::PointCloud2> global_map_msg_cache_;
    std::uint64_t global_map_msg_revision_ = 0;
    std::optional<std::uint64_t> global_map_last_published_revision_;
    std::string optimization_log_path_;
    std::mutex optimization_log_mutex_;

    std::vector<geometry_msgs::PoseStamped> localization_path_;
    ros::Time last_tf_stamp_;
    std::size_t frame_count_ = 0;
    std::size_t keyframe_count_ = 0;
    std::size_t loop_count_ = 0;
    RelocalizationOutputAuthority relocalization_output_authority_;
    bool shutdown_called_ = false;
};

}  // namespace n3mapping

int main(int argc, char** argv)
{
    if (n3mapping::productBuildIdentityRequested(argc, argv)) {
        std::cout << n3mapping::productBuildIdentityJson() << '\n';
        return 0;
    }
    ros::init(argc, argv, "n3mapping_node");
    const char* configured_log_dir = std::getenv("GLOG_log_dir");
    FLAGS_log_dir = configured_log_dir && *configured_log_dir
        ? configured_log_dir : std::string(N3MAPPING_SOURCE_DIR) + "/logs";
    std::filesystem::create_directories(FLAGS_log_dir);
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
    FLAGS_stderrthreshold = google::GLOG_FATAL;
    FLAGS_logbufsecs = 0;
    google::InitGoogleLogging(argv[0]);
    {
        n3mapping::N3MappingNoeticNode node;
        node.spin();
    }
    google::ShutdownGoogleLogging();
    return 0;
}
