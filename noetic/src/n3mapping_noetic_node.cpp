#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt32.h>
#include <std_srvs/Trigger.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping_noetic/config_noetic.h"
#include "n3mapping_noetic/conversions.h"

namespace n3mapping {

class N3MappingNoeticNode {
  public:
    N3MappingNoeticNode()
      : nh_()
      , private_nh_("~")
    {
        loadConfigFromNoetic(private_nh_, &config_);
        global_map_cache_.setVoxelSize(config_.global_map_voxel_size);
        ROS_INFO("%s", config_.toString().c_str());

        run_mode_ = parseCoreRunMode(config_.mode);
        core_ = std::make_unique<N3MappingCore>(config_);
        core_->setExternalDenseTrajectoryRecordingEnabled(true);
        initializeOptimizationLogging();

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

  private:
    using SyncPolicy =
        message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void initializeRosInterfaces()
    {
        const int sync_queue_size = std::max(1, config_.sync_queue_size);
        cloud_sub_.subscribe(nh_, config_.cloud_topic, static_cast<uint32_t>(sync_queue_size));
        odom_sub_.subscribe(nh_, config_.odom_topic, static_cast<uint32_t>(sync_queue_size));
        SyncPolicy sync_policy(static_cast<uint32_t>(sync_queue_size));
        sync_policy.setMaxIntervalDuration(ros::Duration(config_.sync_time_tolerance));
        sync_ = std::make_unique<Synchronizer>(
            static_cast<const SyncPolicy&>(sync_policy), cloud_sub_, odom_sub_);
        sync_->registerCallback(boost::bind(&N3MappingNoeticNode::syncCallback, this, _1, _2));
        dense_odom_sub_ = nh_.subscribe(
            config_.odom_topic, static_cast<uint32_t>(sync_queue_size), &N3MappingNoeticNode::denseOdomCallback, this);

        odom_pub_ = nh_.advertise<nav_msgs::Odometry>(config_.output_odom_topic, 10);
        path_pub_ = nh_.advertise<nav_msgs::Path>(config_.output_path_topic, 10);
        cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_body_topic, 10);
        cloud_world_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_world_topic, 10);
        loop_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/n3mapping/loop_closure_markers", 10, true);
        global_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/n3mapping/global_map", 1, true);
        relocalization_lock_pub_ = nh_.advertise<std_msgs::UInt32>("/n3mapping/relocalization_lock", 10);
        save_map_srv_ = nh_.advertiseService("/n3mapping/save_map", &N3MappingNoeticNode::handleSaveMap, this);

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
        core::BackendOutput output_for_clouds;
        std_msgs::Header cloud_header;
        bool publish_clouds = false;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            const auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);

            const auto output = core_->processFrame(run_mode_, frame);

            if (run_mode_ == CoreRunMode::MAP_EXTENSION && output.relocalization_locked && !output.accepted_keyframe) {
                ROS_INFO_THROTTLE(2.0, "Initial relocalization successful for map extension");
            }
            if (!output.success && !output.accepted_keyframe && run_mode_ != CoreRunMode::LOCALIZATION) {
                ++frame_count_;
                return;
            }

            publishOdometry(output.T_world_lidar, cloud_msg->header, *odom_msg);
            publishPath(cloud_msg->header, &output.T_world_lidar);

            if (output.relocalization_locked) {
                publishRelocalizationLock(cloud_msg->header, output.T_world_lidar);
            }
            if (output.accepted_keyframe) {
                ++keyframe_count_;
                if (run_mode_ == CoreRunMode::MAPPING || run_mode_ == CoreRunMode::MAP_EXTENSION) {
                    const char* context = run_mode_ == CoreRunMode::MAP_EXTENSION ? "map_extension_incremental" : "mapping_incremental";
                    logOptimizationResult(context, cloud_msg->header.stamp.toSec(), &output.T_world_lidar);
                }
            }
            ++frame_count_;
            output_for_clouds = output;
            cloud_header = cloud_msg->header;
            publish_clouds = true;
        }

        if (publish_clouds) {
            publishPointClouds(output_for_clouds, cloud_header);
        }
    }

    void denseOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        if (!core_ || !coreRunModeSavesMap(run_mode_)) {
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
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            keyframes = snapshotKeyframesForGlobalMapLocked();
        }
        publishGlobalMap(keyframes);
    }

    bool handleSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        std::string error;
        std::string warning;
        std::vector<Keyframe::Ptr> global_map_keyframes;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (!coreRunModeSavesMap(run_mode_)) {
                res.success = false;
                res.message = std::string("save_disabled_in_mode:") + coreRunModeName(run_mode_);
                ROS_WARN("save_map service rejected: %s", res.message.c_str());
                return true;
            }
            res.success = saveMapPbstreamAndSnapshotLocked(&error, &global_map_keyframes);
        }
        if (res.success && config_.save_global_map_on_shutdown) {
            saveDebugGlobalMap(global_map_keyframes, &warning);
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
                                          std::vector<Keyframe::Ptr>* global_map_keyframes)
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
            *global_map_keyframes = snapshotKeyframesForGlobalMapLocked();
        }
        return true;
    }

    void saveDebugGlobalMap(const std::vector<Keyframe::Ptr>& keyframes, std::string* warning)
    {
        if (warning) {
            warning->clear();
        }
        if (keyframes.empty()) {
            if (warning) *warning = "save_global_map_empty";
            return;
        }
        std::lock_guard<std::mutex> global_map_lock(global_map_mutex_);
        const auto cloud = global_map_cache_.update(keyframes);
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

    void publishOdometry(const Eigen::Isometry3d& pose,
                         const std_msgs::Header& header,
                         const nav_msgs::Odometry& input_odom)
    {
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = config_.world_frame;
        odom_msg.child_frame_id = config_.body_frame;
        odom_msg.pose.pose = makePose(pose);
        odom_msg.twist = transformTwistToOutputFrame(pose, input_odom);
        odom_pub_.publish(odom_msg);

        ros::Time tf_stamp = odom_msg.header.stamp;
        if (tf_stamp.isZero()) {
            tf_stamp = ros::Time::now();
        }
        if (!last_tf_stamp_.isZero() && tf_stamp <= last_tf_stamp_) {
            ROS_WARN_THROTTLE(2.0,
                              "Skip non-increasing TF stamp map->%s: current=%.6f last=%.6f",
                              config_.body_frame.c_str(),
                              tf_stamp.toSec(),
                              last_tf_stamp_.toSec());
            return;
        }

        geometry_msgs::TransformStamped tf;
        tf.header = odom_msg.header;
        tf.header.stamp = tf_stamp;
        tf.child_frame_id = config_.body_frame;
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        tf.transform.rotation = odom_msg.pose.pose.orientation;
        tf_broadcaster_.sendTransform(tf);
        last_tf_stamp_ = tf_stamp;
    }

    geometry_msgs::TwistWithCovariance transformTwistToOutputFrame(
        const Eigen::Isometry3d& output_pose,
        const nav_msgs::Odometry& input_odom) const
    {
        geometry_msgs::TwistWithCovariance twist = input_odom.twist;
        const Eigen::Isometry3d input_pose = odometryPoseToIsometry(input_odom);
        const Eigen::Matrix3d rotation = output_pose.linear() * input_pose.linear().transpose();

        const Eigen::Vector3d linear(twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z);
        const Eigen::Vector3d angular(twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z);
        const Eigen::Vector3d linear_out = rotation * linear;
        const Eigen::Vector3d angular_out = rotation * angular;
        twist.twist.linear.x = linear_out.x();
        twist.twist.linear.y = linear_out.y();
        twist.twist.linear.z = linear_out.z();
        twist.twist.angular.x = angular_out.x();
        twist.twist.angular.y = angular_out.y();
        twist.twist.angular.z = angular_out.z();

        Eigen::Matrix<double, 6, 6> covariance;
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                covariance(row, col) = twist.covariance[row * 6 + col];
            }
        }
        Eigen::Matrix<double, 6, 6> transform = Eigen::Matrix<double, 6, 6>::Zero();
        transform.block<3, 3>(0, 0) = rotation;
        transform.block<3, 3>(3, 3) = rotation;
        const Eigen::Matrix<double, 6, 6> covariance_out = transform * covariance * transform.transpose();
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                twist.covariance[row * 6 + col] = covariance_out(row, col);
            }
        }

        return twist;
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

    void publishPointClouds(const core::BackendOutput& output, const std_msgs::Header& header)
    {
        if (output.cloud_body && !output.cloud_body->empty()) {
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*output.cloud_body, msg);
            msg.header = header;
            msg.header.frame_id = config_.body_frame;
            cloud_body_pub_.publish(msg);
        }

        if (output.cloud_world && !output.cloud_world->empty()) {
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*output.cloud_world, msg);
            msg.header = header;
            msg.header.frame_id = config_.world_frame;
            cloud_world_pub_.publish(msg);
        }
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

    std::vector<Keyframe::Ptr> snapshotKeyframesForGlobalMapLocked() const
    {
        std::vector<Keyframe::Ptr> snapshots;
        const auto keyframes = core_ ? core_->getAllKeyframes() : std::vector<Keyframe::Ptr>{};
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
                                 sensor_msgs::PointCloud2* out_msg)
    {
        if (!out_msg) {
            return false;
        }

        std::lock_guard<std::mutex> lock(global_map_mutex_);
        const auto cloud = global_map_cache_.update(keyframes);
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
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            keyframes = snapshotKeyframesForGlobalMapLocked();
        }
        publishGlobalMap(keyframes);
    }

    void publishGlobalMap(const std::vector<Keyframe::Ptr>& keyframes)
    {
        sensor_msgs::PointCloud2 msg;
        if (!refreshGlobalMapMessage(keyframes, &msg)) {
            return;
        }
        global_map_pub_.publish(msg);
    }

    void publishRelocalizationLock(const std_msgs::Header&, const Eigen::Isometry3d& pose)
    {
        std_msgs::UInt32 msg;
        msg.data = ++relocalization_lock_count_;
        relocalization_lock_pub_.publish(msg);

        const Eigen::Quaterniond q(pose.rotation());
        const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
        ROS_INFO("Relocalization lock #%u pose=(%.3f, %.3f, %.3f) yaw=%.3f",
                 msg.data,
                 pose.translation().x(),
                 pose.translation().y(),
                 pose.translation().z(),
                 yaw);
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
        if (coreRunModeSavesMap(run_mode_)) {
            std::string error;
            std::string warning;
            std::vector<Keyframe::Ptr> global_map_keyframes;
            bool saved = false;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                saved = saveMapPbstreamAndSnapshotLocked(&error, &global_map_keyframes);
            }
            if (saved && config_.save_global_map_on_shutdown) {
                saveDebugGlobalMap(global_map_keyframes, &warning);
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
    CoreRunMode run_mode_ = CoreRunMode::MAPPING;
    std::unique_ptr<N3MappingCore> core_;
    std::mutex data_mutex_;
    std::mutex global_map_mutex_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::unique_ptr<Synchronizer> sync_;
    ros::Subscriber dense_odom_sub_;

    ros::Publisher odom_pub_;
    ros::Publisher path_pub_;
    ros::Publisher cloud_body_pub_;
    ros::Publisher cloud_world_pub_;
    ros::Publisher loop_marker_pub_;
    ros::Publisher global_map_pub_;
    ros::Publisher relocalization_lock_pub_;
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
    uint32_t relocalization_lock_count_ = 0;
    bool shutdown_called_ = false;
};

}  // namespace n3mapping

int main(int argc, char** argv)
{
    ros::init(argc, argv, "n3mapping_node");
    n3mapping::N3MappingNoeticNode node;
    ros::spin();
    return 0;
}
