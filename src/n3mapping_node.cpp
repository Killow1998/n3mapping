// N3Mapping node: ROS I/O orchestration for mapping, localization, and map extension.
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <optional>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <string_view>
#include <thread>
#include <visualization_msgs/MarkerArray.h>
#include <boost/make_shared.hpp>
#include <glog/logging.h>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/mode_handlers.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {

static std::atomic<bool> g_shutdown_requested{false};

class OptimizationLogSink : public google::LogSink {
public:
    explicit OptimizationLogSink(const std::string& file_path) : file_path_(file_path) { openFile(); }

    void send(google::LogSeverity severity, const char*, const char*, int, const struct ::tm*,
              const char* message, size_t message_len) override {
        std::string_view view(message, message_len);
        if (view.find(tag_) == std::string_view::npos) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_.is_open()) openFile();
        if (!file_.is_open()) return;
        file_.write(message, static_cast<std::streamsize>(message_len));
        file_ << '\n';
        file_.flush();
        if (severity >= google::GLOG_ERROR) file_.flush();
    }

    ~OptimizationLogSink() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.close();
    }

private:
    void openFile() { file_.open(file_path_, std::ios::out | std::ios::trunc); }

    const std::string tag_ = "[OPTIMIZATION]";
    std::string file_path_;
    std::ofstream file_;
    std::mutex mutex_;
};

enum class RunMode { MAPPING, LOCALIZATION, MAP_EXTENSION };

class N3MappingNode {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;

    N3MappingNode(ros::NodeHandle nh, ros::NodeHandle pnh)
        : nh_(std::move(nh)), pnh_(std::move(pnh)) {
        config_.loadFromROS(pnh_);
        config_.print();
        parseRunMode();
        initializeComponents();
        initializeROS();
        initializeOptimizationLogging();

        if (run_mode_ == RunMode::LOCALIZATION || run_mode_ == RunMode::MAP_EXTENSION) {
            loadMap();
            if (map_loaded_) loadAndPublishGlobalMap();
        }

        ROS_INFO("N3Mapping initialized. Mode: %s", config_.mode.c_str());
    }

    ~N3MappingNode() { shutdown(); }

    void shutdown() {
        if (shutdown_called_) return;
        shutdown_called_ = true;

        if (loop_timer_.hasStarted()) loop_timer_.stop();
        if (global_map_timer_.hasStarted()) global_map_timer_.stop();

        if (run_mode_ == RunMode::MAPPING || run_mode_ == RunMode::MAP_EXTENSION) saveMapOnShutdown();

        if (optimization_log_sink_) {
            google::RemoveLogSink(optimization_log_sink_.get());
            optimization_log_sink_.reset();
        }

        printStatistics();
    }

private:
    void parseRunMode() {
        if (config_.mode == "mapping") run_mode_ = RunMode::MAPPING;
        else if (config_.mode == "localization") run_mode_ = RunMode::LOCALIZATION;
        else if (config_.mode == "map_extension") run_mode_ = RunMode::MAP_EXTENSION;
        else {
            ROS_WARN("Unknown mode '%s', use mapping", config_.mode.c_str());
            run_mode_ = RunMode::MAPPING;
        }
    }

    void initializeComponents() {
        keyframe_manager_ = std::make_unique<KeyframeManager>(config_);
        point_cloud_matcher_ = std::make_unique<PointCloudMatcher>(config_);
        loop_detector_ = std::make_unique<LoopDetector>(config_);
        loop_closure_manager_ = std::make_unique<LoopClosureManager>(config_);
        graph_optimizer_ = std::make_unique<GraphOptimizer>(config_);
        map_serializer_ = std::make_unique<MapSerializer>(config_);
        world_localizing_ = std::make_unique<WorldLocalizing>(config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_);
        mapping_resuming_ = std::make_unique<MappingResuming>(
            config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_, *graph_optimizer_, *map_serializer_, *world_localizing_);

        ModePublishCallbacks callbacks{
            [this](const Eigen::Isometry3d& pose, const std_msgs::Header& header) { publishOdometry(pose, header); },
            [this](const std_msgs::Header& header, const Eigen::Isometry3d* pose) { publishPath(header, pose); },
            [this](PointCloud::Ptr cloud, const Eigen::Isometry3d& pose, const std_msgs::Header& header) {
                publishPointClouds(cloud, pose, header);
            },
            [this](const std::string& context, double timestamp, const Eigen::Isometry3d* pose) {
                logOptimizationResult(context, timestamp, pose);
            }
        };

        mapping_mode_handler_ = std::make_unique<MappingModeHandler>(
            config_, *keyframe_manager_, *loop_detector_, *graph_optimizer_,
            loop_queue_mutex_, loop_detection_queue_, callbacks, [this]() { ++keyframe_count_; });

        localization_mode_handler_ = std::make_unique<LocalizationModeHandler>(*world_localizing_, callbacks);

        map_resuming_mode_handler_ = std::make_unique<MapResumingModeHandler>(
            config_, *keyframe_manager_, *graph_optimizer_, *world_localizing_, *mapping_resuming_,
            callbacks, [this]() { ++keyframe_count_; });
    }

    void initializeROS() {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();

        cloud_sub_.subscribe(nh_, config_.cloud_topic, 10);
        odom_sub_.subscribe(nh_, config_.odom_topic, 10);

        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), cloud_sub_, odom_sub_);
        sync_->registerCallback(std::bind(&N3MappingNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        odom_pub_ = nh_.advertise<nav_msgs::Odometry>(config_.output_odom_topic, 10);
        path_pub_ = nh_.advertise<nav_msgs::Path>(config_.output_path_topic, 10);
        cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_body_topic, 10);
        cloud_world_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_world_topic, 10);
        loop_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/n3mapping/loop_closure_markers", 10);
        global_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/n3mapping/global_map", 1, true);

        loop_timer_ = nh_.createTimer(ros::Duration(0.1), &N3MappingNode::loopDetectionTimerCallback, this);
        global_map_timer_ = nh_.createTimer(ros::Duration(10.0), &N3MappingNode::globalMapTimerCallback, this);
    }

    void initializeOptimizationLogging() {
        std::error_code ec;
        std::filesystem::create_directories(config_.map_save_path, ec);
        optimization_log_path_ = config_.map_save_path + "/optimization.log";
        optimization_log_sink_ = std::make_unique<OptimizationLogSink>(optimization_log_path_);
        google::AddLogSink(optimization_log_sink_.get());
    }

    void loadMap() {
        if (config_.map_path.empty()) {
            ROS_ERROR("Map path is empty");
            return;
        }

        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (!mapping_resuming_->loadExistingMap(config_.map_path)) {
                ROS_ERROR("Failed to load map for extension: %s", config_.map_path.c_str());
                return;
            }
            ROS_INFO("Loaded map for extension with %zu keyframes", mapping_resuming_->getOriginalKeyframeCount());
        } else {
            if (!map_serializer_->loadMap(config_.map_path, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                ROS_ERROR("Failed to load map: %s", config_.map_path.c_str());
                return;
            }
            ROS_INFO("Loaded map with %zu keyframes", keyframe_manager_->size());
        }
        map_loaded_ = true;
    }

    void loadAndPublishGlobalMap() {
        std::filesystem::path pbstream_path(config_.map_path);
        std::filesystem::path global_map_path = pbstream_path.parent_path() / "global_map.pcd";

        if (!std::filesystem::exists(global_map_path)) {
            ROS_WARN("Global map PCD not found: %s", global_map_path.c_str());
            return;
        }

        auto global_map = boost::make_shared<PointCloud>();
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(global_map_path.string(), *global_map) == -1) {
            ROS_ERROR("Failed to load global map: %s", global_map_path.c_str());
            return;
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global_map, msg);
        msg.header.frame_id = config_.world_frame;
        msg.header.stamp = ros::Time::now();
        global_map_pub_.publish(msg);
    }

    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                      const nav_msgs::OdometryConstPtr& odom_msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);

        auto cloud = boost::make_shared<PointCloud>();
        pcl::fromROSMsg(*cloud_msg, *cloud);

        Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
        pose_odom.translation() << odom_msg->pose.pose.position.x,
                                   odom_msg->pose.pose.position.y,
                                   odom_msg->pose.pose.position.z;
        Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w,
                             odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y,
                             odom_msg->pose.pose.orientation.z);
        pose_odom.linear() = q.toRotationMatrix();

        double timestamp = cloud_msg->header.stamp.toSec();

        if (run_mode_ == RunMode::MAPPING) {
            mapping_mode_handler_->process(timestamp, pose_odom, cloud, cloud_msg->header);
        } else if (run_mode_ == RunMode::LOCALIZATION) {
            localization_mode_handler_->process(map_loaded_, pose_odom, cloud, cloud_msg->header);
        } else {
            map_resuming_mode_handler_->process(map_loaded_, timestamp, pose_odom, cloud, cloud_msg->header);
        }

        frame_count_++;
    }

    void loopDetectionTimerCallback(const ros::TimerEvent&) {
        if (run_mode_ == RunMode::LOCALIZATION) return;
        if (run_mode_ == RunMode::MAP_EXTENSION && !map_loaded_) return;

        std::vector<int64_t> keyframes_to_check;
        {
            std::lock_guard<std::mutex> lock(loop_queue_mutex_);
            if (loop_detection_queue_.empty()) return;
            keyframes_to_check.swap(loop_detection_queue_);
        }

        for (int64_t query_id : keyframes_to_check) {
            // Throttle: only check every loop_kf_gap keyframes
            if (query_id - last_loop_check_id_ < config_.loop_kf_gap) continue;

            auto query_kf = keyframe_manager_->getKeyframe(query_id);
            if (!query_kf) continue;

            // --- Distance-based candidate search (replaces SC) ---
            // Find keyframes that are spatially close (XY) but temporally far (ID gap)
            struct DistCandidate {
                int64_t match_id;
                double xy_dist;
            };
            std::vector<DistCandidate> candidates;
            {
                auto all_kfs = keyframe_manager_->getAllKeyframes();
                Eigen::Vector3d q_pos = query_kf->pose_optimized.translation();
                int64_t last_candidate_id = -1000;

                for (const auto& kf : all_kfs) {
                    int64_t id_gap = std::abs(query_id - kf->id);
                    if (id_gap < config_.loop_closest_id_th) continue;

                    // Enforce min_id_interval between consecutive candidates
                    if (std::abs(kf->id - last_candidate_id) < config_.loop_min_id_interval) continue;

                    Eigen::Vector3d m_pos = kf->pose_optimized.translation();
                    double dx = q_pos.x() - m_pos.x();
                    double dy = q_pos.y() - m_pos.y();
                    double xy_dist = std::sqrt(dx * dx + dy * dy);

                    if (xy_dist < config_.loop_max_range) {
                        candidates.push_back({kf->id, xy_dist});
                        last_candidate_id = kf->id;
                    }
                }
                // Sort by XY distance, keep top candidates
                std::sort(candidates.begin(), candidates.end(),
                          [](const DistCandidate& a, const DistCandidate& b) { return a.xy_dist < b.xy_dist; });
                const size_t max_candidates = 5;
                if (candidates.size() > max_candidates) candidates.resize(max_candidates);
            }

            if (candidates.empty()) continue;
            last_loop_check_id_ = query_id;  // update throttle counter

            LOG(INFO) << "[LOOP] Distance-based search: query=" << query_id
                      << " found " << candidates.size() << " candidates"
                      << " (closest xy=" << (candidates.empty() ? 0.0 : candidates[0].xy_dist) << ")";

            std::vector<VerifiedLoop> verified_loops;
            verified_loops.reserve(candidates.size());

            for (const auto& candidate : candidates) {
                auto match_kf = keyframe_manager_->getKeyframe(candidate.match_id);
                if (!match_kf || !query_kf->cloud || !match_kf->cloud ||
                    query_kf->cloud->empty() || match_kf->cloud->empty()) {
                    continue;
                }

                auto source = keyframe_manager_->buildSubmapInRootFrame(query_id, 0, candidate.match_id);
                auto target = keyframe_manager_->buildSubmapInRootFrame(candidate.match_id, config_.gicp_submap_size, candidate.match_id);
                if (!source || source->empty() || !target || target->empty()) continue;

                MatchResult match_result = point_cloud_matcher_->alignCloud(
                    target, source, Eigen::Isometry3d::Identity());

                VerifiedLoop loop;
                loop.query_id = query_id;
                loop.match_id = candidate.match_id;
                loop.fitness_score = match_result.fitness_score;
                loop.inlier_ratio = match_result.inlier_ratio;
                loop.information = config_.loop_use_icp_information
                    ? match_result.information
                    : Eigen::Matrix<double, 6, 6>::Identity();

                bool fitness_ok = match_result.fitness_score < config_.loop_fitness_threshold;
                bool inlier_ok = match_result.inlier_ratio >= config_.loop_min_inlier_ratio;

                double icp_translation = match_result.T_target_source.translation().norm();
                double icp_rotation = Eigen::AngleAxisd(match_result.T_target_source.rotation()).angle();
                bool geom_ok = icp_translation <= config_.loop_max_icp_translation &&
                               icp_rotation <= config_.loop_max_icp_rotation;

                loop.verified = match_result.converged && fitness_ok && inlier_ok && geom_ok;

                if (loop.verified) {
                    loop.T_match_query = match_result.T_target_source.inverse();
                    LOG(INFO) << "[LOOP] VERIFIED query=" << loop.query_id
                              << " match=" << loop.match_id
                              << " xy_dist=" << candidate.xy_dist
                              << " fitness=" << loop.fitness_score
                              << " inlier=" << loop.inlier_ratio
                              << " icp_t=" << icp_translation
                              << " icp_r=" << icp_rotation;
                } else {
                    LOG(INFO) << "[LOOP] REJECTED query=" << query_id
                            << " match=" << candidate.match_id
                            << " xy_dist=" << candidate.xy_dist
                            << " fitness=" << match_result.fitness_score
                            << " inlier=" << match_result.inlier_ratio
                            << " icp_t=" << icp_translation
                            << " icp_r=" << icp_rotation
                            << " (ok: conv=" << match_result.converged
                            << " fit=" << fitness_ok
                            << " inl=" << inlier_ok
                            << " geom=" << geom_ok << ")";
                }

                verified_loops.push_back(loop);
            }

            if (verified_loops.empty()) continue;

            auto valid_loops = loop_closure_manager_->filterValidLoops(verified_loops);
            auto best_loops = loop_closure_manager_->selectBestPerQuery(valid_loops);
            if (best_loops.empty()) continue;

            publishLoopMarkers(best_loops);

            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                auto edges = loop_closure_manager_->buildLoopEdges(best_loops, LoopEdgeDirection::MatchToQuery);
                if (edges.empty()) continue;

                loop_closure_manager_->applyEdges(edges, *graph_optimizer_);
                loop_count_ += edges.size();

                auto poses = graph_optimizer_->getOptimizedPoses();
                keyframe_manager_->updateOptimizedPoses(poses);
                logOptimizationResult("loop_closure", ros::Time::now().toSec(), nullptr);

                for (const auto& loop : best_loops) {
                    ROS_INFO("Loop accepted: %ld -> %ld (fitness=%.3f, inlier=%.3f)",
                             loop.match_id, loop.query_id, loop.fitness_score, loop.inlier_ratio);
                }
            }
        }
    }

    void globalMapTimerCallback(const ros::TimerEvent&) {
        if (keyframe_count_ < 2) return;
        if (global_map_pub_.getNumSubscribers() == 0) return;

        std::lock_guard<std::mutex> lock(data_mutex_);

        auto keyframes = keyframe_manager_->getAllKeyframes();
        if (keyframes.empty()) return;

        auto global_map = boost::make_shared<PointCloud>();
        int skip = std::max(1, static_cast<int>(keyframes.size()) / 500);  // sparse vis for large maps
        for (size_t i = 0; i < keyframes.size(); i += skip) {
            const auto& kf = keyframes[i];
            if (!kf || !kf->cloud || kf->cloud->empty()) continue;
            PointCloud transformed;
            pcl::transformPointCloud(*kf->cloud, transformed, kf->pose_optimized.matrix().cast<float>());
            *global_map += transformed;
        }

        if (config_.global_map_voxel_size > 0.0 && !global_map->empty()) {
            auto ds = boost::make_shared<PointCloud>();
            pcl::VoxelGrid<pcl::PointXYZI> vf;
            vf.setInputCloud(global_map);
            vf.setLeafSize(config_.global_map_voxel_size, config_.global_map_voxel_size, config_.global_map_voxel_size);
            vf.filter(*ds);
            global_map = ds;
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global_map, msg);
        msg.header.frame_id = config_.world_frame;
        msg.header.stamp = ros::Time::now();
        global_map_pub_.publish(msg);
    }

    void publishOdometry(const Eigen::Isometry3d& pose, const std_msgs::Header& header) {
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = config_.world_frame;
        odom_msg.child_frame_id = config_.body_frame;

        odom_msg.pose.pose.position.x = pose.translation().x();
        odom_msg.pose.pose.position.y = pose.translation().y();
        odom_msg.pose.pose.position.z = pose.translation().z();

        Eigen::Quaterniond q(pose.rotation());
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();

        odom_pub_.publish(odom_msg);

        geometry_msgs::TransformStamped tf;
        tf.header = odom_msg.header;
        tf.child_frame_id = config_.body_frame;
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        tf.transform.rotation = odom_msg.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

    void publishPath(const std_msgs::Header& header, const Eigen::Isometry3d* current_pose) {
        nav_msgs::Path path_msg;
        path_msg.header = header;
        path_msg.header.frame_id = config_.world_frame;

        if (run_mode_ == RunMode::LOCALIZATION) {
            if (current_pose) {
                geometry_msgs::PoseStamped pose;
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
                localization_path_.push_back(pose);
            }
            path_msg.poses = localization_path_;
        } else {
            for (const auto& kf : keyframe_manager_->getAllKeyframes()) {
                if (!kf) continue;
                geometry_msgs::PoseStamped pose;
                pose.header = path_msg.header;
                pose.pose.position.x = kf->pose_optimized.translation().x();
                pose.pose.position.y = kf->pose_optimized.translation().y();
                pose.pose.position.z = kf->pose_optimized.translation().z();
                Eigen::Quaterniond q(kf->pose_optimized.rotation());
                pose.pose.orientation.w = q.w();
                pose.pose.orientation.x = q.x();
                pose.pose.orientation.y = q.y();
                pose.pose.orientation.z = q.z();
                path_msg.poses.push_back(pose);
            }
            if (current_pose) {
                geometry_msgs::PoseStamped pose;
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
            }
        }

        path_pub_.publish(path_msg);
    }

    void publishLoopMarkers(const std::vector<VerifiedLoop>& new_loops) {
        // Accumulate new loops into history
        for (const auto& loop : new_loops) {
            accepted_loops_.push_back(loop);
        }

        visualization_msgs::MarkerArray markers;
        // Clear old markers first
        visualization_msgs::Marker clear_marker;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        markers.markers.push_back(clear_marker);

        // Color palette — visually distinct colors cycling through hue
        auto indexToColor = [](int idx) -> std_msgs::ColorRGBA {
            // Golden-angle based hue rotation for maximum distinction
            float hue = std::fmod(idx * 137.508f, 360.0f);  // golden angle ≈ 137.508°
            float s = 0.9f, v = 0.95f;
            // HSV to RGB
            float c = v * s;
            float h2 = hue / 60.0f;
            float x = c * (1.0f - std::fabs(std::fmod(h2, 2.0f) - 1.0f));
            float r1 = 0, g1 = 0, b1 = 0;
            if      (h2 < 1) { r1 = c; g1 = x; }
            else if (h2 < 2) { r1 = x; g1 = c; }
            else if (h2 < 3) { g1 = c; b1 = x; }
            else if (h2 < 4) { g1 = x; b1 = c; }
            else if (h2 < 5) { r1 = x; b1 = c; }
            else              { r1 = c; b1 = x; }
            float m = v - c;
            std_msgs::ColorRGBA color;
            color.r = r1 + m; color.g = g1 + m; color.b = b1 + m; color.a = 0.95f;
            return color;
        };

        // Publish ALL historical loops with distinct colors
        int marker_id = 0;
        for (size_t i = 0; i < accepted_loops_.size(); ++i) {
            const auto& loop = accepted_loops_[i];
            auto match_kf = keyframe_manager_->getKeyframe(loop.match_id);
            auto query_kf = keyframe_manager_->getKeyframe(loop.query_id);
            if (!match_kf || !query_kf) continue;

            auto color = indexToColor(static_cast<int>(i));

            geometry_msgs::Point p_match;
            p_match.x = match_kf->pose_optimized.translation().x();
            p_match.y = match_kf->pose_optimized.translation().y();
            p_match.z = match_kf->pose_optimized.translation().z();

            geometry_msgs::Point p_query;
            p_query.x = query_kf->pose_optimized.translation().x();
            p_query.y = query_kf->pose_optimized.translation().y();
            p_query.z = query_kf->pose_optimized.translation().z();

            // Line connecting match <-> query
            visualization_msgs::Marker line;
            line.header.frame_id = config_.world_frame;
            line.header.stamp = ros::Time::now();
            line.ns = "loop_lines";
            line.id = marker_id++;
            line.type = visualization_msgs::Marker::LINE_LIST;
            line.action = visualization_msgs::Marker::ADD;
            line.scale.x = 0.15;
            line.color = color;
            line.lifetime = ros::Duration(0);
            line.points.push_back(p_match);
            line.points.push_back(p_query);
            markers.markers.push_back(line);

            // Sphere at match end
            visualization_msgs::Marker sphere_m;
            sphere_m.header = line.header;
            sphere_m.ns = "loop_match";
            sphere_m.id = marker_id++;
            sphere_m.type = visualization_msgs::Marker::SPHERE;
            sphere_m.action = visualization_msgs::Marker::ADD;
            sphere_m.pose.position = p_match;
            sphere_m.pose.orientation.w = 1.0;
            sphere_m.scale.x = sphere_m.scale.y = sphere_m.scale.z = 0.4;
            sphere_m.color = color;
            sphere_m.lifetime = ros::Duration(0);
            markers.markers.push_back(sphere_m);

            // Sphere at query end
            visualization_msgs::Marker sphere_q;
            sphere_q.header = line.header;
            sphere_q.ns = "loop_query";
            sphere_q.id = marker_id++;
            sphere_q.type = visualization_msgs::Marker::SPHERE;
            sphere_q.action = visualization_msgs::Marker::ADD;
            sphere_q.pose.position = p_query;
            sphere_q.pose.orientation.w = 1.0;
            sphere_q.scale.x = sphere_q.scale.y = sphere_q.scale.z = 0.3;
            sphere_q.color = color;
            sphere_q.color.a = 0.7f;  // slightly transparent for query
            sphere_q.lifetime = ros::Duration(0);
            markers.markers.push_back(sphere_q);
        }

        loop_marker_pub_.publish(markers);
    }

    void publishPointClouds(PointCloud::Ptr cloud, const Eigen::Isometry3d& pose, const std_msgs::Header& header) {
        sensor_msgs::PointCloud2 cloud_body_msg;
        pcl::toROSMsg(*cloud, cloud_body_msg);
        cloud_body_msg.header = header;
        cloud_body_msg.header.frame_id = config_.body_frame;
        cloud_body_pub_.publish(cloud_body_msg);

        auto cloud_world = boost::make_shared<PointCloud>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_world, transform);

        sensor_msgs::PointCloud2 cloud_world_msg;
        pcl::toROSMsg(*cloud_world, cloud_world_msg);
        cloud_world_msg.header = header;
        cloud_world_msg.header.frame_id = config_.world_frame;
        cloud_world_pub_.publish(cloud_world_msg);
    }

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose) {
        auto keyframes = keyframe_manager_->getAllKeyframes();
        std::vector<Keyframe::Ptr> valid;
        valid.reserve(keyframes.size());
        for (const auto& kf : keyframes) {
            if (kf) valid.push_back(kf);
        }
        std::sort(valid.begin(), valid.end(),
                  [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) { return a->id < b->id; });

        std::optional<int64_t> latest_id;
        if (!valid.empty()) latest_id = valid.back()->id;

        LOG(INFO) << "[OPTIMIZATION] context=" << context
                  << " time=" << timestamp
                  << " keyframes=" << valid.size()
                  << " latest_id=" << (latest_id ? std::to_string(*latest_id) : std::string("none"));

        if (current_pose) {
            Eigen::Quaterniond q(current_pose->rotation());
            LOG(INFO) << "[OPTIMIZATION] current t="
                      << current_pose->translation().x() << ","
                      << current_pose->translation().y() << ","
                      << current_pose->translation().z() << " q="
                      << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
        }
    }

    void saveMapOnShutdown() {
        std::lock_guard<std::mutex> lock(data_mutex_);

        std::string map_file = config_.map_save_path + "/n3map.pbstream";
        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (!mapping_resuming_->saveExtendedMap(map_file)) {
                ROS_ERROR("Failed to save extended map");
            }
        } else {
            if (!map_serializer_->saveMap(map_file, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                ROS_ERROR("Failed to save map");
            }
        }

        if (config_.save_global_map_on_shutdown) {
            std::string global_map_file = config_.map_save_path + "/global_map.pcd";
            map_serializer_->saveGlobalMap(global_map_file, *keyframe_manager_);
        }
    }

    void printStatistics() {
        ROS_INFO("========== N3Mapping Statistics ==========");
        ROS_INFO("Frames processed: %zu", frame_count_);
        ROS_INFO("Keyframes: %zu", keyframe_count_);
        ROS_INFO("Loop closures: %zu", loop_count_);
        ROS_INFO("==========================================");
    }

private:
    Config config_;
    RunMode run_mode_ = RunMode::MAPPING;

    std::unique_ptr<KeyframeManager> keyframe_manager_;
    std::unique_ptr<PointCloudMatcher> point_cloud_matcher_;
    std::unique_ptr<LoopDetector> loop_detector_;
    std::unique_ptr<LoopClosureManager> loop_closure_manager_;
    std::unique_ptr<GraphOptimizer> graph_optimizer_;
    std::unique_ptr<MapSerializer> map_serializer_;
    std::unique_ptr<WorldLocalizing> world_localizing_;
    std::unique_ptr<MappingResuming> mapping_resuming_;
    std::unique_ptr<MappingModeHandler> mapping_mode_handler_;
    std::unique_ptr<LocalizationModeHandler> localization_mode_handler_;
    std::unique_ptr<MapResumingModeHandler> map_resuming_mode_handler_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    ros::Publisher odom_pub_;
    ros::Publisher path_pub_;
    ros::Publisher cloud_body_pub_;
    ros::Publisher cloud_world_pub_;
    ros::Publisher loop_marker_pub_;
    ros::Publisher global_map_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    ros::Timer loop_timer_;
    ros::Timer global_map_timer_;

    std::mutex data_mutex_;
    std::mutex loop_queue_mutex_;
    std::vector<int64_t> loop_detection_queue_;

    bool map_loaded_ = false;
    bool shutdown_called_ = false;
    std::unique_ptr<OptimizationLogSink> optimization_log_sink_;
    std::string optimization_log_path_;
    std::vector<geometry_msgs::PoseStamped> localization_path_;

    size_t frame_count_ = 0;
    size_t keyframe_count_ = 0;
    size_t loop_count_ = 0;
    int64_t last_loop_check_id_ = -1000;  // throttle: last query ID that triggered loop check
    std::vector<VerifiedLoop> accepted_loops_;  // all accepted loops for visualization

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
};

} // namespace n3mapping

void signalHandler(int) {
    n3mapping::g_shutdown_requested = true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "n3mapping_node");

    // glog: 所有输出通过 RuntimeLogSink → runtime.log，不生成任何默认日志文件
    FLAGS_logtostderr = true;           // 让 glog 以为输出到 stderr
    FLAGS_stderrthreshold = 4;          // 但阈值设为 FATAL+1，实际不输出任何东西到 stderr
    FLAGS_alsologtostderr = false;
    FLAGS_colorlogtostderr = false;
    FLAGS_minloglevel = 0;
    FLAGS_logbufsecs = 0;
    google::InitGoogleLogging(argv[0]);
#ifdef N3MAPPING_SOURCE_DIR
    std::string log_dir = std::string(N3MAPPING_SOURCE_DIR) + "/map";
#else
    std::string log_dir = "./map";
#endif
    std::filesystem::create_directories(log_dir);
    std::string runtime_log = log_dir + "/runtime.log";
    { std::ofstream(runtime_log, std::ios::trunc); }  // 清空旧日志
    // Use a custom log sink to write everything to runtime.log
    static std::ofstream runtime_log_stream(runtime_log, std::ios::out | std::ios::app);
    class RuntimeLogSink : public google::LogSink {
    public:
        explicit RuntimeLogSink(std::ofstream& os) : os_(os) {}
        void send(google::LogSeverity severity, const char* full_filename,
                  const char* base_filename, int line, const struct ::tm* tm_time,
                  const char* message, size_t message_len) override {
            (void)full_filename; (void)base_filename; (void)line; (void)tm_time;
            const char* sev = (severity == google::GLOG_INFO) ? "I" :
                              (severity == google::GLOG_WARNING) ? "W" :
                              (severity == google::GLOG_ERROR) ? "E" : "F";
            std::lock_guard<std::mutex> lock(mu_);
            os_ << sev << " ";
            os_.write(message, static_cast<std::streamsize>(message_len));
            os_ << "\n";
            os_.flush();
        }
    private:
        std::ofstream& os_;
        std::mutex mu_;
    };
    static RuntimeLogSink runtime_sink(runtime_log_stream);
    google::AddLogSink(&runtime_sink);

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    n3mapping::N3MappingNode node(nh, pnh);

    ros::AsyncSpinner spinner(2);
    spinner.start();

    while (ros::ok() && !n3mapping::g_shutdown_requested) {
        ros::Duration(0.001).sleep();
    }

    node.shutdown();
    ros::shutdown();
    google::ShutdownGoogleLogging();
    return 0;
}
