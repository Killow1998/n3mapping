/**
 * @file n3mapping_node.cpp
 * @brief N3Mapping 主节点实现
 *
 * 后端 SLAM 优化节点，接收 FAST-LIO2 前端数据，
 * 通过图优化构建全局一致性地图。
 *
 * Requirements: 1.1, 1.2, 1.3, 5.4, 5.5, 6.1, 6.2, 6.3, 7.1, 7.8, 9.6, 10.2, 10.3,
 * 10.4, 11.1
 */

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <unordered_map>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <limits>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <optional>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <string_view>
#include <boost/make_shared.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <thread>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Header.h>

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

// 全局节点指针用于信号处理
static std::atomic<bool> g_shutdown_requested{ false };

class OptimizationLogSink : public google::LogSink
{
  public:
    explicit OptimizationLogSink(const std::string& file_path)
      : file_path_(file_path)
    {
        openFile();
    }

    void send(google::LogSeverity severity, const char*, const char*, int, const struct ::tm*, const char* message, size_t message_len) override
    {
        std::string_view view(message, message_len);
        if (view.find(tag_) == std::string_view::npos) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_.is_open()) {
            openFile();
        }
        if (!file_.is_open()) {
            return;
        }
        file_.write(message, static_cast<std::streamsize>(message_len));
        file_ << '\n';
        file_.flush();
        if (severity >= google::GLOG_ERROR) {
            file_.flush();
        }
    }

    ~OptimizationLogSink() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
    }

  private:
    void openFile() { file_.open(file_path_, std::ios::out | std::ios::trunc); }

    const std::string tag_ = "[OPTIMIZATION]";
    std::string file_path_;
    std::ofstream file_;
    std::mutex mutex_;
};

/**
 * @brief 运行模式枚举
 */
enum class RunMode
{
    MAPPING,      // 建图模式
    LOCALIZATION, // 重定位模式
    MAP_EXTENSION // 地图续建模式
};

/**
 * @brief N3Mapping 主节点
 */
class N3MappingNode
{
  public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;

        N3MappingNode(ros::NodeHandle nh, ros::NodeHandle pnh)
            : nh_(std::move(nh))
            , pnh_(std::move(pnh))
    {
        // 加载配置
                config_.loadFromROS(pnh_);
                config_.print();

        // 解析运行模式
        parseRunMode();

        // 初始化组件
        initializeComponents();

        // 初始化 ROS 接口
        initializeROS();

        initializeOptimizationLogging();

        // 根据模式加载地图
        if (run_mode_ == RunMode::LOCALIZATION || run_mode_ == RunMode::MAP_EXTENSION) {
            loadMap();
            // 定位模式下加载并发布全局地图用于可视化
            if (map_loaded_) {
                loadAndPublishGlobalMap();
            }
        }

        ROS_INFO("N3Mapping node initialized. Mode: %s", config_.mode.c_str());
    }

    ~N3MappingNode() { shutdown(); }

    /**
     * @brief 关闭节点，保存地图
     */
    void shutdown()
    {
        if (shutdown_called_) return;
        shutdown_called_ = true;

        ROS_INFO("Shutting down N3Mapping node...");

        if (loop_timer_.hasStarted()) {
            loop_timer_.stop();
        }

        // 保存地图
        if (run_mode_ == RunMode::MAPPING || run_mode_ == RunMode::MAP_EXTENSION) {
            saveMapOnShutdown();
        }

        if (optimization_log_sink_) {
            google::RemoveLogSink(optimization_log_sink_.get());
            optimization_log_sink_.reset();
        }

        // 输出统计信息
        printStatistics();
    }

  private:
    /**
     * @brief 解析运行模式
     */
    void parseRunMode()
    {
        if (config_.mode == "mapping") {
            run_mode_ = RunMode::MAPPING;
        } else if (config_.mode == "localization") {
            run_mode_ = RunMode::LOCALIZATION;
        } else if (config_.mode == "map_extension") {
            run_mode_ = RunMode::MAP_EXTENSION;
        } else {
            ROS_WARN("Unknown mode '%s', defaulting to mapping", config_.mode.c_str());
            run_mode_ = RunMode::MAPPING;
        }
    }

    /**
     * @brief 初始化组件
     */
    void initializeComponents()
    {
        keyframe_manager_ = std::make_unique<KeyframeManager>(config_);
        point_cloud_matcher_ = std::make_unique<PointCloudMatcher>(config_);
        loop_detector_ = std::make_unique<LoopDetector>(config_);
        loop_closure_manager_ = std::make_unique<LoopClosureManager>(config_);
        graph_optimizer_ = std::make_unique<GraphOptimizer>(config_);
        map_serializer_ = std::make_unique<MapSerializer>(config_);
        world_localizing_ = std::make_unique<WorldLocalizing>(config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_);
        mapping_resuming_ = std::make_unique<MappingResuming>(
          config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_, *graph_optimizer_, *map_serializer_, *world_localizing_);

        ModePublishCallbacks publish_callbacks{
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
          config_,
          *keyframe_manager_,
          *loop_detector_,
          *graph_optimizer_,
          loop_queue_mutex_,
          loop_detection_queue_,
          publish_callbacks,
                    [this]() { ++keyframe_count_; });

        localization_mode_handler_ = std::make_unique<LocalizationModeHandler>(*world_localizing_, publish_callbacks);

                map_resuming_mode_handler_ = std::make_unique<MapResumingModeHandler>(
          config_,
          *keyframe_manager_,
          *graph_optimizer_,
          *world_localizing_,
          *mapping_resuming_,
          publish_callbacks,
                    [this]() { ++keyframe_count_; });
    }

    /**
     * @brief 初始化 ROS 接口
     */
    void initializeROS()
    {
        // TF 广播器
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();

        // 订阅者
        cloud_sub_.subscribe(nh_, config_.cloud_topic, 10);
        odom_sub_.subscribe(nh_, config_.odom_topic, 10);

        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), cloud_sub_, odom_sub_);
        sync_->registerCallback(std::bind(&N3MappingNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // 发布者
                odom_pub_ = nh_.advertise<nav_msgs::Odometry>(config_.output_odom_topic, 10);
                path_pub_ = nh_.advertise<nav_msgs::Path>(config_.output_path_topic, 10);
                cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_body_topic, 10);
                cloud_world_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_world_topic, 10);
                loop_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/n3mapping/loop_closure_markers", 10);

                // 全局地图发布者 (latched topic)
                global_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/n3mapping/global_map", 1, true);

                loop_timer_ = nh_.createTimer(ros::Duration(0.1), &N3MappingNode::loopDetectionTimerCallback, this);
    }

    void initializeOptimizationLogging()
    {
        std::error_code ec;
        std::filesystem::create_directories(config_.map_save_path, ec);
        optimization_log_path_ = config_.map_save_path + "/optimization.log";
        optimization_log_sink_ = std::make_unique<OptimizationLogSink>(optimization_log_path_);
        google::AddLogSink(optimization_log_sink_.get());
    }

    /**
     * @brief 构建以 center_id 为中心的局部子图（点云变换到 center 坐标系）
     */
    PointCloud::Ptr buildLocalSubmapInCenterFrame(int64_t center_id, int range)
    {
        auto center_kf = keyframe_manager_->getKeyframe(center_id);
        if (!center_kf || !center_kf->cloud || center_kf->cloud->empty()) {
            return boost::make_shared<PointCloud>();
        }

        Eigen::Matrix4f T_center_inv = center_kf->pose_optimized.matrix().cast<float>().inverse();
        auto submap = boost::make_shared<PointCloud>();

        for (int64_t id = center_id - range; id <= center_id + range; ++id) {
            auto kf = keyframe_manager_->getKeyframe(id);
            if (!kf || !kf->cloud || kf->cloud->empty()) continue;

            Eigen::Matrix4f T_kf = kf->pose_optimized.matrix().cast<float>();
            Eigen::Matrix4f T_center_kf = T_center_inv * T_kf;

            PointCloud transformed;
            pcl::transformPointCloud(*kf->cloud, transformed, T_center_kf);
            *submap += transformed;
        }

        // 下采样
        if (config_.gicp_downsampling_resolution > 0.0) {
            pcl::VoxelGrid<pcl::PointXYZI> voxel;
            voxel.setLeafSize(config_.gicp_downsampling_resolution, config_.gicp_downsampling_resolution, config_.gicp_downsampling_resolution);
            voxel.setInputCloud(submap);
            auto filtered = boost::make_shared<PointCloud>();
            voxel.filter(*filtered);
            submap = filtered;
        }
        return submap;
    }

    /**
     * @brief 加载地图
     */
    void loadMap()
    {
        if (config_.map_path.empty()) {
            ROS_ERROR("Map path is empty!");
            return;
        }

        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (!mapping_resuming_->loadExistingMap(config_.map_path)) {
                ROS_ERROR("Failed to load map for extension: %s", config_.map_path.c_str());
                return;
            }
            ROS_INFO("Loaded map for extension with %zu keyframes", mapping_resuming_->getOriginalKeyframeCount());
        } else {
            // 定位模式
            if (!map_serializer_->loadMap(config_.map_path, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                ROS_ERROR("Failed to load map: %s", config_.map_path.c_str());
                return;
            }
            ROS_INFO("Loaded map with %zu keyframes", keyframe_manager_->size());
        }
        map_loaded_ = true;
    }

    /**
     * @brief 加载并发布全局地图 PCD
     *
     * 在定位模式启动时调用，从 map_path 同目录加载 global_map.pcd，
     * 通过 transient_local QoS 发布使 RViz 可以可视化参考地图。
     */
    void loadAndPublishGlobalMap()
    {
        // 从 map_path (pbstream) 推导 global_map.pcd 的路径
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

        ROS_INFO("Loaded global map: %s (%zu points)", global_map_path.c_str(), global_map->size());

        // 转换为 ROS 消息并发布
        sensor_msgs::PointCloud2 global_map_msg;
        pcl::toROSMsg(*global_map, global_map_msg);
        global_map_msg.header.frame_id = config_.world_frame;
        global_map_msg.header.stamp = ros::Time::now();
        global_map_pub_.publish(global_map_msg);

        ROS_INFO("Published global map on /n3mapping/global_map");
    }

    /**
     * @brief 同步回调处理
     * Requirements: 10.2, 5.4, 5.5
     */
    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, const nav_msgs::OdometryConstPtr& odom_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 转换点云
        auto cloud = boost::make_shared<PointCloud>();
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // 转换位姿
        Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
        pose_odom.translation() << odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;
        Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w,
                             odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y,
                             odom_msg->pose.pose.orientation.z);
        pose_odom.linear() = q.toRotationMatrix();

        double timestamp = cloud_msg->header.stamp.toSec();

        switch (run_mode_) {
            case RunMode::MAPPING:
                processMappingMode(timestamp, pose_odom, cloud, cloud_msg->header);
                break;
            case RunMode::LOCALIZATION:
                processLocalizationMode(timestamp, pose_odom, cloud, cloud_msg->header);
                break;
            case RunMode::MAP_EXTENSION:
                processMapExtensionMode(timestamp, pose_odom, cloud, cloud_msg->header);
                break;
        }

        frame_count_++;
    }

    /**
     * @brief 建图模式处理
     */
    void processMappingMode(double timestamp, const Eigen::Isometry3d& pose_odom, PointCloud::Ptr cloud, const std_msgs::Header& header)
    {
        mapping_mode_handler_->process(timestamp, pose_odom, cloud, header);
    }

    /**
     * @brief 定位模式处理
     *
     * 定位策略：
     * 1. 首次启动时进行全局重定位 (ScanContext + ICP)
     * 2. 重定位成功后，使用里程计增量预测 + ICP 配准融合
     * 3. 跟踪失败时重新进行全局重定位
     */
    void processLocalizationMode(double timestamp,
                                 const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud,
                                 const std_msgs::Header& header)
    {
        (void)timestamp;
        localization_mode_handler_->process(map_loaded_, pose_odom, cloud, header);
    }

    /**
     * @brief 地图续建模式处理
     */
    void processMapExtensionMode(double timestamp,
                                 const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud,
                                 const std_msgs::Header& header)
    {
        map_resuming_mode_handler_->process(map_loaded_, timestamp, pose_odom, cloud, header);
    }

    /**
     * @brief 回环检测定时器回调
     */
    void loopDetectionTimerCallback(const ros::TimerEvent&)
    {
        // 定位模式不做回环优化；建图允许；续建需先加载地图
        if (run_mode_ == RunMode::LOCALIZATION) {
            return;
        }
        if (run_mode_ == RunMode::MAP_EXTENSION && map_loaded_ == false) {
            return;
        }
        std::vector<int64_t> keyframes_to_check;
        {
            std::lock_guard<std::mutex> lock(loop_queue_mutex_);
            if (loop_detection_queue_.empty()) {
                return;
            }
            keyframes_to_check.swap(loop_detection_queue_);
        }

        for (int64_t kf_id : keyframes_to_check) {
            auto kf = keyframe_manager_->getKeyframe(kf_id);
            if (!kf) continue;

            // 检测回环候选
            auto candidates = loop_detector_->detectLoopCandidates(kf_id);
            if (candidates.empty()) continue;

            std::vector<VerifiedLoop> verified_loops;
            verified_loops.reserve(candidates.size());

            for (const auto& candidate : candidates) {
                auto match_kf = keyframe_manager_->getKeyframe(candidate.match_id);
                if (!match_kf || !kf->cloud || !match_kf->cloud || kf->cloud->empty() || match_kf->cloud->empty()) {
                    continue;
                }

                auto submap = keyframe_manager_->buildLocalSubmap(candidate.match_id, config_.gicp_submap_size);
                if (!submap || submap->empty()) {
                    submap = boost::make_shared<PointCloud>();
                    Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
                    pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
                }
                if (!submap || submap->empty()) {
                    continue;
                }

                Eigen::Isometry3d init_guess = kf->pose_optimized;
                init_guess.prerotate(Eigen::AngleAxisd(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ()));

                MatchResult match_result = point_cloud_matcher_->alignCloud(submap, kf->cloud, init_guess);

                VerifiedLoop loop;
                loop.query_id = candidate.query_id;
                loop.match_id = candidate.match_id;
                loop.fitness_score = match_result.fitness_score;
                loop.inlier_ratio = match_result.inlier_ratio;
                // 信息矩阵: 默认为零 (让 graph_optimizer 使用配置的固定噪声模型)
                // ICP Hessian 通常过度自信，会导致优化器过度信任回环约束
                if (config_.loop_use_icp_information) {
                    loop.information = match_result.information;
                }
                loop.verified = match_result.success;
                if (match_result.success) {
                    Eigen::Isometry3d corrected_query_pose = match_result.T_target_source;
                    loop.T_match_query = corrected_query_pose.inverse() * match_kf->pose_optimized;

                    // 一致性门控: 检查 ICP 测量值与当前位姿估计的偏差
                    const Eigen::Isometry3d current_T_query_match = kf->pose_optimized.inverse() * match_kf->pose_optimized;
                    const Eigen::Isometry3d consistency_delta = loop.T_match_query.inverse() * current_T_query_match;
                    const double pre_trans_err = consistency_delta.translation().norm();
                    const double pre_rot_err = Eigen::AngleAxisd(consistency_delta.rotation()).angle();

                    // Z 分量检查: 防止错误的 Z 约束导致地图分层
                    const double z_diff = std::abs(loop.T_match_query.translation().z());

                    if (pre_trans_err > config_.loop_max_pre_translation_error || pre_rot_err > config_.loop_max_pre_rotation_error) {
                        ROS_WARN(
                                    "Loop rejected (consistency gate): %ld -> %ld "
                                    "(pre_t=%.2f > %.2f || pre_r=%.2f > %.2f)",
                                    loop.query_id,
                                    loop.match_id,
                                    pre_trans_err,
                                    config_.loop_max_pre_translation_error,
                                    pre_rot_err,
                                    config_.loop_max_pre_rotation_error);
                        LOG(INFO) << "[OPTIMIZATION] loop_rejected query=" << loop.query_id << " match=" << loop.match_id
                                  << " pre_t=" << pre_trans_err << " pre_r=" << pre_rot_err << " fitness=" << loop.fitness_score
                                  << " inlier=" << loop.inlier_ratio;
                        loop.verified = false;
                    } else if (z_diff > config_.loop_max_z_diff) {
                        ROS_WARN(
                                    "Loop rejected (z gate): %ld -> %ld "
                                    "(z_diff=%.2f > %.2f)",
                                    loop.query_id,
                                    loop.match_id,
                                    z_diff,
                                    config_.loop_max_z_diff);
                        LOG(INFO) << "[OPTIMIZATION] loop_rejected_z query=" << loop.query_id << " match=" << loop.match_id
                                  << " z_diff=" << z_diff << " fitness=" << loop.fitness_score
                                  << " inlier=" << loop.inlier_ratio;
                        loop.verified = false;
                    }
                }
                verified_loops.push_back(loop);
            }
            if (!verified_loops.empty()) {
                auto valid_loops = loop_closure_manager_->filterValidLoops(verified_loops);
                auto best_loops = loop_closure_manager_->selectBestPerQuery(valid_loops);
                if (!best_loops.empty()) {
                    struct LoopDiag
                    {
                        int64_t match_id;
                        int64_t query_id;
                        double fitness;
                        double inlier_ratio;
                        double pre_trans_error;
                        double pre_rot_error;
                        Eigen::Isometry3d measured_T_match_query;
                    };
                    std::vector<LoopDiag> loop_diags;
                    loop_diags.reserve(best_loops.size());

                    for (const auto& loop : best_loops) {
                        auto match_kf = keyframe_manager_->getKeyframe(loop.match_id);
                        auto query_kf = keyframe_manager_->getKeyframe(loop.query_id);
                        if (!match_kf || !query_kf) {
                            continue;
                        }
                        const Eigen::Isometry3d current_T_query_match = query_kf->pose_optimized.inverse() * match_kf->pose_optimized;
                        const Eigen::Isometry3d delta = loop.T_match_query.inverse() * current_T_query_match;
                        const double trans_error = delta.translation().norm();
                        const double rot_error = Eigen::AngleAxisd(delta.rotation()).angle();

                        // 详细诊断日志
                        const auto& qt = query_kf->pose_optimized.translation();
                        const auto& mt = match_kf->pose_optimized.translation();
                        const auto& meas_t = loop.T_match_query.translation();
                        const Eigen::Quaterniond meas_q(loop.T_match_query.rotation());
                        LOG(INFO) << "[OPTIMIZATION] loop_edge query=" << loop.query_id << " match=" << loop.match_id
                                  << " fitness=" << loop.fitness_score << " inlier=" << loop.inlier_ratio << " pre_err_t=" << trans_error
                                  << " pre_err_r=" << rot_error;
                        LOG(INFO) << "[OPTIMIZATION] loop_pose query_t=" << qt.x() << "," << qt.y() << "," << qt.z() << " match_t=" << mt.x()
                                  << "," << mt.y() << "," << mt.z();
                        LOG(INFO) << "[OPTIMIZATION] loop_meas t=" << meas_t.x() << "," << meas_t.y() << "," << meas_t.z()
                                  << " q=" << meas_q.w() << "," << meas_q.x() << "," << meas_q.y() << "," << meas_q.z();
                        LOG(INFO) << "[OPTIMIZATION] loop_info diag=" << loop.information(0, 0) << "," << loop.information(1, 1) << ","
                                  << loop.information(2, 2) << "," << loop.information(3, 3) << "," << loop.information(4, 4) << ","
                                  << loop.information(5, 5);

                        ROS_INFO(
                                    "Loop detected: %ld -> %ld (fitness=%.3f, inlier_ratio=%.3f, pre_err_t=%.3f, pre_err_rot=%.3f)",
                                    loop.match_id,
                                    loop.query_id,
                                    loop.fitness_score,
                                    loop.inlier_ratio,
                                    trans_error,
                                    rot_error);

                        loop_diags.push_back(LoopDiag{
                          loop.match_id, loop.query_id, loop.fitness_score, loop.inlier_ratio, trans_error, rot_error, loop.T_match_query });
                    }

                    publishLoopMarkers(best_loops);

                    std::lock_guard<std::mutex> lock(data_mutex_);
                    auto edges = loop_closure_manager_->buildLoopEdges(best_loops, LoopEdgeDirection::QueryToMatch);
                    loop_closure_manager_->applyEdges(edges, *graph_optimizer_);
                    loop_count_ += edges.size();

                    auto poses = graph_optimizer_->getOptimizedPoses();
                    keyframe_manager_->updateOptimizedPoses(poses);
                    logOptimizationResult("loop_closure", ros::Time::now().toSec(), nullptr);

                    for (const auto& diag : loop_diags) {
                        auto match_kf = keyframe_manager_->getKeyframe(diag.match_id);
                        auto query_kf = keyframe_manager_->getKeyframe(diag.query_id);
                        if (!match_kf || !query_kf) {
                            continue;
                        }
                        const Eigen::Isometry3d post_T_query_match = query_kf->pose_optimized.inverse() * match_kf->pose_optimized;
                        const Eigen::Isometry3d post_delta = diag.measured_T_match_query.inverse() * post_T_query_match;
                        const double post_trans_error = post_delta.translation().norm();
                        const double post_rot_error = Eigen::AngleAxisd(post_delta.rotation()).angle();

                        const auto& pqt = query_kf->pose_optimized.translation();
                        const auto& pmt = match_kf->pose_optimized.translation();
                        LOG(INFO) << "[OPTIMIZATION] loop_result query=" << diag.query_id << " match=" << diag.match_id
                                  << " pre_t=" << diag.pre_trans_error << " pre_r=" << diag.pre_rot_error << " post_t=" << post_trans_error
                                  << " post_r=" << post_rot_error;
                        LOG(INFO) << "[OPTIMIZATION] loop_post_pose query_t=" << pqt.x() << "," << pqt.y() << "," << pqt.z()
                                  << " match_t=" << pmt.x() << "," << pmt.y() << "," << pmt.z();

                                                ROS_INFO(
                          "Loop optimized: %ld -> %ld (fitness=%.3f, inlier_ratio=%.3f, pre_t=%.3f, pre_r=%.3f, post_t=%.3f, post_r=%.3f)",
                          diag.match_id,
                          diag.query_id,
                          diag.fitness,
                          diag.inlier_ratio,
                          diag.pre_trans_error,
                          diag.pre_rot_error,
                          post_trans_error,
                          post_rot_error);
                    }
                }
            }
        }
    }

    /**
     * @brief 发布里程计
     */
    void publishOdometry(const Eigen::Isometry3d& pose, const std_msgs::Header& header)
    {
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

        // 发布 TF
        geometry_msgs::TransformStamped tf;
        tf.header = odom_msg.header;
        tf.child_frame_id = config_.body_frame;
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        tf.transform.rotation = odom_msg.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

    /**
     * @brief 发布路径
     */
    void publishPath(const std_msgs::Header& header, const Eigen::Isometry3d* current_pose)
    {
        nav_msgs::Path path_msg;
        path_msg.header = header;
        path_msg.header.frame_id = config_.world_frame;

        if (run_mode_ == RunMode::LOCALIZATION) {
            // 定位模式: 只累积发布实时定位轨迹
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
            // 建图/续建模式: 按 version 分段构建彩色轨迹 marker
            // 同时保留 nav_msgs/Path 发布全部轨迹（用于数据记录/导航）
            std::vector<geometry_msgs::Point> loaded_points; // 旧地图 (蓝色)
            std::vector<geometry_msgs::Point> new_points;    // 新建帧 (绿色)

            for (const auto& kf : keyframe_manager_->getAllKeyframes()) {
                if (!kf) continue;
                geometry_msgs::PoseStamped pose;
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

                geometry_msgs::Point pt;
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

                geometry_msgs::Point pt;
                pt.x = current_pose->translation().x();
                pt.y = current_pose->translation().y();
                pt.z = current_pose->translation().z();
                new_points.push_back(pt);
            }

            // 发布彩色轨迹 marker 到 loop_closure_markers 话题
            publishPathMarkers(loaded_points, new_points, header);
        }

        path_pub_.publish(path_msg);
    }

    /**
     * @brief 发布分版本彩色轨迹 Marker
     *
     * 旧地图关键帧用蓝色 LINE_STRIP，新建关键帧用绿色 LINE_STRIP，
     * 发布到 loop_closure_markers 话题与回环线条共存。
     */
    void publishPathMarkers(const std::vector<geometry_msgs::Point>& loaded_points,
                            const std::vector<geometry_msgs::Point>& new_points,
                            const std_msgs::Header& header)
    {
        visualization_msgs::MarkerArray markers;

        // 旧地图轨迹 (蓝色)
        if (loaded_points.size() >= 2) {
            visualization_msgs::Marker loaded_marker;
            loaded_marker.header.frame_id = config_.world_frame;
            loaded_marker.header.stamp = header.stamp;
            loaded_marker.ns = "path_loaded";
            loaded_marker.id = 0;
            loaded_marker.type = visualization_msgs::Marker::LINE_STRIP;
            loaded_marker.action = visualization_msgs::Marker::ADD;
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
            visualization_msgs::Marker new_marker;
            new_marker.header.frame_id = config_.world_frame;
            new_marker.header.stamp = header.stamp;
            new_marker.ns = "path_new";
            new_marker.id = 0;
            new_marker.type = visualization_msgs::Marker::LINE_STRIP;
            new_marker.action = visualization_msgs::Marker::ADD;
            new_marker.scale.x = 0.15;
            new_marker.color.r = 0.1f;
            new_marker.color.g = 1.0f;
            new_marker.color.b = 0.2f;
            new_marker.color.a = 0.9f;
            new_marker.points = new_points;
            markers.markers.push_back(new_marker);
        }

        if (!markers.markers.empty()) {
            loop_marker_pub_.publish(markers);
        }
    }

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose)
    {
        auto keyframes = keyframe_manager_->getAllKeyframes();
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

        LOG(INFO) << "[OPTIMIZATION] context=" << context << " time=" << timestamp << " keyframes=" << valid_keyframes.size()
                  << " latest_id=" << (latest_id ? std::to_string(*latest_id) : std::string("none"));

        if (current_pose) {
            Eigen::Quaterniond q(current_pose->rotation());
            LOG(INFO) << "[OPTIMIZATION] current t=" << current_pose->translation().x() << "," << current_pose->translation().y() << ","
                      << current_pose->translation().z() << " q=" << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
        }
    }

    void publishLoopMarkers(const std::vector<VerifiedLoop>& loops)
    {
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker clear_marker;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        markers.markers.push_back(clear_marker);

        int marker_id = 0;
        for (const auto& loop : loops) {
            auto match_kf = keyframe_manager_->getKeyframe(loop.match_id);
            auto query_kf = keyframe_manager_->getKeyframe(loop.query_id);
            if (!match_kf || !query_kf) {
                continue;
            }

            visualization_msgs::Marker marker;
            marker.header.frame_id = config_.world_frame;
            marker.header.stamp = ros::Time::now();
            marker.ns = "loop_closure";
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::LINE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x = 0.1;
            marker.color.r = 1.0f;
            marker.color.g = 0.2f;
            marker.color.b = 0.2f;
            marker.color.a = 0.9f;

            geometry_msgs::Point p_match;
            p_match.x = match_kf->pose_optimized.translation().x();
            p_match.y = match_kf->pose_optimized.translation().y();
            p_match.z = match_kf->pose_optimized.translation().z();

            geometry_msgs::Point p_query;
            p_query.x = query_kf->pose_optimized.translation().x();
            p_query.y = query_kf->pose_optimized.translation().y();
            p_query.z = query_kf->pose_optimized.translation().z();

            marker.points.push_back(p_match);
            marker.points.push_back(p_query);

            markers.markers.push_back(marker);
        }

        loop_marker_pub_.publish(markers);
    }

    /**
     * @brief 发布点云
     */
    void publishPointClouds(PointCloud::Ptr cloud, const Eigen::Isometry3d& pose, const std_msgs::Header& header)
    {
        // Body frame 点云
        sensor_msgs::PointCloud2 cloud_body_msg;
        pcl::toROSMsg(*cloud, cloud_body_msg);
        cloud_body_msg.header = header;
        cloud_body_msg.header.frame_id = config_.body_frame;
        cloud_body_pub_.publish(cloud_body_msg);

        // World frame 点云
        auto cloud_world = boost::make_shared<PointCloud>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_world, transform);

        sensor_msgs::PointCloud2 cloud_world_msg;
        pcl::toROSMsg(*cloud_world, cloud_world_msg);
        cloud_world_msg.header = header;
        cloud_world_msg.header.frame_id = config_.world_frame;
        cloud_world_pub_.publish(cloud_world_msg);
    }

    /**
     * @brief 关闭时保存地图
     */
    void saveMapOnShutdown()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        std::string map_file = config_.map_save_path + "/n3map.pbstream";

        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (mapping_resuming_->saveExtendedMap(map_file)) {
                ROS_INFO("Extended map saved to: %s", map_file.c_str());
            } else {
                ROS_ERROR("Failed to save extended map");
            }
        } else {
            if (map_serializer_->saveMap(map_file, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                ROS_INFO("Map saved to: %s", map_file.c_str());
            } else {
                ROS_ERROR("Failed to save map");
            }
        }

        // 保存全局点云
        if (config_.save_global_map_on_shutdown) {
            std::string global_map_file = config_.map_save_path + "/global_map.pcd";
            if (map_serializer_->saveGlobalMap(global_map_file, *keyframe_manager_)) {
                ROS_INFO("Global map saved to: %s", global_map_file.c_str());
            }
        }
    }

    /**
     * @brief 打印统计信息
     */
    void printStatistics()
    {
        ROS_INFO("========== N3Mapping Statistics ==========");
        ROS_INFO("Total frames processed: %zu", frame_count_);
        ROS_INFO("Total keyframes: %zu", keyframe_count_);
        ROS_INFO("Loop closures detected: %zu", loop_count_);
        ROS_INFO("===========================================");
    }

  private:
    // 配置
    Config config_;
    RunMode run_mode_ = RunMode::MAPPING;

    // 核心组件
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

    // ROS 接口
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

    double last_reloc_verify_time_ = 0.0; // 上次周期性全局验证的时间戳（秒）

    // 线程和同步
    std::mutex data_mutex_;
    std::mutex loop_queue_mutex_;
    std::vector<int64_t> loop_detection_queue_;

    // 状态
    bool map_loaded_ = false;
    bool shutdown_called_ = false;
    std::unique_ptr<OptimizationLogSink> optimization_log_sink_;
    std::string optimization_log_path_;
    std::vector<geometry_msgs::PoseStamped> localization_path_; // 定位模式累积轨迹

    // 统计
    size_t frame_count_ = 0;
    size_t keyframe_count_ = 0;
    size_t loop_count_ = 0;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
};

} // namespace n3mapping

// 信号处理
void
signalHandler(int signum)
{
    (void)signum; // 避免未使用参数警告
    n3mapping::g_shutdown_requested = true;
}

int
main(int argc, char** argv)
{
    ros::init(argc, argv, "n3mapping_node");

    // 初始化 glog，日志写入单一文件而不是终端
    google::InitGoogleLogging(argv[0]);
#ifdef N3MAPPING_SOURCE_DIR
    std::string log_dir = std::string(N3MAPPING_SOURCE_DIR) + "/map";
#else
    std::string log_dir = "./map";
#endif
    std::filesystem::create_directories(log_dir);
    FLAGS_stderrthreshold = google::GLOG_ERROR;  // 只有 ERROR 及以上才输出到终端
    FLAGS_alsologtostderr = false;
    FLAGS_colorlogtostderr = false;
    FLAGS_minloglevel = 0;
    FLAGS_logbufsecs = 0;  // 日志实时刷新，不缓冲
    FLAGS_stop_logging_if_full_disk = true;
    // 所有级别统一写入同一个 INFO 日志文件，禁止 WARNING/ERROR 单独创建文件
    std::string log_prefix = log_dir + "/n3mapping.";
    google::SetLogDestination(google::GLOG_INFO, log_prefix.c_str());
    google::SetLogDestination(google::GLOG_WARNING, "");  // 禁止单独的 WARNING 文件
    google::SetLogDestination(google::GLOG_ERROR, "");    // 禁止单独的 ERROR 文件
    google::SetLogSymlink(google::GLOG_INFO, "n3mapping.log");
    google::SetLogSymlink(google::GLOG_WARNING, "");
    google::SetLogSymlink(google::GLOG_ERROR, "");

    // 注册信号处理
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
