/**
 * @file n3mapping_node.cpp
 * @brief N3Mapping 主节点实现
 *
 * ROS2 wrapper for the ROS-free n3mapping core backend.
 * Receives external LIO cloud/odometry frames and publishes ROS outputs.
 *
 * Requirements: 1.1, 1.2, 1.3, 5.4, 5.5, 6.1, 6.2, 6.3, 7.1, 7.8, 9.6, 10.2, 10.3,
 * 10.4, 11.1
 */

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/memory.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <limits>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string_view>
#include <std_msgs/msg/u_int32.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <visualization_msgs/msg/marker_array.hpp>

#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/ros2/config_ros2.h"
#include "n3mapping/ros2/conversions.h"

namespace n3mapping {

// 全局节点指针用于信号处理
static std::atomic<bool> g_shutdown_requested{ false };

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
class N3MappingNode : public rclcpp::Node
{
  public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;

    N3MappingNode()
      : Node("n3mapping_node")
    {
        // 加载配置
        loadConfigFromRos2(this, &config_);
        RCLCPP_INFO(this->get_logger(), "%s", config_.toString().c_str());

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

        RCLCPP_INFO(this->get_logger(), "N3Mapping node initialized. Mode: %s", config_.mode.c_str());
    }

    ~N3MappingNode() { shutdown(); }

    /**
     * @brief 关闭节点，保存地图
     */
    void shutdown()
    {
        if (shutdown_called_) return;
        shutdown_called_ = true;

        RCLCPP_INFO(this->get_logger(), "Shutting down N3Mapping node...");

        if (loop_timer_) {
            loop_timer_->cancel();
        }

        // 保存地图
        if (run_mode_ == RunMode::MAPPING || run_mode_ == RunMode::MAP_EXTENSION) {
            saveMapOnShutdown();
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
            RCLCPP_WARN(this->get_logger(), "Unknown mode '%s', defaulting to mapping", config_.mode.c_str());
            run_mode_ = RunMode::MAPPING;
        }
    }

    /**
     * @brief 初始化组件
     */
    void initializeComponents()
    {
        n3mapping_core_ = std::make_unique<N3MappingCore>(config_);
    }

    /**
     * @brief 初始化 ROS 接口
     */
    void initializeROS()
    {
        // TF 广播器
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // 订阅者
        cloud_sub_.subscribe(this, config_.cloud_topic);
        odom_sub_.subscribe(this, config_.odom_topic);

        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), cloud_sub_, odom_sub_);
        sync_->registerCallback(std::bind(&N3MappingNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

        // 发布者
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(config_.output_odom_topic, 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>(config_.output_path_topic, 10);
        cloud_body_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(config_.output_cloud_body_topic, 10);
        cloud_world_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(config_.output_cloud_world_topic, 10);
        rclcpp::QoS marker_qos(10);
        marker_qos.transient_local();
        loop_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/n3mapping/loop_closure_markers", marker_qos);
        relocalization_lock_pub_ = this->create_publisher<std_msgs::msg::UInt32>("/n3mapping/relocalization_lock", 10);

        // 全局地图发布者 (transient_local QoS，类似 ROS1 latched topic)
        rclcpp::QoS global_map_qos(1);
        global_map_qos.transient_local();
        global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/global_map", global_map_qos);

        loop_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        loop_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(100), std::bind(&N3MappingNode::loopDetectionTimerCallback, this), loop_callback_group_);
    }

    void initializeOptimizationLogging()
    {
        std::error_code ec;
        std::filesystem::create_directories(config_.map_save_path, ec);
        optimization_log_path_ = config_.map_save_path + "/optimization.log";
        std::ofstream file(optimization_log_path_, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            RCLCPP_WARN(this->get_logger(), "Failed to open optimization log: %s", optimization_log_path_.c_str());
        }
    }

    /**
     * @brief 加载地图
     */
    void loadMap()
    {
        if (config_.map_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Map path is empty!");
            return;
        }

        if (!n3mapping_core_->loadMap(config_.map_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load map: %s", config_.map_path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded map with %zu keyframes", n3mapping_core_->getAllKeyframes().size());
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
            RCLCPP_WARN(this->get_logger(), "Global map PCD not found: %s", global_map_path.c_str());
            return;
        }

        auto global_map = pcl::make_shared<PointCloud>();
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(global_map_path.string(), *global_map) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load global map: %s", global_map_path.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Loaded global map: %s (%zu points)", global_map_path.c_str(), global_map->size());

        // 转换为 ROS 消息并发布
        sensor_msgs::msg::PointCloud2 global_map_msg;
        pcl::toROSMsg(*global_map, global_map_msg);
        global_map_msg.header.frame_id = config_.world_frame;
        global_map_msg.header.stamp = this->get_clock()->now();
        global_map_pub_->publish(global_map_msg);

        RCLCPP_INFO(this->get_logger(), "Published global map on /n3mapping/global_map");
    }

    /**
     * @brief 同步回调处理
     * Requirements: 10.2, 5.4, 5.5
     */
    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);
        auto cloud = frame.undistorted_cloud;
        const Eigen::Isometry3d pose_odom = frame.T_world_lidar;
        double timestamp = rclcpp::Time(cloud_msg->header.stamp).seconds();

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
    void processMappingMode(double timestamp, const Eigen::Isometry3d& pose_odom, PointCloud::Ptr cloud, const std_msgs::msg::Header& header)
    {
        core::LioFrame frame;
        frame.stamp.nsec = static_cast<int64_t>(timestamp * 1e9);
        frame.T_world_lidar = pose_odom;
        frame.undistorted_cloud = cloud;
        frame.pose_valid = true;

        const auto output = n3mapping_core_->processMappingFrame(frame);
        const Eigen::Isometry3d publish_pose = output.T_world_lidar;
        publishOdometry(publish_pose, header);
        publishPath(header, &publish_pose);
        publishPointClouds(cloud, publish_pose, header);

        if (output.accepted_keyframe) {
            ++keyframe_count_;
            logOptimizationResult("mapping_incremental", timestamp, &publish_pose);
            publishGlobalMap();
        }
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
                                 const std_msgs::msg::Header& header)
    {
        core::LioFrame frame;
        frame.stamp.nsec = static_cast<int64_t>(timestamp * 1e9);
        frame.T_world_lidar = pose_odom;
        frame.undistorted_cloud = cloud;
        frame.pose_valid = true;

        const auto output = n3mapping_core_->processLocalizationFrame(frame);
        if (output.relocalization_locked) {
            publishRelocalizationLock(header, output.T_world_lidar);
        }
        publishOdometry(output.T_world_lidar, header);
        publishPath(header, &output.T_world_lidar);
        publishPointClouds(cloud, output.T_world_lidar, header);
    }

    /**
     * @brief 地图续建模式处理
     */
    void processMapExtensionMode(double timestamp,
                                 const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud,
                                 const std_msgs::msg::Header& header)
    {
        core::LioFrame frame;
        frame.stamp.nsec = static_cast<int64_t>(timestamp * 1e9);
        frame.T_world_lidar = pose_odom;
        frame.undistorted_cloud = cloud;
        frame.pose_valid = true;

        const auto output = n3mapping_core_->processMapExtensionFrame(frame);
        if (output.relocalization_locked && !output.accepted_keyframe) {
            RCLCPP_INFO(this->get_logger(), "Initial relocalization successful for map extension");
            return;
        }
        if (!output.success && !output.accepted_keyframe) {
            return;
        }

        publishOdometry(output.T_world_lidar, header);
        publishPath(header, &output.T_world_lidar);
        publishPointClouds(cloud, output.T_world_lidar, header);

        if (output.accepted_keyframe) {
            ++keyframe_count_;
            logOptimizationResult("map_extension_incremental", timestamp, &output.T_world_lidar);
            publishGlobalMap();
        }
    }

    /**
     * @brief 回环检测定时器回调
     */
    void loopDetectionTimerCallback()
    {
        if (run_mode_ != RunMode::MAPPING) {
            return;
        }

        const auto result = n3mapping_core_->processPendingLoopClosures();
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
            publishGlobalMap();
        }
    }

    /**
     * @brief 发布里程计
     */
    void publishOdometry(const Eigen::Isometry3d& pose, const std_msgs::msg::Header& header)
    {
        nav_msgs::msg::Odometry odom_msg;
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

        odom_pub_->publish(odom_msg);

        // 发布 TF
        geometry_msgs::msg::TransformStamped tf;
        tf.header = odom_msg.header;
        tf.child_frame_id = config_.body_frame;
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        tf.transform.rotation = odom_msg.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

    void publishRelocalizationLock(const std_msgs::msg::Header& header, const Eigen::Isometry3d& pose)
    {
        if (!relocalization_lock_pub_) {
            return;
        }

        std_msgs::msg::UInt32 msg;
        msg.data = ++relocalization_lock_count_;
        relocalization_lock_pub_->publish(msg);

        Eigen::Quaterniond q(pose.rotation());
        const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
        RCLCPP_INFO(this->get_logger(),
                    "Relocalization lock event #%u @ %.3f pose=(%.3f, %.3f, %.3f) yaw=%.3f",
                    msg.data,
                    rclcpp::Time(header.stamp).seconds(),
                    pose.translation().x(),
                    pose.translation().y(),
                    pose.translation().z(),
                    yaw);
    }

    /**
     * @brief 发布路径
     */
    void publishPath(const std_msgs::msg::Header& header, const Eigen::Isometry3d* current_pose)
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header = header;
        path_msg.header.frame_id = config_.world_frame;

        if (run_mode_ == RunMode::LOCALIZATION) {
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
                localization_path_.push_back(pose);
            }
            path_msg.poses = localization_path_;
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

    /**
     * @brief 发布分版本彩色轨迹 Marker
     *
     * 旧地图关键帧用蓝色 LINE_STRIP，新建关键帧用绿色 LINE_STRIP，
     * 发布到 loop_closure_markers 话题与回环线条共存。
     */
    void publishPathMarkers(const std::vector<geometry_msgs::msg::Point>& loaded_points,
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

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose)
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

    void appendOptimizationLogLine(const std::string& line)
    {
        std::lock_guard<std::mutex> lock(optimization_log_mutex_);
        std::ofstream file(optimization_log_path_, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            return;
        }
        file << line << '\n';
    }

    void publishLoopMarkers(const std::vector<VerifiedLoop>& loops)
    {
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(clear_marker);

        int marker_id = 0;
        for (const auto& loop : loops) {
            auto match_kf = getKeyframeForPublishing(loop.match_id);
            auto query_kf = getKeyframeForPublishing(loop.query_id);
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

    std::vector<Keyframe::Ptr> getKeyframesForPublishing() const
    {
        if (n3mapping_core_) {
            return n3mapping_core_->getAllKeyframes();
        }
        return {};
    }

    Keyframe::Ptr getKeyframeForPublishing(int64_t id) const
    {
        if (n3mapping_core_) {
            return n3mapping_core_->getKeyframe(id);
        }
        return nullptr;
    }

    /**
     * @brief 发布点云
     */
    void publishPointClouds(PointCloud::Ptr cloud, const Eigen::Isometry3d& pose, const std_msgs::msg::Header& header)
    {
        // Body frame 点云
        sensor_msgs::msg::PointCloud2 cloud_body_msg;
        pcl::toROSMsg(*cloud, cloud_body_msg);
        cloud_body_msg.header = header;
        cloud_body_msg.header.frame_id = config_.body_frame;
        cloud_body_pub_->publish(cloud_body_msg);

        // World frame 点云
        auto cloud_world = pcl::make_shared<PointCloud>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_world, transform);

        sensor_msgs::msg::PointCloud2 cloud_world_msg;
        pcl::toROSMsg(*cloud_world, cloud_world_msg);
        cloud_world_msg.header = header;
        cloud_world_msg.header.frame_id = config_.world_frame;
        cloud_world_pub_->publish(cloud_world_msg);
    }

    void publishGlobalMap()
    {
        if (!n3mapping_core_) {
            return;
        }
        auto global_map = n3mapping_core_->buildGlobalMap();
        if (!global_map || global_map->empty()) {
            return;
        }

        sensor_msgs::msg::PointCloud2 global_map_msg;
        pcl::toROSMsg(*global_map, global_map_msg);
        global_map_msg.header.frame_id = config_.world_frame;
        global_map_msg.header.stamp = this->get_clock()->now();
        global_map_pub_->publish(global_map_msg);
    }

    /**
     * @brief 关闭时保存地图
     */
    void saveMapOnShutdown()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        std::string map_file = config_.map_save_path + "/n3map.pbstream";

        const bool saved = n3mapping_core_
                             ? n3mapping_core_->saveMap(map_file)
                             : false;
        if (saved) {
            RCLCPP_INFO(this->get_logger(), "Map saved to: %s", map_file.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save map");
        }

        // 保存全局点云
        if (config_.save_global_map_on_shutdown) {
            std::string global_map_file = config_.map_save_path + "/global_map.pcd";
            const bool saved = ((run_mode_ == RunMode::MAPPING || run_mode_ == RunMode::MAP_EXTENSION) && n3mapping_core_)
                                 ? n3mapping_core_->saveGlobalMap(global_map_file)
                                 : false;
            if (saved) {
                RCLCPP_INFO(this->get_logger(), "Global map saved to: %s", global_map_file.c_str());
            }
        }
    }

    /**
     * @brief 打印统计信息
     */
    void printStatistics()
    {
        RCLCPP_INFO(this->get_logger(), "========== N3Mapping Statistics ==========");
        RCLCPP_INFO(this->get_logger(), "Total frames processed: %zu", frame_count_);
        RCLCPP_INFO(this->get_logger(), "Total keyframes: %zu", keyframe_count_);
        RCLCPP_INFO(this->get_logger(), "Loop closures detected: %zu", loop_count_);
        RCLCPP_INFO(this->get_logger(), "===========================================");
    }

  private:
    // 配置
    Config config_;
    RunMode run_mode_ = RunMode::MAPPING;

    // 核心组件
    std::unique_ptr<N3MappingCore> n3mapping_core_;

    // ROS 接口
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_body_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_world_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr relocalization_lock_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::TimerBase::SharedPtr loop_timer_;
    rclcpp::CallbackGroup::SharedPtr loop_callback_group_;
    rclcpp::TimerBase::SharedPtr reloc_verify_timer_;
    rclcpp::CallbackGroup::SharedPtr reloc_verify_callback_group_;

    double last_reloc_verify_time_ = 0.0; // 上次周期性全局验证的时间戳（秒）

    // 线程和同步
    std::mutex data_mutex_;

    // 状态
    bool map_loaded_ = false;
    bool shutdown_called_ = false;
    std::string optimization_log_path_;
    std::mutex optimization_log_mutex_;
    std::vector<geometry_msgs::msg::PoseStamped> localization_path_; // 定位模式累积轨迹

    // 统计
    size_t frame_count_ = 0;
    size_t keyframe_count_ = 0;
    size_t loop_count_ = 0;
    uint32_t relocalization_lock_count_ = 0;
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
    rclcpp::init(argc, argv);

    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    auto node = std::make_shared<n3mapping::N3MappingNode>();

    // 使用多线程执行器
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    while (rclcpp::ok() && !n3mapping::g_shutdown_requested) {
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    node->shutdown();
    rclcpp::shutdown();
    return 0;
}
