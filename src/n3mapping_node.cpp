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
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <csignal>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <limits>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_extension_module.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_module.h"

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
        config_.loadFromROS(this);
        config_.print(this->get_logger());

        // 解析运行模式
        parseRunMode();

        // 初始化组件
        initializeComponents();

        // 初始化 ROS 接口
        initializeROS();

        // 根据模式加载地图
        if (run_mode_ == RunMode::LOCALIZATION || run_mode_ == RunMode::MAP_EXTENSION) {
            loadMap();
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
        keyframe_manager_ = std::make_unique<KeyframeManager>(config_);
        point_cloud_matcher_ = std::make_unique<PointCloudMatcher>(config_);
        loop_detector_ = std::make_unique<LoopDetector>(config_);
        graph_optimizer_ = std::make_unique<GraphOptimizer>(config_);
        map_serializer_ = std::make_unique<MapSerializer>(config_);
        relocalization_module_ = std::make_unique<RelocalizationModule>(config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_);
        map_extension_module_ = std::make_unique<MapExtensionModule>(
          config_, *keyframe_manager_, *loop_detector_, *point_cloud_matcher_, *graph_optimizer_, *map_serializer_, *relocalization_module_);
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

        loop_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        loop_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(100), std::bind(&N3MappingNode::loopDetectionTimerCallback, this), loop_callback_group_);
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

        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (!map_extension_module_->loadExistingMap(config_.map_path)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load map for extension: %s", config_.map_path.c_str());
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Loaded map for extension with %zu keyframes", map_extension_module_->getOriginalKeyframeCount());
        } else {
            // 定位模式
            if (!map_serializer_->loadMap(config_.map_path, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load map: %s", config_.map_path.c_str());
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Loaded map with %zu keyframes", keyframe_manager_->size());
        }
        map_loaded_ = true;
    }

    /**
     * @brief 同步回调处理
     * Requirements: 10.2, 5.4, 5.5
     */
    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 转换点云
        auto cloud = std::make_shared<PointCloud>();
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // 转换位姿
        Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
        pose_odom.translation() << odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;
        Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w,
                             odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y,
                             odom_msg->pose.pose.orientation.z);
        pose_odom.linear() = q.toRotationMatrix();

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
        // 检查是否需要添加关键帧
        if (!keyframe_manager_->shouldAddKeyframe(pose_odom)) {
            // 发布当前位姿
            publishOdometry(pose_odom, header);
            publishPointClouds(cloud, pose_odom, header);
            return;
        }

        // 添加关键帧
        int64_t kf_id = keyframe_manager_->addKeyframe(timestamp, pose_odom, cloud);
        loop_detector_->addDescriptor(kf_id, cloud);

        // 添加图优化因子
        if (kf_id == 0) {
            graph_optimizer_->addPriorFactor(kf_id, pose_odom);
        } else {
            auto prev_kf = keyframe_manager_->getKeyframe(kf_id - 1);
            if (prev_kf) {
                EdgeInfo edge;
                edge.from_id = kf_id - 1;
                edge.to_id = kf_id;
                edge.measurement = prev_kf->pose_odom.inverse() * pose_odom;
                edge.information = Eigen::Matrix<double, 6, 6>::Identity();
                edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
                edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
                edge.type = EdgeType::ODOMETRY;
                graph_optimizer_->addOdometryEdge(edge);
            }
        }

        // 增量优化
        graph_optimizer_->incrementalOptimize();

        // 更新优化后位姿
        Eigen::Isometry3d optimized_pose = pose_odom;
        if (graph_optimizer_->hasNode(kf_id)) {
            try {
                optimized_pose = graph_optimizer_->getOptimizedPose(kf_id);
                std::map<int64_t, Eigen::Isometry3d> pose_update;
                pose_update[kf_id] = optimized_pose;
                keyframe_manager_->updateOptimizedPoses(pose_update);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Failed to get optimized pose: %s", e.what());
            }
        }

        // 添加到回环检测队列
        {
            std::lock_guard<std::mutex> lock(loop_queue_mutex_);
            loop_detection_queue_.push_back(kf_id);
        }

        // 发布
        Eigen::Isometry3d publish_pose = optimized_pose;
        publishOdometry(publish_pose, header);
        publishPath();
        publishPointClouds(cloud, publish_pose, header);

        keyframe_count_++;
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
        if (!map_loaded_) {
            LOG_EVERY_N(WARNING, 10) << "Map not loaded, cannot perform localization";
        }

        Eigen::Isometry3d pose_map;
        bool success = false;

        if (relocalization_module_->isRelocalized()) {
            auto result = relocalization_module_->trackLocalization(cloud, pose_odom);
            if (result.success) {
                pose_map = result.pose_in_map;
                success = true;
            }
        }
        if (!relocalization_module_->isRelocalized() || !success) {
            auto result = relocalization_module_->relocalize(cloud, pose_odom);
            if (result.success) {
                pose_map = result.pose_in_map;
                success = true;
            }
        }

        if (success) {
            publishOdometry(pose_map, header);
            publishPointClouds(cloud, pose_map, header);
        }
    }

    /**
     * @brief 地图续建模式处理
     */
    void processMapExtensionMode(double timestamp,
                                 const Eigen::Isometry3d& pose_odom,
                                 PointCloud::Ptr cloud,
                                 const std_msgs::msg::Header& header)
    {
        if (!map_loaded_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Map not loaded for extension");
            return;
        }

        auto state = map_extension_module_->getState();

        if (state == MapExtensionState::MAP_LOADED) {
            // 需要先重定位
            if (map_extension_module_->performInitialRelocalization(cloud, pose_odom)) {
                RCLCPP_INFO(this->get_logger(), "Initial relocalization successful for map extension");
            }
            return;
        }

        if (state != MapExtensionState::RELOCALIZED && state != MapExtensionState::EXTENDING) {
            return;
        }

        // 检查是否需要添加关键帧
        if (!keyframe_manager_->shouldAddKeyframe(pose_odom)) {
            auto T_map_odom = relocalization_module_->getMapToOdomTransform();
            Eigen::Isometry3d pose_map = T_map_odom * pose_odom;
            publishOdometry(pose_map, header);
            publishPointClouds(cloud, pose_map, header);
            return;
        }

        // 处理新关键帧
        auto T_map_odom = relocalization_module_->getMapToOdomTransform();
        Eigen::Isometry3d pose_map = T_map_odom * pose_odom;

        int64_t kf_id = map_extension_module_->processNewKeyframe(timestamp, pose_odom, cloud);
        if (kf_id >= 0) {
            // 检测新旧关键帧之间的回环
            int cross_loops = map_extension_module_->detectCrossLoops(kf_id);
            if (cross_loops > 0) {
                RCLCPP_INFO(this->get_logger(), "Detected %d cross-loops for keyframe %ld", cross_loops, kf_id);
            }

            // 增量优化
            graph_optimizer_->incrementalOptimize();

            // 更新优化后位姿
            if (graph_optimizer_->hasNode(kf_id)) {
                try {
                    Eigen::Isometry3d optimized_pose = graph_optimizer_->getOptimizedPose(kf_id);
                    std::map<int64_t, Eigen::Isometry3d> pose_update;
                    pose_update[kf_id] = optimized_pose;
                    keyframe_manager_->updateOptimizedPoses(pose_update);
                    pose_map = optimized_pose;
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "Failed to get optimized pose: %s", e.what());
                }
            }

            keyframe_count_++;
        }

        publishOdometry(pose_map, header);
        publishPath();
        publishPointClouds(cloud, pose_map, header);
    }

    /**
     * @brief 回环检测定时器回调
     */
    void loopDetectionTimerCallback()
    {
        if (run_mode_ == RunMode::LOCALIZATION || map_loaded_ == false) {
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

            // 构建关键帧映射用于验证
            std::map<int64_t, Keyframe::Ptr> keyframes_map;
            keyframes_map[kf_id] = kf;
            for (const auto& candidate : candidates) {
                auto match_kf = keyframe_manager_->getKeyframe(candidate.match_id);
                if (match_kf) {
                    keyframes_map[candidate.match_id] = match_kf;
                }
            }

            // 验证回环候选
            auto verified_loops = loop_detector_->verifyLoopCandidatesBatch(candidates, keyframes_map, *point_cloud_matcher_);
            if (!verified_loops.empty()) {
                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    for (const auto& loop : verified_loops) {
                        EdgeInfo edge;
                        edge.from_id = loop.query_id;
                        edge.to_id = loop.match_id;
                        edge.measurement = loop.T_match_query;
                        edge.information = loop.information;
                        edge.type = EdgeType::LOOP;

                        graph_optimizer_->addLoopEdge(edge);
                        loop_count_++;

                        RCLCPP_INFO(
                          this->get_logger(), "Loop detected: %ld -> %ld (score: %.3f)", loop.query_id, loop.match_id, loop.fitness_score);
                    }
                    graph_optimizer_->optimize();
                    // 获取并更新优化后的位姿
                    auto poses = graph_optimizer_->getOptimizedPoses();
                    keyframe_manager_->updateOptimizedPoses(poses);
                }
            }
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

    /**
     * @brief 发布路径
     */
    void publishPath()
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = config_.world_frame;

        for (const auto& kf : keyframe_manager_->getAllKeyframes()) {
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
        }

        path_pub_->publish(path_msg);
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
        auto cloud_world = std::make_shared<PointCloud>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_world, transform);

        sensor_msgs::msg::PointCloud2 cloud_world_msg;
        pcl::toROSMsg(*cloud_world, cloud_world_msg);
        cloud_world_msg.header = header;
        cloud_world_msg.header.frame_id = config_.world_frame;
        cloud_world_pub_->publish(cloud_world_msg);
    }

    /**
     * @brief 关闭时保存地图
     */
    void saveMapOnShutdown()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        std::string map_file = config_.map_save_path + "/n3map.pbstream";

        if (run_mode_ == RunMode::MAP_EXTENSION) {
            if (map_extension_module_->saveExtendedMap(map_file)) {
                RCLCPP_INFO(this->get_logger(), "Extended map saved to: %s", map_file.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save extended map");
            }
        } else {
            if (map_serializer_->saveMap(map_file, *keyframe_manager_, *loop_detector_, *graph_optimizer_)) {
                RCLCPP_INFO(this->get_logger(), "Map saved to: %s", map_file.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save map");
            }
        }

        // 保存全局点云
        if (config_.save_global_map_on_shutdown) {
            std::string global_map_file = config_.map_save_path + "/global_map.pcd";
            if (map_serializer_->saveGlobalMap(global_map_file, *keyframe_manager_)) {
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
    std::unique_ptr<KeyframeManager> keyframe_manager_;
    std::unique_ptr<PointCloudMatcher> point_cloud_matcher_;
    std::unique_ptr<LoopDetector> loop_detector_;
    std::unique_ptr<GraphOptimizer> graph_optimizer_;
    std::unique_ptr<MapSerializer> map_serializer_;
    std::unique_ptr<RelocalizationModule> relocalization_module_;
    std::unique_ptr<MapExtensionModule> map_extension_module_;

    // ROS 接口
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_body_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_world_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::TimerBase::SharedPtr loop_timer_;
    rclcpp::CallbackGroup::SharedPtr loop_callback_group_;
    rclcpp::TimerBase::SharedPtr reloc_verify_timer_;
    rclcpp::CallbackGroup::SharedPtr reloc_verify_callback_group_;

    double last_reloc_verify_time_ = 0.0; // 上次周期性全局验证的时间戳（秒）

    // 线程和同步
    std::mutex data_mutex_;
    std::mutex loop_queue_mutex_;
    std::vector<int64_t> loop_detection_queue_;

    // 状态
    bool map_loaded_ = false;
    bool shutdown_called_ = false;

    // 统计
    size_t frame_count_ = 0;
    size_t keyframe_count_ = 0;
    size_t loop_count_ = 0;
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
