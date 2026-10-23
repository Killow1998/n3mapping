#include "n3mapping_node.h"
#include "n3mapping/humble/config_humble.h"
#include <filesystem>
#include <fstream>

namespace n3mapping {

std::atomic<bool> g_shutdown_requested{false};

N3MappingNode::N3MappingNode(
  std::optional<std::string> authority_probe_scenario)
  : Node("n3mapping_node"),
    authority_probe_scenario_(std::move(authority_probe_scenario))
{
    input_linear_velocity_frame_ = parseInputLinearVelocityFrame(
        this->declare_parameter<std::string>("input_linear_velocity_frame", "child"));
    localization_status_pub_ = this->create_publisher<std_msgs::msg::String>(
        this->declare_parameter<std::string>("output_status_topic", "/localization/status"), 1);
    if (authority_probe_scenario_) {
        initializeAuthorityProbe();
        return;
    }
    product_profile_v1_ =
      this->declare_parameter<bool>("product_profile_v1", false);
    if (product_profile_v1_) {
        const std::string map_path =
          this->declare_parameter<std::string>("map_path", "");
        const std::string atlas_path =
          this->declare_parameter<std::string>("reloc_atlas_path", "");
        if (map_path.empty() || atlas_path.empty()) {
            throw std::runtime_error(
              "product_profile_v1 requires non-empty map_path and "
              "reloc_atlas_path");
        }
        config_ = makeProductLocalizationConfig(map_path, atlas_path);
    } else {
        loadConfigFromHumble(this, &config_);
    }
    RCLCPP_INFO(this->get_logger(), "%s", config_.toString().c_str());

    run_mode_ = parseCoreRunMode(config_.mode);
    if (config_.mode != coreRunModeName(run_mode_)) {
        RCLCPP_WARN(this->get_logger(), "Unknown mode '%s', defaulting to mapping", config_.mode.c_str());
    }

    // 初始化组件
    initializeComponents();

    // 初始化 ROS 接口
    initializeROS();

    if (coreRunModeSavesMap(run_mode_)) {
        initializeOptimizationLogging();
    }

    // 根据模式加载地图
    if (coreRunModeLoadsMap(run_mode_)) {
        loadMap();
        // 定位模式下加载并发布全局地图用于可视化
        if (map_loaded_) {
            loadAndPublishGlobalMap();
        }
    }

    RCLCPP_INFO(this->get_logger(), "N3Mapping node initialized. Mode: %s", config_.mode.c_str());
}

void N3MappingNode::shutdown()
{
    if (shutdown_called_.exchange(true)) return;

    RCLCPP_INFO(this->get_logger(), "Shutting down N3Mapping node...");

    if (loop_timer_) {
        loop_timer_->cancel();
    }
    if (authority_probe_timer_) {
        authority_probe_timer_->cancel();
    }
    stopLiveOutput();
    if (authority_probe_scenario_) {
        return;
    }

    // 保存地图
    if (coreRunModeSavesMap(run_mode_)) {
        saveMapOnShutdown();
    }

    // 输出统计信息
    printStatistics();
}

void N3MappingNode::initializeComponents()
{
    n3mapping_core_ = std::make_unique<N3MappingCore>(
        config_, N3MappingCore::WorldCloudOutput::Disabled);
    realtime_output_ = std::make_unique<core::RealtimeOutput>(
        run_mode_, config_.world_frame, std::max(1, config_.sync_queue_size));
    n3mapping_core_->setExternalDenseTrajectoryRecordingEnabled(
      run_mode_ == CoreRunMode::MAPPING);
    global_map_cache_.setVoxelSize(config_.global_map_voxel_size);
    if (config_.reloc_debug_enable) {
        performance_debug_logger_ =
          std::make_unique<RuntimePerformanceDebugLogger>(config_);
        if (performance_debug_logger_->ready()) {
            RCLCPP_INFO(
              this->get_logger(), "Runtime performance JSONL: %s",
              performance_debug_logger_->path().c_str());
        } else {
            RCLCPP_WARN(
              this->get_logger(),
              "Failed to open runtime performance JSONL: %s",
              performance_debug_logger_->path().c_str());
        }
    }
}

void N3MappingNode::initializeROS()
{
    // TF 广播器
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    const int sync_queue_size = std::max(1, config_.sync_queue_size);
    rmw_qos_profile_t sync_qos = rmw_qos_profile_default;
    sync_qos.depth = static_cast<std::size_t>(sync_queue_size);

    // 订阅者
    cloud_sub_.subscribe(this, config_.cloud_topic, sync_qos);
    odom_sub_.subscribe(this, config_.odom_topic, sync_qos);
    if (run_mode_ == CoreRunMode::MAPPING) {
        dense_odom_sub_ =
          this->create_subscription<nav_msgs::msg::Odometry>(
            config_.odom_topic,
            rclcpp::QoS(static_cast<std::size_t>(sync_queue_size)),
            std::bind(
              &N3MappingNode::denseOdomCallback, this,
              std::placeholders::_1));
    }

    if (product_profile_v1_) {
        ExactSyncPolicy sync_policy(static_cast<uint32_t>(sync_queue_size));
        exact_sync_ =
          std::make_unique<
            message_filters::Synchronizer<ExactSyncPolicy>>(
            static_cast<const ExactSyncPolicy&>(sync_policy),
            cloud_sub_, odom_sub_);
        exact_sync_->registerCallback(
          std::bind(
            &N3MappingNode::syncCallback, this,
            std::placeholders::_1, std::placeholders::_2));
    } else {
        ApproxSyncPolicy sync_policy(
          static_cast<uint32_t>(sync_queue_size));
        sync_policy.setMaxIntervalDuration(
          rclcpp::Duration::from_seconds(config_.sync_time_tolerance));
        approx_sync_ =
          std::make_unique<
            message_filters::Synchronizer<ApproxSyncPolicy>>(
            static_cast<const ApproxSyncPolicy&>(sync_policy),
            cloud_sub_, odom_sub_);
        approx_sync_->registerCallback(
          std::bind(
            &N3MappingNode::syncCallback, this,
            std::placeholders::_1, std::placeholders::_2));
    }

    initializeLiveSubscriptions();

    // 发布者
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(config_.output_odom_topic, 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(config_.output_path_topic, 10);
    cloud_body_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(config_.output_cloud_body_topic, 10);
    cloud_world_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(config_.output_cloud_world_topic, 10);
    rclcpp::QoS marker_qos(10);
    marker_qos.transient_local();
    loop_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/n3mapping/loop_closure_markers", marker_qos);
    relocalization_lock_pub_ = this->create_publisher<std_msgs::msg::UInt32>("/n3mapping/relocalization_lock", 10);
    rclcpp::QoS relocalization_status_qos(1);
    relocalization_status_qos.reliable();
    relocalization_status_qos.transient_local();
    relocalization_status_pub_ = this->create_publisher<msg::RelocalizationStatus>(
      "/n3mapping/relocalization_status", relocalization_status_qos);
    rclcpp::QoS relocalization_pose_qos(1);
    relocalization_pose_qos.reliable();
    relocalization_pose_qos.durability_volatile();
    relocalization_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/n3mapping/relocalization_pose", relocalization_pose_qos);
    if (coreRunModeSavesMap(run_mode_)) {
        save_map_srv_ =
          this->create_service<std_srvs::srv::Trigger>(
            "/n3mapping/save_map",
            std::bind(&N3MappingNode::handleSaveMap, this,
                      std::placeholders::_1,
                      std::placeholders::_2));
    }

    // 全局地图发布者 (transient_local QoS，等价于 latched topic 语义)
    rclcpp::QoS global_map_qos(1);
    global_map_qos.transient_local();
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/global_map", global_map_qos);

    loop_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&N3MappingNode::loopDetectionTimerCallback, this));
    const double global_map_hz = std::max(0.1, config_.global_map_publish_hz);
    global_map_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(1.0 / global_map_hz)),
      std::bind(&N3MappingNode::globalMapTimerCallback, this));
}

void N3MappingNode::initializeOptimizationLogging()
{
    std::error_code ec;
    std::filesystem::create_directories(config_.map_save_path, ec);
    optimization_log_path_ = config_.map_save_path + "/optimization.log";
    std::ofstream file(optimization_log_path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        RCLCPP_WARN(this->get_logger(), "Failed to open optimization log: %s", optimization_log_path_.c_str());
    }
}

void N3MappingNode::loadMap()
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
    resetGlobalMapCache();
}

void N3MappingNode::printStatistics()
{
    RCLCPP_INFO(this->get_logger(), "========== N3Mapping Statistics ==========");
    RCLCPP_INFO(this->get_logger(), "Total frames processed: %zu", frame_count_);
    RCLCPP_INFO(this->get_logger(), "Total keyframes: %zu", keyframe_count_);
    RCLCPP_INFO(this->get_logger(), "Loop closures detected: %zu", loop_count_);
    RCLCPP_INFO(this->get_logger(), "===========================================");
}

N3MappingNode::~N3MappingNode() { shutdown(); }

}  // namespace n3mapping
