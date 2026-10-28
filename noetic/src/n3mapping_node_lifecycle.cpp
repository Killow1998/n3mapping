#include "n3mapping_node.h"
#include "n3mapping_noetic/config_noetic.h"
#include <algorithm>
#include <stdexcept>

namespace n3mapping {

N3MappingNoeticNode::N3MappingNoeticNode()
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
    map_artifacts_ = std::make_unique<MapArtifacts>(config_);
    if (config_.reloc_debug_enable) performance_debug_logger_ = std::make_unique<RuntimePerformanceDebugLogger>(config_);
    ROS_INFO("%s", config_.toString().c_str());

    run_mode_ = parseCoreRunMode(config_.mode);
    core_ = std::make_unique<N3MappingCore>(
        config_, N3MappingCore::WorldCloudOutput::Disabled);
    realtime_output_ = std::make_unique<core::RealtimeOutput>(
        run_mode_, config_.world_frame, std::max(1, config_.sync_queue_size));
    core_->setExternalDenseTrajectoryRecordingEnabled(
        run_mode_ == CoreRunMode::MAPPING);
    if (coreRunModeSavesMap(run_mode_)) {
        initializeOptimizationLogging();
    }

    initializeROS();

    if (coreRunModeLoadsMap(run_mode_)) {
        loadMap();
    }

    ROS_INFO("N3Mapping Noetic node initialized. Mode: %s", config_.mode.c_str());
}

N3MappingNoeticNode::~N3MappingNoeticNode() { shutdown(); }

bool N3MappingNoeticNode::handleStartMapExtension(
    std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    res.success = core_->startMapExtension(&res.message);
    if (!res.success) return true;
    res.message = "map_extension";
    if (run_mode_ == CoreRunMode::MAP_EXTENSION) return true;
    pending_localization_frame_.take();
    run_mode_ = CoreRunMode::MAP_EXTENSION;
    config_.mode = coreRunModeName(run_mode_);
    realtime_output_->setMode(run_mode_);
    map_artifacts_->setMode(run_mode_);
    initializeOptimizationLogging();
    return true;
}

void N3MappingNoeticNode::spin()
{
    startLiveOutput();
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

void N3MappingNoeticNode::initializeROS()
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
    initializeLiveSubscriptions();
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
    save_map_srv_ = nh_.advertiseService(
        "/n3mapping/save_map", &N3MappingNoeticNode::handleSaveMap, this);
    start_extension_srv_ = nh_.advertiseService(
        "/n3mapping/start_map_extension",
        &N3MappingNoeticNode::handleStartMapExtension, this);

    loop_timer_ = nh_.createTimer(ros::Duration(0.1), &N3MappingNoeticNode::loopDetectionTimerCallback, this);
    const double global_map_hz = std::max(0.1, config_.global_map_publish_hz);
    global_map_timer_ =
        nh_.createTimer(ros::Duration(1.0 / global_map_hz), &N3MappingNoeticNode::globalMapTimerCallback, this);
}

void N3MappingNoeticNode::shutdown()
{
    if (shutdown_called_) return;
    shutdown_called_ = true;
    stopLiveOutput();
    if (coreRunModeSavesMap(run_mode_)) saveMapOnShutdown();
    ROS_INFO("N3Mapping statistics: frames=%zu keyframes=%zu loops=%zu",
             frame_count_,
             keyframe_count_,
             loop_count_);
}

}  // namespace n3mapping
