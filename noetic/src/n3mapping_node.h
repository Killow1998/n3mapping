#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt32.h>
#include <std_srvs/Trigger.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <n3mapping/RelocalizationStatus.h>
#include "n3mapping/async_log.h"
#include "n3mapping/runtime_events.h"
#include "n3mapping/runtime_performance_debug_logger.h"
#include "n3mapping/core/latest_frame_slot.h"
#include "n3mapping/core/map_artifacts.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/realtime_output.h"
#include "n3mapping/ros_output_contract.h"

namespace n3mapping {
class RosLogging {
public:
    RosLogging();
    ~RosLogging();
};
// Owns ROS resources and scheduling; shared core owns SLAM and output decisions.
class N3MappingNoeticNode {
  public:
    N3MappingNoeticNode();

    ~N3MappingNoeticNode();

    void spin();

  private:
    using ApproxSyncPolicy =
        message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using ExactSyncPolicy =
        message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using ApproxSynchronizer =
        message_filters::Synchronizer<ApproxSyncPolicy>;
    using ExactSynchronizer =
        message_filters::Synchronizer<ExactSyncPolicy>;

    void initializeROS();
    void initializeLiveSubscriptions();
    void startLiveOutput();
    void stopLiveOutput();
    void saveMapOnShutdown();

    void loadMap();

    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                      const nav_msgs::OdometryConstPtr& odom_msg);

    void processSynchronizedFrame(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                  const nav_msgs::OdometryConstPtr& odom_msg);

    void liveOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg);

    void denseOdomCallback(const nav_msgs::OdometryConstPtr& odom_msg);

    void loopDetectionTimerCallback(const ros::TimerEvent&);

    void globalMapTimerCallback(const ros::TimerEvent&);

    bool handleSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res);
    bool handleStartMapExtension(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res);

    void publishLiveTransform(const Eigen::Isometry3d& pose,
                              const std_msgs::Header& header);

    void publishPath(const std_msgs::Header& header, const Eigen::Isometry3d* current_pose);

    void liveCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud,
                           const nav_msgs::OdometryConstPtr& odom);

    void initializeOptimizationLogging();

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose);

    void appendOptimizationLogLine(const std::string& line);

    void resetGlobalMapCache();

    void markGlobalMapFullRebuildRequired();

    bool refreshGlobalMapMessage(const MapKeyframeSnapshot& snapshot,
                                 sensor_msgs::PointCloud2* out_msg);

    void publishGlobalMap();

    void publishGlobalMap(const MapKeyframeSnapshot& snapshot);

    void publishLegacyRelocalizationLock(std::uint64_t lock_epoch);

    void publishAuthoritativeRelocalizationPose(
        const std_msgs::Header& header, const Eigen::Isometry3d& pose,
        std::uint64_t lock_epoch);

    void publishRelocalizationStatus(const std_msgs::Header& header,
                                     const core::BackendOutput& output,
                                     std::uint64_t lock_epoch);

    void publishLoopMarkers(const std::vector<VerifiedLoop>& loops);

    void publishPathMarkers(const std::vector<geometry_msgs::Point>& loaded_points,
                            const std::vector<geometry_msgs::Point>& new_points,
                            const std_msgs::Header& header);

    void addPathMarker(const std::string& ns,
                       int id,
                       const std::vector<geometry_msgs::Point>& points,
                       float r,
                       float g,
                       float b,
                       const std_msgs::Header& header,
                       visualization_msgs::MarkerArray* markers);

    geometry_msgs::Pose makePose(const Eigen::Isometry3d& pose) const;

    geometry_msgs::Point toPoint(const Eigen::Isometry3d& pose) const;

    void shutdown();

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
    std::unique_ptr<core::RealtimeOutput> realtime_output_;
    std::optional<bool> last_logged_backend_authorized_;
    ros::Time previous_backend_authorized_stamp_;
    std::optional<std::chrono::steady_clock::time_point>
        previous_backend_authorized_update_;
    std::string last_logged_live_reason_;
    std::optional<std::chrono::steady_clock::time_point> last_live_odom_receive_;
    ros::CallbackQueue live_odom_queue_;
    ros::CallbackQueue live_cloud_queue_;
    ros::Subscriber live_odom_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> live_cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> live_paired_odom_sub_;
    std::unique_ptr<ApproxSynchronizer> live_approx_sync_;
    std::unique_ptr<ExactSynchronizer> live_exact_sync_;
    std::unique_ptr<ros::AsyncSpinner> live_odom_spinner_;
    std::unique_ptr<ros::AsyncSpinner> live_cloud_spinner_;
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
    ros::ServiceServer start_extension_srv_;
    ros::Timer loop_timer_;
    ros::Timer global_map_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::unique_ptr<MapArtifacts> map_artifacts_;
    std::optional<sensor_msgs::PointCloud2> global_map_msg_cache_;
    std::uint64_t global_map_msg_revision_ = 0;
    std::optional<std::uint64_t> global_map_last_published_revision_;
    RuntimeEvents runtime_events_;
    std::unique_ptr<RuntimePerformanceDebugLogger> performance_debug_logger_;
    std::uint64_t performance_frame_index_ = 0, performance_loop_index_ = 0;
    ros::Publisher logging_status_pub_;
    ros::WallTimer logging_status_timer_;
    std::string optimization_log_path_;

    std::vector<geometry_msgs::PoseStamped> localization_path_;
    ros::Time last_tf_stamp_;
    std::size_t frame_count_ = 0;
    std::size_t keyframe_count_ = 0;
    std::size_t loop_count_ = 0;
    bool shutdown_called_ = false;
};

}  // namespace n3mapping
