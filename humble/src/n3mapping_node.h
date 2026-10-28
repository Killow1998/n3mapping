#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include "n3mapping/async_log.h"
#include "n3mapping/runtime_events.h"
#include "n3mapping/core/latest_frame_slot.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/realtime_output.h"
#include "n3mapping/core/map_artifacts.h"
#include "n3mapping/humble/callback_worker.h"
#include "n3mapping/humble/conversions.h"
#include "n3mapping/msg/relocalization_status.hpp"
#include "n3mapping/ros_output_contract.h"
#include "n3mapping/runtime_performance_debug_logger.h"

namespace n3mapping {
class RosLogging {
public:
    RosLogging();
    ~RosLogging();
};
// Shared by the process signal handler and the finite authority probe.
extern std::atomic<bool> g_shutdown_requested;

// Owns the ROS subscriptions, publication workers and shutdown transaction.
// Definitions are grouped by responsibility; backend decisions stay in core.
class N3MappingNode : public rclcpp::Node
{
    friend class HumbleOutputTestAccess;
  public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
    using ApproxSyncPolicy =
      message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;
    using ExactSyncPolicy =
      message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;

    explicit N3MappingNode(
      std::optional<std::string> authority_probe_scenario = std::nullopt);

    ~N3MappingNode() override;

    /**
     * @brief 关闭节点，保存地图
     */
    void shutdown();

    void startLiveOutput();

    void processPendingLocalizationFrame();

  private:
    void initializeAuthorityProbe();

    static core::BackendOutput makeAuthorityProbeOutput(
      const std::string& label, int event_index);

    void authorityProbeTick();

    /**
     * @brief 初始化组件
     */
    void initializeComponents();

    /**
     * @brief 初始化 ROS 接口
     */
    void initializeROS();

    void initializeLiveSubscriptions();

    void stopLiveOutput();

    int64_t liveOutputNow();

    void liveOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr odom);

    void liveCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                           const nav_msgs::msg::Odometry::ConstSharedPtr& odom);

  protected:
    // Conversion is outside the core and backend locks. This boundary also lets
    // contract tests pause real publication during conversion and invalidate it.
    virtual sensor_msgs::msg::PointCloud2 transformLiveCloud(
        const sensor_msgs::msg::PointCloud2& cloud, const Eigen::Isometry3d& pose);

  private:
    void initializeOptimizationLogging();

    /**
     * @brief 加载地图
     */
    void loadMap();

    /**
     * @brief Load global_map.pcd when available, otherwise publish a map built from loaded keyframes.
     */

    /**
     * @brief 同步回调处理
     * Requirements: 10.2, 5.4, 5.5
     */
    struct SynchronizedFrame {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
        nav_msgs::msg::Odometry::ConstSharedPtr odom;
    };

    void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                      const nav_msgs::msg::Odometry::ConstSharedPtr& odom);

    void processSynchronizedFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg);

    RelocalizationPublicationPlan publishRelocalizationOutput(
      const std_msgs::msg::Header& header, const nav_msgs::msg::Odometry& source,
      core::BackendOutput& output);

    void denseOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg);

    /**
     * @brief 回环检测定时器回调
     */
    void loopDetectionTimerCallback();

    void globalMapTimerCallback();

    /**
     * @brief 发布里程计
     */
    geometry_msgs::msg::Pose makePose(const Eigen::Isometry3d& pose) const;

    void publishLiveTransform(const Eigen::Isometry3d& pose,
                              const std_msgs::msg::Header& source_header);

    void publishLegacyRelocalizationLock(std::uint64_t lock_epoch);

    void publishAuthoritativeRelocalizationPose(
      const std_msgs::msg::Header& header, const Eigen::Isometry3d& pose,
      std::uint64_t lock_epoch);

    void publishRelocalizationStatus(
      const std_msgs::msg::Header& header, const core::BackendOutput& output,
      std::uint64_t lock_epoch);

    /**
     * @brief 发布路径
     */
    void publishPath(const std_msgs::msg::Header& header, const Eigen::Isometry3d* current_pose);

    /**
     * @brief 发布分版本彩色轨迹 Marker
     *
     * 旧地图关键帧用蓝色 LINE_STRIP，新建关键帧用绿色 LINE_STRIP，
     * 发布到 loop_closure_markers 话题与回环线条共存。
     */
    void publishPathMarkers(const std::vector<geometry_msgs::msg::Point>& loaded_points,
                            const std::vector<geometry_msgs::msg::Point>& new_points,
                            const std_msgs::msg::Header& header);

    void logOptimizationResult(const std::string& context, double timestamp, const Eigen::Isometry3d* current_pose);

    void appendOptimizationLogLine(const std::string& line);

    void publishLoopMarkers(const std::vector<VerifiedLoop>& loops);

    std::vector<Keyframe::Ptr> getKeyframesForPublishing() const;

    Keyframe::Ptr getKeyframeForPublishing(int64_t id) const;

    void resetGlobalMapCache();

    void markGlobalMapFullRebuildRequired();

    bool refreshGlobalMapMessage(const MapKeyframeSnapshot& snapshot,
                                 sensor_msgs::msg::PointCloud2* out_msg);


    void publishGlobalMap(const MapKeyframeSnapshot& snapshot);

    /**
     * @brief 关闭时保存地图
     */
  protected:
    virtual void saveMapOnShutdown();

  private:
    void handleSaveMap(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handleStartMapExtension(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    /**
     * @brief 打印统计信息
     */
    void printStatistics();

  private:
    // 配置
    Config config_;
    InputLinearVelocityFrame input_linear_velocity_frame_ = InputLinearVelocityFrame::CHILD;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr localization_status_pub_;
    CoreRunMode run_mode_ = CoreRunMode::MAPPING;
    bool product_profile_v1_ = false;

    // 核心组件
    std::unique_ptr<N3MappingCore> n3mapping_core_;
    std::unique_ptr<RuntimePerformanceDebugLogger>
      performance_debug_logger_;

    // ROS 接口
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>>
      approx_sync_;
    std::unique_ptr<message_filters::Synchronizer<ExactSyncPolicy>>
      exact_sync_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr dense_odom_sub_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_body_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_world_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr relocalization_lock_pub_;
    rclcpp::Publisher<msg::RelocalizationStatus>::SharedPtr
      relocalization_status_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      relocalization_pose_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_map_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_extension_srv_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::TimerBase::SharedPtr loop_timer_;
    rclcpp::TimerBase::SharedPtr global_map_timer_;
    rclcpp::TimerBase::SharedPtr authority_probe_timer_;
    std::unique_ptr<MapArtifacts> map_artifacts_;
    std::optional<std::uint64_t> global_map_last_published_revision_;

    // 线程和同步
    std::mutex data_mutex_;
    std::mutex global_map_mutex_;

    // 状态
    bool map_loaded_ = false;
    std::atomic<bool> shutdown_called_{false};
    RuntimeEvents runtime_events_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr logging_status_pub_;
    rclcpp::TimerBase::SharedPtr logging_status_timer_;
    std::string optimization_log_path_;
    nav_msgs::msg::Path localization_path_; // 定位模式累积轨迹
    std::vector<std::pair<int64_t, int64_t>> loop_marker_history_;
    std::optional<rclcpp::Time> last_tf_stamp_;

    // 统计
    bool odometry_divergence_reported_ = false;
    std::atomic<bool> performance_log_failure_reported_{false};
    uint64_t performance_frame_index_ = 0;
    uint64_t performance_loop_cycle_index_ = 0;
    std::optional<std::chrono::steady_clock::time_point>
      last_performance_callback_started_;
    std::optional<double> last_performance_sensor_timestamp_;
    size_t frame_count_ = 0;
    size_t keyframe_count_ = 0;
    size_t loop_count_ = 0;
    std::unique_ptr<core::RealtimeOutput> realtime_output_;
    std::unique_ptr<CallbackWorker> live_odom_worker_;
    std::unique_ptr<CallbackWorker> live_cloud_worker_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr live_odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> live_cloud_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> live_paired_odom_sub_;
    std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> live_approx_sync_;
    std::unique_ptr<message_filters::Synchronizer<ExactSyncPolicy>> live_exact_sync_;
    core::LatestFrameSlot<SynchronizedFrame> pending_localization_frame_;
    std::atomic<int64_t> authority_probe_now_nsec_{0};
    std::optional<std::string> authority_probe_scenario_;
    int authority_probe_tick_ = 0;
};

}  // namespace n3mapping
