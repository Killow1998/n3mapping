#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
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
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt32.h>
#include <std_srvs/Trigger.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "n3mapping/core/n3mapping_core.h"
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
        ROS_INFO("%s", config_.toString().c_str());

        run_mode_ = parseCoreRunMode(config_.mode);
        core_ = std::make_unique<N3MappingCore>(config_);

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
        cloud_sub_.subscribe(nh_, config_.cloud_topic, 10);
        odom_sub_.subscribe(nh_, config_.odom_topic, 10);
        sync_ = std::make_unique<Synchronizer>(SyncPolicy(10), cloud_sub_, odom_sub_);
        sync_->registerCallback(boost::bind(&N3MappingNoeticNode::syncCallback, this, _1, _2));

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
    }

    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                      const nav_msgs::OdometryConstPtr& odom_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        const auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);

        const auto output = core_->processFrame(run_mode_, frame);

        if (run_mode_ == CoreRunMode::MAP_EXTENSION && output.relocalization_locked && !output.accepted_keyframe) {
            ROS_INFO_THROTTLE(2.0, "Initial relocalization successful for map extension");
        }
        if (!output.success && !output.accepted_keyframe && run_mode_ != CoreRunMode::LOCALIZATION) {
            return;
        }

        publishOdometry(output.T_world_lidar, cloud_msg->header);
        publishPath(cloud_msg->header, &output.T_world_lidar);
        publishPointClouds(output, cloud_msg->header);

        if (output.relocalization_locked) {
            publishRelocalizationLock(cloud_msg->header, output.T_world_lidar);
        }
        if (output.accepted_keyframe) {
            ++keyframe_count_;
            publishGlobalMap();
        }
        ++frame_count_;
    }

    void loopTimerCallback(const ros::TimerEvent&)
    {
        if (!coreRunModeProcessesLoopClosures(run_mode_)) {
            return;
        }

        std::lock_guard<std::mutex> lock(data_mutex_);
        const auto result = core_->processPendingLoopClosures();
        if (!result.accepted_loops.empty()) {
            publishLoopMarkers(result.accepted_loops);
        }
        if (result.optimized) {
            loop_count_ += result.edge_count;
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = config_.world_frame;
            publishPath(header, nullptr);
            publishGlobalMap();
            ROS_INFO("Loop optimization: edges=%zu accepted=%zu pose_updates=%zu",
                     result.edge_count,
                     result.accepted_loops.size(),
                     result.pose_update_count);
        }
    }

    void globalMapTimerCallback(const ros::TimerEvent&)
    {
        if (global_map_pub_.getNumSubscribers() == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(data_mutex_);
        publishGlobalMap();
    }

    bool handleSaveMap(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std::string error;
        res.success = core_ && core_->saveMapSnapshot(&error);
        res.message = res.success ? ("saved:" + config_.map_save_path) : (error.empty() ? "save_failed" : error);
        if (res.success) {
            ROS_INFO("save_map service finished: %s", res.message.c_str());
        } else {
            ROS_ERROR("save_map service failed: %s", res.message.c_str());
        }
        return true;
    }

    void publishOdometry(const Eigen::Isometry3d& pose, const std_msgs::Header& header)
    {
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = config_.world_frame;
        odom_msg.child_frame_id = config_.body_frame;
        odom_msg.pose.pose = makePose(pose);
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

    void publishGlobalMap()
    {
        auto global_map = core_ ? core_->buildGlobalMap() : nullptr;
        if (!global_map || global_map->empty()) {
            return;
        }
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global_map, msg);
        msg.header.frame_id = config_.world_frame;
        msg.header.stamp = ros::Time::now();
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
            std::lock_guard<std::mutex> lock(data_mutex_);
            std::string error;
            if (core_ && core_->saveMapSnapshot(&error)) {
                ROS_INFO("Map snapshot saved under: %s", config_.map_save_path.c_str());
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

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::unique_ptr<Synchronizer> sync_;

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
