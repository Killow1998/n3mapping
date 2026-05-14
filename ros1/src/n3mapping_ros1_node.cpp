#include <memory>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping_ros1/config_ros1.h"
#include "n3mapping_ros1/conversions.h"

namespace n3mapping {

class N3MappingRos1Node {
  public:
    N3MappingRos1Node()
      : nh_()
      , private_nh_("~")
    {
        loadConfigFromRos1(private_nh_, &config_);
        core_ = std::make_unique<N3MappingCore>(config_);

        cloud_sub_.subscribe(nh_, config_.cloud_topic, 10);
        odom_sub_.subscribe(nh_, config_.odom_topic, 10);
        sync_ = std::make_unique<Synchronizer>(SyncPolicy(10), cloud_sub_, odom_sub_);
        sync_->registerCallback(boost::bind(&N3MappingRos1Node::syncCallback, this, _1, _2));

        odom_pub_ = nh_.advertise<nav_msgs::Odometry>(config_.output_odom_topic, 10);
        cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.output_cloud_body_topic, 10);
    }

  private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void syncCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                      const nav_msgs::OdometryConstPtr& odom_msg)
    {
        const auto frame = toCoreLioFrame(*cloud_msg, *odom_msg);
        core::BackendOutput output;
        if (config_.mode == "localization") {
            output = core_->processLocalizationFrame(frame);
        } else if (config_.mode == "map_extension") {
            output = core_->processMapExtensionFrame(frame);
        } else {
            output = core_->processMappingFrame(frame);
        }

        nav_msgs::Odometry odom_out;
        odom_out.header = cloud_msg->header;
        odom_out.header.frame_id = config_.world_frame;
        odom_out.child_frame_id = config_.body_frame;
        odom_out.pose.pose.position.x = output.T_world_lidar.translation().x();
        odom_out.pose.pose.position.y = output.T_world_lidar.translation().y();
        odom_out.pose.pose.position.z = output.T_world_lidar.translation().z();
        const Eigen::Quaterniond q(output.T_world_lidar.rotation());
        odom_out.pose.pose.orientation.w = q.w();
        odom_out.pose.pose.orientation.x = q.x();
        odom_out.pose.pose.orientation.y = q.y();
        odom_out.pose.pose.orientation.z = q.z();
        odom_pub_.publish(odom_out);

        if (frame.undistorted_cloud) {
            sensor_msgs::PointCloud2 cloud_body;
            pcl::toROSMsg(*frame.undistorted_cloud, cloud_body);
            cloud_body.header = cloud_msg->header;
            cloud_body.header.frame_id = config_.body_frame;
            cloud_body_pub_.publish(cloud_body);
        }
    }

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    Config config_;
    std::unique_ptr<N3MappingCore> core_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::unique_ptr<Synchronizer> sync_;
    ros::Publisher odom_pub_;
    ros::Publisher cloud_body_pub_;
};

}  // namespace n3mapping

int main(int argc, char** argv)
{
    ros::init(argc, argv, "n3mapping_ros1_node");
    n3mapping::N3MappingRos1Node node;
    ros::spin();
    return 0;
}
