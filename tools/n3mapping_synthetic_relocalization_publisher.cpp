#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "n3mapping/core/n3mapping_core.h"

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;

struct Options {
    std::string map_path;
    std::string cloud_topic = "/cloud_registered_body";
    std::string odom_topic = "/Odometry";
    int64_t query_id = -1;
    int repeat = 20;
    double rate_hz = 5.0;
    double dropout = 0.0;
    double noise_sigma = 0.0;
    double fake_odom_yaw_deg = 90.0;
    double fake_odom_tx = 20.0;
    double fake_odom_ty = -10.0;
    double fake_odom_tz = 1.0;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " --map /path/to/n3map.pbstream [options]\n"
        << "Options:\n"
        << "  --query_id ID             Keyframe id to synthesize. Default: middle keyframe\n"
        << "  --cloud_topic TOPIC       Default: /cloud_registered_body\n"
        << "  --odom_topic TOPIC        Default: /Odometry\n"
        << "  --repeat N                Number of synchronized frames to publish. Default: 20, 0 means until shutdown\n"
        << "  --rate_hz HZ              Publish rate. Default: 5\n"
        << "  --dropout R               Random point dropout ratio [0,1). Default: 0\n"
        << "  --noise_sigma M           XYZ Gaussian noise in meters. Default: 0\n"
        << "  --fake_odom_yaw_deg DEG   Fake map->odom yaw. Default: 90\n"
        << "  --fake_odom_tx M          Fake map->odom x. Default: 20\n"
        << "  --fake_odom_ty M          Fake map->odom y. Default: -10\n"
        << "  --fake_odom_tz M          Fake map->odom z. Default: 1\n";
}

bool parseArgs(int argc, char** argv, Options* options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "--map") {
            if (const char* v = needValue(arg)) options->map_path = v; else return false;
        } else if (arg == "--query_id") {
            if (const char* v = needValue(arg)) options->query_id = std::stoll(v); else return false;
        } else if (arg == "--cloud_topic") {
            if (const char* v = needValue(arg)) options->cloud_topic = v; else return false;
        } else if (arg == "--odom_topic") {
            if (const char* v = needValue(arg)) options->odom_topic = v; else return false;
        } else if (arg == "--repeat") {
            if (const char* v = needValue(arg)) options->repeat = std::max(0, std::stoi(v)); else return false;
        } else if (arg == "--rate_hz") {
            if (const char* v = needValue(arg)) options->rate_hz = std::max(0.1, std::stod(v)); else return false;
        } else if (arg == "--dropout") {
            if (const char* v = needValue(arg)) options->dropout = std::clamp(std::stod(v), 0.0, 0.95); else return false;
        } else if (arg == "--noise_sigma") {
            if (const char* v = needValue(arg)) options->noise_sigma = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--fake_odom_yaw_deg") {
            if (const char* v = needValue(arg)) options->fake_odom_yaw_deg = std::stod(v); else return false;
        } else if (arg == "--fake_odom_tx") {
            if (const char* v = needValue(arg)) options->fake_odom_tx = std::stod(v); else return false;
        } else if (arg == "--fake_odom_ty") {
            if (const char* v = needValue(arg)) options->fake_odom_ty = std::stod(v); else return false;
        } else if (arg == "--fake_odom_tz") {
            if (const char* v = needValue(arg)) options->fake_odom_tz = std::stod(v); else return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (options->map_path.empty()) {
        std::cerr << "--map is required\n";
        return false;
    }
    return true;
}

Config makePublisherConfig()
{
    Config config;
    config.rhpd_enabled = true;
    config.rhpd_dist_threshold = 100.0;
    config.reloc_static_agg_enable = false;
    return config;
}

Eigen::Isometry3d makeFakeMapOdom(const Options& options)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(options.fake_odom_tx, options.fake_odom_ty, options.fake_odom_tz);
    pose.linear() = Eigen::AngleAxisd(options.fake_odom_yaw_deg * M_PI / 180.0,
                                      Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return pose;
}

Cloud::Ptr perturbCloud(const Cloud::Ptr& input, const Options& options, std::uint32_t seed)
{
    auto output = pcl::make_shared<Cloud>();
    if (!input) return output;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, options.noise_sigma);
    output->reserve(input->size());

    for (const auto& point : input->points) {
        if (keep_dist(rng) < options.dropout) continue;
        pcl::PointXYZI out = point;
        out.x = static_cast<float>(out.x + noise(rng));
        out.y = static_cast<float>(out.y + noise(rng));
        out.z = static_cast<float>(out.z + noise(rng));
        output->push_back(out);
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = input->is_dense;
    return output;
}

nav_msgs::msg::Odometry makeOdomMsg(const Eigen::Isometry3d& pose,
                                    const rclcpp::Time& stamp)
{
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "odom";
    msg.child_frame_id = "body";
    msg.pose.pose.position.x = pose.translation().x();
    msg.pose.pose.position.y = pose.translation().y();
    msg.pose.pose.position.z = pose.translation().z();
    const Eigen::Quaterniond q(pose.rotation());
    msg.pose.pose.orientation.w = q.w();
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    return msg;
}

}  // namespace
}  // namespace n3mapping

int main(int argc, char** argv)
{
    using namespace n3mapping;

    const std::vector<std::string> app_args =
        rclcpp::init_and_remove_ros_arguments(argc, const_cast<const char* const*>(argv));
    std::vector<char*> app_argv;
    app_argv.reserve(app_args.size());
    for (const auto& arg : app_args) {
        app_argv.push_back(const_cast<char*>(arg.c_str()));
    }

    Options options;
    if (!parseArgs(static_cast<int>(app_argv.size()), app_argv.data(), &options)) {
        rclcpp::shutdown();
        return 1;
    }

    N3MappingCore map_core(makePublisherConfig());
    if (!map_core.loadMap(options.map_path)) {
        std::cerr << "Failed to load map: " << options.map_path << "\n";
        rclcpp::shutdown();
        return 1;
    }

    auto keyframes = map_core.getAllKeyframes();
    keyframes.erase(std::remove_if(keyframes.begin(), keyframes.end(), [](const Keyframe::Ptr& kf) {
        return !kf || !kf->cloud || kf->cloud->empty();
    }), keyframes.end());
    std::sort(keyframes.begin(), keyframes.end(), [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) {
        return a->id < b->id;
    });
    if (keyframes.empty()) {
        std::cerr << "Loaded map has no keyframes with point clouds\n";
        rclcpp::shutdown();
        return 1;
    }

    Keyframe::Ptr query_kf;
    if (options.query_id >= 0) {
        const auto it = std::find_if(keyframes.begin(), keyframes.end(), [&](const Keyframe::Ptr& kf) {
            return kf->id == options.query_id;
        });
        if (it == keyframes.end()) {
            std::cerr << "query_id not found or has empty cloud: " << options.query_id << "\n";
            rclcpp::shutdown();
            return 1;
        }
        query_kf = *it;
    } else {
        query_kf = keyframes[keyframes.size() / 2];
    }

    auto query_cloud = perturbCloud(query_kf->cloud, options, static_cast<std::uint32_t>(1000 + query_kf->id));
    if (!query_cloud || query_cloud->empty()) {
        std::cerr << "Synthetic query cloud is empty after dropout\n";
        rclcpp::shutdown();
        return 1;
    }

    const Eigen::Isometry3d T_map_odom_fake = makeFakeMapOdom(options);
    const Eigen::Isometry3d T_odom_lidar = T_map_odom_fake.inverse() * query_kf->pose_optimized;

    auto node = std::make_shared<rclcpp::Node>("n3mapping_synthetic_relocalization_publisher");
    auto cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>(options.cloud_topic, 10);
    auto odom_pub = node->create_publisher<nav_msgs::msg::Odometry>(options.odom_topic, 10);

    RCLCPP_INFO(node->get_logger(),
                "Publishing synthetic relocalization query: map=%s query_id=%ld points=%zu cloud_topic=%s odom_topic=%s repeat=%d",
                options.map_path.c_str(),
                query_kf->id,
                query_cloud->size(),
                options.cloud_topic.c_str(),
                options.odom_topic.c_str(),
                options.repeat);

    rclcpp::Rate rate(options.rate_hz);
    int published = 0;
    while (rclcpp::ok() && (options.repeat == 0 || published < options.repeat)) {
        const rclcpp::Time stamp = node->now();

        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*query_cloud, cloud_msg);
        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = "body";

        auto odom_msg = makeOdomMsg(T_odom_lidar, stamp);
        cloud_pub->publish(cloud_msg);
        odom_pub->publish(odom_msg);

        rclcpp::spin_some(node);
        rate.sleep();
        ++published;
    }

    rclcpp::shutdown();
    return 0;
}
