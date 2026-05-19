#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "n3mapping/core/n3mapping_core.h"

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;

struct Options {
    std::string map_path;
    std::string world_frame = "map";
    int64_t query_id = -1;
    int repeat = 0;
    double rate_hz = 1.0;
    double dropout = 0.3;
    double noise_sigma = 0.02;
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
        << "  --world_frame FRAME       RViz frame id. Default: map\n"
        << "  --repeat N                Publish count. Default: 0 means until shutdown\n"
        << "  --rate_hz HZ              Publish rate. Default: 1\n"
        << "  --dropout R               Random point dropout ratio [0,1). Default: 0.3\n"
        << "  --noise_sigma M           XYZ Gaussian noise in meters. Default: 0.02\n"
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
        } else if (arg == "--world_frame") {
            if (const char* v = needValue(arg)) options->world_frame = v; else return false;
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

Config makeVisualizerConfig()
{
    Config config;
    config.gicp_max_iterations = 60;
    config.gicp_fitness_threshold = 0.8;
    config.gicp_max_correspondence_distance = 2.0;
    config.gicp_submap_size = 1;
    config.rhpd_enabled = true;
    config.rhpd_dist_threshold = 100.0;
    config.rhpd_num_candidates = 10;
    config.rhpd_preselect_candidates = 100;
    config.rhpd_yaw_hypotheses = 4;
    config.reloc_num_candidates = 10;
    config.reloc_temporal_window_size = 1;
    config.reloc_lock_log_likelihood_threshold = -100.0;
    config.reloc_lock_min_winner_streak = 1;
    config.reloc_lock_min_converged_updates = 1;
    config.reloc_lock_min_margin = 0.0;
    config.reloc_min_confidence = 0.0;
    config.reloc_min_inlier_ratio = 0.0;
    config.reloc_static_agg_enable = false;
    config.reloc_ambiguity_min_basin_separation = 1000.0;
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

Cloud::Ptr transformCloud(const Cloud::Ptr& cloud, const Eigen::Isometry3d& pose)
{
    auto output = pcl::make_shared<Cloud>();
    if (!cloud) return output;
    pcl::transformPointCloud(*cloud, *output, pose.matrix().cast<float>());
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = true;
    return output;
}

core::LioFrame makeFrame(const Cloud::Ptr& query_cloud, const Eigen::Isometry3d& T_odom_lidar)
{
    core::LioFrame frame;
    frame.stamp.nsec = 1;
    frame.T_world_lidar = T_odom_lidar;
    frame.undistorted_cloud = query_cloud;
    frame.pose_valid = true;
    return frame;
}

sensor_msgs::msg::PointCloud2 makeCloudMsg(const Cloud::Ptr& cloud,
                                           const std::string& frame_id,
                                           const rclcpp::Time& stamp)
{
    sensor_msgs::msg::PointCloud2 msg;
    if (cloud) {
        pcl::toROSMsg(*cloud, msg);
    }
    msg.header.frame_id = frame_id;
    msg.header.stamp = stamp;
    return msg;
}

geometry_msgs::msg::Point makePoint(const Eigen::Vector3d& p)
{
    geometry_msgs::msg::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
}

void setColor(std_msgs::msg::ColorRGBA* color, float r, float g, float b, float a)
{
    color->r = r;
    color->g = g;
    color->b = b;
    color->a = a;
}

visualization_msgs::msg::Marker makePoseArrow(const std::string& ns,
                                               int id,
                                               const std::string& frame_id,
                                               const rclcpp::Time& stamp,
                                               const Eigen::Isometry3d& pose,
                                               float r,
                                               float g,
                                               float b)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 1.4;
    marker.scale.y = 0.18;
    marker.scale.z = 0.18;
    setColor(&marker.color, r, g, b, 0.95f);
    marker.pose.position = makePoint(pose.translation());
    const Eigen::Quaterniond q(pose.rotation());
    marker.pose.orientation.w = q.w();
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    return marker;
}

visualization_msgs::msg::Marker makeText(const std::string& frame_id,
                                          const rclcpp::Time& stamp,
                                          const Eigen::Vector3d& position,
                                          const std::string& text)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "synthetic_relocalization_text";
    marker.id = 100;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.z = 1.0;
    setColor(&marker.color, 1.0f, 1.0f, 1.0f, 1.0f);
    marker.pose.position = makePoint(position);
    marker.text = text;
    return marker;
}

visualization_msgs::msg::Marker makeErrorLines(const std::string& frame_id,
                                                const rclcpp::Time& stamp,
                                                const Eigen::Isometry3d& before,
                                                const Eigen::Isometry3d& after,
                                                const Eigen::Isometry3d& gt)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "synthetic_relocalization_error_lines";
    marker.id = 200;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.08;
    setColor(&marker.color, 1.0f, 1.0f, 0.0f, 0.9f);
    marker.points.push_back(makePoint(before.translation()));
    marker.points.push_back(makePoint(gt.translation()));
    marker.points.push_back(makePoint(after.translation()));
    marker.points.push_back(makePoint(gt.translation()));
    return marker;
}

double yawDeg(const Eigen::Isometry3d& pose)
{
    const Eigen::Vector3d x = pose.rotation().col(0);
    return std::atan2(x.y(), x.x()) * 180.0 / M_PI;
}

double yawErrorDeg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b)
{
    double diff = (yawDeg(a) - yawDeg(b)) * M_PI / 180.0;
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    return std::abs(diff) * 180.0 / M_PI;
}

}  // namespace
}  // namespace n3mapping

int main(int argc, char** argv)
{
    using namespace n3mapping;

    rclcpp::init(argc, argv);
    Options options;
    if (!parseArgs(argc, argv, &options)) {
        rclcpp::shutdown();
        return 1;
    }

    N3MappingCore catalog(makeVisualizerConfig());
    if (!catalog.loadMap(options.map_path)) {
        std::cerr << "Failed to load map: " << options.map_path << "\n";
        rclcpp::shutdown();
        return 1;
    }

    auto keyframes = catalog.getAllKeyframes();
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

    const Eigen::Isometry3d T_map_lidar_gt = query_kf->pose_optimized;
    const Eigen::Isometry3d T_map_odom_fake = makeFakeMapOdom(options);
    const Eigen::Isometry3d T_odom_lidar = T_map_odom_fake.inverse() * T_map_lidar_gt;
    const Eigen::Isometry3d T_map_lidar_before = T_odom_lidar;

    N3MappingCore localizer(makeVisualizerConfig());
    if (!localizer.loadMap(options.map_path)) {
        std::cerr << "Failed to reload map for localization: " << options.map_path << "\n";
        rclcpp::shutdown();
        return 1;
    }
    const auto output = localizer.processLocalizationFrame(makeFrame(query_cloud, T_odom_lidar));
    const Eigen::Isometry3d T_map_lidar_after = output.T_world_lidar;

    auto global_map = catalog.buildGlobalMap();
    auto before_cloud = transformCloud(query_cloud, T_map_lidar_before);
    auto after_cloud = transformCloud(query_cloud, T_map_lidar_after);
    auto gt_cloud = transformCloud(query_cloud, T_map_lidar_gt);

    const double before_t = (T_map_lidar_before.translation() - T_map_lidar_gt.translation()).norm();
    const double after_t = (T_map_lidar_after.translation() - T_map_lidar_gt.translation()).norm();
    const double before_y = yawErrorDeg(T_map_lidar_before, T_map_lidar_gt);
    const double after_y = yawErrorDeg(T_map_lidar_after, T_map_lidar_gt);

    auto node = std::make_shared<rclcpp::Node>("n3mapping_synthetic_relocalization_visualizer");
    rclcpp::QoS qos(1);
    qos.transient_local();
    auto global_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/synthetic/global_map", qos);
    auto before_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/synthetic/query_before", qos);
    auto after_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/synthetic/query_after", qos);
    auto gt_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/n3mapping/synthetic/query_gt", qos);
    auto marker_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/n3mapping/synthetic/relocalization_markers", qos);

    RCLCPP_INFO(node->get_logger(),
                "Synthetic relocalization visualization query_id=%ld success=%d lock=%d matched=%ld before_t=%.3f after_t=%.3f before_yaw=%.3f after_yaw=%.3f",
                query_kf->id,
                output.success,
                output.relocalization_locked,
                output.matched_keyframe_id,
                before_t,
                after_t,
                before_y,
                after_y);
    RCLCPP_INFO(node->get_logger(), "RViz topics: /n3mapping/synthetic/global_map, query_before(red), query_after(green), query_gt(blue), relocalization_markers");

    rclcpp::Rate rate(options.rate_hz);
    int published = 0;
    while (rclcpp::ok() && (options.repeat == 0 || published < options.repeat)) {
        const auto stamp = node->now();

        global_pub->publish(makeCloudMsg(global_map, options.world_frame, stamp));
        before_pub->publish(makeCloudMsg(before_cloud, options.world_frame, stamp));
        after_pub->publish(makeCloudMsg(after_cloud, options.world_frame, stamp));
        gt_pub->publish(makeCloudMsg(gt_cloud, options.world_frame, stamp));

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(makePoseArrow("synthetic_pose_before", 0, options.world_frame, stamp, T_map_lidar_before, 1.0f, 0.1f, 0.1f));
        markers.markers.push_back(makePoseArrow("synthetic_pose_after", 1, options.world_frame, stamp, T_map_lidar_after, 0.1f, 1.0f, 0.1f));
        markers.markers.push_back(makePoseArrow("synthetic_pose_gt", 2, options.world_frame, stamp, T_map_lidar_gt, 0.1f, 0.4f, 1.0f));
        markers.markers.push_back(makeErrorLines(options.world_frame, stamp, T_map_lidar_before, T_map_lidar_after, T_map_lidar_gt));
        std::ostringstream text;
        text << "query_id=" << query_kf->id
             << " matched=" << output.matched_keyframe_id
             << " lock=" << output.relocalization_locked
             << "\nbefore: " << before_t << " m, " << before_y << " deg"
             << "\nafter:  " << after_t << " m, " << after_y << " deg";
        markers.markers.push_back(makeText(options.world_frame, stamp, T_map_lidar_gt.translation() + Eigen::Vector3d(0.0, 0.0, 2.5), text.str()));
        marker_pub->publish(markers);

        rclcpp::spin_some(node);
        rate.sleep();
        ++published;
    }

    rclcpp::shutdown();
    return 0;
}
