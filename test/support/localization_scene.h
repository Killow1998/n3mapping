#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <pcl/common/transforms.h>
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping::test::synthetic_scene {
using Cloud = core::LioFrame::PointCloud;

inline Eigen::Isometry3d makePose(double x, double y, double yaw)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(x, y, 0.0);
    pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return pose;
}

inline Cloud::Ptr makeWorldScene()
{
    auto cloud = pcl::make_shared<Cloud>();

    auto addPoint = [&](double x, double y, double z, float intensity) {
        pcl::PointXYZI point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.z = static_cast<float>(z);
        point.intensity = intensity;
        cloud->push_back(point);
    };

    for (double x = -4.0; x <= 14.0; x += 0.25) {
        for (double z = -0.2; z <= 2.2; z += 0.35) {
            addPoint(x, -3.0, z, 40.0f);
            addPoint(x, 3.0, z, 45.0f);
        }
    }

    for (double x = -4.0; x <= 14.0; x += 0.35) {
        for (double y = -3.0; y <= 3.0; y += 0.35) {
            addPoint(x, y, -0.25, 20.0f);
        }
    }

    const std::vector<Eigen::Vector2d> pillars = {
        {1.0, -1.2}, {4.3, 1.1}, {7.0, -1.8}, {10.5, 1.7}
    };
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        for (double a = 0.0; a < 2.0 * M_PI; a += 0.25) {
            for (double z = -0.2; z <= 2.6 + 0.2 * static_cast<double>(i); z += 0.25) {
                addPoint(pillars[i].x() + 0.22 * std::cos(a),
                         pillars[i].y() + 0.22 * std::sin(a),
                         z,
                         static_cast<float>(80 + 10 * i));
            }
        }
    }

    for (double z = 0.2; z <= 1.8; z += 0.2) {
        addPoint(5.8, -2.95, z, 150.0f);
        addPoint(5.8, -2.5, z, 150.0f);
        addPoint(6.4, 2.95, z, 180.0f);
    }

    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

inline Cloud::Ptr synthesizeBodyCloud(const Cloud::Ptr& world_scene,
                               const Eigen::Isometry3d& T_map_lidar,
                               double range_max,
                               double dropout_ratio,
                               double noise_sigma,
                               std::uint32_t seed)
{
    auto body = pcl::make_shared<Cloud>();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    const Eigen::Isometry3d T_lidar_map = T_map_lidar.inverse();

    for (const auto& point_map : world_scene->points) {
        const Eigen::Vector3d p_map(point_map.x, point_map.y, point_map.z);
        const Eigen::Vector3d p_body = T_lidar_map * p_map;
        const double range = p_body.norm();
        if (range < 0.5 || range > range_max) {
            continue;
        }
        if (p_body.z() < -2.0 || p_body.z() > 4.0) {
            continue;
        }
        if (keep_dist(rng) < dropout_ratio) {
            continue;
        }

        pcl::PointXYZI out;
        out.x = static_cast<float>(p_body.x() + noise(rng));
        out.y = static_cast<float>(p_body.y() + noise(rng));
        out.z = static_cast<float>(p_body.z() + noise(rng));
        out.intensity = point_map.intensity;
        body->push_back(out);
    }

    body->width = static_cast<std::uint32_t>(body->size());
    body->height = 1;
    body->is_dense = true;
    return body;
}

inline core::LioFrame makeFrame(std::int64_t stamp_nsec,
                         const Eigen::Isometry3d& pose,
                         const Cloud::Ptr& body_cloud)
{
    core::LioFrame frame;
    frame.stamp.nsec = stamp_nsec;
    frame.T_world_lidar = pose;
    frame.undistorted_cloud = body_cloud;
    frame.pose_valid = true;
    return frame;
}

inline Config makeSyntheticRelocConfig()
{
    Config config;
    // The synthetic scene is fixed, so the platform never appears to move and
    // the static-start guard would keep the map empty. This suite is testing
    // relocalization against a map, not when mapping should begin.
    config.mapping_static_start_guard_enable = false;
    config.keyframe_distance_threshold = 1.0;
    config.keyframe_angle_threshold = 0.3;
    config.gicp_downsampling_resolution = 0.15;
    config.gicp_max_correspondence_distance = 1.5;
    config.gicp_max_iterations = 60;
    config.gicp_fitness_threshold = 0.8;
    config.gicp_submap_size = 1;
    config.sc_num_exclude_recent = 0;
    config.sc_num_candidates = 8;
    config.rhpd_enabled = true;
    config.rhpd_dist_threshold = 100.0;
    config.rhpd_num_candidates = 8;
    config.rhpd_preselect_candidates = 50;
    config.rhpd_submap_kf_radius = 0;
    config.rhpd_submap_voxel_size = 0.0;
    config.rhpd_yaw_hypotheses = 4;
    config.reloc_num_candidates = 8;
    config.reloc_temporal_window_size = 1;
    config.reloc_lock_log_likelihood_threshold = -100.0;
    config.reloc_lock_min_winner_streak = 1;
    config.reloc_lock_min_converged_updates = 1;
    config.reloc_lock_min_margin = 0.0;
    config.reloc_min_confidence = 0.0;
    config.reloc_min_inlier_ratio = 0.0;
    config.reloc_static_agg_enable = false;
    config.reloc_ambiguity_min_basin_separation = 1000.0;
    config.global_map_voxel_size = 0.0;
    return config;
}


}  // namespace n3mapping::test::synthetic_scene
