#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"
#include "n3mapping/synthetic_relocalization_query.h"

namespace n3mapping {
namespace test {
namespace {

using Cloud = core::LioFrame::PointCloud;

Eigen::Isometry3d makePose(double x, double y, double yaw)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(x, y, 0.0);
    pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return pose;
}

Cloud::Ptr makeWorldScene()
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

Cloud::Ptr synthesizeBodyCloud(const Cloud::Ptr& world_scene,
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

core::LioFrame makeFrame(std::int64_t stamp_nsec,
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

Config makeSyntheticRelocConfig()
{
    Config config;
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

double yawError(const Eigen::Isometry3d& estimated, const Eigen::Isometry3d& expected)
{
    const Eigen::Matrix3d R = expected.rotation().transpose() * estimated.rotation();
    return std::abs(Eigen::AngleAxisd(R).angle());
}

}  // namespace

TEST(SyntheticRelocalizationTest, CoreRelocalizesBodyCloudWithFakeOdomFrame)
{
    Config config = makeSyntheticRelocConfig();
    const auto world_scene = makeWorldScene();
    const std::vector<Eigen::Isometry3d> trajectory = {
        makePose(0.0, 0.0, 0.0),
        makePose(1.5, 0.0, 0.02),
        makePose(3.0, 0.0, -0.03),
        makePose(4.5, 0.2, 0.05),
        makePose(6.0, 0.2, 0.0),
        makePose(7.5, -0.1, -0.04),
        makePose(9.0, -0.1, 0.03)
    };

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "n3mapping_synthetic_relocalization";
    std::filesystem::create_directories(dir);
    const std::filesystem::path map_path = dir / "synthetic.pbstream";

    {
        N3MappingCore mapper(config);
        for (std::size_t i = 0; i < trajectory.size(); ++i) {
            auto body_cloud = synthesizeBodyCloud(world_scene, trajectory[i], 12.0, 0.0, 0.0, 100U + i);
            ASSERT_GT(body_cloud->size(), 200U);
            auto output = mapper.processMappingFrame(makeFrame(
                static_cast<std::int64_t>(i + 1) * 1000000000LL, trajectory[i], body_cloud));
            ASSERT_TRUE(output.success);
            ASSERT_TRUE(output.accepted_keyframe);
        }
        ASSERT_TRUE(mapper.saveMap(map_path.string()));
    }

    N3MappingCore localizer(config);
    ASSERT_TRUE(localizer.loadMap(map_path.string()));

    const std::size_t query_index = 4;
    const Eigen::Isometry3d T_map_lidar_gt = trajectory[query_index];
    auto query_cloud = synthesizeBodyCloud(world_scene, T_map_lidar_gt, 12.0, 0.35, 0.01, 900U);
    ASSERT_GT(query_cloud->size(), 150U);

    Eigen::Isometry3d T_map_odom_fake = makePose(20.0, -8.0, M_PI / 2.0);
    T_map_odom_fake.translation().z() = 1.0;
    const Eigen::Isometry3d T_odom_lidar_input = T_map_odom_fake.inverse() * T_map_lidar_gt;

    const auto output = localizer.processLocalizationFrame(
        makeFrame(10000000000LL, T_odom_lidar_input, query_cloud));

    ASSERT_TRUE(output.success);
    EXPECT_TRUE(output.relocalization_locked);
    EXPECT_GE(output.matched_keyframe_id, 0);
    const double map_error = (output.T_world_lidar.translation() - T_map_lidar_gt.translation()).norm();
    const double odom_passthrough_error =
        (T_odom_lidar_input.translation() - T_map_lidar_gt.translation()).norm();
    EXPECT_GT(odom_passthrough_error, 10.0);
    EXPECT_LT(map_error, 1.0);
    EXPECT_LT(yawError(output.T_world_lidar, T_map_lidar_gt), 0.2);

    std::filesystem::remove(map_path);
    std::filesystem::remove_all(dir);
}

TEST(SyntheticRelocalizationTest, RaycastQuerySynthesisSupportsNewPoseAndFov)
{
    auto world = pcl::make_shared<Cloud>();
    auto add = [&](double x, double y, double z, float intensity) {
        pcl::PointXYZI p;
        p.x = static_cast<float>(x);
        p.y = static_cast<float>(y);
        p.z = static_cast<float>(z);
        p.intensity = intensity;
        world->push_back(p);
    };
    add(5.0, 0.0, 0.0, 1.0f);
    add(6.0, 0.0, 0.0, 2.0f);
    add(-5.0, 0.0, 0.0, 3.0f);
    add(4.0, 1.0, 1.0, 4.0f);
    world->width = static_cast<std::uint32_t>(world->size());
    world->height = 1;
    world->is_dense = true;

    synthetic::PoseJitterOptions jitter;
    jitter.z_m = 0.5;
    jitter.roll_pitch_deg = 2.0;
    std::mt19937 rng(42);
    const auto query_pose = synthetic::applyUniformPoseJitter(makePose(0.0, 0.0, 0.0), jitter, &rng);
    EXPECT_NEAR(query_pose.translation().x(), 0.0, 1e-9);
    EXPECT_NEAR(query_pose.translation().y(), 0.0, 1e-9);
    EXPECT_NE(query_pose.translation().z(), 0.0);

    synthetic::QuerySynthesisOptions options;
    options.dropout = 0.0;
    options.noise_sigma = 0.0;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.fov_azimuth_deg = 120.0;
    options.fov_vertical_deg = 90.0;
    options.raycast_azimuth_resolution_deg = 10.0;
    options.raycast_vertical_resolution_deg = 10.0;

    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(world, query_pose, options, 7U);
    ASSERT_FALSE(query->empty());
    bool saw_front = false;
    bool saw_back = false;
    bool saw_occluded_wall = false;
    for (const auto& p : query->points) {
        if (p.x > 0.0f) saw_front = true;
        if (p.x < 0.0f) saw_back = true;
        if (std::abs(p.intensity - 2.0f) < 1e-3f) saw_occluded_wall = true;
    }
    EXPECT_TRUE(saw_front);
    EXPECT_FALSE(saw_back);
    EXPECT_FALSE(saw_occluded_wall);
    EXPECT_LT(query->size(), world->size());
}

}  // namespace test
}  // namespace n3mapping
