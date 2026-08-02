#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
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

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, query_pose, options, 7U, nullptr, &visibility);
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
    EXPECT_TRUE(visibility.raycast_enabled);
    EXPECT_EQ(visibility.input_map_points, world->size());
    EXPECT_GT(visibility.same_ray_occluded_points, 0U);
    EXPECT_EQ(visibility.output_points, query->size());
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(visibility));
}

TEST(SyntheticRelocalizationTest, RaycastFovEnvelopeWrapsAcrossZeroAzimuth)
{
    auto world = pcl::make_shared<Cloud>();
    auto reference = pcl::make_shared<Cloud>();
    auto addPolar = [](Cloud* cloud, double range, double azimuth_deg, float intensity) {
        const double azimuth = azimuth_deg * M_PI / 180.0;
        pcl::PointXYZI point;
        point.x = static_cast<float>(range * std::cos(azimuth));
        point.y = static_cast<float>(range * std::sin(azimuth));
        point.z = 0.0f;
        point.intensity = intensity;
        cloud->push_back(point);
    };
    addPolar(world.get(), 5.0, 355.0, 1.0f);
    addPolar(world.get(), 5.0, 5.0, 2.0f);
    addPolar(world.get(), 5.0, 180.0, 3.0f);
    for (double azimuth : {350.0, 352.0, 354.0, 356.0, 0.0, 4.0, 6.0, 8.0, 10.0}) {
        addPolar(reference.get(), 4.0, azimuth, 10.0f);
    }

    synthetic::QuerySynthesisOptions options;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.raycast_azimuth_resolution_deg = 1.0;
    options.raycast_vertical_resolution_deg = 1.0;
    options.occlusion_dilation_bins = 0;

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, Eigen::Isometry3d::Identity(), options, 11U, reference, &visibility);

    ASSERT_EQ(query->size(), 2U);
    bool saw_355 = false;
    bool saw_5 = false;
    bool saw_back = false;
    for (const auto& point : query->points) {
        saw_355 = saw_355 || std::abs(point.intensity - 1.0f) < 1e-6f;
        saw_5 = saw_5 || std::abs(point.intensity - 2.0f) < 1e-6f;
        saw_back = saw_back || std::abs(point.intensity - 3.0f) < 1e-6f;
    }
    EXPECT_TRUE(saw_355);
    EXPECT_TRUE(saw_5);
    EXPECT_FALSE(saw_back);
    EXPECT_FALSE(visibility.envelope_azimuth_full);
    EXPECT_LT(visibility.envelope_azimuth_start_deg, 360.0);
    EXPECT_LT(visibility.envelope_azimuth_span_deg, 30.0);
}

TEST(SyntheticRelocalizationTest, RaycastRenderingAndVisibilityStatsAreDeterministic)
{
    const auto world = makeWorldScene();
    synthetic::QuerySynthesisOptions options;
    options.dropout = 0.25;
    options.noise_sigma = 0.01;
    options.range_min = 0.5;
    options.range_max = 20.0;
    options.raycast_azimuth_resolution_deg = 2.0;
    options.raycast_vertical_resolution_deg = 2.0;

    synthetic::QueryVisibilityStats first_stats;
    synthetic::QueryVisibilityStats second_stats;
    const auto first = synthetic::synthesizeBodyCloudFromMapCloud(
        world, makePose(2.0, 0.0, 0.1), options, 12345U, nullptr, &first_stats);
    const auto second = synthetic::synthesizeBodyCloudFromMapCloud(
        world, makePose(2.0, 0.0, 0.1), options, 12345U, nullptr, &second_stats);

    ASSERT_EQ(first->size(), second->size());
    EXPECT_EQ(first_stats.input_map_points, second_stats.input_map_points);
    EXPECT_EQ(first_stats.range_eligible_points, second_stats.range_eligible_points);
    EXPECT_EQ(first_stats.fov_eligible_points, second_stats.fov_eligible_points);
    EXPECT_EQ(first_stats.occupied_ray_bins, second_stats.occupied_ray_bins);
    EXPECT_EQ(first_stats.same_ray_occluded_points, second_stats.same_ray_occluded_points);
    EXPECT_EQ(first_stats.dropout_suppressed_points, second_stats.dropout_suppressed_points);
    EXPECT_EQ(first_stats.occlusion_suppressed_bins, second_stats.occlusion_suppressed_bins);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(first_stats));
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(second_stats));
    EXPECT_EQ(synthetic::fingerprintPointSequenceFNV1a64(*first),
              synthetic::fingerprintPointSequenceFNV1a64(*second));
    for (std::size_t i = 0; i < first->size(); ++i) {
        EXPECT_FLOAT_EQ(first->points[i].x, second->points[i].x);
        EXPECT_FLOAT_EQ(first->points[i].y, second->points[i].y);
        EXPECT_FLOAT_EQ(first->points[i].z, second->points[i].z);
        EXPECT_FLOAT_EQ(first->points[i].intensity, second->points[i].intensity);
    }
}

TEST(SyntheticRelocalizationTest, RangeAndFovFilteringCanProduceExplicitEmptyVisibility)
{
    auto world = pcl::make_shared<Cloud>();
    auto add = [&](double x, double y, double z) {
        pcl::PointXYZI point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.z = static_cast<float>(z);
        world->push_back(point);
    };
    add(50.0, 0.0, 0.0);   // Outside range.
    add(-2.0, 0.0, 0.0);   // Inside range, behind a forward-facing FOV.

    synthetic::QuerySynthesisOptions options;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.fov_azimuth_deg = 90.0;
    options.fov_vertical_deg = 60.0;
    options.raycast_azimuth_resolution_deg = 1.0;
    options.raycast_vertical_resolution_deg = 1.0;

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, Eigen::Isometry3d::Identity(), options, 21U, nullptr, &visibility);

    EXPECT_TRUE(query->empty());
    EXPECT_EQ(visibility.input_map_points, 2U);
    EXPECT_EQ(visibility.finite_points, 2U);
    EXPECT_EQ(visibility.range_eligible_points, 1U);
    EXPECT_EQ(visibility.fov_eligible_points, 0U);
    EXPECT_EQ(visibility.occupied_ray_bins, 0U);
    EXPECT_EQ(visibility.output_points, 0U);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(visibility));
}

TEST(SyntheticRelocalizationTest, NeighborDilationSuppressesBackgroundWithoutDeletingRemoteSurface)
{
    auto world = pcl::make_shared<Cloud>();
    auto addPolar = [&](double range, double azimuth_deg, float intensity) {
        const double azimuth = azimuth_deg * M_PI / 180.0;
        pcl::PointXYZI point;
        point.x = static_cast<float>(range * std::cos(azimuth));
        point.y = static_cast<float>(range * std::sin(azimuth));
        point.z = 0.0f;
        point.intensity = intensity;
        world->push_back(point);
    };
    addPolar(2.0, 0.0, 1.0f);    // Foreground.
    addPolar(5.0, 1.2, 2.0f);    // Adjacent-bin background: should be suppressed.
    addPolar(5.0, 10.2, 3.0f);   // Remote surface: outside dilation neighborhood.

    synthetic::QuerySynthesisOptions options;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.raycast_azimuth_resolution_deg = 1.0;
    options.raycast_vertical_resolution_deg = 1.0;
    options.occlusion_dilation_bins = 1;
    options.occlusion_depth_tolerance_m = 0.1;

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, Eigen::Isometry3d::Identity(), options, 22U, nullptr, &visibility);

    bool saw_foreground = false;
    bool saw_background = false;
    bool saw_remote = false;
    for (const auto& point : query->points) {
        saw_foreground = saw_foreground || std::abs(point.intensity - 1.0f) < 1e-6f;
        saw_background = saw_background || std::abs(point.intensity - 2.0f) < 1e-6f;
        saw_remote = saw_remote || std::abs(point.intensity - 3.0f) < 1e-6f;
    }
    EXPECT_TRUE(saw_foreground);
    EXPECT_FALSE(saw_background);
    EXPECT_TRUE(saw_remote);
    EXPECT_EQ(visibility.fov_eligible_points, 3U);
    EXPECT_EQ(visibility.occupied_ray_bins, 3U);
    EXPECT_EQ(visibility.occlusion_suppressed_bins, 1U);
    EXPECT_EQ(visibility.output_points, 2U);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(visibility));
}

TEST(SyntheticRelocalizationTest, RaycastDisabledPreservesLegacyPointOrderAndConservation)
{
    auto world = pcl::make_shared<Cloud>();
    for (int i = 0; i < 3; ++i) {
        pcl::PointXYZI point;
        point.x = static_cast<float>(1.0 + i);
        point.y = static_cast<float>(0.1 * i);
        point.z = 0.0f;
        point.intensity = static_cast<float>(10 + i);
        world->push_back(point);
    }

    synthetic::QuerySynthesisOptions options;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.raycast_azimuth_resolution_deg = 0.0;
    options.raycast_vertical_resolution_deg = 0.0;

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, Eigen::Isometry3d::Identity(), options, 23U, nullptr, &visibility);

    ASSERT_EQ(query->size(), world->size());
    EXPECT_FALSE(visibility.raycast_enabled);
    EXPECT_EQ(visibility.input_map_points, 3U);
    EXPECT_EQ(visibility.finite_points, 3U);
    EXPECT_EQ(visibility.range_eligible_points, 3U);
    EXPECT_EQ(visibility.fov_eligible_points, 3U);
    EXPECT_EQ(visibility.azimuth_bins, 0U);
    EXPECT_EQ(visibility.vertical_bins, 0U);
    EXPECT_EQ(visibility.output_points, 3U);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(visibility));
    EXPECT_EQ(synthetic::fingerprintPointSequenceFNV1a64(*world),
              synthetic::fingerprintPointSequenceFNV1a64(*query));
}

TEST(SyntheticRelocalizationTest, NonFiniteMapPointsAndEmptyMapsRemainFiniteSafe)
{
    auto world = pcl::make_shared<Cloud>();
    pcl::PointXYZI finite;
    finite.x = 2.0f;
    finite.y = 0.0f;
    finite.z = 0.0f;
    finite.intensity = 1.0f;
    world->push_back(finite);
    pcl::PointXYZI nonfinite;
    nonfinite.x = std::numeric_limits<float>::quiet_NaN();
    nonfinite.y = 0.0f;
    nonfinite.z = 0.0f;
    nonfinite.intensity = 2.0f;
    world->push_back(nonfinite);

    synthetic::QuerySynthesisOptions options;
    options.range_min = 0.1;
    options.range_max = 10.0;
    options.raycast_azimuth_resolution_deg = 1.0;
    options.raycast_vertical_resolution_deg = 1.0;

    synthetic::QueryVisibilityStats visibility;
    const auto query = synthetic::synthesizeBodyCloudFromMapCloud(
        world, Eigen::Isometry3d::Identity(), options, 24U, nullptr, &visibility);
    ASSERT_EQ(query->size(), 1U);
    EXPECT_TRUE(std::isfinite(query->front().x));
    EXPECT_EQ(visibility.input_map_points, 2U);
    EXPECT_EQ(visibility.finite_points, 1U);
    EXPECT_EQ(visibility.range_eligible_points, 1U);
    EXPECT_EQ(visibility.fov_eligible_points, 1U);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(visibility));

    synthetic::QueryVisibilityStats empty_visibility;
    const auto empty_world = pcl::make_shared<Cloud>();
    const auto empty_query = synthetic::synthesizeBodyCloudFromMapCloud(
        empty_world, Eigen::Isometry3d::Identity(), options, 25U, nullptr, &empty_visibility);
    EXPECT_TRUE(empty_query->empty());
    EXPECT_EQ(empty_visibility.input_map_points, 0U);
    EXPECT_EQ(empty_visibility.finite_points, 0U);
    EXPECT_TRUE(synthetic::queryVisibilityStatsConserved(empty_visibility));
    EXPECT_EQ(synthetic::fingerprintPointSequenceFNV1a64(*empty_query),
              14695981039346656037ULL);
}

TEST(SyntheticRelocalizationTest, QueryPoseManifestParsesStrictStandaloneAndEpisodeSchemas)
{
    {
        std::istringstream csv(
            "query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
            "7,1.0,-2.0,0.5,3.0,-4.0,90.0\n");
        synthetic::QueryPoseManifest manifest;
        std::string error;
        ASSERT_TRUE(synthetic::parseQueryPoseManifestCsv(csv, &manifest, &error)) << error;
        ASSERT_FALSE(manifest.has_episode_columns);
        ASSERT_EQ(manifest.entries.size(), 1U);
        EXPECT_EQ(manifest.entries[0].query_id, 7);
        EXPECT_EQ(manifest.entries[0].episode_id, 7);
        EXPECT_EQ(manifest.entries[0].frame_index, 0);
        EXPECT_NEAR(manifest.entries[0].T_map_lidar.translation().x(), 1.0, 1e-12);
        EXPECT_NEAR(manifest.entries[0].T_map_lidar.translation().y(), -2.0, 1e-12);
    }
    {
        std::istringstream csv(
            "query_id,episode_id,frame_index,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
            "10,3,0,0,0,0,0,0,0\n"
            "11,3,1,0.1,0,0,0,0,1\n"
            "12,4,0,1,2,3,4,5,6\n");
        synthetic::QueryPoseManifest manifest;
        std::string error;
        ASSERT_TRUE(synthetic::parseQueryPoseManifestCsv(csv, &manifest, &error)) << error;
        ASSERT_TRUE(manifest.has_episode_columns);
        ASSERT_EQ(manifest.entries.size(), 3U);
        EXPECT_EQ(manifest.entries[1].episode_id, 3);
        EXPECT_EQ(manifest.entries[1].frame_index, 1);
        EXPECT_EQ(manifest.entries[2].episode_id, 4);
    }
}

TEST(SyntheticRelocalizationTest, QueryPoseManifestRejectsAmbiguousOrNonDeterministicRows)
{
    {
        std::istringstream csv(
            "query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
            "1,0,0,0,0,0,nan\n");
        synthetic::QueryPoseManifest manifest;
        std::string error;
        EXPECT_FALSE(synthetic::parseQueryPoseManifestCsv(csv, &manifest, &error));
        EXPECT_NE(error.find("invalid_yaw_deg"), std::string::npos);
    }
    {
        std::istringstream csv(
            "query_id,episode_id,frame_index,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
            "1,9,0,0,0,0,0,0,0\n"
            "2,9,2,0,0,0,0,0,0\n");
        synthetic::QueryPoseManifest manifest;
        std::string error;
        EXPECT_FALSE(synthetic::parseQueryPoseManifestCsv(csv, &manifest, &error));
        EXPECT_NE(error.find("frame_index_expected_1_got_2"), std::string::npos);
    }
    {
        std::istringstream csv(
            "query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
            "5,0,0,0,0,0,0\n"
            "5,1,0,0,0,0,0\n");
        synthetic::QueryPoseManifest manifest;
        std::string error;
        EXPECT_FALSE(synthetic::parseQueryPoseManifestCsv(csv, &manifest, &error));
        EXPECT_NE(error.find("duplicate_query_id"), std::string::npos);
    }
}

TEST(SyntheticRelocalizationTest, EpisodeStateClearsTrackLossAndLocksOnNextFrameReentry)
{
    auto transition = synthetic::advanceEpisodeFrameState(
        false, false, true, true);
    EXPECT_FALSE(transition.locked_before_frame);
    EXPECT_TRUE(transition.locked_after_frame);
    EXPECT_FALSE(transition.ever_locked_before_frame);
    EXPECT_TRUE(transition.ever_locked_after_frame);
    EXPECT_TRUE(transition.localization_attempted);
    EXPECT_FALSE(transition.tracking_processed);
    EXPECT_TRUE(transition.lock_event);
    EXPECT_FALSE(transition.tracking_loss_event);
    EXPECT_EQ(transition.stage, synthetic::EpisodeFrameStage::INITIAL_LOCK);
    EXPECT_STREQ(synthetic::episodeLockEventTypeName(transition.stage), "initial_lock");

    transition = synthetic::advanceEpisodeFrameState(
        transition.locked_after_frame,
        transition.ever_locked_after_frame,
        false,
        false);
    EXPECT_TRUE(transition.locked_before_frame);
    EXPECT_FALSE(transition.locked_after_frame);
    EXPECT_TRUE(transition.ever_locked_before_frame);
    EXPECT_TRUE(transition.ever_locked_after_frame);
    EXPECT_FALSE(transition.localization_attempted);
    EXPECT_TRUE(transition.tracking_processed);
    EXPECT_FALSE(transition.lock_event);
    EXPECT_TRUE(transition.tracking_loss_event);
    EXPECT_EQ(transition.stage, synthetic::EpisodeFrameStage::TRACKING_LOSS);

    transition = synthetic::advanceEpisodeFrameState(
        transition.locked_after_frame,
        transition.ever_locked_after_frame,
        true,
        true);
    EXPECT_FALSE(transition.locked_before_frame);
    EXPECT_TRUE(transition.locked_after_frame);
    EXPECT_TRUE(transition.ever_locked_before_frame);
    EXPECT_TRUE(transition.localization_attempted);
    EXPECT_FALSE(transition.tracking_processed);
    EXPECT_TRUE(transition.lock_event);
    EXPECT_FALSE(transition.tracking_loss_event);
    EXPECT_EQ(transition.stage, synthetic::EpisodeFrameStage::REENTRY_LOCK);
    EXPECT_STREQ(synthetic::episodeLockEventTypeName(transition.stage), "reentry_lock");
}

TEST(SyntheticRelocalizationTest, EpisodeStateAuditsSameFrameTrackingLossAndReentryLock)
{
    const auto ordinary_tracking = synthetic::advanceEpisodeFrameState(
        true, true, true, false);
    EXPECT_TRUE(ordinary_tracking.locked_before_frame);
    EXPECT_TRUE(ordinary_tracking.locked_after_frame);
    EXPECT_FALSE(ordinary_tracking.localization_attempted);
    EXPECT_TRUE(ordinary_tracking.tracking_processed);
    EXPECT_FALSE(ordinary_tracking.lock_event);
    EXPECT_FALSE(ordinary_tracking.tracking_loss_event);
    EXPECT_EQ(ordinary_tracking.stage, synthetic::EpisodeFrameStage::TRACKING);

    const auto transition = synthetic::advanceEpisodeFrameState(
        true, true, true, true);

    EXPECT_TRUE(transition.locked_before_frame);
    EXPECT_TRUE(transition.locked_after_frame);
    EXPECT_TRUE(transition.ever_locked_before_frame);
    EXPECT_TRUE(transition.ever_locked_after_frame);
    EXPECT_TRUE(transition.localization_attempted);
    EXPECT_TRUE(transition.tracking_processed);
    EXPECT_TRUE(transition.lock_event);
    EXPECT_FALSE(transition.tracking_loss_event);
    EXPECT_EQ(transition.stage, synthetic::EpisodeFrameStage::REENTRY_LOCK);
    EXPECT_STREQ(synthetic::episodeFrameStageName(transition.stage), "reentry_lock");
    EXPECT_STREQ(synthetic::episodeLockEventTypeName(transition.stage), "reentry_lock");
}

TEST(SyntheticRelocalizationTest, ProductQueryGenerationSupportUsesFrozenHundredPointBoundaries)
{
    const auto relaxed = synthetic::queryGenerationThresholds(false);
    EXPECT_EQ(relaxed.minimum_query_points, 1U);
    EXPECT_EQ(relaxed.minimum_occupied_ray_bins, 1U);
    EXPECT_EQ(
        synthetic::evaluateQueryGenerationSupport(1U, 0U, false, relaxed),
        synthetic::QueryGenerationSupport::SUFFICIENT);

    const auto product = synthetic::queryGenerationThresholds(true);
    EXPECT_EQ(product.minimum_query_points, 100U);
    EXPECT_EQ(product.minimum_occupied_ray_bins, 100U);
    EXPECT_EQ(
        synthetic::evaluateQueryGenerationSupport(99U, 100U, true, product),
        synthetic::QueryGenerationSupport::INSUFFICIENT_QUERY_POINTS);
    EXPECT_EQ(
        synthetic::evaluateQueryGenerationSupport(100U, 99U, true, product),
        synthetic::QueryGenerationSupport::INSUFFICIENT_OCCUPIED_RAY_BINS);
    EXPECT_EQ(
        synthetic::evaluateQueryGenerationSupport(100U, 100U, true, product),
        synthetic::QueryGenerationSupport::SUFFICIENT);
}

TEST(SyntheticRelocalizationTest, PerQueryCsvPrecisionRoundTripsLargePoseCoordinates)
{
    EXPECT_EQ(synthetic::kEvaluationCsvDoublePrecision, 17);
    const std::vector<double> pose_values = {
        123.45678901234567,
        -234.56789012345678,
        101.00000000000013,
        12.345678901234567,
        -23.456789012345678,
        179.99999999999997,
    };

    std::ostringstream serialized;
    serialized << std::setprecision(synthetic::kEvaluationCsvDoublePrecision);
    for (std::size_t i = 0; i < pose_values.size(); ++i) {
        if (i > 0) serialized << ',';
        serialized << pose_values[i];
    }

    std::istringstream input(serialized.str());
    for (const double expected : pose_values) {
        std::string field;
        ASSERT_TRUE(std::getline(input, field, ','));
        const double decoded = std::stod(field);
        EXPECT_LE(std::abs(decoded - expected), 1e-12);
    }
}

}  // namespace test
}  // namespace n3mapping
