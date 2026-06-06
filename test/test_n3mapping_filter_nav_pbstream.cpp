#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3map.pb.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeSectorCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto add = [&](float x, float y, int count) {
        for (int i = 0; i < count; ++i) {
            pcl::PointXYZI point;
            point.x = x;
            point.y = y + static_cast<float>(i - count / 2) * 0.01f;
            point.z = static_cast<float>(i % 5) * 0.05f;
            point.intensity = 1.0f;
            cloud->push_back(point);
        }
    };
    add(2.0f, 0.0f, 50);    // front
    add(0.0f, 2.0f, 50);    // left side
    add(0.0f, -2.0f, 50);   // right side
    add(-2.0f, 0.0f, 50);   // rear sector
    auto addAngle = [&](double angle_deg, int count) {
        const double rad = angle_deg * M_PI / 180.0;
        for (int i = 0; i < count; ++i) {
            pcl::PointXYZI point;
            point.x = static_cast<float>(2.0 * std::cos(rad));
            (void)i;
            point.y = static_cast<float>(2.0 * std::sin(rad));
            point.z = 0.0f;
            point.intensity = static_cast<float>(angle_deg);
            cloud->push_back(point);
        }
    };
    addAngle(150.0, 10);     // outside default rear sector
    addAngle(-150.0, 10);    // outside default rear sector
    addAngle(157.5, 10);     // inclusive boundary, removed
    addAngle(-157.5, 10);    // inclusive boundary, removed
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr makeRangeSelfCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto add = [&](float x, float y, float z) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.intensity = 1.0f;
        cloud->push_back(point);
    };
    add(2.0f, 0.0f, 0.0f);   // kept
    add(4.0f, 0.0f, 0.0f);   // max-range removed
    add(0.25f, 0.0f, 0.0f);  // self-box removed
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr makeRearOnlyCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (int i = 0; i < 32; ++i) {
        pcl::PointXYZI point;
        point.x = -2.0f;
        point.y = static_cast<float>(i - 16) * 0.01f;
        point.z = 0.0f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

bool parseMap(const std::filesystem::path& path, N3Map* map) {
    std::ifstream ifs(path, std::ios::binary);
    return ifs.is_open() && map->ParseFromIstream(&ifs);
}

std::filesystem::path findFilterTool() {
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::filesystem::path dir = self.parent_path();
    const std::filesystem::path candidates[] = {
        dir / "n3mapping_filter_nav_pbstream",
        dir.parent_path() / "n3mapping_filter_nav_pbstream",
        std::filesystem::current_path() / "n3mapping_filter_nav_pbstream",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

bool hasDescriptorDifference(const N3Map& before, const N3Map& after) {
    if (before.keyframes_size() == 0 || after.keyframes_size() == 0) return false;
    const auto& lhs = before.keyframes(0);
    const auto& rhs = after.keyframes(0);
    if (lhs.sc_descriptor().values_size() != rhs.sc_descriptor().values_size() ||
        lhs.rhpd_descriptor().values_size() != rhs.rhpd_descriptor().values_size()) {
        return true;
    }
    double sc_delta = 0.0;
    for (int i = 0; i < lhs.sc_descriptor().values_size(); ++i) {
        sc_delta += std::abs(lhs.sc_descriptor().values(i) - rhs.sc_descriptor().values(i));
    }
    double rhpd_delta = 0.0;
    for (int i = 0; i < lhs.rhpd_descriptor().values_size(); ++i) {
        rhpd_delta += std::abs(lhs.rhpd_descriptor().values(i) - rhs.rhpd_descriptor().values(i));
    }
    return sc_delta > 1e-6 && rhpd_delta > 1e-6;
}

}  // namespace

TEST(N3MappingFilterNavPbstreamTest, RemovesOnlyRearSectorAndRecomputesDescriptors) {
    const auto tool = findFilterTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_filter_nav_pbstream executable not found";

    Config config;
    config.map_save_path = (std::filesystem::temp_directory_path() / "n3mapping_nav_filter_test").string();
    config.world_frame = "world";
    config.body_frame = "base_link";
    config.rhpd_submap_voxel_size = 0.0;
    std::filesystem::remove_all(config.map_save_path);
    std::filesystem::create_directories(config.map_save_path);

    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    GraphOptimizer optimizer(config);
    MapSerializer serializer(config);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = keyframes.addKeyframe(1.0, pose, makeSectorCloud());
    auto cloud = keyframes.getKeyframe(kf_id)->cloud;
    keyframes.updateDescriptor(kf_id, loop_detector.addDescriptor(kf_id, cloud));
    keyframes.getKeyframe(kf_id)->rhpd_descriptor = loop_detector.addRHPD(kf_id, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::filesystem::path input = std::filesystem::path(config.map_save_path) / "input.pbstream";
    const std::filesystem::path output = std::filesystem::path(config.map_save_path) / "n3map_nav_filtered.pbstream";
    ASSERT_TRUE(serializer.saveMap(input.string(), keyframes, loop_detector, optimizer));

    const std::filesystem::path debug_dir = std::filesystem::path(config.map_save_path) / "debug";
    const std::string command = tool.string() +
        " --input " + input.string() +
        " --output " + output.string() +
        " --debug-dir " + debug_dir.string();
    ASSERT_EQ(std::system(command.c_str()), 0);

    KeyframeManager filtered_keyframes(config);
    LoopDetector filtered_loops(config);
    GraphOptimizer filtered_optimizer(config);
    ASSERT_TRUE(serializer.loadMap(output.string(), filtered_keyframes, filtered_loops, filtered_optimizer));
    auto filtered_kf = filtered_keyframes.getKeyframe(kf_id);
    ASSERT_NE(filtered_kf, nullptr);
    ASSERT_NE(filtered_kf->cloud, nullptr);

    bool has_front = false;
    bool has_left = false;
    bool has_right = false;
    bool has_rear = false;
    bool has_angle_150 = false;
    bool has_angle_157_5 = false;
    for (const auto& point : filtered_kf->cloud->points) {
        if (point.x > 1.5f) has_front = true;
        if (point.y > 1.5f) has_left = true;
        if (point.y < -1.5f) has_right = true;
        if (point.x < -1.5f && std::abs(point.y) < 0.5f) has_rear = true;
        if (std::abs(point.intensity - 150.0f) < 1e-3f || std::abs(point.intensity + 150.0f) < 1e-3f) {
            has_angle_150 = true;
        }
        if (std::abs(point.intensity - 157.5f) < 1e-3f || std::abs(point.intensity + 157.5f) < 1e-3f) {
            has_angle_157_5 = true;
        }
    }
    EXPECT_TRUE(has_front);
    EXPECT_TRUE(has_left);
    EXPECT_TRUE(has_right);
    EXPECT_TRUE(has_angle_150);
    EXPECT_FALSE(has_angle_157_5);
    EXPECT_FALSE(has_rear);
    EXPECT_EQ(filtered_kf->cloud->size(), 170U);

    N3Map before;
    N3Map after;
    ASSERT_TRUE(parseMap(input, &before));
    ASSERT_TRUE(parseMap(output, &after));
    EXPECT_TRUE(hasDescriptorDifference(before, after));
    EXPECT_EQ(after.metadata().map_frame(), "world");
    EXPECT_EQ(after.metadata().body_frame(), "base_link");
    EXPECT_TRUE(after.metadata().nav_cloud_filter_applied());
    EXPECT_TRUE(after.metadata().descriptors_recomputed_from_filtered_cloud());
    EXPECT_EQ(after.metadata().nav_filter_raw_points(), 240U);
    EXPECT_EQ(after.metadata().nav_filter_kept_points(), 170U);
    EXPECT_EQ(after.metadata().nav_filter_removed_points(), 70U);
    EXPECT_NE(after.metadata().nav_cloud_filter_policy().find("rear_sector_width_deg=45"), std::string::npos);

    EXPECT_TRUE(std::filesystem::exists(debug_dir / "global_map_nav_filtered_debug.pcd"));
    EXPECT_TRUE(std::filesystem::exists(debug_dir / "removed_by_nav_filter.pcd"));
    EXPECT_TRUE(std::filesystem::exists(debug_dir / "removed_by_nav_filter_voxel.pcd"));
    EXPECT_TRUE(std::filesystem::exists(debug_dir / "nav_filter_report.json"));
    {
        std::ifstream report(debug_dir / "nav_filter_report.json");
        const std::string json((std::istreambuf_iterator<char>(report)), std::istreambuf_iterator<char>());
        EXPECT_NE(json.find("\"removed_ratio\""), std::string::npos);
        EXPECT_NE(json.find("\"per_keyframe\""), std::string::npos);
    }

    const std::filesystem::path second_dir = std::filesystem::path(config.map_save_path) / "second";
    std::filesystem::create_directories(second_dir);
    const std::filesystem::path second_output = second_dir / "n3map_nav_filtered_second.pbstream";
    const std::string second_command = tool.string() +
        " --input " + output.string() +
        " --output " + second_output.string() +
        " --debug-dir " + second_dir.string();
    ASSERT_EQ(std::system(second_command.c_str()), 0);
    EXPECT_TRUE(std::filesystem::exists(second_dir / "removed_by_nav_filter.pcd"));
    EXPECT_TRUE(std::filesystem::exists(second_dir / "removed_by_nav_filter_voxel.pcd"));

    std::filesystem::remove_all(config.map_save_path);
}

TEST(N3MappingFilterNavPbstreamTest, AppliesRangeAndSelfBoxFilters) {
    const auto tool = findFilterTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_filter_nav_pbstream executable not found";

    Config config;
    config.map_save_path = (std::filesystem::temp_directory_path() / "n3mapping_nav_filter_range_self_test").string();
    config.rhpd_submap_voxel_size = 0.0;
    std::filesystem::remove_all(config.map_save_path);
    std::filesystem::create_directories(config.map_save_path);

    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    GraphOptimizer optimizer(config);
    MapSerializer serializer(config);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = keyframes.addKeyframe(1.0, pose, makeRangeSelfCloud());
    auto cloud = keyframes.getKeyframe(kf_id)->cloud;
    keyframes.updateDescriptor(kf_id, loop_detector.addDescriptor(kf_id, cloud));
    keyframes.getKeyframe(kf_id)->rhpd_descriptor = loop_detector.addRHPD(kf_id, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::filesystem::path input = std::filesystem::path(config.map_save_path) / "input.pbstream";
    const std::filesystem::path output = std::filesystem::path(config.map_save_path) / "filtered.pbstream";
    const std::filesystem::path debug_dir = std::filesystem::path(config.map_save_path) / "debug";
    ASSERT_TRUE(serializer.saveMap(input.string(), keyframes, loop_detector, optimizer));

    const std::string command = tool.string() +
        " --input " + input.string() +
        " --output " + output.string() +
        " --debug-dir " + debug_dir.string() +
        " --rear-sector-enable false --max-range 3.0 --self-filter-enable true";
    ASSERT_EQ(std::system(command.c_str()), 0);

    KeyframeManager filtered_keyframes(config);
    LoopDetector filtered_loops(config);
    GraphOptimizer filtered_optimizer(config);
    ASSERT_TRUE(serializer.loadMap(output.string(), filtered_keyframes, filtered_loops, filtered_optimizer));
    auto filtered_kf = filtered_keyframes.getKeyframe(kf_id);
    ASSERT_NE(filtered_kf, nullptr);
    ASSERT_NE(filtered_kf->cloud, nullptr);
    ASSERT_EQ(filtered_kf->cloud->size(), 1U);
    EXPECT_NEAR(filtered_kf->cloud->front().x, 2.0f, 1e-6f);

    N3Map filtered;
    ASSERT_TRUE(parseMap(output, &filtered));
    EXPECT_EQ(filtered.metadata().nav_filter_raw_points(), 3U);
    EXPECT_EQ(filtered.metadata().nav_filter_kept_points(), 1U);
    EXPECT_EQ(filtered.metadata().nav_filter_removed_points(), 2U);
    EXPECT_TRUE(std::filesystem::exists(debug_dir / "removed_by_nav_filter_voxel.pcd"));

    std::filesystem::remove_all(config.map_save_path);
}

TEST(N3MappingFilterNavPbstreamTest, RejectsInputOutputSamePath) {
    const auto tool = findFilterTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_filter_nav_pbstream executable not found";

    Config config;
    config.map_save_path = (std::filesystem::temp_directory_path() / "n3mapping_nav_filter_same_path_test").string();
    config.rhpd_submap_voxel_size = 0.0;
    std::filesystem::remove_all(config.map_save_path);
    std::filesystem::create_directories(config.map_save_path);

    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    GraphOptimizer optimizer(config);
    MapSerializer serializer(config);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = keyframes.addKeyframe(1.0, pose, makeSectorCloud());
    auto cloud = keyframes.getKeyframe(kf_id)->cloud;
    keyframes.updateDescriptor(kf_id, loop_detector.addDescriptor(kf_id, cloud));
    keyframes.getKeyframe(kf_id)->rhpd_descriptor = loop_detector.addRHPD(kf_id, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::filesystem::path input = std::filesystem::path(config.map_save_path) / "input.pbstream";
    ASSERT_TRUE(serializer.saveMap(input.string(), keyframes, loop_detector, optimizer));

    const std::string command = tool.string() +
        " --input " + input.string() +
        " --output " + input.string();
    EXPECT_NE(std::system(command.c_str()), 0);

    N3Map after;
    ASSERT_TRUE(parseMap(input, &after));
    EXPECT_FALSE(after.metadata().nav_cloud_filter_applied());
    ASSERT_EQ(after.keyframes_size(), 1);
    EXPECT_EQ(after.keyframes(0).cloud().num_points(), 240U);

    std::filesystem::remove_all(config.map_save_path);
}

TEST(N3MappingFilterNavPbstreamTest, FailsFastWhenFilterWouldEmptyKeyframe) {
    const auto tool = findFilterTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_filter_nav_pbstream executable not found";

    Config config;
    config.map_save_path = (std::filesystem::temp_directory_path() / "n3mapping_nav_filter_empty_kf_test").string();
    config.rhpd_submap_voxel_size = 0.0;
    std::filesystem::remove_all(config.map_save_path);
    std::filesystem::create_directories(config.map_save_path);

    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    GraphOptimizer optimizer(config);
    MapSerializer serializer(config);

    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const int64_t kf_id = keyframes.addKeyframe(1.0, pose, makeRearOnlyCloud());
    auto cloud = keyframes.getKeyframe(kf_id)->cloud;
    keyframes.updateDescriptor(kf_id, loop_detector.addDescriptor(kf_id, cloud));
    keyframes.getKeyframe(kf_id)->rhpd_descriptor = loop_detector.addRHPD(kf_id, cloud);
    optimizer.addPriorFactor(kf_id, pose);

    const std::filesystem::path input = std::filesystem::path(config.map_save_path) / "input.pbstream";
    const std::filesystem::path output = std::filesystem::path(config.map_save_path) / "filtered.pbstream";
    const std::filesystem::path debug_dir = std::filesystem::path(config.map_save_path) / "debug";
    ASSERT_TRUE(serializer.saveMap(input.string(), keyframes, loop_detector, optimizer));

    const std::string command = tool.string() +
        " --input " + input.string() +
        " --output " + output.string() +
        " --debug-dir " + debug_dir.string();
    EXPECT_NE(std::system(command.c_str()), 0);
    EXPECT_FALSE(std::filesystem::exists(output));
    ASSERT_TRUE(std::filesystem::exists(debug_dir / "nav_filter_report.json"));
    {
        std::ifstream report(debug_dir / "nav_filter_report.json");
        const std::string json((std::istreambuf_iterator<char>(report)), std::istreambuf_iterator<char>());
        EXPECT_NE(json.find("\"empty_result_keyframes\": [0]"), std::string::npos);
        EXPECT_NE(json.find("\"kept_points\": 0"), std::string::npos);
    }

    std::filesystem::remove_all(config.map_save_path);
}

}  // namespace test
}  // namespace n3mapping
