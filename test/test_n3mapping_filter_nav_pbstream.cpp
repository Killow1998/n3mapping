#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
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

    const std::string command = tool.string() + " " + input.string() + " " + output.string();
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
    for (const auto& point : filtered_kf->cloud->points) {
        if (point.x > 1.5f) has_front = true;
        if (point.y > 1.5f) has_left = true;
        if (point.y < -1.5f) has_right = true;
        if (point.x < -1.5f) has_rear = true;
    }
    EXPECT_TRUE(has_front);
    EXPECT_TRUE(has_left);
    EXPECT_TRUE(has_right);
    EXPECT_FALSE(has_rear);
    EXPECT_EQ(filtered_kf->cloud->size(), 150U);

    N3Map before;
    N3Map after;
    ASSERT_TRUE(parseMap(input, &before));
    ASSERT_TRUE(parseMap(output, &after));
    EXPECT_TRUE(hasDescriptorDifference(before, after));

    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(config.map_save_path) / "global_map_nav_filtered_debug.pcd"));
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(config.map_save_path) / "removed_by_nav_filter.pcd"));
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(config.map_save_path) / "nav_filter_report.json"));

    const std::filesystem::path second_dir = std::filesystem::path(config.map_save_path) / "second";
    std::filesystem::create_directories(second_dir);
    const std::filesystem::path second_output = second_dir / "n3map_nav_filtered_second.pbstream";
    const std::string second_command = tool.string() + " " + output.string() + " " + second_output.string();
    ASSERT_EQ(std::system(second_command.c_str()), 0);
    EXPECT_TRUE(std::filesystem::exists(second_dir / "removed_by_nav_filter.pcd"));

    std::filesystem::remove_all(config.map_save_path);
}

}  // namespace test
}  // namespace n3mapping
