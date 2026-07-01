#include "n3mapping/relocalization_debug_logger.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

namespace n3mapping {
namespace {

std::filesystem::path makeTempDir()
{
    return std::filesystem::temp_directory_path() /
           ("n3mapping_relocalization_debug_logger_test_" + std::to_string(::getpid()));
}

std::vector<std::string> readLines(const std::filesystem::path& path)
{
    std::vector<std::string> lines;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

LoopCandidate makeCandidate()
{
    LoopCandidate candidate;
    candidate.match_id = 7;
    candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
    candidate.rhpd_distance = 2.5;
    candidate.sc_distance = 0.2;
    candidate.fused_score = 0.42;
    candidate.yaw_diff_rad = 0.1f;
    return candidate;
}

TEST(RelocalizationDebugLoggerTest, ResolvePathUsesMapSavePathWhenDebugPathEmpty)
{
    Config config;
    config.map_save_path = "/tmp/n3mapping_map";
    config.reloc_debug_path.clear();

    EXPECT_EQ(RelocalizationDebugLogger::resolvePath(config), "/tmp/n3mapping_map/relocalization_debug.jsonl");

    config.reloc_debug_path = "/tmp/custom_relocalization_debug.jsonl";
    EXPECT_EQ(RelocalizationDebugLogger::resolvePath(config), "/tmp/custom_relocalization_debug.jsonl");
}

TEST(RelocalizationDebugLoggerTest, WritesRelocalizationRejectAsSingleLineJson)
{
    const auto temp_dir = makeTempDir();
    std::filesystem::remove_all(temp_dir);
    const auto path = temp_dir / "relocalization_debug.jsonl";

    RelocalizationDebugEvent event;
    event.processing_time = 1.0;
    event.query_index = 4;
    event.query_cloud.mode = "stationary";
    event.query_cloud.frame_count = 1;
    event.query_cloud.motion_translation_m = 0.0;
    event.query_cloud.motion_rotation_rad = 0.0;
    event.query_cloud.raw_points = 100;
    event.query_cloud.downsampled_points = 80;
    event.motion_query_cloud.mode = "motion_submap";
    event.motion_query_cloud.frame_count = 3;
    event.motion_query_cloud.motion_translation_m = 1.2;
    event.motion_query_cloud.motion_rotation_rad = 0.1;
    event.motion_query_cloud.raw_points = 300;
    event.motion_query_cloud.downsampled_points = 180;
    event.motion_query_cloud.candidate_count = 1;
    event.motion_query_cloud.top_candidates.push_back(makeCandidate());
    event.candidate_count = 1;
    event.top_candidates.push_back(makeCandidate());
    event.lock_result = "rejected";
    event.reject_reason = "bad\nreason";

    ASSERT_TRUE(RelocalizationDebugLogger::appendRelocalization(path.string(), event));

    const auto lines = readLines(path);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].front(), '{');
    EXPECT_EQ(lines[0].back(), '}');
    EXPECT_NE(lines[0].find("\"record_type\":\"relocalize\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"query_mode\":\"stationary\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"query_frame_count\":1"), std::string::npos);
    EXPECT_NE(lines[0].find("\"motion_query_mode\":\"motion_submap\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"motion_query_frame_count\":3"), std::string::npos);
    EXPECT_NE(lines[0].find("\"motion_query_candidate_count\":1"), std::string::npos);
    EXPECT_NE(lines[0].find("\"candidate_count\":1"), std::string::npos);
    EXPECT_NE(lines[0].find("\"lock_result\":\"rejected\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"reject_reason\":\"bad\\nreason\""), std::string::npos);

    std::filesystem::remove_all(temp_dir);
}

TEST(RelocalizationDebugLoggerTest, AppendsTrackingFailure)
{
    const auto temp_dir = makeTempDir();
    std::filesystem::remove_all(temp_dir);
    const auto path = temp_dir / "relocalization_debug.jsonl";

    RelocTrackingDebugEvent event;
    event.processing_time = 2.0;
    event.query_index = 8;
    event.predicted_pose.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
    event.nearest_kf_id = -1;
    event.submap_size = 0;
    event.icp_converged = false;
    event.retry_used = true;
    event.consecutive_track_failures = 2;
    event.result_success = false;
    event.reject_reason = "nearest_keyframe_missing";

    ASSERT_TRUE(RelocalizationDebugLogger::appendTracking(path.string(), event));

    const auto lines = readLines(path);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_NE(lines[0].find("\"record_type\":\"tracking\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"nearest_kf_id\":-1"), std::string::npos);
    EXPECT_NE(lines[0].find("\"retry_used\":true"), std::string::npos);
    EXPECT_NE(lines[0].find("\"reject_reason\":\"nearest_keyframe_missing\""), std::string::npos);

    std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace n3mapping
