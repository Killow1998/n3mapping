#include "n3mapping/loop_debug_logger.h"

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
           ("n3mapping_loop_debug_logger_test_" + std::to_string(::getpid()));
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

LoopDebugCandidateEvent makeRejectedCandidateEvent()
{
    LoopDebugCandidateEvent event;
    event.processing_time = 1.25;
    event.query_timestamp = 42.5;
    event.candidate.query_id = 10;
    event.candidate.match_id = 2;
    event.candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
    event.candidate.rhpd_distance = 3.5;
    event.candidate.sc_distance = 0.25;
    event.candidate.fused_score = 4.0;
    event.candidate.yaw_diff_rad = 0.1f;
    event.icp_converged = false;
    event.fitness_score = 9.0;
    event.inlier_ratio = 0.0;
    event.icp_translation_norm = 1.0;
    event.icp_rotation_norm = 0.2;
    event.residual.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
    event.has_loop_measurement = true;
    event.loop_measurement_match_query.translation() = Eigen::Vector3d(4.0, 5.0, 6.0);
    event.gate_result = "rejected";
    event.reject_reason = "bad\nreason";
    event.loop_information.diagonal() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    return event;
}

TEST(LoopDebugLoggerTest, ResolvePathUsesMapSavePathWhenDebugPathEmpty)
{
    Config config;
    config.map_save_path = "/tmp/n3mapping_map";
    config.loop_debug_path.clear();

    EXPECT_EQ(LoopDebugLogger::resolvePath(config), "/tmp/n3mapping_map/loop_debug.jsonl");

    config.loop_debug_path = "/tmp/custom_loop_debug.jsonl";
    EXPECT_EQ(LoopDebugLogger::resolvePath(config), "/tmp/custom_loop_debug.jsonl");
}

TEST(LoopDebugLoggerTest, WritesRejectedCandidateAsSingleLineJson)
{
    const auto temp_dir = makeTempDir();
    std::filesystem::remove_all(temp_dir);
    const auto path = temp_dir / "loop_debug.jsonl";

    ASSERT_TRUE(LoopDebugLogger::appendCandidate(path.string(), makeRejectedCandidateEvent()));

    const auto lines = readLines(path);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].front(), '{');
    EXPECT_EQ(lines[0].back(), '}');
    EXPECT_NE(lines[0].find("\"record_type\":\"candidate\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"gate_result\":\"rejected\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"reject_reason\":\"bad\\nreason\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"measurement_z\":6"), std::string::npos);
    EXPECT_NE(lines[0].find("\"loop_information_diag\":[1,2,3,4,5,6]"), std::string::npos);

    std::filesystem::remove_all(temp_dir);
}

TEST(LoopDebugLoggerTest, AppendsOptimizationSummary)
{
    const auto temp_dir = makeTempDir();
    std::filesystem::remove_all(temp_dir);
    const auto path = temp_dir / "loop_debug.jsonl";

    ASSERT_TRUE(LoopDebugLogger::appendCandidate(path.string(), makeRejectedCandidateEvent()));

    LoopDebugOptimizationEvent event;
    event.processing_time = 2.0;
    event.accepted_edge_count = 3;
    event.accepted_edges = {{2, 10}, {4, 12}};
    event.loop_residual_translation_before = 1.0;
    event.loop_residual_translation_after = 0.2;
    event.loop_residual_rotation_before = 0.5;
    event.loop_residual_rotation_after = 0.1;
    event.loop_residual_translation_axes_before = Eigen::Vector3d(1.0, 2.0, 3.0);
    event.loop_residual_translation_axes_after = Eigen::Vector3d(0.1, 0.2, 0.3);
    event.loop_residual_rpy_axes_before = Eigen::Vector3d(0.4, 0.5, 0.6);
    event.loop_residual_rpy_axes_after = Eigen::Vector3d(0.04, 0.05, 0.06);
    event.mean_pose_update_translation = 0.05;
    event.max_pose_update_translation = 0.1;
    event.mean_pose_update_rotation = 0.02;
    event.max_pose_update_rotation = 0.04;
    ASSERT_TRUE(LoopDebugLogger::appendOptimizationSummary(path.string(), event));

    const auto lines = readLines(path);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_NE(lines[1].find("\"record_type\":\"optimization_summary\""), std::string::npos);
    EXPECT_NE(lines[1].find("\"accepted_edge_count\":3"), std::string::npos);
    EXPECT_NE(lines[1].find("\"accepted_edges\":[{\"from_id\":2,\"to_id\":10},{\"from_id\":4,\"to_id\":12}]"), std::string::npos);
    EXPECT_NE(lines[1].find("\"loop_residual_translation_before\":1"), std::string::npos);
    EXPECT_NE(lines[1].find("\"loop_residual_x_before\":1"), std::string::npos);
    EXPECT_NE(lines[1].find("\"loop_residual_z_after\":0.29999999999999999"), std::string::npos);
    EXPECT_NE(lines[1].find("\"loop_residual_pitch_before\":0.5"), std::string::npos);
    EXPECT_NE(lines[1].find("\"loop_residual_yaw_after\":0.059999999999999998"), std::string::npos);

    std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace n3mapping
