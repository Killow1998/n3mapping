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
    event.edge_mode = "planar_xy_yaw";
    event.vertical_observability_score = 0.25;
    event.vertical_downweighted = true;
    event.source_z_span = 1.5;
    event.target_z_span = 2.5;
    event.z_overlap_ratio_before = 0.4;
    event.z_overlap_ratio_after = 0.8;
    event.source_z_robust_span = 1.1;
    event.target_z_robust_span = 2.1;
    event.z_robust_overlap_ratio_before = 0.3;
    event.z_robust_overlap_ratio_after = 0.7;
    event.source_target_z_centroid_delta_before = -1.2;
    event.source_target_z_centroid_delta_after = 0.1;
    event.vertical_information_ratio = 0.75;
    event.vertical_hypothesis_count = 7;
    event.best_z_offset_m = -0.5;
    event.best_z_offset_fitness = 0.11;
    event.zero_z_fitness = 0.12;
    event.fitness_gap_zero_vs_best = 0.01;
    event.z_hypothesis_spread_m = 1.0;
    event.vertical_ambiguity_score = 0.25;
    event.vertical_hypothesis_edge_recommendation = "planar_xy_yaw";
    event.heightmap_overlap_cell_count = 12;
    event.heightmap_overlap_ratio = 0.75;
    event.heightmap_ground_dz_median = 0.4;
    event.heightmap_ground_dz_p90 = 1.2;
    event.heightmap_ground_dz_max = 1.8;
    event.heightmap_ground_support_ratio = 0.6;
    event.heightmap_vertical_consistency_score = 0.27;
    event.graph_trial_success = true;
    event.graph_trial_residual_x_after = 0.1;
    event.graph_trial_residual_y_after = 0.2;
    event.graph_trial_residual_z_after = 0.3;
    event.graph_trial_residual_roll_after = 0.01;
    event.graph_trial_residual_pitch_after = 0.02;
    event.graph_trial_residual_yaw_after = 0.03;
    event.graph_trial_residual_translation_norm_after = 0.4;
    event.graph_trial_residual_rotation_norm_after = 0.05;
    event.graph_trial_mean_pose_update_translation = 0.06;
    event.graph_trial_max_pose_update_translation = 0.07;
    event.graph_trial_mean_pose_update_rotation = 0.08;
    event.graph_trial_max_pose_update_rotation = 0.09;
    event.graph_trial_existing_loop_residual_delta = -0.1;
    event.graph_trial_odom_residual_delta = 0.2;
    event.graph_trial_consistency_score = 0.75;
    event.graph_trial_recommendation = "trial_success_score_only";
    event.segment_pair_count = 4;
    event.segment_valid_pair_count = 3;
    event.segment_consensus_inlier_count = 2;
    event.segment_consensus_ratio = 2.0 / 3.0;
    event.segment_translation_median = 0.3;
    event.segment_translation_std = 0.4;
    event.segment_yaw_median = 0.05;
    event.segment_yaw_std = 0.06;
    event.segment_z_std = 0.07;
    event.segment_roll_pitch_std = 0.08;
    event.segment_direction = "same";
    event.segment_recommendation = "consistent";
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
    EXPECT_NE(lines[0].find("\"edge_mode\":\"planar_xy_yaw\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_observability_score\":0.25"), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_downweighted\":true"), std::string::npos);
    EXPECT_NE(lines[0].find("\"source_z_span\":1.5"), std::string::npos);
    EXPECT_NE(lines[0].find("\"target_z_span\":2.5"), std::string::npos);
    EXPECT_NE(lines[0].find("\"z_overlap_ratio_before\":0.40000000000000002"), std::string::npos);
    EXPECT_NE(lines[0].find("\"z_overlap_ratio_after\":0.80000000000000004"), std::string::npos);
    EXPECT_NE(lines[0].find("\"source_z_robust_span\":1.1000000000000001"), std::string::npos);
    EXPECT_NE(lines[0].find("\"target_z_robust_span\":2.1000000000000001"), std::string::npos);
    EXPECT_NE(lines[0].find("\"z_robust_overlap_ratio_before\":0.29999999999999999"), std::string::npos);
    EXPECT_NE(lines[0].find("\"z_robust_overlap_ratio_after\":0.69999999999999996"), std::string::npos);
    EXPECT_NE(lines[0].find("\"source_target_z_centroid_delta_before\":-1.2"), std::string::npos);
    EXPECT_NE(lines[0].find("\"source_target_z_centroid_delta_after\":0.10000000000000001"), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_information_ratio\":0.75"), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_hypothesis_count\":7"), std::string::npos);
    EXPECT_NE(lines[0].find("\"best_z_offset_m\":-0.5"), std::string::npos);
    EXPECT_NE(lines[0].find("\"best_z_offset_fitness\":0.11"), std::string::npos);
    EXPECT_NE(lines[0].find("\"zero_z_fitness\":0.12"), std::string::npos);
    EXPECT_NE(lines[0].find("\"fitness_gap_zero_vs_best\":0.01"), std::string::npos);
    EXPECT_NE(lines[0].find("\"z_hypothesis_spread_m\":1"), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_ambiguity_score\":0.25"), std::string::npos);
    EXPECT_NE(lines[0].find("\"vertical_hypothesis_edge_recommendation\":\"planar_xy_yaw\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"heightmap_overlap_cell_count\":12"), std::string::npos);
    EXPECT_NE(lines[0].find("\"heightmap_ground_dz_p90\":1.2"), std::string::npos);
    EXPECT_NE(lines[0].find("\"heightmap_vertical_consistency_score\":0.27000000000000002"), std::string::npos);
    EXPECT_NE(lines[0].find("\"graph_trial_success\":true"), std::string::npos);
    EXPECT_NE(lines[0].find("\"graph_trial_residual_z_after\":0.29999999999999999"), std::string::npos);
    EXPECT_NE(lines[0].find("\"graph_trial_consistency_score\":0.75"), std::string::npos);
    EXPECT_NE(lines[0].find("\"graph_trial_recommendation\":\"trial_success_score_only\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_pair_count\":4"), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_valid_pair_count\":3"), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_consensus_inlier_count\":2"), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_consensus_ratio\":0.66666666666666663"), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_translation_median\":0.29999999999999999"), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_direction\":\"same\""), std::string::npos);
    EXPECT_NE(lines[0].find("\"segment_recommendation\":\"consistent\""), std::string::npos);
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
