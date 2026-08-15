#include "n3mapping/relocalization_debug_logger.h"
#include "n3mapping/registration_observability.h"
#include "n3mapping/runtime_performance_debug_logger.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

namespace n3mapping {
namespace {

std::filesystem::path makeTempDir() {
  return std::filesystem::temp_directory_path() /
         ("n3mapping_relocalization_debug_logger_test_" +
          std::to_string(::getpid()));
}

std::vector<std::string> readLines(const std::filesystem::path &path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  return lines;
}

LoopCandidate makeCandidate() {
  LoopCandidate candidate;
  candidate.match_id = 7;
  candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
  candidate.rhpd_distance = 2.5;
  candidate.sc_distance = 0.2;
  candidate.fused_score = 0.42;
  candidate.yaw_diff_rad = 0.1f;
  return candidate;
}

MatchResult makeObservableMatch() {
  MatchResult match;
  match.success = true;
  match.converged = true;
  match.information.setZero();
  match.information.diagonal() << 4.0, 9.0, 16.0, 25.0, 36.0, 49.0;
  match.optimizer_error = 36.0;
  match.num_inliers = 9;
  match.fitness_score = 4.0;
  match.inlier_ratio = 0.75;
  match.iterations = 7;
  match.termination = MatchTermination::Converged;
  return match;
}

TEST(RegistrationObservabilityTest, SummarizesExistingInformationWithoutGate) {
  const RegistrationObservability observability =
      analyzeRegistrationObservability(makeObservableMatch(),
                                       "registration_endpoint", true, true);

  EXPECT_TRUE(observability.available);
  EXPECT_TRUE(observability.information_finite);
  EXPECT_TRUE(observability.spectrum_valid);
  EXPECT_TRUE(observability.positive_definite);
  EXPECT_EQ(observability.numerical_rank, 6);
  EXPECT_NEAR(observability.condition_number, 49.0 / 4.0, 1e-12);
  EXPECT_NEAR(observability.residual_scale, 2.0, 1e-12);
  EXPECT_EQ(observability.selected_pose_source, "registration_endpoint");
  EXPECT_TRUE(observability.information_at_selected_pose);
  EXPECT_TRUE(observability.production_quality);
  RegistrationObservability::Vector6d expected_eigenvalues;
  expected_eigenvalues << 4.0, 9.0, 16.0, 25.0, 36.0, 49.0;
  EXPECT_TRUE(observability.information_eigenvalues.isApprox(
      expected_eigenvalues, 1e-12));
  EXPECT_TRUE(observability.rotational_marginal_eigenvalues.isApprox(
      Eigen::Vector3d(25.0, 36.0, 49.0), 1e-12));
}

TEST(RegistrationObservabilityTest, UsesSchurRotationalMarginal) {
  MatchResult match = makeObservableMatch();
  match.information.setZero();
  match.information.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 4.0;
  match.information.block<3, 3>(3, 3) =
      Eigen::Vector3d(10.0, 11.0, 12.0).asDiagonal();
  match.information.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * 2.0;
  match.information.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * 2.0;

  const RegistrationObservability observability =
      analyzeRegistrationObservability(match, "motion_prediction", false,
                                       true);

  ASSERT_TRUE(observability.rotational_marginal_valid);
  const Eigen::Matrix3d expected_marginal =
      Eigen::Vector3d(9.0, 10.0, 11.0).asDiagonal();
  EXPECT_TRUE(observability.rotational_marginal_information.isApprox(
      expected_marginal, 1e-12));
  EXPECT_TRUE(observability.rotational_marginal_eigenvalues.isApprox(
      Eigen::Vector3d(9.0, 10.0, 11.0), 1e-12));
  EXPECT_EQ(observability.selected_pose_source, "motion_prediction");
  EXPECT_FALSE(observability.information_at_selected_pose);
}

TEST(RegistrationObservabilityTest, ExposesRankDeficiencyAndInvalidInput) {
  MatchResult rank_deficient = makeObservableMatch();
  rank_deficient.information.setZero();
  rank_deficient.information.diagonal() << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;
  const RegistrationObservability ranked = analyzeRegistrationObservability(
      rank_deficient, "descriptor_initial", false, true);
  ASSERT_TRUE(ranked.spectrum_valid);
  EXPECT_FALSE(ranked.positive_definite);
  EXPECT_EQ(ranked.numerical_rank, 3);
  EXPECT_TRUE(std::isinf(ranked.condition_number));

  MatchResult invalid = makeObservableMatch();
  invalid.information(0, 0) = std::numeric_limits<double>::quiet_NaN();
  const RegistrationObservability rejected = analyzeRegistrationObservability(
      invalid, "registration_endpoint", true, false);
  EXPECT_TRUE(rejected.available);
  EXPECT_FALSE(rejected.information_finite);
  EXPECT_FALSE(rejected.spectrum_valid);
}

TEST(RelocalizationDebugLoggerTest,
     ResolvePathUsesMapSavePathWhenDebugPathEmpty) {
  Config config;
  config.map_save_path = "/tmp/n3mapping_map";
  config.reloc_debug_path.clear();

  EXPECT_EQ(RelocalizationDebugLogger::resolvePath(config),
            "/tmp/n3mapping_map/relocalization_debug.jsonl");

  config.reloc_debug_path = "/tmp/custom_relocalization_debug.jsonl";
  EXPECT_EQ(RelocalizationDebugLogger::resolvePath(config),
            "/tmp/custom_relocalization_debug.jsonl");
}

TEST(RuntimePerformanceDebugLoggerTest,
     UsesSeparateSiblingPathAndHonorsDebugSwitch) {
  Config config;
  config.map_save_path = "/tmp/n3mapping_map";
  EXPECT_EQ(RuntimePerformanceDebugLogger::resolvePath(config),
            "/tmp/n3mapping_map/runtime_performance_debug.jsonl");

  config.reloc_debug_path = "/tmp/custom/relocalization.jsonl";
  EXPECT_EQ(RuntimePerformanceDebugLogger::resolvePath(config),
            "/tmp/custom/runtime_performance_debug.jsonl");

  RuntimePerformanceDebugLogger disabled(config);
  EXPECT_FALSE(disabled.enabled());
  EXPECT_FALSE(disabled.ready());
}

TEST(RuntimePerformanceDebugLoggerTest, AppendsVersionedFrameAndLoopCycle) {
  const auto temp_dir = makeTempDir() / "runtime_performance";
  std::filesystem::remove_all(temp_dir);
  Config config;
  config.reloc_debug_enable = true;
  config.reloc_debug_path =
      (temp_dir / "relocalization_debug.jsonl").string();
  {
    RuntimePerformanceDebugLogger logger(config);
    ASSERT_TRUE(logger.enabled());
    ASSERT_TRUE(logger.ready());

    RuntimePerformanceDebugEvent event;
    event.mode = "map_extension";
    event.processing_time = 10.0;
    event.frame_index = 7;
    event.sensor_timestamp = 123.0;
    event.sensor_delta_ms = 100.0;
    event.callback_interarrival_ms = 105.0;
    event.input_points = 4096;
    event.core_success = true;
    event.accepted_keyframe = true;
    event.keyframe_id = 218;
    event.matched_keyframe_id = 53;
    event.relocalization_state = "FULL_6DOF_LOCKED";
    event.pose_source = "GEOMETRICALLY_CORRECTED";
    event.relocalization_decision = "loaded_map_tracking_geometric";
    event.relocalization_locked = false;
    event.tracking_attempted = true;
    event.published_global_pose = true;
    event.published_body_cloud = true;
    event.published_world_cloud = true;
    event.callback_lock_wait_ms = 0.5;
    event.ros_conversion_ms = 1.0;
    event.core_frame_ms = 20.0;
    event.loaded_map_tracking_ms = 12.0;
    event.keyframe_gate_ms = 0.1;
    event.keyframe_commit_ms = 4.0;
    event.graph_update_ms = 2.5;
    event.descriptor_update_ms = 1.0;
    event.post_commit_refresh_ms = 0.25;
    event.authority_publish_ms = 0.2;
    event.odometry_path_publish_ms = 0.3;
    event.callback_locked_ms = 22.0;
    event.cloud_publish_ms = 3.0;
    event.callback_total_ms = 25.5;
    ASSERT_TRUE(logger.append(event));

    RuntimePerformanceLoopEvent loop_event;
    loop_event.mode = "mapping";
    loop_event.processing_time = 11.0;
    loop_event.cycle_index = 3;
    loop_event.queued_keyframe_count = 1;
    loop_event.detected_candidate_count = 4;
    loop_event.place_candidate_count = 1;
    loop_event.accepted_loop_count = 1;
    loop_event.edge_count = 1;
    loop_event.optimized = true;
    loop_event.lock_wait_ms = 0.5;
    loop_event.core_ms = 40.0;
    loop_event.publish_ms = 2.0;
    loop_event.total_ms = 43.0;
    ASSERT_TRUE(logger.append(loop_event));
  }

  const auto lines = readLines(temp_dir / "runtime_performance_debug.jsonl");
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_NE(lines[0].find(
                "\"schema\":\"n3mapping_runtime_performance_v2\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"record_type\":\"runtime_frame\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"mode\":\"map_extension\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"tracking_attempted\":true"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"frame_index\":7"), std::string::npos);
  EXPECT_NE(lines[0].find("\"graph_update_ms\":2.5"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"initial_relocalization_ms\":null"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"callback_total_ms\":25.5"),
            std::string::npos);
  EXPECT_NE(lines[1].find("\"record_type\":\"loop_cycle\""),
            std::string::npos);
  EXPECT_NE(lines[1].find("\"detected_candidate_count\":4"),
            std::string::npos);
  EXPECT_NE(lines[1].find("\"total_ms\":43"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(RelocalizationDebugLoggerTest,
     WritesRelocalizationRejectAsSingleLineJson) {
  const auto temp_dir = makeTempDir();
  std::filesystem::remove_all(temp_dir);
  const auto path = temp_dir / "relocalization_debug.jsonl";

  RelocalizationDebugEvent event;
  event.processing_time = 1.0;
  event.query_timestamp = 123.456;
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
  RelocDebugBasinBestSummary basin_best;
  basin_best.basin_center_id = 7;
  basin_best.matched_kf_id = 8;
  basin_best.candidate = makeCandidate();
  basin_best.pose_in_map.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  basin_best.visibility_consistency_ratio = 0.8;
  basin_best.visibility_evidence_log_odds = 1.5;
  basin_best.visibility_known_bins = 8;
  basin_best.visibility_unknown_bins = 2;
  basin_best.visibility_known_fraction = 0.8;
  basin_best.visibility_consistent_given_known = 0.75;
  basin_best.visibility_foreground_conflict_given_known = 0.125;
  basin_best.visibility_evidence_log_odds_given_known = 1.25;
  basin_best.registration_observability = analyzeRegistrationObservability(
      makeObservableMatch(), "descriptor_initial", false, true);
  event.basin_best_results.push_back(basin_best);
  RelocDebugHypothesisSummary hypothesis;
  hypothesis.seed_match_id = 7;
  hypothesis.last_match_id = 8;
  hypothesis.visibility_updates = 3;
  hypothesis.mean_visibility_consistency = 0.8;
  hypothesis.mean_visibility_evidence = 1.5;
  hypothesis.registration_observability = analyzeRegistrationObservability(
      makeObservableMatch(), "registration_endpoint", true, true);
  event.hypotheses.push_back(hypothesis);
  event.winner_seed_match_id = 7;
  event.winner_last_match_id = 8;
  event.runner_up_seed_match_id = 9;
  event.runner_up_last_match_id = 10;
  event.visibility_margin = 1.2;
  event.visibility_ratio = 3.3;
  event.winner_pose_translation_delta = 0.12;
  event.winner_pose_rotation_delta = 0.03;
  event.evidence_motion_translation = 1.2;
  event.evidence_motion_rotation = 0.1;
  event.moving_visibility_required = true;
  event.moving_visibility_passed = false;
  event.lock_result = "rejected";
  event.reject_reason = "bad\nreason";

  ASSERT_TRUE(
      RelocalizationDebugLogger::appendRelocalization(path.string(), event));

  const auto lines = readLines(path);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].front(), '{');
  EXPECT_EQ(lines[0].back(), '}');
  EXPECT_NE(lines[0].find("\"record_type\":\"relocalize\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"query_timestamp\":123.456"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"query_mode\":\"stationary\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"query_frame_count\":1"), std::string::npos);
  EXPECT_NE(lines[0].find("\"motion_query_mode\":\"motion_submap\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"motion_query_frame_count\":3"), std::string::npos);
  EXPECT_NE(lines[0].find("\"motion_query_candidate_count\":1"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"winner_pose_translation_delta\":"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"winner_pose_rotation_delta\":"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"evidence_motion_translation\":1.2"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"moving_visibility_required\":true"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"moving_visibility_passed\":false"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"candidate_count\":1"), std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_evidence_log_odds\":1.5"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_known_bins\":8"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_unknown_bins\":2"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_known_fraction\":0.8"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_consistent_given_known\":0.75"),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"visibility_foreground_conflict_given_known\":0.125"),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"visibility_evidence_log_odds_given_known\":1.25"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"mean_visibility_evidence\":1.5"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"winner_seed_match_id\":7"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"winner_last_match_id\":8"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"runner_up_seed_match_id\":9"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"runner_up_last_match_id\":10"),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"information_layout\":\"translation_xyz_rotation_xyz\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"selected_pose_source\":\"descriptor_initial\""),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"information_at_selected_pose\":false"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"observable_dofs_numerical\":6"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"residual_scale\":2"), std::string::npos);
  EXPECT_NE(lines[0].find("\"full_information\":[[4,0,0,0,0,0]"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"full_eigenvalues_ascending\":["),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"rotational_marginal_eigenvalues_ascending\":["),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_margin\":1.2"), std::string::npos);
  EXPECT_NE(lines[0].find("\"lock_result\":\"rejected\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"reject_reason\":\"bad\\nreason\""),
            std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(RelocalizationDebugLoggerTest, AppendsTrackingFailure) {
  const auto temp_dir = makeTempDir();
  std::filesystem::remove_all(temp_dir);
  const auto path = temp_dir / "relocalization_debug.jsonl";

  RelocTrackingDebugEvent event;
  event.processing_time = 2.0;
  event.query_index = 8;
  event.strict_loaded_map = true;
  event.predicted_pose.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  event.nearest_kf_id = -1;
  event.submap_size = 0;
  event.tracking_total_ms = 12.5;
  event.nearest_keyframe_ms = 0.25;
  event.loaded_map_cache_ms = 1.5;
  event.submap_build_ms = 2.5;
  event.target_prepare_ms = 3.5;
  event.loaded_map_target_cache_miss = true;
  event.localization_target_cache_enabled = false;
  event.source_prepare_ms = 0.75;
  event.registration_ms = 4.0;
  event.visibility_ms = 0.5;
  event.visibility_prediction_ms = 0.25;
  event.visibility_registration_ms = 0.5;
  event.visibility_prediction_valid = true;
  event.visibility_prediction_consistency_ratio = 0.8;
  event.visibility_prediction_evidence_log_odds = 1.1;
  event.visibility_registration_valid = true;
  event.visibility_registration_consistency_ratio = 0.7;
  event.visibility_registration_evidence_log_odds = 0.9;
  event.visibility_selected_pose_source = "motion_prediction";
  event.visibility_registration_delta_translation_m = 0.2;
  event.visibility_registration_delta_rotation_rad = 0.02;
  event.visibility_registration_would_accept = true;
  event.icp_converged = false;
  event.retry_used = true;
  event.consecutive_track_failures = 2;
  event.result_success = false;
  event.reject_reason = "nearest_keyframe_missing";

  ASSERT_TRUE(RelocalizationDebugLogger::appendTracking(path.string(), event));

  const auto lines = readLines(path);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("\"record_type\":\"tracking\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"strict_loaded_map\":true"), std::string::npos);
  EXPECT_NE(lines[0].find("\"nearest_kf_id\":-1"), std::string::npos);
  EXPECT_NE(lines[0].find("\"tracking_total_ms\":12.5"), std::string::npos);
  EXPECT_NE(lines[0].find("\"nearest_keyframe_ms\":0.25"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"loaded_map_target_cache_hit\":false"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"loaded_map_target_cache_miss\":true"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"localization_target_cache_enabled\":false"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"localization_target_cache_entry_bytes\":0"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"retry_registration_ms\":null"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_ms\":0.5"), std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_prediction_ms\":0.25"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"visibility_registration_ms\":0.5"),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"visibility_selected_pose_source\":\"motion_prediction\""),
            std::string::npos);
  EXPECT_NE(lines[0].find(
                "\"visibility_registration_would_accept\":true"),
            std::string::npos);
  EXPECT_NE(lines[0].find("\"retry_used\":true"), std::string::npos);
  EXPECT_NE(lines[0].find("\"reject_reason\":\"nearest_keyframe_missing\""),
            std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

} // namespace
} // namespace n3mapping
