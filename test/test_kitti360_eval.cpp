#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace n3mapping {
namespace test {
namespace {

std::filesystem::path makeTempDir(const std::string& name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        (name + "_" + std::to_string(static_cast<long long>(stamp)));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string shellQuote(const std::filesystem::path& path)
{
    std::string value = path.string();
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::filesystem::path findKittiEvalTool()
{
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::filesystem::path dir = self.parent_path();
    const std::filesystem::path candidates[] = {
        dir / "n3mapping_kitti360_eval",
        dir.parent_path() / "n3mapping_kitti360_eval",
        std::filesystem::current_path() / "n3mapping_kitti360_eval",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

std::filesystem::path findLoopDebugAnalyzer()
{
    const std::filesystem::path source_tool =
        std::filesystem::path(N3MAPPING_SOURCE_DIR) / "tools" / "n3mapping_loop_debug_analyze.py";
    if (std::filesystem::exists(source_tool)) return source_tool;
    return {};
}

std::filesystem::path findEvalMatrixTool()
{
    const std::filesystem::path source_tool =
        std::filesystem::path(N3MAPPING_SOURCE_DIR) / "tools" / "n3mapping_eval_matrix.py";
    if (std::filesystem::exists(source_tool)) return source_tool;
    return {};
}

void writeFakeBin(const std::filesystem::path& path, int frame_index)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    for (int ring = 0; ring < 8; ++ring) {
        for (int sector = 0; sector < 24; ++sector) {
            const float angle = static_cast<float>(sector) * static_cast<float>(2.0 * M_PI / 24.0);
            const float radius = 3.0f + 0.15f * static_cast<float>(ring);
            const float values[4] = {
                radius * std::cos(angle),
                radius * std::sin(angle),
                -0.5f + 0.2f * static_cast<float>(ring),
                static_cast<float>(frame_index + ring)
            };
            output.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
    }
    ASSERT_TRUE(output.good());
}

std::filesystem::path makeMiniKitti360Fixture(
    const std::string& sequence,
    const std::string& calibration = "1 0 0 0 0 1 0 0 0 0 1 0\n")
{
    const auto root = makeTempDir("n3mapping_kitti360_eval_fixture");
    const auto lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data";
    const auto pose_dir = root / "data_poses" / sequence;
    const auto calibration_dir = root / "calibration";
    std::filesystem::create_directories(lidar_dir);
    std::filesystem::create_directories(pose_dir);
    std::filesystem::create_directories(calibration_dir);

    std::ofstream poses(pose_dir / "poses.txt");
    if (!poses.is_open()) {
        throw std::runtime_error("failed to open synthetic poses.txt");
    }
    for (int i = 0; i < 6; ++i) {
        const int frame_id = i + 1;
        const std::string frame_name = "000000000" + std::to_string(frame_id);
        writeFakeBin(lidar_dir / (frame_name + ".bin"), i);
        poses << frame_id << " 1 0 0 " << static_cast<double>(i) * 1.5
              << " 0 1 0 0"
              << " 0 0 1 0\n";
    }
    std::ofstream calib(calibration_dir / "calib_cam_to_velo.txt");
    if (!calib.is_open()) {
        throw std::runtime_error("failed to open synthetic calibration");
    }
    calib << calibration;
    return root;
}

}  // namespace

TEST(N3MappingKitti360EvalTest, MappingLoopWritesEvaluationArtifacts)
{
    const auto tool = findKittiEvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_eval executable not found";
    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeMiniKitti360Fixture(sequence);
    const auto output = makeTempDir("n3mapping_kitti360_eval_mapping_output");

    const std::string command = shellQuote(tool) +
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 5"
        " --stride 1"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    EXPECT_TRUE(std::filesystem::exists(output / "metrics.json"));
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_est.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_gt.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "keyframes_gt.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "accepted_loops.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "loop_debug.jsonl"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"mode\": \"mapping_loop\""), std::string::npos);
    EXPECT_NE(metrics.find("\"frames_processed\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"accepted_keyframes\""), std::string::npos);
    EXPECT_NE(metrics.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_lidar_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_gt_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_matched_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_selected_count\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_time_diff_max_s\": 0"), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_loaded\": true"), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_mode_requested\": \"auto\""), std::string::npos);
    const std::string loops = readTextFile(output / "accepted_loops.csv");
    EXPECT_NE(loops.find("query_id,match_id,fitness_score,inlier_ratio,verified,edge_mode,vertical_observability_score,vertical_downweighted,source_z_span,target_z_span,z_overlap_ratio_before,z_overlap_ratio_after,source_z_robust_span,target_z_robust_span,z_robust_overlap_ratio_before,z_robust_overlap_ratio_after,source_target_z_centroid_delta_before,source_target_z_centroid_delta_after,vertical_information_ratio"), std::string::npos);
    EXPECT_NE(loops.find("vertical_hypothesis_count,best_z_offset_m,best_z_offset_fitness,zero_z_fitness,fitness_gap_zero_vs_best,z_hypothesis_spread_m,vertical_ambiguity_score,vertical_hypothesis_edge_recommendation,heightmap_overlap_cell_count"), std::string::npos);
    const std::string keyframes_gt = readTextFile(output / "keyframes_gt.csv");
    EXPECT_NE(keyframes_gt.find("keyframe_id,frame_id,x,y,z,qx,qy,qz,qw"), std::string::npos);
}

TEST(N3MappingKitti360EvalTest, LoopDebugAnalyzerLabelsCandidatesWithGroundTruth)
{
    const auto analyzer = findLoopDebugAnalyzer();
    ASSERT_FALSE(analyzer.empty()) << "n3mapping_loop_debug_analyze.py not found";
    const auto input = makeTempDir("n3mapping_loop_debug_analyze_input");
    const auto output = makeTempDir("n3mapping_loop_debug_analyze_output");

    {
        std::ofstream keyframes(input / "keyframes_gt.csv");
        ASSERT_TRUE(keyframes.is_open());
        keyframes << "keyframe_id,frame_id,x,y,z,qx,qy,qz,qw\n";
        keyframes << "0,10,0,0,0,0,0,0,1\n";
        keyframes << "1,20,1,0,0,0,0,0,1\n";
        keyframes << "2,30,20,0,0,0,0,0,1\n";
        keyframes << "3,40,0.5,0.2,0,0,0,0,1\n";
        keyframes << "4,50,0.2,0,0,0,0,0,1\n";
    }
    {
        std::ofstream debug(input / "loop_debug.jsonl");
        ASSERT_TRUE(debug.is_open());
        debug << "{\"record_type\":\"candidate\",\"query_id\":3,\"match_id\":0,"
              << "\"candidate_source\":\"rhpd_primary\",\"gate_result\":\"accepted\","
              << "\"reject_reason\":\"\",\"fitness_score\":0.1,\"inlier_ratio\":0.9,"
              << "\"residual_z\":0.8,\"residual_roll\":0,\"residual_pitch\":0,"
              << "\"residual_yaw\":0,\"measurement_x\":0.5,\"measurement_y\":0.2,"
              << "\"measurement_z\":0.8,\"measurement_roll\":0,\"measurement_pitch\":0,"
              << "\"measurement_yaw\":0,\"edge_mode\":\"planar_xy_yaw\","
              << "\"vertical_observability_score\":0.25,\"vertical_downweighted\":true,"
              << "\"vertical_hypothesis_count\":7,\"best_z_offset_m\":-0.5,"
              << "\"best_z_offset_fitness\":0.11,\"zero_z_fitness\":0.12,"
              << "\"fitness_gap_zero_vs_best\":0.01,\"z_hypothesis_spread_m\":1.0,"
              << "\"vertical_ambiguity_score\":0.25,"
              << "\"vertical_hypothesis_edge_recommendation\":\"planar_xy_yaw\","
              << "\"heightmap_overlap_cell_count\":4,\"heightmap_overlap_ratio\":1.0,"
              << "\"heightmap_ground_dz_median\":0.8,\"heightmap_ground_dz_p90\":0.8,"
              << "\"heightmap_ground_dz_max\":0.8,\"heightmap_ground_support_ratio\":1.0,"
              << "\"heightmap_vertical_consistency_score\":0.55}\n";
        debug << "{\"record_type\":\"candidate\",\"query_id\":2,\"match_id\":0,"
              << "\"candidate_source\":\"rhpd_primary\",\"gate_result\":\"accepted\","
              << "\"reject_reason\":\"\",\"fitness_score\":0.1,\"inlier_ratio\":0.9,"
              << "\"residual_z\":0.0,\"residual_roll\":0,\"residual_pitch\":0,"
              << "\"residual_yaw\":0,\"measurement_x\":20,\"measurement_y\":0,"
              << "\"measurement_z\":0,\"measurement_roll\":0,\"measurement_pitch\":0,"
              << "\"measurement_yaw\":0,\"edge_mode\":\"full6dof\","
              << "\"vertical_observability_score\":1,\"vertical_downweighted\":false}\n";
        debug << "{\"record_type\":\"candidate\",\"query_id\":1,\"match_id\":0,"
              << "\"candidate_source\":\"rhpd_primary\",\"gate_result\":\"rejected\","
              << "\"reject_reason\":\"fitness_threshold\",\"fitness_score\":2.0,"
              << "\"inlier_ratio\":0.1,\"residual_z\":0.0,\"residual_roll\":0,"
              << "\"residual_pitch\":0,\"residual_yaw\":0,\"measurement_x\":1,"
              << "\"measurement_y\":0,\"measurement_z\":0,\"measurement_roll\":0,"
              << "\"measurement_pitch\":0,\"measurement_yaw\":0}\n";
        debug << "{\"record_type\":\"candidate\",\"query_id\":4,\"match_id\":2,"
              << "\"candidate_source\":\"rhpd_primary\",\"gate_result\":\"accepted\","
              << "\"reject_reason\":\"\",\"fitness_score\":0.05,\"inlier_ratio\":0.9,"
              << "\"residual_z\":0.0,\"residual_roll\":0,\"residual_pitch\":0,"
              << "\"residual_yaw\":0,\"measurement_x\":-19.8,\"measurement_y\":0,"
              << "\"measurement_z\":0,\"measurement_roll\":0,\"measurement_pitch\":0,"
              << "\"measurement_yaw\":0,\"edge_mode\":\"full6dof\","
              << "\"vertical_observability_score\":1,\"vertical_downweighted\":false}\n";
        debug << "{\"record_type\":\"candidate\",\"query_id\":4,\"match_id\":0,"
              << "\"candidate_source\":\"rhpd_primary\",\"gate_result\":\"rejected\","
              << "\"reject_reason\":\"not_selected\",\"fitness_score\":0.1,\"inlier_ratio\":0.9,"
              << "\"residual_z\":0.0,\"residual_roll\":0,\"residual_pitch\":0,"
              << "\"residual_yaw\":0,\"measurement_x\":0.2,\"measurement_y\":0,"
              << "\"measurement_z\":0,\"measurement_roll\":0,\"measurement_pitch\":0,"
              << "\"measurement_yaw\":0}\n";
    }
    {
        std::ofstream accepted(input / "accepted_loops.csv");
        ASSERT_TRUE(accepted.is_open());
        accepted << "query_id,match_id,fitness_score,inlier_ratio,verified\n";
        accepted << "3,0,0.1,0.9,true\n";
        accepted << "2,0,0.1,0.9,true\n";
        accepted << "4,2,0.05,0.9,true\n";
    }

    const std::string command =
        "python3 " + shellQuote(analyzer) +
        " --loop_debug " + shellQuote(input / "loop_debug.jsonl") +
        " --keyframes_gt " + shellQuote(input / "keyframes_gt.csv") +
        " --accepted_loops " + shellQuote(input / "accepted_loops.csv") +
        " --output " + shellQuote(output) +
        " --loop_translation_threshold 5"
        " --loop_yaw_threshold_deg 45"
        " --min_id_gap 1"
        " --z_drift_threshold 0.5";
    ASSERT_EQ(std::system(command.c_str()), 0);

    const std::string diagnosis = readTextFile(output / "loop_diagnosis.json");
    EXPECT_NE(diagnosis.find("\"candidate_count\": 5"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_true_loop\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_false_loop\": 2"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"icp_reject_true_loop\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"verification_reject_true_loop\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"true_loop_not_selected\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_true_loop_bad_z\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_true_loop_good\": 0"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_planar_xy_yaw\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"vertical_hypothesis_candidate_count\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_true_loop_bad_z_planar_recommended\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"heightmap_candidate_count\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_true_loop_bad_z_heightmap_high\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"heightmap_separates_bad_z_count\": 0"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"accepted_full6dof\": 2"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"query_selection_failure_count\": 1"), std::string::npos);
    EXPECT_NE(diagnosis.find("\"z_drift_suspect_count\": 1"), std::string::npos);

    const std::string labeled = readTextFile(output / "loop_candidates_labeled.csv");
    EXPECT_NE(labeled.find("accepted_true_loop"), std::string::npos);
    EXPECT_NE(labeled.find("accepted_false_loop"), std::string::npos);
    EXPECT_NE(labeled.find("accepted_true_loop_bad_z"), std::string::npos);
    EXPECT_NE(labeled.find("verification_reject_true_loop"), std::string::npos);
    EXPECT_NE(labeled.find("planar_xy_yaw"), std::string::npos);
    EXPECT_NE(labeled.find("icp_error_to_gt_z"), std::string::npos);
    EXPECT_NE(labeled.find("vertical_hypothesis_edge_recommendation"), std::string::npos);
    EXPECT_NE(labeled.find("heightmap_ground_dz_p90"), std::string::npos);

    const std::string query_summary = readTextFile(output / "loop_query_summary.csv");
    EXPECT_NE(query_summary.find("selection_failure"), std::string::npos);
    EXPECT_NE(query_summary.find("4,2,1,1,2,False"), std::string::npos);
}

TEST(N3MappingKitti360EvalTest, EvalMatrixSummarizesRunArtifacts)
{
    const auto matrix_tool = findEvalMatrixTool();
    ASSERT_FALSE(matrix_tool.empty()) << "n3mapping_eval_matrix.py not found";
    const auto run = makeTempDir("n3mapping_eval_matrix_run");
    const auto analysis = run / "loop_gt_analysis";
    const auto output = makeTempDir("n3mapping_eval_matrix_output");
    std::filesystem::create_directories(analysis);

    {
        std::ofstream metrics(run / "metrics.json");
        ASSERT_TRUE(metrics.is_open());
        metrics << "{\n"
                << "  \"mode\": \"mapping_loop\",\n"
                << "  \"sequence\": \"synthetic_sequence\",\n"
                << "  \"odom_source\": \"gt\",\n"
                << "  \"frames_processed\": 3,\n"
                << "  \"accepted_keyframes\": 2,\n"
                << "  \"accepted_loop_count\": 2,\n"
                << "  \"dense_trajectory_count\": 3,\n"
                << "  \"alignment_input_lidar_count\": 3,\n"
                << "  \"alignment_input_gt_count\": 3,\n"
                << "  \"alignment_matched_count\": 3,\n"
                << "  \"alignment_selected_count\": 3,\n"
                << "  \"alignment_dropped_lidar_count\": 0,\n"
                << "  \"alignment_dropped_gt_count\": 0,\n"
                << "  \"alignment_time_diff_median_s\": 0,\n"
                << "  \"alignment_time_diff_p95_s\": 0,\n"
                << "  \"alignment_time_diff_max_s\": 0\n"
                << "}\n";
    }
    {
        std::ofstream est(run / "trajectory_est.txt");
        std::ofstream gt(run / "trajectory_gt.txt");
        ASSERT_TRUE(est.is_open());
        ASSERT_TRUE(gt.is_open());
        gt << "1 0 0 0 0 0 0 1\n";
        gt << "2 1 0 0 0 0 0 1\n";
        gt << "3 2 0 0 0 0 0 1\n";
        est << "1 0 0 0 0 0 0 1\n";
        est << "2 1.2 0 0.1 0 0 0 1\n";
        est << "3 2.4 0 0.2 0 0 0 1\n";
    }
    {
        std::ofstream diagnosis(analysis / "loop_diagnosis.json");
        ASSERT_TRUE(diagnosis.is_open());
        diagnosis << "{\n"
                  << "  \"candidate_count\": 4,\n"
                  << "  \"gt_loop_pair_count\": 2,\n"
                  << "  \"accepted_candidate_count\": 2,\n"
                  << "  \"accepted_true_loop\": 2,\n"
                  << "  \"accepted_true_loop_good\": 1,\n"
                  << "  \"accepted_true_loop_bad_z\": 1,\n"
                  << "  \"accepted_true_loop_bad_z_measurement\": 1,\n"
                  << "  \"accepted_true_loop_bad_z_after\": 1,\n"
                  << "  \"accepted_true_loop_corrected_z\": 0,\n"
                  << "  \"accepted_true_loop_bad_roll_pitch\": 0,\n"
                  << "  \"accepted_false_loop\": 0,\n"
                  << "  \"accepted_full6dof\": 1,\n"
                  << "  \"accepted_planar_xy_yaw\": 1,\n"
                  << "  \"vertical_hypothesis_candidate_count\": 1,\n"
                  << "  \"vertical_hypothesis_planar_recommendation_count\": 1,\n"
                  << "  \"vertical_hypothesis_full6dof_recommendation_count\": 0,\n"
                  << "  \"accepted_true_loop_bad_z_with_vertical_hypothesis\": 1,\n"
                  << "  \"accepted_true_loop_bad_z_planar_recommended\": 1,\n"
                  << "  \"heightmap_candidate_count\": 1,\n"
                  << "  \"accepted_true_loop_bad_z_heightmap_high\": 1,\n"
                  << "  \"heightmap_separates_bad_z_count\": 1,\n"
                  << "  \"icp_reject_true_loop\": 1,\n"
                  << "  \"verification_reject_true_loop\": 1,\n"
                  << "  \"true_loop_not_selected\": 0,\n"
                  << "  \"retrieval_false_positive\": 2,\n"
                  << "  \"retrieval_miss_estimate\": 1,\n"
                  << "  \"z_drift_suspect_count\": 1,\n"
                  << "  \"optimization_summary_count\": 2,\n"
                  << "  \"optimization_high_residual_z_after_count\": 1,\n"
                  << "  \"optimization_max_residual_z_after\": 0.8,\n"
                  << "  \"failure_class_counts\": {\n"
                  << "    \"accepted_true_loop_good\": 1,\n"
                  << "    \"accepted_true_loop_bad_z\": 1,\n"
                  << "    \"verification_reject_true_loop\": 1\n"
                  << "  },\n"
                  << "  \"edge_mode_counts\": {\n"
                  << "    \"full6dof\": 1,\n"
                  << "    \"planar_xy_yaw\": 1\n"
                  << "  }\n"
                  << "}\n";
    }

    const std::string command =
        "python3 " + shellQuote(matrix_tool) +
        " --run baseline=" + shellQuote(run) +
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    const std::string csv = readTextFile(output / "matrix_summary.csv");
    EXPECT_NE(csv.find("baseline"), std::string::npos);
    EXPECT_NE(csv.find("synthetic_sequence"), std::string::npos);
    EXPECT_NE(csv.find("true,false"), std::string::npos);
    const std::string json = readTextFile(output / "matrix_summary.json");
    EXPECT_NE(json.find("\"loop_precision\": 1.0"), std::string::npos);
    EXPECT_NE(json.find("\"loop_accepted_true_loop_bad_z\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"loop_accepted_true_loop_bad_z_planar_recommended\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"loop_accepted_true_loop_bad_z_heightmap_high\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"loop_verification_reject_true_loop\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"loop_accepted_planar_xy_yaw\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"trajectory_pair_count\": 3"), std::string::npos);
    EXPECT_NE(json.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(json.find("\"alignment_matched_count\": 3"), std::string::npos);
}

TEST(N3MappingKitti360EvalTest, CalibrationModeChangesGroundTruthPoseAndMetrics)
{
    const auto tool = findKittiEvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_eval executable not found";
    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeMiniKitti360Fixture(sequence, "1 0 0 10 0 1 0 0 0 0 1 0\n");
    const auto cam_to_velo_output = makeTempDir("n3mapping_kitti360_eval_cam_to_velo_output");
    const auto velo_to_cam_output = makeTempDir("n3mapping_kitti360_eval_velo_to_cam_output");

    const std::string common =
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 1"
        " --stride 1";

    const std::string direct_command = shellQuote(tool) + common +
        " --calib_mode cam_to_velo"
        " --output " + shellQuote(cam_to_velo_output);
    ASSERT_EQ(std::system(direct_command.c_str()), 0);

    const std::string inverse_command = shellQuote(tool) + common +
        " --calib_mode velo_to_cam"
        " --output " + shellQuote(velo_to_cam_output);
    ASSERT_EQ(std::system(inverse_command.c_str()), 0);

    const std::string direct_traj = readTextFile(cam_to_velo_output / "trajectory_gt.txt");
    const std::string inverse_traj = readTextFile(velo_to_cam_output / "trajectory_gt.txt");
    EXPECT_NE(direct_traj.find("1 10.000000000"), std::string::npos);
    EXPECT_NE(inverse_traj.find("1 -10.000000000"), std::string::npos);

    const std::string direct_metrics = readTextFile(cam_to_velo_output / "metrics.json");
    const std::string inverse_metrics = readTextFile(velo_to_cam_output / "metrics.json");
    EXPECT_NE(direct_metrics.find("\"calib_mode_used\": \"cam_to_velo\""), std::string::npos);
    EXPECT_NE(inverse_metrics.find("\"calib_mode_used\": \"velo_to_cam\""), std::string::npos);
    EXPECT_NE(direct_metrics.find("\"first_pose_delta_translation_m\": 20"), std::string::npos);
}

TEST(N3MappingKitti360EvalTest, MalformedCalibrationIsReportedInMetrics)
{
    const auto tool = findKittiEvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_eval executable not found";
    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeMiniKitti360Fixture(sequence, "malformed calibration\n");
    const auto output = makeTempDir("n3mapping_kitti360_eval_bad_calib_output");

    const std::string command = shellQuote(tool) +
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 1"
        " --stride 1"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"calib_loaded\": false"), std::string::npos);
    EXPECT_NE(metrics.find("failed to parse calib_cam_to_velo.txt"), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_mode_used\": \"identity\""), std::string::npos);
}

TEST(N3MappingKitti360EvalTest, RelocalizationWritesMetricsAndDebug)
{
    const auto tool = findKittiEvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_eval executable not found";
    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeMiniKitti360Fixture(sequence);
    const auto output = makeTempDir("n3mapping_kitti360_eval_reloc_output");

    const std::string command = shellQuote(tool) +
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode relocalization"
        " --max_frames 6"
        " --stride 1"
        " --build_map_frames 3"
        " --dropout 0.1"
        " --noise 0.01"
        " --fake_yaw 15"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    EXPECT_TRUE(std::filesystem::exists(output / "metrics.json"));
    EXPECT_TRUE(std::filesystem::exists(output / "relocalization_debug.jsonl"));
    EXPECT_TRUE(std::filesystem::exists(output / "relocalization_queries.csv"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"mode\": \"relocalization\""), std::string::npos);
    EXPECT_NE(metrics.find("\"query_count\": 3"), std::string::npos);
    EXPECT_NE(metrics.find("\"lock_success_rate\""), std::string::npos);
    EXPECT_NE(metrics.find("\"pose_success_rate\""), std::string::npos);
    EXPECT_NE(metrics.find("\"lock_precision\""), std::string::npos);
    EXPECT_NE(metrics.find("\"false_lock_rate\""), std::string::npos);
    EXPECT_NE(metrics.find("\"pose_error_at_lock_p95_m\""), std::string::npos);
    EXPECT_NE(metrics.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_matched_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"median_translation_error_m\""), std::string::npos);
    EXPECT_NE(metrics.find("\"p95_yaw_error_deg\""), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_loaded\": true"), std::string::npos);
    const std::string queries = readTextFile(output / "relocalization_queries.csv");
    EXPECT_NE(queries.find("pose_success,lock_correct,false_lock,lock_latency_frames,failure_class"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
