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
    EXPECT_TRUE(std::filesystem::exists(output / "accepted_loops.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "loop_debug.jsonl"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"mode\": \"mapping_loop\""), std::string::npos);
    EXPECT_NE(metrics.find("\"frames_processed\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"accepted_keyframes\""), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_loaded\": true"), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_mode_requested\": \"auto\""), std::string::npos);
    const std::string loops = readTextFile(output / "accepted_loops.csv");
    EXPECT_NE(loops.find("query_id,match_id,fitness_score,inlier_ratio,verified"), std::string::npos);
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
    EXPECT_NE(metrics.find("\"median_translation_error_m\""), std::string::npos);
    EXPECT_NE(metrics.find("\"p95_yaw_error_deg\""), std::string::npos);
    EXPECT_NE(metrics.find("\"calib_loaded\": true"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
