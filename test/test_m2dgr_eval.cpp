#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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

std::filesystem::path findM2DGREvalTool()
{
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::filesystem::path dir = self.parent_path();
    const std::filesystem::path candidates[] = {
        dir / "n3mapping_m2dgr_eval",
        dir.parent_path() / "n3mapping_m2dgr_eval",
        std::filesystem::current_path() / "n3mapping_m2dgr_eval",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

void writeFakePcd(const std::filesystem::path& path, int frame_index)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    ASSERT_TRUE(out.is_open());
    const int rings = 8;
    const int sectors = 24;
    const int points = rings * sectors;
    out << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n"
        << "FIELDS x y z intensity\n"
        << "SIZE 4 4 4 4\n"
        << "TYPE F F F F\n"
        << "COUNT 1 1 1 1\n"
        << "WIDTH " << points << "\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << points << "\n"
        << "DATA ascii\n";
    for (int ring = 0; ring < rings; ++ring) {
        for (int sector = 0; sector < sectors; ++sector) {
            const double angle = static_cast<double>(sector) * 2.0 * 3.14159265358979323846 / sectors;
            const double radius = 3.0 + 0.1 * ring;
            out << radius * std::cos(angle) << ' '
                << radius * std::sin(angle) << ' '
                << -0.4 + 0.15 * ring << ' '
                << frame_index + ring << '\n';
        }
    }
}

std::filesystem::path makeMiniM2DGRFixture(const std::string& sequence)
{
    const auto root = makeTempDir("n3mapping_m2dgr_fixture");
    const auto cloud_dir = root / sequence / "velodyne_points";
    std::filesystem::create_directories(cloud_dir);
    std::ofstream gt(root / sequence / "groundtruth.txt");
    if (!gt.is_open()) {
        throw std::runtime_error("failed to write synthetic M2DGR GT");
    }
    for (int i = 0; i < 6; ++i) {
        const double timestamp = 1000.0 + 0.1 * static_cast<double>(i);
        std::ostringstream name;
        name.setf(std::ios::fixed);
        name.precision(9);
        name << timestamp << ".pcd";
        writeFakePcd(cloud_dir / name.str(), i);
        gt.setf(std::ios::fixed);
        gt.precision(9);
        gt << timestamp << ' '
           << static_cast<double>(i) * 1.5 << " 0 0 0 0 0 1\n";
    }
    return root;
}

std::filesystem::path writeMiniM2DGRCalibration(
    const std::filesystem::path& root)
{
    const auto path = root / "calibration_results.txt";
    std::ofstream calibration(path);
    if (!calibration.is_open()) {
        throw std::runtime_error("failed to write synthetic M2DGR calibration");
    }
    calibration
        << "%% Xsens IMU\n"
        << "% Extrinsic [to LIDAR]\n"
        << "data: [1, 0, 0, 0.15905,\n"
        << "       0, 1, 0, 0.00067,\n"
        << "       0, 0, 1, -0.16824]\n"
        << "%% leica\n"
        << "% Extrinsic [to LIDAR]\n"
        << "data: [1, 0, 0, -0.21374,\n"
        << "       0, 1, 0, 0.00146,\n"
        << "       0, 0, 1, 0.68356]\n";
    return path;
}

void writeRotatedEcefLikeGroundTruth(const std::filesystem::path& root,
                                     const std::string& sequence)
{
    std::ofstream gt(root / sequence / "groundtruth.txt");
    ASSERT_TRUE(gt.is_open());
    const double origin_x = -2853538.25608;
    const double origin_y = 4667400.61047;
    const double origin_z = 3268263.155;
    const double longitude = std::atan2(origin_y, origin_x);
    const double east_x = -std::sin(longitude);
    const double east_y = std::cos(longitude);
    const double half_sqrt = std::sqrt(0.5);
    for (int i = 0; i < 6; ++i) {
        const double timestamp = 1000.0 + 0.1 * static_cast<double>(i);
        gt.setf(std::ios::fixed);
        gt.precision(9);
        gt << timestamp << ' '
           << origin_x + east_x * static_cast<double>(i) << ' '
           << origin_y + east_y * static_cast<double>(i) << ' '
           << origin_z << ' '
           << "0 0 " << half_sqrt << ' ' << half_sqrt << '\n';
    }
}

TEST(N3MappingM2DGREvalTest, MappingLoopWritesMatrixCompatibleArtifacts)
{
    const auto tool = findM2DGREvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_m2dgr_eval executable not found";
    const std::string sequence = "hall_03";
    const auto root = makeMiniM2DGRFixture(sequence);
    const auto output = makeTempDir("n3mapping_m2dgr_mapping_output");

    const std::string command = shellQuote(tool) +
        " --m2dgr_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 5"
        " --stride 1"
        " --max_time_diff 0.001"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    EXPECT_TRUE(std::filesystem::exists(output / "metrics.json"));
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_est.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_optimized.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_gt.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "keyframes_gt.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "accepted_loops.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "loop_debug.jsonl"));
    EXPECT_TRUE(std::filesystem::exists(output / "n3map.pbstream"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"dataset\": \"M2DGR\""), std::string::npos);
    EXPECT_NE(metrics.find("\"mode\": \"mapping_loop\""), std::string::npos);
    EXPECT_NE(metrics.find("\"frames_processed\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(metrics.find("\"backend_input_contract\": \"gt_pose_plus_lidar\""),
              std::string::npos);
    EXPECT_NE(metrics.find("\"real_lio_safety_filters_applied\": false"),
              std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_lidar_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_gt_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_matched_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_selected_count\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_time_diff_max_s\": 0"), std::string::npos);
    EXPECT_NE(metrics.find("\"trajectory_optimized_semantics\": \"final_dense_after_all_loop_updates\""),
              std::string::npos);
    const std::string loops = readTextFile(output / "accepted_loops.csv");
    EXPECT_NE(loops.find("vertical_hypothesis_count,best_z_offset_m,best_z_offset_fitness,zero_z_fitness,fitness_gap_zero_vs_best,z_hypothesis_spread_m,vertical_ambiguity_score,vertical_hypothesis_edge_recommendation,heightmap_overlap_cell_count"), std::string::npos);
    EXPECT_NE(loops.find("graph_trial_success,graph_trial_residual_x_after"), std::string::npos);
}

TEST(N3MappingM2DGREvalTest, EpisodeManifestSelectsExactMappingFrames)
{
    const auto tool = findM2DGREvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_m2dgr_eval executable not found";
    const std::string sequence = "hall_03";
    const auto root = makeMiniM2DGRFixture(sequence);
    const auto output = makeTempDir("n3mapping_m2dgr_manifest_output");
    const auto manifest_dir = makeTempDir("n3mapping_m2dgr_manifest");
    const auto manifest = manifest_dir / "episode_frames.csv";
    std::ofstream frames(manifest);
    ASSERT_TRUE(frames.is_open());
    frames << "episode_id,role,frame_token\n"
           << "map,map,1000.000000000\n"
           << "map,map,1000.200000000\n"
           << "map,map,1000.400000000\n";
    frames.close();

    const std::string command = shellQuote(tool) +
        " --m2dgr_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_time_diff 0.001"
        " --frame_manifest " + shellQuote(manifest) +
        " --episode_id map"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"frames_processed\": 3"), std::string::npos);
    EXPECT_NE(metrics.find("\"episode_id\": \"map\""), std::string::npos);
    EXPECT_TRUE(std::filesystem::exists(output / "n3map.pbstream"));
}

TEST(N3MappingM2DGREvalTest, NormalizeGtOriginUsesFullFirstPoseFrame)
{
    const auto tool = findM2DGREvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_m2dgr_eval executable not found";
    const std::string sequence = "gate_02";
    const auto root = makeMiniM2DGRFixture(sequence);
    writeRotatedEcefLikeGroundTruth(root, sequence);
    const auto output = makeTempDir("n3mapping_m2dgr_local_frame_output");

    const std::string command = shellQuote(tool) +
        " --m2dgr_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 2"
        " --stride 1"
        " --max_time_diff 0.001"
        " --normalize_gt_origin"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    std::ifstream trajectory(output / "trajectory_gt.txt");
    ASSERT_TRUE(trajectory.is_open());
    double stamp = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 0.0;
    ASSERT_TRUE(trajectory >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw);
    EXPECT_NEAR(x, 0.0, 1e-6);
    EXPECT_NEAR(y, 0.0, 1e-6);
    EXPECT_NEAR(z, 0.0, 1e-6);
    EXPECT_NEAR(qz, 0.0, 1e-6);
    EXPECT_NEAR(qw, 1.0, 1e-6);
    ASSERT_TRUE(trajectory >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw);
    const double longitude = std::atan2(4667400.61047, -2853538.25608);
    EXPECT_NEAR(x, std::cos(longitude), 1e-5);
    EXPECT_NEAR(y, std::sin(longitude), 1e-5);
    EXPECT_NEAR(z, 0.0, 1e-6);
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"gt_position_frame\": \"ecef_to_enu_first_pose\""),
              std::string::npos);
}

TEST(N3MappingM2DGREvalTest, AppliesPinnedGroundTruthSensorToLidarCalibration)
{
    const auto tool = findM2DGREvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_m2dgr_eval executable not found";
    const std::string sequence = "gate_02";
    const auto root = makeMiniM2DGRFixture(sequence);
    const auto calibration = writeMiniM2DGRCalibration(root);
    const auto output = makeTempDir("n3mapping_m2dgr_calibrated_output");

    const std::string command = shellQuote(tool) +
        " --m2dgr_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode mapping_loop"
        " --max_frames 1"
        " --max_time_diff 0.001"
        " --gt_sensor_frame xsens"
        " --calibration_file " + shellQuote(calibration) +
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    std::ifstream trajectory(output / "trajectory_gt.txt");
    ASSERT_TRUE(trajectory.is_open());
    double stamp = 0.0, x = 0.0, y = 0.0, z = 0.0;
    double qx = 0.0, qy = 0.0, qz = 0.0, qw = 0.0;
    ASSERT_TRUE(trajectory >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw);
    EXPECT_NEAR(x, -0.15905, 1e-6);
    EXPECT_NEAR(y, -0.00067, 1e-6);
    EXPECT_NEAR(z, 0.16824, 1e-6);
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"gt_sensor_frame\": \"xsens\""), std::string::npos);
    EXPECT_NE(metrics.find("\"gt_to_lidar_calibration_applied\": true"),
              std::string::npos);
    EXPECT_NE(metrics.find("\"gt_to_lidar_calibration_sha256\": \""),
              std::string::npos);
}

TEST(N3MappingM2DGREvalTest, RelocalizationWritesMetricsAndDebug)
{
    const auto tool = findM2DGREvalTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_m2dgr_eval executable not found";
    const std::string sequence = "hall_03";
    const auto root = makeMiniM2DGRFixture(sequence);
    const auto output = makeTempDir("n3mapping_m2dgr_reloc_output");

    const std::string command = shellQuote(tool) +
        " --m2dgr_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --mode relocalization"
        " --max_frames 6"
        " --stride 1"
        " --max_time_diff 0.001"
        " --build_map_frames 3"
        " --dropout 0.05"
        " --noise 0.01"
        " --fake_x 20"
        " --fake_y -10"
        " --fake_yaw 10"
        " --output " + shellQuote(output);
    ASSERT_EQ(std::system(command.c_str()), 0);

    EXPECT_TRUE(std::filesystem::exists(output / "metrics.json"));
    EXPECT_TRUE(std::filesystem::exists(output / "relocalization_debug.jsonl"));
    EXPECT_TRUE(std::filesystem::exists(output / "relocalization_queries.csv"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"dataset\": \"M2DGR\""), std::string::npos);
    EXPECT_NE(metrics.find("\"mode\": \"relocalization\""), std::string::npos);
    EXPECT_NE(metrics.find("\"query_count\": 3"), std::string::npos);
    EXPECT_NE(metrics.find("\"pose_success_rate\""), std::string::npos);
    EXPECT_NE(metrics.find("\"lock_precision\""), std::string::npos);
    EXPECT_NE(metrics.find("\"false_lock_rate\""), std::string::npos);
    EXPECT_NE(metrics.find("\"pose_error_at_lock_p95_m\""), std::string::npos);
    EXPECT_NE(metrics.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_matched_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"fake_x_m\": 20"), std::string::npos);
    EXPECT_NE(metrics.find("\"fake_y_m\": -10"), std::string::npos);
    const std::string queries = readTextFile(output / "relocalization_queries.csv");
    EXPECT_NE(queries.find("relocalization_state,pose_source"),
              std::string::npos);
    EXPECT_NE(queries.find("pose_success,lock_correct,false_lock,lock_latency_frames,failure_class"), std::string::npos);
}

}  // namespace
}  // namespace test
}  // namespace n3mapping
