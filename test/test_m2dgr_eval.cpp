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
    EXPECT_TRUE(std::filesystem::exists(output / "trajectory_gt.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "keyframes_gt.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "accepted_loops.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "loop_debug.jsonl"));
    const std::string metrics = readTextFile(output / "metrics.json");
    EXPECT_NE(metrics.find("\"dataset\": \"M2DGR\""), std::string::npos);
    EXPECT_NE(metrics.find("\"mode\": \"mapping_loop\""), std::string::npos);
    EXPECT_NE(metrics.find("\"frames_processed\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"odom_source\": \"gt\""), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_lidar_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_input_gt_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_matched_count\": 6"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_selected_count\": 5"), std::string::npos);
    EXPECT_NE(metrics.find("\"alignment_time_diff_max_s\": 0"), std::string::npos);
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
    const std::string queries = readTextFile(output / "relocalization_queries.csv");
    EXPECT_NE(queries.find("pose_success,lock_correct,false_lock,lock_latency_frames,failure_class"), std::string::npos);
}

}  // namespace
}  // namespace test
}  // namespace n3mapping
