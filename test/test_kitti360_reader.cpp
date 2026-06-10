#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace n3mapping {
namespace test {
namespace {

std::filesystem::path makeTempDir(const std::string& name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        (name + "_" + std::to_string(static_cast<long long>(stamp)));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string shellQuote(const std::filesystem::path& path) {
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

std::filesystem::path findKittiReaderTool() {
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::filesystem::path dir = self.parent_path();
    const std::filesystem::path candidates[] = {
        dir / "n3mapping_kitti360_reader",
        dir.parent_path() / "n3mapping_kitti360_reader",
        std::filesystem::current_path() / "n3mapping_kitti360_reader",
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

void writeFakeBin(const std::filesystem::path& path, const std::vector<float>& values) {
    ASSERT_EQ(values.size() % 4, 0U);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    ASSERT_TRUE(output.good());
}

}  // namespace

TEST(N3MappingKitti360ReaderTest, AlignsByFrameIdAndWritesSummaryAndPcd) {
    const auto tool = findKittiReaderTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_reader executable not found";

    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeTempDir("n3mapping_kitti360_reader_root");
    const auto output = makeTempDir("n3mapping_kitti360_reader_output");
    const auto lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data";
    const auto pose_dir = root / "data_poses" / sequence;
    const auto calibration_dir = root / "calibration";
    std::filesystem::create_directories(lidar_dir);
    std::filesystem::create_directories(pose_dir);
    std::filesystem::create_directories(calibration_dir);

    writeFakeBin(lidar_dir / "0000000001.bin", {1.0f, 2.0f, 3.0f, 0.5f});
    writeFakeBin(lidar_dir / "0000000002.bin", {2.0f, 3.0f, 4.0f, 0.6f});
    writeFakeBin(lidar_dir / "0000000004.bin", {4.0f, 5.0f, 6.0f, 0.7f});

    {
        std::ofstream poses(pose_dir / "poses.txt");
        ASSERT_TRUE(poses.is_open());
        poses << "1 1 0 0 0 0 1 0 0 0 0 1 0\n";
        poses << "3 1 0 0 3 0 1 0 0 0 0 1 0\n";
        poses << "4 1 0 0 4 0 1 0 0 0 0 1 0\n";
    }
    {
        std::ofstream calib(calibration_dir / "calib_cam_to_velo.txt");
        ASSERT_TRUE(calib.is_open());
        calib << "calib placeholder\n";
    }

    const std::string command = shellQuote(tool) +
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --output " + shellQuote(output) +
        " --max_frames 10 --dump_first_n 2";
    ASSERT_EQ(std::system(command.c_str()), 0);

    const auto summary_path = output / "summary.json";
    ASSERT_TRUE(std::filesystem::exists(summary_path));
    const std::string summary = readTextFile(summary_path);
    EXPECT_NE(summary.find("\"sequence\": \"" + sequence + "\""), std::string::npos);
    EXPECT_NE(summary.find("\"lidar_bin_count\": 3"), std::string::npos);
    EXPECT_NE(summary.find("\"pose_count\": 3"), std::string::npos);
    EXPECT_NE(summary.find("\"common_frame_count\": 2"), std::string::npos);
    EXPECT_NE(summary.find("\"total_common_frame_count\": 2"), std::string::npos);
    EXPECT_NE(summary.find("\"first_common_frame\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"last_common_frame\": 4"), std::string::npos);
    EXPECT_NE(summary.find("\"missing_pose_count\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"missing_lidar_count\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"calib_loaded\": true"), std::string::npos);

    const auto first_pcd = output / "frame_0000000001.pcd";
    const auto second_pcd = output / "frame_0000000004.pcd";
    EXPECT_TRUE(std::filesystem::exists(first_pcd));
    EXPECT_TRUE(std::filesystem::exists(second_pcd));
    const std::string pcd = readTextFile(first_pcd);
    EXPECT_NE(pcd.find("FIELDS x y z intensity"), std::string::npos);
    EXPECT_NE(pcd.find("1 2 3 0.5"), std::string::npos);
}

TEST(N3MappingKitti360ReaderTest, MaxFramesLimitsProcessedCommonFrames) {
    const auto tool = findKittiReaderTool();
    ASSERT_FALSE(tool.empty()) << "n3mapping_kitti360_reader executable not found";

    const std::string sequence = "2013_05_28_drive_0003_sync";
    const auto root = makeTempDir("n3mapping_kitti360_reader_max_root");
    const auto output = makeTempDir("n3mapping_kitti360_reader_max_output");
    const auto lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data";
    const auto pose_dir = root / "data_poses" / sequence;
    std::filesystem::create_directories(lidar_dir);
    std::filesystem::create_directories(pose_dir);

    writeFakeBin(lidar_dir / "0000000001.bin", {1.0f, 0.0f, 0.0f, 0.1f});
    writeFakeBin(lidar_dir / "0000000002.bin", {2.0f, 0.0f, 0.0f, 0.2f});
    {
        std::ofstream poses(pose_dir / "poses.txt");
        ASSERT_TRUE(poses.is_open());
        poses << "1 1 0 0 0 0 1 0 0 0 0 1 0\n";
        poses << "2 1 0 0 1 0 1 0 0 0 0 1 0\n";
    }

    const std::string command = shellQuote(tool) +
        " --kitti_root " + shellQuote(root) +
        " --sequence " + sequence +
        " --output " + shellQuote(output) +
        " --max_frames 1";
    ASSERT_EQ(std::system(command.c_str()), 0);

    const std::string summary = readTextFile(output / "summary.json");
    EXPECT_NE(summary.find("\"common_frame_count\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"total_common_frame_count\": 2"), std::string::npos);
    EXPECT_NE(summary.find("\"first_common_frame\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"last_common_frame\": 1"), std::string::npos);
    EXPECT_NE(summary.find("\"calib_loaded\": false"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
