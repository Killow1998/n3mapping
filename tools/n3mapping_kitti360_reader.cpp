#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path kitti_root;
    std::string sequence;
    fs::path output_dir;
    int64_t max_frames = -1;
    bool dump_sample_pcd = false;
    int64_t dump_first_n = 0;
};

struct PoseRecord {
    int64_t frame_id = -1;
    std::array<double, 12> values{};
};

struct PointXYZI {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
};

std::string jsonEscape(const std::string& input) {
    std::ostringstream out;
    for (const unsigned char ch : input) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

void printUsage(std::ostream& os) {
    os << "Usage: n3mapping_kitti360_reader "
       << "--kitti_root <path> --sequence <name> --output <dir> [options]\n\n"
       << "Options:\n"
       << "  --max_frames <N>       Process at most N common frame ids.\n"
       << "  --dump_sample_pcd      Dump the first selected lidar frame as PCD.\n"
       << "  --dump_first_n <N>     Dump the first N selected lidar frames as PCD.\n"
       << "  --help                 Show this help.\n";
}

int64_t parseNonNegativeInt(const std::string& value, const std::string& name) {
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < 0) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return static_cast<int64_t>(parsed);
}

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--kitti_root") {
            options.kitti_root = requireValue(arg);
        } else if (arg == "--sequence") {
            options.sequence = requireValue(arg);
        } else if (arg == "--output") {
            options.output_dir = requireValue(arg);
        } else if (arg == "--max_frames") {
            options.max_frames = parseNonNegativeInt(requireValue(arg), arg);
        } else if (arg == "--dump_sample_pcd") {
            options.dump_sample_pcd = true;
        } else if (arg == "--dump_first_n") {
            options.dump_first_n = parseNonNegativeInt(requireValue(arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.kitti_root.empty()) {
        throw std::runtime_error("--kitti_root is required");
    }
    if (options.sequence.empty()) {
        throw std::runtime_error("--sequence is required");
    }
    if (options.output_dir.empty()) {
        throw std::runtime_error("--output is required");
    }
    return options;
}

bool parseFrameIdFromBin(const fs::path& path, int64_t* frame_id) {
    if (!frame_id || path.extension() != ".bin") return false;
    const std::string stem = path.stem().string();
    if (stem.empty()) return false;
    for (const char ch : stem) {
        if (ch < '0' || ch > '9') return false;
    }
    *frame_id = parseNonNegativeInt(stem, "frame id");
    return true;
}

std::map<int64_t, fs::path> listLidarBins(const fs::path& lidar_dir) {
    if (!fs::is_directory(lidar_dir)) {
        throw std::runtime_error("lidar directory does not exist: " + lidar_dir.string());
    }

    std::map<int64_t, fs::path> bins;
    for (const auto& entry : fs::directory_iterator(lidar_dir)) {
        if (!entry.is_regular_file()) continue;
        int64_t frame_id = -1;
        if (!parseFrameIdFromBin(entry.path(), &frame_id)) continue;
        const auto inserted = bins.emplace(frame_id, entry.path());
        if (!inserted.second) {
            throw std::runtime_error("duplicate lidar frame id: " + std::to_string(frame_id));
        }
    }
    return bins;
}

std::map<int64_t, PoseRecord> readPoses(const fs::path& poses_path) {
    std::ifstream input(poses_path);
    if (!input.is_open()) {
        throw std::runtime_error("poses file does not exist: " + poses_path.string());
    }

    std::map<int64_t, PoseRecord> poses;
    std::string line;
    int64_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        PoseRecord pose;
        if (!(iss >> pose.frame_id)) {
            throw std::runtime_error("failed to parse pose frame id at line " + std::to_string(line_number));
        }
        if (pose.frame_id < 0) {
            throw std::runtime_error("negative pose frame id at line " + std::to_string(line_number));
        }
        for (double& value : pose.values) {
            if (!(iss >> value) || !std::isfinite(value)) {
                throw std::runtime_error("failed to parse finite 3x4 pose at line " + std::to_string(line_number));
            }
        }
        std::string trailing;
        if (iss >> trailing) {
            throw std::runtime_error("unexpected trailing token in poses file at line " + std::to_string(line_number));
        }
        const auto inserted = poses.emplace(pose.frame_id, pose);
        if (!inserted.second) {
            throw std::runtime_error("duplicate pose frame id: " + std::to_string(pose.frame_id));
        }
    }
    return poses;
}

bool loadCalibrationDirectory(const fs::path& calibration_dir, std::vector<std::string>* calib_files) {
    if (calib_files) calib_files->clear();
    if (!fs::is_directory(calibration_dir)) {
        return false;
    }
    bool loaded = false;
    for (const auto& entry : fs::directory_iterator(calibration_dir)) {
        if (!entry.is_regular_file()) continue;
        std::ifstream input(entry.path());
        if (!input.is_open()) continue;
        loaded = true;
        if (calib_files) calib_files->push_back(entry.path().filename().string());
    }
    if (calib_files) std::sort(calib_files->begin(), calib_files->end());
    return loaded;
}

std::vector<int64_t> intersectFrameIds(const std::map<int64_t, fs::path>& lidar_bins,
                                       const std::map<int64_t, PoseRecord>& poses) {
    std::vector<int64_t> common;
    auto lidar_it = lidar_bins.begin();
    auto pose_it = poses.begin();
    while (lidar_it != lidar_bins.end() && pose_it != poses.end()) {
        if (lidar_it->first == pose_it->first) {
            common.push_back(lidar_it->first);
            ++lidar_it;
            ++pose_it;
        } else if (lidar_it->first < pose_it->first) {
            ++lidar_it;
        } else {
            ++pose_it;
        }
    }
    return common;
}

size_t countMissingPose(const std::map<int64_t, fs::path>& lidar_bins,
                        const std::map<int64_t, PoseRecord>& poses) {
    size_t missing = 0;
    for (const auto& [frame_id, unused] : lidar_bins) {
        (void)unused;
        if (poses.find(frame_id) == poses.end()) ++missing;
    }
    return missing;
}

size_t countMissingLidar(const std::map<int64_t, PoseRecord>& poses,
                         const std::map<int64_t, fs::path>& lidar_bins) {
    size_t missing = 0;
    for (const auto& [frame_id, unused] : poses) {
        (void)unused;
        if (lidar_bins.find(frame_id) == lidar_bins.end()) ++missing;
    }
    return missing;
}

std::vector<PointXYZI> readKittiBin(const fs::path& path) {
    const auto byte_size = fs::file_size(path);
    constexpr uintmax_t kPointBytes = sizeof(float) * 4;
    if (byte_size % kPointBytes != 0) {
        throw std::runtime_error("KITTI360 lidar bin size is not a multiple of 16 bytes: " + path.string());
    }
    const uintmax_t point_count = byte_size / kPointBytes;
    if (point_count > static_cast<uintmax_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("KITTI360 lidar bin contains too many points: " + path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open lidar bin: " + path.string());
    }

    std::vector<PointXYZI> points(static_cast<size_t>(point_count));
    input.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(byte_size));
    if (!input) {
        throw std::runtime_error("failed to read lidar bin: " + path.string());
    }
    return points;
}

void writeAsciiPcd(const fs::path& path, const std::vector<PointXYZI>& points) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open output PCD: " + path.string());
    }

    output << "# .PCD v0.7 - Point Cloud Data file format\n";
    output << "VERSION 0.7\n";
    output << "FIELDS x y z intensity\n";
    output << "SIZE 4 4 4 4\n";
    output << "TYPE F F F F\n";
    output << "COUNT 1 1 1 1\n";
    output << "WIDTH " << points.size() << "\n";
    output << "HEIGHT 1\n";
    output << "VIEWPOINT 0 0 0 1 0 0 0\n";
    output << "POINTS " << points.size() << "\n";
    output << "DATA ascii\n";
    output << std::setprecision(9);
    for (const auto& point : points) {
        output << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.intensity << '\n';
    }
}

std::string zeroPaddedFrameName(int64_t frame_id) {
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame_id;
    return out.str();
}

void writeSummaryJson(const fs::path& path,
                      const Options& options,
                      size_t lidar_bin_count,
                      size_t pose_count,
                      size_t total_common_count,
                      const std::vector<int64_t>& selected_common,
                      size_t missing_pose_count,
                      size_t missing_lidar_count,
                      bool calib_loaded,
                      const std::vector<std::string>& calib_files,
                      size_t dumped_pcd_count) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open summary output: " + path.string());
    }

    output << "{\n";
    output << "  \"sequence\": \"" << jsonEscape(options.sequence) << "\",\n";
    output << "  \"kitti_root\": \"" << jsonEscape(options.kitti_root.string()) << "\",\n";
    output << "  \"lidar_bin_count\": " << lidar_bin_count << ",\n";
    output << "  \"pose_count\": " << pose_count << ",\n";
    output << "  \"common_frame_count\": " << selected_common.size() << ",\n";
    output << "  \"total_common_frame_count\": " << total_common_count << ",\n";
    if (selected_common.empty()) {
        output << "  \"first_common_frame\": null,\n";
        output << "  \"last_common_frame\": null,\n";
    } else {
        output << "  \"first_common_frame\": " << selected_common.front() << ",\n";
        output << "  \"last_common_frame\": " << selected_common.back() << ",\n";
    }
    output << "  \"missing_pose_count\": " << missing_pose_count << ",\n";
    output << "  \"missing_lidar_count\": " << missing_lidar_count << ",\n";
    output << "  \"calib_loaded\": " << (calib_loaded ? "true" : "false") << ",\n";
    output << "  \"calib_files\": [";
    for (size_t i = 0; i < calib_files.size(); ++i) {
        if (i > 0) output << ", ";
        output << "\"" << jsonEscape(calib_files[i]) << "\"";
    }
    output << "],\n";
    if (options.max_frames >= 0) {
        output << "  \"max_frames\": " << options.max_frames << ",\n";
    } else {
        output << "  \"max_frames\": null,\n";
    }
    output << "  \"dumped_pcd_count\": " << dumped_pcd_count << "\n";
    output << "}\n";
}

int run(const Options& options) {
    const fs::path lidar_dir = options.kitti_root / "data_3d_raw" / options.sequence / "velodyne_points" / "data";
    const fs::path poses_path = options.kitti_root / "data_poses" / options.sequence / "poses.txt";
    const fs::path calibration_dir = options.kitti_root / "calibration";

    const auto lidar_bins = listLidarBins(lidar_dir);
    const auto poses = readPoses(poses_path);
    std::vector<std::string> calib_files;
    const bool calib_loaded = loadCalibrationDirectory(calibration_dir, &calib_files);

    const std::vector<int64_t> all_common = intersectFrameIds(lidar_bins, poses);
    std::vector<int64_t> selected_common = all_common;
    if (options.max_frames >= 0 && static_cast<size_t>(options.max_frames) < selected_common.size()) {
        selected_common.resize(static_cast<size_t>(options.max_frames));
    }

    fs::create_directories(options.output_dir);

    size_t dumped_pcd_count = 0;
    int64_t requested_dump_count = options.dump_first_n;
    if (options.dump_sample_pcd) {
        requested_dump_count = std::max<int64_t>(requested_dump_count, 1);
    }
    if (requested_dump_count > 0) {
        const size_t dump_count = std::min<size_t>(selected_common.size(), static_cast<size_t>(requested_dump_count));
        for (size_t i = 0; i < dump_count; ++i) {
            const int64_t frame_id = selected_common[i];
            const auto lidar_it = lidar_bins.find(frame_id);
            if (lidar_it == lidar_bins.end()) {
                throw std::runtime_error("internal error: selected frame missing lidar bin");
            }
            const std::vector<PointXYZI> points = readKittiBin(lidar_it->second);
            const fs::path pcd_path =
                options.output_dir / ("frame_" + zeroPaddedFrameName(frame_id) + ".pcd");
            writeAsciiPcd(pcd_path, points);
            ++dumped_pcd_count;
        }
        if (options.dump_sample_pcd && !selected_common.empty()) {
            const fs::path first_frame_pcd =
                options.output_dir / ("frame_" + zeroPaddedFrameName(selected_common.front()) + ".pcd");
            const fs::path sample_pcd = options.output_dir / "sample.pcd";
            if (first_frame_pcd != sample_pcd) {
                fs::copy_file(first_frame_pcd, sample_pcd, fs::copy_options::overwrite_existing);
            }
        }
    }

    writeSummaryJson(options.output_dir / "summary.json",
                     options,
                     lidar_bins.size(),
                     poses.size(),
                     all_common.size(),
                     selected_common,
                     countMissingPose(lidar_bins, poses),
                     countMissingLidar(poses, lidar_bins),
                     calib_loaded,
                     calib_files,
                     dumped_pcd_count);

    std::cout << "KITTI360 reader summary written: "
              << (options.output_dir / "summary.json").string()
              << " common_frames=" << selected_common.size()
              << " total_common_frames=" << all_common.size()
              << " dumped_pcd=" << dumped_pcd_count << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        return run(options);
    } catch (const std::exception& e) {
        std::cerr << "n3mapping_kitti360_reader: " << e.what() << std::endl;
        std::cerr << "Run with --help for usage." << std::endl;
        return 1;
    }
}
