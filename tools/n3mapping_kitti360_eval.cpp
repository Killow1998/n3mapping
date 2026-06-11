#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/n3mapping_core.h"

namespace fs = std::filesystem;

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;

struct Options {
    fs::path kitti_root;
    std::string sequence;
    std::string mode;
    std::string calib_mode = "auto";
    fs::path output_dir;
    fs::path map_path;
    int64_t max_frames = -1;
    int stride = 1;
    int build_map_frames = 0;
    double dropout = 0.0;
    double noise_sigma = 0.0;
    double fake_yaw_deg = 0.0;
    double pose_translation_threshold_m = 1.0;
    double pose_yaw_threshold_deg = 10.0;
};

struct PoseRecord {
    int64_t frame_id = -1;
    Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
};

struct KittiFrame {
    int64_t frame_id = -1;
    fs::path lidar_bin;
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
};

struct Range3 {
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    size_t count = 0;
};

struct CalibrationDiagnostics {
    bool calib_loaded = false;
    std::string warning;
    std::string requested_mode = "auto";
    std::string used_mode = "identity";
    Eigen::Isometry3d T_cam_velo = Eigen::Isometry3d::Identity();
    int64_t first_frame_id = -1;
    Eigen::Isometry3d first_pose_cam_to_velo = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d first_pose_velo_to_cam = Eigen::Isometry3d::Identity();
    double first_pose_delta_translation_m = std::numeric_limits<double>::quiet_NaN();
    double first_pose_delta_yaw_deg = std::numeric_limits<double>::quiet_NaN();
    Range3 first_cloud_lidar_range;
    Range3 first_cloud_world_cam_to_velo_range;
    Range3 first_cloud_world_velo_to_cam_range;
};

struct AlignedFrames {
    std::vector<KittiFrame> frames;
    CalibrationDiagnostics calibration;
};

struct MappingStats {
    size_t frames_processed = 0;
    size_t successful_frames = 0;
    size_t accepted_keyframes = 0;
    size_t accepted_loops = 0;
};

struct RelocResult {
    int64_t frame_id = -1;
    bool success = false;
    bool lock = false;
    int64_t matched_keyframe_id = -1;
    double translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
};

class EvalProgressLog {
public:
    explicit EvalProgressLog(const fs::path& path)
        : start_(Clock::now())
    {
        out_.open(path);
        if (!out_.is_open()) {
            throw std::runtime_error("failed to open eval progress log: " + path.string());
        }
        out_ << "frame_index,frame_id,event,elapsed_ms,keyframe_id,accepted_keyframe,success,loop_edge_count,accepted_loops\n";
        out_.flush();
    }

    void write(size_t frame_index,
               int64_t frame_id,
               const std::string& event,
               int64_t keyframe_id = -1,
               bool accepted_keyframe = false,
               bool success = false,
               size_t loop_edge_count = 0,
               size_t accepted_loops = 0)
    {
        out_ << frame_index << ','
             << frame_id << ','
             << event << ','
             << elapsedMs() << ','
             << keyframe_id << ','
             << (accepted_keyframe ? "true" : "false") << ','
             << (success ? "true" : "false") << ','
             << loop_edge_count << ','
             << accepted_loops << '\n';
        out_.flush();
    }

private:
    using Clock = std::chrono::steady_clock;

    double elapsedMs() const
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

    Clock::time_point start_;
    std::ofstream out_;
};

std::string jsonEscape(const std::string& input)
{
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

void printUsage(std::ostream& os)
{
    os << "Usage: n3mapping_kitti360_eval --kitti_root <path> --sequence <name> "
       << "--mode mapping_loop|relocalization --output <dir> [options]\n\n"
       << "Options:\n"
       << "  --max_frames <N>                 Maximum selected common frames.\n"
       << "  --stride <N>                     Use every Nth common frame. Default: 1.\n"
       << "  --calib_mode auto|cam_to_velo|velo_to_cam\n"
       << "                                   KITTI360 calibration direction. Default: auto.\n"
       << "  --map <n3map.pbstream>           Existing map for relocalization mode.\n"
       << "  --build_map_frames <N>           Build a temporary map from first N selected frames when --map is absent.\n"
       << "  --dropout <R>                    Query point dropout ratio for relocalization [0,0.95]. Default: 0.\n"
       << "  --noise <M>                      Query XYZ Gaussian noise sigma in meters. Default: 0.\n"
       << "  --fake_yaw <DEG>                 Fake map->odom yaw for relocalization. Default: 0.\n"
       << "  --fake_yaw_deg <DEG>             Alias for --fake_yaw.\n"
       << "  --pose_translation_threshold <M> Pose success translation threshold. Default: 1.\n"
       << "  --pose_yaw_threshold_deg <DEG>   Pose success yaw threshold. Default: 10.\n"
       << "  --help                           Show this help.\n";
}

int64_t parseInt64(const std::string& value, const std::string& name)
{
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return static_cast<int64_t>(parsed);
}

double parseDouble(const std::string& value, const std::string& name)
{
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid " + name + ": " + value);
    }
    return parsed;
}

Options parseArgs(int argc, char** argv)
{
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
        } else if (arg == "--mode") {
            options.mode = requireValue(arg);
        } else if (arg == "--calib_mode") {
            options.calib_mode = requireValue(arg);
        } else if (arg == "--output") {
            options.output_dir = requireValue(arg);
        } else if (arg == "--max_frames") {
            options.max_frames = parseInt64(requireValue(arg), arg);
        } else if (arg == "--stride") {
            options.stride = std::max<int>(1, static_cast<int>(parseInt64(requireValue(arg), arg)));
        } else if (arg == "--map") {
            options.map_path = requireValue(arg);
        } else if (arg == "--build_map_frames") {
            options.build_map_frames = std::max<int>(0, static_cast<int>(parseInt64(requireValue(arg), arg)));
        } else if (arg == "--dropout") {
            options.dropout = std::clamp(parseDouble(requireValue(arg), arg), 0.0, 0.95);
        } else if (arg == "--noise") {
            options.noise_sigma = std::max(0.0, parseDouble(requireValue(arg), arg));
        } else if (arg == "--fake_yaw" || arg == "--fake_yaw_deg") {
            options.fake_yaw_deg = parseDouble(requireValue(arg), arg);
        } else if (arg == "--pose_translation_threshold") {
            options.pose_translation_threshold_m = std::max(0.0, parseDouble(requireValue(arg), arg));
        } else if (arg == "--pose_yaw_threshold_deg") {
            options.pose_yaw_threshold_deg = std::max(0.0, parseDouble(requireValue(arg), arg));
        } else if (arg == "--help" || arg == "-h") {
            printUsage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.kitti_root.empty()) throw std::runtime_error("--kitti_root is required");
    if (options.sequence.empty()) throw std::runtime_error("--sequence is required");
    if (options.output_dir.empty()) throw std::runtime_error("--output is required");
    if (options.mode != "mapping_loop" && options.mode != "relocalization") {
        throw std::runtime_error("--mode must be mapping_loop or relocalization");
    }
    if (options.calib_mode != "auto" &&
        options.calib_mode != "cam_to_velo" &&
        options.calib_mode != "velo_to_cam") {
        throw std::runtime_error("--calib_mode must be auto, cam_to_velo, or velo_to_cam");
    }
    if (options.max_frames < -1) throw std::runtime_error("--max_frames must be non-negative");
    return options;
}

bool parseFrameIdFromBin(const fs::path& path, int64_t* frame_id)
{
    if (!frame_id || path.extension() != ".bin") return false;
    const std::string stem = path.stem().string();
    if (stem.empty()) return false;
    for (const char ch : stem) {
        if (ch < '0' || ch > '9') return false;
    }
    *frame_id = parseInt64(stem, "frame id");
    return *frame_id >= 0;
}

std::map<int64_t, fs::path> listLidarBins(const fs::path& lidar_dir)
{
    if (!fs::is_directory(lidar_dir)) {
        throw std::runtime_error("lidar directory does not exist: " + lidar_dir.string());
    }

    std::map<int64_t, fs::path> bins;
    for (const auto& entry : fs::directory_iterator(lidar_dir)) {
        if (!entry.is_regular_file()) continue;
        int64_t frame_id = -1;
        if (!parseFrameIdFromBin(entry.path(), &frame_id)) continue;
        if (!bins.emplace(frame_id, entry.path()).second) {
            throw std::runtime_error("duplicate lidar frame id: " + std::to_string(frame_id));
        }
    }
    return bins;
}

Eigen::Isometry3d poseFromTwelve(const std::array<double, 12>& values)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.matrix()(0, 0) = values[0];
    pose.matrix()(0, 1) = values[1];
    pose.matrix()(0, 2) = values[2];
    pose.matrix()(0, 3) = values[3];
    pose.matrix()(1, 0) = values[4];
    pose.matrix()(1, 1) = values[5];
    pose.matrix()(1, 2) = values[6];
    pose.matrix()(1, 3) = values[7];
    pose.matrix()(2, 0) = values[8];
    pose.matrix()(2, 1) = values[9];
    pose.matrix()(2, 2) = values[10];
    pose.matrix()(2, 3) = values[11];
    return pose;
}

std::map<int64_t, PoseRecord> readPoses(const fs::path& poses_path)
{
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
        int64_t frame_id = -1;
        if (!(iss >> frame_id) || frame_id < 0) {
            throw std::runtime_error("invalid pose frame id at line " + std::to_string(line_number));
        }
        std::array<double, 12> values{};
        for (double& value : values) {
            if (!(iss >> value) || !std::isfinite(value)) {
                throw std::runtime_error("failed to parse finite 3x4 pose at line " + std::to_string(line_number));
            }
        }
        PoseRecord record;
        record.frame_id = frame_id;
        record.T_world_cam = poseFromTwelve(values);
        if (!poses.emplace(frame_id, record).second) {
            throw std::runtime_error("duplicate pose frame id: " + std::to_string(frame_id));
        }
    }
    return poses;
}

std::vector<double> extractDoubles(const std::string& text)
{
    std::vector<double> values;
    std::string token;
    std::istringstream stream(text);
    while (stream >> token) {
        if (!token.empty() && token.back() == ':') continue;
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(token.c_str(), &end);
        if (errno == 0 && end != token.c_str() && *end == '\0' && std::isfinite(value)) {
            values.push_back(value);
        }
    }
    return values;
}

double yawRad(const Eigen::Isometry3d& pose);
double angleDiffRad(double a, double b);

CalibrationDiagnostics readCalibration(const fs::path& calibration_dir, const Options& options)
{
    CalibrationDiagnostics diagnostics;
    diagnostics.requested_mode = options.calib_mode;
    diagnostics.used_mode = options.calib_mode == "velo_to_cam" ? "velo_to_cam" : "cam_to_velo";

    const fs::path path = calibration_dir / "calib_cam_to_velo.txt";
    if (!fs::is_regular_file(path)) {
        diagnostics.warning = "missing calib_cam_to_velo.txt; using identity transform";
        diagnostics.used_mode = "identity";
        std::cerr << "Warning: " << diagnostics.warning << ".\n";
        return diagnostics;
    }
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto values = extractDoubles(buffer.str());
    if (values.size() < 12) {
        diagnostics.warning = "failed to parse calib_cam_to_velo.txt; using identity transform";
        diagnostics.used_mode = "identity";
        std::cerr << "Warning: " << diagnostics.warning << ".\n";
        return diagnostics;
    }
    std::array<double, 12> first_twelve{};
    std::copy_n(values.begin(), 12, first_twelve.begin());
    diagnostics.calib_loaded = true;
    diagnostics.T_cam_velo = poseFromTwelve(first_twelve);
    return diagnostics;
}

Eigen::Isometry3d applyCalibrationMode(const Eigen::Isometry3d& T_world_cam,
                                       const CalibrationDiagnostics& calibration)
{
    if (!calibration.calib_loaded) {
        return T_world_cam;
    }
    if (calibration.used_mode == "velo_to_cam") {
        return T_world_cam * calibration.T_cam_velo.inverse();
    }
    return T_world_cam * calibration.T_cam_velo;
}

AlignedFrames loadAlignedFrames(const Options& options)
{
    const fs::path lidar_dir = options.kitti_root / "data_3d_raw" / options.sequence / "velodyne_points" / "data";
    const fs::path poses_path = options.kitti_root / "data_poses" / options.sequence / "poses.txt";
    const fs::path calibration_dir = options.kitti_root / "calibration";
    const auto lidar_bins = listLidarBins(lidar_dir);
    const auto poses = readPoses(poses_path);
    AlignedFrames aligned;
    aligned.calibration = readCalibration(calibration_dir, options);

    int stride_count = 0;
    for (const auto& [frame_id, bin_path] : lidar_bins) {
        auto pose_it = poses.find(frame_id);
        if (pose_it == poses.end()) continue;
        if ((stride_count++ % options.stride) != 0) continue;
        KittiFrame frame;
        frame.frame_id = frame_id;
        frame.lidar_bin = bin_path;
        frame.T_world_lidar = applyCalibrationMode(pose_it->second.T_world_cam, aligned.calibration);
        aligned.frames.push_back(frame);
        if (options.max_frames >= 0 && static_cast<int64_t>(aligned.frames.size()) >= options.max_frames) {
            break;
        }
    }
    if (aligned.frames.empty()) {
        throw std::runtime_error("no common KITTI360 lidar/pose frame ids selected");
    }
    const auto pose_it = poses.find(aligned.frames.front().frame_id);
    if (pose_it != poses.end()) {
        aligned.calibration.first_frame_id = aligned.frames.front().frame_id;
        aligned.calibration.first_pose_cam_to_velo = pose_it->second.T_world_cam * aligned.calibration.T_cam_velo;
        aligned.calibration.first_pose_velo_to_cam = pose_it->second.T_world_cam * aligned.calibration.T_cam_velo.inverse();
        aligned.calibration.first_pose_delta_translation_m =
            (aligned.calibration.first_pose_cam_to_velo.translation() -
             aligned.calibration.first_pose_velo_to_cam.translation()).norm();
        aligned.calibration.first_pose_delta_yaw_deg =
            std::abs(angleDiffRad(yawRad(aligned.calibration.first_pose_cam_to_velo),
                                  yawRad(aligned.calibration.first_pose_velo_to_cam))) *
            180.0 / M_PI;
    }
    return aligned;
}

Cloud::Ptr readKittiBinCloud(const fs::path& path)
{
    const auto byte_size = fs::file_size(path);
    constexpr uintmax_t kPointBytes = sizeof(float) * 4;
    if (byte_size % kPointBytes != 0) {
        throw std::runtime_error("KITTI360 lidar bin size is not a multiple of 16 bytes: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open lidar bin: " + path.string());
    }
    auto cloud = pcl::make_shared<Cloud>();
    const size_t point_count = static_cast<size_t>(byte_size / kPointBytes);
    cloud->reserve(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        float raw[4]{};
        input.read(reinterpret_cast<char*>(raw), sizeof(raw));
        if (!input) {
            throw std::runtime_error("failed to read lidar bin: " + path.string());
        }
        if (!std::isfinite(raw[0]) || !std::isfinite(raw[1]) ||
            !std::isfinite(raw[2]) || !std::isfinite(raw[3])) {
            continue;
        }
        pcl::PointXYZI point;
        point.x = raw[0];
        point.y = raw[1];
        point.z = raw[2];
        point.intensity = raw[3];
        cloud->push_back(point);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

void updateRange(Range3* range, double x, double y, double z)
{
    if (!range || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
    range->min_x = std::min(range->min_x, x);
    range->min_y = std::min(range->min_y, y);
    range->min_z = std::min(range->min_z, z);
    range->max_x = std::max(range->max_x, x);
    range->max_y = std::max(range->max_y, y);
    range->max_z = std::max(range->max_z, z);
    ++range->count;
}

void populateCalibrationCloudSummary(CalibrationDiagnostics* calibration, const KittiFrame& first_frame)
{
    if (!calibration) return;
    const auto cloud = readKittiBinCloud(first_frame.lidar_bin);
    for (const auto& point : cloud->points) {
        updateRange(&calibration->first_cloud_lidar_range, point.x, point.y, point.z);
        const Eigen::Vector3d p_lidar(point.x, point.y, point.z);
        const Eigen::Vector3d p_direct = calibration->first_pose_cam_to_velo * p_lidar;
        const Eigen::Vector3d p_inverse = calibration->first_pose_velo_to_cam * p_lidar;
        updateRange(&calibration->first_cloud_world_cam_to_velo_range,
                    p_direct.x(),
                    p_direct.y(),
                    p_direct.z());
        updateRange(&calibration->first_cloud_world_velo_to_cam_range,
                    p_inverse.x(),
                    p_inverse.y(),
                    p_inverse.z());
    }
}

Cloud::Ptr perturbCloud(const Cloud::Ptr& input_cloud, double dropout, double noise_sigma, uint32_t seed)
{
    auto cloud = pcl::make_shared<Cloud>();
    if (!input_cloud) return cloud;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    cloud->reserve(input_cloud->size());
    for (const auto& point : input_cloud->points) {
        if (keep_dist(rng) < dropout) continue;
        pcl::PointXYZI out = point;
        if (noise_sigma > 0.0) {
            out.x += static_cast<float>(noise(rng));
            out.y += static_cast<float>(noise(rng));
            out.z += static_cast<float>(noise(rng));
        }
        cloud->push_back(out);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

core::LioFrame makeFrame(const KittiFrame& frame, const Eigen::Isometry3d& pose, const Cloud::Ptr& cloud)
{
    core::LioFrame lio;
    lio.stamp.nsec = frame.frame_id * 100000000LL;
    lio.T_world_lidar = pose;
    lio.undistorted_cloud = cloud;
    lio.pose_valid = true;
    lio.covariance_valid = false;
    return lio;
}

double yawRad(const Eigen::Isometry3d& pose)
{
    return std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
}

double angleDiffRad(double a, double b)
{
    double d = a - b;
    while (d > M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

double yawErrorDeg(const Eigen::Isometry3d& estimate, const Eigen::Isometry3d& gt)
{
    return std::abs(angleDiffRad(yawRad(estimate), yawRad(gt))) * 180.0 / M_PI;
}

double percentile(std::vector<double> values, double q)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) {
        return !std::isfinite(v);
    }), values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double idx = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double t = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

void writeTrajectoryLine(std::ofstream& out, int64_t frame_id, const Eigen::Isometry3d& pose)
{
    const Eigen::Quaterniond q(pose.rotation());
    out << frame_id << ' ' << std::fixed << std::setprecision(9)
        << pose.translation().x() << ' '
        << pose.translation().y() << ' '
        << pose.translation().z() << ' '
        << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
}

void writeJsonDoubleOrNull(std::ostream& out, double value)
{
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

void writePoseSummaryJson(std::ostream& out, const Eigen::Isometry3d& pose)
{
    out << "{\"x\":" << pose.translation().x()
        << ",\"y\":" << pose.translation().y()
        << ",\"z\":" << pose.translation().z()
        << ",\"yaw_deg\":" << yawRad(pose) * 180.0 / M_PI << "}";
}

void writeRangeJson(std::ostream& out, const Range3& range)
{
    out << "{\"count\":" << range.count
        << ",\"x_min\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.min_x);
    out << ",\"x_max\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.max_x);
    out << ",\"y_min\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.min_y);
    out << ",\"y_max\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.max_y);
    out << ",\"z_min\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.min_z);
    out << ",\"z_max\":";
    writeJsonDoubleOrNull(out, range.count == 0 ? std::numeric_limits<double>::quiet_NaN() : range.max_z);
    out << "}";
}

void writeCalibrationJson(std::ostream& out, const CalibrationDiagnostics& calibration)
{
    out << "  \"calibration\": {\n"
        << "    \"calib_loaded\": " << (calibration.calib_loaded ? "true" : "false") << ",\n"
        << "    \"warning\": \"" << jsonEscape(calibration.warning) << "\",\n"
        << "    \"calib_mode_requested\": \"" << jsonEscape(calibration.requested_mode) << "\",\n"
        << "    \"calib_mode_used\": \"" << jsonEscape(calibration.used_mode) << "\",\n"
        << "    \"first_frame_id\": " << calibration.first_frame_id << ",\n"
        << "    \"first_pose_cam_to_velo\": ";
    writePoseSummaryJson(out, calibration.first_pose_cam_to_velo);
    out << ",\n    \"first_pose_velo_to_cam\": ";
    writePoseSummaryJson(out, calibration.first_pose_velo_to_cam);
    out << ",\n    \"first_pose_delta_translation_m\": ";
    writeJsonDoubleOrNull(out, calibration.first_pose_delta_translation_m);
    out << ",\n    \"first_pose_delta_yaw_deg\": ";
    writeJsonDoubleOrNull(out, calibration.first_pose_delta_yaw_deg);
    out << ",\n    \"first_cloud_lidar_range\": ";
    writeRangeJson(out, calibration.first_cloud_lidar_range);
    out << ",\n    \"first_cloud_world_cam_to_velo_range\": ";
    writeRangeJson(out, calibration.first_cloud_world_cam_to_velo_range);
    out << ",\n    \"first_cloud_world_velo_to_cam_range\": ";
    writeRangeJson(out, calibration.first_cloud_world_velo_to_cam_range);
    out << "\n  }";
}

Config makeEvalConfig(const Options& options)
{
    Config config;
    config.mode = options.mode == "mapping_loop" ? "mapping" : "localization";
    config.map_save_path = options.output_dir.string();
    config.map_path = options.map_path.string();
    config.loop_debug_enable = options.mode == "mapping_loop";
    config.loop_debug_path = (options.output_dir / "loop_debug.jsonl").string();
    config.reloc_debug_enable = options.mode == "relocalization";
    config.reloc_debug_path = (options.output_dir / "relocalization_debug.jsonl").string();
    return config;
}

void touchFile(const fs::path& path)
{
    std::ofstream out(path, std::ios::app);
}

void writeAcceptedLoopsHeader(std::ofstream& out)
{
    out << "query_id,match_id,fitness_score,inlier_ratio,verified\n";
}

void writeKeyframesGtHeader(std::ofstream& out)
{
    out << "keyframe_id,frame_id,x,y,z,qx,qy,qz,qw\n";
}

void writeKeyframeGt(std::ofstream& out, int64_t keyframe_id, const KittiFrame& frame)
{
    const Eigen::Quaterniond q(frame.T_world_lidar.rotation());
    out << keyframe_id << ','
        << frame.frame_id << ','
        << std::setprecision(17)
        << frame.T_world_lidar.translation().x() << ','
        << frame.T_world_lidar.translation().y() << ','
        << frame.T_world_lidar.translation().z() << ','
        << q.x() << ','
        << q.y() << ','
        << q.z() << ','
        << q.w() << '\n';
}

void writeAcceptedLoop(std::ofstream& out, const VerifiedLoop& loop)
{
    out << loop.query_id << ','
        << loop.match_id << ','
        << loop.fitness_score << ','
        << loop.inlier_ratio << ','
        << (loop.verified ? "true" : "false") << '\n';
}

int runMappingLoop(const Options& options, const AlignedFrames& aligned)
{
    fs::create_directories(options.output_dir);
    touchFile(options.output_dir / "loop_debug.jsonl");
    EvalProgressLog progress(options.output_dir / "eval_progress.csv");
    std::ofstream trajectory_est(options.output_dir / "trajectory_est.txt");
    std::ofstream trajectory_gt(options.output_dir / "trajectory_gt.txt");
    std::ofstream keyframes_gt(options.output_dir / "keyframes_gt.csv");
    std::ofstream accepted_loops(options.output_dir / "accepted_loops.csv");
    if (!trajectory_est.is_open() || !trajectory_gt.is_open() ||
        !keyframes_gt.is_open() || !accepted_loops.is_open()) {
        throw std::runtime_error("failed to open mapping_loop output files");
    }
    writeKeyframesGtHeader(keyframes_gt);
    writeAcceptedLoopsHeader(accepted_loops);

    Config config = makeEvalConfig(options);
    N3MappingCore core(config);
    MappingStats stats;
    const auto& frames = aligned.frames;
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        const auto& frame = frames[frame_index];
        progress.write(frame_index, frame.frame_id, "read_cloud_start");
        auto cloud = readKittiBinCloud(frame.lidar_bin);
        progress.write(frame_index, frame.frame_id, "read_cloud_done");
        progress.write(frame_index, frame.frame_id, "process_mapping_start");
        auto output = core.processMappingFrame(makeFrame(frame, frame.T_world_lidar, cloud));
        progress.write(frame_index,
                       frame.frame_id,
                       "process_mapping_done",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success);
        ++stats.frames_processed;
        if (output.success) ++stats.successful_frames;
        if (output.accepted_keyframe) ++stats.accepted_keyframes;
        progress.write(frame_index,
                       frame.frame_id,
                       "write_outputs_start",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success);
        writeTrajectoryLine(trajectory_gt, frame.frame_id, frame.T_world_lidar);
        writeTrajectoryLine(trajectory_est, frame.frame_id, output.T_world_lidar);
        if (output.accepted_keyframe && output.keyframe_id >= 0) {
            writeKeyframeGt(keyframes_gt, output.keyframe_id, frame);
        }
        trajectory_gt.flush();
        trajectory_est.flush();
        keyframes_gt.flush();
        progress.write(frame_index,
                       frame.frame_id,
                       "write_outputs_done",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success);

        progress.write(frame_index,
                       frame.frame_id,
                       "process_loops_start",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success);
        const auto loop_result = core.processPendingLoopClosures();
        stats.accepted_loops += loop_result.accepted_loops.size();
        for (const auto& loop : loop_result.accepted_loops) {
            writeAcceptedLoop(accepted_loops, loop);
        }
        accepted_loops.flush();
        progress.write(frame_index,
                       frame.frame_id,
                       "process_loops_done",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success,
                       static_cast<size_t>(loop_result.edge_count),
                       loop_result.accepted_loops.size());
    }
    progress.write(frames.size(), -1, "final_process_loops_start");
    const auto final_loop_result = core.processPendingLoopClosures();
    stats.accepted_loops += final_loop_result.accepted_loops.size();
    for (const auto& loop : final_loop_result.accepted_loops) {
        writeAcceptedLoop(accepted_loops, loop);
    }
    accepted_loops.flush();
    progress.write(frames.size(),
                   -1,
                   "final_process_loops_done",
                   -1,
                   false,
                   true,
                   static_cast<size_t>(final_loop_result.edge_count),
                   final_loop_result.accepted_loops.size());

    const auto dense = core.getDenseOptimizedTrajectory();
    std::ofstream metrics(options.output_dir / "metrics.json");
    metrics << "{\n"
            << "  \"mode\": \"mapping_loop\",\n"
            << "  \"sequence\": \"" << jsonEscape(options.sequence) << "\",\n"
            << "  \"frames_processed\": " << stats.frames_processed << ",\n"
            << "  \"successful_frames\": " << stats.successful_frames << ",\n"
            << "  \"accepted_keyframes\": " << stats.accepted_keyframes << ",\n"
            << "  \"accepted_loop_count\": " << stats.accepted_loops << ",\n"
            << "  \"dense_trajectory_count\": " << dense.size() << ",\n"
            << "  \"stride\": " << options.stride << ",\n";
    writeCalibrationJson(metrics, aligned.calibration);
    metrics << "\n"
            << "}\n";
    std::cout << "mapping_loop frames=" << stats.frames_processed
              << " keyframes=" << stats.accepted_keyframes
              << " loops=" << stats.accepted_loops
              << " output=" << options.output_dir << "\n";
    return 0;
}

Eigen::Isometry3d fakeMapToOdom(double fake_yaw_deg)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = Eigen::AngleAxisd(fake_yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return T;
}

bool buildTemporaryMap(const Options& options,
                       const std::vector<KittiFrame>& frames,
                       int build_count,
                       fs::path* map_path)
{
    if (!map_path) return false;
    Config config = makeEvalConfig(options);
    config.mode = "mapping";
    config.loop_debug_enable = false;
    config.reloc_debug_enable = false;
    config.map_save_path = (options.output_dir / "temporary_map").string();
    N3MappingCore mapper(config);

    for (int i = 0; i < build_count; ++i) {
        const auto& frame = frames[static_cast<size_t>(i)];
        auto cloud = readKittiBinCloud(frame.lidar_bin);
        mapper.processMappingFrame(makeFrame(frame, frame.T_world_lidar, cloud));
        mapper.processPendingLoopClosures();
    }
    fs::create_directories(options.output_dir / "temporary_map");
    *map_path = options.output_dir / "temporary_map" / "n3map.pbstream";
    return mapper.saveMap(map_path->string());
}

bool poseSuccess(const RelocResult& result, const Options& options)
{
    return result.success &&
        std::isfinite(result.translation_error_m) &&
        std::isfinite(result.yaw_error_deg) &&
        result.translation_error_m <= options.pose_translation_threshold_m &&
        result.yaw_error_deg <= options.pose_yaw_threshold_deg;
}

int runRelocalization(const Options& options, const AlignedFrames& aligned)
{
    fs::create_directories(options.output_dir);
    touchFile(options.output_dir / "relocalization_debug.jsonl");
    const auto& frames = aligned.frames;

    fs::path map_path = options.map_path;
    int query_start = 0;
    if (map_path.empty()) {
        int build_count = options.build_map_frames;
        if (build_count <= 0) {
            build_count = std::max<int>(1, static_cast<int>(frames.size() / 2));
        }
        build_count = std::min<int>(build_count, static_cast<int>(frames.size()));
        if (build_count >= static_cast<int>(frames.size())) {
            throw std::runtime_error("relocalization needs at least one query frame after the temporary map frames");
        }
        if (!buildTemporaryMap(options, frames, build_count, &map_path)) {
            throw std::runtime_error("failed to build temporary KITTI360 relocalization map");
        }
        query_start = build_count;
    }

    Config config = makeEvalConfig(options);
    config.map_path = map_path.string();
    config.mode = "localization";
    N3MappingCore localizer(config);
    if (!localizer.loadMap(map_path.string())) {
        throw std::runtime_error("failed to load relocalization map: " + map_path.string());
    }

    const Eigen::Isometry3d T_map_odom = fakeMapToOdom(options.fake_yaw_deg);
    std::vector<RelocResult> results;
    results.reserve(frames.size() - static_cast<size_t>(query_start));
    for (size_t i = static_cast<size_t>(query_start); i < frames.size(); ++i) {
        const auto& frame = frames[i];
        auto cloud = perturbCloud(readKittiBinCloud(frame.lidar_bin),
                                  options.dropout,
                                  options.noise_sigma,
                                  static_cast<uint32_t>(frame.frame_id));
        const Eigen::Isometry3d T_odom_lidar = T_map_odom.inverse() * frame.T_world_lidar;
        const auto output = localizer.processLocalizationFrame(makeFrame(frame, T_odom_lidar, cloud));
        RelocResult result;
        result.frame_id = frame.frame_id;
        result.success = output.success;
        result.lock = output.relocalization_locked;
        result.matched_keyframe_id = output.matched_keyframe_id;
        if (output.success) {
            result.translation_error_m =
                (output.T_world_lidar.translation() - frame.T_world_lidar.translation()).norm();
            result.yaw_error_deg = yawErrorDeg(output.T_world_lidar, frame.T_world_lidar);
        }
        results.push_back(result);
    }

    std::vector<double> translation_errors;
    std::vector<double> yaw_errors;
    size_t locks = 0;
    size_t pose_successes = 0;
    for (const auto& result : results) {
        if (result.lock) ++locks;
        if (poseSuccess(result, options)) ++pose_successes;
        if (result.success) {
            translation_errors.push_back(result.translation_error_m);
            yaw_errors.push_back(result.yaw_error_deg);
        }
    }

    std::ofstream metrics(options.output_dir / "metrics.json");
    metrics << "{\n"
            << "  \"mode\": \"relocalization\",\n"
            << "  \"sequence\": \"" << jsonEscape(options.sequence) << "\",\n"
            << "  \"map_path\": \"" << jsonEscape(map_path.string()) << "\",\n"
            << "  \"query_count\": " << results.size() << ",\n"
            << "  \"lock_success_rate\": " << (results.empty() ? 0.0 : static_cast<double>(locks) / results.size()) << ",\n"
            << "  \"pose_success_rate\": " << (results.empty() ? 0.0 : static_cast<double>(pose_successes) / results.size()) << ",\n"
            << "  \"median_translation_error_m\": ";
    writeJsonDoubleOrNull(metrics, percentile(translation_errors, 0.5));
    metrics << ",\n  \"p95_translation_error_m\": ";
    writeJsonDoubleOrNull(metrics, percentile(translation_errors, 0.95));
    metrics << ",\n  \"median_yaw_error_deg\": ";
    writeJsonDoubleOrNull(metrics, percentile(yaw_errors, 0.5));
    metrics << ",\n  \"p95_yaw_error_deg\": ";
    writeJsonDoubleOrNull(metrics, percentile(yaw_errors, 0.95));
    metrics << ",\n"
            << "  \"dropout\": " << options.dropout << ",\n"
            << "  \"noise_sigma\": " << options.noise_sigma << ",\n"
            << "  \"fake_yaw_deg\": " << options.fake_yaw_deg << ",\n";
    writeCalibrationJson(metrics, aligned.calibration);
    metrics << "\n"
            << "}\n";

    std::ofstream per_query(options.output_dir / "relocalization_queries.csv");
    per_query << "frame_id,success,lock,matched_keyframe_id,translation_error_m,yaw_error_deg\n";
    for (const auto& result : results) {
        per_query << result.frame_id << ','
                  << (result.success ? "true" : "false") << ','
                  << (result.lock ? "true" : "false") << ','
                  << result.matched_keyframe_id << ','
                  << result.translation_error_m << ','
                  << result.yaw_error_deg << '\n';
    }

    std::cout << "relocalization queries=" << results.size()
              << " locks=" << locks
              << " pose_successes=" << pose_successes
              << " output=" << options.output_dir << "\n";
    return 0;
}

}  // namespace
}  // namespace n3mapping

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    try {
        const auto options = n3mapping::parseArgs(argc, argv);
        auto aligned = n3mapping::loadAlignedFrames(options);
        n3mapping::populateCalibrationCloudSummary(&aligned.calibration, aligned.frames.front());
        if (options.mode == "mapping_loop") {
            return n3mapping::runMappingLoop(options, aligned);
        }
        return n3mapping::runRelocalization(options, aligned);
    } catch (const std::exception& e) {
        std::cerr << "n3mapping_kitti360_eval: " << e.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }
}
