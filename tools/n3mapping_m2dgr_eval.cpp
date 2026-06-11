#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cctype>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/n3mapping_core.h"

namespace fs = std::filesystem;

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;

struct Options {
    fs::path m2dgr_root;
    std::string sequence;
    fs::path lidar_dir;
    fs::path gt_path;
    std::string mode;
    fs::path output_dir;
    fs::path map_path;
    int64_t max_frames = -1;
    int stride = 1;
    double max_time_diff = 0.05;
    int build_map_frames = 0;
    double dropout = 0.0;
    double noise_sigma = 0.0;
    double fake_yaw_deg = 0.0;
    double pose_translation_threshold_m = 1.0;
    double pose_yaw_threshold_deg = 10.0;
};

struct LidarRecord {
    double stamp = 0.0;
    fs::path path;
};

struct PoseRecord {
    double stamp = 0.0;
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
};

struct M2DGRFrame {
    size_t frame_index = 0;
    double lidar_stamp = 0.0;
    double gt_stamp = 0.0;
    double time_diff = 0.0;
    fs::path cloud_path;
    Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
};

struct MappingStats {
    size_t frames_processed = 0;
    size_t successful_frames = 0;
    size_t accepted_keyframes = 0;
    size_t accepted_loops = 0;
};

struct RelocResult {
    size_t frame_index = 0;
    double timestamp = 0.0;
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
        out_ << "frame_index,timestamp,event,elapsed_ms,keyframe_id,accepted_keyframe,success,loop_edge_count,accepted_loops\n";
        out_.flush();
    }

    void write(size_t frame_index,
               double timestamp,
               const std::string& event,
               int64_t keyframe_id = -1,
               bool accepted_keyframe = false,
               bool success = false,
               size_t loop_edge_count = 0,
               size_t accepted_loops = 0)
    {
        out_ << frame_index << ','
             << std::fixed << std::setprecision(9) << timestamp << ','
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
    os << "Usage: n3mapping_m2dgr_eval --m2dgr_root <path> --sequence <name> "
       << "--mode mapping_loop|relocalization --output <dir> [options]\n\n"
       << "This tool expects M2DGR lidar clouds extracted from rosbag into .pcd or .bin files,\n"
       << "plus a TUM-format ground-truth trajectory: timestamp x y z qx qy qz qw.\n\n"
       << "Options:\n"
       << "  --lidar_dir <dir>                Explicit extracted lidar cloud directory.\n"
       << "  --gt <path>                      Explicit TUM ground-truth trajectory file.\n"
       << "  --max_frames <N>                 Maximum aligned frames.\n"
       << "  --stride <N>                     Use every Nth aligned frame. Default: 1.\n"
       << "  --max_time_diff <S>              Max lidar/GT timestamp difference. Default: 0.05.\n"
       << "  --map <n3map.pbstream>           Existing map for relocalization mode.\n"
       << "  --build_map_frames <N>           Build temporary map from first N frames when --map is absent.\n"
       << "  --dropout <R>                    Query point dropout ratio for relocalization [0,0.95].\n"
       << "  --noise <M>                      Query XYZ Gaussian noise sigma in meters.\n"
       << "  --fake_yaw <DEG>                 Fake map->odom yaw for relocalization.\n"
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

        if (arg == "--m2dgr_root") {
            options.m2dgr_root = requireValue(arg);
        } else if (arg == "--sequence") {
            options.sequence = requireValue(arg);
        } else if (arg == "--lidar_dir") {
            options.lidar_dir = requireValue(arg);
        } else if (arg == "--gt") {
            options.gt_path = requireValue(arg);
        } else if (arg == "--mode") {
            options.mode = requireValue(arg);
        } else if (arg == "--output") {
            options.output_dir = requireValue(arg);
        } else if (arg == "--max_frames") {
            options.max_frames = parseInt64(requireValue(arg), arg);
        } else if (arg == "--stride") {
            options.stride = static_cast<int>(parseInt64(requireValue(arg), arg));
        } else if (arg == "--max_time_diff") {
            options.max_time_diff = parseDouble(requireValue(arg), arg);
        } else if (arg == "--map") {
            options.map_path = requireValue(arg);
        } else if (arg == "--build_map_frames") {
            options.build_map_frames = static_cast<int>(parseInt64(requireValue(arg), arg));
        } else if (arg == "--dropout") {
            options.dropout = parseDouble(requireValue(arg), arg);
        } else if (arg == "--noise") {
            options.noise_sigma = parseDouble(requireValue(arg), arg);
        } else if (arg == "--fake_yaw" || arg == "--fake_yaw_deg") {
            options.fake_yaw_deg = parseDouble(requireValue(arg), arg);
        } else if (arg == "--pose_translation_threshold") {
            options.pose_translation_threshold_m = parseDouble(requireValue(arg), arg);
        } else if (arg == "--pose_yaw_threshold_deg") {
            options.pose_yaw_threshold_deg = parseDouble(requireValue(arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.m2dgr_root.empty()) {
        throw std::runtime_error("--m2dgr_root is required");
    }
    if (options.sequence.empty()) {
        throw std::runtime_error("--sequence is required");
    }
    if (options.mode != "mapping_loop" && options.mode != "relocalization") {
        throw std::runtime_error("--mode must be mapping_loop or relocalization");
    }
    if (options.output_dir.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (options.stride <= 0) {
        throw std::runtime_error("--stride must be positive");
    }
    if (options.max_frames < -1) {
        throw std::runtime_error("--max_frames must be non-negative or -1");
    }
    if (options.max_time_diff < 0.0 || !std::isfinite(options.max_time_diff)) {
        throw std::runtime_error("--max_time_diff must be non-negative");
    }
    if (options.build_map_frames < 0) {
        throw std::runtime_error("--build_map_frames must be non-negative");
    }
    if (options.dropout < 0.0 || options.dropout >= 0.95) {
        throw std::runtime_error("--dropout must be in [0, 0.95)");
    }
    if (options.noise_sigma < 0.0) {
        throw std::runtime_error("--noise must be non-negative");
    }
    return options;
}

bool isCloudFile(const fs::path& path)
{
    return path.extension() == ".pcd" || path.extension() == ".bin";
}

bool parseStampFromStem(const fs::path& path, double* stamp)
{
    if (!stamp) return false;
    std::string stem = path.stem().string();
    if (stem.empty()) return false;
    size_t begin = 0;
    while (begin < stem.size() && !(std::isdigit(stem[begin]) || stem[begin] == '-' || stem[begin] == '+')) {
        ++begin;
    }
    if (begin >= stem.size()) return false;
    size_t end = begin;
    while (end < stem.size() &&
           (std::isdigit(stem[end]) || stem[end] == '.' || stem[end] == '-' || stem[end] == '+' ||
            stem[end] == 'e' || stem[end] == 'E')) {
        ++end;
    }
    try {
        size_t parsed_chars = 0;
        const double parsed = std::stod(stem.substr(begin, end - begin), &parsed_chars);
        if (parsed_chars == 0 || !std::isfinite(parsed)) return false;
        *stamp = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

fs::path firstExistingDirectory(const std::vector<fs::path>& candidates)
{
    for (const auto& candidate : candidates) {
        if (fs::is_directory(candidate)) return candidate;
    }
    return {};
}

fs::path firstExistingFile(const std::vector<fs::path>& candidates)
{
    for (const auto& candidate : candidates) {
        if (fs::is_regular_file(candidate)) return candidate;
    }
    return {};
}

fs::path resolveLidarDir(const Options& options)
{
    if (!options.lidar_dir.empty()) {
        if (!fs::is_directory(options.lidar_dir)) {
            throw std::runtime_error("lidar directory does not exist: " + options.lidar_dir.string());
        }
        return options.lidar_dir;
    }
    const fs::path seq = options.m2dgr_root / options.sequence;
    const fs::path resolved = firstExistingDirectory({
        seq / "velodyne_points",
        seq / "velodyne_points" / "data",
        seq / "lidar",
        seq / "pcd",
        seq / "clouds",
        seq,
    });
    if (resolved.empty()) {
        throw std::runtime_error("failed to infer lidar directory under " + seq.string() +
                                 "; use --lidar_dir for extracted M2DGR point clouds");
    }
    return resolved;
}

fs::path resolveGtPath(const Options& options)
{
    if (!options.gt_path.empty()) {
        if (!fs::is_regular_file(options.gt_path)) {
            throw std::runtime_error("ground-truth file does not exist: " + options.gt_path.string());
        }
        return options.gt_path;
    }
    const fs::path seq = options.m2dgr_root / options.sequence;
    const fs::path resolved = firstExistingFile({
        seq / "groundtruth.txt",
        seq / "gt.txt",
        seq / "pose.txt",
        seq / "poses.txt",
        options.m2dgr_root / "groundtruth" / (options.sequence + ".txt"),
        options.m2dgr_root / "ground_truth" / (options.sequence + ".txt"),
        options.m2dgr_root / "gt" / (options.sequence + ".txt"),
        options.m2dgr_root / (options.sequence + ".txt"),
    });
    if (resolved.empty()) {
        throw std::runtime_error("failed to infer TUM ground-truth file for sequence " +
                                 options.sequence + "; use --gt");
    }
    return resolved;
}

std::vector<LidarRecord> listLidarRecords(const fs::path& lidar_dir)
{
    std::vector<LidarRecord> records;
    for (const auto& entry : fs::directory_iterator(lidar_dir)) {
        if (!entry.is_regular_file() || !isCloudFile(entry.path())) continue;
        double stamp = 0.0;
        if (!parseStampFromStem(entry.path(), &stamp)) continue;
        records.push_back({stamp, entry.path()});
    }
    std::sort(records.begin(), records.end(), [](const LidarRecord& a, const LidarRecord& b) {
        return a.stamp < b.stamp;
    });
    for (size_t i = 1; i < records.size(); ++i) {
        if (records[i - 1].stamp == records[i].stamp) {
            throw std::runtime_error("duplicate lidar timestamp: " + std::to_string(records[i].stamp));
        }
    }
    if (records.empty()) {
        throw std::runtime_error("no .pcd/.bin lidar clouds found in " + lidar_dir.string());
    }
    return records;
}

std::vector<PoseRecord> readTumGroundTruth(const fs::path& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open ground-truth file: " + path.string());
    }
    std::vector<PoseRecord> poses;
    std::string line;
    int64_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double stamp = 0.0;
        double x = 0.0, y = 0.0, z = 0.0;
        double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
        if (!(iss >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            throw std::runtime_error("failed to parse TUM pose at line " + std::to_string(line_number));
        }
        if (!std::isfinite(stamp) || !std::isfinite(x) || !std::isfinite(y) ||
            !std::isfinite(z) || !std::isfinite(qx) || !std::isfinite(qy) ||
            !std::isfinite(qz) || !std::isfinite(qw)) {
            throw std::runtime_error("non-finite TUM pose at line " + std::to_string(line_number));
        }
        Eigen::Quaterniond q(qw, qx, qy, qz);
        if (q.norm() < 1e-9 || !std::isfinite(q.norm())) {
            throw std::runtime_error("invalid quaternion at line " + std::to_string(line_number));
        }
        q.normalize();
        PoseRecord pose;
        pose.stamp = stamp;
        pose.T_world_lidar = Eigen::Isometry3d::Identity();
        pose.T_world_lidar.linear() = q.toRotationMatrix();
        pose.T_world_lidar.translation() = Eigen::Vector3d(x, y, z);
        poses.push_back(pose);
    }
    std::sort(poses.begin(), poses.end(), [](const PoseRecord& a, const PoseRecord& b) {
        return a.stamp < b.stamp;
    });
    for (size_t i = 1; i < poses.size(); ++i) {
        if (poses[i - 1].stamp == poses[i].stamp) {
            throw std::runtime_error("duplicate ground-truth timestamp: " + std::to_string(poses[i].stamp));
        }
    }
    if (poses.empty()) {
        throw std::runtime_error("ground-truth file contains no poses: " + path.string());
    }
    return poses;
}

std::vector<M2DGRFrame> alignFrames(const std::vector<LidarRecord>& lidar,
                                    const std::vector<PoseRecord>& poses,
                                    const Options& options)
{
    std::vector<M2DGRFrame> aligned;
    size_t pose_index = 0;
    int64_t accepted_count = 0;
    for (const auto& record : lidar) {
        while (pose_index + 1 < poses.size() &&
               std::abs(poses[pose_index + 1].stamp - record.stamp) <=
                   std::abs(poses[pose_index].stamp - record.stamp)) {
            ++pose_index;
        }
        const double diff = std::abs(poses[pose_index].stamp - record.stamp);
        if (diff > options.max_time_diff) continue;
        if (accepted_count % options.stride == 0) {
            M2DGRFrame frame;
            frame.frame_index = aligned.size();
            frame.lidar_stamp = record.stamp;
            frame.gt_stamp = poses[pose_index].stamp;
            frame.time_diff = diff;
            frame.cloud_path = record.path;
            frame.T_world_lidar = poses[pose_index].T_world_lidar;
            aligned.push_back(frame);
            if (options.max_frames >= 0 &&
                static_cast<int64_t>(aligned.size()) >= options.max_frames) {
                break;
            }
        }
        ++accepted_count;
    }
    if (aligned.empty()) {
        throw std::runtime_error("no lidar/GT frames aligned; try increasing --max_time_diff");
    }
    return aligned;
}

Cloud::Ptr readBinCloud(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open lidar bin: " + path.string());
    }
    const std::streamsize size = input.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(4 * sizeof(float)) != 0) {
        throw std::runtime_error("lidar bin size is not a multiple of XYZI float records: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    const size_t point_count = static_cast<size_t>(size) / (4 * sizeof(float));
    auto cloud = pcl::make_shared<Cloud>();
    cloud->reserve(point_count);
    for (size_t i = 0; i < point_count; ++i) {
        float values[4] = {};
        input.read(reinterpret_cast<char*>(values), sizeof(values));
        if (!input.good()) {
            throw std::runtime_error("failed to read lidar bin: " + path.string());
        }
        if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
            !std::isfinite(values[2]) || !std::isfinite(values[3])) {
            continue;
        }
        pcl::PointXYZI point;
        point.x = values[0];
        point.y = values[1];
        point.z = values[2];
        point.intensity = values[3];
        cloud->push_back(point);
    }
    return cloud;
}

Cloud::Ptr readPcdCloud(const fs::path& path)
{
    auto cloud = pcl::make_shared<Cloud>();
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(path.string(), *cloud) < 0) {
        throw std::runtime_error("failed to load PCD cloud: " + path.string());
    }
    auto filtered = pcl::make_shared<Cloud>();
    filtered->reserve(cloud->size());
    for (const auto& point : cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z) || !std::isfinite(point.intensity)) {
            continue;
        }
        filtered->push_back(point);
    }
    return filtered;
}

Cloud::Ptr readCloud(const fs::path& path)
{
    auto cloud = path.extension() == ".pcd" ? readPcdCloud(path) : readBinCloud(path);
    if (!cloud || cloud->empty()) {
        throw std::runtime_error("lidar cloud is empty after finite filtering: " + path.string());
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

core::LioFrame makeFrame(const M2DGRFrame& frame,
                         const Eigen::Isometry3d& pose,
                         const Cloud::Ptr& cloud)
{
    core::LioFrame out;
    out.stamp.nsec = static_cast<int64_t>(std::llround(frame.lidar_stamp * 1e9));
    out.T_world_lidar = pose;
    out.undistorted_cloud = cloud;
    out.pose_valid = true;
    return out;
}

Eigen::Isometry3d fakeMapToOdom(double fake_yaw_deg)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = Eigen::AngleAxisd(fake_yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return T;
}

double yawFromRotation(const Eigen::Matrix3d& rotation)
{
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

double angleDiff(double a, double b)
{
    double diff = a - b;
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    return diff;
}

std::vector<double> sortedFinite(std::vector<double> values)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) {
        return !std::isfinite(v);
    }), values.end());
    std::sort(values.begin(), values.end());
    return values;
}

double percentile(std::vector<double> values, double q)
{
    values = sortedFinite(std::move(values));
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double idx = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double t = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

void writeJsonDoubleOrNull(std::ostream& out, double value)
{
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

void writeTrajectoryLine(std::ofstream& out, double timestamp, const Eigen::Isometry3d& pose)
{
    const Eigen::Quaterniond q(pose.rotation());
    out << std::fixed << std::setprecision(9)
        << timestamp << ' '
        << pose.translation().x() << ' '
        << pose.translation().y() << ' '
        << pose.translation().z() << ' '
        << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
}

void writeKeyframesGtHeader(std::ofstream& out)
{
    out << "keyframe_id,frame_id,x,y,z,qx,qy,qz,qw\n";
}

void writeKeyframeGt(std::ofstream& out, int64_t keyframe_id, const M2DGRFrame& frame)
{
    const Eigen::Quaterniond q(frame.T_world_lidar.rotation());
    out << keyframe_id << ','
        << frame.frame_index << ','
        << std::setprecision(17)
        << frame.T_world_lidar.translation().x() << ','
        << frame.T_world_lidar.translation().y() << ','
        << frame.T_world_lidar.translation().z() << ','
        << q.x() << ','
        << q.y() << ','
        << q.z() << ','
        << q.w() << '\n';
}

void writeAcceptedLoopsHeader(std::ofstream& out)
{
    out << "query_id,match_id,fitness_score,inlier_ratio,verified,edge_mode,vertical_observability_score,"
           "vertical_downweighted,source_z_span,target_z_span,z_overlap_ratio_before,z_overlap_ratio_after,"
           "source_z_robust_span,target_z_robust_span,z_robust_overlap_ratio_before,z_robust_overlap_ratio_after,"
           "source_target_z_centroid_delta_before,source_target_z_centroid_delta_after,vertical_information_ratio\n";
}

void writeAcceptedLoop(std::ofstream& out, const VerifiedLoop& loop)
{
    out << loop.query_id << ','
        << loop.match_id << ','
        << loop.fitness_score << ','
        << loop.inlier_ratio << ','
        << (loop.verified ? "true" : "false") << ','
        << loopEdgeModeName(loop.edge_mode) << ','
        << loop.vertical_observability_score << ','
        << (loop.vertical_downweighted ? "true" : "false") << ','
        << loop.source_z_span << ','
        << loop.target_z_span << ','
        << loop.z_overlap_ratio_before << ','
        << loop.z_overlap_ratio_after << ','
        << loop.source_z_robust_span << ','
        << loop.target_z_robust_span << ','
        << loop.z_robust_overlap_ratio_before << ','
        << loop.z_robust_overlap_ratio_after << ','
        << loop.source_target_z_centroid_delta_before << ','
        << loop.source_target_z_centroid_delta_after << ','
        << loop.vertical_information_ratio << '\n';
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

Cloud::Ptr makeQueryCloud(const Cloud::Ptr& source, const Options& options, size_t seed)
{
    if (!source) return source;
    auto query = pcl::make_shared<Cloud>();
    query->reserve(source->size());
    std::mt19937 rng(static_cast<uint32_t>(seed + 17U));
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, options.noise_sigma);
    for (const auto& point : source->points) {
        if (options.dropout > 0.0 && keep_dist(rng) < options.dropout) continue;
        pcl::PointXYZI p = point;
        if (options.noise_sigma > 0.0) {
            p.x += static_cast<float>(noise(rng));
            p.y += static_cast<float>(noise(rng));
            p.z += static_cast<float>(noise(rng));
        }
        query->push_back(p);
    }
    query->width = static_cast<uint32_t>(query->size());
    query->height = 1;
    query->is_dense = false;
    return query;
}

bool buildTemporaryMap(const Options& options,
                       const std::vector<M2DGRFrame>& frames,
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
        auto cloud = readCloud(frame.cloud_path);
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
        result.translation_error_m <= options.pose_translation_threshold_m &&
        result.yaw_error_deg <= options.pose_yaw_threshold_deg;
}

int runMappingLoop(const Options& options, const std::vector<M2DGRFrame>& frames)
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

    N3MappingCore core(makeEvalConfig(options));
    MappingStats stats;
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& frame = frames[i];
        progress.write(i, frame.lidar_stamp, "read_cloud_start");
        auto cloud = readCloud(frame.cloud_path);
        progress.write(i, frame.lidar_stamp, "read_cloud_done");
        progress.write(i, frame.lidar_stamp, "process_mapping_start");
        const auto output = core.processMappingFrame(makeFrame(frame, frame.T_world_lidar, cloud));
        progress.write(i,
                       frame.lidar_stamp,
                       "process_mapping_done",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success);
        ++stats.frames_processed;
        if (output.success) ++stats.successful_frames;
        if (output.accepted_keyframe) ++stats.accepted_keyframes;
        writeTrajectoryLine(trajectory_gt, frame.lidar_stamp, frame.T_world_lidar);
        writeTrajectoryLine(trajectory_est, frame.lidar_stamp, output.T_world_lidar);
        if (output.accepted_keyframe && output.keyframe_id >= 0) {
            writeKeyframeGt(keyframes_gt, output.keyframe_id, frame);
        }
        trajectory_gt.flush();
        trajectory_est.flush();
        keyframes_gt.flush();

        progress.write(i, frame.lidar_stamp, "process_loops_start", output.keyframe_id, output.accepted_keyframe, output.success);
        const auto loop_result = core.processPendingLoopClosures();
        stats.accepted_loops += loop_result.accepted_loops.size();
        for (const auto& loop : loop_result.accepted_loops) {
            writeAcceptedLoop(accepted_loops, loop);
        }
        accepted_loops.flush();
        progress.write(i,
                       frame.lidar_stamp,
                       "process_loops_done",
                       output.keyframe_id,
                       output.accepted_keyframe,
                       output.success,
                       static_cast<size_t>(loop_result.edge_count),
                       loop_result.accepted_loops.size());
    }
    progress.write(frames.size(), -1.0, "final_process_loops_start");
    const auto final_loop_result = core.processPendingLoopClosures();
    stats.accepted_loops += final_loop_result.accepted_loops.size();
    for (const auto& loop : final_loop_result.accepted_loops) {
        writeAcceptedLoop(accepted_loops, loop);
    }
    accepted_loops.flush();

    const auto dense = core.getDenseOptimizedTrajectory();
    std::ofstream metrics(options.output_dir / "metrics.json");
    metrics << "{\n"
            << "  \"mode\": \"mapping_loop\",\n"
            << "  \"dataset\": \"M2DGR\",\n"
            << "  \"sequence\": \"" << jsonEscape(options.sequence) << "\",\n"
            << "  \"frames_processed\": " << stats.frames_processed << ",\n"
            << "  \"successful_frames\": " << stats.successful_frames << ",\n"
            << "  \"accepted_keyframes\": " << stats.accepted_keyframes << ",\n"
            << "  \"accepted_loop_count\": " << stats.accepted_loops << ",\n"
            << "  \"dense_trajectory_count\": " << dense.size() << ",\n"
            << "  \"stride\": " << options.stride << ",\n"
            << "  \"max_time_diff\": " << options.max_time_diff << ",\n"
            << "  \"lidar_dir\": \"" << jsonEscape(resolveLidarDir(options).string()) << "\",\n"
            << "  \"gt_path\": \"" << jsonEscape(resolveGtPath(options).string()) << "\"\n"
            << "}\n";
    std::cout << "m2dgr mapping_loop frames=" << stats.frames_processed
              << " keyframes=" << stats.accepted_keyframes
              << " loops=" << stats.accepted_loops
              << " output=" << options.output_dir << "\n";
    return 0;
}

int runRelocalization(const Options& options, const std::vector<M2DGRFrame>& frames)
{
    fs::create_directories(options.output_dir);
    touchFile(options.output_dir / "relocalization_debug.jsonl");

    fs::path map_path = options.map_path;
    int build_count = options.build_map_frames;
    if (map_path.empty()) {
        if (build_count <= 0) {
            build_count = std::min<int>(static_cast<int>(frames.size()) / 2, 50);
        }
        if (build_count <= 0 || build_count >= static_cast<int>(frames.size())) {
            throw std::runtime_error("not enough frames to build temporary relocalization map");
        }
        if (!buildTemporaryMap(options, frames, build_count, &map_path)) {
            throw std::runtime_error("failed to build temporary M2DGR relocalization map");
        }
    }

    Config config = makeEvalConfig(options);
    config.mode = "localization";
    config.map_path = map_path.string();
    N3MappingCore core(config);
    if (!core.loadMap(map_path.string())) {
        throw std::runtime_error("failed to load relocalization map: " + map_path.string());
    }

    const Eigen::Isometry3d T_map_odom = fakeMapToOdom(options.fake_yaw_deg);
    std::vector<RelocResult> results;
    const size_t start = map_path == options.map_path ? 0U : static_cast<size_t>(build_count);
    for (size_t i = start; i < frames.size(); ++i) {
        const auto& frame = frames[i];
        auto source_cloud = readCloud(frame.cloud_path);
        auto query_cloud = makeQueryCloud(source_cloud, options, i);
        RelocResult result;
        result.frame_index = i;
        result.timestamp = frame.lidar_stamp;
        const Eigen::Isometry3d T_odom_lidar = T_map_odom.inverse() * frame.T_world_lidar;
        const auto output = core.processLocalizationFrame(makeFrame(frame, T_odom_lidar, query_cloud));
        result.success = output.success;
        result.lock = output.relocalization_locked;
        result.matched_keyframe_id = output.matched_keyframe_id;
        const Eigen::Vector3d dt = output.T_world_lidar.translation() - frame.T_world_lidar.translation();
        result.translation_error_m = dt.norm();
        result.yaw_error_deg = std::abs(angleDiff(
            yawFromRotation(output.T_world_lidar.rotation()),
            yawFromRotation(frame.T_world_lidar.rotation()))) * 180.0 / M_PI;
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
            << "  \"dataset\": \"M2DGR\",\n"
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
            << "  \"fake_yaw_deg\": " << options.fake_yaw_deg << ",\n"
            << "  \"max_time_diff\": " << options.max_time_diff << "\n"
            << "}\n";

    std::ofstream per_query(options.output_dir / "relocalization_queries.csv");
    per_query << "frame_index,timestamp,success,lock,matched_keyframe_id,translation_error_m,yaw_error_deg\n";
    for (const auto& result : results) {
        per_query << result.frame_index << ','
                  << std::fixed << std::setprecision(9) << result.timestamp << ','
                  << (result.success ? "true" : "false") << ','
                  << (result.lock ? "true" : "false") << ','
                  << result.matched_keyframe_id << ','
                  << result.translation_error_m << ','
                  << result.yaw_error_deg << '\n';
    }

    std::cout << "m2dgr relocalization queries=" << results.size()
              << " locks=" << locks
              << " pose_successes=" << pose_successes
              << " output=" << options.output_dir << "\n";
    return 0;
}

int run(const Options& options)
{
    const auto lidar_dir = resolveLidarDir(options);
    const auto gt_path = resolveGtPath(options);
    const auto lidar = listLidarRecords(lidar_dir);
    const auto poses = readTumGroundTruth(gt_path);
    const auto frames = alignFrames(lidar, poses, options);
    if (options.mode == "mapping_loop") {
        return runMappingLoop(options, frames);
    }
    return runRelocalization(options, frames);
}

}  // namespace
}  // namespace n3mapping

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    try {
        return n3mapping::run(n3mapping::parseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "n3mapping_m2dgr_eval: " << e.what() << "\n";
        return 1;
    }
}
