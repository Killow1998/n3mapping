#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <string>
#include <unordered_set>
#include <vector>
#include <Eigen/Geometry>

namespace n3mapping::synthetic {

// Strict CSV schemas accepted by parseQueryPoseManifestCsv():
//   query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg
//   query_id,episode_id,frame_index,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg
// The short schema treats every row as an independent one-frame episode.
struct QueryPoseManifestEntry {
    std::int64_t query_id = -1;
    std::int64_t episode_id = -1;
    std::int64_t frame_index = 0;
    double x_m = 0.0;
    double y_m = 0.0;
    double z_m = 0.0;
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    Eigen::Isometry3d T_map_lidar = Eigen::Isometry3d::Identity();
};

struct QueryPoseManifest {
    bool has_episode_columns = false;
    std::vector<QueryPoseManifestEntry> entries;
};

inline std::string trimManifestField(const std::string& input)
{
    const auto first = input.find_first_not_of(" \t\r");
    if (first == std::string::npos) return {};
    const auto last = input.find_last_not_of(" \t\r");
    return input.substr(first, last - first + 1);
}

inline std::vector<std::string> splitManifestCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        const std::size_t comma = line.find(',', begin);
        fields.push_back(trimManifestField(line.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin)));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return fields;
}

inline bool parseManifestInt64(const std::string& text, std::int64_t& value)
{
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoll(text.c_str(), &end, 10);
    if (end == text.c_str() || end != text.c_str() + text.size() ||
        errno == ERANGE) return false;
    value = parsed;
    return true;
}

inline bool parseManifestFiniteDouble(const std::string& text, double& value)
{
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || end != text.c_str() + text.size() ||
        errno == ERANGE || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

inline bool setManifestError(std::string* error, const std::string& message)
{
    if (error) *error = message;
    return false;
}

inline bool parseQueryPoseManifestCsv(std::istream& input,
                                      QueryPoseManifest* manifest,
                                      std::string* error)
{
    if (!manifest) return setManifestError(error, "manifest_output_is_null");
    *manifest = QueryPoseManifest{};

    std::string line;
    if (!std::getline(input, line)) {
        return setManifestError(error, "manifest_is_empty");
    }
    const auto header = splitManifestCsvLine(line);
    const std::vector<std::string> standalone_header = {
        "query_id", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"};
    const std::vector<std::string> episode_header = {
        "query_id", "episode_id", "frame_index", "x_m", "y_m", "z_m",
        "roll_deg", "pitch_deg", "yaw_deg"};
    if (header == standalone_header) {
        manifest->has_episode_columns = false;
    } else if (header == episode_header) {
        manifest->has_episode_columns = true;
    } else {
        return setManifestError(
            error,
            "manifest_header_mismatch: expected exactly 'query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg' "
            "or 'query_id,episode_id,frame_index,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg'");
    }

    std::unordered_set<std::int64_t> query_ids;
    std::unordered_set<std::int64_t> closed_episode_ids;
    std::int64_t active_episode_id = -1;
    std::int64_t expected_frame_index = 0;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (trimManifestField(line).empty()) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": blank_row");
        }
        if (line.find('"') != std::string::npos) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": quoted_fields_not_supported");
        }
        const auto fields = splitManifestCsvLine(line);
        const std::size_t expected_fields = manifest->has_episode_columns ? 9U : 7U;
        if (fields.size() != expected_fields) {
            return setManifestError(
                error,
                "manifest_line_" + std::to_string(line_number) + ": expected_" +
                    std::to_string(expected_fields) + "_fields_got_" + std::to_string(fields.size()));
        }

        QueryPoseManifestEntry entry;
        std::size_t cursor = 0;
        if (!parseManifestInt64(fields[cursor++], entry.query_id) || entry.query_id < 0) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": invalid_query_id");
        }
        if (!query_ids.insert(entry.query_id).second) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": duplicate_query_id");
        }

        if (manifest->has_episode_columns) {
            if (!parseManifestInt64(fields[cursor++], entry.episode_id) || entry.episode_id < 0) {
                return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": invalid_episode_id");
            }
            if (!parseManifestInt64(fields[cursor++], entry.frame_index) || entry.frame_index < 0) {
                return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": invalid_frame_index");
            }
            if (entry.episode_id != active_episode_id) {
                if (closed_episode_ids.count(entry.episode_id) > 0) {
                    return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": episode_rows_not_contiguous");
                }
                if (active_episode_id >= 0) closed_episode_ids.insert(active_episode_id);
                active_episode_id = entry.episode_id;
                expected_frame_index = 0;
            }
            if (entry.frame_index != expected_frame_index) {
                return setManifestError(
                    error,
                    "manifest_line_" + std::to_string(line_number) + ": frame_index_expected_" +
                        std::to_string(expected_frame_index) + "_got_" + std::to_string(entry.frame_index));
            }
            ++expected_frame_index;
        } else {
            entry.episode_id = entry.query_id;
            entry.frame_index = 0;
        }

        double* pose_fields[] = {
            &entry.x_m, &entry.y_m, &entry.z_m,
            &entry.roll_deg, &entry.pitch_deg, &entry.yaw_deg};
        const char* pose_names[] = {
            "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"};
        for (std::size_t i = 0; i < 6; ++i) {
            if (!parseManifestFiniteDouble(fields[cursor++], *pose_fields[i])) {
                return setManifestError(
                    error,
                    "manifest_line_" + std::to_string(line_number) + ": invalid_" + pose_names[i]);
            }
        }

        entry.T_map_lidar.translation() = Eigen::Vector3d(entry.x_m, entry.y_m, entry.z_m);
        entry.T_map_lidar.linear() =
            (Eigen::AngleAxisd(entry.yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(entry.pitch_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(entry.roll_deg * M_PI / 180.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
        manifest->entries.push_back(entry);
    }

    if (manifest->entries.empty()) {
        return setManifestError(error, "manifest_has_no_query_rows");
    }
    return true;
}

inline bool readQueryPoseManifestCsv(const std::string& path,
                                     QueryPoseManifest* manifest,
                                     std::string* error)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return setManifestError(error, "manifest_open_failed: " + path);
    }
    return parseQueryPoseManifestCsv(input, manifest, error);
}

} // namespace n3mapping::synthetic
