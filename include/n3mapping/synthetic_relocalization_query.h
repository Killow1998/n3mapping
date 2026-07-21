#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {
namespace synthetic {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;

struct QuerySynthesisOptions {
    double dropout = 0.0;
    double noise_sigma = 0.0;
    double range_min = 0.5;
    double range_max = 30.0;
    // <= 0 means infer the scan angular envelope from the reference keyframe cloud.
    double fov_azimuth_deg = 0.0;
    double fov_vertical_deg = 0.0;
    double raycast_azimuth_resolution_deg = 1.0;
    double raycast_vertical_resolution_deg = 1.0;
    int occlusion_dilation_bins = 1;
    double occlusion_depth_tolerance_m = 0.3;
};

struct QueryVisibilityStats {
    std::size_t input_map_points = 0;
    std::size_t finite_points = 0;
    std::size_t range_eligible_points = 0;
    std::size_t fov_eligible_points = 0;
    std::size_t azimuth_bins = 0;
    std::size_t vertical_bins = 0;
    std::size_t occupied_ray_bins = 0;
    std::size_t same_ray_occluded_points = 0;
    std::size_t dropout_suppressed_points = 0;
    std::size_t occlusion_suppressed_bins = 0;
    std::size_t output_points = 0;
    bool raycast_enabled = false;
    bool envelope_valid = false;
    bool envelope_azimuth_full = true;
    double envelope_azimuth_start_deg = 0.0;
    double envelope_azimuth_span_deg = 360.0;
    double envelope_vertical_min_deg = -30.0;
    double envelope_vertical_max_deg = 30.0;
};

inline bool queryVisibilityStatsConserved(const QueryVisibilityStats& stats)
{
    if (stats.finite_points > stats.input_map_points ||
        stats.range_eligible_points > stats.finite_points ||
        stats.fov_eligible_points > stats.range_eligible_points) {
        return false;
    }
    if (stats.raycast_enabled) {
        if (stats.azimuth_bins == 0 || stats.vertical_bins == 0 ||
            stats.azimuth_bins > std::numeric_limits<std::size_t>::max() / stats.vertical_bins ||
            stats.occupied_ray_bins > stats.azimuth_bins * stats.vertical_bins ||
            stats.same_ray_occluded_points > stats.fov_eligible_points ||
            stats.occupied_ray_bins != stats.fov_eligible_points - stats.same_ray_occluded_points ||
            stats.dropout_suppressed_points > stats.occupied_ray_bins ||
            stats.occlusion_suppressed_bins >
                stats.occupied_ray_bins - stats.dropout_suppressed_points) {
            return false;
        }
        return stats.output_points ==
               stats.occupied_ray_bins - stats.dropout_suppressed_points -
                   stats.occlusion_suppressed_bins;
    }
    return stats.azimuth_bins == 0 && stats.vertical_bins == 0 &&
           stats.occupied_ray_bins == 0 && stats.same_ray_occluded_points == 0 &&
           stats.occlusion_suppressed_bins == 0 &&
           stats.dropout_suppressed_points <= stats.fov_eligible_points &&
           stats.output_points == stats.fov_eligible_points - stats.dropout_suppressed_points;
}

inline constexpr char kQueryCloudFingerprintAlgorithm[] =
    "fnv1a64_xyz_intensity_float32_le_v1";
inline constexpr int kEvaluationCsvDoublePrecision =
    std::numeric_limits<double>::max_digits10;

struct QueryGenerationThresholds {
    std::size_t minimum_query_points = 1;
    std::size_t minimum_occupied_ray_bins = 1;
};

enum class QueryGenerationSupport {
    SUFFICIENT,
    INSUFFICIENT_QUERY_POINTS,
    INSUFFICIENT_OCCUPIED_RAY_BINS,
};

inline QueryGenerationThresholds queryGenerationThresholds(bool product_default)
{
    return product_default
        ? QueryGenerationThresholds{100U, 100U}
        : QueryGenerationThresholds{1U, 1U};
}

inline QueryGenerationSupport evaluateQueryGenerationSupport(
    std::size_t query_points,
    std::size_t occupied_ray_bins,
    bool enforce_occupied_ray_bins,
    const QueryGenerationThresholds& thresholds)
{
    if (query_points < thresholds.minimum_query_points) {
        return QueryGenerationSupport::INSUFFICIENT_QUERY_POINTS;
    }
    if (enforce_occupied_ray_bins &&
        occupied_ray_bins < thresholds.minimum_occupied_ray_bins) {
        return QueryGenerationSupport::INSUFFICIENT_OCCUPIED_RAY_BINS;
    }
    return QueryGenerationSupport::SUFFICIENT;
}

inline const char* queryGenerationSupportReason(QueryGenerationSupport support)
{
    switch (support) {
    case QueryGenerationSupport::SUFFICIENT: return "ok";
    case QueryGenerationSupport::INSUFFICIENT_QUERY_POINTS:
        return "insufficient_query_points";
    case QueryGenerationSupport::INSUFFICIENT_OCCUPIED_RAY_BINS:
        return "insufficient_occupied_ray_bins";
    }
    return "unknown_query_generation_support";
}

inline std::uint64_t fingerprintPointSequenceFNV1a64(const Cloud& cloud)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
                  "fingerprint requires IEEE-754 float32 storage");
    constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t hash = kOffsetBasis;
    const auto append_float32_le = [&](float value, std::uint64_t* state) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            *state ^= static_cast<std::uint8_t>((bits >> shift) & 0xffU);
            *state *= kPrime;
        }
    };
    for (const auto& point : cloud.points) {
        append_float32_le(point.x, &hash);
        append_float32_le(point.y, &hash);
        append_float32_le(point.z, &hash);
        append_float32_le(point.intensity, &hash);
    }
    return hash;
}

enum class EpisodeFrameStage {
    INITIAL_LOCALIZATION,
    INITIAL_LOCK,
    REENTRY_LOCALIZATION,
    REENTRY_LOCK,
    TRACKING,
    TRACKING_LOSS,
};

struct EpisodeFrameTransition {
    bool locked_before_frame = false;
    bool locked_after_frame = false;
    bool ever_locked_before_frame = false;
    bool ever_locked_after_frame = false;
    bool localization_attempted = false;
    bool tracking_processed = false;
    bool lock_event = false;
    bool tracking_loss_event = false;
    EpisodeFrameStage stage = EpisodeFrameStage::INITIAL_LOCALIZATION;
};

inline const char* episodeFrameStageName(EpisodeFrameStage stage)
{
    switch (stage) {
    case EpisodeFrameStage::INITIAL_LOCALIZATION: return "initial_localization";
    case EpisodeFrameStage::INITIAL_LOCK: return "initial_lock";
    case EpisodeFrameStage::REENTRY_LOCALIZATION: return "reentry_localization";
    case EpisodeFrameStage::REENTRY_LOCK: return "reentry_lock";
    case EpisodeFrameStage::TRACKING: return "tracking";
    case EpisodeFrameStage::TRACKING_LOSS: return "tracking_loss";
    }
    return "unknown";
}

inline const char* episodeLockEventTypeName(EpisodeFrameStage stage)
{
    if (stage == EpisodeFrameStage::INITIAL_LOCK) return "initial_lock";
    if (stage == EpisodeFrameStage::REENTRY_LOCK) return "reentry_lock";
    return "none";
}

// Audit-only wrapper state for processLocalizationFrame(). The runtime localizer
// remains the source of output_success/output_relocalization_locked; this helper
// only makes the episode transition and its scoring/denominator semantics explicit.
inline EpisodeFrameTransition advanceEpisodeFrameState(
    bool current_locked,
    bool ever_locked,
    bool output_success,
    bool output_relocalization_locked)
{
    EpisodeFrameTransition transition;
    transition.locked_before_frame = current_locked;
    transition.locked_after_frame = current_locked;
    transition.ever_locked_before_frame = ever_locked;
    transition.ever_locked_after_frame = ever_locked;
    transition.localization_attempted = !current_locked;
    transition.tracking_processed = current_locked;

    if (output_relocalization_locked) {
        transition.localization_attempted = true;
        transition.lock_event = true;
        transition.locked_after_frame = true;
        transition.ever_locked_after_frame = true;
        transition.stage = ever_locked
            ? EpisodeFrameStage::REENTRY_LOCK
            : EpisodeFrameStage::INITIAL_LOCK;
        return transition;
    }

    if (current_locked) {
        if (output_success) {
            transition.stage = EpisodeFrameStage::TRACKING;
        } else {
            transition.locked_after_frame = false;
            transition.tracking_loss_event = true;
            transition.stage = EpisodeFrameStage::TRACKING_LOSS;
        }
        return transition;
    }

    transition.stage = ever_locked
        ? EpisodeFrameStage::REENTRY_LOCALIZATION
        : EpisodeFrameStage::INITIAL_LOCALIZATION;
    return transition;
}

struct PoseJitterOptions {
    double xy_m = 0.0;
    double z_m = 0.0;
    double yaw_deg = 0.0;
    double roll_pitch_deg = 0.0;
};

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

inline bool parseManifestInt64(const std::string& text, std::int64_t* value)
{
    if (!value || text.empty()) return false;
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(text, &consumed, 10);
        if (consumed != text.size()) return false;
        *value = static_cast<std::int64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parseManifestFiniteDouble(const std::string& text, double* value)
{
    if (!value || text.empty()) return false;
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(parsed)) return false;
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
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
        if (!parseManifestInt64(fields[cursor++], &entry.query_id) || entry.query_id < 0) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": invalid_query_id");
        }
        if (!query_ids.insert(entry.query_id).second) {
            return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": duplicate_query_id");
        }

        if (manifest->has_episode_columns) {
            if (!parseManifestInt64(fields[cursor++], &entry.episode_id) || entry.episode_id < 0) {
                return setManifestError(error, "manifest_line_" + std::to_string(line_number) + ": invalid_episode_id");
            }
            if (!parseManifestInt64(fields[cursor++], &entry.frame_index) || entry.frame_index < 0) {
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
            if (!parseManifestFiniteDouble(fields[cursor++], pose_fields[i])) {
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

inline double degToRad(double deg)
{
    return deg * M_PI / 180.0;
}

inline Eigen::Isometry3d applyUniformPoseJitter(const Eigen::Isometry3d& base,
                                                const PoseJitterOptions& options,
                                                std::mt19937* rng)
{
    if (!rng || (options.xy_m <= 0.0 && options.z_m <= 0.0 &&
                 options.yaw_deg <= 0.0 && options.roll_pitch_deg <= 0.0)) {
        return base;
    }

    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * M_PI);
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    if (options.xy_m > 0.0) {
        const double r = options.xy_m * std::sqrt(std::max(0.0, (unit(*rng) + 1.0) * 0.5));
        const double a = angle(*rng);
        t.x() = r * std::cos(a);
        t.y() = r * std::sin(a);
    }
    if (options.z_m > 0.0) {
        t.z() = options.z_m * unit(*rng);
    }

    const double roll = options.roll_pitch_deg > 0.0 ? degToRad(options.roll_pitch_deg * unit(*rng)) : 0.0;
    const double pitch = options.roll_pitch_deg > 0.0 ? degToRad(options.roll_pitch_deg * unit(*rng)) : 0.0;
    const double yaw = options.yaw_deg > 0.0 ? degToRad(options.yaw_deg * unit(*rng)) : 0.0;

    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    delta.translation() = t;
    delta.linear() =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return base * delta;
}

inline double wrap360(double deg)
{
    while (deg < 0.0) deg += 360.0;
    while (deg >= 360.0) deg -= 360.0;
    return deg;
}

struct ObservedScanEnvelope {
    bool valid = false;
    bool azimuth_full = true;
    double azimuth_start_deg = 0.0;
    double azimuth_span_deg = 360.0;
    double vertical_min_deg = -30.0;
    double vertical_max_deg = 30.0;
};

inline double percentileSorted(const std::vector<double>& sorted, double q)
{
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) {
        return sorted[lo];
    }
    const double alpha = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - alpha) + sorted[hi] * alpha;
}

inline ObservedScanEnvelope estimateObservedScanEnvelope(const Cloud::Ptr& reference_cloud,
                                                         const QuerySynthesisOptions& options)
{
    ObservedScanEnvelope envelope;
    if (!reference_cloud || reference_cloud->empty()) {
        envelope.azimuth_full = options.fov_azimuth_deg <= 0.0 || options.fov_azimuth_deg >= 360.0;
        envelope.azimuth_start_deg = envelope.azimuth_full ? 0.0 : -options.fov_azimuth_deg * 0.5;
        envelope.azimuth_span_deg = envelope.azimuth_full ? 360.0 : options.fov_azimuth_deg;
        const double vertical = options.fov_vertical_deg > 0.0 ? options.fov_vertical_deg : 60.0;
        envelope.vertical_min_deg = -vertical * 0.5;
        envelope.vertical_max_deg = vertical * 0.5;
        return envelope;
    }

    std::vector<double> azimuths;
    std::vector<double> verticals;
    azimuths.reserve(reference_cloud->size());
    verticals.reserve(reference_cloud->size());
    for (const auto& p : reference_cloud->points) {
        const Eigen::Vector3d v(p.x, p.y, p.z);
        const double range = v.norm();
        if (!std::isfinite(range) || range < options.range_min || range > options.range_max) {
            continue;
        }
        const double horizontal = std::hypot(v.x(), v.y());
        azimuths.push_back(wrap360(std::atan2(v.y(), v.x()) * 180.0 / M_PI));
        verticals.push_back(std::atan2(v.z(), horizontal) * 180.0 / M_PI);
    }
    if (azimuths.size() < 8 || verticals.size() < 8) {
        return estimateObservedScanEnvelope(nullptr, options);
    }

    std::sort(azimuths.begin(), azimuths.end());
    std::sort(verticals.begin(), verticals.end());

    double largest_gap = -1.0;
    std::size_t gap_index = 0;
    for (std::size_t i = 0; i < azimuths.size(); ++i) {
        const double a = azimuths[i];
        const double b = (i + 1 < azimuths.size()) ? azimuths[i + 1] : azimuths.front() + 360.0;
        const double gap = b - a;
        if (gap > largest_gap) {
            largest_gap = gap;
            gap_index = i;
        }
    }

    envelope.valid = true;
    envelope.azimuth_span_deg = std::max(1.0, 360.0 - largest_gap);
    envelope.azimuth_full = envelope.azimuth_span_deg >= 330.0;
    if (options.fov_azimuth_deg > 0.0) {
        envelope.azimuth_span_deg = std::min(envelope.azimuth_span_deg, options.fov_azimuth_deg);
        envelope.azimuth_full = envelope.azimuth_span_deg >= 330.0;
    }
    const double observed_start = wrap360(azimuths[(gap_index + 1) % azimuths.size()]);
    if (envelope.azimuth_full) {
        envelope.azimuth_start_deg = 0.0;
        envelope.azimuth_span_deg = 360.0;
    } else {
        const double observed_center = wrap360(observed_start + (360.0 - largest_gap) * 0.5);
        envelope.azimuth_start_deg = wrap360(observed_center - envelope.azimuth_span_deg * 0.5);
    }

    const double observed_vmin = percentileSorted(verticals, 0.01);
    const double observed_vmax = percentileSorted(verticals, 0.99);
    if (options.fov_vertical_deg > 0.0) {
        const double center = 0.5 * (observed_vmin + observed_vmax);
        const double half = 0.5 * std::min(options.fov_vertical_deg, std::max(1.0, observed_vmax - observed_vmin));
        envelope.vertical_min_deg = center - half;
        envelope.vertical_max_deg = center + half;
    } else {
        envelope.vertical_min_deg = observed_vmin;
        envelope.vertical_max_deg = observed_vmax;
    }
    return envelope;
}

inline bool withinObservedEnvelope(double azimuth_deg,
                                   double vertical_deg,
                                   const ObservedScanEnvelope& envelope)
{
    if (vertical_deg < envelope.vertical_min_deg || vertical_deg > envelope.vertical_max_deg) {
        return false;
    }
    if (envelope.azimuth_full) {
        return true;
    }
    const double rel = wrap360(azimuth_deg - envelope.azimuth_start_deg);
    return rel <= envelope.azimuth_span_deg;
}

inline void addNoisyPoint(const pcl::PointXYZI& point_body,
                          double noise_sigma,
                          std::mt19937* rng,
                          Cloud* output)
{
    pcl::PointXYZI out = point_body;
    if (rng && noise_sigma > 0.0) {
        std::normal_distribution<double> noise(0.0, noise_sigma);
        out.x = static_cast<float>(out.x + noise(*rng));
        out.y = static_cast<float>(out.y + noise(*rng));
        out.z = static_cast<float>(out.z + noise(*rng));
    }
    output->push_back(out);
}

inline Cloud::Ptr synthesizeBodyCloudFromMapCloud(const Cloud::Ptr& map_cloud,
                                                  const Eigen::Isometry3d& T_map_lidar,
                                                  const QuerySynthesisOptions& options,
                                                  std::uint32_t seed,
                                                  const Cloud::Ptr& reference_cloud = nullptr,
                                                  QueryVisibilityStats* visibility_stats = nullptr)
{
    if (visibility_stats) {
        *visibility_stats = QueryVisibilityStats{};
        visibility_stats->input_map_points = map_cloud ? map_cloud->size() : 0U;
    }
    auto body = pcl::make_shared<Cloud>();
    if (!map_cloud || map_cloud->empty()) {
        return body;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    const Eigen::Isometry3d T_lidar_map = T_map_lidar.inverse();

    struct Hit {
        bool valid = false;
        double range = std::numeric_limits<double>::infinity();
        pcl::PointXYZI point;
    };

    const bool use_raycast = options.raycast_azimuth_resolution_deg > 0.0 &&
                             options.raycast_vertical_resolution_deg > 0.0;
    const ObservedScanEnvelope envelope = estimateObservedScanEnvelope(reference_cloud, options);
    if (visibility_stats) {
        visibility_stats->raycast_enabled = use_raycast;
        visibility_stats->envelope_valid = envelope.valid;
        visibility_stats->envelope_azimuth_full = envelope.azimuth_full;
        visibility_stats->envelope_azimuth_start_deg = envelope.azimuth_start_deg;
        visibility_stats->envelope_azimuth_span_deg = envelope.azimuth_span_deg;
        visibility_stats->envelope_vertical_min_deg = envelope.vertical_min_deg;
        visibility_stats->envelope_vertical_max_deg = envelope.vertical_max_deg;
    }
    const double az_fov = std::clamp(envelope.azimuth_span_deg, 1.0, 360.0);
    const double vertical_fov = std::clamp(envelope.vertical_max_deg - envelope.vertical_min_deg, 1.0, 180.0);
    const int az_bins = use_raycast
        ? std::max(1, static_cast<int>(std::ceil(az_fov / options.raycast_azimuth_resolution_deg)))
        : 0;
    const int vertical_bins = use_raycast
        ? std::max(1, static_cast<int>(std::ceil(vertical_fov / options.raycast_vertical_resolution_deg)))
        : 0;
    if (visibility_stats) {
        visibility_stats->azimuth_bins = static_cast<std::size_t>(az_bins);
        visibility_stats->vertical_bins = static_cast<std::size_t>(vertical_bins);
    }
    std::vector<Hit> hits;
    if (use_raycast) {
        hits.resize(static_cast<std::size_t>(az_bins) * static_cast<std::size_t>(vertical_bins));
    } else {
        body->reserve(map_cloud->size());
    }

    for (const auto& point_map : map_cloud->points) {
        const Eigen::Vector3d p_body =
            T_lidar_map * Eigen::Vector3d(point_map.x, point_map.y, point_map.z);
        const double range = p_body.norm();
        if (!p_body.array().isFinite().all() || !std::isfinite(range)) {
            continue;
        }
        if (visibility_stats) ++visibility_stats->finite_points;
        if (range < options.range_min || range > options.range_max) {
            continue;
        }
        if (visibility_stats) ++visibility_stats->range_eligible_points;

        double az = 0.0;
        double vert = 0.0;
        const double horizontal_range = std::hypot(p_body.x(), p_body.y());
        az = wrap360(std::atan2(p_body.y(), p_body.x()) * 180.0 / M_PI);
        vert = std::atan2(p_body.z(), horizontal_range) * 180.0 / M_PI;
        if (!withinObservedEnvelope(az, vert, envelope)) {
            continue;
        }
        if (visibility_stats) ++visibility_stats->fov_eligible_points;

        pcl::PointXYZI point_body;
        point_body.x = static_cast<float>(p_body.x());
        point_body.y = static_cast<float>(p_body.y());
        point_body.z = static_cast<float>(p_body.z());
        point_body.intensity = point_map.intensity;

        if (!use_raycast) {
            if (keep_dist(rng) >= options.dropout) {
                addNoisyPoint(point_body, options.noise_sigma, &rng, body.get());
            } else if (visibility_stats) {
                ++visibility_stats->dropout_suppressed_points;
            }
            continue;
        }

        const double az_norm = envelope.azimuth_full ? az : wrap360(az - envelope.azimuth_start_deg);
        const double vert_norm = vert - envelope.vertical_min_deg;
        const int az_idx = std::clamp(
            static_cast<int>(std::floor(az_norm / std::max(1e-6, options.raycast_azimuth_resolution_deg))),
            0,
            az_bins - 1);
        const int vert_idx = std::clamp(
            static_cast<int>(std::floor(vert_norm / std::max(1e-6, options.raycast_vertical_resolution_deg))),
            0,
            vertical_bins - 1);
        Hit& hit = hits[static_cast<std::size_t>(vert_idx) * static_cast<std::size_t>(az_bins) +
                        static_cast<std::size_t>(az_idx)];
        if (!hit.valid) {
            if (visibility_stats) ++visibility_stats->occupied_ray_bins;
            hit.valid = true;
            hit.range = range;
            hit.point = point_body;
        } else {
            if (visibility_stats) ++visibility_stats->same_ray_occluded_points;
            if (range < hit.range) {
                hit.range = range;
                hit.point = point_body;
            }
        }
    }

    if (use_raycast) {
        body->reserve(hits.size());
        const int dilation = std::max(0, options.occlusion_dilation_bins);
        for (int row = 0; row < vertical_bins; ++row) {
            for (int col = 0; col < az_bins; ++col) {
                const auto& hit = hits[static_cast<std::size_t>(row) * static_cast<std::size_t>(az_bins) +
                                       static_cast<std::size_t>(col)];
                if (!hit.valid) {
                    continue;
                }
                if (keep_dist(rng) < options.dropout) {
                    if (visibility_stats) ++visibility_stats->dropout_suppressed_points;
                    continue;
                }
                double nearest = hit.range;
                for (int dr = -dilation; dr <= dilation; ++dr) {
                    const int rr = row + dr;
                    if (rr < 0 || rr >= vertical_bins) continue;
                    for (int dc = -dilation; dc <= dilation; ++dc) {
                        int cc = col + dc;
                        if (envelope.azimuth_full) {
                            cc = (cc % az_bins + az_bins) % az_bins;
                        } else if (cc < 0 || cc >= az_bins) {
                            continue;
                        }
                        const auto& neighbor =
                            hits[static_cast<std::size_t>(rr) * static_cast<std::size_t>(az_bins) +
                                 static_cast<std::size_t>(cc)];
                        if (neighbor.valid) {
                            nearest = std::min(nearest, neighbor.range);
                        }
                    }
                }
                if (hit.range > nearest + std::max(0.0, options.occlusion_depth_tolerance_m)) {
                    if (visibility_stats) ++visibility_stats->occlusion_suppressed_bins;
                    continue;
                }
                addNoisyPoint(hit.point, options.noise_sigma, &rng, body.get());
            }
        }
    }

    body->width = static_cast<std::uint32_t>(body->size());
    body->height = 1;
    body->is_dense = true;
    if (visibility_stats) visibility_stats->output_points = body->size();
    return body;
}

}  // namespace synthetic
}  // namespace n3mapping
