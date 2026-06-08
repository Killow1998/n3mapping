#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
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

struct PoseJitterOptions {
    double xy_m = 0.0;
    double z_m = 0.0;
    double yaw_deg = 0.0;
    double roll_pitch_deg = 0.0;
};

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
                                                  const Cloud::Ptr& reference_cloud = nullptr)
{
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
    const double az_fov = std::clamp(envelope.azimuth_span_deg, 1.0, 360.0);
    const double vertical_fov = std::clamp(envelope.vertical_max_deg - envelope.vertical_min_deg, 1.0, 180.0);
    const int az_bins = use_raycast
        ? std::max(1, static_cast<int>(std::ceil(az_fov / options.raycast_azimuth_resolution_deg)))
        : 0;
    const int vertical_bins = use_raycast
        ? std::max(1, static_cast<int>(std::ceil(vertical_fov / options.raycast_vertical_resolution_deg)))
        : 0;
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
        if (!std::isfinite(range) || range < options.range_min || range > options.range_max) {
            continue;
        }

        double az = 0.0;
        double vert = 0.0;
        const double horizontal_range = std::hypot(p_body.x(), p_body.y());
        az = wrap360(std::atan2(p_body.y(), p_body.x()) * 180.0 / M_PI);
        vert = std::atan2(p_body.z(), horizontal_range) * 180.0 / M_PI;
        if (!withinObservedEnvelope(az, vert, envelope)) {
            continue;
        }

        pcl::PointXYZI point_body;
        point_body.x = static_cast<float>(p_body.x());
        point_body.y = static_cast<float>(p_body.y());
        point_body.z = static_cast<float>(p_body.z());
        point_body.intensity = point_map.intensity;

        if (!use_raycast) {
            if (keep_dist(rng) >= options.dropout) {
                addNoisyPoint(point_body, options.noise_sigma, &rng, body.get());
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
        if (!hit.valid || range < hit.range) {
            hit.valid = true;
            hit.range = range;
            hit.point = point_body;
        }
    }

    if (use_raycast) {
        body->reserve(hits.size());
        const int dilation = std::max(0, options.occlusion_dilation_bins);
        for (int row = 0; row < vertical_bins; ++row) {
            for (int col = 0; col < az_bins; ++col) {
                const auto& hit = hits[static_cast<std::size_t>(row) * static_cast<std::size_t>(az_bins) +
                                       static_cast<std::size_t>(col)];
                if (!hit.valid || keep_dist(rng) < options.dropout) {
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
                    continue;
                }
                addNoisyPoint(hit.point, options.noise_sigma, &rng, body.get());
            }
        }
    }

    body->width = static_cast<std::uint32_t>(body->size());
    body->height = 1;
    body->is_dense = true;
    return body;
}

}  // namespace synthetic
}  // namespace n3mapping
