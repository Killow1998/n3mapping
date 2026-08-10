#include "n3mapping/visibility_consistency.h"

#include "n3mapping/config.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <pcl/common/point_tests.h>

namespace n3mapping {
namespace {

double percentile(std::vector<double> values, double quantile) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double index =
      std::clamp(quantile, 0.0, 1.0) * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  if (lower == upper)
    return values[lower];
  const double alpha = index - static_cast<double>(lower);
  return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

struct SphericalDepthImage {
  int azimuth_bins = 0;
  int elevation_bins = 0;
  double resolution_deg = 1.0;
  std::vector<double> ranges;
  // Populated only for the predicted image, and only when occlusion awareness
  // is on: the accumulated return in each bearing closest to what was measured.
  std::vector<double> match_ranges;
};

SphericalDepthImage makeDepthImage(double resolution_deg) {
  SphericalDepthImage image;
  image.resolution_deg = resolution_deg;
  image.azimuth_bins =
      std::max(1, static_cast<int>(std::ceil(360.0 / resolution_deg)));
  image.elevation_bins =
      std::max(1, static_cast<int>(std::ceil(180.0 / resolution_deg)));
  image.ranges.assign(static_cast<std::size_t>(image.azimuth_bins) *
                          static_cast<std::size_t>(image.elevation_bins),
                      std::numeric_limits<double>::infinity());
  return image;
}

void insertPoint(const Eigen::Vector3d &point,
                 const VisibilityConsistencyOptions &options,
                 SphericalDepthImage *image,
                 const SphericalDepthImage *measured = nullptr) {
  if (!image || !point.array().isFinite().all())
    return;
  const double range = point.norm();
  if (!std::isfinite(range) || range < options.range_min_m ||
      range > options.range_max_m)
    return;

  double azimuth_deg = std::atan2(point.y(), point.x()) * 180.0 / M_PI + 180.0;
  if (azimuth_deg >= 360.0)
    azimuth_deg -= 360.0;
  if (azimuth_deg < 0.0)
    azimuth_deg += 360.0;
  const double elevation_deg =
      std::atan2(point.z(), std::hypot(point.x(), point.y())) * 180.0 / M_PI +
      90.0;
  const int azimuth_index = std::clamp(
      static_cast<int>(std::floor(azimuth_deg / image->resolution_deg)), 0,
      image->azimuth_bins - 1);
  const int elevation_index = std::clamp(
      static_cast<int>(std::floor(elevation_deg / image->resolution_deg)), 0,
      image->elevation_bins - 1);
  const std::size_t index =
      static_cast<std::size_t>(elevation_index) *
          static_cast<std::size_t>(image->azimuth_bins) +
      static_cast<std::size_t>(azimuth_index);
  double &stored = image->ranges[index];
  stored = std::min(stored, range);
  // A spot the session drove through twice contributes twice the surface, and
  // from any one vantage much of it is occluded yet still lands in this
  // bearing. Keeping only the nearest return therefore shortens the prediction
  // in proportion to how thick the map is there. Remember as well the return
  // that best matches the measurement, so occlusion can be told apart from
  // contradiction.
  if (measured != nullptr && !image->match_ranges.empty()) {
    const double measured_range = measured->ranges[index];
    if (std::isfinite(measured_range)) {
      double &best = image->match_ranges[index];
      if (!std::isfinite(best) ||
          std::abs(range - measured_range) < std::abs(best - measured_range)) {
        best = range;
      }
    }
  }
}

} // namespace

VisibilityConsistencyResult evaluateVisibilityConsistency(
    const pcl::PointCloud<pcl::PointXYZI> &map_cloud,
    const pcl::PointCloud<pcl::PointXYZI> &query_cloud,
    const Eigen::Isometry3d &T_map_lidar,
    const VisibilityConsistencyOptions &options) {
  VisibilityConsistencyResult result;
  if (map_cloud.empty() || query_cloud.empty() ||
      !T_map_lidar.matrix().array().isFinite().all() ||
      options.range_max_m <= options.range_min_m ||
      options.range_tolerance_m <= 0.0) {
    return result;
  }

  std::vector<double> query_ranges;
  query_ranges.reserve(query_cloud.size());
  for (const auto &point : query_cloud) {
    if (!pcl::isFinite(point))
      continue;
    const double range = Eigen::Vector3d(point.x, point.y, point.z).norm();
    if (std::isfinite(range) && range >= options.range_min_m &&
        range <= options.range_max_m) {
      query_ranges.push_back(range);
    }
  }
  if (query_ranges.empty())
    return result;

  const double median_query_range = percentile(query_ranges, 0.5);
  const double inferred_resolution_deg =
      std::atan2(options.range_tolerance_m,
                 std::max(options.range_min_m, median_query_range)) *
      180.0 / M_PI;
  const double resolution_deg =
      std::clamp(inferred_resolution_deg,
                 std::max(1e-3, options.min_angular_resolution_deg),
                 std::max(options.min_angular_resolution_deg,
                          options.max_angular_resolution_deg));
  auto observed = makeDepthImage(resolution_deg);
  auto predicted = makeDepthImage(resolution_deg);

  for (const auto &point : query_cloud) {
    if (!pcl::isFinite(point))
      continue;
    insertPoint(Eigen::Vector3d(point.x, point.y, point.z), options, &observed);
  }
  if (options.occlusion_aware) {
    predicted.match_ranges.assign(predicted.ranges.size(),
                                  std::numeric_limits<double>::infinity());
  }
  const Eigen::Isometry3d T_lidar_map = T_map_lidar.inverse();
  for (const auto &point : map_cloud) {
    if (!pcl::isFinite(point))
      continue;
    insertPoint(T_lidar_map * Eigen::Vector3d(point.x, point.y, point.z),
                options, &predicted,
                options.occlusion_aware ? &observed : nullptr);
  }

  std::vector<double> residuals;
  residuals.reserve(observed.ranges.size());
  for (std::size_t i = 0; i < observed.ranges.size(); ++i) {
    const bool has_observation = std::isfinite(observed.ranges[i]);
    const bool has_prediction = std::isfinite(predicted.ranges[i]);
    if (has_observation)
      ++result.observed_bins;
    if (has_prediction)
      ++result.predicted_bins;
    if (!has_observation || !has_prediction)
      continue;
    ++result.common_bins;
    // With occlusion awareness the bearing is judged against the accumulated
    // return that best explains the measurement; without it, against the
    // nearest, which is the published behaviour.
    const double predicted_range =
        (options.occlusion_aware && !predicted.match_ranges.empty() &&
         std::isfinite(predicted.match_ranges[i]))
            ? predicted.match_ranges[i]
            : predicted.ranges[i];
    const double signed_error = predicted_range - observed.ranges[i];
    const double absolute_error = std::abs(signed_error);
    residuals.push_back(absolute_error);
    if (absolute_error <= options.range_tolerance_m)
      ++result.consistent_bins;
    if (signed_error < -options.range_tolerance_m)
      ++result.foreground_conflict_bins;
  }

  result.known_bins = result.common_bins;
  result.unknown_bins = result.observed_bins - result.common_bins;
  if (result.observed_bins > 0) {
    result.known_fraction =
        static_cast<double>(result.known_bins) /
        static_cast<double>(result.observed_bins);
  }
  constexpr double kJeffreysPrior = 0.5;
  if (result.known_bins > 0) {
    const double known_denominator =
        static_cast<double>(result.known_bins);
    result.consistent_given_known =
        static_cast<double>(result.consistent_bins) / known_denominator;
    result.foreground_conflict_given_known =
        static_cast<double>(result.foreground_conflict_bins) /
        known_denominator;
    result.evidence_log_odds_given_known = std::log(
        (static_cast<double>(result.consistent_bins) + kJeffreysPrior) /
        (static_cast<double>(result.known_bins - result.consistent_bins) +
         kJeffreysPrior));
  }

  if (result.observed_bins == 0 || result.predicted_bins == 0)
    return result;
  const double observed_denominator = static_cast<double>(result.observed_bins);
  result.valid = true;
  result.angular_resolution_deg = resolution_deg;
  result.observed_coverage =
      static_cast<double>(result.common_bins) / observed_denominator;
  result.consistency_ratio =
      static_cast<double>(result.consistent_bins) / observed_denominator;
  result.foreground_conflict_ratio =
      static_cast<double>(result.foreground_conflict_bins) /
      observed_denominator;
  // Jeffreys-prior smoothing keeps exact agreement/disagreement finite without
  // a tuned clamp. The score is the per-observation log odds that the pose
  // explains a measured first return rather than contradicting it.
  result.evidence_log_odds = std::log(
      (static_cast<double>(result.consistent_bins) + kJeffreysPrior) /
      (static_cast<double>(result.observed_bins - result.consistent_bins) +
       kJeffreysPrior));
  result.median_abs_range_error_m = percentile(residuals, 0.5);
  result.p90_abs_range_error_m = percentile(residuals, 0.9);
  return result;
}

VisibilityConsistencyOptions visibilityOptionsFromConfig(const Config& config) {
  VisibilityConsistencyOptions options;
  options.range_min_m = 0.5;
  options.range_max_m = std::max(1.0, config.rhpd_max_range);
  options.range_tolerance_m =
      std::max(0.05, 3.0 * std::max(config.global_map_voxel_size,
                                    config.gicp_downsampling_resolution));
  options.occlusion_aware = config.reloc_visibility_occlusion_aware;
  return options;
}

} // namespace n3mapping
