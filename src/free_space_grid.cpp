// free_space_grid.cpp
#include "n3mapping/free_space_grid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <flann/algorithms/dist.h>
#include <flann/algorithms/kdtree_single_index.h>
#include <flann/util/result_set.h>

namespace n3mapping {

bool FreeSpaceGrid::Score::significantlyBetterThan(const Score& other, double sigmas) const {
  return value - other.value > sigmas * std::hypot(stddev, other.stddev);
}

bool FreeSpaceGrid::build(const pcl::PointCloud<pcl::PointXYZI> &map_cloud,
                          const std::vector<Eigen::Vector3d> &observation_origins,
                          double resolution, double max_ray_length,
                          int occupied_min_points) {
  valid_ = false;
  occupied_.clear();
  free_.clear();
  occupied_cells_ = 0;
  free_cells_ = 0;
  build_seconds_ = 0.0;
  if (map_cloud.empty() || observation_origins.empty() || resolution <= 0.0)
    return false;
  const auto started = std::chrono::steady_clock::now();
  resolution_ = resolution;

  Eigen::Vector3d lo =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d hi = -lo;
  for (const auto &pt : map_cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3d p(pt.x, pt.y, pt.z);
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  for (const auto &o : observation_origins) {
    lo = lo.cwiseMin(o);
    hi = hi.cwiseMax(o);
  }
  if (!lo.allFinite() || !hi.allFinite())
    return false;

  origin_ = lo - Eigen::Vector3d::Constant(2.0 * resolution_);
  const Eigen::Vector3d extent =
      hi + Eigen::Vector3d::Constant(2.0 * resolution_) - origin_;
  for (int axis = 0; axis < 3; ++axis) {
    dims_[axis] =
        static_cast<int>(std::ceil(extent[axis] / resolution_)) + 1;
    if (dims_[axis] <= 0)
      return false;
  }
  const std::size_t cells = static_cast<std::size_t>(dims_.x()) *
                            static_cast<std::size_t>(dims_.y()) *
                            static_cast<std::size_t>(dims_.z());

  // Occupancy counts live in a saturating byte per cell only while the grid is
  // being built; the grid itself keeps one bit per cell per layer.
  {
    std::vector<std::uint8_t> counts(cells, 0);
    for (const auto &pt : map_cloud.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;
      const Eigen::Vector3i c = cellOf(Eigen::Vector3d(pt.x, pt.y, pt.z));
      if (!inside(c))
        continue;
      std::uint8_t &n = counts[indexOf(c)];
      if (n < 255)
        ++n;
    }
    const std::uint8_t min_points =
        static_cast<std::uint8_t>(std::max(1, occupied_min_points));
    occupied_.assign(cells, false);
    for (std::size_t i = 0; i < cells; ++i) {
      if (counts[i] >= min_points) {
        occupied_[i] = true;
        ++occupied_cells_;
      }
    }
  }

  // Ray origins: the nearest position the map was observed from. This is an
  // approximation of the true per-point observation frame, and it is the same
  // approximation the offline validation used.
  // Use the same float coordinates, L2 distance and 15-point leaves as
  // KdTreeFLANN. Reuse its one-neighbor result storage instead of allocating
  // a query vector and entering a one-query OpenMP region for every map point.
  std::vector<float> coordinates;
  std::vector<std::size_t> origin_indices;
  coordinates.reserve(observation_origins.size() * 3);
  origin_indices.reserve(observation_origins.size());
  for (std::size_t i = 0; i < observation_origins.size(); ++i) {
    const auto point = observation_origins[i].cast<float>().eval();
    if (!point.allFinite()) continue;
    origin_indices.push_back(i);
    for (int axis = 0; axis < 3; ++axis) coordinates.push_back(point[axis]);
  }
  if (origin_indices.empty()) return false;
  flann::KDTreeSingleIndex<flann::L2_Simple<float>> origin_tree(
      flann::Matrix<float>(coordinates.data(), origin_indices.size(), 3),
      flann::KDTreeSingleIndexParams(15));
  origin_tree.buildIndex();
  flann::KNNSimpleResultSet<float> nearest(1);
  const flann::SearchParams search(-1, 0.0f);

  free_.assign(cells, false);
  for (const auto &pt : map_cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3d end(pt.x, pt.y, pt.z);
    const float probe[3] = {pt.x, pt.y, pt.z};
    nearest.clear();
    origin_tree.findNeighbors(nearest, probe, search);
    if (nearest.size() == 0) continue;
    std::size_t nearest_index = 0;
    float nearest_distance = 0.0f;
    nearest.copy(&nearest_index, &nearest_distance, 1, true);
    if (nearest_distance > max_ray_length * max_ray_length) continue;
    const Eigen::Vector3d start = observation_origins[origin_indices[nearest_index]];
    const Eigen::Vector3d delta = end - start;
    const double length = delta.norm();
    if (length < resolution_)
      continue;
    const Eigen::Vector3d dir = delta / length;
    const int steps = static_cast<int>(length / resolution_);
    for (int s = 1; s < steps; ++s) {
      const Eigen::Vector3i c =
          cellOf(start + dir * (static_cast<double>(s) * resolution_));
      if (!inside(c))
        break;
      const std::size_t idx = indexOf(c);
      if (occupied_[idx])
        break;
      if (!free_[idx]) {
        free_[idx] = true;
        ++free_cells_;
      }
    }
  }

  build_seconds_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  valid_ = true;
  return true;
}

FreeSpaceGrid::Score
FreeSpaceGrid::score(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
                     const Eigen::Isometry3d &T_map_sensor) const {
  Score out;
  if (!valid_ || cloud_in_sensor.empty())
    return out;
  double occupied_hits = 0.0;
  double free_hits = 0.0;
  std::size_t scored = 0;
  for (const auto &pt : cloud_in_sensor.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3d point = T_map_sensor * Eigen::Vector3d(pt.x, pt.y, pt.z);
    if (!inside(cellOf(point)))
      continue;
    ++scored;
    // Hard cell lookup can flip a whole planar return at a submillimeter
    // boundary crossing. Interpolate the existing evidence, not the lock gates.
    const Eigen::Vector3d local = (point - origin_) / resolution_ -
                                  Eigen::Vector3d::Constant(0.5);
    const Eigen::Vector3i lower = local.array().floor().cast<int>();
    const Eigen::Vector3d fraction = local - lower.cast<double>();
    for (int x = 0; x < 2; ++x) {
      const double wx = x ? fraction.x() : 1.0 - fraction.x();
      for (int y = 0; y < 2; ++y) {
        const double wxy = wx * (y ? fraction.y() : 1.0 - fraction.y());
        for (int z = 0; z < 2; ++z) {
          const Eigen::Vector3i cell = lower + Eigen::Vector3i(x, y, z);
          if (!inside(cell)) continue;
          const double weight = wxy * (z ? fraction.z() : 1.0 - fraction.z());
          const std::size_t idx = indexOf(cell);
          if (occupied_[idx]) occupied_hits += weight;
          else if (free_[idx]) free_hits += weight;
        }
      }
    }
  }
  if (scored == 0)
    return out;
  const double n = static_cast<double>(scored);
  const double known = occupied_hits + free_hits;
  out.valid = true;
  out.scored_points = static_cast<int>(scored);
  out.known_points = known;
  out.occupied_points = occupied_hits;
  out.free_points = free_hits;
  out.occupied_fraction = occupied_hits / n;
  out.free_fraction = free_hits / n;
  out.known_fraction = known / n;
  if (known > 0) {
    out.occupied_given_known = occupied_hits / known;
    out.free_given_known = free_hits / known;
  }
  out.value = out.occupied_fraction - out.free_fraction;
  // Interpolation assigns mass to exclusive occupied/free/unknown labels.
  // Their second moment is p_o + p_f; retain one sample per query point.
  const double variance =
      (out.occupied_fraction + out.free_fraction -
       (out.occupied_fraction - out.free_fraction) *
           (out.occupied_fraction - out.free_fraction)) /
      n;
  out.stddev = std::sqrt(std::max(variance, 0.0));
  return out;
}

} // namespace n3mapping
