// free_space_grid.cpp
#include "n3mapping/free_space_grid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/kdtree/kdtree_flann.h>

namespace n3mapping {

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
  pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  origin_cloud->points.reserve(observation_origins.size());
  for (const auto &o : observation_origins) {
    origin_cloud->points.emplace_back(static_cast<float>(o.x()),
                                      static_cast<float>(o.y()),
                                      static_cast<float>(o.z()));
  }
  origin_cloud->width = origin_cloud->points.size();
  origin_cloud->height = 1;
  pcl::KdTreeFLANN<pcl::PointXYZ> origin_tree;
  origin_tree.setInputCloud(origin_cloud);

  free_.assign(cells, false);
  std::vector<int> nn_index(1);
  std::vector<float> nn_dist_sq(1);
  for (const auto &pt : map_cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3d end(pt.x, pt.y, pt.z);
    pcl::PointXYZ probe(pt.x, pt.y, pt.z);
    if (origin_tree.nearestKSearch(probe, 1, nn_index, nn_dist_sq) < 1)
      continue;
    if (nn_dist_sq[0] > max_ray_length * max_ray_length)
      continue;
    const Eigen::Vector3d start = observation_origins[nn_index[0]];
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
  std::size_t occupied_hits = 0;
  std::size_t free_hits = 0;
  std::size_t scored = 0;
  for (const auto &pt : cloud_in_sensor.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3i c =
        cellOf(T_map_sensor * Eigen::Vector3d(pt.x, pt.y, pt.z));
    if (!inside(c))
      continue;
    const std::size_t idx = indexOf(c);
    ++scored;
    if (occupied_[idx])
      ++occupied_hits;
    else if (free_[idx])
      ++free_hits;
  }
  if (scored == 0)
    return out;
  const double n = static_cast<double>(scored);
  const std::size_t known = occupied_hits + free_hits;
  out.valid = true;
  out.scored_points = static_cast<int>(scored);
  out.known_points = static_cast<int>(known);
  out.occupied_points = static_cast<int>(occupied_hits);
  out.free_points = static_cast<int>(free_hits);
  out.occupied_fraction = static_cast<double>(occupied_hits) / n;
  out.free_fraction = static_cast<double>(free_hits) / n;
  out.known_fraction = static_cast<double>(known) / n;
  if (known > 0) {
    out.occupied_given_known = static_cast<double>(occupied_hits) /
                               static_cast<double>(known);
    out.free_given_known = static_cast<double>(free_hits) /
                           static_cast<double>(known);
  }
  out.value = out.occupied_fraction - out.free_fraction;
  // Occupied and free are mutually exclusive, so the per-point variable takes
  // +1/-1/0 and its iid variance is E[X^2] - E[X]^2 with E[X^2] = p_o + p_f.
  // The previous expression treated the two as independent Bernoullis and
  // understated the variance by 2*p_o*p_f/n.
  const double variance =
      (out.occupied_fraction + out.free_fraction -
       (out.occupied_fraction - out.free_fraction) *
           (out.occupied_fraction - out.free_fraction)) /
      n;
  out.stddev = std::sqrt(std::max(variance, 0.0));
  return out;
}

} // namespace n3mapping
