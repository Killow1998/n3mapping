// free_space_grid.h - 3D occupancy and free-space grid for relocalization.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

// A 3D occupancy and free-space grid, built the way Cartographer builds one:
// every cell a mapping ray traversed before its first return is free, and a
// return landing there contradicts the map.
//
// Why 3D. A 2D projection of the same idea was measured on today20 and left one
// wrong lock 10 m from truth: collapsing height merges a ray that passed
// through a column at one height with structure present at another, so the true
// pose itself scored as contradicted. Scoring in 3D removed that failure and
// took the worst wrong lock from 33.1 m to 1.1 m.
//
// Resolution is not free to choose. At 0.10 m the map point cloud does not fill
// its own surfaces, the occupied hit rate collapses from 20-95% to 2-4%, and
// the score stops discriminating. 0.15 m, 0.20 m and 0.25 m produced
// case-for-case identical results, so the working range is a plateau rather
// than a tuned point; 0.30 m reintroduced a wrong lock.
class FreeSpaceGrid {
public:
  struct Score {
    bool valid = false;
    double occupied_fraction = 0.0;
    double free_fraction = 0.0;
    int scored_points = 0;
    // occupied_fraction - free_fraction, with the binomial sampling standard
    // deviation of that difference. The deviation is what makes a survival test
    // possible without inventing a threshold: two hypotheses are only separated
    // when they differ by more than the noise of the measurement itself.
    double value = 0.0;
    double stddev = 0.0;
  };

  // Builds from the assembled map cloud plus the positions the map was observed
  // from. Every map point is an occupied return; the ray from its nearest
  // observation position up to that return marks free cells, stopping at the
  // first occupied cell so a surface never has free space behind it. Map points
  // farther than max_ray_length from any observation position contribute
  // occupancy but cast no ray. Returns false if there is nothing to build from.
  bool build(const pcl::PointCloud<pcl::PointXYZI> &map_cloud,
             const std::vector<Eigen::Vector3d> &observation_origins,
             double resolution, double max_ray_length, int occupied_min_points);

  bool valid() const { return valid_; }
  double resolution() const { return resolution_; }
  std::size_t occupiedCells() const { return occupied_cells_; }
  std::size_t freeCells() const { return free_cells_; }
  double buildSeconds() const { return build_seconds_; }

  // Scores a sensor-frame cloud placed at T_map_sensor.
  Score score(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
              const Eigen::Isometry3d &T_map_sensor) const;

private:
  bool inside(const Eigen::Vector3i &c) const {
    return c.x() >= 0 && c.y() >= 0 && c.z() >= 0 && c.x() < dims_.x() &&
           c.y() < dims_.y() && c.z() < dims_.z();
  }
  Eigen::Vector3i cellOf(const Eigen::Vector3d &p) const {
    const Eigen::Vector3d local = (p - origin_) / resolution_;
    return Eigen::Vector3i(static_cast<int>(std::floor(local.x())),
                           static_cast<int>(std::floor(local.y())),
                           static_cast<int>(std::floor(local.z())));
  }
  std::size_t indexOf(const Eigen::Vector3i &c) const {
    return (static_cast<std::size_t>(c.x()) *
                static_cast<std::size_t>(dims_.y()) +
            static_cast<std::size_t>(c.y())) *
               static_cast<std::size_t>(dims_.z()) +
           static_cast<std::size_t>(c.z());
  }

  bool valid_ = false;
  double resolution_ = 0.0;
  double build_seconds_ = 0.0;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  Eigen::Vector3i dims_ = Eigen::Vector3i::Zero();
  std::vector<bool> occupied_;
  std::vector<bool> free_;
  std::size_t occupied_cells_ = 0;
  std::size_t free_cells_ = 0;
};

} // namespace n3mapping
