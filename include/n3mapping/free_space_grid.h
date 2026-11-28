#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {
class FreeSpaceGrid {
public:
  struct Score {
    bool valid = false;
    double occupied_fraction = 0.0;
    double free_fraction = 0.0;
    int scored_points = 0;

    double known_points = 0.0;
    double occupied_points = 0.0;
    double free_points = 0.0;
    double known_fraction = 0.0;
    double occupied_given_known = 0.0;
    double free_given_known = 0.0;

    double value = 0.0;
    double stddev = 0.0;
    bool significantlyBetterThan(const Score& other, double sigmas) const;
    bool hasAbsoluteConflict() const;
  };

  bool build(const pcl::PointCloud<pcl::PointXYZI> &map_cloud,
             const std::vector<Eigen::Vector3d> &observation_origins,
             double resolution, double max_ray_length, int occupied_min_points);

  bool valid() const { return valid_; }
  double resolution() const { return resolution_; }
  std::size_t occupiedCells() const { return occupied_cells_; }
  std::size_t freeCells() const { return free_cells_; }
  double buildSeconds() const { return build_seconds_; }

  bool write(std::ostream& output) const;
  bool map(const char* bytes, std::size_t size, std::shared_ptr<const void> owner);

  Score score(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
              const Eigen::Isometry3d &T_map_sensor) const;
  bool hasOccupiedBarrier(const Eigen::Vector3d& from,
                          const Eigen::Vector3d& to, int min_occupied_voxels = 2) const;

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
  using BlockCoordinate = std::array<std::uint32_t, 3>;
  struct BlockHash {
    std::size_t operator()(const BlockCoordinate& coordinate) const {
      return (std::size_t(coordinate[0]) * 73856093) ^
             (std::size_t(coordinate[1]) * 19349663) ^
             (std::size_t(coordinate[2]) * 83492791);
    }
  };
  struct Block {
    BlockCoordinate coordinate{};
    std::uint32_t reserved = 0;
    std::array<std::uint8_t, 64> occupied{};
    std::array<std::uint8_t, 64> free{};
  };
  static_assert(sizeof(Block) == 144);

  static BlockCoordinate blockOf(const Eigen::Vector3i& cell) {
    return {std::uint32_t(cell.x() >> 3), std::uint32_t(cell.y() >> 3),
            std::uint32_t(cell.z() >> 3)};
  }
  static int bitOf(const Eigen::Vector3i& cell) {
    return (cell.x() & 7) * 64 + (cell.y() & 7) * 8 + (cell.z() & 7);
  }
  Block& addBlock(const BlockCoordinate& coordinate);
  const char* findBlock(const BlockCoordinate& coordinate) const;

  bool valid_ = false;
  double resolution_ = 0.0;
  double build_seconds_ = 0.0;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  Eigen::Vector3i dims_ = Eigen::Vector3i::Zero();
  std::unordered_map<BlockCoordinate, std::size_t, BlockHash> block_index_;
  std::vector<Block> blocks_;
  std::shared_ptr<const void> mapped_owner_;
  const char* mapped_blocks_ = nullptr;
  std::size_t occupied_cells_ = 0;
  std::size_t free_cells_ = 0;
};

}
