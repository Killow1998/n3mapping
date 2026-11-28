#include "n3mapping/free_space_grid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <flann/algorithms/dist.h>
#include <flann/algorithms/kdtree_single_index.h>
#include <flann/util/result_set.h>

namespace n3mapping {
bool FreeSpaceGrid::Score::hasAbsoluteConflict() const {
  return valid && known_points >= 32.0 && known_fraction >= 0.20 &&
      free_given_known > 0.80;
}

bool FreeSpaceGrid::Score::significantlyBetterThan(const Score& other, double sigmas) const {
  return value - other.value > sigmas * std::hypot(stddev, other.stddev);
}

FreeSpaceGrid::Block& FreeSpaceGrid::addBlock(const BlockCoordinate& coordinate) {
  const auto [index, inserted] = block_index_.try_emplace(coordinate, blocks_.size());
  if (inserted) blocks_.push_back(Block{coordinate});
  return blocks_[index->second];
}

const char* FreeSpaceGrid::findBlock(const BlockCoordinate& coordinate) const {
  const auto index = block_index_.find(coordinate);
  if (index == block_index_.end()) return nullptr;
  return mapped_blocks_ ? mapped_blocks_ + index->second * sizeof(Block)
                        : reinterpret_cast<const char*>(&blocks_[index->second]);
}

bool FreeSpaceGrid::build(const pcl::PointCloud<pcl::PointXYZI> &map_cloud,
                          const std::vector<Eigen::Vector3d> &observation_origins,
                          double resolution, double max_ray_length,
                          int occupied_min_points) {
  *this = FreeSpaceGrid{};
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
    const double dimension = std::ceil(extent[axis] / resolution_) + 1;
    if (!std::isfinite(dimension) || dimension < 1 ||
        dimension > std::numeric_limits<int>::max())
      return false;
    dims_[axis] = static_cast<int>(dimension);
  }

  {
    std::unordered_map<BlockCoordinate, std::array<std::uint8_t, 512>, BlockHash> counts;
    const auto min_points = static_cast<std::uint8_t>(std::max(1, occupied_min_points));
    for (const auto &pt : map_cloud.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;
      const Eigen::Vector3i c = cellOf(Eigen::Vector3d(pt.x, pt.y, pt.z));
      if (!inside(c)) continue;
      const auto coordinate = blockOf(c);
      const int bit = bitOf(c);
      auto& count = counts[coordinate][bit];
      if (count < min_points && ++count == min_points) {
        addBlock(coordinate).occupied[bit / 8] |= std::uint8_t(1u << (bit % 8));
        ++occupied_cells_;
      }
    }
  }

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
    Block* block = nullptr;
    for (int s = 1; s < steps; ++s) {
      const Eigen::Vector3i c =
          cellOf(start + dir * (static_cast<double>(s) * resolution_));
      if (!inside(c))
        break;
      const auto coordinate = blockOf(c);
      if (!block || block->coordinate != coordinate) block = &addBlock(coordinate);
      const int bit = bitOf(c);
      const auto mask = std::uint8_t(1u << (bit % 8));
      if (block->occupied[bit / 8] & mask) break;
      auto& byte = block->free[bit / 8];
      if ((byte & mask) == 0) {
        byte |= mask;
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

bool FreeSpaceGrid::hasOccupiedBarrier(const Eigen::Vector3d& from,
                                       const Eigen::Vector3d& to,
                                       int min_occupied_voxels) const {
  constexpr double endpoint_margin = 0.4;
  const Eigen::Vector3d delta = to - from;
  const double length = delta.norm();
  if (!valid_ || length <= 2.0 * endpoint_margin) return false;

  const Eigen::Vector3d direction = delta / length;
  Eigen::Vector3i previous = Eigen::Vector3i::Constant(-1);
  int occupied = 0;
  for (double distance = endpoint_margin; distance <= length - endpoint_margin;
       distance += resolution_) {
    const Eigen::Vector3i cell = cellOf(from + direction * distance);
    if (!inside(cell) || cell == previous) continue;
    previous = cell;
    const auto* bits = findBlock(blockOf(cell));
    if (!bits) continue;
    const int bit = bitOf(cell);
    if ((bits[offsetof(Block, occupied) + bit / 8] & (1u << (bit % 8))) &&
        ++occupied >= min_occupied_voxels) return true;
  }
  return false;
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
          const auto* bits = findBlock(blockOf(cell));
          if (!bits) continue;
          const int bit = bitOf(cell);
          const auto mask = std::uint8_t(1u << (bit % 8));
          if (bits[offsetof(Block, occupied) + bit / 8] & mask) occupied_hits += weight;
          else if (bits[offsetof(Block, free) + bit / 8] & mask) free_hits += weight;
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

  const double variance =
      (out.occupied_fraction + out.free_fraction -
       (out.occupied_fraction - out.free_fraction) *
           (out.occupied_fraction - out.free_fraction)) /
      n;
  out.stddev = std::sqrt(std::max(variance, 0.0));
  return out;
}

}
