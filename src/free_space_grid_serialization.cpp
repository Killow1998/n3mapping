#include "n3mapping/free_space_grid.h"

#include <cstring>
#include <limits>
#include <ostream>

namespace n3mapping {
namespace {
struct FreeSpaceHeader {
  double resolution;
  std::array<double, 3> origin;
  std::array<std::uint32_t, 3> dimensions;
  std::uint32_t reserved = 0;
  std::uint64_t blocks;
};
static_assert(sizeof(FreeSpaceHeader) == 56);
}

bool FreeSpaceGrid::write(std::ostream& output) const {
  const FreeSpaceHeader header{
      resolution_, {origin_.x(), origin_.y(), origin_.z()},
      {std::uint32_t(dims_.x()), std::uint32_t(dims_.y()), std::uint32_t(dims_.z())},
      0, block_index_.size()};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  const char* bytes = mapped_blocks_ ? mapped_blocks_ : reinterpret_cast<const char*>(blocks_.data());
  if (header.blocks) output.write(bytes, header.blocks * sizeof(Block));
  return output.good();
}

bool FreeSpaceGrid::map(const char* bytes, std::size_t size, std::shared_ptr<const void> owner) {
  if (size < sizeof(FreeSpaceHeader)) return false;
  FreeSpaceHeader header;
  std::memcpy(&header, bytes, sizeof(header));
  const auto payload = size - sizeof(header);
  if (!std::isfinite(header.resolution) || header.resolution <= 0 ||
      header.reserved || payload % sizeof(Block) || header.blocks != payload / sizeof(Block))
    return false;

  FreeSpaceGrid grid;
  grid.resolution_ = header.resolution;
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(header.origin[axis]) || !header.dimensions[axis] ||
        header.dimensions[axis] > std::uint32_t(std::numeric_limits<int>::max()))
      return false;
    grid.dims_[axis] = static_cast<int>(header.dimensions[axis]);
    grid.origin_[axis] = header.origin[axis];
  }
  grid.block_index_.reserve(header.blocks);
  grid.mapped_blocks_ = bytes + sizeof(header);
  for (std::size_t index = 0; index < header.blocks; ++index) {
    Block block;
    std::memcpy(&block, grid.mapped_blocks_ + index * sizeof(Block), sizeof(Block));
    int extent[3];
    for (int axis = 0; axis < 3; ++axis) {
      if (block.coordinate[axis] > (header.dimensions[axis] - 1) / 8) return false;
      extent[axis] = std::min(8u, header.dimensions[axis] - block.coordinate[axis] * 8);
    }
    if (block.reserved || !grid.block_index_.try_emplace(block.coordinate, index).second)
      return false;
    const unsigned z_mask = (1u << extent[2]) - 1;
    bool nonempty = false;
    for (int offset = 0; offset < 64; ++offset) {
      const auto occupied = block.occupied[offset];
      const auto free = block.free[offset];
      const auto bits = occupied | free;
      const auto allowed = offset / 8 < extent[0] && offset % 8 < extent[1] ? z_mask : 0u;
      if ((occupied & free) || (bits & ~allowed)) return false;
      nonempty |= bits != 0;
      grid.occupied_cells_ += __builtin_popcount(occupied);
      grid.free_cells_ += __builtin_popcount(free);
    }
    if (!nonempty) return false;
  }
  grid.mapped_owner_ = std::move(owner);
  grid.valid_ = true;
  *this = std::move(grid);
  return true;
}
}
