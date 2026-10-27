#include "n3mapping/free_space_grid.h"
#include "localization_atlas.pb.h"

#include <limits>

namespace n3mapping {
void FreeSpaceGrid::encode(LocalizationAtlasFreeSpaceGrid& output) const {
  output.Clear();
  if (!valid_) return;
  output.set_resolution(resolution_);
  for (int axis = 0; axis < 3; ++axis) {
    output.add_origin(origin_[axis]);
    output.add_dimensions(dims_[axis]);
  }
  std::string occupied((occupied_.size() + 7) / 8, '\0');
  std::string free(occupied.size(), '\0');
  for (std::size_t i = 0; i < occupied_.size(); ++i) {
    if (occupied_[i]) occupied[i / 8] |= static_cast<char>(1u << (i % 8));
    if (free_[i]) free[i / 8] |= static_cast<char>(1u << (i % 8));
  }
  output.set_occupied(std::move(occupied));
  output.set_free(std::move(free));
}

bool FreeSpaceGrid::decode(const LocalizationAtlasFreeSpaceGrid& input) {
  *this = FreeSpaceGrid{};
  if (!std::isfinite(input.resolution()) || input.resolution() <= 0 ||
      input.origin_size() != 3 || input.dimensions_size() != 3)
    return false;
  std::size_t cells = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const auto dimension = input.dimensions(axis);
    if (!std::isfinite(input.origin(axis)) || dimension == 0 ||
        dimension > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        cells > std::numeric_limits<std::size_t>::max() / dimension)
      return false;
    cells *= dimension;
    dims_[axis] = static_cast<int>(dimension);
    origin_[axis] = input.origin(axis);
  }
  const auto bytes = cells / 8 + (cells % 8 != 0);
  if (input.occupied().size() != bytes || input.free().size() != bytes)
    return false;
  for (std::size_t i = 0; i < bytes; ++i)
    if ((input.occupied()[i] & input.free()[i]) != 0) return false;
  if (cells % 8 != 0 &&
      ((static_cast<unsigned char>(input.occupied().back()) |
        static_cast<unsigned char>(input.free().back())) >> (cells % 8)) != 0)
    return false;
  resolution_ = input.resolution();
  occupied_.resize(cells);
  free_.resize(cells);
  for (std::size_t i = 0; i < cells; ++i) {
    const unsigned mask = 1u << (i % 8);
    occupied_[i] = (static_cast<unsigned char>(input.occupied()[i / 8]) & mask) != 0;
    free_[i] = (static_cast<unsigned char>(input.free()[i / 8]) & mask) != 0;
    occupied_cells_ += occupied_[i];
    free_cells_ += free_[i];
  }
  valid_ = true;
  return true;
}
} // namespace n3mapping
