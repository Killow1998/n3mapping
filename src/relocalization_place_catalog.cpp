#include "n3mapping/relocalization_place_catalog.h"
#include "n3mapping/config.h"
#include <algorithm>
#include <cmath>
#include <tuple>
namespace n3mapping {
namespace {
bool sameCells(const RHPDCells &a, const RHPDCells &b) {
  return a.indices == b.indices && a.values == b.values;
}
bool samePlace(const std::shared_ptr<const RHPDPlace> &a,
               const std::shared_ptr<const RHPDPlace> &b) {
  if (a == b)
    return true;
  if (!a || !b || a->auxiliary != b->auxiliary ||
      !sameCells(a->ring_height, b->ring_height))
    return false;
  for (size_t p = 0; p < a->planes.size(); ++p)
    if (!sameCells(a->planes[p], b->planes[p]))
      return false;
  return true;
}
} // namespace

RelocalizationPlaceCatalog compileRelocalizationPlaceCatalog(
    const Config &config, const std::vector<Keyframe::Ptr> &keyframes,
    const RHPDPlaceCache &frames, const RHPDPlaceCache &submaps,
    const RHPDHeadingCache &headings) {
  using Cell = std::tuple<int, int, int>;
  std::map<Cell, std::vector<Keyframe::Ptr>> representatives;
  std::map<int64_t, std::vector<int64_t>> place_views;
  // Preserve separately observed headings and floors. The representative is
  // only a retrieval identity; all views retain their own pose and descriptor.
  constexpr double viewpoint_radius = 0.25;
  const double angle =
      (360.0 / std::max(4, config.sc_num_sectors)) * M_PI / 180.0;
  for (const auto &keyframe : keyframes) {
    if (!keyframe)
      continue;
    const auto frame = frames.find(keyframe->id);
    const auto submap = submaps.find(keyframe->id);
    if (frame == frames.end() && submap == submaps.end())
      continue;
    const auto position = keyframe->getPosition();
    const Eigen::Vector3i cell =
        (position / viewpoint_radius).array().floor().cast<int>();
    int64_t owner = keyframe->id;
    for (int x = -1; x <= 1; ++x)
      for (int y = -1; y <= 1; ++y)
        for (int z = -1; z <= 1; ++z) {
          auto found =
              representatives.find({cell.x() + x, cell.y() + y, cell.z() + z});
          if (found == representatives.end())
            continue;
          for (const auto &reference : found->second) {
            if ((position - reference->getPosition()).squaredNorm() <=
                    viewpoint_radius * viewpoint_radius &&
                Eigen::AngleAxisd(reference->getRotation().transpose() *
                                  keyframe->getRotation())
                        .angle() <= angle)
              owner = std::min(owner, reference->id);
          }
        }
    if (owner == keyframe->id)
      representatives[{cell.x(), cell.y(), cell.z()}].push_back(keyframe);
    // Equal features always have equal query scores. Keep the lowest-ID view,
    // exactly as the ranking tie-break would; retain every differing view.
    auto &members = place_views[owner];
    const auto duplicate =
        std::find_if(members.begin(), members.end(), [&](int64_t id) {
          const auto old_frame = frames.find(id), old_submap = submaps.find(id);
          const auto old_heading = headings.find(id),
                     new_heading = headings.find(keyframe->id);
          const auto a =
              old_heading == headings.end() ? nullptr : old_heading->second;
          const auto b =
              new_heading == headings.end() ? nullptr : new_heading->second;
          const bool same_heading =
              a == b ||
              (a && b && a->rings == b->rings && a->sectors == b->sectors &&
               a->offsets == b->offsets && a->ring_indices == b->ring_indices &&
               a->values == b->values && a->column_norms == b->column_norms &&
               a->sector_key == b->sector_key && a->ring_key == b->ring_key);
          return same_heading &&
                 samePlace(frame == frames.end() ? nullptr : frame->second,
                           old_frame == frames.end() ? nullptr
                                                     : old_frame->second) &&
                 samePlace(submap == submaps.end() ? nullptr : submap->second,
                           old_submap == submaps.end() ? nullptr
                                                       : old_submap->second);
        });
    if (duplicate == members.end())
      members.push_back(keyframe->id);
  }
  RelocalizationPlaceCatalog result;
  result.reserve(place_views.size());
  for (auto &[id, members] : place_views)
    result.push_back(std::move(members));
  return result;
}

} // namespace n3mapping
