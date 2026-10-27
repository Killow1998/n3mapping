#include "n3mapping/relocalization_place_index.h"

#include <cmath>
#include <map>
#include <tuple>

namespace n3mapping {
void RelocalizationPlaceIndex::prepare(
    const RHPDPlaceCache &cached_frames,
    DescriptorSearchPerformance *performance,
    const RelocalizationPlaceCatalog *cached_catalog) {
  if (!config_.rhpd_enabled)
    return;
  const auto revision = keyframe_manager_.revision();
  if (revision == view_revision_ && !places_.empty())
    return;
  DescriptorPhaseTimer timer(performance ? &performance->index_rebuild
                                         : nullptr);
  const auto keyframes = keyframe_manager_.getAllKeyframes();
  if (revision.generation != frame_rhpd_revision_.generation ||
      revision.structure_revision != frame_rhpd_revision_.structure_revision) {
    RHPDPlaceCache updated;
    for (const auto &keyframe : keyframes) {
      if (!keyframe || !keyframe->cloud || keyframe->cloud->empty())
        continue;
      const auto cached = cached_frames.find(keyframe->id);
      if (cached != cached_frames.end()) {
        updated.emplace(*cached);
        if (performance)
          ++performance->cached_frames;
      } else {
        // A structure revision can replace existing frames, not just append.
        auto place = std::make_shared<const RHPDPlace>(
            descriptor_.compute(keyframe->cloud));
        if (!place->isZero())
          updated.emplace(keyframe->id, std::move(place));
      }
    }
    frames_.swap(updated);
    frame_rhpd_revision_ = revision;
    frame_rhpd_indexed_keyframes_ = keyframes.size();
    if (performance)
      performance->indexed_keyframes += frames_.size();
  }
  const auto submaps = loop_detector_.getRHPDManager().snapshot();
  RHPDHeadingCache headings;
  for (const auto &[id, heading] : loop_detector_.getDescriptors())
    headings.emplace(id, heading);
  const auto catalog =
      cached_catalog ? *cached_catalog
                     : compileRelocalizationPlaceCatalog(
                           config_, keyframes, frames_, submaps, headings);
  places_.clear();
  retrieval_view_count_ = 0;
  for (const auto &members : catalog) {
    Place place;
    for (const auto id : members) {
      const auto frame = frames_.find(id);
      const auto submap = submaps.find(id);
      const auto heading = headings.find(id);
      place.views.push_back(
          {id, frame == frames_.end() ? nullptr : frame->second,
           submap == submaps.end() ? nullptr : submap->second,
           heading == headings.end() ? nullptr : heading->second});
    }
    retrieval_view_count_ += place.views.size();
    places_.push_back(std::move(place));
  }
  view_revision_ = revision;
}

} // namespace n3mapping
