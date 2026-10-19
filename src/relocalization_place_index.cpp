#include "n3mapping/relocalization_place_index.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <glog/logging.h>

namespace n3mapping {
namespace {

bool sameGenerationAndStructure(const KeyframeMapRevision &lhs,
                                const KeyframeMapRevision &rhs) {
  return lhs.generation == rhs.generation &&
         lhs.structure_revision == rhs.structure_revision;
}

} // namespace

RelocalizationPlaceIndex::RelocalizationPlaceIndex(
    const Config &config, KeyframeManager &keyframe_manager,
    LoopDetector &loop_detector)
    : config_(config), keyframe_manager_(keyframe_manager),
      loop_detector_(loop_detector),
      frame_rhpd_manager_(
          loop_detector.getRHPDManager().getDescriptorParams()) {}

std::vector<LoopCandidate>
RelocalizationPlaceIndex::search(const PointCloudT::Ptr &cloud,
    DescriptorSearchPerformance *performance) {
  DescriptorPhaseTimer search_timer(performance ? &performance->total : nullptr);
  std::vector<LoopCandidate> candidates;
  if (!cloud || cloud->empty()) {
    return candidates;
  }

  DescriptorPhaseTimer generation_timer(performance ? &performance->generation : nullptr);
  if (performance) performance->input_points += cloud->size();
  const auto &rhpd_manager = loop_detector_.getRHPDManager();
  RHPDPlace query_rhpd;
  RHPDHeadingPtr query_sc;
  const bool use_rhpd = config_.rhpd_enabled && rhpd_manager.size() > 0;
  if (use_rhpd && config_.rhpd_use_sc_yaw) {
    auto features = loop_detector_.computeFeatures(cloud);
    query_rhpd = std::move(features.place);
    query_sc = std::move(features.heading);
  } else if (use_rhpd) {
    query_rhpd = loop_detector_.computeRHPD(cloud);
  } else {
    query_sc = loop_detector_.computeHeading(cloud);
  }

  generation_timer.finish();
  if (performance) {
    performance->place_values += query_rhpd.storedValues();
    performance->heading_values += query_sc ? query_sc->values.size() : 0;
  }

  if (use_rhpd) {
    if (query_rhpd.isZero()) {
      LOG(WARNING) << "[Reloc/RHPD] query descriptor is zero.";
    } else {
      candidates = searchRHPDCandidates(query_rhpd, query_sc, performance);
    }

    VLOG(1) << "[Reloc/RHPDPrimary] kept=" << candidates.size()
            << " dist_thr=" << config_.rhpd_dist_threshold;
    for (std::size_t i = 0; i < std::min<std::size_t>(candidates.size(), 5);
         ++i) {
      const auto &candidate = candidates[i];
      VLOG(1) << "  [rhpd " << i << "] kf=" << candidate.match_id
              << " rhpd=" << candidate.rhpd_distance
              << " sc=" << candidate.sc_distance
              << " yaw=" << candidate.yaw_diff_rad
              << " score=" << candidate.fused_score;
    }
    if (!candidates.empty()) {
      if (performance) performance->returned_candidates += candidates.size();
      return candidates;
    }
  }

  if (!query_sc) {
    return candidates;
  }

  std::vector<LoopCandidate> sc_ranked;
  auto descriptors = loop_detector_.getDescriptors();
  sc_ranked.reserve(descriptors.size());
  for (int i = 0; i < static_cast<int>(descriptors.size()); ++i) {
    const auto &[keyframe_id, descriptor] = descriptors[i];
    if (!descriptor) {
      continue;
    }
    DescriptorPhaseTimer heading_timer(performance ? &performance->heading : nullptr);
    if (performance) ++performance->heading_comparisons;
    auto [sc_dist, yaw_shift] =
        loop_detector_.computeDistance(query_sc, descriptor);
    heading_timer.finish();
    if (sc_dist > config_.reloc_sc_dist_threshold) {
      continue;
    }
    LoopCandidate candidate;
    candidate.query_id = -1;
    candidate.match_id = keyframe_id;
    candidate.sc_distance = sc_dist;
    candidate.yaw_diff_rad =
        static_cast<float>(yaw_shift) *
        static_cast<float>(loop_detector_.getHeadingSectorAngleDeg()) *
        static_cast<float>(M_PI / 180.0);
    candidate.source_flags = LoopCandidate::SOURCE_SC;
    candidate.candidate_source = LoopCandidate::Source::ScanContextFallback;
    candidate.sc_rank = i;
    candidate.fused_score = sc_dist;
    sc_ranked.push_back(candidate);
  }
  std::sort(sc_ranked.begin(), sc_ranked.end(),
            [](const LoopCandidate &lhs, const LoopCandidate &rhs) {
              return lhs.sc_distance < rhs.sc_distance;
            });
  const int keep = std::min(config_.reloc_num_candidates,
                            static_cast<int>(sc_ranked.size()));
  for (int i = 0; i < keep; ++i) {
    sc_ranked[i].fused_rank = i;
    candidates.push_back(sc_ranked[i]);
  }
  VLOG(1) << "[Reloc/SCFallback] kept=" << candidates.size()
          << " rhpd_enabled=" << config_.rhpd_enabled
          << " rhpd_db=" << rhpd_manager.size();
  if (performance) performance->returned_candidates += candidates.size();
  return candidates;
}

void RelocalizationPlaceIndex::rebuildFrameRHPDIndexIfNeeded(
    DescriptorSearchPerformance *performance) {
  const KeyframeMapRevision current_revision = keyframe_manager_.revision();
  const std::size_t current_size = keyframe_manager_.size();
  if (sameGenerationAndStructure(frame_rhpd_revision_, current_revision)) {
    return;
  }

  DescriptorPhaseTimer rebuild_timer(performance ? &performance->index_rebuild : nullptr);
  frame_rhpd_manager_.clear();
  std::size_t indexed = 0;
  for (const auto &keyframe : keyframe_manager_.getAllKeyframes()) {
    if (!keyframe || !keyframe->cloud || keyframe->cloud->empty()) {
      continue;
    }
    RHPDPlace descriptor = loop_detector_.computeRHPD(keyframe->cloud);
    if (descriptor.isZero()) {
      continue;
    }
    frame_rhpd_manager_.add(keyframe->id, descriptor);
    ++indexed;
  }
  if (performance) performance->indexed_keyframes += indexed;
  frame_rhpd_indexed_keyframes_ = current_size;
  frame_rhpd_revision_ = current_revision;
  VLOG(1) << "[Reloc/RHPDFrame] rebuilt frame-level index: indexed=" << indexed
          << " keyframes=" << current_size;
}

std::vector<LoopCandidate> RelocalizationPlaceIndex::searchRHPDCandidates(
    const RHPDPlace &query, const RHPDHeadingPtr &heading,
    DescriptorSearchPerformance *performance) {
  rebuildFrameRHPDIndexIfNeeded(performance);
  const auto &submaps = loop_detector_.getRHPDManager();
  // A coarse-key rank is not a lower bound on RHPD distance. In particular,
  // it must not discard a place before its discriminating heading is scored.
  // Merge both observation scales by their exact place distance; they share
  // the same keyframe heading and all subsequent quality gates.
  std::map<int64_t, LoopCandidate> places;
  const auto append = [&](const RHPDManager &database, LoopCandidate::Source source) {
    const auto ranked = database.search(query, database.size(), 0, performance);
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
      const auto [id, distance] = ranked[rank];
      if (distance > config_.rhpd_dist_threshold) continue;
      const auto existing = places.find(id);
      if (existing != places.end() && existing->second.rhpd_distance <= distance)
        continue;
      LoopCandidate candidate;
      candidate.query_id = -1;
      candidate.match_id = id;
      candidate.rhpd_distance = distance;
      candidate.source_flags = LoopCandidate::SOURCE_RHPD;
      candidate.candidate_source = source;
      candidate.rhpd_rank = rank;
      places[id] = candidate;
    }
  };
  append(submaps, LoopCandidate::Source::RhpdPrimary);
  append(frame_rhpd_manager_, LoopCandidate::Source::RhpdFrame);
  std::vector<LoopCandidate> ordered;
  ordered.reserve(places.size());
  for (const auto &[id, candidate] : places) ordered.push_back(candidate);
  std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
    if (a.rhpd_distance != b.rhpd_distance) return a.rhpd_distance < b.rhpd_distance;
    return a.match_id < b.match_id;
  });
  const auto better = [](const auto &a, const auto &b) {
    if (a.fused_score != b.fused_score) return a.fused_score < b.fused_score;
    if (a.rhpd_distance != b.rhpd_distance) return a.rhpd_distance < b.rhpd_distance;
    return a.match_id < b.match_id;
  };
  const auto limit = static_cast<std::size_t>(std::max(1, config_.reloc_num_candidates));
  std::vector<LoopCandidate> selected;
  selected.reserve(limit + 1);
  const double rounding_slack = 1e-12 *
      (1.0 + config_.sc_aux_weight / std::max(1e-6, config_.sc_aux_veto_threshold));
  for (auto candidate : ordered) {
    const double place_score = config_.rhpd_primary_weight * candidate.rhpd_distance /
                              std::max(1e-6, config_.rhpd_dist_threshold);
    // Heading distance and its configured weight are nonnegative. Once this
    // bound cannot beat the worst retained fused score, no later place can.
    // Keep equal scores eligible so distance/id tie-breaking remains exact.
    // Cosine distance can be a few ulps below zero at an exact match.
    if (selected.size() == limit &&
        place_score > selected.back().fused_score + rounding_slack) break;
    const auto reference = config_.rhpd_use_sc_yaw
                               ? loop_detector_.getDescriptor(candidate.match_id)
                               : RHPDHeadingPtr{};
    if (heading && reference) {
      DescriptorPhaseTimer timer(performance ? &performance->heading : nullptr);
      if (performance) ++performance->heading_comparisons;
      const auto [distance, shift] = loop_detector_.computeDistance(heading, reference);
      candidate.sc_distance = distance;
      candidate.yaw_diff_rad = static_cast<float>(shift) *
          static_cast<float>(loop_detector_.getHeadingSectorAngleDeg()) *
          static_cast<float>(M_PI / 180.0);
      candidate.source_flags |= LoopCandidate::SOURCE_SC;
      if (config_.sc_aux_veto_enabled && distance > config_.sc_aux_veto_threshold) continue;
    }
    const double heading_score = candidate.fromSC() && std::isfinite(candidate.sc_distance)
        ? config_.sc_aux_weight * candidate.sc_distance /
              std::max(1e-6, config_.sc_aux_veto_threshold) : 0.0;
    candidate.fused_score = place_score + heading_score;
    selected.insert(std::lower_bound(selected.begin(), selected.end(), candidate, better), candidate);
    if (selected.size() > limit) selected.pop_back();
  }
  for (std::size_t rank = 0; rank < selected.size(); ++rank) selected[rank].fused_rank = rank;
  return selected;
}

void RelocalizationPlaceIndex::resetMapDerivedState() {
  frame_rhpd_manager_.clear();
  frame_rhpd_indexed_keyframes_ = 0;
  frame_rhpd_revision_ = {};
}

std::size_t RelocalizationPlaceIndex::indexedKeyframes() const {
  return frame_rhpd_indexed_keyframes_;
}

} // namespace n3mapping
