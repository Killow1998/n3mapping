#include "n3mapping/relocalization_place_index.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <glog/logging.h>

namespace n3mapping {
RelocalizationPlaceIndex::RelocalizationPlaceIndex(
    const Config &config, KeyframeManager &keyframe_manager,
    LoopDetector &loop_detector)
    : config_(config), keyframe_manager_(keyframe_manager),
      loop_detector_(loop_detector),
      descriptor_(loop_detector.getRHPDManager().getDescriptorParams()) {}

std::vector<LoopCandidate>
RelocalizationPlaceIndex::search(const PointCloudT::Ptr &cloud,
                                 DescriptorSearchPerformance *performance) {
  DescriptorPhaseTimer search_timer(performance ? &performance->total
                                                : nullptr);
  std::vector<LoopCandidate> candidates;
  if (!cloud || cloud->empty()) {
    return candidates;
  }

  DescriptorPhaseTimer generation_timer(performance ? &performance->generation
                                                    : nullptr);
  if (performance)
    performance->input_points += cloud->size();
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
      if (performance)
        performance->returned_candidates += candidates.size();
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
    DescriptorPhaseTimer heading_timer(performance ? &performance->heading
                                                   : nullptr);
    if (performance)
      ++performance->heading_comparisons;
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
  if (performance)
    performance->returned_candidates += candidates.size();
  return candidates;
}

std::vector<LoopCandidate> RelocalizationPlaceIndex::searchRHPDCandidates(
    const RHPDPlace &query, const RHPDHeadingPtr &heading,
    DescriptorSearchPerformance *performance) {
  prepare({}, performance);
  if (performance) {
    performance->retrieval_views += retrieval_view_count_;
    performance->retrieval_places += places_.size();
  }
  struct RankedView {
    LoopCandidate candidate;
    const View *reference;
  };
  struct RankedPlace {
    std::vector<RankedView> views;
  };
  std::vector<RankedPlace> ordered;
  ordered.reserve(places_.size());
  const auto prepared = descriptor_.prepareQuery(query);
  DescriptorPhaseTimer place_timer(performance ? &performance->place : nullptr);
  for (const auto &place : places_) {
    RankedPlace ranked;
    ranked.views.reserve(place.views.size());
    for (const auto &view : place.views) {
      LoopCandidate candidate;
      candidate.query_id = -1;
      candidate.match_id = view.id;
      candidate.source_flags = LoopCandidate::SOURCE_RHPD;
      candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
      candidate.rhpd_distance =
          view.submap ? descriptor_.distance(prepared, *view.submap)
                      : std::numeric_limits<double>::infinity();
      if (performance)
        performance->place_comparisons += bool(view.submap);
      if (view.frame) {
        const double distance = descriptor_.distance(prepared, *view.frame);
        if (performance)
          ++performance->place_comparisons;
        if (distance < candidate.rhpd_distance) {
          candidate.rhpd_distance = distance;
          candidate.candidate_source = LoopCandidate::Source::RhpdFrame;
        }
      }
      if (candidate.rhpd_distance <= config_.rhpd_dist_threshold)
        ranked.views.push_back({candidate, &view});
    }
    std::sort(ranked.views.begin(), ranked.views.end(),
              [](const auto &a, const auto &b) {
                if (a.candidate.rhpd_distance != b.candidate.rhpd_distance)
                  return a.candidate.rhpd_distance < b.candidate.rhpd_distance;
                return a.candidate.match_id < b.candidate.match_id;
              });
    if (!ranked.views.empty())
      ordered.push_back(std::move(ranked));
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
    const auto &x = a.views.front().candidate, &y = b.views.front().candidate;
    if (x.rhpd_distance != y.rhpd_distance)
      return x.rhpd_distance < y.rhpd_distance;
    return x.match_id < y.match_id;
  });
  place_timer.finish();
  const auto better = [](const auto &a, const auto &b) {
    if (a.fused_score != b.fused_score)
      return a.fused_score < b.fused_score;
    if (a.rhpd_distance != b.rhpd_distance)
      return a.rhpd_distance < b.rhpd_distance;
    return a.match_id < b.match_id;
  };
  const auto limit =
      static_cast<size_t>(std::max(1, config_.reloc_num_candidates));
  const double weight =
      config_.rhpd_primary_weight / std::max(1e-6, config_.rhpd_dist_threshold);
  const double slack =
      1e-12 * (1.0 + config_.sc_aux_weight /
                         std::max(1e-6, config_.sc_aux_veto_threshold));
  std::vector<LoopCandidate> selected;
  for (const auto &place : ordered) {
    if (selected.size() == limit &&
        weight * place.views.front().candidate.rhpd_distance >
            selected.back().fused_score + slack)
      break;
    LoopCandidate best;
    bool found = false;
    // Match the already compiled multi-view place; no spatial grouping or
    // representative selection changes the map index during a query.
    for (const auto &item : place.views) {
      auto candidate = item.candidate;
      const double place_score = weight * candidate.rhpd_distance;
      if (found && place_score > best.fused_score + slack)
        break;
      const auto &reference = item.reference->heading;
      if (heading && reference && config_.rhpd_use_sc_yaw) {
        DescriptorPhaseTimer timer(performance ? &performance->heading
                                               : nullptr);
        if (performance)
          ++performance->heading_comparisons;
        const auto [distance, shift] =
            loop_detector_.computeDistance(heading, reference);
        candidate.sc_distance = distance;
        candidate.yaw_diff_rad =
            static_cast<float>(shift) *
            static_cast<float>(loop_detector_.getHeadingSectorAngleDeg()) *
            static_cast<float>(M_PI / 180.0);
        candidate.source_flags |= LoopCandidate::SOURCE_SC;
        if (config_.sc_aux_veto_enabled &&
            distance > config_.sc_aux_veto_threshold)
          continue;
      }
      candidate.fused_score =
          place_score +
          (candidate.fromSC() && std::isfinite(candidate.sc_distance)
               ? config_.sc_aux_weight * candidate.sc_distance /
                     std::max(1e-6, config_.sc_aux_veto_threshold)
               : 0.0);
      if (!found || better(candidate, best)) {
        best = candidate;
        found = true;
      }
    }
    if (!found)
      continue;
    selected.insert(
        std::lower_bound(selected.begin(), selected.end(), best, better), best);
    if (selected.size() > limit)
      selected.pop_back();
  }
  for (size_t rank = 0; rank < selected.size(); ++rank)
    selected[rank].fused_rank = rank;
  return selected;
}

void RelocalizationPlaceIndex::resetMapDerivedState() {
  frames_.clear();
  places_.clear();
  retrieval_view_count_ = 0;
  view_revision_ = {};
  frame_rhpd_indexed_keyframes_ = 0;
  frame_rhpd_revision_ = {};
}

std::size_t RelocalizationPlaceIndex::indexedKeyframes() const {
  return frame_rhpd_indexed_keyframes_;
}

} // namespace n3mapping
