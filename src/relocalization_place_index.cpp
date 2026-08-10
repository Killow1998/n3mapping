#include "n3mapping/relocalization_place_index.h"

#include <algorithm>
#include <cmath>

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
RelocalizationPlaceIndex::search(const PointCloudT::Ptr &cloud) {
  std::vector<LoopCandidate> candidates;
  if (!cloud || cloud->empty()) {
    return candidates;
  }

  const auto &rhpd_manager = loop_detector_.getRHPDManager();
  Eigen::VectorXd query_rhpd;
  Eigen::MatrixXd query_sc;
  if (config_.rhpd_use_sc_yaw || !config_.rhpd_enabled ||
      rhpd_manager.size() == 0) {
    query_sc = loop_detector_.makeScanContext(cloud);
  }

  if (config_.rhpd_enabled && rhpd_manager.size() > 0) {
    query_rhpd = loop_detector_.computeRHPD(cloud);
    if (query_rhpd.size() != RHPD_DIM || query_rhpd.isZero()) {
      LOG(WARNING) << "[Reloc/RHPD] query descriptor is zero.";
    } else {
      const int preselect = std::max(config_.rhpd_num_candidates * 3,
                                     config_.reloc_num_candidates * 3);
      auto top_k = rhpd_manager.search(query_rhpd, std::max(1, preselect),
                                       config_.rhpd_preselect_candidates);
      std::vector<LoopCandidate> ranked;
      ranked.reserve(top_k.size());

      for (int i = 0; i < static_cast<int>(top_k.size()); ++i) {
        const auto &item = top_k[i];
        if (item.second > config_.rhpd_dist_threshold) {
          continue;
        }

        LoopCandidate candidate;
        candidate.query_id = -1;
        candidate.match_id = item.first;
        candidate.rhpd_distance = item.second;
        candidate.source_flags = LoopCandidate::SOURCE_RHPD;
        candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
        candidate.rhpd_rank = i;

        Eigen::MatrixXd match_sc = loop_detector_.getDescriptor(item.first);
        if (config_.rhpd_use_sc_yaw && query_sc.size() > 0 &&
            match_sc.size() > 0) {
          auto [sc_dist, yaw_shift] =
              loop_detector_.computeDistance(query_sc, match_sc);
          candidate.sc_distance = sc_dist;
          candidate.yaw_diff_rad =
              static_cast<float>(yaw_shift) *
              static_cast<float>(
                  loop_detector_.getScanContextSectorAngleDeg()) *
              static_cast<float>(M_PI / 180.0);
          candidate.source_flags |= LoopCandidate::SOURCE_SC;
          if (config_.sc_aux_veto_enabled &&
              sc_dist > config_.sc_aux_veto_threshold) {
            continue;
          }
        }

        const double rhpd_norm = candidate.rhpd_distance /
                                 std::max(1e-6, config_.rhpd_dist_threshold);
        const double sc_norm =
            std::isfinite(candidate.sc_distance)
                ? candidate.sc_distance /
                      std::max(1e-6, config_.sc_aux_veto_threshold)
                : 1.0;
        candidate.fused_score = config_.rhpd_primary_weight * rhpd_norm +
                                config_.sc_aux_weight * sc_norm;
        ranked.push_back(candidate);
      }

      std::sort(ranked.begin(), ranked.end(),
                [](const LoopCandidate &lhs, const LoopCandidate &rhs) {
                  if (lhs.fused_score != rhs.fused_score) {
                    return lhs.fused_score < rhs.fused_score;
                  }
                  return lhs.rhpd_distance < rhs.rhpd_distance;
                });

      const int keep = std::min(config_.reloc_num_candidates,
                                static_cast<int>(ranked.size()));
      candidates.reserve(keep);
      for (int i = 0; i < keep; ++i) {
        ranked[i].fused_rank = i;
        candidates.push_back(ranked[i]);
      }

      appendFrameRHPDCandidates(query_rhpd, query_sc, candidates);
      std::sort(candidates.begin(), candidates.end(),
                [](const LoopCandidate &lhs, const LoopCandidate &rhs) {
                  if (lhs.fused_score != rhs.fused_score) {
                    return lhs.fused_score < rhs.fused_score;
                  }
                  return lhs.rhpd_distance < rhs.rhpd_distance;
                });
      if (static_cast<int>(candidates.size()) > config_.reloc_num_candidates) {
        candidates.resize(std::max(1, config_.reloc_num_candidates));
      }
      for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        candidates[i].fused_rank = i;
      }
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
      return candidates;
    }
  }

  if (query_sc.size() == 0) {
    return candidates;
  }

  std::vector<LoopCandidate> sc_ranked;
  auto descriptors = loop_detector_.getDescriptors();
  sc_ranked.reserve(descriptors.size());
  for (int i = 0; i < static_cast<int>(descriptors.size()); ++i) {
    const auto &[keyframe_id, descriptor] = descriptors[i];
    if (descriptor.size() == 0) {
      continue;
    }
    auto [sc_dist, yaw_shift] =
        loop_detector_.computeDistance(query_sc, descriptor);
    if (sc_dist > config_.reloc_sc_dist_threshold) {
      continue;
    }
    LoopCandidate candidate;
    candidate.query_id = -1;
    candidate.match_id = keyframe_id;
    candidate.sc_distance = sc_dist;
    candidate.yaw_diff_rad =
        static_cast<float>(yaw_shift) *
        static_cast<float>(loop_detector_.getScanContextSectorAngleDeg()) *
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
  return candidates;
}

void RelocalizationPlaceIndex::rebuildFrameRHPDIndexIfNeeded() {
  const KeyframeMapRevision current_revision = keyframe_manager_.revision();
  const std::size_t current_size = keyframe_manager_.size();
  if (sameGenerationAndStructure(frame_rhpd_revision_, current_revision)) {
    return;
  }

  frame_rhpd_manager_.clear();
  std::size_t indexed = 0;
  for (const auto &keyframe : keyframe_manager_.getAllKeyframes()) {
    if (!keyframe || !keyframe->cloud || keyframe->cloud->empty()) {
      continue;
    }
    Eigen::VectorXd descriptor = loop_detector_.computeRHPD(keyframe->cloud);
    if (descriptor.size() != RHPD_DIM || descriptor.isZero()) {
      continue;
    }
    frame_rhpd_manager_.add(keyframe->id, descriptor);
    ++indexed;
  }
  frame_rhpd_indexed_keyframes_ = current_size;
  frame_rhpd_revision_ = current_revision;
  VLOG(1) << "[Reloc/RHPDFrame] rebuilt frame-level index: indexed=" << indexed
          << " keyframes=" << current_size;
}

void RelocalizationPlaceIndex::appendFrameRHPDCandidates(
    const Eigen::VectorXd &query_rhpd, const Eigen::MatrixXd &query_sc,
    std::vector<LoopCandidate> &candidates) {
  if (query_rhpd.size() != RHPD_DIM || query_rhpd.isZero()) {
    return;
  }

  rebuildFrameRHPDIndexIfNeeded();
  if (frame_rhpd_manager_.size() == 0) {
    return;
  }

  const int top_k = std::max(config_.reloc_num_candidates * 2,
                             config_.rhpd_num_candidates * 2);
  const int preselect = std::max(config_.rhpd_preselect_candidates, top_k * 5);
  auto frame_top =
      frame_rhpd_manager_.search(query_rhpd, std::max(1, top_k), preselect);
  for (int i = 0; i < static_cast<int>(frame_top.size()); ++i) {
    const auto &item = frame_top[i];
    if (item.second > config_.rhpd_dist_threshold) {
      continue;
    }

    LoopCandidate candidate;
    candidate.query_id = -1;
    candidate.match_id = item.first;
    candidate.rhpd_distance = item.second;
    candidate.source_flags = LoopCandidate::SOURCE_RHPD;
    candidate.candidate_source = LoopCandidate::Source::RhpdFrame;
    candidate.rhpd_rank = i;

    Eigen::MatrixXd match_sc = loop_detector_.getDescriptor(item.first);
    if (config_.rhpd_use_sc_yaw && query_sc.size() > 0 && match_sc.size() > 0) {
      auto [sc_dist, yaw_shift] =
          loop_detector_.computeDistance(query_sc, match_sc);
      candidate.sc_distance = sc_dist;
      candidate.yaw_diff_rad =
          static_cast<float>(yaw_shift) *
          static_cast<float>(loop_detector_.getScanContextSectorAngleDeg()) *
          static_cast<float>(M_PI / 180.0);
      candidate.source_flags |= LoopCandidate::SOURCE_SC;
      if (config_.sc_aux_veto_enabled &&
          sc_dist > config_.sc_aux_veto_threshold) {
        continue;
      }
    }

    const double rhpd_norm =
        candidate.rhpd_distance / std::max(1e-6, config_.rhpd_dist_threshold);
    const double sc_norm =
        std::isfinite(candidate.sc_distance)
            ? candidate.sc_distance /
                  std::max(1e-6, config_.sc_aux_veto_threshold)
            : 1.0;
    candidate.fused_score = config_.rhpd_primary_weight * rhpd_norm +
                            config_.sc_aux_weight * sc_norm;

    auto existing = std::find_if(candidates.begin(), candidates.end(),
                                 [&](const LoopCandidate &old) {
                                   return old.match_id == candidate.match_id;
                                 });
    if (existing == candidates.end()) {
      candidates.push_back(candidate);
    } else if (candidate.fused_score < existing->fused_score ||
               (candidate.fused_score == existing->fused_score &&
                candidate.rhpd_distance < existing->rhpd_distance)) {
      *existing = candidate;
    }
  }
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
