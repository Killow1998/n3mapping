#include "n3mapping/relocalization_debug_logger.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace n3mapping {

namespace {

std::string jsonEscape(const std::string &value) {
  std::ostringstream oss;
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      oss << "\\\"";
      break;
    case '\\':
      oss << "\\\\";
      break;
    case '\b':
      oss << "\\b";
      break;
    case '\f':
      oss << "\\f";
      break;
    case '\n':
      oss << "\\n";
      break;
    case '\r':
      oss << "\\r";
      break;
    case '\t':
      oss << "\\t";
      break;
    default:
      if (ch < 0x20) {
        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec << std::setfill(' ');
      } else {
        oss << static_cast<char>(ch);
      }
      break;
    }
  }
  return oss.str();
}

void appendComma(std::ostream &os, bool *first) {
  if (*first) {
    *first = false;
    return;
  }
  os << ',';
}

void appendString(std::ostream &os, bool *first, const char *key,
                  const std::string &value) {
  appendComma(os, first);
  os << '"' << key << "\":\"" << jsonEscape(value) << '"';
}

void appendBool(std::ostream &os, bool *first, const char *key, bool value) {
  appendComma(os, first);
  os << '"' << key << "\":" << (value ? "true" : "false");
}

void appendNumberValue(std::ostream &os, double value) {
  if (std::isfinite(value)) {
    os << std::setprecision(17) << value;
  } else {
    os << "null";
  }
}

void appendNumber(std::ostream &os, bool *first, const char *key,
                  double value) {
  appendComma(os, first);
  os << '"' << key << "\":";
  appendNumberValue(os, value);
}

void appendInteger(std::ostream &os, bool *first, const char *key,
                   int64_t value) {
  appendComma(os, first);
  os << '"' << key << "\":" << value;
}

void appendSize(std::ostream &os, bool *first, const char *key,
                std::size_t value) {
  appendComma(os, first);
  os << '"' << key << "\":" << value;
}

template <typename Derived>
void appendVector(std::ostream &os, bool *first, const char *key,
                  const Eigen::MatrixBase<Derived> &values) {
  appendComma(os, first);
  os << '"' << key << "\":[";
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    appendNumberValue(os, values(i));
  }
  os << ']';
}

template <typename Derived>
void appendMatrix(std::ostream &os, bool *first, const char *key,
                  const Eigen::MatrixBase<Derived> &matrix) {
  appendComma(os, first);
  os << '"' << key << "\":[";
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    if (row > 0) {
      os << ',';
    }
    os << '[';
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (col > 0) {
        os << ',';
      }
      appendNumberValue(os, matrix(row, col));
    }
    os << ']';
  }
  os << ']';
}

void appendRegistrationObservability(
    std::ostream &os, bool *first,
    const RegistrationObservability &observability) {
  appendComma(os, first);
  os << "\"registration_observability\":";
  if (!observability.available) {
    os << "null";
    return;
  }

  bool item_first = true;
  os << '{';
  appendString(os, &item_first, "information_layout",
               "translation_xyz_rotation_xyz");
  appendString(os, &item_first, "selected_pose_source",
               observability.selected_pose_source);
  appendBool(os, &item_first, "information_at_selected_pose",
             observability.information_at_selected_pose);
  appendBool(os, &item_first, "success", observability.success);
  appendBool(os, &item_first, "converged", observability.converged);
  appendBool(os, &item_first, "production_quality",
             observability.production_quality);
  appendString(os, &item_first, "termination",
               matchTerminationName(observability.termination));
  appendSize(os, &item_first, "iterations", observability.iterations);
  appendNumber(os, &item_first, "optimizer_error",
               observability.optimizer_error);
  appendNumber(os, &item_first, "residual_scale",
               observability.residual_scale);
  appendNumber(os, &item_first, "fitness_score",
               observability.fitness_score);
  appendSize(os, &item_first, "num_inliers", observability.num_inliers);
  appendNumber(os, &item_first, "inlier_ratio", observability.inlier_ratio);
  appendBool(os, &item_first, "information_finite",
             observability.information_finite);
  appendNumber(os, &item_first, "symmetry_max_abs",
               observability.symmetry_max_abs);
  appendBool(os, &item_first, "spectrum_valid",
             observability.spectrum_valid);
  appendBool(os, &item_first, "positive_definite",
             observability.positive_definite);
  appendInteger(os, &item_first, "observable_dofs_numerical",
                observability.numerical_rank);
  appendNumber(os, &item_first, "numerical_rank_tolerance",
               observability.numerical_rank_tolerance);
  appendNumber(os, &item_first, "condition_number",
               observability.condition_number);
  appendMatrix(os, &item_first, "full_information",
               observability.information);
  appendVector(os, &item_first, "full_eigenvalues_ascending",
               observability.information_eigenvalues);
  appendMatrix(os, &item_first, "full_eigenvectors_columns",
               observability.information_eigenvectors);
  appendVector(os, &item_first, "translation_block_eigenvalues_ascending",
               observability.translation_block_eigenvalues);
  appendVector(os, &item_first, "rotation_block_eigenvalues_ascending",
               observability.rotation_block_eigenvalues);
  appendBool(os, &item_first, "rotational_marginal_valid",
             observability.rotational_marginal_valid);
  appendMatrix(os, &item_first, "rotational_marginal_information",
               observability.rotational_marginal_information);
  appendVector(os, &item_first,
               "rotational_marginal_eigenvalues_ascending",
               observability.rotational_marginal_eigenvalues);
  appendMatrix(os, &item_first,
               "rotational_marginal_eigenvectors_columns",
               observability.rotational_marginal_eigenvectors);
  os << '}';
}

const char *candidateSourceName(LoopCandidate::Source source) {
  switch (source) {
  case LoopCandidate::Source::RhpdPrimary:
    return "rhpd_primary";
  case LoopCandidate::Source::ScanContextFallback:
    return "scan_context_fallback";
  case LoopCandidate::Source::RhpdFrame:
    return "rhpd_frame";
  case LoopCandidate::Source::SpatialRadius:
    return "spatial_radius";
  case LoopCandidate::Source::Unknown:
  default:
    return "unknown";
  }
}

void appendCandidateObject(std::ostream &os, const LoopCandidate &candidate) {
  bool first = true;
  os << '{';
  appendInteger(os, &first, "match_id", candidate.match_id);
  appendString(os, &first, "candidate_source",
               candidateSourceName(candidate.candidate_source));
  appendNumber(os, &first, "rhpd_distance", candidate.rhpd_distance);
  appendNumber(os, &first, "sc_distance", candidate.sc_distance);
  appendNumber(os, &first, "fused_score", candidate.fused_score);
  appendNumber(os, &first, "yaw_diff_rad",
               static_cast<double>(candidate.yaw_diff_rad));
  os << '}';
}

void appendCandidates(std::ostream &os, bool *first, const char *key,
                      const std::vector<LoopCandidate> &candidates) {
  appendComma(os, first);
  os << '"' << key << "\":[";
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    appendCandidateObject(os, candidates[i]);
  }
  os << ']';
}

void appendPose(std::ostream &os, bool *first, const char *key,
                const Eigen::Isometry3d &pose);

void appendBasins(std::ostream &os, bool *first,
                  const std::vector<RelocDebugBasinSummary> &basins) {
  appendComma(os, first);
  os << "\"basins\":[";
  for (std::size_t i = 0; i < basins.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    bool basin_first = true;
    os << '{';
    appendInteger(os, &basin_first, "center_match_id",
                  basins[i].center_match_id);
    appendSize(os, &basin_first, "member_count",
               basins[i].member_match_ids.size());
    appendComma(os, &basin_first);
    os << "\"member_match_ids\":[";
    for (std::size_t j = 0; j < basins[i].member_match_ids.size(); ++j) {
      if (j > 0) {
        os << ',';
      }
      os << basins[i].member_match_ids[j];
    }
    os << "]}";
  }
  os << ']';
}

void appendBasinBest(std::ostream &os, bool *first,
                     const std::vector<RelocDebugBasinBestSummary> &results) {
  appendComma(os, first);
  os << "\"per_basin_best\":[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    bool item_first = true;
    os << '{';
    appendInteger(os, &item_first, "basin_center_id",
                  results[i].basin_center_id);
    appendInteger(os, &item_first, "matched_kf_id", results[i].matched_kf_id);
    appendPose(os, &item_first, "pose_in_map", results[i].pose_in_map);
    appendComma(os, &item_first);
    os << "\"candidate\":";
    appendCandidateObject(os, results[i].candidate);
    appendNumber(os, &item_first, "fitness_score", results[i].fitness_score);
    appendNumber(os, &item_first, "inlier_ratio", results[i].inlier_ratio);
    appendNumber(os, &item_first, "selection_score",
                 results[i].selection_score);
    appendNumber(os, &item_first, "log_likelihood", results[i].log_likelihood);
    appendNumber(os, &item_first, "visibility_consistency_ratio",
                 results[i].visibility_consistency_ratio);
    appendNumber(os, &item_first, "visibility_observed_coverage",
                 results[i].visibility_observed_coverage);
    appendNumber(os, &item_first, "visibility_foreground_conflict_ratio",
                 results[i].visibility_foreground_conflict_ratio);
    appendNumber(os, &item_first, "visibility_evidence_log_odds",
                 results[i].visibility_evidence_log_odds);
    appendSize(os, &item_first, "visibility_known_bins",
               results[i].visibility_known_bins);
    appendSize(os, &item_first, "visibility_unknown_bins",
               results[i].visibility_unknown_bins);
    appendNumber(os, &item_first, "visibility_known_fraction",
                 results[i].visibility_known_fraction);
    appendNumber(os, &item_first, "visibility_consistent_given_known",
                 results[i].visibility_consistent_given_known);
    appendNumber(os, &item_first,
                 "visibility_foreground_conflict_given_known",
                 results[i].visibility_foreground_conflict_given_known);
    appendNumber(os, &item_first, "visibility_evidence_log_odds_given_known",
                 results[i].visibility_evidence_log_odds_given_known);
    appendRegistrationObservability(
        os, &item_first, results[i].registration_observability);
    os << '}';
  }
  os << ']';
}

void appendHypotheses(
    std::ostream &os, bool *first,
    const std::vector<RelocDebugHypothesisSummary> &hypotheses) {
  appendComma(os, first);
  os << "\"hypotheses\":[";
  for (std::size_t i = 0; i < hypotheses.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    bool item_first = true;
    os << '{';
    appendInteger(os, &item_first, "seed_match_id",
                  hypotheses[i].seed_match_id);
    appendInteger(os, &item_first, "last_match_id",
                  hypotheses[i].last_match_id);
    appendPose(os, &item_first, "pose_in_map", hypotheses[i].pose_in_map);
    appendNumber(os, &item_first, "cumulative_log_likelihood",
                 hypotheses[i].cumulative_log_likelihood);
    appendInteger(os, &item_first, "num_updates", hypotheses[i].num_updates);
    appendInteger(os, &item_first, "converged_updates",
                  hypotheses[i].converged_updates);
    appendInteger(os, &item_first, "visibility_updates",
                  hypotheses[i].visibility_updates);
    appendNumber(os, &item_first, "mean_visibility_consistency",
                 hypotheses[i].mean_visibility_consistency);
    appendNumber(os, &item_first, "mean_visibility_evidence",
                 hypotheses[i].mean_visibility_evidence);
    appendRegistrationObservability(
        os, &item_first, hypotheses[i].registration_observability);
    appendBool(os, &item_first, "alive", hypotheses[i].alive);
    os << '}';
  }
  os << ']';
}

void appendQueryCloudSummary(std::ostream &os, bool *first, const char *prefix,
                             const RelocQueryCloudDebugSummary &summary) {
  const std::string base(prefix);
  appendString(os, first, (base + "_mode").c_str(), summary.mode);
  appendInteger(os, first, (base + "_frame_count").c_str(),
                summary.frame_count);
  appendNumber(os, first, (base + "_motion_translation_m").c_str(),
               summary.motion_translation_m);
  appendNumber(os, first, (base + "_motion_rotation_rad").c_str(),
               summary.motion_rotation_rad);
  appendSize(os, first, (base + "_raw_points").c_str(), summary.raw_points);
  appendSize(os, first, (base + "_downsampled_points").c_str(),
             summary.downsampled_points);
  appendSize(os, first, (base + "_candidate_count").c_str(),
             summary.candidate_count);
  appendCandidates(os, first, (base + "_top_candidates").c_str(),
                   summary.top_candidates);
}

void appendPose(std::ostream &os, bool *first, const char *key,
                const Eigen::Isometry3d &pose) {
  appendComma(os, first);
  const Eigen::Quaterniond q(pose.rotation());
  os << '"' << key << "\":{";
  bool pose_first = true;
  appendNumber(os, &pose_first, "x", pose.translation().x());
  appendNumber(os, &pose_first, "y", pose.translation().y());
  appendNumber(os, &pose_first, "z", pose.translation().z());
  appendNumber(os, &pose_first, "qx", q.x());
  appendNumber(os, &pose_first, "qy", q.y());
  appendNumber(os, &pose_first, "qz", q.z());
  appendNumber(os, &pose_first, "qw", q.w());
  os << '}';
}

bool appendLine(const std::string &path, const std::string &line) {
  try {
    const std::filesystem::path fs_path(path);
    const auto parent = fs_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream file(path, std::ios::out | std::ios::app);
    if (!file.is_open()) {
      return false;
    }
    file << line << '\n';
    return file.good();
  } catch (const std::exception &) {
    return false;
  }
}

} // namespace

std::string RelocalizationDebugLogger::resolvePath(const Config &config) {
  if (!config.reloc_debug_path.empty()) {
    return config.reloc_debug_path;
  }
  if (config.map_save_path.empty()) {
    return "relocalization_debug.jsonl";
  }
  return (std::filesystem::path(config.map_save_path) /
          "relocalization_debug.jsonl")
      .string();
}

bool RelocalizationDebugLogger::appendRelocalization(
    const std::string &path, const RelocalizationDebugEvent &event) {
  std::ostringstream os;
  bool first = true;
  os << '{';
  appendString(os, &first, "record_type", "relocalize");
  appendNumber(os, &first, "processing_time", event.processing_time);
  appendNumber(os, &first, "query_timestamp", event.query_timestamp);
  appendSize(os, &first, "query_index", event.query_index);
  appendQueryCloudSummary(os, &first, "query", event.query_cloud);
  appendQueryCloudSummary(os, &first, "motion_query", event.motion_query_cloud);
  appendSize(os, &first, "candidate_count", event.candidate_count);
  appendCandidates(os, &first, "top_candidates", event.top_candidates);
  appendBasins(os, &first, event.basins);
  appendBasinBest(os, &first, event.basin_best_results);
  appendHypotheses(os, &first, event.hypotheses);
  appendInteger(os, &first, "winner_seed_match_id",
                event.winner_seed_match_id);
  appendInteger(os, &first, "winner_last_match_id",
                event.winner_last_match_id);
  appendInteger(os, &first, "runner_up_seed_match_id",
                event.runner_up_seed_match_id);
  appendInteger(os, &first, "runner_up_last_match_id",
                event.runner_up_last_match_id);
  appendNumber(os, &first, "temporal_hypothesis_score",
               event.temporal_hypothesis_score);
  appendNumber(os, &first, "log_likelihood", event.log_likelihood);
  appendInteger(os, &first, "winner_streak", event.winner_streak);
  appendNumber(os, &first, "winner_pose_translation_delta",
               event.winner_pose_translation_delta);
  appendNumber(os, &first, "winner_pose_rotation_delta",
               event.winner_pose_rotation_delta);
  appendNumber(os, &first, "evidence_motion_translation",
               event.evidence_motion_translation);
  appendNumber(os, &first, "evidence_motion_rotation",
               event.evidence_motion_rotation);
  appendBool(os, &first, "moving_visibility_required",
             event.moving_visibility_required);
  appendBool(os, &first, "moving_visibility_passed",
             event.moving_visibility_passed);
  appendNumber(os, &first, "margin", event.margin);
  appendNumber(os, &first, "ratio", event.ratio);
  appendNumber(os, &first, "visibility_margin", event.visibility_margin);
  appendNumber(os, &first, "visibility_ratio", event.visibility_ratio);
  appendNumber(os, &first, "basin_separation", event.basin_separation);
  appendBool(os, &first, "lock_accepted", event.lock_accepted);
  appendString(os, &first, "lock_result", event.lock_result);
  appendString(os, &first, "reject_reason", event.reject_reason);
  os << '}';
  return appendLine(path, os.str());
}

bool RelocalizationDebugLogger::appendTracking(
    const std::string &path, const RelocTrackingDebugEvent &event) {
  std::ostringstream os;
  bool first = true;
  os << '{';
  appendString(os, &first, "record_type", "tracking");
  appendNumber(os, &first, "processing_time", event.processing_time);
  appendSize(os, &first, "query_index", event.query_index);
  appendBool(os, &first, "strict_loaded_map", event.strict_loaded_map);
  appendPose(os, &first, "predicted_pose", event.predicted_pose);
  appendInteger(os, &first, "nearest_kf_id", event.nearest_kf_id);
  appendSize(os, &first, "submap_size", event.submap_size);
  appendNumber(os, &first, "tracking_total_ms", event.tracking_total_ms);
  appendNumber(os, &first, "nearest_keyframe_ms",
               event.nearest_keyframe_ms);
  appendNumber(os, &first, "loaded_map_cache_ms",
               event.loaded_map_cache_ms);
  appendNumber(os, &first, "submap_build_ms", event.submap_build_ms);
  appendNumber(os, &first, "target_prepare_ms", event.target_prepare_ms);
  appendBool(os, &first, "loaded_map_target_cache_hit",
             event.loaded_map_target_cache_hit);
  appendBool(os, &first, "loaded_map_target_cache_miss",
             event.loaded_map_target_cache_miss);
  appendNumber(os, &first, "source_prepare_ms", event.source_prepare_ms);
  appendNumber(os, &first, "registration_ms", event.registration_ms);
  appendNumber(os, &first, "retry_registration_ms",
               event.retry_registration_ms);
  appendNumber(os, &first, "visibility_ms", event.visibility_ms);
  appendBool(os, &first, "icp_converged", event.icp_converged);
  appendNumber(os, &first, "fitness_score", event.fitness_score);
  appendNumber(os, &first, "inlier_ratio", event.inlier_ratio);
  appendBool(os, &first, "retry_used", event.retry_used);
  appendInteger(os, &first, "consecutive_track_failures",
                event.consecutive_track_failures);
  appendBool(os, &first, "result_success", event.result_success);
  appendString(os, &first, "reject_reason", event.reject_reason);
  os << '}';
  return appendLine(path, os.str());
}

} // namespace n3mapping
