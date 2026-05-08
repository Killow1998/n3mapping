// WorldLocalizing: global relocalization via RHPD + ICP, and tracking localization with T_map_odom.
#include "n3mapping/world_localizing.h"

#include <glog/logging.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/memory.h>

namespace n3mapping {
namespace {
constexpr int kRelocMaxBasinCount = 3;
constexpr int kRelocPerBasinVerifyCount = 3;
constexpr double kRelocBasinAssignRadiusXY = 4.0;
constexpr double kRelocBasinAmbiguousFitnessGap = 0.03;
} // namespace

WorldLocalizing::WorldLocalizing(const Config& config,
                                 KeyframeManager& keyframe_manager,
                                 LoopDetector& loop_detector,
                                 PointCloudMatcher& matcher)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , loop_detector_(loop_detector)
    , matcher_(matcher)
    , is_relocalized_(false)
    , T_map_odom_(Eigen::Isometry3d::Identity())
    , last_matched_id_(-1)
    , last_odom_pose_(Eigen::Isometry3d::Identity())
    , consecutive_track_failures_(0)
    , hypothesis_window_count_(0)
    , last_window_winner_seed_id_(-1)
    , winner_streak_(0) {}

RelocResult WorldLocalizing::relocalize(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose) {
    RelocResult result;
    result.success = false;

    if (!cloud || cloud->empty()) {
        LOG(WARNING) << "Empty point cloud for relocalization.";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (keyframe_manager_.size() == 0) {
        LOG(WARNING) << "No keyframes available for relocalization.";
        return result;
    }

    PointCloudT::Ptr query_cloud = buildRelocQueryCloud(cloud, odom_pose);
    if (!query_cloud || query_cloud->empty()) {
        LOG(WARNING) << "Relocalization query cloud is empty after aggregation.";
        return result;
    }

    if (pending_hypotheses_.empty()) {
        auto candidates = searchCandidates(query_cloud);
        if (candidates.empty()) {
            LOG(WARNING) << "No relocalization candidates found. Map has " << keyframe_manager_.size()
                         << " keyframes, RHPD candidates=" << config_.rhpd_num_candidates
                         << " dist_thr=" << config_.rhpd_dist_threshold;
            return result;
        }

        struct BasinGroup {
            int64_t center_match_id = -1;
            Eigen::Vector3d center_pos = Eigen::Vector3d::Zero();
            std::vector<LoopCandidate> members;
        };
        std::vector<BasinGroup> basins;
        basins.reserve(kRelocMaxBasinCount);

        // Candidates are already sorted by coarse descriptor distance.
        // Build up to top-K basins in XY, then verify per-basin with local-submap ICP.
        for (const auto& candidate : candidates) {
            auto kf = keyframe_manager_.getKeyframe(candidate.match_id);
            if (!kf) continue;
            const Eigen::Vector3d pos = kf->pose_optimized.translation();

            int best_basin = -1;
            double best_dist = std::numeric_limits<double>::max();
            for (size_t i = 0; i < basins.size(); ++i) {
                const double d = (basins[i].center_pos.head<2>() - pos.head<2>()).norm();
                if (d < best_dist) {
                    best_dist = d;
                    best_basin = static_cast<int>(i);
                }
            }

            if (best_basin >= 0 && best_dist <= kRelocBasinAssignRadiusXY) {
                basins[best_basin].members.push_back(candidate);
            } else if (static_cast<int>(basins.size()) < kRelocMaxBasinCount) {
                BasinGroup bg;
                bg.center_match_id = candidate.match_id;
                bg.center_pos = pos;
                bg.members.push_back(candidate);
                basins.push_back(std::move(bg));
            }
        }

        struct BasinBest {
            int64_t basin_center_id = -1;
            int64_t matched_kf_id = -1;
            LoopCandidate candidate;
            MatchResult match;
        };
        std::vector<BasinBest> basin_best_results;
        basin_best_results.reserve(basins.size());

        for (const auto& basin : basins) {
            BasinBest best;
            best.basin_center_id = basin.center_match_id;
            best.match.fitness_score = std::numeric_limits<double>::max();
            int verified = 0;

            for (const auto& candidate : basin.members) {
                if (verified >= kRelocPerBasinVerifyCount) break;
                ++verified;

                MatchResult match_result;
                int64_t matched_kf_id = -1;
                if (!evaluateSingleCandidate(query_cloud, candidate, match_result, matched_kf_id)) continue;

                const double scale = config_.gicp_fitness_threshold * 0.5;
                const double confidence = std::exp(-match_result.fitness_score / std::max(1e-6, scale));
                const bool valid = match_result.converged &&
                                   match_result.fitness_score < config_.gicp_fitness_threshold &&
                                   match_result.inlier_ratio >= config_.reloc_min_inlier_ratio &&
                                   confidence >= config_.reloc_min_confidence;
                if (!valid) continue;

                if (match_result.fitness_score < best.match.fitness_score) {
                    best.candidate = candidate;
                    best.match = match_result;
                    best.matched_kf_id = matched_kf_id;
                }
            }

            if (best.matched_kf_id >= 0) {
                basin_best_results.push_back(best);
            }
        }

        std::sort(basin_best_results.begin(), basin_best_results.end(),
                  [](const BasinBest& a, const BasinBest& b) {
                      return a.match.fitness_score < b.match.fitness_score;
                  });

        if (basin_best_results.size() >= 2) {
            const auto& b1 = basin_best_results[0];
            const auto& b2 = basin_best_results[1];
            auto kf1 = keyframe_manager_.getKeyframe(b1.matched_kf_id);
            auto kf2 = keyframe_manager_.getKeyframe(b2.matched_kf_id);
            const double basin_sep = (kf1 && kf2)
                ? (kf1->pose_optimized.translation().head<2>() - kf2->pose_optimized.translation().head<2>()).norm()
                : 0.0;
            const double fitness_gap = std::abs(b1.match.fitness_score - b2.match.fitness_score);

            if (basin_sep >= config_.reloc_ambiguity_min_basin_separation &&
                fitness_gap <= kRelocBasinAmbiguousFitnessGap) {
                LOG(WARNING) << "AMBIGUOUS_BASIN_REJECT init: top1_basin=" << b1.basin_center_id
                             << " top1_kf=" << b1.matched_kf_id
                             << " top1_fit=" << b1.match.fitness_score
                             << " top2_basin=" << b2.basin_center_id
                             << " top2_kf=" << b2.matched_kf_id
                             << " top2_fit=" << b2.match.fitness_score
                             << " basin_sep=" << basin_sep
                             << " fitness_gap=" << fitness_gap;
                return result;
            }
        }

        LOG(INFO) << "Relocalization basin init: coarse_candidates=" << candidates.size()
                  << " basins=" << basins.size()
                  << " verified_basins=" << basin_best_results.size();

        for (const auto& bb : basin_best_results) {
            RelocHypothesis hyp;
            hyp.seed_match_id = bb.basin_center_id;
            hyp.last_match_id = bb.matched_kf_id;
            hyp.T_map_odom = bb.match.T_target_source * odom_pose.inverse();
            hyp.cumulative_log_likelihood = computeRelocLogLikelihood(bb.candidate, bb.match);
            hyp.num_updates = 1;
            hyp.converged_updates = 1;
            hyp.alive = true;
            pending_hypotheses_.push_back(hyp);
        }
        hypothesis_window_count_ = 1;

        if (pending_hypotheses_.empty()) {
            LOG(WARNING) << "Relocalization init failed: no valid ICP hypothesis.";
            return result;
        }
    } else {
        hypothesis_window_count_++;
        for (auto& hyp : pending_hypotheses_) {
            if (!hyp.alive) continue;

            Eigen::Isometry3d predicted_pose = hyp.T_map_odom * odom_pose;
            int64_t nearest_kf_id = findNearestKeyframe(predicted_pose);
            if (nearest_kf_id < 0) {
                hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
                continue;
            }

            auto submap = keyframe_manager_.buildLocalSubmap(nearest_kf_id, config_.gicp_submap_size);
            if (!submap || submap->empty()) {
                hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
                continue;
            }

            MatchResult mr = matcher_.alignCloud(submap, query_cloud, predicted_pose);
            hyp.cumulative_log_likelihood += computeTrackLogLikelihood(mr, predicted_pose);
            hyp.num_updates += 1;
            if (mr.converged) {
                hyp.T_map_odom = mr.T_target_source * odom_pose.inverse();
                hyp.last_match_id = nearest_kf_id;
                hyp.converged_updates += 1;
            }
        }
    }

    std::vector<const RelocHypothesis*> ranked_hypotheses;
    ranked_hypotheses.reserve(pending_hypotheses_.size());
    for (const auto& hyp : pending_hypotheses_) {
        if (hyp.alive) ranked_hypotheses.push_back(&hyp);
    }
    std::sort(ranked_hypotheses.begin(), ranked_hypotheses.end(), [](const RelocHypothesis* a, const RelocHypothesis* b) {
        return a->cumulative_log_likelihood > b->cumulative_log_likelihood;
    });

    if (!ranked_hypotheses.empty()) {
        const RelocHypothesis* top1 = ranked_hypotheses[0];
        const RelocHypothesis* top2 = ranked_hypotheses.size() > 1 ? ranked_hypotheses[1] : nullptr;
        const double top1_ll = top1->cumulative_log_likelihood;
        const double top2_ll = top2 ? top2->cumulative_log_likelihood : -std::numeric_limits<double>::infinity();
        const double margin = top2 ? (top1_ll - top2_ll) : std::numeric_limits<double>::infinity();

        if (top1->seed_match_id == last_window_winner_seed_id_) {
            winner_streak_ += 1;
        } else {
            last_window_winner_seed_id_ = top1->seed_match_id;
            winner_streak_ = 1;
        }

        LOG(INFO) << "[Reloc/Stability] window=" << hypothesis_window_count_
                  << "/" << config_.reloc_temporal_window_size
                  << " top1(seed=" << top1->seed_match_id << ",last_kf=" << top1->last_match_id
                  << ",ll=" << top1_ll << ",conv_updates=" << top1->converged_updates
                  << ",updates=" << top1->num_updates << ")"
                  << " top2(seed=" << (top2 ? top2->seed_match_id : -1)
                  << ",last_kf=" << (top2 ? top2->last_match_id : -1)
                  << ",ll=" << (top2 ? top2_ll : -1e9) << ")"
                  << " margin=" << margin
                  << " winner_streak=" << winner_streak_;
    }

    const int effective_temporal_window = std::max(1, config_.reloc_temporal_window_size);
    if (hypothesis_window_count_ < effective_temporal_window) {
        LOG(INFO) << "Relocalization pending: window " << hypothesis_window_count_
                  << "/" << effective_temporal_window
                  << ", active hypotheses=" << pending_hypotheses_.size();
        return result;
    }

    const RelocHypothesis* top1 = ranked_hypotheses.empty() ? nullptr : ranked_hypotheses[0];
    const RelocHypothesis* top2 = ranked_hypotheses.size() > 1 ? ranked_hypotheses[1] : nullptr;
    const double top1_ll = top1 ? top1->cumulative_log_likelihood : -std::numeric_limits<double>::infinity();
    const double top2_ll = top2 ? top2->cumulative_log_likelihood : -std::numeric_limits<double>::infinity();
    const double margin = top2 ? (top1_ll - top2_ll) : std::numeric_limits<double>::infinity();
    const int effective_min_winner_streak = std::min(
        std::max(1, config_.reloc_lock_min_winner_streak), effective_temporal_window);
    const int effective_min_converged_updates = std::min(
        std::max(1, config_.reloc_lock_min_converged_updates), effective_temporal_window);
    const double effective_min_margin = std::max(0.0, config_.reloc_lock_min_margin);

    const bool pass_loglik = top1 && (top1_ll >= config_.reloc_lock_log_likelihood_threshold);
    const bool pass_margin = top1 && (margin >= effective_min_margin);
    const bool pass_winner_streak = top1 && (winner_streak_ >= effective_min_winner_streak);
    const bool pass_converged_updates = top1 && (top1->converged_updates >= effective_min_converged_updates);
    const bool top2_viable = top2 &&
                             (top2_ll >= config_.reloc_lock_log_likelihood_threshold) &&
                             (top2->converged_updates >= effective_min_converged_updates);
    const double ratio = (top2 && top2_ll > 1e-6) ? (top1_ll / top2_ll) : std::numeric_limits<double>::infinity();
    double basin_separation = 0.0;
    bool basin_separated = false;
    if (top1 && top2) {
        auto kf1 = keyframe_manager_.getKeyframe(top1->last_match_id);
        auto kf2 = keyframe_manager_.getKeyframe(top2->last_match_id);
        if (kf1 && kf2) {
            basin_separation = (kf1->pose_optimized.translation() - kf2->pose_optimized.translation()).norm();
            basin_separated = basin_separation >= config_.reloc_ambiguity_min_basin_separation;
        }
    }
    const bool ambiguous = top2_viable &&
                           basin_separated &&
                           (margin < config_.reloc_ambiguity_min_margin) &&
                           (ratio < config_.reloc_ambiguity_min_ratio);

    if (!ambiguous && pass_loglik && pass_margin && pass_winner_streak && pass_converged_updates) {
        const RelocHypothesis& best = *top1;
        T_map_odom_ = best.T_map_odom;
        is_relocalized_ = true;
        last_matched_id_ = best.last_match_id;
        last_odom_pose_ = odom_pose;
        consecutive_track_failures_ = 0;

        result.success = true;
        result.matched_keyframe_id = best.last_match_id;
        result.pose_in_map = T_map_odom_ * odom_pose;
        result.confidence = std::max(0.0, std::min(1.0, best.cumulative_log_likelihood / 10.0));
        result.fitness_score = 0.0;

        LOG(INFO) << "Relocalization locked after temporal window. KF=" << result.matched_keyframe_id
                  << ", cumulative_loglik=" << best.cumulative_log_likelihood
                  << ", guard(loglik=" << pass_loglik
                  << ", margin=" << pass_margin
                  << ", winner_streak=" << pass_winner_streak
                  << ", converged_updates=" << pass_converged_updates
                  << ", ambiguous=" << ambiguous << ")"
                  << ", margin_value=" << margin
                  << ", winner_streak_value=" << winner_streak_
                  << ", converged_updates_value=" << best.converged_updates
                  << ", ratio_value=" << ratio
                  << ", basin_separation=" << basin_separation;
    } else {
        if (ambiguous) {
            LOG(WARNING) << "AMBIGUOUS_REJECT relocalization lock: top1_seed=" << (top1 ? top1->seed_match_id : -1)
                         << " top1_kf=" << (top1 ? top1->last_match_id : -1)
                         << " top1_ll=" << top1_ll
                         << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
                         << " top2_kf=" << (top2 ? top2->last_match_id : -1)
                         << " top2_ll=" << top2_ll
                         << " margin=" << margin
                         << " ratio=" << ratio
                         << " basin_separation=" << basin_separation
                         << " top2_viable=" << top2_viable
                         << " thresholds(margin>=" << config_.reloc_ambiguity_min_margin
                         << ", ratio>=" << config_.reloc_ambiguity_min_ratio
                         << ", basin_sep>=" << config_.reloc_ambiguity_min_basin_separation << ")";
        } else {
            LOG(WARNING) << "Relocalization rejected after temporal window by stability guard."
                         << " top1_seed=" << (top1 ? top1->seed_match_id : -1)
                         << " top1_kf=" << (top1 ? top1->last_match_id : -1)
                         << " top1_ll=" << top1_ll
                         << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
                         << " top2_ll=" << top2_ll
                         << " margin=" << margin
                         << " ratio=" << ratio
                         << " basin_separation=" << basin_separation
                         << " winner_streak=" << winner_streak_
                         << " top1_converged_updates=" << (top1 ? top1->converged_updates : 0)
                         << " require(loglik>=" << config_.reloc_lock_log_likelihood_threshold
                         << ", margin>=" << effective_min_margin
                         << ", winner_streak>=" << effective_min_winner_streak
                         << ", converged_updates>=" << effective_min_converged_updates << ")"
                         << " pass(loglik=" << pass_loglik
                         << ", margin=" << pass_margin
                         << ", winner_streak=" << pass_winner_streak
                         << ", converged_updates=" << pass_converged_updates << ")";
        }
    }

    clearRelocHypotheses();

    return result;
}

RelocResult WorldLocalizing::trackLocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose) {
    Eigen::Isometry3d predicted_pose = T_map_odom_ * odom_pose;

    RelocResult result;
    result.success = false;
    result.pose_in_map = predicted_pose;

    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
        is_relocalized_ = false;
        return result;
    }

    if (!cloud || cloud->empty()) {
        LOG(WARNING) << "Empty point cloud for tracking.";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    int64_t nearest_kf_id = findNearestKeyframe(predicted_pose);
    if (nearest_kf_id < 0) {
        consecutive_track_failures_++;
        if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
            is_relocalized_ = false;
            result.success = false;
        } else {
            result.success = true;
        }
        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.5;
        last_odom_pose_ = odom_pose;
        return result;
    }

    // Build submap — use larger range when tracking is unstable
    int submap_range = config_.gicp_submap_size;
    if (consecutive_track_failures_ > 0)
        submap_range = std::max(submap_range, config_.reloc_track_unstable_submap_size);

    auto submap = keyframe_manager_.buildLocalSubmap(nearest_kf_id, submap_range);
    if (!submap || submap->empty()) {
        consecutive_track_failures_++;
        if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
            is_relocalized_ = false;
            result.success = false;
        } else {
            result.success = true;
        }
        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.5;
        last_odom_pose_ = odom_pose;
        return result;
    }

    MatchResult match_result = matcher_.alignCloud(submap, cloud, predicted_pose);

    // If ICP failed with standard params and we have recent failures, retry with wider search
    const bool icp_failed = !match_result.converged ||
                            match_result.fitness_score >= config_.gicp_fitness_threshold ||
                            match_result.inlier_ratio < config_.reloc_min_inlier_ratio;
    if (icp_failed && consecutive_track_failures_ < config_.reloc_track_retry_max_failures) {
        // Save and widen correspondence distance
        auto saved = matcher_.getSettings();
        auto wide = saved;
        wide.max_correspondence_distance = saved.max_correspondence_distance * config_.reloc_track_retry_corr_scale;
        wide.max_iterations = config_.reloc_track_retry_max_iterations;
        matcher_.setSettings(wide);
        auto retry = matcher_.alignCloud(submap, cloud, predicted_pose);
        matcher_.setSettings(saved);
        if (retry.converged && retry.fitness_score < match_result.fitness_score)
            match_result = retry;
    }

    double scale = config_.gicp_fitness_threshold * 0.5;
    double current_confidence = std::exp(-match_result.fitness_score / scale);
    current_confidence = std::max(0.0, std::min(1.0, current_confidence));

    const bool icp_ok = match_result.converged &&
                        match_result.fitness_score < config_.gicp_fitness_threshold &&
                        match_result.inlier_ratio >= config_.reloc_min_inlier_ratio;

    double delta_translation = std::numeric_limits<double>::infinity();
    if (match_result.converged) {
        const Eigen::Isometry3d delta = predicted_pose.inverse() * match_result.T_target_source;
        delta_translation = delta.translation().norm();
    }

    if (icp_ok) {
        Eigen::Isometry3d T_map_odom_icp = match_result.T_target_source * odom_pose.inverse();

        double alpha;
        if (delta_translation <= 0.5) {
            alpha = std::min(current_confidence * 0.3, 0.2);
        } else if (delta_translation <= 2.0) {
            alpha = std::min(current_confidence * 0.15, 0.1);
        } else {
            alpha = std::min(current_confidence * 0.08, 0.05);
        }
        alpha = std::max(alpha, 0.01);

        Eigen::Vector3d t_current = T_map_odom_.translation();
        Eigen::Vector3d t_target = T_map_odom_icp.translation();
        Eigen::Vector3d t_new = t_current + alpha * (t_target - t_current);

        double alpha_z = alpha * 0.3;
        t_new.z() = t_current.z() + alpha_z * (t_target.z() - t_current.z());

        Eigen::Quaterniond q_current(T_map_odom_.rotation());
        Eigen::Quaterniond q_target(T_map_odom_icp.rotation());
        Eigen::Quaterniond q_new = q_current.slerp(alpha, q_target);

        T_map_odom_.translation() = t_new;
        T_map_odom_.linear() = q_new.toRotationMatrix();

        result.success = true;
        result.matched_keyframe_id = nearest_kf_id;
        result.pose_in_map = T_map_odom_ * odom_pose;
        result.fitness_score = match_result.fitness_score;
        result.confidence = current_confidence;

        last_matched_id_ = nearest_kf_id;
        last_odom_pose_ = odom_pose;
        consecutive_track_failures_ = 0;

        VLOG(1) << "Tracking OK: fitness=" << match_result.fitness_score
                << " conf=" << current_confidence << " alpha=" << alpha
                << " delta_t=" << delta_translation;
    } else {
        consecutive_track_failures_++;
        if (consecutive_track_failures_ <= 3 || consecutive_track_failures_ % 5 == 0) {
            LOG(WARNING) << "Tracking failed (x" << consecutive_track_failures_
                         << "): converged=" << match_result.converged
                         << " fitness=" << match_result.fitness_score
                         << " inlier=" << match_result.inlier_ratio
                         << " nearest_kf=" << nearest_kf_id
                         << " submap_pts=" << (submap ? submap->size() : 0)
                         << " cloud_pts=" << cloud->size()
                         << " pos=(" << predicted_pose.translation().x()
                         << "," << predicted_pose.translation().y()
                         << "," << predicted_pose.translation().z() << ")";
        }

        if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
            is_relocalized_ = false;
            result.success = false;
        } else {
            // Keep odometry-based continuity for transient dropouts.
            result.success = true;
        }

        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.2;
        last_odom_pose_ = odom_pose;
    }

    return result;
}

bool WorldLocalizing::isRelocalized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_relocalized_;
}

Eigen::Isometry3d WorldLocalizing::getMapToOdomTransform() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return T_map_odom_;
}

void WorldLocalizing::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_relocalized_ = false;
    T_map_odom_ = Eigen::Isometry3d::Identity();
    last_matched_id_ = -1;
    last_odom_pose_ = Eigen::Isometry3d::Identity();
    consecutive_track_failures_ = 0;
    query_frame_buffer_.clear();
    clearRelocHypotheses();
}

void WorldLocalizing::setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom) {
    std::lock_guard<std::mutex> lock(mutex_);
    T_map_odom_ = T_map_odom;
    is_relocalized_ = true;
    query_frame_buffer_.clear();
    clearRelocHypotheses();
}

int64_t WorldLocalizing::getLastMatchedKeyframeId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_matched_id_;
}

std::vector<LoopCandidate> WorldLocalizing::searchCandidates(const PointCloudT::Ptr& cloud) {
    std::vector<LoopCandidate> candidates;
    if (!cloud || cloud->empty()) return candidates;

    const auto& rhpd_mgr = loop_detector_.getRHPDManager();
    Eigen::MatrixXd query_sc;
    if (config_.rhpd_use_sc_yaw || !config_.rhpd_enabled || rhpd_mgr.size() == 0) {
        query_sc = loop_detector_.makeScanContext(cloud);
    }

    if (config_.rhpd_enabled && rhpd_mgr.size() > 0) {
        Eigen::VectorXd query_rhpd = loop_detector_.computeRHPD(cloud);
        if (query_rhpd.size() != RHPD_DIM || query_rhpd.isZero()) {
            LOG(WARNING) << "[Reloc/RHPD] query descriptor is zero.";
        } else {
            const int preselect = std::max(config_.rhpd_num_candidates * 3, config_.reloc_num_candidates * 3);
            auto top_k = rhpd_mgr.search(query_rhpd, std::max(1, preselect), config_.rhpd_preselect_candidates);
            std::vector<LoopCandidate> ranked;
            ranked.reserve(top_k.size());

            for (int i = 0; i < static_cast<int>(top_k.size()); ++i) {
                const auto& item = top_k[i];
                if (item.second > config_.rhpd_dist_threshold) continue;

                LoopCandidate c;
                c.query_id = -1;
                c.match_id = item.first;
                c.rhpd_distance = item.second;
                c.source_flags = LoopCandidate::SOURCE_RHPD;
                c.candidate_source = LoopCandidate::Source::RhpdPrimary;
                c.rhpd_rank = i;

                Eigen::MatrixXd match_sc = loop_detector_.getDescriptor(item.first);
                if (config_.rhpd_use_sc_yaw && query_sc.size() > 0 && match_sc.size() > 0) {
                    auto [sc_dist, yaw_shift] = loop_detector_.computeDistance(query_sc, match_sc);
                    c.sc_distance = sc_dist;
                    c.yaw_diff_rad = static_cast<float>(yaw_shift) *
                        static_cast<float>(loop_detector_.getScanContextSectorAngleDeg()) *
                        static_cast<float>(M_PI / 180.0);
                    c.source_flags |= LoopCandidate::SOURCE_SC;
                    if (config_.sc_aux_veto_enabled && sc_dist > config_.sc_aux_veto_threshold) {
                        continue;
                    }
                }

                const double rhpd_norm = c.rhpd_distance / std::max(1e-6, config_.rhpd_dist_threshold);
                const double sc_norm = std::isfinite(c.sc_distance)
                    ? c.sc_distance / std::max(1e-6, config_.sc_aux_veto_threshold)
                    : 1.0;
                c.fused_score = config_.rhpd_primary_weight * rhpd_norm + config_.sc_aux_weight * sc_norm;
                ranked.push_back(c);
            }

            std::sort(ranked.begin(), ranked.end(), [](const LoopCandidate& a, const LoopCandidate& b) {
                if (a.fused_score != b.fused_score) return a.fused_score < b.fused_score;
                return a.rhpd_distance < b.rhpd_distance;
            });

            const int keep = std::min(config_.reloc_num_candidates, static_cast<int>(ranked.size()));
            candidates.reserve(keep);
            for (int i = 0; i < keep; ++i) {
                ranked[i].fused_rank = i;
                candidates.push_back(ranked[i]);
            }
        }

        LOG(INFO) << "[Reloc/RHPDPrimary] kept=" << candidates.size()
                  << " dist_thr=" << config_.rhpd_dist_threshold;
        for (size_t i = 0; i < std::min<size_t>(candidates.size(), 5); ++i) {
            const auto& c = candidates[i];
            LOG(INFO) << "  [rhpd " << i << "] kf=" << c.match_id
                      << " rhpd=" << c.rhpd_distance
                      << " sc=" << c.sc_distance
                      << " yaw=" << c.yaw_diff_rad
                      << " score=" << c.fused_score;
        }
        if (!candidates.empty()) return candidates;
    }

    if (query_sc.size() == 0) return candidates;

    std::vector<LoopCandidate> sc_ranked;
    auto descriptors = loop_detector_.getDescriptors();
    sc_ranked.reserve(descriptors.size());
    for (int i = 0; i < static_cast<int>(descriptors.size()); ++i) {
        const auto& [kf_id, desc] = descriptors[i];
        if (desc.size() == 0) continue;
        auto [sc_dist, yaw_shift] = loop_detector_.computeDistance(query_sc, desc);
        if (sc_dist > config_.reloc_sc_dist_threshold) continue;
        LoopCandidate c;
        c.query_id = -1;
        c.match_id = kf_id;
        c.sc_distance = sc_dist;
        c.yaw_diff_rad = static_cast<float>(yaw_shift) *
            static_cast<float>(loop_detector_.getScanContextSectorAngleDeg()) *
            static_cast<float>(M_PI / 180.0);
        c.source_flags = LoopCandidate::SOURCE_SC;
        c.candidate_source = LoopCandidate::Source::ScanContextFallback;
        c.sc_rank = i;
        c.fused_score = sc_dist;
        sc_ranked.push_back(c);
    }
    std::sort(sc_ranked.begin(), sc_ranked.end(), [](const LoopCandidate& a, const LoopCandidate& b) {
        return a.sc_distance < b.sc_distance;
    });
    const int keep = std::min(config_.reloc_num_candidates, static_cast<int>(sc_ranked.size()));
    for (int i = 0; i < keep; ++i) {
        sc_ranked[i].fused_rank = i;
        candidates.push_back(sc_ranked[i]);
    }
    LOG(INFO) << "[Reloc/SCFallback] kept=" << candidates.size()
              << " rhpd_enabled=" << config_.rhpd_enabled
              << " rhpd_db=" << rhpd_mgr.size();

    return candidates;
}

RelocResult WorldLocalizing::verifyCandidates(const PointCloudT::Ptr& cloud,
                                              const std::vector<LoopCandidate>& candidates) {
    RelocResult best_result;
    double best_score = std::numeric_limits<double>::max();
    double best_descriptor_distance = std::numeric_limits<double>::max();
    const double fitness_epsilon = 1e-4;

    for (const auto& candidate : candidates) {
        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf) continue;

        auto submap = keyframe_manager_.buildLocalSubmap(candidate.match_id, config_.gicp_submap_size);
        if (!submap || submap->empty()) {
            if (!match_kf->cloud || match_kf->cloud->empty()) continue;
            submap = pcl::make_shared<PointCloudT>();
            Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
            pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
        }

        std::vector<double> yaws_to_try;
        if (config_.rhpd_use_sc_yaw && candidate.fromSC() && std::isfinite(candidate.sc_distance)) {
            const double base_yaw = static_cast<double>(candidate.yaw_diff_rad);
            yaws_to_try = {base_yaw, base_yaw + M_PI};
        } else {
            const int n = std::max(1, config_.rhpd_yaw_hypotheses);
            for (int i = 0; i < n; ++i) {
                yaws_to_try.push_back(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
            }
        }

        MatchResult best_match;
        best_match.fitness_score = std::numeric_limits<double>::max();
        for (double yaw : yaws_to_try) {
            Eigen::Isometry3d init_guess = match_kf->pose_optimized;
            init_guess.linear() = match_kf->pose_optimized.linear() *
                Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            MatchResult mr = matcher_.alignCloud(submap, cloud, init_guess);
            if (mr.converged && mr.fitness_score < best_match.fitness_score) {
                best_match = mr;
            }
        }
        const MatchResult& match_result = best_match;

        LOG(INFO) << "[Reloc] candidate match_id=" << candidate.match_id
                << " rhpd=" << candidate.rhpd_distance
                << " sc=" << candidate.sc_distance
                << " converged=" << match_result.converged
                << " fitness=" << match_result.fitness_score
                << " inlier=" << match_result.inlier_ratio;

        double scale = config_.gicp_fitness_threshold * 0.5;
        double current_confidence = std::exp(-match_result.fitness_score / scale);
        current_confidence = std::max(0.0, std::min(1.0, current_confidence));

        if (match_result.converged &&
            match_result.fitness_score < config_.gicp_fitness_threshold &&
            match_result.inlier_ratio >= config_.reloc_min_inlier_ratio &&
            current_confidence >= config_.reloc_min_confidence) {

            const bool better_fitness = match_result.fitness_score + fitness_epsilon < best_score;
            const bool fitness_tie = std::abs(match_result.fitness_score - best_score) <= fitness_epsilon;
            const double desc_distance = std::isfinite(candidate.rhpd_distance) ? candidate.rhpd_distance : candidate.sc_distance;
            const bool better_desc = desc_distance < best_descriptor_distance;

            if (better_fitness || (fitness_tie && better_desc)) {
                best_score = match_result.fitness_score;
                best_descriptor_distance = desc_distance;
                best_result.success = true;
                best_result.matched_keyframe_id = candidate.match_id;
                best_result.pose_in_map = match_result.T_target_source;
                best_result.fitness_score = match_result.fitness_score;
                best_result.confidence = current_confidence;
            }
        }
    }

    return best_result;
}

bool WorldLocalizing::evaluateSingleCandidate(const PointCloudT::Ptr& cloud,
                                              const LoopCandidate& candidate,
                                              MatchResult& best_match,
                                              int64_t& matched_kf_id) {
    auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
    if (!match_kf) return false;

    auto submap = keyframe_manager_.buildLocalSubmap(candidate.match_id, config_.gicp_submap_size);
    if (!submap || submap->empty()) {
        if (!match_kf->cloud || match_kf->cloud->empty()) return false;
        submap = pcl::make_shared<PointCloudT>();
        Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
        pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
    }

    std::vector<double> yaws_to_try;
    if (config_.rhpd_use_sc_yaw && candidate.fromSC() && std::isfinite(candidate.sc_distance)) {
        const double base_yaw = static_cast<double>(candidate.yaw_diff_rad);
        yaws_to_try = {base_yaw, base_yaw + M_PI};
    } else {
        const int n = std::max(1, config_.rhpd_yaw_hypotheses);
        for (int i = 0; i < n; ++i) {
            yaws_to_try.push_back(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
        }
    }

    best_match.fitness_score = std::numeric_limits<double>::max();
    for (double yaw : yaws_to_try) {
        Eigen::Isometry3d init_guess = match_kf->pose_optimized;
        init_guess.linear() = match_kf->pose_optimized.linear() *
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        MatchResult mr = matcher_.alignCloud(submap, cloud, init_guess);
        if (mr.converged && mr.fitness_score < best_match.fitness_score) {
            best_match = mr;
            matched_kf_id = candidate.match_id;
        }
    }

    return best_match.converged;
}

WorldLocalizing::PointCloudT::Ptr WorldLocalizing::buildRelocQueryCloud(
    const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose) {
    auto current_cloud = pcl::make_shared<PointCloudT>(*cloud);

    QueryFrame frame;
    frame.cloud = current_cloud;
    frame.odom_pose = odom_pose;
    query_frame_buffer_.push_back(std::move(frame));

    const int max_frames = std::max(1, config_.reloc_static_agg_max_frames);
    while (static_cast<int>(query_frame_buffer_.size()) > max_frames) {
        query_frame_buffer_.pop_front();
    }

    if (!config_.reloc_static_agg_enable || max_frames <= 1) {
        return current_cloud;
    }

    std::vector<const QueryFrame*> selected;
    selected.reserve(max_frames);
    for (auto it = query_frame_buffer_.rbegin();
         it != query_frame_buffer_.rend() && static_cast<int>(selected.size()) < max_frames; ++it) {
        const Eigen::Isometry3d delta = odom_pose.inverse() * it->odom_pose;
        const double delta_t = delta.translation().norm();
        const double delta_r = Eigen::AngleAxisd(delta.rotation()).angle();
        if (delta_t <= config_.reloc_static_agg_max_translation &&
            delta_r <= config_.reloc_static_agg_max_rotation) {
            selected.push_back(&(*it));
        } else {
            break;
        }
    }

    if (static_cast<int>(selected.size()) < std::max(1, config_.reloc_static_agg_min_frames)) {
        return current_cloud;
    }

    auto merged = pcl::make_shared<PointCloudT>();
    size_t raw_points = 0;
    for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
        const QueryFrame* src = *it;
        raw_points += src->cloud->size();
        const Eigen::Matrix4f T_curr_from_src = (odom_pose.inverse() * src->odom_pose).matrix().cast<float>();
        PointCloudT transformed;
        pcl::transformPointCloud(*src->cloud, transformed, T_curr_from_src);
        *merged += transformed;
    }

    if (config_.reloc_static_agg_voxel_size > 1e-4 && !merged->empty()) {
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setLeafSize(config_.reloc_static_agg_voxel_size,
                          config_.reloc_static_agg_voxel_size,
                          config_.reloc_static_agg_voxel_size);
        voxel.setInputCloud(merged);
        PointCloudT downsampled;
        voxel.filter(downsampled);
        *merged = downsampled;
    }

    LOG(INFO) << "[Reloc/Aggregation] static_frames=" << selected.size()
              << " raw_points=" << raw_points
              << " aggregated_points=" << merged->size()
              << " motion_gate(t<=" << config_.reloc_static_agg_max_translation
              << ", r<=" << config_.reloc_static_agg_max_rotation << ")";
    return merged->empty() ? current_cloud : merged;
}

double WorldLocalizing::computeRelocLogLikelihood(const LoopCandidate& candidate,
                                                  const MatchResult& match_result) const {
    if (!match_result.converged) return -config_.reloc_hypothesis_not_converged_penalty;
    const double fit_scale = std::max(1e-6, config_.gicp_fitness_threshold);
    const double inlier_term = config_.reloc_reloc_inlier_weight * std::max(0.0, match_result.inlier_ratio);
    const double fitness_term = -match_result.fitness_score / fit_scale;
    const double desc_distance = std::isfinite(candidate.rhpd_distance) ? candidate.rhpd_distance : candidate.sc_distance;
    const double desc_term = -config_.reloc_reloc_desc_dist_weight * desc_distance;
    return fitness_term + inlier_term + desc_term;
}

double WorldLocalizing::computeTrackLogLikelihood(const MatchResult& match_result,
                                                  const Eigen::Isometry3d& predicted_pose) const {
    if (!match_result.converged) return -config_.reloc_hypothesis_not_converged_penalty;

    const double fit_scale = std::max(1e-6, config_.gicp_fitness_threshold);
    const double fitness_term = -match_result.fitness_score / fit_scale;
    const double inlier_term = config_.reloc_reloc_inlier_weight * std::max(0.0, match_result.inlier_ratio);

    const Eigen::Isometry3d delta = predicted_pose.inverse() * match_result.T_target_source;
    const double delta_t = delta.translation().norm();
    const double motion_term = -config_.reloc_track_motion_weight * delta_t;

    return fitness_term + inlier_term + motion_term;
}

void WorldLocalizing::clearRelocHypotheses() {
    pending_hypotheses_.clear();
    hypothesis_window_count_ = 0;
    last_window_winner_seed_id_ = -1;
    winner_streak_ = 0;
}

int64_t WorldLocalizing::findNearestKeyframe(const Eigen::Isometry3d& pose) const {
    int64_t nearest_id = -1;
    double min_distance = std::numeric_limits<double>::max();

    for (const auto& kf : keyframe_manager_.getAllKeyframes()) {
        if (!kf) continue;
        double distance = (kf->pose_optimized.translation() - pose.translation()).norm();
        if (distance < config_.reloc_search_radius && distance < min_distance) {
            min_distance = distance;
            nearest_id = kf->id;
        }
    }

    return nearest_id;
}

} // namespace n3mapping
