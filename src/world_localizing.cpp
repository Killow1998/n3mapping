// WorldLocalizing: global relocalization via ScanContext + ICP, and tracking localization with T_map_odom.
#include "n3mapping/world_localizing.h"

#include <glog/logging.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <boost/make_shared.hpp>
#include <pcl/common/transforms.h>

namespace n3mapping {

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
    , hypothesis_window_count_(0) {}

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

    if (pending_hypotheses_.empty()) {
        auto candidates = searchCandidates(cloud);
        if (candidates.empty()) {
            LOG(WARNING) << "No relocalization candidates found. Map has " << keyframe_manager_.size()
                         << " keyframes, SC threshold=" << config_.reloc_sc_dist_threshold;
            return result;
        }

        LOG(INFO) << "Relocalization: initialize " << candidates.size() << " hypotheses";
        for (const auto& candidate : candidates) {
            MatchResult match_result;
            int64_t matched_kf_id = -1;
            if (!evaluateSingleCandidate(cloud, candidate, match_result, matched_kf_id)) continue;

            RelocHypothesis hyp;
            hyp.seed_match_id = candidate.match_id;
            hyp.last_match_id = matched_kf_id;
            hyp.T_map_odom = match_result.T_target_source * odom_pose.inverse();
            hyp.cumulative_log_likelihood = computeRelocLogLikelihood(candidate, match_result);
            hyp.num_updates = 1;
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

            MatchResult mr = matcher_.alignCloud(submap, cloud, predicted_pose);
            hyp.cumulative_log_likelihood += computeTrackLogLikelihood(mr, predicted_pose);
            hyp.num_updates += 1;
            if (mr.converged) {
                hyp.T_map_odom = mr.T_target_source * odom_pose.inverse();
                hyp.last_match_id = nearest_kf_id;
            }
        }
    }

    if (hypothesis_window_count_ < config_.reloc_temporal_window_size) {
        LOG(INFO) << "Relocalization pending: window " << hypothesis_window_count_
                  << "/" << config_.reloc_temporal_window_size
                  << ", active hypotheses=" << pending_hypotheses_.size();
        return result;
    }

    auto best_it = std::max_element(
        pending_hypotheses_.begin(), pending_hypotheses_.end(),
        [](const RelocHypothesis& a, const RelocHypothesis& b) {
            return a.cumulative_log_likelihood < b.cumulative_log_likelihood;
        });

    if (best_it != pending_hypotheses_.end() &&
        best_it->cumulative_log_likelihood >= config_.reloc_lock_log_likelihood_threshold) {
        T_map_odom_ = best_it->T_map_odom;
        is_relocalized_ = true;
        last_matched_id_ = best_it->last_match_id;
        last_odom_pose_ = odom_pose;
        consecutive_track_failures_ = 0;

        result.success = true;
        result.matched_keyframe_id = best_it->last_match_id;
        result.pose_in_map = T_map_odom_ * odom_pose;
        result.confidence = std::max(0.0, std::min(1.0, best_it->cumulative_log_likelihood / 10.0));
        result.fitness_score = 0.0;

        LOG(INFO) << "Relocalization locked after temporal window. KF=" << result.matched_keyframe_id
                  << ", cumulative_loglik=" << best_it->cumulative_log_likelihood;
    } else {
        double best_ll = (best_it != pending_hypotheses_.end()) ? best_it->cumulative_log_likelihood : -1e9;
        LOG(WARNING) << "Relocalization rejected after temporal window. best_loglik="
                     << best_ll << " < " << config_.reloc_lock_log_likelihood_threshold;
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
    clearRelocHypotheses();
}

void WorldLocalizing::setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom) {
    std::lock_guard<std::mutex> lock(mutex_);
    T_map_odom_ = T_map_odom;
    is_relocalized_ = true;
    clearRelocHypotheses();
}

int64_t WorldLocalizing::getLastMatchedKeyframeId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_matched_id_;
}

int WorldLocalizing::getConsecutiveTrackFailures() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return consecutive_track_failures_;
}

int WorldLocalizing::getMaxTrackFailures() const {
    return config_.reloc_max_track_failures;
}

std::vector<LoopCandidate> WorldLocalizing::searchCandidates(const PointCloudT::Ptr& cloud) {
    std::vector<LoopCandidate> candidates;

    // ---- RHPD path ----
    if (config_.rhpd_enabled) {
        const auto& rhpd_mgr = loop_detector_.getRHPDManager();
        if (rhpd_mgr.size() == 0) {
            LOG(WARNING) << "[Reloc/RHPD] No RHPD descriptors in database.";
            return candidates;
        }

        Eigen::VectorXd query_desc = loop_detector_.computeRHPD(cloud);
        if (query_desc.isZero()) {
            LOG(WARNING) << "[Reloc/RHPD] Failed to compute RHPD for query cloud.";
            return candidates;
        }

        auto top_k = rhpd_mgr.search(query_desc, config_.rhpd_num_candidates);
        LOG(INFO) << "[Reloc/RHPD] Top candidates (db size=" << rhpd_mgr.size() << "):";
        for (size_t i = 0; i < top_k.size(); ++i) {
            LOG(INFO) << "  [" << i << "] kf=" << top_k[i].first << " dist=" << top_k[i].second;
            if (top_k[i].second > config_.rhpd_dist_threshold) break;
            LoopCandidate c;
            c.query_id  = -1;
            c.match_id  = top_k[i].first;
            c.sc_distance = top_k[i].second;
            c.yaw_diff_rad = 0.0f;  // RHPD handles 180° internally; ICP will recover full pose
            candidates.push_back(c);
        }
        return candidates;
    }

    // ---- Fallback: SC path ----
    Eigen::MatrixXd descriptor = loop_detector_.makeScanContext(cloud);
    if (descriptor.size() == 0) return candidates;

    auto all_descriptors = loop_detector_.getDescriptors();
    if (all_descriptors.empty()) return candidates;

    std::vector<std::tuple<double, int, int64_t>> distances;
    for (const auto& [kf_id, kf_desc] : all_descriptors) {
        if (kf_desc.size() == 0) continue;
        Eigen::MatrixXd desc_copy = descriptor;
        Eigen::MatrixXd kf_desc_copy = kf_desc;
        auto [dist, yaw_shift] = loop_detector_.computeDistance(desc_copy, kf_desc_copy);
        distances.emplace_back(dist, yaw_shift, kf_id);
    }

    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    int num_candidates = std::min(config_.reloc_num_candidates, static_cast<int>(distances.size()));
    for (int i = 0; i < num_candidates; ++i) {
        double dist = std::get<0>(distances[i]);
        int yaw_shift = std::get<1>(distances[i]);
        int64_t kf_id = std::get<2>(distances[i]);

        if (dist < config_.reloc_sc_dist_threshold) {
            LoopCandidate candidate;
            candidate.query_id = -1;
            candidate.match_id = kf_id;
            candidate.sc_distance = dist;
            auto [rows, cols] = loop_detector_.getDescriptorDimensions();
            double sector_angle = 360.0 / static_cast<double>(cols);
            candidate.yaw_diff_rad = static_cast<float>(yaw_shift) * static_cast<float>(sector_angle) * static_cast<float>(M_PI / 180.0);
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

RelocResult WorldLocalizing::verifyCandidates(const PointCloudT::Ptr& cloud,
                                              const std::vector<LoopCandidate>& candidates) {
    RelocResult best_result;
    double best_score = std::numeric_limits<double>::max();
    double best_sc_distance = std::numeric_limits<double>::max();
    const double fitness_epsilon = 1e-4;

    for (const auto& candidate : candidates) {
        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf) continue;

        auto submap = keyframe_manager_.buildLocalSubmap(candidate.match_id, config_.gicp_submap_size);
        if (!submap || submap->empty()) {
            if (!match_kf->cloud || match_kf->cloud->empty()) continue;
            submap = boost::make_shared<PointCloudT>();
            Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
            pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
        }

        // Build list of initial guesses to try.
        // RHPD: yaw_diff_rad == 0 means unknown orientation → try 4 quadrant yaws.
        // SC:   yaw_diff_rad is estimated → try only that yaw.
        std::vector<double> yaws_to_try;
        if (config_.rhpd_enabled && std::abs(candidate.yaw_diff_rad) < 1e-6f) {
            yaws_to_try = {0.0, M_PI_2, M_PI, 3.0 * M_PI_2};
        } else {
            yaws_to_try = {static_cast<double>(candidate.yaw_diff_rad)};
        }

        MatchResult best_match;
        best_match.fitness_score = std::numeric_limits<double>::max();
        for (double yaw : yaws_to_try) {
            Eigen::Isometry3d init_guess = match_kf->pose_optimized;
            init_guess.prerotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
            MatchResult mr = matcher_.alignCloud(submap, cloud, init_guess);
            if (mr.converged && mr.fitness_score < best_match.fitness_score) {
                best_match = mr;
            }
        }
        const MatchResult& match_result = best_match;

        LOG(INFO) << "[Reloc] candidate match_id=" << candidate.match_id
                << " dist=" << candidate.sc_distance
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
            const bool better_sc = candidate.sc_distance < best_sc_distance;

            if (better_fitness || (fitness_tie && better_sc)) {
                best_score = match_result.fitness_score;
                best_sc_distance = candidate.sc_distance;
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
        submap = boost::make_shared<PointCloudT>();
        Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
        pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
    }

    std::vector<double> yaws_to_try;
    if (config_.rhpd_enabled && std::abs(candidate.yaw_diff_rad) < 1e-6f) {
        yaws_to_try = {0.0, M_PI_2, M_PI, 3.0 * M_PI_2};
    } else {
        yaws_to_try = {static_cast<double>(candidate.yaw_diff_rad)};
    }

    best_match.fitness_score = std::numeric_limits<double>::max();
    for (double yaw : yaws_to_try) {
        Eigen::Isometry3d init_guess = match_kf->pose_optimized;
        init_guess.prerotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        MatchResult mr = matcher_.alignCloud(submap, cloud, init_guess);
        if (mr.converged && mr.fitness_score < best_match.fitness_score) {
            best_match = mr;
            matched_kf_id = candidate.match_id;
        }
    }

    return best_match.converged;
}

double WorldLocalizing::computeRelocLogLikelihood(const LoopCandidate& candidate,
                                                  const MatchResult& match_result) const {
    if (!match_result.converged) return -config_.reloc_hypothesis_not_converged_penalty;
    const double fit_scale = std::max(1e-6, config_.gicp_fitness_threshold);
    const double inlier_term = config_.reloc_reloc_inlier_weight * std::max(0.0, match_result.inlier_ratio);
    const double fitness_term = -match_result.fitness_score / fit_scale;
    const double desc_term = -config_.reloc_reloc_desc_dist_weight * candidate.sc_distance;
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
