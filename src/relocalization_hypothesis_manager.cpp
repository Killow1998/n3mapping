#include "n3mapping/relocalization_hypothesis_manager.h"

#include <algorithm>
#include <utility>

namespace n3mapping {

bool RelocalizationHypothesisManager::empty() const {
  return hypotheses_.empty();
}

std::size_t RelocalizationHypothesisManager::size() const {
  return hypotheses_.size();
}

std::vector<RelocalizationHypothesis> &
RelocalizationHypothesisManager::hypotheses() {
  return hypotheses_;
}

const std::vector<RelocalizationHypothesis> &
RelocalizationHypothesisManager::hypotheses() const {
  return hypotheses_;
}

void RelocalizationHypothesisManager::start(
    std::vector<Hypothesis> hypotheses,
    const Eigen::Isometry3d &start_odom_pose) {
  reset();
  hypotheses_ = std::move(hypotheses);
  for (auto &hypothesis : hypotheses_) {
    hypothesis.first_observation = 1;
    hypothesis.first_odom_pose = start_odom_pose;
  }
  if (!hypotheses_.empty()) {
    window_count_ = 1;
  }
}

void RelocalizationHypothesisManager::advanceWindow() {
  if (!hypotheses_.empty()) ++window_count_;
}

void RelocalizationHypothesisManager::appendForNextObservation(
    std::vector<Hypothesis> hypotheses, const Eigen::Isometry3d &odom_pose) {
  for (auto &hypothesis : hypotheses) {
    hypothesis.first_observation = window_count_ + 1;
    hypothesis.first_odom_pose = odom_pose;
    hypotheses_.push_back(std::move(hypothesis));
  }
}

int RelocalizationHypothesisManager::windowCount() const {
  return window_count_;
}

int RelocalizationHypothesisManager::persistedFrames() const {
  return persisted_frames_;
}

WinnerStability RelocalizationHypothesisManager::observeWinner(
    const Eigen::Isometry3d &winner_map_odom,
    const Eigen::Isometry3d &current_odom_pose, double max_translation_delta,
    double max_rotation_delta) {
  WinnerStability stability;
  if (has_last_winner_transform_) {
    const Eigen::Isometry3d previous_pose =
        last_winner_map_odom_ * current_odom_pose;
    const Eigen::Isometry3d current_pose = winner_map_odom * current_odom_pose;
    const Eigen::Isometry3d delta = previous_pose.inverse() * current_pose;
    stability.translation_delta = delta.translation().norm();
    stability.rotation_delta = Eigen::AngleAxisd(delta.rotation()).angle();
    stability.same_physical_pose =
        stability.translation_delta < max_translation_delta &&
        stability.rotation_delta < max_rotation_delta;
  }

  winner_streak_ = stability.same_physical_pose ? winner_streak_ + 1 : 1;
  last_winner_map_odom_ = winner_map_odom;
  has_last_winner_transform_ = true;
  stability.streak = winner_streak_;
  return stability;
}

int RelocalizationHypothesisManager::winnerStreak() const {
  return winner_streak_;
}

HypothesisMotionBaseline RelocalizationHypothesisManager::motionBaseline(
    const Hypothesis *hypothesis, const Eigen::Isometry3d &current_odom_pose) const {
  HypothesisMotionBaseline baseline;
  if (!hypothesis || window_count_ <= hypothesis->first_observation) {
    return baseline;
  }
  const Eigen::Isometry3d motion =
      current_odom_pose.inverse() * hypothesis->first_odom_pose;
  baseline.translation = motion.translation().norm();
  baseline.rotation = Eigen::AngleAxisd(motion.rotation()).angle();
  return baseline;
}

HypothesisPersistenceResult RelocalizationHypothesisManager::finishWindow(
    bool success, bool persistence_enabled, int max_persist_frames) {
  HypothesisPersistenceResult result;
  if (success || !persistence_enabled) {
    result.any_alive = std::any_of(
        hypotheses_.begin(), hypotheses_.end(),
        [](const Hypothesis &hypothesis) { return hypothesis.alive; });
    reset();
    return result;
  }

  ++persisted_frames_;
  result.persisted_frames = persisted_frames_;
  result.any_alive = std::any_of(
      hypotheses_.begin(), hypotheses_.end(),
      [](const Hypothesis &hypothesis) { return hypothesis.alive; });
  if (!result.any_alive || persisted_frames_ > max_persist_frames) {
    result.reseeded = true;
    reset();
  }
  return result;
}

void RelocalizationHypothesisManager::reset() {
  hypotheses_.clear();
  window_count_ = 0;
  has_last_winner_transform_ = false;
  last_winner_map_odom_ = Eigen::Isometry3d::Identity();
  winner_streak_ = 0;
  persisted_frames_ = 0;
}

} // namespace n3mapping
