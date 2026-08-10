// Temporal state and lifecycle for relocalization hypotheses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

namespace n3mapping {

struct RelocalizationHypothesis {
  int64_t seed_match_id = -1;
  int64_t last_match_id = -1;
  Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
  double cumulative_log_likelihood = 0.0;
  int num_updates = 0;
  int converged_updates = 0;
  double visibility_consistency_sum = 0.0;
  double visibility_evidence_sum = 0.0;
  int visibility_updates = 0;
  bool alive = true;
  double last_rot_info_min = 0.0;
  double last_trans_info_min = 0.0;
  int last_iterations = 0;
  // Diagnostics only: which pose the recorded registration numbers describe,
  // and whether that registration would have passed the production gate.
  bool last_pose_is_refined = false;
  bool last_production_quality = false;
  double last_rot_info_marginal_min = 0.0;
  int last_termination = 0;
  double last_inlier_ratio = 0.0;
  double last_fitness = 0.0;
};

struct WinnerStability {
  int streak = 0;
  bool same_physical_pose = false;
  double translation_delta = std::numeric_limits<double>::quiet_NaN();
  double rotation_delta = std::numeric_limits<double>::quiet_NaN();
};

struct HypothesisMotionBaseline {
  double translation = 0.0;
  double rotation = 0.0;
};

struct HypothesisPersistenceResult {
  bool reseeded = false;
  bool any_alive = false;
  int persisted_frames = 0;
};

class RelocalizationHypothesisManager {
public:
  using Hypothesis = RelocalizationHypothesis;

  bool empty() const;
  std::size_t size() const;
  std::vector<Hypothesis> &hypotheses();
  const std::vector<Hypothesis> &hypotheses() const;

  void start(std::vector<Hypothesis> hypotheses,
             const Eigen::Isometry3d &start_odom_pose);
  void advanceWindow();
  int windowCount() const;
  int persistedFrames() const;

  WinnerStability observeWinner(const Eigen::Isometry3d &winner_map_odom,
                                const Eigen::Isometry3d &current_odom_pose,
                                double max_translation_delta,
                                double max_rotation_delta);
  int winnerStreak() const;
  HypothesisMotionBaseline
  motionBaseline(const Eigen::Isometry3d &current_odom_pose) const;

  HypothesisPersistenceResult
  finishWindow(bool success, bool persistence_enabled, int max_persist_frames);
  void reset();

private:
  std::vector<Hypothesis> hypotheses_;
  int window_count_ = 0;
  Eigen::Isometry3d window_start_odom_pose_ = Eigen::Isometry3d::Identity();
  bool has_last_winner_transform_ = false;
  Eigen::Isometry3d last_winner_map_odom_ = Eigen::Isometry3d::Identity();
  int winner_streak_ = 0;
  int persisted_frames_ = 0;
};

} // namespace n3mapping
