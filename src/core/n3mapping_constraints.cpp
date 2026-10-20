// Graph constraints for new mapping keyframes. Raw odometry stays unchanged.
#include "n3mapping/core/n3mapping_core.h"
#include "../constraint_refinement.h"

namespace n3mapping {
bool N3MappingCore::addOdometryConstraint(int64_t keyframe_id,
                                          const Eigen::Isometry3d &pose) {
  auto prev_kf = session_->keyframeManager().getKeyframe(keyframe_id - 1);
  if (!prev_kf) {
    return false;
  }

  EdgeInfo edge;
  edge.from_id = keyframe_id - 1;
  edge.to_id = keyframe_id;
  edge.measurement = prev_kf->pose_odom.inverse() * pose;
  const auto current = session_->keyframeManager().getKeyframe(keyframe_id);
  if (config_.icp_refine_use_gicp && current) {
    auto& matcher = session_->pointCloudMatcher();
    const auto target = matcher.prepareTargetCloud(prev_kf->cloud);
    const auto source = matcher.prepareSourceCloud(current->cloud);
    const auto refined = refineConstraint(config_, matcher, target, source, edge.measurement);
    if (refined.success) {
      edge.measurement = refined.T_target_source;
    }
  }
  edge.information = Eigen::Matrix<double, 6, 6>::Identity();
  edge.information.block<3, 3>(0, 0) *=
      1.0 / (config_.odom_noise_position * config_.odom_noise_position);
  edge.information.block<3, 3>(3, 3) *=
      1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
  edge.type = EdgeType::ODOMETRY;
  session_->graphOptimizer().addOdometryEdge(edge);
  return true;
}

}  // namespace n3mapping
