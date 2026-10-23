// Graph constraints for new mapping keyframes. Raw odometry stays unchanged.
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"

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
  // This edge represents the frontend's motion estimate. Re-registering its
  // deskewed scans would reuse correlated evidence and assign the result the
  // odometry noise below. Keep geometric corrections in the loop/plane factors,
  // where they retain their own admission checks and uncertainty model.
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
