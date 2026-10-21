#include "association.h"
#include <map>

namespace n3mapping {
namespace {
bool planar(const PlaneMoment& moment) {
  if (moment.count < 6) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment.covariance());
  const auto v = eigen.eigenvalues();
  return eigen.info() == Eigen::Success && v[1] >= .0025 &&
         v[0] <= .0004 && v[0] <= .02 * v[1];
}
void combine(PlaneMoment& target, const PlaneMoment& source) {
  target.count += source.count;
  target.sum += source.sum;
  target.square += source.square;
}
}  // namespace

std::vector<PlaneMoment> extractPlaneMoments(const Keyframe::PointCloudT& cloud) {
  std::map<planar_geometry::Cell, PlaneMoment> cells;
  for (const auto& p : cloud) {
    const Eigen::Vector3d point(p.x, p.y, p.z);
    if (!point.allFinite()) continue;
    planar_geometry::Cell cell;
    for (int axis = 0; axis < 3; ++axis) cell[axis] = std::floor(point[axis]);
    cells[cell].add(point);
  }
  std::vector<PlaneMoment> result;
  for (const auto& [cell, moment] : cells)
    if (planar(moment)) result.push_back(moment);
  return result;
}

namespace planar_geometry {
std::vector<std::pair<size_t, Moment>> associateToPlanes(
    const std::vector<Moment>& moments, const gtsam::Pose3& pose,
    const std::vector<SharedPlane>& planes,
    const std::map<int64_t, Eigen::Isometry3d>& reference_poses) {
  // Bound support by the observed footprints, rather than extending a saved
  // plane to infinity. Ambiguous associations are omitted, never tie-broken.
  std::vector<Eigen::AlignedBox3d> support(planes.size());
  for (size_t i = 0; i < planes.size(); ++i)
    for (const auto& observation : planes[i].observations) {
      const auto it = reference_poses.find(observation.node_id);
      if (it == reference_poses.end()) continue;
      const auto& transform = it->second;
      const auto& moment = observation.moment;
      const Eigen::Vector3d center = transform * moment.mean();
      const Eigen::Matrix3d covariance = transform.linear() * moment.covariance() *
                                          transform.linear().transpose();
      const Eigen::Vector3d radius = 2. * covariance.diagonal().cwiseMax(0.).cwiseSqrt() +
                                     Eigen::Vector3d::Ones();
      support[i].extend(center - radius);
      support[i].extend(center + radius);
    }
  std::map<size_t, Moment> grouped;
  const double normal_gate = std::cos(5. * M_PI / 180.);
  for (const auto& moment : moments) {
    if (!planar(moment)) continue;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment.covariance());
    const Eigen::Vector3d center = pose.transformFrom(moment.mean());
    Eigen::Vector3d normal = pose.rotation().matrix() * eigen.eigenvectors().col(0);
    if (normal.dot(pose.translation() - center) < 0) normal = -normal;
    size_t match = planes.size();
    bool ambiguous = false;
    for (size_t i = 0; i < planes.size(); ++i) {
      const auto& plane = planes[i];
      if (!support[i].contains(center) || plane.normal.dot(normal) < normal_gate ||
          std::abs(plane.normal.dot(center - plane.origin) - plane.offset) >= .1)
        continue;
      if (match != planes.size()) { ambiguous = true; break; }
      match = i;
    }
    if (!ambiguous && match != planes.size()) combine(grouped[match], moment);
  }
  std::vector<std::pair<size_t, Moment>> result;
  for (const auto& [index, moment] : grouped) {
    if (!planar(moment)) continue;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment.covariance());
    const Eigen::Vector3d center = pose.transformFrom(moment.mean());
    Eigen::Vector3d normal = pose.rotation().matrix() * eigen.eigenvectors().col(0);
    if (normal.dot(pose.translation() - center) < 0) normal = -normal;
    const auto& plane = planes[index];
    if (plane.normal.dot(normal) >= normal_gate &&
        std::abs(plane.normal.dot(center - plane.origin) - plane.offset) < .1)
      result.emplace_back(index, moment);
  }
  return result;
}
}  // namespace planar_geometry
}  // namespace n3mapping
