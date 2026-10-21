#pragma once
#include "n3mapping/keyframe.h"
#include <vector>

namespace n3mapping {
// Body-frame sufficient statistics. No transformed cloud or truth is retained.
struct PlaneMoment {
  int count = 0;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  Eigen::Matrix3d square = Eigen::Matrix3d::Zero();
  void add(const Eigen::Vector3d& p) {
    ++count;
    sum += p;
    square.noalias() += p * p.transpose();
  }
  Eigen::Vector3d mean() const { return sum / count; }
  Eigen::Matrix3d covariance() const {
    return square / count - mean() * mean().transpose();
  }
};
struct PlaneObservation {
  int64_t node_id = -1;
  PlaneMoment moment;
  double information_weight = 1.;
};
struct SharedPlane {
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  double offset = 0;
  double sigma = .01;
  std::vector<PlaneObservation> observations;
};
struct GeometryOptimizationStats {
  size_t planes = 0, observations = 0;
  double association_ms = 0, solve_ms = 0;
  double initial_cost = 0, final_cost = 0;
  double max_translation_update_m = 0, max_rotation_update_rad = 0;
  bool optimized = false;
};
std::vector<SharedPlane> buildSharedPlanes(
    const std::vector<Keyframe::Ptr>& keyframes,
    GeometryOptimizationStats& stats);
// Body-frame patches for a newly appended scan. No map or future pose is used.
std::vector<PlaneMoment> extractPlaneMoments(const Keyframe::PointCloudT& cloud);
// Only identical acquisition time, raw pose and point payload share an identity.
// Nearby/repeated views with different measurements remain independent here.
std::vector<size_t> geometrySourceIds(const std::vector<Keyframe::Ptr>& keyframes);
}  // namespace n3mapping
