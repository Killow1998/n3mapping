#include "n3mapping/graph_optimizer.h"
#include "n3mapping/graph_factor_noise.h"
#include "planar_geometry/moment_factor.h"
#include <chrono>
#include <iostream>
#include <tbb/global_control.h>

namespace n3mapping {
std::vector<SharedPlane> GraphOptimizer::sharedPlanes() const {
  auto planes = shared_planes_;
  for (size_t i = 0; i < planes.size(); ++i) {
    const auto& value = current_estimate_.at<gtsam::OrientedPlane3>(gtsam::Symbol('p', i));
    planes[i].normal = value.normal().unitVector();
    planes[i].offset = value.distance();
  }
  return planes;
}

Eigen::Isometry3d GraphOptimizer::gaugePrior() const {
  for (const auto* factors : {&graph_, &new_factors_})
    for (const auto& factor : *factors) {
      const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get());
      if (prior) return gtsamToEigen(prior->prior());
    }
  throw std::logic_error("Missing graph gauge");
}

bool GraphOptimizer::replaceSharedPlanes(const std::vector<SharedPlane>& planes,
                                        bool optimize,
                                        GeometryOptimizationStats& stats) {
  if (!new_factors_.empty() || !new_values_.empty()) return false;
  try {
    gtsam::NonlinearFactorGraph candidate;
    for (const auto& factor : graph_)
      if (!dynamic_cast<const planar_geometry::MomentPlaneFactor*>(factor.get()))
        candidate.push_back(factor);
    auto values = current_estimate_.empty() ? initial_values_ : current_estimate_;
    for (size_t i = 0; i < shared_planes_.size(); ++i)
      values.erase(gtsam::Symbol('p', i));
    for (size_t i = 0; i < planes.size(); ++i) {
      const auto& plane = planes[i];
      if (!plane.normal.allFinite() || !plane.origin.allFinite() ||
          std::abs(plane.normal.norm() - 1.) > 1e-6 ||
          !std::isfinite(plane.offset) || !std::isfinite(plane.sigma) ||
          plane.sigma < .01 || plane.observations.size() < 3)
        throw std::invalid_argument("Invalid shared plane");
      const auto key = gtsam::Symbol('p', i);
      values.insert(key, gtsam::OrientedPlane3(gtsam::Unit3(plane.normal), plane.offset));
      std::set<int64_t> observed;
      for (const auto& observation : plane.observations) {
        const auto noise = planar_geometry::momentNoise(
            config_, plane.sigma, observation.information_weight);
        const auto& m = observation.moment;
        if (!node_ids_.count(observation.node_id) || !observed.insert(observation.node_id).second ||
            m.count < 6 || !m.sum.allFinite() || !m.square.allFinite())
          throw std::invalid_argument("Invalid plane observation");
        const Eigen::Matrix3d covariance = m.covariance();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
        if (!covariance.isApprox(covariance.transpose(), 1e-9) ||
            eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() < -1e-7)
          throw std::invalid_argument("Invalid plane covariance");
        candidate.add(planar_geometry::MomentPlaneFactor(
            gtsam::Symbol('x', observation.node_id), key, m, plane.origin, noise));
      }
    }
    stats.initial_cost = candidate.error(values);
    if (optimize) {
      const auto start = std::chrono::steady_clock::now();
      tbb::global_control serial(tbb::global_control::max_allowed_parallelism, 1);
      gtsam::LevenbergMarquardtParams params;
      params.maxIterations = 20;
      params.diagonalDamping = true;
      values = gtsam::LevenbergMarquardtOptimizer(candidate, values, params).optimize();
      stats.solve_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    }
    stats.final_cost = candidate.error(values);
    if (!std::isfinite(stats.final_cost) ||
        stats.final_cost > stats.initial_cost + 1e-6 * (1 + stats.initial_cost))
      throw std::runtime_error("Geometry solve did not decrease finite cost");
    for (int64_t id : node_ids_)
      if (!values.at<gtsam::Pose3>(gtsam::Symbol('x', id)).matrix().allFinite())
        throw std::runtime_error("Non-finite geometry pose");
    for (int64_t id : node_ids_) {
      const auto key = gtsam::Symbol('x', id);
      const auto& before = current_estimate_.at<gtsam::Pose3>(key);
      const auto& after = values.at<gtsam::Pose3>(key);
      stats.max_translation_update_m = std::max(stats.max_translation_update_m,
          (before.translation()-after.translation()).norm());
      stats.max_rotation_update_rad = std::max(stats.max_rotation_update_rad,
          gtsam::Rot3::Logmap(before.rotation().between(after.rotation())).norm());
    }
    graph_ = std::move(candidate);
    current_estimate_ = values;
    initial_values_ = values;
    last_good_estimate_ = values;
    shared_planes_ = planes;
    isam2_ = createISAM2(); // Next graph update builds its own transactional solver.
    stats.optimized = optimize;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Joint geometry unchanged: " << e.what() << '\n';
    return false;
  }
}
}  // namespace n3mapping
