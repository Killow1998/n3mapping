#pragma once
#include "registration_search.h"
#include <type_traits>
#include "registration_reduction.h"
#include "point_cloud_target_view.h"
#include <small_gicp/factors/plane_icp_factor.hpp>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>

namespace n3mapping::registration_detail {
namespace {

struct ReciprocalRejector : small_gicp::DistanceRejector {
  const small_gicp::UnsafeKdTree<small_gicp::PointCloud>* source_tree = nullptr;

  template <typename Target, typename Source>
  bool operator()(const Target& target, const Source&, const Eigen::Isometry3d& pose,
                  std::size_t target_id, std::size_t source_id, double distance) const {
    if (distance > max_dist_sq) return true;
    std::size_t back;
    double back_distance;
    return !source_tree->nearest_neighbor_search(
        pose.inverse() * small_gicp::traits::point(target, target_id),
        &back, &back_distance) || back != source_id;
  }
};

// Each upstream factor already owns its last correspondence. Reuse that index
// only to bound the next exact search; the factor and optimizer remain intact.
template <typename Factor>
struct HintedFactor : Factor {
  using Factor::Factor;

  template <typename Target, typename Source, typename Search, typename Rejector>
  bool linearize(const Target& target, const Source& source,
                 const Search& search, const Eigen::Isometry3d& pose,
                 std::size_t source_index, const Rejector& rejector,
                 Eigen::Matrix<double, 6, 6>* H,
                 Eigen::Matrix<double, 6, 1>* b, double* error) {
    const auto hinted_search = search.withHint(this->target_index);
    return Factor::linearize(target, source, hinted_search, pose,
                             source_index, rejector, H, b, error);
  }
};

template <typename Factor, typename Target, typename Search, typename Rejector = small_gicp::DistanceRejector>
small_gicp::RegistrationResult alignWithFactor(
    const Target& target, const small_gicp::PointCloud& source,
    const Search& search, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting, const Rejector& rejector = {}) {
  small_gicp::Registration<HintedFactor<Factor>, RegistrationReduction,
                         small_gicp::NullFactor, Rejector> registration;
  registration.reduction.num_threads = setting.num_threads;
  registration.rejector = rejector;
  registration.rejector.max_dist_sq =
      setting.max_correspondence_distance * setting.max_correspondence_distance;
  registration.criteria.rotation_eps = setting.rotation_eps;
  registration.criteria.translation_eps = setting.translation_eps;
  registration.optimizer.max_iterations = setting.max_iterations;
  registration.optimizer.verbose = setting.verbose;
  return registration.align(target, source, search, initial_pose);
}
template <typename Target, typename Search>
small_gicp::RegistrationResult dispatch(
    const Target& target, const small_gicp::PointCloud& source,
    const Search& search, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting, bool reciprocal,
    [[maybe_unused]] const CorrespondenceSearch::Tree* native_tree) {
  switch (setting.type) {
    case small_gicp::RegistrationSetting::ICP:
      return alignWithFactor<small_gicp::ICPFactor>(target, source, search, initial_pose, setting);
    case small_gicp::RegistrationSetting::PLANE_ICP:
      return alignWithFactor<small_gicp::PointToPlaneICPFactor>(target, source, search, initial_pose, setting);
    case small_gicp::RegistrationSetting::GICP:
      if (reciprocal) {
        const small_gicp::UnsafeKdTree<small_gicp::PointCloud> source_tree(
            source, small_gicp::KdTreeBuilderOMP(setting.num_threads));
        ReciprocalRejector rejector;
        rejector.source_tree = &source_tree;
        return alignWithFactor<small_gicp::GICPFactor>(
            target, source, search, initial_pose, setting, rejector);
      }
      return alignWithFactor<small_gicp::GICPFactor>(target, source, search, initial_pose, setting);
    default:
      if constexpr (std::is_same_v<Target, TargetView>) {
        if (target.lazy_normals)
          for (std::size_t i = 0; i < target.cloud.size(); ++i) target.normal(i);
        return small_gicp::align(target.cloud, source, *native_tree, initial_pose, setting);
      }
      return {};
  }
}
} // namespace
} // namespace n3mapping::registration_detail
