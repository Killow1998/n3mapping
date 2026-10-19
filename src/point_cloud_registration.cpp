#include "registration_search.h"
#include "registration_reduction.h"

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/registration/registration.hpp>

namespace n3mapping::registration_detail {
namespace {
// Each upstream factor already owns its last correspondence. Reuse that index
// only to bound the next exact search; the factor and optimizer remain intact.
template <typename Factor>
struct HintedFactor : Factor {
  using Factor::Factor;

  template <typename Target, typename Source, typename Rejector>
  bool linearize(const Target& target, const Source& source,
                 const CorrespondenceSearch& search, const Eigen::Isometry3d& pose,
                 std::size_t source_index, const Rejector& rejector,
                 Eigen::Matrix<double, 6, 6>* H,
                 Eigen::Matrix<double, 6, 1>* b, double* error) {
    const auto hinted_search = search.withHint(this->target_index);
    return Factor::linearize(target, source, hinted_search, pose,
                             source_index, rejector, H, b, error);
  }
};

template <typename Factor>
small_gicp::RegistrationResult alignWithFactor(
    const small_gicp::PointCloud& target, const small_gicp::PointCloud& source,
    const CorrespondenceSearch::Tree& tree, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting) {
  small_gicp::Registration<HintedFactor<Factor>, RegistrationReduction> registration;
  registration.reduction.num_threads = setting.num_threads;
  registration.rejector.max_dist_sq =
      setting.max_correspondence_distance * setting.max_correspondence_distance;
  registration.criteria.rotation_eps = setting.rotation_eps;
  registration.criteria.translation_eps = setting.translation_eps;
  registration.optimizer.max_iterations = setting.max_iterations;
  registration.optimizer.verbose = setting.verbose;
  const CorrespondenceSearch search(tree, registration.rejector.max_dist_sq);
  return registration.align(target, source, search, initial_pose);
}
}  // namespace

small_gicp::RegistrationResult align(
    const small_gicp::PointCloud& target, const small_gicp::PointCloud& source,
    const CorrespondenceSearch::Tree& tree, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting) {
  switch (setting.type) {
    case small_gicp::RegistrationSetting::ICP:
      return alignWithFactor<small_gicp::ICPFactor>(target, source, tree, initial_pose, setting);
    case small_gicp::RegistrationSetting::PLANE_ICP:
      return alignWithFactor<small_gicp::PointToPlaneICPFactor>(target, source, tree, initial_pose, setting);
    case small_gicp::RegistrationSetting::GICP:
      return alignWithFactor<small_gicp::GICPFactor>(target, source, tree, initial_pose, setting);
    default:
      return small_gicp::align(target, source, tree, initial_pose, setting);
  }
}
}  // namespace n3mapping::registration_detail
