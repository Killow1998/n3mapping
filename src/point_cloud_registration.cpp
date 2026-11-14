#include "registration_solver.h"

namespace n3mapping::registration_detail {
small_gicp::RegistrationResult align(
    const small_gicp::PointCloud& target, const small_gicp::PointCloud& source,
    const CorrespondenceSearch::Tree& tree, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting, const SearchBounds* bounds,
    const LazyTargetNormals* lazy_normals, bool reciprocal) {
  const CorrespondenceSearch search(tree, setting.max_correspondence_distance *
                                          setting.max_correspondence_distance, bounds);
  return dispatch(TargetView{target, lazy_normals}, source, search, initial_pose, setting, reciprocal, &tree);
}
} // namespace n3mapping::registration_detail
