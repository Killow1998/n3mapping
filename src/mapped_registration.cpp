#include "registration_solver.h"

namespace n3mapping::registration_detail {
small_gicp::RegistrationResult align(
    const PointCloudMatcher::PreparedTargetLevel& target, const small_gicp::PointCloud& source,
    const Eigen::Isometry3d& initial_pose, const small_gicp::RegistrationSetting& setting,
    bool reciprocal) {
  if (!target.mapped)
    return align(*target.cloud, source, *target.kdtree, initial_pose, setting,
                 target.search_bounds.get(), target.lazy_normals.get(), reciprocal);
  const MappedCorrespondenceSearch search(*target.mapped, setting.max_correspondence_distance *
                                          setting.max_correspondence_distance, target.use_occupied_bounds);
  return dispatch(MappedTargetView{*target.mapped}, source, search, initial_pose, setting, reciprocal, nullptr);
}
} // namespace n3mapping::registration_detail
