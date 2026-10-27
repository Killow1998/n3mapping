#include "causal_submap.h"
#include "n3mapping/cloud_utils.h"
#include <pcl/common/transforms.h>
namespace n3mapping {
Keyframe::PointCloudT::Ptr
buildCausalSubmap(const std::map<int64_t, Keyframe::Ptr> &keyframes,
                  int64_t center_id, int range, int64_t root_id,
                  bool use_odometry) {
  auto submap = pcl::make_shared<Keyframe::PointCloudT>();

  auto root_it = keyframes.find(root_id);
  if (root_it == keyframes.end())
    return submap;

  const auto &root = *root_it->second;
  const Eigen::Matrix4f T_root_inv =
      root.pose_optimized.matrix().cast<float>().inverse();
  const auto T_odom_root_inv = root.pose_odom.inverse();

  for (int64_t id = center_id - range; id <= center_id; ++id) {
    auto it = keyframes.find(id);
    if (it == keyframes.end())
      continue;
    auto kf = it->second;
    if (!kf->cloud || kf->cloud->empty())
      continue;
    if (use_odometry && kf->session_id != root.session_id)
      continue;
    // A newly inserted frame still has its raw pose while its neighbors
    // may already be corrected. Never mix those domains in a descriptor.
    const Eigen::Matrix4f T_root_kf =
        use_odometry
            ? (T_odom_root_inv * kf->pose_odom).matrix().cast<float>().eval()
            : (T_root_inv * kf->pose_optimized.matrix().cast<float>()).eval();
    Keyframe::PointCloudT transformed;
    pcl::transformPointCloud(*kf->cloud, transformed, T_root_kf);
    *submap += transformed;
  }
  return submap;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr
maybeVoxelizeRhpdCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                       double voxel_size) {
  if (!cloud || cloud->empty() || voxel_size <= 1e-4)
    return cloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered;
  if (!safeVoxelGridFilter<pcl::PointXYZI>(cloud, voxel_size, &filtered)) {
    return cloud;
  }
  return filtered->empty() ? cloud : filtered;
}
} // namespace n3mapping
