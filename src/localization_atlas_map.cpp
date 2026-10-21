// Global reference assembly. Packed scratch removes PointXYZI's SIMD padding;
// PCL retains the same voxel ordering, centroid arithmetic and intensity data.
#include "n3mapping/localization_atlas.h"
#include "n3mapping/pcl_compat.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>

namespace n3mapping {
LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::buildGlobalMap(
    const Config& config, const std::vector<Keyframe::Ptr>& keyframes) {
  size_t count = 0;
  for (const auto& frame : keyframes)
    if (frame && frame->cloud) count += frame->cloud->size();
  const auto assemble_raw = [&]() {
    auto raw = pcl::make_shared<PointCloudT>();
    raw->reserve(count);
    for (const auto& frame : keyframes) {
      if (!frame || !frame->cloud || frame->cloud->empty()) continue;
      PointCloudT transformed;
      pcl::transformPointCloud(*frame->cloud, transformed,
                               frame->pose_optimized.matrix().cast<float>());
      *raw += transformed;
    }
    return raw;
  };
  constexpr size_t stride = 4 * sizeof(float);
  const double leaf = std::max(1e-3, config.global_map_voxel_size);
  if (!std::isfinite(leaf) ||
      count > std::numeric_limits<std::uint32_t>::max() / stride)
    return assemble_raw();
  auto packed = pcl::make_shared<pcl::PCLPointCloud2>();
  for (const char* name : {"x", "y", "z", "intensity"}) {
    pcl::PCLPointField field;
    field.name = name;
    field.offset = packed->fields.size() * sizeof(float);
    field.count = 1;
    field.datatype = pcl::PCLPointField::FLOAT32;
    packed->fields.push_back(field);
  }
  packed->point_step = stride;
  packed->height = 1;
  packed->is_dense = true;
  packed->data.resize(count * stride);
  Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d high = -low;
  size_t finite = 0;
  for (const auto& frame : keyframes) {
    if (!frame || !frame->cloud || frame->cloud->empty()) continue;
    PointCloudT transformed;
    pcl::transformPointCloud(*frame->cloud, transformed,
                             frame->pose_optimized.matrix().cast<float>());
    for (const auto& point : transformed) {
      const Eigen::Vector3d position(point.x, point.y, point.z);
      if (!position.allFinite()) continue;
      low = low.cwiseMin(position);
      high = high.cwiseMax(position);
      const float fields[] = {point.x, point.y, point.z, point.intensity};
      std::memcpy(packed->data.data() + finite * stride, fields, stride);
      ++finite;
    }
  }
  long double cells = 1;
  for (int axis = 0; axis < 3; ++axis)
    cells *= std::floor((high[axis] - low[axis]) / leaf) + 1.L;
  if (finite == 0 || !std::isfinite(cells) || cells <= 0 ||
      cells > std::numeric_limits<int>::max()) {
    // Preserve the established no-filter fallback for empty/unsafe grids,
    // including non-finite points. Release scratch before assembling it.
    packed.reset();
    return assemble_raw();
  }
  packed->data.resize(finite * stride);
  packed->width = finite;
  packed->row_step = finite * stride;
  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
  voxel.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf),
                    static_cast<float>(leaf));
  voxel.setInputCloud(packed);
  pcl::PCLPointCloud2 filtered;
  voxel.filter(filtered);
  auto output = pcl::make_shared<PointCloudT>();
  pcl::fromPCLPointCloud2(filtered, *output);
  return output;
}
} // namespace n3mapping
