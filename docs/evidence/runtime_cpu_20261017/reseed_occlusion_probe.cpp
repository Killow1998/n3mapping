// Experimental use of the existing occlusion-aware observation model.
#include "n3mapping/visibility_consistency.h"
#include "n3mapping/config.h"
namespace n3mapping {
VisibilityConsistencyOptions visibilityOptionsFromConfig(const Config& config) {
  VisibilityConsistencyOptions options;
  options.range_min_m = 0.5;
  options.range_max_m = std::max(1.0, config.rhpd_max_range);
  options.range_tolerance_m = std::max(0.05, 3.0 * std::max(
      config.global_map_voxel_size, config.gicp_downsampling_resolution));
  options.occlusion_aware = true;
  return options;
}
}
