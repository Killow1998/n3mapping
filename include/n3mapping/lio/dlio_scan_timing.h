// ROS-free DLIO scan timing extraction from raw lidar point offsets.
#pragma once

#include <vector>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {
namespace dlio {

struct ScanTiming {
    double stamp_begin = 0.0;
    double stamp_end = 0.0;
    double stamp_median = 0.0;
    std::vector<double> unique_point_timestamps;
    bool has_point_timing = false;
    bool valid = false;
};

ScanTiming computeScanTiming(const core::RawLidarFrame& frame);

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
