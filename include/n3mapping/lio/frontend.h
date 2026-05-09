// Common C++ frontend interface for external LIO, FAST-LIO2, and DLIO adapters.
#pragma once

#include <functional>
#include <optional>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {

struct LioTimingStats {
    double preprocess_ms = 0.0;
    double odometry_ms = 0.0;
    double total_ms = 0.0;
};

struct LioDebugCallbacks {
    std::function<void(const core::LioFrame&)> odom;
    std::function<void(const core::LioFrame::PointCloud::ConstPtr&)> deskewed_cloud;
    std::function<void(const core::LioFrame::PointCloud::ConstPtr&)> local_map;
    std::function<void(const LioTimingStats&)> timing;
};

class LioFrontend {
public:
    virtual ~LioFrontend() = default;

    virtual void addImu(const core::ImuSample& imu) = 0;
    virtual std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame) = 0;
    virtual void reset() = 0;
    virtual void setDebugCallbacks(const LioDebugCallbacks& callbacks) {
        (void)callbacks;
    }
};

}  // namespace lio
}  // namespace n3mapping
