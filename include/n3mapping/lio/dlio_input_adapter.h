// ROS-free input packet builder for the future DLIO core.
#pragma once

#include <vector>

#include "n3mapping/core/types.h"
#include "n3mapping/lio/dlio_cloud_adapter.h"
#include "n3mapping/lio/imu_sample_buffer.h"

namespace n3mapping {
namespace lio {
namespace dlio {

struct InputPacket {
    core::TimeStamp stamp_begin;
    core::TimeStamp stamp_end;
    PointCloud::Ptr cloud;
    std::vector<core::ImuSample> imu_samples;
    CloudAdapterStats cloud_stats;
    TimeEncoding time_encoding = TimeEncoding::OusterOffsetNs;
    bool has_complete_imu_window = false;
};

InputPacket buildInputPacket(const core::RawLidarFrame& frame,
                             const ImuSampleBuffer& imu_buffer,
                             const CloudAdapterOptions& options = {});

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
