#include "n3mapping/lio/dlio_input_adapter.h"

#include <algorithm>

namespace n3mapping {
namespace lio {
namespace dlio {

InputPacket buildInputPacket(const core::RawLidarFrame& frame,
                             const ImuSampleBuffer& imu_buffer,
                             const CloudAdapterOptions& options) {
    InputPacket packet;
    packet.stamp_begin = frame.stamp_begin;
    packet.stamp_end = frame.stamp_end;

    const auto converted = cloudFromRawLidar(frame, options);
    packet.cloud = converted.cloud;
    packet.cloud_stats = converted.stats;
    packet.time_encoding = converted.resolved_time_encoding;

    const int64_t begin_nsec = std::min(frame.stamp_begin.nsec, frame.stamp_end.nsec);
    const int64_t end_nsec = std::max(frame.stamp_begin.nsec, frame.stamp_end.nsec);
    packet.imu_samples = imu_buffer.samplesInRange(begin_nsec, end_nsec);
    packet.has_complete_imu_window =
        !packet.imu_samples.empty() &&
        packet.imu_samples.front().stamp.nsec <= begin_nsec &&
        packet.imu_samples.back().stamp.nsec >= end_nsec;
    return packet;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
