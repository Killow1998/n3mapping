#include "n3mapping/lio/dlio_scan_timing.h"

#include <algorithm>
#include <cstdint>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

double secondsFromNsec(int64_t nsec) {
    return static_cast<double>(nsec) * 1.0e-9;
}

double pointStampSeconds(const core::RawLidarFrame& frame, size_t index) {
    const uint32_t offset_ns =
        index < frame.point_time_offsets_ns.size()
            ? frame.point_time_offsets_ns[index]
            : 0u;
    return secondsFromNsec(frame.stamp_begin.nsec) +
           static_cast<double>(offset_ns) * 1.0e-9;
}

}  // namespace

ScanTiming computeScanTiming(const core::RawLidarFrame& frame) {
    ScanTiming timing;
    timing.stamp_begin = secondsFromNsec(frame.stamp_begin.nsec);
    timing.stamp_end = secondsFromNsec(frame.stamp_end.nsec);
    if (timing.stamp_end < timing.stamp_begin) {
        std::swap(timing.stamp_begin, timing.stamp_end);
    }

    if (!frame.points || frame.points->empty()) {
        timing.stamp_median = 0.5 * (timing.stamp_begin + timing.stamp_end);
        return timing;
    }

    timing.valid = true;
    timing.has_point_timing = !frame.point_time_offsets_ns.empty();
    if (!timing.has_point_timing) {
        timing.stamp_median = 0.5 * (timing.stamp_begin + timing.stamp_end);
        timing.unique_point_timestamps.push_back(timing.stamp_median);
        return timing;
    }

    timing.unique_point_timestamps.reserve(frame.points->size());
    for (size_t i = 0; i < frame.points->size(); ++i) {
        timing.unique_point_timestamps.push_back(pointStampSeconds(frame, i));
    }
    std::sort(timing.unique_point_timestamps.begin(),
              timing.unique_point_timestamps.end());
    timing.unique_point_timestamps.erase(
        std::unique(timing.unique_point_timestamps.begin(),
                    timing.unique_point_timestamps.end()),
        timing.unique_point_timestamps.end());

    if (!timing.unique_point_timestamps.empty()) {
        timing.stamp_begin =
            std::min(timing.stamp_begin, timing.unique_point_timestamps.front());
        timing.stamp_end =
            std::max(timing.stamp_end, timing.unique_point_timestamps.back());
        timing.stamp_median =
            timing.unique_point_timestamps[timing.unique_point_timestamps.size() / 2];
    }
    return timing;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
