// ROS-free IMU sample buffer shared by in-process LIO frontends.
#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {

class ImuSampleBuffer {
public:
    explicit ImuSampleBuffer(size_t max_samples = 2000);

    void add(const core::ImuSample& sample);
    void clear();

    bool empty() const { return samples_.empty(); }
    size_t size() const { return samples_.size(); }
    size_t maxSamples() const { return max_samples_; }

    std::optional<core::ImuSample> latest() const;
    std::vector<core::ImuSample> samplesInRange(int64_t start_nsec,
                                                int64_t end_nsec) const;

private:
    size_t max_samples_;
    std::deque<core::ImuSample> samples_;
};

}  // namespace lio
}  // namespace n3mapping
