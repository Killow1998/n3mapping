#include "n3mapping/lio/imu_sample_buffer.h"

#include <algorithm>

namespace n3mapping {
namespace lio {

ImuSampleBuffer::ImuSampleBuffer(size_t max_samples)
    : max_samples_(std::max<size_t>(1, max_samples)) {}

void ImuSampleBuffer::add(const core::ImuSample& sample) {
    samples_.push_back(sample);
    while (samples_.size() > max_samples_) {
        samples_.pop_front();
    }
}

void ImuSampleBuffer::clear() {
    samples_.clear();
}

std::optional<core::ImuSample> ImuSampleBuffer::latest() const {
    if (samples_.empty()) {
        return std::nullopt;
    }
    return samples_.back();
}

std::vector<core::ImuSample> ImuSampleBuffer::samplesInRange(
    int64_t start_nsec, int64_t end_nsec) const {
    if (end_nsec < start_nsec) {
        return {};
    }

    std::vector<core::ImuSample> result;
    for (const auto& sample : samples_) {
        if (sample.stamp.nsec >= start_nsec && sample.stamp.nsec <= end_nsec) {
            result.push_back(sample);
        }
    }
    return result;
}

}  // namespace lio
}  // namespace n3mapping
