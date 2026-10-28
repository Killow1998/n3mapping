#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace n3mapping::core {

// Admission and backend consumption may run on separate callback lanes.
// Keep the complete payload
// together so dropping queued work cannot mix a cloud with another frame's pose.
template <typename Frame>
class LatestFrameSlot {
 public:
  void submit(int64_t stamp_nsec, Frame frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_ || stamp_nsec >= pending_stamp_nsec_) {
      pending_ = std::move(frame);
      pending_stamp_nsec_ = stamp_nsec;
    }
  }

  std::optional<Frame> take() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto frame = std::move(pending_);
    pending_.reset();
    return frame;
  }

 private:
  std::mutex mutex_;
  int64_t pending_stamp_nsec_ = 0;
  std::optional<Frame> pending_;
};

}  // namespace n3mapping::core
