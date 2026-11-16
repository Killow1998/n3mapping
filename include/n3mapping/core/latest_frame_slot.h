#pragma once

#include <cstdint>
#include <optional>
#include <utility>

namespace n3mapping::core {

template <typename Frame>
class LatestFrameSlot {
 public:
  void submit(int64_t stamp_nsec, Frame frame) {
    if (!pending_ || stamp_nsec >= pending_stamp_nsec_) {
      pending_ = std::move(frame);
      pending_stamp_nsec_ = stamp_nsec;
    }
  }

  std::optional<Frame> take() {
    auto frame = std::move(pending_);
    pending_.reset();
    return frame;
  }

 private:
  int64_t pending_stamp_nsec_ = 0;
  std::optional<Frame> pending_;
};

}
