#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

#include <rclcpp/rclcpp.hpp>

namespace n3mapping {

// A serial ROS callback lane with its own executor. Explicit association keeps
// the node's backend executor from claiming this group's subscriptions.
class CallbackWorker {
  friend class HumbleOutputTestAccess;
 public:
  explicit CallbackWorker(rclcpp::Node& node)
      : group_(node.create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive, false)),
        executor_(std::make_unique<rclcpp::executors::SingleThreadedExecutor>()) {
    executor_->add_callback_group(group_, node.get_node_base_interface());
  }
  ~CallbackWorker() { stop(); }
  CallbackWorker(const CallbackWorker&) = delete;
  CallbackWorker& operator=(const CallbackWorker&) = delete;

  rclcpp::SubscriptionOptions options() const {
    rclcpp::SubscriptionOptions options;
    options.callback_group = group_;
    return options;
  }
  void start() {
    if (!thread_.joinable()) {
      stopping_ = false;
      thread_ = std::thread([this] {
        while (!stopping_ && rclcpp::ok()) {
          executor_->spin_once(std::chrono::milliseconds(50));
        }
      });
    }
  }
  void stop() {
    stopping_ = true;
    executor_->cancel();
    if (thread_.joinable()) thread_.join();
  }

 private:
  rclcpp::CallbackGroup::SharedPtr group_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread thread_;
  std::atomic<bool> stopping_{false};
};

}  // namespace n3mapping
