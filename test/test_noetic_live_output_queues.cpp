#include <gtest/gtest.h>

#include <ros/callback_queue.h>

#include <functional>
#include <future>
#include <thread>

#include "n3mapping/core/realtime_odometry.h"

namespace {

class Callback final : public ros::CallbackInterface {
 public:
  explicit Callback(std::function<void()> function) : function_(std::move(function)) {}
  CallResult call() override {
    function_();
    return Success;
  }
 private:
  std::function<void()> function_;
};

// Exercise the real ROS callback queues without init, master, topics or services.
// The wrapper guard checks that production wires these same distinct queues.
bool odomCompletesBeforeCloudRelease(bool separate_queues) {
  ros::CallbackQueue cloud_queue;
  ros::CallbackQueue odom_queue;
  n3mapping::core::RealtimeOdometry live(4);
  const auto pose = Eigen::Isometry3d::Identity();
  live.updateCorrection(90, "odom", "body", pose, pose, true);
  std::promise<void> cloud_started;
  std::promise<void> release_cloud;
  auto release = release_cloud.get_future();
  std::promise<bool> odom_completed;
  auto completion = odom_completed.get_future();
  cloud_queue.addCallback(ros::CallbackInterfacePtr(new Callback([&] {
    const auto projection = live.projectScan(100, 100, "odom", "body", pose);
    cloud_started.set_value();
    release.wait();
    EXPECT_TRUE(projection);
  })));
  auto& destination = separate_queues ? odom_queue : cloud_queue;
  destination.addCallback(ros::CallbackInterfacePtr(new Callback([&] {
    odom_completed.set_value(live.project(101, 101, "odom", "body", pose).has_value());
  })));
  std::thread cloud_worker([&] { cloud_queue.callAvailable(); });
  cloud_started.get_future().wait();
  std::thread odom_worker;
  if (separate_queues) odom_worker = std::thread([&] { odom_queue.callAvailable(); });
  const bool completed = completion.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  release_cloud.set_value();
  cloud_worker.join();
  if (odom_worker.joinable()) odom_worker.join();
  EXPECT_TRUE(completion.get());
  return completed;
}

TEST(NoeticLiveOutputQueues, SlowCloudDoesNotDelayIndependentOdomQueue) {
  EXPECT_TRUE(odomCompletesBeforeCloudRelease(true));
}

TEST(NoeticLiveOutputQueues, OriginalSingleQueueAblationDoesDelayOdom) {
  EXPECT_FALSE(odomCompletesBeforeCloudRelease(false));
}

}  // namespace
