#pragma once
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace n3mapping {
// One isolated job at a time, on a reusable thread.
class BackgroundWorker {
public:
  ~BackgroundWorker() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  template <class Function>
  std::future<void> submit(Function&& function) {
    std::packaged_task<void()> next(std::forward<Function>(function));
    auto result = next.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_ || task_.valid()) throw std::logic_error("Background worker already queued");
      if (!thread_.joinable()) thread_ = std::thread([this] { run(); });
      task_ = std::move(next);
    }
    ready_.notify_one();
    return result;
  }

private:
  void run() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || task_.valid(); });
        if (!task_.valid()) return;
        task = std::move(task_);
      }
      task();  // Exceptions are delivered by the existing future.get() path.
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
  std::packaged_task<void()> task_;
  std::thread thread_;
};

} // namespace n3mapping
