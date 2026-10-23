#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace n3mapping {

struct LogOptions {
  std::string directory;
  std::string legacy_directory;
  std::uint64_t total_bytes = 100000000;
  std::size_t queue_bytes = 2 * 1024 * 1024;
  std::size_t record_bytes = 64 * 1024;
  std::chrono::milliseconds batch_interval{200};
  static LogOptions fromEnvironment();
};

struct LogStatistics {
  std::uint64_t accepted = 0, written = 0, dropped = 0, dropped_critical = 0;
  std::uint64_t oversize = 0, write_failures = 0, lost_on_write = 0;
  std::uint64_t batches = 0, bytes_written = 0, retained_bytes = 0;
  std::uint64_t queue_bytes = 0, peak_queue_bytes = 0, aggregated = 0;
  std::uint64_t recovered_bytes = 0;
  std::string last_failed_path;
  bool storage_ready = false;
  std::string json() const;
};

// One process owner and one disk worker. Producers never access the filesystem.
class AsyncLog {
public:
  explicit AsyncLog(LogOptions options);
  ~AsyncLog();
  AsyncLog(const AsyncLog&) = delete;
  AsyncLog& operator=(const AsyncLog&) = delete;
  static AsyncLog& instance();
  bool append(const std::string& path, std::string line, bool critical = false);
  // Bounded in-memory counters; the worker emits one summary per interval.
  void count(const std::string& event, double duration_ms = 0.0);
  LogStatistics statistics() const;
  // Lifecycle/test barrier only. Never call from a live callback.
  bool drain(std::chrono::milliseconds timeout = std::chrono::seconds(5));
  void stop();
  const std::string& directory() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Keep existing JSONL destinations while sharing their instance-wide budget.
bool appendLogLine(const std::string& path, std::string line, bool critical = false);
void logEvent(const std::string& event, const std::string& detail = {}, bool critical = false);
class LogOperation {
public:
  LogOperation(std::string operation, std::string target);
  ~LogOperation();
  void succeed() { succeeded_ = true; }
private:
  std::string operation_, target_;
  std::chrono::steady_clock::time_point start_;
  bool succeeded_ = false;
};
// Install glog forwarding before ROS starts; remove it after ROS shuts down.
class ProcessLogging {
public:
  explicit ProcessLogging(const char* program);
  ~ProcessLogging();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void forwardRuntimeLog(int severity, const char* name, const std::string& message);

} // namespace n3mapping
