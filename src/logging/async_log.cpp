#include "n3mapping/async_log.h"
#include "log_store.h"
#include <json/json.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace n3mapping {
namespace {
using Clock = std::chrono::steady_clock;
std::string compact(const Json::Value& value) {
  Json::StreamWriterBuilder writer; writer["indentation"] = "";
  return Json::writeString(writer, value);
}
std::uint64_t environmentLimit(const char* name, std::uint64_t fallback) {
  const char* text = std::getenv(name);
  if (!text || !*text) return fallback;
  std::size_t used = 0; const auto value = std::stoull(text, &used);
  if (text[used] || text[0] == '-') throw std::invalid_argument(name);
  return value;
}
}
LogOptions LogOptions::fromEnvironment() {
  LogOptions options;
  if (const auto* path = std::getenv("N3MAPPING_LOG_DIR")) options.directory = path;
  if (options.directory.empty()) {
    const char* ros_home = std::getenv("ROS_HOME");
    const char* user_home = std::getenv("HOME");
    options.directory = (ros_home && *ros_home ? std::string(ros_home) :
        std::string(user_home ? user_home : ".") + "/.ros") + "/n3mapping/logs";
  }
  if (const auto* legacy = std::getenv("GLOG_log_dir")) options.legacy_directory = legacy;
#ifdef N3MAPPING_SOURCE_DIR
  else if (!std::getenv("N3MAPPING_LOG_DIR")) options.legacy_directory = std::string(N3MAPPING_SOURCE_DIR) + "/logs";
#endif
  options.total_bytes = environmentLimit("N3MAPPING_LOG_TOTAL_BYTES", options.total_bytes);
  return options;
}
std::string LogStatistics::json() const {
  Json::Value value; value["schema"] = "n3mapping_logging_status_v1";
#define FIELD(name) value[#name] = Json::UInt64(name)
  FIELD(accepted); FIELD(written); FIELD(dropped); FIELD(dropped_critical);
  FIELD(oversize); FIELD(write_failures); FIELD(lost_on_write); FIELD(batches);
  FIELD(bytes_written); FIELD(retained_bytes); FIELD(queue_bytes);
  FIELD(peak_queue_bytes); FIELD(aggregated); FIELD(recovered_bytes);
#undef FIELD
  value["last_failed_path"] = last_failed_path;
  value["storage_ready"] = storage_ready;
  return compact(value);
}
struct AsyncLog::Impl {
  struct Record { std::string path, line; };
  struct Counter { std::uint64_t count = 0; double sum = 0, maximum = 0; };
  LogOptions options;
  mutable std::mutex mutex;
  std::condition_variable changed, completed;
  std::deque<Record> queue;
  std::map<std::string, Counter> counters;
  std::size_t queue_bytes = 0;
  bool stopping = false, force = false, urgent = false;
  std::uint64_t finished = 0, drain_requested = 0, drain_completed = 0;
  std::thread worker;
#define ATOMIC(name) std::atomic<std::uint64_t> name{0}
  ATOMIC(accepted); ATOMIC(written); ATOMIC(dropped); ATOMIC(dropped_critical);
  ATOMIC(oversize); ATOMIC(write_failures); ATOMIC(lost_on_write); ATOMIC(batches);
  ATOMIC(bytes_written); ATOMIC(retained_bytes); ATOMIC(peak_queue_bytes); ATOMIC(aggregated);
#undef ATOMIC
  std::atomic<std::uint64_t> recovered_bytes{0};
  mutable std::mutex failure_mutex;
  std::string last_failed_path;
  std::atomic<bool> storage_ready{false};
  std::atomic<std::uint64_t> queued{0};
  explicit Impl(LogOptions input) : options(std::move(input)) {
    if (options.directory.empty() || options.total_bytes < 1048576 || options.total_bytes > 100000000 ||
        options.queue_bytes < 1024 || options.record_bytes == 0 ||
        options.record_bytes > 65536 || options.batch_interval.count() < 20)
      throw std::invalid_argument("Invalid bounded logging options");
    worker = std::thread([this] { run(); });
  }
  LogStatistics statistics() const {
    LogStatistics s;
#define COPY(name) s.name = name.load(std::memory_order_relaxed)
    COPY(accepted); COPY(written); COPY(dropped); COPY(dropped_critical); COPY(oversize);
    COPY(write_failures); COPY(lost_on_write); COPY(batches); COPY(bytes_written);
    COPY(retained_bytes); COPY(peak_queue_bytes); COPY(aggregated); COPY(recovered_bytes);
#undef COPY
    { std::lock_guard<std::mutex> lock(failure_mutex); s.last_failed_path = last_failed_path; }
    s.queue_bytes = queued.load(); s.storage_ready = storage_ready.load(); return s;
  }
  void failedWrite(const std::string& path, std::uint64_t records = 0) {
    ++write_failures; lost_on_write += records;
    std::lock_guard<std::mutex> lock(failure_mutex); last_failed_path = path;
  }
  void run() {
    LogStore store(options);
    storage_ready = store.ready();
    if (!store.ready()) failedWrite(options.directory);
    auto health_due = Clock::now();
    std::uint64_t reported_drops = 0, reported_failures = 0;
    auto next_batch = Clock::now() + options.batch_interval;
    while (true) {
      std::deque<Record> pending; std::map<std::string, Counter> summaries;
      bool stop = false; std::uint64_t barrier = 0; bool flush = false;
      {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait_until(lock, next_batch, [this] { return stopping || force || urgent; });
        // Error bursts may wake promptly, but cannot turn every record into I/O.
        if (urgent && !stopping && !force && Clock::now() < next_batch) {
          urgent = false;
          changed.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopping || force; });
        }
        barrier = drain_requested; flush = force;
        pending.swap(queue); summaries.swap(counters);
        queue_bytes = 0; queued = 0; force = false; urgent = false; stop = stopping;
      }
      const auto count = pending.size();
      std::size_t accounted = 0;
      try {
        // Coalesce complete records by destination, with bounded write batches.
        struct Batch { std::string data; std::size_t records = 0; };
        std::map<std::string, Batch> output;
        auto write = [&](const std::string& path, Batch& batch) {
          if (batch.data.empty()) return;
          ++batches;
          if (store.append(path, batch.data)) {
            written += batch.records; bytes_written += batch.data.size();
          } else {
            failedWrite(path, batch.records);
          }
          accounted += batch.records;
          batch = {};
        };
        for (auto& record : pending) {
          auto& batch = output[record.path];
          if (batch.data.size() + record.line.size() + 1 > 65536) write(record.path, batch);
          batch.data += record.line; batch.data += '\n'; ++batch.records;
        }
        for (auto& item : output) write(item.first, item.second);
        if (!summaries.empty()) {
          Json::Value value; value["schema"] = "n3mapping_events_v1";
          value["event"] = "aggregate";
          for (const auto& item : summaries) {
            auto& counter = value["counters"][item.first];
            counter["count"] = Json::UInt64(item.second.count);
            counter["sum_ms"] = item.second.sum; counter["max_ms"] = item.second.maximum;
          }
          const auto line = compact(value) + "\n";
          ++batches;
          if (store.append(options.directory + "/events.jsonl", line)) bytes_written += line.size();
          else failedWrite(options.directory + "/events.jsonl");
        }
        if (Clock::now() >= health_due || stop || flush) {
          recovered_bytes = store.recoveredBytes();
          if (dropped.load() != reported_drops || write_failures.load() != reported_failures || stop) {
            auto record = statistics().json(); record.pop_back();
            record += ",\"event\":\"logger_health\"}\n";
            if (!store.append(options.directory + "/events.jsonl", record)) failedWrite(options.directory + "/events.jsonl");
            else { ++batches; bytes_written += record.size(); }
            reported_drops = dropped.load(); reported_failures = write_failures.load();
          }
          retained_bytes = store.retainedBytes();
          storage_ready = store.ready();
          if (!store.health(statistics().json())) failedWrite(options.directory + "/logger_health.json");
          health_due = Clock::now() + std::chrono::seconds(1);
        }
      } catch (...) { failedWrite(options.directory, count - accounted); }
      {
        std::lock_guard<std::mutex> lock(mutex); finished += count; drain_completed = barrier;
        completed.notify_all();
      }
      if (stop) break;
      next_batch = Clock::now() + options.batch_interval;
    }
  }
};
AsyncLog::AsyncLog(LogOptions options) : impl_(new Impl(std::move(options))) {}
AsyncLog::~AsyncLog() { stop(); }
AsyncLog& AsyncLog::instance() { static AsyncLog log(LogOptions::fromEnvironment()); return log; }
const std::string& AsyncLog::directory() const { return impl_->options.directory; }
bool AsyncLog::append(const std::string& path, std::string line, bool critical) {
  auto& p = *impl_;
  auto reject = [&] { ++p.dropped; if (critical) ++p.dropped_critical; return false; };
  if (line.size() + 1 > p.options.record_bytes || path.size() > 1024) { ++p.oversize; return reject(); }
  Impl::Record record{path, std::move(line)};
  const auto size = sizeof(record) + record.path.capacity() + record.line.capacity();
  std::unique_lock<std::mutex> lock(p.mutex);
  if (p.stopping) return reject();
  const auto limit = critical ? p.options.queue_bytes : p.options.queue_bytes * 3 / 4;
  if (p.queue_bytes + size > limit) return reject();
  p.queue.push_back(std::move(record));
  p.queue_bytes += size; p.queued = p.queue_bytes;
  p.peak_queue_bytes = std::max(p.peak_queue_bytes.load(), static_cast<std::uint64_t>(p.queue_bytes));
  ++p.accepted;
  if (critical) { p.urgent = true; p.changed.notify_one(); }
  return true;
}
void AsyncLog::count(const std::string& event, double duration_ms) {
  auto& p = *impl_; std::unique_lock<std::mutex> lock(p.mutex);
  if (p.stopping || event.size() > 128) { ++p.dropped; return; }
  if (p.counters.size() == 64 && !p.counters.count(event)) { ++p.dropped; return; }
  auto& counter = p.counters[event]; ++counter.count; ++p.aggregated;
  if (std::isfinite(duration_ms)) { counter.sum += duration_ms; counter.maximum = std::max(counter.maximum, duration_ms); }
}
LogStatistics AsyncLog::statistics() const { return impl_->statistics(); }
bool AsyncLog::drain(std::chrono::milliseconds timeout) {
  auto& p = *impl_; std::unique_lock<std::mutex> lock(p.mutex);
  const auto target = p.accepted.load(); const auto barrier = ++p.drain_requested; p.force = true; p.changed.notify_one();
  return p.completed.wait_for(lock, timeout, [&] { return p.finished >= target && p.drain_completed >= barrier; });
}
void AsyncLog::stop() {
  if (!impl_ || !impl_->worker.joinable()) return;
  { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->stopping = true; impl_->changed.notify_one(); }
  impl_->worker.join();
}
bool appendLogLine(const std::string& path, std::string line, bool critical) {
  return AsyncLog::instance().append(path, std::move(line), critical);
}
void logEvent(const std::string& event, const std::string& detail, bool critical) {
  Json::Value value; value["schema"] = "n3mapping_events_v1"; value["event"] = event;
  value["detail"] = detail;
  value["time_ns"] = Json::Int64(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  auto& log = AsyncLog::instance(); log.append(log.directory() + "/events.jsonl", compact(value), critical);
}
LogOperation::LogOperation(std::string operation, std::string target)
    : operation_(std::move(operation)), target_(std::move(target)), start_(Clock::now()) {
  logEvent(operation_ + "_begin", target_, true);
}
LogOperation::~LogOperation() {
  try {
    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    logEvent(operation_ + (succeeded_ ? "_success" : "_failure"),
        target_ + " elapsed_ms=" + std::to_string(ms), true);
  } catch (...) { }
}
} // namespace n3mapping
