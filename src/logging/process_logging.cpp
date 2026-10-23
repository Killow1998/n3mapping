#include "n3mapping/async_log.h"
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <string>

namespace n3mapping {
namespace {
thread_local bool fatal_message = false;
class GlogForwarder final : public google::LogSink {
  void send(google::LogSeverity severity, const char*, const char* file, int line,
            const struct tm*, const char* message, std::size_t size) override {
    fatal_message = severity == google::GLOG_FATAL;
    try {
      if (size > 32768) AsyncLog::instance().count("glog_message_truncated");
      forwardRuntimeLog(severity >= google::GLOG_ERROR ? 40 : severity >= google::GLOG_WARNING ? 30 : 20,
          file, std::string(message, std::min<std::size_t>(size, 32768)) + " line=" + std::to_string(line));
    } catch (...) { }
  }
  void WaitTillSent() override {
    if (fatal_message) AsyncLog::instance().drain(std::chrono::milliseconds(200));
    fatal_message = false;
  }
};
}
struct ProcessLogging::Impl { GlogForwarder sink; bool owns_glog = false; };
ProcessLogging::ProcessLogging(const char* program) : impl_(new Impl) {
  (void)AsyncLog::instance();
  google::InitGoogleLogging(program); impl_->owns_glog = true;
  FLAGS_logtostderr = false; FLAGS_alsologtostderr = false;
  FLAGS_stderrthreshold = google::GLOG_FATAL;
  for (int level = 0; level < google::NUM_SEVERITIES; ++level) {
    google::SetLogDestination(level, ""); google::SetLogSymlink(level, "");
  }
  google::AddLogSink(&impl_->sink);
  logEvent("process_start", program, true);
}
ProcessLogging::~ProcessLogging() {
  logEvent("process_stop", {}, true);
  google::RemoveLogSink(&impl_->sink);
  if (impl_->owns_glog) google::ShutdownGoogleLogging();
  AsyncLog::instance().stop();
}
void forwardRuntimeLog(int severity, const char* name, const std::string& message) {
  auto& log = AsyncLog::instance();
  // Warnings/errors retain their content; ordinary repeated messages are also
  // bounded by the shared queue and disk budget.
  const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  log.append(log.directory() + "/runtime.log", std::to_string(time) + " severity=" +
      std::to_string(severity) + " source=" + (name ? name : "unknown") + " " + message,
      severity >= 30);
}
} // namespace n3mapping
