#pragma once
#include "n3mapping/async_log.h"
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace n3mapping {
// Worker-only filesystem owner. Catalog entries are the only external files
// subject to retention. Unregistered files are never removed.
class LogStore {
public:
  explicit LogStore(const LogOptions& options);
  ~LogStore();
  bool append(const std::string& path, const std::string& data);
  bool health(const std::string& json);
  std::uint64_t retainedBytes() const;
  bool ready() const { return ready_; }
  std::uint64_t recoveredBytes() const { return recovered_bytes_; }
private:
  struct File { std::string path; std::uint64_t size; std::int64_t time; };
  bool initialize();
  bool enrollLegacy();
  bool catalog();
  bool repairTail(const std::string& path);
  bool registerPath(const std::string& path);
  bool trim(std::uint64_t incoming);
  std::vector<File> files() const;
  LogOptions options_;
  std::vector<std::string> paths_;
  std::set<std::string> repaired_paths_;
  int lock_fd_ = -1;
  bool ready_ = false;
  std::uint64_t sequence_ = 0, recovered_bytes_ = 0;
  static constexpr std::uint64_t metadata_reserve = 576 * 1024;
};
} // namespace n3mapping
