#include "log_store.h"
#include <json/json.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace n3mapping {
namespace fs = std::filesystem;
namespace {
bool writeFile(const std::string& path, const std::string& data, bool append) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW |
      O_NONBLOCK | (append ? O_APPEND : 0), 0600);
  if (fd < 0) return false;
  struct stat st{};
  bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1;
  if (ok && !append) ok = ::ftruncate(fd, 0) == 0;
  std::size_t offset = 0;
  while (ok && offset < data.size()) {
    const auto n = ::write(fd, data.data() + offset, data.size() - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { ok = false; break; }
    offset += static_cast<std::size_t>(n);
  }
  if (!ok && append && S_ISREG(st.st_mode)) (void)::ftruncate(fd, st.st_size);
  if (::close(fd) != 0) ok = false;
  return ok;
}
bool legacyName(const std::string& path) {
  const auto name = fs::path(path).filename().string();
  if (name.rfind("n3mapping_node.", 0) != 0) return false;
  for (const auto* level : {"INFO", "WARNING", "ERROR", "FATAL"}) {
    const auto token = std::string(".log.") + level + ".";
    const auto position = name.find(token);
    if (position == std::string::npos) continue;
    const auto suffix = name.substr(position + token.size());
    if (!suffix.empty() && suffix.find_first_not_of("0123456789.-") == std::string::npos) return true;
  }
  return false;
}
bool ownedPath(const std::string& path) {
  return path.size() <= 1024 && fs::path(path).is_absolute() &&
      (fs::path(path).extension() == ".log" || fs::path(path).extension() == ".jsonl" || legacyName(path));
}
std::string compact(const Json::Value& value) {
  Json::StreamWriterBuilder writer; writer["indentation"] = "";
  return Json::writeString(writer, value) + "\n";
}
}
LogStore::LogStore(const LogOptions& options) : options_(options) { ready_ = initialize(); }
LogStore::~LogStore() { if (lock_fd_ >= 0) ::close(lock_fd_); }
bool LogStore::initialize() {
  try {
    options_.directory = fs::absolute(options_.directory).lexically_normal().string();
    fs::create_directories(options_.directory);
    lock_fd_ = ::open((options_.directory + "/.lock").c_str(),
        O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (lock_fd_ < 0 || ::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) return false;
    const auto index = options_.directory + "/catalog.json";
    if (fs::exists(index)) {
      if (fs::is_symlink(index) || fs::file_size(index) > 262144) return false;
      std::ifstream input(index); Json::Value value; input >> value;
      if (value["schema"] != "n3mapping_log_catalog_v1" || !value["paths"].isArray() ||
          value["paths"].size() > 32) return false;
      for (const auto& path : value["paths"]) {
        if (!path.isString() || !ownedPath(path.asString())) return false;
        if (std::find(paths_.begin(), paths_.end(), path.asString()) == paths_.end())
          paths_.push_back(path.asString());
      }
    }
    if (!enrollLegacy()) return false;
    for (const auto& path : paths_) {
      if (!repairTail(path)) return false;
      repaired_paths_.insert(path);
    }
    return trim(0);
  } catch (...) { return false; }
}
bool LogStore::enrollLegacy() {
  if (options_.legacy_directory.empty() || !fs::exists(options_.legacy_directory)) return true;
  std::vector<File> legacy;
  for (const auto& entry : fs::directory_iterator(options_.legacy_directory)) {
    const auto path = fs::absolute(entry.path()).lexically_normal().string();
    struct stat st{};
    if (ownedPath(path) && legacyName(path) && ::lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
        st.st_nlink == 1 && std::find(paths_.begin(), paths_.end(), path) == paths_.end())
      legacy.push_back({path, static_cast<std::uint64_t>(st.st_size), static_cast<std::int64_t>(st.st_mtime)});
  }
  std::sort(legacy.begin(), legacy.end(), [](const File& a, const File& b) { return a.time < b.time; });
  // Reserve room for active streams; only recognized n3mapping glog files may expire.
  while (!legacy.empty() && legacy.size() + paths_.size() > 24) {
    fs::remove(legacy.front().path); legacy.erase(legacy.begin());
  }
  for (const auto& entry : legacy) paths_.push_back(entry.path);
  return legacy.empty() || (catalog() && trim(0));
}
bool LogStore::repairTail(const std::string& path) {
  struct stat st{};
  if (::lstat(path.c_str(), &st) != 0) return errno == ENOENT;
  if (!S_ISREG(st.st_mode) || st.st_nlink != 1) return false;
  if (st.st_size == 0) return true;
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0) return false;
  const auto size = std::min<std::int64_t>(st.st_size, 65536);
  std::string tail(static_cast<std::size_t>(size), '\0');
  const auto n = ::pread(fd, tail.data(), tail.size(), st.st_size - size);
  bool ok = n == size;
  if (ok && tail.back() != '\n') {
    const auto newline = tail.rfind('\n');
    const auto keep = newline == std::string::npos ? 0 : st.st_size - size + newline + 1;
    ok = ::ftruncate(fd, keep) == 0;
    if (ok) recovered_bytes_ += st.st_size - keep;
  }
  ::close(fd); return ok;
}
bool LogStore::catalog() {
  Json::Value value; value["schema"] = "n3mapping_log_catalog_v1";
  value["paths"] = Json::Value(Json::arrayValue);
  for (const auto& path : paths_) value["paths"].append(path);
  const auto path = options_.directory + "/catalog.json";
  const auto data = compact(value);
  if (data.size() > 262144 || !writeFile(path + ".tmp", data, false)) return false;
  std::error_code error; fs::rename(path + ".tmp", path, error); return !error;
}
std::vector<LogStore::File> LogStore::files() const {
  std::vector<File> result;
  for (const auto& path : paths_) {
    const fs::path base(path); std::error_code error;
    fs::directory_iterator it(base.parent_path(), error), end;
    if (error) {
      if (error == std::errc::no_such_file_or_directory || error == std::errc::not_a_directory) continue;
      throw fs::filesystem_error("log inventory", base.parent_path(), error);
    }
    const auto prefix = base.filename().string() + ".n3m.";
    for (; it != end; it.increment(error)) {
      if (error) throw fs::filesystem_error("log inventory", base.parent_path(), error);
      const auto name = it->path().filename().string();
      const auto suffix = name.size() > prefix.size() ? name.substr(prefix.size()) : "";
      const bool archive = name.compare(0, prefix.size(), prefix) == 0 &&
          !suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos;
      if (it->path() != base && !archive) continue;
      struct stat st{};
      if (::lstat(it->path().c_str(), &st) == 0 && S_ISREG(st.st_mode))
        result.push_back({it->path().string(), static_cast<std::uint64_t>(st.st_size),
          static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec});
    }
    if (error) throw fs::filesystem_error("log inventory", base.parent_path(), error);
  }
  return result;
}
std::uint64_t LogStore::retainedBytes() const {
  std::uint64_t size = 0, metadata = 0;
  std::set<std::string> registered;
  for (const auto& file : files())
    if (registered.insert(file.path).second) size += file.size;
  // Unknown files consume the budget but may never be deleted by retention.
  for (const auto& entry : fs::directory_iterator(options_.directory)) {
    if (!entry.is_regular_file()) continue;
    const auto name = entry.path().filename().string();
    if (name == ".lock" || name == "catalog.json" || name == "catalog.json.tmp" ||
        name == "logger_health.json" || name == "logger_health.json.tmp")
      metadata += entry.file_size();
    else if (!registered.count(entry.path().string())) size += entry.file_size();
  }
  return size + std::max(metadata, metadata_reserve);
}
bool LogStore::trim(std::uint64_t incoming) {
  auto entries = files();
  auto retained = retainedBytes();
  std::sort(entries.begin(), entries.end(), [](const File& a, const File& b) {
    return a.time < b.time;
  });
  for (const auto& file : entries) {
    if (retained + incoming <= options_.total_bytes) return true;
    std::error_code error;
    if (!fs::remove(file.path, error) || error) return false;
    retained -= file.size;
  }
  return retained + incoming <= options_.total_bytes;
}
bool LogStore::registerPath(const std::string& path) {
  if (std::find(paths_.begin(), paths_.end(), path) != paths_.end()) return true;
  if (paths_.size() == 32) {
    // Retention may discard a stream's old files, but never forget live files.
    const auto entries = files();
    for (const auto& file : entries)
      if (file.path == paths_.front() || file.path.rfind(paths_.front() + ".n3m.", 0) == 0) {
        std::error_code error; fs::remove(file.path, error); if (error) return false;
      }
    repaired_paths_.erase(paths_.front());
    paths_.erase(paths_.begin());
  }
  paths_.push_back(path);
  if (catalog()) return true;
  paths_.pop_back();
  return false;
}
bool LogStore::append(const std::string& requested, const std::string& data) {
  try {
    if (!ready_) return false;
    const auto path = fs::absolute(requested).lexically_normal().string();
    if (!ownedPath(path) || data.size() > options_.total_bytes - metadata_reserve) return false;
    fs::create_directories(fs::path(path).parent_path());
    struct stat existing{};
    if (::lstat(path.c_str(), &existing) == 0 &&
        (!S_ISREG(existing.st_mode) || existing.st_nlink != 1)) return false;
    if (!registerPath(path)) return false;
    if (!repaired_paths_.count(path)) {
      if (!repairTail(path)) return false;
      repaired_paths_.insert(path);
    }
    const auto segment = std::min<std::uint64_t>(10000000, (options_.total_bytes - metadata_reserve) / 4);
    std::error_code error;
    if (fs::exists(path) && !fs::is_symlink(path) && fs::is_regular_file(path) &&
        fs::file_size(path) + data.size() > segment) {
      auto suffix = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      const auto rotated = path + ".n3m." + std::to_string(suffix + sequence_++);
      fs::rename(path, rotated, error); if (error) return false;
    }
    if (!trim(data.size())) return false;
    const bool written = writeFile(path, data, true);
    if (!written) repaired_paths_.erase(path);
    return written;
  } catch (...) { return false; }
}
bool LogStore::health(const std::string& json) {
  if (!ready_ || json.size() > 8192) return false;
  const auto path = options_.directory + "/logger_health.json";
  if (!writeFile(path + ".tmp", json + "\n", false)) return false;
  std::error_code error; fs::rename(path + ".tmp", path, error); return !error;
}
} // namespace n3mapping
