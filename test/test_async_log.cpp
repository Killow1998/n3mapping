#include "n3mapping/async_log.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>

namespace n3mapping {
namespace {
struct Logs {
  std::filesystem::path root;
  Logs() {
    auto pattern = (std::filesystem::temp_directory_path() / "n3m-logs-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
    const char* made = ::mkdtemp(name.data()); if (!made) throw std::runtime_error("mkdtemp");
    root = made;
  }
  ~Logs() { std::error_code error; std::filesystem::remove_all(root, error); }
  LogOptions options() const { LogOptions o; o.directory = (root / "control").string(); return o; }
  std::string path(const char* name) const { return (root / "data" / name).string(); }
  std::uint64_t bytes() const {
    std::uint64_t result = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
      if (entry.is_regular_file()) result += entry.file_size();
    return result;
  }
};
std::string read(const std::string& path) {
  std::ifstream input(path); return {std::istreambuf_iterator<char>(input), {}};
}
TEST(AsyncLogTest, BoundedFloodPreservesReservedCriticalEventAndBatchesWrites) {
  Logs fixture; auto options = fixture.options(); options.queue_bytes = 16384;
  options.record_bytes = 4096; options.batch_interval = std::chrono::seconds(1);
  AsyncLog log(options);
  std::vector<double> latencies; latencies.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    const auto start = std::chrono::steady_clock::now();
    log.append(fixture.path("flood.log"), std::string(128, 'x'));
    latencies.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
  }
  ASSERT_TRUE(log.append(fixture.path("critical.log"), "critical_after_flood", true));
  for (int i = 0; i < 10000; ++i) log.count("live", 0.25);
  ASSERT_TRUE(log.drain());
  const auto stats = log.statistics();
  EXPECT_GT(stats.dropped, 0u); EXPECT_EQ(stats.dropped_critical, 0u);
  EXPECT_LE(stats.peak_queue_bytes, options.queue_bytes);
  EXPECT_EQ(stats.accepted, stats.written); EXPECT_EQ(stats.write_failures, 0u);
  EXPECT_LT(stats.batches, stats.written / 2);
  EXPECT_EQ(stats.aggregated, 10000u);
  EXPECT_NE(read(fixture.path("critical.log")).find("critical_after_flood"), std::string::npos);
  EXPECT_NE(read(options.directory + "/events.jsonl").find("aggregate"), std::string::npos);
  std::sort(latencies.begin(), latencies.end());
  RecordProperty("append_p99_us", std::to_string(latencies[latencies.size() * 99 / 100]));
  RecordProperty("append_max_us", std::to_string(latencies.back()));
}
TEST(AsyncLogTest, AllDestinationsAndArchivesShareBudgetAcrossRestart) {
  Logs fixture; auto options = fixture.options(); options.total_bytes = 1048576;
  for (int run = 0; run < 3; ++run) {
    AsyncLog log(options);
    for (int batch = 0; batch < 30; ++batch) {
      for (int i = 0; i < 24; ++i)
        ASSERT_TRUE(log.append(fixture.path(i % 2 ? "first.jsonl" : "second.log"), std::string(4000, 'a' + run)));
      ASSERT_TRUE(log.drain());
      EXPECT_LE(fixture.bytes(), options.total_bytes);
    }
    EXPECT_EQ(log.statistics().write_failures, 0u);
    log.stop(); EXPECT_LE(fixture.bytes(), options.total_bytes);
  }
  EXPECT_NE(read(options.directory + "/catalog.json").find("first.jsonl"), std::string::npos);
}
TEST(AsyncLogTest, FailedWriteIsObservableAndRecoveryDoesNotBlockOtherEvents) {
  Logs fixture; auto options = fixture.options(); AsyncLog log(options);
  std::filesystem::create_directories(fixture.path("blocked.log"));
  ASSERT_TRUE(log.append(fixture.path("blocked.log"), "cannot_write", true));
  ASSERT_TRUE(log.append(fixture.path("healthy.log"), "still_running", true));
  ASSERT_TRUE(log.drain());
  const auto failed = log.statistics();
  EXPECT_EQ(failed.lost_on_write, 1u); EXPECT_GT(failed.write_failures, 0u);
  EXPECT_NE(read(options.directory + "/logger_health.json").find("write_failures"), std::string::npos);
  EXPECT_NE(read(fixture.path("healthy.log")).find("still_running"), std::string::npos);
  std::filesystem::remove(fixture.path("blocked.log"));
  ASSERT_TRUE(log.append(fixture.path("blocked.log"), "recovered", true));
  ASSERT_TRUE(log.drain());
  EXPECT_EQ(log.statistics().write_failures, failed.write_failures);
  EXPECT_NE(read(fixture.path("blocked.log")).find("recovered"), std::string::npos);
}
TEST(AsyncLogTest, LegacyNodeLogsShareRetentionButUnrelatedFilesArePreserved) {
  Logs fixture; auto options = fixture.options(); options.total_bytes = 1048576;
  options.legacy_directory = (fixture.root / "old").string();
  std::filesystem::create_directories(options.legacy_directory);
  const auto legacy = options.legacy_directory + "/n3mapping_node.host.user.log.INFO.20261023-000000.123";
  { std::ofstream file(legacy); file << std::string(900000, 'x') << '\n'; }
  const auto other = options.legacy_directory + "/other_service.log";
  { std::ofstream file(other); file << "untouched"; }
  AsyncLog log(options);
  ASSERT_TRUE(log.append(fixture.path("new.log"), "new_instance")); ASSERT_TRUE(log.drain());
  EXPECT_LE(fixture.bytes(), options.total_bytes);
  EXPECT_EQ(read(other), "untouched"); EXPECT_EQ(log.statistics().write_failures, 0u);
}
TEST(AsyncLogTest, RestartRepairsIncompleteRecordBeforeAppending) {
  Logs fixture; auto options = fixture.options();
  { AsyncLog log(options); ASSERT_TRUE(log.append(fixture.path("crash.jsonl"), "{\"complete\":true}")); ASSERT_TRUE(log.drain()); }
  { std::ofstream file(fixture.path("crash.jsonl"), std::ios::app); file << "{\"partial\":"; }
  AsyncLog restarted(options);
  ASSERT_TRUE(restarted.append(fixture.path("crash.jsonl"), "{\"next\":true}"));
  ASSERT_TRUE(restarted.drain());
  EXPECT_EQ(read(fixture.path("crash.jsonl")), "{\"complete\":true}\n{\"next\":true}\n");
  EXPECT_GT(restarted.statistics().recovered_bytes, 0u);
}
TEST(AsyncLogTest, UnknownArchiveNamesConsumeQuotaWithoutBeingDeleted) {
  Logs fixture; auto options = fixture.options(); options.total_bytes = 1048576;
  const auto path = options.directory + "/events.jsonl.n3m.unrelated";
  std::filesystem::create_directories(options.directory);
  { std::ofstream file(path); file << std::string(800000, 'u'); }
  AsyncLog log(options);
  ASSERT_TRUE(log.append(fixture.path("new.log"), "does_not_fit")); ASSERT_TRUE(log.drain());
  EXPECT_FALSE(log.statistics().storage_ready);
  EXPECT_GT(log.statistics().write_failures, 0u);
  EXPECT_EQ(std::filesystem::file_size(path), 800000u);
  EXPECT_LE(fixture.bytes(), options.total_bytes);
}
TEST(AsyncLogTest, ConcurrentOwnerCannotWriteOrRotateTheFirstOwnersLogs) {
  Logs fixture; auto options = fixture.options(); AsyncLog owner(options);
  ASSERT_TRUE(owner.append(fixture.path("owner.log"), "first")); ASSERT_TRUE(owner.drain());
  {
    AsyncLog contender(options);
    ASSERT_TRUE(contender.append(fixture.path("owner.log"), "must_not_write"));
    ASSERT_TRUE(contender.drain());
    EXPECT_FALSE(contender.statistics().storage_ready);
    EXPECT_GT(contender.statistics().write_failures, 0u);
  }
  ASSERT_TRUE(owner.append(fixture.path("owner.log"), "still_owner")); ASSERT_TRUE(owner.drain());
  EXPECT_EQ(read(fixture.path("owner.log")), "first\nstill_owner\n");
  EXPECT_EQ(owner.statistics().write_failures, 0u);
}
TEST(AsyncLogTest, OversizeAndUnsafeDestinationsAreCountedWithoutChangingTheirTargets) {
  Logs fixture; auto options = fixture.options(); AsyncLog log(options);
  std::filesystem::create_directories(fixture.root / "data");
  { std::ofstream file(fixture.path("original.log")); file << "keep"; }
  std::filesystem::create_symlink(fixture.path("original.log"), fixture.path("alias.log"));
  ASSERT_FALSE(log.append(fixture.path("oversize.log"), std::string(65536, 'x'), true));
  std::string excess_capacity = "short"; excess_capacity.reserve(options.queue_bytes * 2);
  ASSERT_FALSE(log.append(fixture.path("capacity.log"), std::move(excess_capacity)));
  ASSERT_TRUE(log.append(fixture.path("alias.log"), "overwrite", true));
  ASSERT_TRUE(log.drain());
  EXPECT_EQ(read(fixture.path("original.log")), "keep");
  EXPECT_EQ(log.statistics().oversize, 1u); EXPECT_EQ(log.statistics().dropped_critical, 1u);
  EXPECT_GT(log.statistics().write_failures, 0u);
}
} // namespace
} // namespace n3mapping
