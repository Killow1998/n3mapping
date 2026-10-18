// Replay the production Humble node with input/output helpers in one process.
// N3M_DDS_REPLAY=1 exercises actual DDS subscriptions and callback workers.
// Cross-process/network transport is outside this measurement.
#define N3MAPPING_NODE_NO_MAIN
#ifndef N3M_NODE_SOURCE
#define N3M_NODE_SOURCE "../../../humble/src/n3mapping_node.cpp"
#endif
#include N3M_NODE_SOURCE
#include <sqlite3.h>
#include <rclcpp/serialization.hpp>
#include <deque>
#include <map>
#include <sys/resource.h>
#include <ctime>
#include <cstring>
#include <iomanip>

using Clock = std::chrono::steady_clock;
using Cloud = sensor_msgs::msg::PointCloud2;
using Odom = nav_msgs::msg::Odometry;
auto& pathPoses(nav_msgs::msg::Path& path) { return path.poses; }
const auto& pathPoses(const nav_msgs::msg::Path& path) { return path.poses; }
auto& pathPoses(std::vector<geometry_msgs::msg::PoseStamped>& path) { return path; }
const auto& pathPoses(const std::vector<geometry_msgs::msg::PoseStamped>& path) { return path; }
double cpuMs() {
  timespec value{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value);
  return value.tv_sec * 1000.0 + value.tv_nsec / 1e6;
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
namespace n3mapping {
class HumbleOutputTestAccess {
 public:
  static void logger(N3MappingNode& node, const std::string& directory) {
    auto config = node.config_;
    config.reloc_debug_enable = true;
    config.reloc_debug_path = directory + "/reloc.jsonl";
    node.performance_debug_logger_ = std::make_unique<RuntimePerformanceDebugLogger>(config);
  }
  static void clock(N3MappingNode& node, int64_t now) { node.authority_probe_now_nsec_ = now; }
  static void odom(N3MappingNode& node, const Odom::SharedPtr& source) {
    node.liveOdomCallback(source);
  }
  static void cloud(N3MappingNode& node, const Cloud::SharedPtr& cloud, const Odom::SharedPtr& source) {
    node.liveCloudCallback(cloud, source);
  }
  static void submit(N3MappingNode& node, const Cloud::SharedPtr& cloud, const Odom::SharedPtr& source) {
    node.syncCallback(cloud, source);
  }
  static void pathProbe(N3MappingNode& node) {
    std_msgs::msg::Header header;
    header.frame_id = "body";
    header.stamp = rclcpp::Time(1700000000000000000LL);
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header;
    pose.header.frame_id = "map";
    pose.pose.orientation.w = 1.0;
    for (const int count : {1800, 18000, 72000, 180000}) {
      pathPoses(node.localization_path_).resize(count, pose);
      const int repeats = 100;
      const auto start = Clock::now();
      const double cpu = cpuMs();
      for (int i = 0; i < repeats; ++i) node.publishPath(header, nullptr);
      std::cout << "PATH_PROBE count=" << count << " repeats=" << repeats
                << " wall_ms=" << elapsed(start) << " cpu_ms=" << cpuMs() - cpu << '\n';
    }
  }
  static double processedStamp(const N3MappingNode& node) {
    return node.last_performance_sensor_timestamp_.value_or(0.0);
  }
  static std::size_t frames(const N3MappingNode& node) { return node.frame_count_; }
  static std::size_t poses(const N3MappingNode& node) { return pathPoses(node.localization_path_).size(); }
};
}
template <typename Message>
std::shared_ptr<Message> decode(sqlite3_stmt* statement) {
  const auto size = sqlite3_column_bytes(statement, 2);
  rclcpp::SerializedMessage serialized(size);
  auto& raw = serialized.get_rcl_serialized_message();
  std::memcpy(raw.buffer, sqlite3_column_blob(statement, 2), size);
  raw.buffer_length = size;
  auto message = std::make_shared<Message>();
  rclcpp::Serialization<Message>().deserialize_message(&serialized, message.get());
  return message;
}
int main(int argc, char** argv) {
  if (argc < 5) { std::cerr << "BAG_DB OUTPUT_DIR MAX_FRAMES --ros-args ...\n"; return 2; }
  const std::string database = argv[1], directory = argv[2];
  const int limit = std::stoi(argv[3]);
  std::filesystem::create_directories(directory);
  rclcpp::init(argc - 3, argv + 3);
  using Access = n3mapping::HumbleOutputTestAccess;
  const auto load_start = Clock::now();
  auto node = std::make_shared<n3mapping::N3MappingNode>();
  std::cerr << "MAP_READY wall_ms=" << elapsed(load_start) << '\n';
  if (std::getenv("N3M_PATH_PROBE")) { Access::pathProbe(*node); node->shutdown(); rclcpp::shutdown(); return 0; }
  Access::logger(*node, directory);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  node->startLiveOutput();
  const bool dds = std::getenv("N3M_DDS_REPLAY") != nullptr;
  auto io = std::make_shared<rclcpp::Node>("n3mapping_replay_io");
  auto odom_pub = io->create_publisher<Odom>("/Odometry", 10);
  auto cloud_pub = io->create_publisher<Cloud>("/cloud_registered_body", 10);
  std::atomic<int> output_odometry{0}, output_body{0}, output_world{0}, output_paths{0}, output_maps{0};
  std::ofstream published_poses(directory + "/published_poses.csv");
  published_poses << std::setprecision(17)
                  << "topic,stamp_ns,tx,ty,tz,qx,qy,qz,qw\n";
  auto odom_sub = io->create_subscription<Odom>("/n3mapping/odometry", 100,
      [&](Odom::ConstSharedPtr message) {
        ++output_odometry;
        const auto& pose = message->pose.pose;
        published_poses << "/n3mapping/odometry,"
            << rclcpp::Time(message->header.stamp).nanoseconds() << ','
            << pose.position.x << ',' << pose.position.y << ',' << pose.position.z << ','
            << pose.orientation.x << ',' << pose.orientation.y << ','
            << pose.orientation.z << ',' << pose.orientation.w << '\n';
      });
  auto body_sub = io->create_subscription<Cloud>("/n3mapping/cloud_body", 10,
      [&](Cloud::ConstSharedPtr) { ++output_body; });
  auto world_sub = io->create_subscription<Cloud>("/n3mapping/cloud_world", 10,
      [&](Cloud::ConstSharedPtr) { ++output_world; });
  auto path_sub = io->create_subscription<nav_msgs::msg::Path>("/n3mapping/path", 10,
      [&](nav_msgs::msg::Path::ConstSharedPtr) { ++output_paths; });
  auto map_sub = io->create_subscription<Cloud>("/n3mapping/global_map", rclcpp::QoS(1).transient_local(),
      [&](Cloud::ConstSharedPtr) { ++output_maps; });
  rclcpp::executors::SingleThreadedExecutor collector;
  collector.add_node(io);
  std::atomic<bool> collecting{true};
  std::thread observer([&] { while (collecting) collector.spin_once(std::chrono::milliseconds(50)); });
  const auto discovery = Clock::now();
  while (dds && (odom_pub->get_subscription_count() < 3 || cloud_pub->get_subscription_count() < 2)) {
    if (elapsed(discovery) > 3000) throw std::runtime_error("Local DDS discovery failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  std::mutex mutex;
  using Pair = std::pair<Cloud::SharedPtr, Odom::SharedPtr>;
  std::deque<Pair> queued;
  std::atomic<bool> done{false};
  std::atomic<int> input_frames{0};
  std::atomic<int64_t> last_input_stamp{0};
  std::exception_ptr input_error;
  const auto start = Clock::now();
  const double start_cpu = cpuMs();
  std::thread input([&] {
    try {
      sqlite3* db = nullptr;
      if (sqlite3_open_v2(database.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot open bag database");
      sqlite3_stmt* statement = nullptr;
      const char* sql = "SELECT messages.timestamp,topics.name,messages.data FROM messages JOIN topics ON messages.topic_id=topics.id ORDER BY messages.timestamp";
      if (sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot read bag messages");
      std::map<int64_t, Cloud::SharedPtr> clouds;
      std::map<int64_t, Odom::SharedPtr> odometry;
      int64_t first = 0;
      while (sqlite3_step(statement) == SQLITE_ROW && (limit <= 0 || input_frames < limit)) {
        const int64_t recorded = sqlite3_column_int64(statement, 0);
        if (first == 0) first = recorded;
        std::this_thread::sleep_until(start + std::chrono::nanoseconds(recorded - first));
        const std::string topic = reinterpret_cast<const char*>(sqlite3_column_text(statement, 1));
        int64_t stamp;
        if (topic == "/Odometry") {
          auto message = decode<Odom>(statement);
          stamp = rclcpp::Time(message->header.stamp).nanoseconds();
          Access::clock(*node, stamp);
          if (dds) odom_pub->publish(*message);
          else Access::odom(*node, message);
          odometry[stamp] = std::move(message);
        } else if (topic == "/cloud_registered_body") {
          auto message = decode<Cloud>(statement);
          stamp = rclcpp::Time(message->header.stamp).nanoseconds();
          Access::clock(*node, stamp);
          if (dds) cloud_pub->publish(*message);
          clouds[stamp] = std::move(message);
        } else continue;
        auto cloud = clouds.find(stamp);
        auto odom = odometry.find(stamp);
        if (cloud != clouds.end() && odom != odometry.end()) {
          if (!dds) {
            Access::cloud(*node, cloud->second, odom->second);
            std::lock_guard<std::mutex> lock(mutex);
            queued.emplace_back(cloud->second, odom->second);
          }
          last_input_stamp = stamp;
          clouds.erase(cloud);
          odometry.erase(odom);
          ++input_frames;
        }
      }
      sqlite3_finalize(statement);
      sqlite3_close(db);
    } catch (...) { input_error = std::current_exception(); }
    done = true;
  });
  std::ofstream output(directory + "/process.csv");
  output << std::setprecision(17) << "wall_ms,cpu_ms,input_frames,processed_frames,path_poses,rss_kib\n";
  auto record = [&] {
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    output << elapsed(start) << ',' << cpuMs() - start_cpu << ',' << input_frames.load()
           << ',' << Access::frames(*node) << ',' << Access::poses(*node) << ',' << usage.ru_maxrss << '\n';
    output.flush();
  };
  std::size_t prior_frames = 0;
  while (rclcpp::ok()) {
    executor.spin_some();
    std::deque<Pair> arrivals;
    bool finished;
    { std::lock_guard<std::mutex> lock(mutex); arrivals.swap(queued); finished = done && arrivals.empty(); }
    // Keep the production single-owner frame slot on the backend thread.
    for (const auto& frame : arrivals) Access::submit(*node, frame.first, frame.second);
    node->processPendingLocalizationFrame();
    if (Access::frames(*node) != prior_frames) { record(); prior_frames = Access::frames(*node); }
    if (dds) finished = done &&
        std::abs(Access::processedStamp(*node) - last_input_stamp.load() / 1e9) < 1e-5;
    if (finished) break;
    if (done && elapsed(start) > 10000 && input_error) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  input.join();
  record();
  std::cerr << "COMPLETE wall_ms=" << elapsed(start) << " cpu_ms=" << cpuMs() - start_cpu
            << " inputs=" << input_frames << " processed=" << Access::frames(*node)
            << " poses=" << Access::poses(*node) << '\n';
  // Allow already-published messages to reach the counters after measurement.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  collecting = false;
  collector.cancel();
  observer.join();
  std::cerr << "OUTPUT odometry=" << output_odometry << " body=" << output_body
            << " world=" << output_world << " paths=" << output_paths << " maps=" << output_maps << '\n';
  node->shutdown();
  rclcpp::shutdown();
  if (input_error) std::rethrow_exception(input_error);
}
