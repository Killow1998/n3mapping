#include <chrono>
#include <json/json.h>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <sys/resource.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/realtime_output.h"

using Clock = std::chrono::steady_clock;
double cpuMs() {
  timespec t{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
double wallMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "MAP ATLAS MANIFEST OUTPUT_CSV RATE MAX_FRAMES\n";
    return 2;
  }
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  auto config = n3mapping::makeProductLocalizationConfig(argv[1], argv[2]);
  if (const auto* threads = std::getenv("N3M_BENCH_THREADS"))
    config.num_threads = std::stoi(threads);
  if (const auto* debug = std::getenv("N3M_BENCH_DEBUG")) {
    config.reloc_debug_enable = true;
    config.reloc_debug_path = debug;
    FLAGS_v = 1;
  }
  n3mapping::N3MappingCore core(config,
      n3mapping::N3MappingCore::WorldCloudOutput::Disabled);
  const auto loading = Clock::now();
  const auto load_cpu = cpuMs();
  if (!core.loadMap(argv[1])) return 3;
  std::cerr << "MAP_READY wall_ms=" << wallMs(loading)
            << " cpu_ms=" << cpuMs() - load_cpu << '\n';
  if (config.reloc_debug_enable) {
    std::ofstream keyframes(config.reloc_debug_path + ".keyframes.csv");
    keyframes << std::setprecision(17) << "id,x,y,z\n";
    for (const auto& keyframe : core.getAllKeyframes()) {
      const auto& p = keyframe->pose_optimized.translation();
      keyframes << keyframe->id << ',' << p.x() << ',' << p.y() << ',' << p.z() << '\n';
    }
  }
  n3mapping::core::RealtimeOutput live(
      n3mapping::CoreRunMode::LOCALIZATION, "map", config.sync_queue_size);
  std::ifstream manifest(argv[3]);
  std::string row;
  std::getline(manifest, row);
  if (row.find("episode_id,frame_index,stamp_ns,pcd_path,tx,ty,tz,qx,qy,qz,qw") != 0)
    return 4;
  std::ofstream output(argv[4]);
  output << std::setprecision(17);
  std::ofstream costs(std::string(argv[4]) + ".costs.jsonl");
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  output << "frame,stamp_ns,points,success,lock,state,pose_source,decision,seed,support,matched,live,wall_ms,cpu_ms,cumulative_wall_ms,cumulative_cpu_ms,late_ms,source_ms,target_ms,submap_ms,tracking_ms,registration_ms,visibility_ms,reloc_source_ms,reloc_align_ms,reloc_visibility_ms,cache_hit,cache_miss,candidates,alignments,rss_kib";
  for (int i = 0; i < 16; ++i) output << ",pose" << i;
  for (int i = 0; i < 16; ++i) output << ",output_pose" << i;
  output << '\n';
  const double rate = std::stod(argv[5]);
  const int max_frames = std::stoi(argv[6]);
  const auto start = Clock::now();
  const double start_cpu = cpuMs();
  int index = 0;
  int64_t first_stamp = 0;
  while (std::getline(manifest, row) && (max_frames <= 0 || index < max_frames)) {
    std::istringstream fields(row);
    std::vector<std::string> f;
    std::string value;
    while (std::getline(fields, value, ',')) f.push_back(value);
    if (f.size() != 11) return 5;
    n3mapping::core::LioFrame frame;
    frame.stamp.nsec = std::stoll(f[2]);
    if (index == 0) first_stamp = frame.stamp.nsec;
    const double elapsed_sensor_ms = (frame.stamp.nsec - first_stamp) / 1e6;
    const auto due = start + std::chrono::nanoseconds(
        rate > 0 ? static_cast<int64_t>((frame.stamp.nsec - first_stamp) / rate) : 0);
    if (rate > 0) std::this_thread::sleep_until(due);
    const double late = rate > 0 ? wallMs(due) : 0.0;
    const auto frame_start = Clock::now();
    const double frame_cpu = cpuMs();
    auto cloud = std::make_shared<n3mapping::core::LioFrame::PointCloud>();
    auto path = std::filesystem::path(f[3]);
    if (path.is_relative()) path = std::filesystem::path(argv[3]).parent_path() / path;
    if (pcl::io::loadPCDFile(path.string(), *cloud) != 0) return 6;
    frame.undistorted_cloud = cloud;
    frame.pose_valid = true;
    frame.T_world_lidar.translation() = Eigen::Vector3d(
        std::stod(f[4]), std::stod(f[5]), std::stod(f[6]));
    Eigen::Quaterniond q(std::stod(f[10]), std::stod(f[7]),
                         std::stod(f[8]), std::stod(f[9]));
    frame.T_world_lidar.linear() = q.normalized().toRotationMatrix();
    auto result = core.processLocalizationFrame(frame);
    n3mapping::core::OdometrySource source;
    source.stamp_nsec = frame.stamp.nsec;
    source.pose = frame.T_world_lidar;
    source.source_frame = "camera_init";
    source.body_frame = "body";
    live.acceptBackend(source, frame.stamp.nsec / 1e9, result);
    auto emitted = live.observeOdometry(source, frame.stamp.nsec, true);
    const double process_ms = wallMs(frame_start);
    const double process_cpu = cpuMs() - frame_cpu;
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const auto& t = result.tracking_performance;
    const auto& r = result.relocalization_performance;
    output << index << ',' << frame.stamp.nsec << ',' << cloud->size()
           << ',' << result.success << ',' << result.relocalization_locked
           << ',' << static_cast<int>(result.relocalization_state)
           << ',' << static_cast<int>(result.pose_source)
           << ',' << result.relocalization_decision
           << ',' << result.relocalization_seed_keyframe_id
           << ',' << result.relocalization_support_keyframe_id
           << ',' << result.matched_keyframe_id << ',' << emitted.pose.has_value()
           << ',' << process_ms << ',' << process_cpu << ',' << wallMs(start)
           << ',' << cpuMs() - start_cpu << ',' << late
           << ',' << t.source_prepare_ms << ',' << t.target_prepare_ms
           << ',' << t.submap_build_ms << ',' << t.tracking_total_ms
           << ',' << t.registration_ms << ',' << t.visibility_ms
           << ',' << r.source_prepare_ms << ',' << r.align_ms << ',' << r.visibility_ms
           << ',' << t.localization_target_cache_hit << ',' << t.localization_target_cache_miss
           << ',' << r.candidates << ',' << r.alignments << ',' << usage.ru_maxrss;
    for (int i = 0; i < 16; ++i) output << ',' << result.T_world_lidar.matrix().data()[i];
    for (int i = 0; i < 16; ++i)
      output << ',' << (emitted.pose ? emitted.pose->pose.matrix().data()[i]
                                    : std::numeric_limits<double>::quiet_NaN());
    output << '\n';
    output.flush();
    if (r.available) {
      auto record = n3mapping::relocalizationPerformanceJson(r);
      record["frame"] = index;
      costs << Json::writeString(writer, record) << '\n';
      costs.flush();
    }
    if (index % 100 == 0)
      std::cerr << "FRAME " << index << " state=" << static_cast<int>(result.relocalization_state)
                << " sensor_ms=" << elapsed_sensor_ms << " cpu_ms=" << process_cpu
                << " wall_ms=" << process_ms << '\n';
    ++index;
  }
  std::cerr << "COMPLETE frames=" << index << " wall_ms=" << wallMs(start)
            << " cpu_ms=" << cpuMs() - start_cpu << '\n';
}
