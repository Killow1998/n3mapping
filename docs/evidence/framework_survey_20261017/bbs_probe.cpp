#include <sys/resource.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cpu_bbs3d/bbs3d.hpp>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

std::vector<Eigen::Vector3d> load(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing input");
  std::vector<Eigen::Vector3d> points;
  double xyz[3];
  while (in.read(reinterpret_cast<char*>(xyz), sizeof(xyz)))
    points.emplace_back(xyz[0], xyz[1], xyz[2]);
  if (points.empty()) throw std::runtime_error("empty input");
  return points;
}
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  auto map = load(argv[1]), query = load(argv[2]);
  int threads = std::stoi(argv[3]), timeout_ms = std::stoi(argv[4]);
  cpu::BBS3D bbs;
  bbs.set_num_threads(threads);
  const auto start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();
  bbs.set_tar_points(map, 0.5, 6);
  auto build_end = std::chrono::steady_clock::now();
  auto cpu_build_end = std::clock();
  bbs.set_trans_search_range(map);
  bbs.set_angular_search_range(Eigen::Vector3d(0, 0, 0),
                               Eigen::Vector3d(0, 0, 2 * M_PI));
  bbs.set_src_points(query);
  bbs.set_score_threshold_percentage(0.6);
  bbs.set_timeout_duration_in_msec(timeout_ms);
  bbs.enable_timeout();
  bbs.localize();
  auto end = std::chrono::steady_clock::now();
  auto cpu_end = std::clock();
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout
      << std::setprecision(12) << "RESULT {\"points\":" << map.size()
      << ",\"query_points\":" << query.size() << ",\"threads\":" << threads
      << ",\"build_ms\":"
      << std::chrono::duration<double, std::milli>(build_end - start).count()
      << ",\"build_cpu_ms\":"
      << 1000.0 * (cpu_build_end - cpu_start) / CLOCKS_PER_SEC
      << ",\"query_ms\":"
      << std::chrono::duration<double, std::milli>(end - build_end).count()
      << ",\"query_cpu_ms\":"
      << 1000.0 * (cpu_end - cpu_build_end) / CLOCKS_PER_SEC
      << ",\"rss_kib\":" << usage.ru_maxrss
      << ",\"localized\":" << (bbs.has_localized() ? "true" : "false")
      << ",\"timeout\":" << (bbs.has_timed_out() ? "true" : "false")
      << ",\"score\":" << bbs.get_best_score_percentage();
  if (bbs.has_localized()) {
    const auto T = bbs.get_global_pose();
    std::cout << ",\"xyz\":[" << T(0, 3) << ',' << T(1, 3) << ',' << T(2, 3)
              << "],\"yaw\":" << std::atan2(T(1, 0), T(0, 0));
  }
  std::cout << "}" << std::endl;
}
