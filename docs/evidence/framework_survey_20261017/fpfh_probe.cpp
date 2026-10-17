#include <sys/resource.h>
#include <tbb/global_control.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <kiss_matcher/FasterPFH.hpp>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                            std::stoi(argv[2]));
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<Eigen::Vector3f> points;
  double p[3];
  while (f.read(reinterpret_cast<char*>(p), sizeof(p)))
    points.emplace_back(p[0], p[1], p[2]);
  if (points.empty()) return 3;
  kiss_matcher::FasterPFH feature(1.5f, 2.5f, .99f);
  std::vector<Eigen::Vector3f> keypoints;
  std::vector<Eigen::VectorXf> descriptors;
  const auto start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();
  feature.setInputCloud(points);
  feature.ComputeFeature(keypoints, descriptors);
  auto end = std::chrono::steady_clock::now();
  auto cpu_end = std::clock();
  bool finite = true;
  for (const auto& d : descriptors) finite = finite && d.allFinite();
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "RESULT {\"points\":" << points.size()
            << ",\"keypoints\":" << keypoints.size()
            << ",\"descriptors\":" << descriptors.size()
            << ",\"finite\":" << (finite ? "true" : "false") << ",\"wall_ms\":"
            << std::chrono::duration<double, std::milli>(end - start).count()
            << ",\"cpu_ms\":" << 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC
            << ",\"rss_kib\":" << usage.ru_maxrss << "}\n";
}
