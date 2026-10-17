#include <pcl/io/pcd_io.h>
#include <sys/resource.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "n3mapping/point_cloud_matcher.h"

using Cloud = n3mapping::PointCloudMatcher::SmallGicpCloud;
std::size_t bytes(const Cloud& c) {
  return c.points.capacity() * sizeof(Eigen::Vector4d) +
         c.normals.capacity() * sizeof(Eigen::Vector4d) +
         c.covs.capacity() * sizeof(Eigen::Matrix4d);
}
uint64_t fingerprint = 14695981039346656037ULL;
void hash(const void* p, size_t n) {
  auto b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) {
    fingerprint ^= b[i];
    fingerprint *= 1099511628211ULL;
  }
}
void cloudHash(const Cloud& c, bool normals, bool covs) {
  for (const auto& p : c.points) hash(p.data(), 4 * sizeof(double));
  if (normals)
    for (const auto& n : c.normals) hash(n.data(), 4 * sizeof(double));
  if (covs)
    for (const auto& v : c.covs) hash(v.data(), 16 * sizeof(double));
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  if (pcl::io::loadPCDFile(argv[1], *cloud) != 0) return 3;
  auto cfg = n3mapping::makeProductLocalizationConfig(argv[1], "");
  cfg.num_threads = std::stoi(argv[2]);
  n3mapping::PointCloudMatcher matcher(cfg);
  const auto start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();
  auto target = matcher.prepareTargetCloud(cloud);
  auto source = matcher.prepareSourceCloud(cloud);
  const auto end = std::chrono::steady_clock::now();
  const auto cpu_end = std::clock();
  size_t payload = 0, total_points = 0;
  for (const auto& l : target.plane_levels) {
    payload += bytes(*l.cloud);
    total_points += l.cloud->size();
    cloudHash(*l.cloud, true, false);
  }
  for (const auto& l : source.plane_levels) {
    payload += bytes(*l.cloud);
    cloudHash(*l.cloud, false, false);
  }
  if (target.has_refine_level) {
    payload += bytes(*target.refine_level.cloud);
    cloudHash(*target.refine_level.cloud, false, true);
  }
  if (source.has_refine_cloud) {
    payload += bytes(*source.refine_cloud);
    cloudHash(*source.refine_cloud, false, true);
  }
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << std::setprecision(17) << "{\"input_points\":" << cloud->size()
            << ",\"plane_target_points\":" << total_points
            << ",\"payload_bytes\":" << payload
            << ",\"estimated_target_bytes\":"
            << n3mapping::estimatePreparedTargetMemoryBytes(target)
            << ",\"required_features_fingerprint\":\"" << fingerprint
            << "\",\"wall_ms\":"
            << std::chrono::duration<double, std::milli>(end - start).count()
            << ",\"cpu_ms\":" << 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC
            << ",\"rss_kib\":" << usage.ru_maxrss << "}\n";
}
