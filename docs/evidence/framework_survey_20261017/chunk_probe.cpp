#include <mcl_3dl/chunked_kdtree.h>
#include <pcl/io/pcd_io.h>
#include <sys/resource.h>

#include <chrono>
#include <iostream>
#include <random>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPCDFile(argv[1], *cloud) != 0) return 3;
  bool chunked = std::string(argv[2]) == "chunk";
  pcl::KdTreeFLANN<pcl::PointXYZ> full;
  mcl_3dl::ChunkedKdtree<pcl::PointXYZ> chunk(20, 1.5);
  auto start = std::chrono::steady_clock::now();
  if (chunked)
    chunk.setInputCloud(cloud);
  else
    full.setInputCloud(cloud);
  auto built = std::chrono::steady_clock::now();
  std::mt19937 rng(3588);
  std::uniform_real_distribution<float> x(-30, 30), y(-40, 30), z(-4, 12);
  std::vector<pcl::PointXYZ> queries;
  for (int i = 0; i < 20000; ++i) queries.emplace_back(x(rng), y(rng), z(rng));
  std::vector<int> ids;
  std::vector<float> distances;
  int matched = 0;
  double sum = 0;
  std::vector<double> all;
  auto qs = std::chrono::steady_clock::now();
  for (const auto& q : queries) {
    ids.clear();
    distances.clear();
    int n = chunked ? chunk.radiusSearch(q, 1.5, ids, distances, 1)
                    : full.radiusSearch(q, 1.5, ids, distances, 1);
    matched += n > 0;
    sum += n ? distances[0] : 0;
    all.push_back(n ? distances[0] : -1);
  }
  auto end = std::chrono::steady_clock::now();
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  int mismatches = 0;
  if (chunked) {
    full.setInputCloud(cloud);
    for (size_t i = 0; i < queries.size(); ++i) {
      int n = full.radiusSearch(queries[i], 1.5, ids, distances, 1);
      double d = n ? distances[0] : -1;
      mismatches += std::abs(d - all[i]) > 1e-6;
    }
  }
  std::cout << "{\"chunked\":" << (chunked ? "true" : "false")
            << ",\"build_ms\":"
            << std::chrono::duration<double, std::milli>(built - start).count()
            << ",\"query_ms\":"
            << std::chrono::duration<double, std::milli>(end - qs).count()
            << ",\"matched\":" << matched << ",\"distance_sum\":" << sum
            << ",\"mismatches\":" << mismatches
            << ",\"rss_kib\":" << usage.ru_maxrss << "}\n";
}
