#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <vector>
using namespace n3mapping;
double cpuMs() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec * 1000. + t.tv_nsec / 1e6;
}
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "MAP OUTPUT_CSV QUERY_PCD...\n";
    return 2;
  }
  Config config;
  KeyframeManager frames(config);
  LoopDetector detector(config);
  GraphOptimizer graph(config);
  MapSerializer serializer(config);
  if (!serializer.loadMap(argv[1], frames, detector, graph))
    return 3;
  const auto database = detector.getDescriptors();
  std::vector<n3mapping::RHPDHeadingPtr> queries;
  for (int i = 3; i < argc; ++i) {
    auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
    if (pcl::io::loadPCDFile(argv[i], *cloud) != 0)
      return 4;
    for (double degrees : {0., 42., 137., 180.}) {
      auto rotated = pcl::make_shared<Keyframe::PointCloudT>();
      Eigen::Affine3f pose = Eigen::Affine3f::Identity();
      pose.linear() =
          Eigen::AngleAxisf(degrees * M_PI / 180., Eigen::Vector3f::UnitZ())
              .toRotationMatrix();
      pcl::transformPointCloud(*cloud, *rotated, pose);
      queries.push_back(detector.computeHeading(rotated));
    }
  }
  std::ofstream out(argv[2]);
  if (!out)
    return 5;
  out << std::setprecision(17) << "query,keyframe,distance,shift\n";
  double cpu = 0;
  for (size_t i = 0; i < queries.size(); ++i)
    for (const auto &[id, reference] : database) {
      const auto begin = cpuMs();
      const auto result = detector.computeDistance(queries[i], reference);
      cpu += cpuMs() - begin;
      out << i << ',' << id << ',' << result.first << ',' << result.second
          << '\n';
    }
  std::cerr << "MATCH_CPU_MS=" << cpu
            << " pairs=" << queries.size() * database.size() << '\n';
}
