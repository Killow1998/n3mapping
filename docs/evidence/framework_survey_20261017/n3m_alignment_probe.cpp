#include <pcl/io/pcd_io.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "n3mapping/point_cloud_matcher.h"

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  auto target_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  if (pcl::io::loadPCDFile(argv[1], *target_cloud) != 0) return 3;
  auto source_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  std::ifstream input(argv[2], std::ios::binary);
  double point[3];
  while (input.read(reinterpret_cast<char*>(point), sizeof(point))) {
    pcl::PointXYZI p;
    p.x = point[0];
    p.y = point[1];
    p.z = point[2];
    p.intensity = 0;
    source_cloud->push_back(p);
  }
  if (source_cloud->empty()) return 4;
  auto config = n3mapping::makeProductLocalizationConfig(argv[1], "");
  // Preparation is identical in both processes, even when registration uses
  // four threads. Upstream parallel voxel collection can reorder input points.
  config.num_threads = 1;
  n3mapping::PointCloudMatcher matcher(config);
  const auto target = matcher.prepareTargetCloud(target_cloud);
  const auto source = matcher.prepareSourceCloud(source_cloud);
  auto setting = matcher.getSettings();
  setting.num_threads = std::stoi(argv[3]);
  std::cout << std::setprecision(17);
  for (const double offset : {0.0, 0.1, 1.0, 5.0}) {
    Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
    initial.linear() =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    initial.translation() = Eigen::Vector3d(
        8.470738410949707 + offset, -6.751821041107178, 1.2010440826416016);
    const auto warm = matcher.alignPrepared(target, source, initial, setting);
    n3mapping::MatchResult result;
    int status_changes = 0;
    double max_pose_change = 0.0;
    constexpr int repeats = 50;
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start = std::clock();
    for (int i = 0; i < repeats; ++i) {
      result = matcher.alignPrepared(target, source, initial, setting);
      status_changes += result.success != warm.success ||
                        result.converged != warm.converged ||
                        result.num_inliers != warm.num_inliers ||
                        result.iterations != warm.iterations;
      max_pose_change = std::max(
          max_pose_change,
          (result.T_target_source.matrix() - warm.T_target_source.matrix())
              .cwiseAbs()
              .maxCoeff());
    }
    const auto cpu_end = std::clock();
    const auto end = std::chrono::steady_clock::now();
    std::cout
        << "{\"offset\":" << offset << ",\"threads\":" << setting.num_threads
        << ",\"batch_size\":" << repeats
        << ",\"status_changes\":" << status_changes
        << ",\"within_batch_pose_delta\":" << max_pose_change << ",\"cpu_ms\":"
        << 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC / repeats
        << ",\"wall_ms\":"
        << std::chrono::duration<double, std::milli>(end - start).count() /
               repeats
        << ",\"success\":" << result.success
        << ",\"converged\":" << result.converged
        << ",\"inliers\":" << result.num_inliers
        << ",\"iterations\":" << result.iterations
        << ",\"error\":" << result.optimizer_error
        << ",\"fitness\":" << result.fitness_score << ",\"pose\":[";
    for (int i = 0; i < 16; ++i)
      std::cout << (i ? "," : "") << result.T_target_source.matrix().data()[i];
    std::cout << "],\"stages\":[";
    for (std::size_t i = 0; i < result.stages.size(); ++i) {
      const auto& s = result.stages[i];
      std::cout << (i ? "," : "") << "{\"name\":\"" << s.stage
                << "\",\"iterations\":" << s.iterations
                << ",\"inliers\":" << s.num_inliers
                << ",\"converged\":" << s.converged
                << ",\"error\":" << s.optimizer_error << "}";
    }
    std::cout << "]}\n";
  }
}
