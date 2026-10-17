#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/features2d.hpp>

#include "DensityMap.hpp"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<Eigen::Vector3d> points;
  double p[3];
  while (f.read(reinterpret_cast<char*>(p), sizeof(p)))
    points.emplace_back(p[0], p[1], p[2]);
  if (points.empty()) return 3;
  auto start = std::chrono::steady_clock::now();
  auto map = map_closures::GenerateDensityMap(
      points, Eigen::Matrix4d::Identity(), .5, .05);
  auto end = std::chrono::steady_clock::now();
  for (auto& p : points) p.z() += 5.0;
  auto shifted = map_closures::GenerateDensityMap(
      points, Eigen::Matrix4d::Identity(), .5, .05);
  auto orb =
      cv::ORB::create(500, 1.0f, 1, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 35);
  std::vector<cv::KeyPoint> kp;
  cv::Mat descriptors;
  orb->detectAndCompute(map.grid, cv::noArray(), kp, descriptors);
  std::cout << "{\"points\":" << points.size() << ",\"density_ms\":"
            << std::chrono::duration<double, std::milli>(end - start).count()
            << ",\"width\":" << map.grid.cols << ",\"height\":" << map.grid.rows
            << ",\"orb_keypoints\":" << kp.size()
            << ",\"vertical_5m_shift_identical\":"
            << (cv::countNonZero(map.grid != shifted.grid) == 0 ? "true"
                                                                : "false")
            << "}\n";
}
