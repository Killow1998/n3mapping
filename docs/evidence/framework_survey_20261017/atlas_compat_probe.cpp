#include <pcl/io/pcd_io.h>

#include <iostream>

#include "n3mapping/localization_atlas.h"
int main(int argc, char** argv) {
  if (argc != 4) return 2;
  auto cfg = n3mapping::makeProductLocalizationConfig(argv[1], argv[2]);
  cfg.num_threads = 1;
  n3mapping::PointCloudMatcher matcher(cfg);
  n3mapping::LocalizationAtlas atlas(cfg, matcher);
  std::string error;
  bool ok;
  if (std::string(argv[3]) == "write") {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (pcl::io::loadPCDFile(argv[1], *cloud) != 0) return 3;
    std::vector<n3mapping::Keyframe::Ptr> frames{n3mapping::Keyframe::create(
        0, 0, Eigen::Isometry3d::Identity(), cloud)};
    ok = atlas.compileAndSave(argv[1], frames, argv[2], false, nullptr, &error);
  } else {
    ok = atlas.load(argv[1], argv[2], nullptr, &error);
  }
  if (!ok) {
    std::cerr << error << '\n';
    return 4;
  }
  uint64_t fingerprint = 14695981039346656037ULL;
  auto hash = [&](const void* ptr, size_t n) {
    auto b = static_cast<const unsigned char*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      fingerprint ^= b[i];
      fingerprint *= 1099511628211ULL;
    }
  };
  const auto& prepared = atlas.preparedTarget();
  for (const auto& l : prepared.plane_levels) {
    for (const auto& p : l.cloud->points) hash(p.data(), sizeof(double) * 4);
    for (const auto& n : l.cloud->normals) hash(n.data(), sizeof(double) * 4);
  }
  if (prepared.has_refine_level) {
    for (const auto& p : prepared.refine_level.cloud->points)
      hash(p.data(), sizeof(double) * 4);
    for (const auto& c : prepared.refine_level.cloud->covs)
      hash(c.data(), sizeof(double) * 16);
  }
  std::cout << "{\"ok\":true,\"required_features_fingerprint\":\""
            << fingerprint << "\"}\n";
}
