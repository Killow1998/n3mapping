#include "association.h"
#include <algorithm>
#include <cmath>
#include <map>
namespace n3mapping::planar_geometry {
double median(std::vector<double> values) {
  auto m = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), m, values.end());
  return *m;
}

std::vector<Plane> associate(const n3mapping::N3Map &map,
                             const std::vector<gtsam::Pose3> &poses,
                             Json::Value &audit) {
  std::map<Cell, std::vector<Patch>> cells;
  int raw = 0, nonplanar = 0;
  for (int i = 0; i < map.keyframes_size(); ++i) {
    std::map<Cell, Moment> scan;
    const auto &points = map.keyframes(i).cloud().points();
    for (int j = 0; j < points.size(); j += 4) {
      const Eigen::Vector3d local(points[j], points[j + 1], points[j + 2]);
      if (!local.allFinite())
        continue;
      const auto world = poses[i].transformFrom(local);
      Cell cell;
      for (int a = 0; a < 3; ++a)
        cell[a] = std::floor(world[a]);
      scan[cell].add(local);
    }
    for (auto &[cell, moment] : scan) {
      if (moment.count < 6)
        continue;
      ++raw;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment.covariance());
      const auto v = eigen.eigenvalues();
      if (v[1] < .0025 || v[0] > .0004 || v[0] > .02 * v[1]) {
        ++nonplanar;
        continue;
      }
      const auto rotation = poses[i].rotation().matrix();
      Eigen::Vector3d normal = rotation * eigen.eigenvectors().col(0);
      const auto center = poses[i].transformFrom(moment.mean());
      if (normal.dot(poses[i].translation() - center) < 0)
        normal = -normal;
      cells[cell].push_back({i, moment, center, normal, std::max(0., v[0])});
    }
  }
  std::vector<Plane> output;
  int outliers = 0, observations = 0;
  for (auto &[cell, all] : cells) {
    std::vector<std::vector<Patch>> groups;
    for (auto &patch : all) {
      bool assigned = false;
      for (auto &group : groups) {
        const auto &seed = group.front();
        if (seed.normal.dot(patch.normal) > std::cos(5 * M_PI / 180.) &&
            std::abs(seed.normal.dot(patch.center - seed.center)) < .1) {
          group.push_back(std::move(patch));
          assigned = true;
          break;
        }
      }
      if (!assigned)
        groups.push_back({std::move(patch)});
    }
    for (auto &patches : groups) {
      if (patches.size() < 3)
        continue;
      Eigen::Matrix3d normals = Eigen::Matrix3d::Zero();
      for (const auto &p : patches)
        normals.noalias() += p.normal * p.normal.transpose();
      Eigen::Vector3d normal =
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(normals)
              .eigenvectors()
              .col(2);
      Eigen::Index axis;
      normal.cwiseAbs().maxCoeff(&axis);
      if (normal[axis] < 0)
        normal = -normal;
      if (normal.dot(patches.front().normal) < 0)
        normal = -normal;
      std::vector<double> offsets;
      for (const auto &p : patches)
        offsets.push_back(normal.dot(p.center));
      const double offset = median(offsets);
      std::vector<Patch> accepted;
      std::vector<double> scatter;
      for (auto &patch : patches) {
        const double delta = normal.dot(patch.center) - offset;
        if (std::abs(normal.dot(patch.normal)) < std::cos(5 * M_PI / 180.) ||
            std::abs(delta) > .1) {
          ++outliers;
          continue;
        }
        scatter.push_back(delta * delta + patch.variance);
        accepted.push_back(std::move(patch));
      }
      if (accepted.size() < 3)
        continue;
      const double sigma = std::sqrt(.0001 + median(scatter));
      const Eigen::Vector3d origin(cell[0] + .5, cell[1] + .5, cell[2] + .5);
      observations += accepted.size();
      output.push_back({normal, origin, offset - normal.dot(origin), sigma,
                        std::move(accepted)});
    }
  }
  audit["raw_patches"] = raw;
  audit["nonplanar_patches"] = nonplanar;
  audit["association_outliers"] = outliers;
  audit["landmarks"] = int(output.size());
  audit["observations"] = observations;
  return output;
}

} // namespace n3mapping::planar_geometry
