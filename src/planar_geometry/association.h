// Geometry association shared by the asynchronous backend and research audit.
#pragma once
#include "moment_factor.h"
#include "n3mapping/keyframe.h"
#include <array>
#include <gtsam/geometry/Pose3.h>
#include <json/json.h>
#include <map>
#include <vector>
namespace n3mapping::planar_geometry {
using Cell = std::array<int, 3>;
struct Patch {
  int index;
  Moment moment;
  Eigen::Vector3d center, normal;
  double variance;
};
struct Plane {
  Eigen::Vector3d normal, origin;
  double offset, sigma;
  std::vector<Patch> patches;
};
double median(std::vector<double> values);
std::vector<Plane> associate(const std::vector<Keyframe::Ptr>& keyframes,
                             const std::vector<gtsam::Pose3> &poses,
                             Json::Value &audit);
std::vector<Plane> mergePlanes(std::vector<Plane> input,
                               const std::vector<gtsam::Pose3> &poses,
                               Json::Value &audit);
std::vector<std::pair<size_t, Moment>> associateToPlanes(
    const std::vector<Moment>& moments, const gtsam::Pose3& pose,
    const std::vector<SharedPlane>& planes,
    const std::map<int64_t, Eigen::Isometry3d>& reference_poses);
} // namespace n3mapping::planar_geometry
