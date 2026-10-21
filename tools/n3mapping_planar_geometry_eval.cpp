// Research-only geometry evaluation on immutable map snapshots.
#include "n3mapping/graph_factor_noise.h"
#include "../src/planar_geometry/association.h"
#include "n3mapping/n3map_proto_utils.h"
#include "planar_geometry/factor_audit.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <iomanip>
#include <iostream>
#include <tbb/global_control.h>
using namespace n3mapping::planar_geometry;
using Clock = std::chrono::steady_clock;
class ExactPoseFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  gtsam::Pose3 measured_;

public:
  ExactPoseFactor(gtsam::Key a, gtsam::Key b, const gtsam::Pose3 &z,
                  const gtsam::SharedNoiseModel &n)
      : NoiseModelFactor2(n, a, b), measured_(z) {}
  gtsam::Vector evaluateError(
      const gtsam::Pose3 &a, const gtsam::Pose3 &b,
      boost::optional<gtsam::Matrix &> H1 = boost::none,
      boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
    const auto relative = a.between(b, H1, H2);
    gtsam::Matrix6 local;
    const auto residual = measured_.localCoordinates(
        relative, boost::none, (H1 || H2) ? &local : nullptr);
    if (H1)
      *H1 = local * (*H1);
    if (H2)
      *H2 = local * (*H2);
    return residual;
  }
};

int main(int argc, char **argv) {
  const bool help = argc == 2 && std::string(argv[1]) == "--help";
  if (argc != 3 || help) {
    std::cerr
        << "MAP.pbstream OUTPUT_DIRECTORY\nResearch-only joint pose/plane "
           "solve; writes CSV/JSON, never installs or overwrites maps.\nUses "
           "stored six-DoF edge information and the default mapping Cauchy "
           "kernel; unsupported factor types are rejected.\n";
    return help ? 0 : 2;
  }
  try {
    const auto began = Clock::now();
    const std::filesystem::path out(argv[2]);
    if (!std::filesystem::create_directory(out))
      throw std::runtime_error("Existing output");
    n3mapping::Config config;

    n3mapping::N3Map map;
    std::string error;
    if (!n3mapping::readN3MapProtoFromFile(argv[1], &map, &error))
      throw std::runtime_error(error);
    if (map.keyframes_size() < 3 || map.floor_attitude_factors_size() != 0)
      throw std::runtime_error("Expected at least three keyframes and no "
                               "unsupported attitude factors");
    for (const auto &edge : map.edges())
      if (edge.constraint_mode() != n3mapping::EdgeProto::FULL_6DOF ||
          (edge.type() != n3mapping::EdgeProto::ODOMETRY &&
           edge.type() != n3mapping::EdgeProto::LOOP))
        throw std::runtime_error(
            "This research driver does not reconstruct planar/session factors");
    std::vector<gtsam::Pose3> poses;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    for (const auto &k : map.keyframes()) {
      if (k.cloud().points_size() % 4 != 0)
        throw std::runtime_error("Malformed packed keyframe cloud");
      poses.emplace_back(n3mapping::poseFromProto(k.pose_optimized()).matrix());
      if (!poses.back().matrix().allFinite())
        throw std::runtime_error("Non-finite input pose");
      initial.insert(gtsam::Symbol('x', k.id()), poses.back());
    }
    gtsam::Vector6 sigmas;
    sigmas << config.prior_noise_rotation, config.prior_noise_rotation,
        config.prior_noise_rotation, config.prior_noise_position,
        config.prior_noise_position, config.prior_noise_position;
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        gtsam::Symbol('x', map.keyframes(0).id()), poses[0],
        gtsam::noiseModel::Diagonal::Sigmas(sigmas)));
    int odometry = 0, loops = 0;
    for (const auto &e : map.edges()) {
      if (!initial.exists(gtsam::Symbol('x', e.from_id())) ||
          !initial.exists(gtsam::Symbol('x', e.to_id())))
        throw std::runtime_error("Dangling graph edge");
      Eigen::Matrix<double, 6, 6> info;
      if (!n3mapping::informationFromProto(e.information(), &info, &error))
        throw std::runtime_error(error);
      const auto explicit_noise =
          n3mapping::makeExplicitFullGraphFactorNoise(info);
      if (!explicit_noise)
        throw std::runtime_error("Missing usable stored edge information");
      const auto noise = e.type() == n3mapping::EdgeProto::LOOP
                             ? n3mapping::makeFullGraphFactorNoise(
                                   info, config,
                                   n3mapping::GraphFactorNoiseRole::
                                       ROBUST_GLOBAL_OR_SESSION_ODOMETRY,
                                   true)
                                   .model
                             : explicit_noise;
      graph.add(ExactPoseFactor(
          gtsam::Symbol('x', e.from_id()), gtsam::Symbol('x', e.to_id()),
          gtsam::Pose3(n3mapping::poseFromProto(e.measurement()).matrix()),
          noise));
      if (e.type() == n3mapping::EdgeProto::LOOP)
        ++loops;
      else
        ++odometry;
    }
    Json::Value report;
    const auto associating = Clock::now();
    std::vector<n3mapping::Keyframe::Ptr> keyframes;
    for (const auto& k : map.keyframes()) {
      auto frame = std::make_shared<n3mapping::Keyframe>();
      frame->id = k.id();
      frame->timestamp = k.timestamp();
      frame->pose_odom = n3mapping::poseFromProto(k.pose_odom());
      if (!n3mapping::pointCloudFromProto(k.cloud(), &frame->cloud, &error))
        throw std::runtime_error(error);
      keyframes.push_back(frame);
    }
    auto planes = associate(keyframes, poses, report);
    const auto merging = Clock::now();
    planes = mergePlanes(std::move(planes), poses, report);
    const auto factors = Clock::now();
    if (planes.empty())
      throw std::runtime_error("No supported shared planes");
    report["association_wall_s"] =
        std::chrono::duration<double>(merging - associating).count();
    report["merging_wall_s"] =
        std::chrono::duration<double>(factors - merging).count();
    bool audited = false;
    const auto sources = n3mapping::geometrySourceIds(keyframes);
    for (int i = 0; i < int(planes.size()); ++i) {
      const auto &plane = planes[i];
      std::map<size_t, size_t> multiplicity;
      for (const auto& patch : plane.patches) ++multiplicity[sources.at(patch.index)];
      if (multiplicity.size() < 3) continue;
      const gtsam::OrientedPlane3 landmark(gtsam::Unit3(plane.normal),
                                           plane.offset);
      const auto key = gtsam::Symbol('p', i);
      initial.insert(key, landmark);
      for (const auto &patch : plane.patches) {
        const auto noise = momentNoise(config, plane.sigma,
            1. / multiplicity.at(sources.at(patch.index)));
        MomentPlaneFactor factor(
            gtsam::Symbol('x', map.keyframes(patch.index).id()), key,
            patch.moment, plane.origin, noise);
        if (!audited) {
          auditFactor(factor, poses[patch.index], landmark, report);
          audited = true;
        }
        graph.add(factor);
      }
    }
    report["mode"] = "shared_connected_planes";
    report["odometry_edges"] = odometry;
    report["loop_edges"] = loops;
    report["initial_cost"] = graph.error(initial);
    std::cout << report << std::endl;
    tbb::global_control cores(tbb::global_control::max_allowed_parallelism, 1);
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    params.diagonalDamping = true;
    params.setVerbosityLM("SUMMARY");
    const auto solving = Clock::now();
    report["geometry_factors_wall_s"] =
        std::chrono::duration<double>(solving - factors).count();
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    const auto result = optimizer.optimize();
    report["solve_wall_s"] =
        std::chrono::duration<double>(Clock::now() - solving).count();
    report["final_cost"] = graph.error(result);
    report["iterations"] = int(optimizer.iterations());
    std::ofstream csv(out / "keyframes.csv");
    csv << std::setprecision(17) << "id,timestamp";
    for (int i = 0; i < 16; ++i)
      csv << ",optimized" << i;
    csv << '\n';
    double maximum = 0;
    for (int i = 0; i < map.keyframes_size(); ++i) {
      const auto &k = map.keyframes(i);
      const auto pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', k.id()));
      maximum = std::max(maximum,
                         (pose.translation() - poses[i].translation()).norm());
      csv << k.id() << ',' << k.timestamp();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          csv << ',' << pose.matrix()(r, c);
      csv << '\n';
    }
    report["max_pose_change_m"] = maximum;
    report["wall_s"] =
        std::chrono::duration<double>(Clock::now() - began).count();
    report["truth_used"] = false;
    report["online_feedback_qualified"] = false;
    std::ofstream(out / "report.json") << report << '\n';
    std::cout << report << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
