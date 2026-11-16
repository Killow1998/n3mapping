#pragma once

#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {
enum class EdgeType { ODOMETRY, LOOP, SESSION_ANCHOR };

enum class EdgeConstraintMode
{
    FULL_6DOF,
    XY_YAW
};

struct EdgeInfo
{
  int64_t from_id;
  int64_t to_id;
  Eigen::Isometry3d measurement;
  Eigen::Matrix<double, 6, 6> information;
  EdgeType type;
  EdgeConstraintMode constraint_mode;

  EdgeInfo()
      : from_id(-1), to_id(-1), measurement(Eigen::Isometry3d::Identity()),
        information(Eigen::Matrix<double, 6, 6>::Identity()),
        type(EdgeType::ODOMETRY),
        constraint_mode(EdgeConstraintMode::FULL_6DOF) {}

  EdgeInfo(int64_t from, int64_t to, const Eigen::Isometry3d &meas,
           const Eigen::Matrix<double, 6, 6> &info, EdgeType t)
      : from_id(from), to_id(to), measurement(meas), information(info), type(t),
        constraint_mode(EdgeConstraintMode::FULL_6DOF) {}
};

struct FloorAttitudeConstraint
{
    int64_t node_id = -1;
    Eigen::Vector3d normal_body = Eigen::Vector3d::UnitZ();
    double sigma_rad = 0.0;
};

class LoopOptimizerInterface
{
  public:
    virtual ~LoopOptimizerInterface() = default;
    virtual void addLoopEdge(const EdgeInfo& edge) = 0;
    virtual bool incrementalOptimize() = 0;
};

}
