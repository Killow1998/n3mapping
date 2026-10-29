#pragma once

#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping {
enum class EdgeType
{
    ODOMETRY, ///< 里程计约束
    LOOP,     ///< Robust geometric edge, including loaded-map tracking; not a measurement-source tag.
    SESSION_ANCHOR  ///< cross-session anchor constraint
};

enum class EdgeConstraintMode
{
    FULL_6DOF,
    XY_YAW
};

struct EdgeInfo
{
    int64_t from_id;                         ///< 起始节点 ID
    int64_t to_id;                           ///< 目标节点 ID
    Eigen::Isometry3d measurement;           ///< 相对位姿测量值
    Eigen::Matrix<double, 6, 6> information; ///< 信息矩阵 (协方差逆)
    EdgeType type;                           ///< 边类型
    EdgeConstraintMode constraint_mode;       ///< 约束维度

    EdgeInfo()
      : from_id(-1)
      , to_id(-1)
      , measurement(Eigen::Isometry3d::Identity())
      , information(Eigen::Matrix<double, 6, 6>::Identity())
      , type(EdgeType::ODOMETRY)
      , constraint_mode(EdgeConstraintMode::FULL_6DOF)
    {
    }

    EdgeInfo(int64_t from, int64_t to, const Eigen::Isometry3d& meas, const Eigen::Matrix<double, 6, 6>& info, EdgeType t)
      : from_id(from)
      , to_id(to)
      , measurement(meas)
      , information(info)
      , type(t)
      , constraint_mode(EdgeConstraintMode::FULL_6DOF)
    {
    }
};

// A floor attitude constraint is a roll/pitch-only observation attached to one
// keyframe: the floor normal as seen in the sensor frame, with the attitude
// noise as a sigma. It is a first-class serializable part of the graph (map
// format 2.4.0), with the same pending/committed lifecycle as edges.
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

}  // namespace n3mapping
