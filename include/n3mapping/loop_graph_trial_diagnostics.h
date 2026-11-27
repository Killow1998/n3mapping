#pragma once

#include <map>
#include <set>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/graph_types.h"

namespace n3mapping {

struct LoopGraphTrialDiagnostics {
    bool success = false;
    bool conflicts_with_map_height = false;
    double residual_x_after = std::numeric_limits<double>::quiet_NaN();
    double residual_y_after = std::numeric_limits<double>::quiet_NaN();
    double residual_z_after = std::numeric_limits<double>::quiet_NaN();
    double residual_roll_after = std::numeric_limits<double>::quiet_NaN();
    double residual_pitch_after = std::numeric_limits<double>::quiet_NaN();
    double residual_yaw_after = std::numeric_limits<double>::quiet_NaN();
    double residual_translation_norm_after = std::numeric_limits<double>::quiet_NaN();
    double residual_rotation_norm_after = std::numeric_limits<double>::quiet_NaN();
    double mean_pose_update_translation = std::numeric_limits<double>::quiet_NaN();
    double max_pose_update_translation = std::numeric_limits<double>::quiet_NaN();
    double mean_pose_update_rotation = std::numeric_limits<double>::quiet_NaN();
    double max_pose_update_rotation = std::numeric_limits<double>::quiet_NaN();
    double existing_loop_residual_delta = std::numeric_limits<double>::quiet_NaN();
    double odom_residual_delta = std::numeric_limits<double>::quiet_NaN();
    double consistency_score = std::numeric_limits<double>::quiet_NaN();
    std::string recommendation = "not_available";
};

LoopGraphTrialDiagnostics computeLoopGraphTrialDiagnostics(
    const Config& config,
    const std::map<int64_t, Eigen::Isometry3d>& poses_before,
    const std::vector<EdgeInfo>& committed_edges,
    const std::vector<EdgeInfo>& candidate_edges,
    const std::set<int64_t>& fixed_node_ids = {});

class GraphOptimizer;
LoopGraphTrialDiagnostics computeLoopGraphTrialDiagnostics(
    const Config& config, const GraphOptimizer& graph,
    const std::vector<EdgeInfo>& candidate_edges);

struct VerifiedLoop;
void assignLoopGraphTrialDiagnostics(
    VerifiedLoop* loop, const LoopGraphTrialDiagnostics& diagnostics);

std::string loopGraphTrialRejectReason(
    const LoopGraphTrialDiagnostics& diagnostics, bool cross_session = false);

}
