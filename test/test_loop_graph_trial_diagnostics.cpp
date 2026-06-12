#include "n3mapping/loop_graph_trial_diagnostics.h"

#include <cmath>
#include <map>
#include <vector>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

Eigen::Isometry3d translated(double x, double y = 0.0, double z = 0.0)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(x, y, z);
    return pose;
}

EdgeInfo makeEdge(int64_t from, int64_t to, const Eigen::Isometry3d& measurement, EdgeType type)
{
    EdgeInfo edge;
    edge.from_id = from;
    edge.to_id = to;
    edge.measurement = measurement;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    edge.type = type;
    return edge;
}

TEST(LoopGraphTrialDiagnosticsTest, ComputesSuccessfulShadowTrialWithoutMutatingCallerState)
{
    Config config;
    config.optimization_iterations = 5;
    std::map<int64_t, Eigen::Isometry3d> poses;
    poses[0] = Eigen::Isometry3d::Identity();
    poses[1] = translated(1.0);
    const std::vector<EdgeInfo> committed_edges{
        makeEdge(0, 1, translated(1.0), EdgeType::ODOMETRY),
    };
    const std::vector<EdgeInfo> candidate_edges{
        makeEdge(0, 1, translated(1.0), EdgeType::LOOP),
    };

    const auto diagnostics = computeLoopGraphTrialDiagnostics(
        config, poses, committed_edges, candidate_edges);

    EXPECT_TRUE(diagnostics.success);
    EXPECT_EQ(diagnostics.recommendation, "trial_success_score_only");
    EXPECT_TRUE(std::isfinite(diagnostics.consistency_score));
    EXPECT_GT(diagnostics.consistency_score, 0.0);
    EXPECT_NEAR(diagnostics.residual_translation_norm_after, 0.0, 1e-6);
    EXPECT_NEAR(diagnostics.max_pose_update_translation, 0.0, 1e-6);
}

TEST(LoopGraphTrialDiagnosticsTest, MissingEndpointReturnsFailedDiagnostics)
{
    Config config;
    std::map<int64_t, Eigen::Isometry3d> poses;
    poses[0] = Eigen::Isometry3d::Identity();
    poses[1] = translated(1.0);
    const std::vector<EdgeInfo> committed_edges{
        makeEdge(0, 1, translated(1.0), EdgeType::ODOMETRY),
    };
    const std::vector<EdgeInfo> candidate_edges{
        makeEdge(0, 99, translated(1.0), EdgeType::LOOP),
    };

    const auto diagnostics = computeLoopGraphTrialDiagnostics(
        config, poses, committed_edges, candidate_edges);

    EXPECT_FALSE(diagnostics.success);
    EXPECT_NE(diagnostics.recommendation, "trial_success_score_only");
    EXPECT_FALSE(std::isfinite(diagnostics.consistency_score));
}

}  // namespace
}  // namespace n3mapping
