#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "../src/core/loop_closure_processing.h"

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

TEST(LoopGraphTrialDiagnosticsTest, RebaseRetainsValidQueryWhenAnotherQueryConflicts)
{
    Config config;
    config.use_robust_kernel = false;
    GraphOptimizer optimizer(config);
    LoopClosureManager manager(config);
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    std::vector<EdgeInfo> odometry;
    for (int i = 0; i < 4; ++i) {
        nodes.emplace_back(i, translated(i));
        if (i) odometry.push_back(makeEdge(i - 1, i, translated(1), EdgeType::ODOMETRY));
    }
    ASSERT_TRUE(optimizer.loadGraph(nodes, odometry));
    LoopClosureOutcome outcome;
    for (int id : {2, 3}) {
        VerifiedLoop loop;
        loop.query_id = id;
        loop.match_id = 0;
        loop.verified = true;
        loop.T_match_query = translated(id == 2 ? 10.0 : 3.0);
        loop.information = Eigen::Matrix<double, 6, 6>::Identity() * 10000.0;
        outcome.summary.accepted_loops.push_back(loop);
        LoopDebugCandidateEvent event;
        event.candidate.query_id = id;
        event.candidate.match_id = 0;
        event.gate_result = "accepted";
        outcome.candidates.push_back(event);
    }
    const auto valid = outcome.summary.accepted_loops.back();
    rebaseLoopOutcome(config, optimizer, manager, outcome);
    ASSERT_EQ(outcome.summary.accepted_loops.size(), 1U);
    EXPECT_EQ(outcome.summary.accepted_loops.front().query_id, 3);
    EXPECT_TRUE(outcome.summary.accepted_loops.front().T_match_query.isApprox(
        valid.T_match_query, 0.0));
    EXPECT_TRUE(outcome.summary.accepted_loops.front().information.isApprox(
        valid.information, 0.0));
    EXPECT_EQ(optimizer.getEdges().size(), odometry.size() + 1);
    EXPECT_EQ(outcome.candidates.front().gate_result, "rejected");
    EXPECT_EQ(outcome.candidates.back().gate_result, "accepted");
}

}  // namespace
}  // namespace n3mapping
