#include "n3mapping/submap_graph_trial.h"

#include "n3mapping/graph_factor_noise.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace n3mapping {
namespace {

Eigen::Isometry3d makePose(const Eigen::Vector3d& translation,
                           double roll = 0.0,
                           double pitch = 0.0,
                           double yaw = 0.0) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = translation;
    pose.linear() =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return pose;
}

Eigen::Matrix<double, 6, 6> information(double translation,
                                        double rotation) {
    Eigen::Matrix<double, 6, 6> result =
        Eigen::Matrix<double, 6, 6>::Zero();
    result.diagonal() << translation, translation, translation,
        rotation, rotation, rotation;
    return result;
}

SubmapGraphNodeProjection makeNode(SubmapId id,
                                   MapSessionId session,
                                   int64_t anchor_keyframe_id,
                                   const Eigen::Isometry3d& pose) {
    SubmapGraphNodeProjection node;
    node.submap_id = id;
    node.session_id = session;
    node.anchor_keyframe_id = anchor_keyframe_id;
    node.keyframe_count = 1;
    node.closed = true;
    node.content_revision = 1;
    node.T_map_submap = pose;
    return node;
}

SubmapGraphEdgeProjection makeCrossEdge(
    std::size_t source_index,
    int64_t from_keyframe,
    int64_t to_keyframe,
    SubmapId from_submap,
    SubmapId to_submap,
    const Eigen::Isometry3d& T_from_submap_from_keyframe,
    const Eigen::Isometry3d& T_to_submap_to_keyframe,
    const Eigen::Isometry3d& source_measurement,
    EdgeType type,
    EdgeConstraintMode mode,
    const Eigen::Matrix<double, 6, 6>& edge_information) {
    SubmapGraphEdgeProjection projection;
    projection.source_edge_index = source_index;
    projection.source_edge.from_id = from_keyframe;
    projection.source_edge.to_id = to_keyframe;
    projection.source_edge.measurement = source_measurement;
    projection.source_edge.information = edge_information;
    projection.source_edge.type = type;
    projection.source_edge.constraint_mode = mode;
    projection.classification = SubmapGraphEdgeClass::CROSS_SUBMAP;
    projection.from_submap_id = from_submap;
    projection.to_submap_id = to_submap;
    projection.T_from_submap_from_keyframe =
        T_from_submap_from_keyframe;
    projection.T_to_submap_to_keyframe =
        T_to_submap_to_keyframe;
    projection.has_projected_measurement = true;
    projection.T_from_submap_to_submap_measurement =
        T_from_submap_from_keyframe * source_measurement *
        T_to_submap_to_keyframe.inverse();
    return projection;
}

SubmapGraphSnapshot makeTwoNodeFullSnapshot(
    const Eigen::Isometry3d& initial_to_pose,
    const Eigen::Isometry3d& true_to_pose) {
    const Eigen::Isometry3d from_internal = makePose(
        Eigen::Vector3d(0.7, -0.2, 0.1), 0.03, -0.04, 0.2);
    const Eigen::Isometry3d to_internal = makePose(
        Eigen::Vector3d(-0.4, 0.3, -0.1), -0.02, 0.06, -0.15);
    const Eigen::Isometry3d source_measurement =
        from_internal.inverse() * true_to_pose * to_internal;

    SubmapGraphSnapshot snapshot;
    snapshot.valid = true;
    snapshot.nodes.push_back(makeNode(
        0, 0, 10, Eigen::Isometry3d::Identity()));
    snapshot.nodes.push_back(makeNode(1, 0, 20, initial_to_pose));
    snapshot.keyframe_ownership.emplace(10, 0);
    snapshot.keyframe_ownership.emplace(20, 1);
    snapshot.edge_projections.push_back(makeCrossEdge(
        0, 10, 20, 0, 1, from_internal, to_internal,
        source_measurement, EdgeType::ODOMETRY,
        EdgeConstraintMode::FULL_6DOF, information(100.0, 100.0)));
    snapshot.source_edge_count = 1;
    snapshot.cross_submap_edge_count = 1;
    snapshot.cross_submap_odometry_edge_count = 1;
    return snapshot;
}

const SubmapGraphTrialNodeResult* findNode(
    const SubmapGraphTrialDiagnostics& diagnostics,
    SubmapId id) {
    for (const auto& node : diagnostics.nodes) {
        if (node.submap_id == id) return &node;
    }
    return nullptr;
}

TEST(SubmapGraphTrialTest,
     Full6DLiftedFactorSolvesWithoutMutatingSnapshot) {
    const Eigen::Isometry3d true_to_pose = makePose(
        Eigen::Vector3d(4.0, -1.0, 0.5), 0.08, -0.05, 0.35);
    const Eigen::Isometry3d initial_to_pose = true_to_pose * makePose(
        Eigen::Vector3d(0.4, -0.25, 0.15), 0.04, -0.03, 0.1);
    const SubmapGraphSnapshot snapshot = makeTwoNodeFullSnapshot(
        initial_to_pose, true_to_pose);
    const auto before = snapshot.nodes;

    Config config;
    config.use_robust_kernel = false;
    config.optimization_iterations = 30;
    const auto trial = evaluateSubmapGraphOptimizationTrial(
        snapshot, config);

    ASSERT_TRUE(trial.valid) << trial.failure_reason;
    EXPECT_TRUE(trial.attempted);
    EXPECT_TRUE(trial.solved);
    EXPECT_EQ(trial.node_count, 2u);
    EXPECT_EQ(trial.gauge_anchor_count, 1u);
    EXPECT_EQ(trial.active_edge_factor_count, 1u);
    EXPECT_EQ(trial.full_6d_factor_count, 1u);
    EXPECT_EQ(trial.xy_yaw_lifted_factor_count, 0u);
    EXPECT_EQ(trial.explicit_information_factor_count, 1u);
    EXPECT_EQ(trial.fallback_noise_factor_count, 0u);
    EXPECT_GT(trial.nonlinear_error_reduction, 1e-3);

    const auto* anchor = findNode(trial, 0);
    const auto* optimized = findNode(trial, 1);
    ASSERT_NE(anchor, nullptr);
    ASSERT_NE(optimized, nullptr);
    EXPECT_TRUE(anchor->gauge_anchor);
    EXPECT_TRUE(anchor->optimized_pose.matrix().isApprox(
        Eigen::Isometry3d::Identity().matrix(), 1e-12));
    EXPECT_TRUE(optimized->optimized_pose.matrix().isApprox(
        true_to_pose.matrix(), 1e-6));
    ASSERT_EQ(snapshot.nodes.size(), before.size());
    for (std::size_t index = 0; index < before.size(); ++index) {
        EXPECT_TRUE(snapshot.nodes[index].T_map_submap.matrix().isApprox(
            before[index].T_map_submap.matrix(), 0.0));
    }
}

TEST(SubmapGraphTrialTest,
     XYYawRemainsLiftedAlongsideFullConstraintAndFloor) {
    const Eigen::Isometry3d true_to_pose = makePose(
        Eigen::Vector3d(3.0, 2.0, 0.4), 0.12, -0.08, 0.45);
    SubmapGraphSnapshot snapshot = makeTwoNodeFullSnapshot(
        true_to_pose * makePose(Eigen::Vector3d(0.2, -0.1, 0.05),
                                0.02, -0.01, 0.08),
        true_to_pose);
    const Eigen::Isometry3d from_internal = makePose(
        Eigen::Vector3d(-0.2, 0.4, 0.1), 0.05, 0.02, -0.3);
    const Eigen::Isometry3d to_internal = makePose(
        Eigen::Vector3d(0.6, -0.3, -0.2), -0.04, 0.03, 0.25);
    const Eigen::Isometry3d source_measurement =
        from_internal.inverse() * true_to_pose * to_internal;
    snapshot.edge_projections.push_back(makeCrossEdge(
        1, 10, 20, 0, 1, from_internal, to_internal,
        source_measurement, EdgeType::LOOP,
        EdgeConstraintMode::XY_YAW, information(25.0, 16.0)));
    snapshot.source_edge_count = 2;
    snapshot.cross_submap_edge_count = 2;
    snapshot.cross_submap_loop_edge_count = 1;

    SubmapGraphFloorProjection floor;
    floor.source_constraint_index = 0;
    floor.source_constraint.node_id = 20;
    floor.source_constraint.normal_body =
        Eigen::Vector3d(0.1, -0.15, 0.98).normalized();
    floor.source_constraint.sigma_rad = 0.05;
    floor.assigned = true;
    floor.submap_id = 1;
    floor.T_submap_keyframe = to_internal;
    snapshot.floor_projections.push_back(floor);
    snapshot.source_floor_constraint_count = 1;
    snapshot.assigned_floor_constraint_count = 1;

    Config config;
    config.use_robust_kernel = true;
    config.robust_kernel_type = "Huber";
    config.optimization_iterations = 30;
    const auto trial = evaluateSubmapGraphOptimizationTrial(
        snapshot, config);

    ASSERT_TRUE(trial.valid) << trial.failure_reason;
    EXPECT_EQ(trial.active_edge_factor_count, 2u);
    EXPECT_EQ(trial.full_6d_factor_count, 1u);
    EXPECT_EQ(trial.xy_yaw_lifted_factor_count, 1u);
    EXPECT_EQ(trial.robust_factor_count, 1u);
    EXPECT_EQ(trial.floor_factor_count, 1u);
    EXPECT_GE(trial.nonlinear_error_reduction, 0.0);
}

TEST(SubmapGraphTrialTest,
     SessionOdometryAndFallbackNoiseProvenanceAreExplicit) {
    SubmapGraphSnapshot snapshot;
    snapshot.valid = true;
    snapshot.nodes.push_back(makeNode(
        0, 0, 0, Eigen::Isometry3d::Identity()));
    snapshot.nodes.push_back(makeNode(
        1, 1, 100, makePose(Eigen::Vector3d(10.0, 0.0, 0.0))));
    snapshot.nodes.push_back(makeNode(
        2, 1, 101, makePose(Eigen::Vector3d(11.0, 0.0, 0.0))));
    snapshot.keyframe_ownership.emplace(0, 0);
    snapshot.keyframe_ownership.emplace(100, 1);
    snapshot.keyframe_ownership.emplace(101, 2);
    snapshot.edge_projections.push_back(makeCrossEdge(
        0, 0, 100, 0, 1, Eigen::Isometry3d::Identity(),
        Eigen::Isometry3d::Identity(),
        makePose(Eigen::Vector3d(10.0, 0.0, 0.0)),
        EdgeType::SESSION_ANCHOR, EdgeConstraintMode::FULL_6DOF,
        Eigen::Matrix<double, 6, 6>::Zero()));
    snapshot.edge_projections.push_back(makeCrossEdge(
        1, 100, 101, 1, 2, Eigen::Isometry3d::Identity(),
        Eigen::Isometry3d::Identity(),
        makePose(Eigen::Vector3d(1.0, 0.0, 0.0)),
        EdgeType::ODOMETRY, EdgeConstraintMode::FULL_6DOF,
        Eigen::Matrix<double, 6, 6>::Zero()));
    snapshot.source_edge_count = 2;
    snapshot.cross_submap_edge_count = 2;
    snapshot.cross_submap_odometry_edge_count = 1;
    snapshot.cross_submap_session_anchor_edge_count = 1;

    Config config;
    config.robust_kernel_type = "Cauchy";
    const auto trial = evaluateSubmapGraphOptimizationTrial(
        snapshot, config);

    ASSERT_TRUE(trial.valid) << trial.failure_reason;
    EXPECT_EQ(trial.robust_factor_count, 2u);
    EXPECT_EQ(trial.session_odometry_factor_count, 1u);
    EXPECT_EQ(trial.explicit_information_factor_count, 0u);
    EXPECT_EQ(trial.fallback_noise_factor_count, 2u);
}

TEST(SubmapGraphTrialTest,
     SharedNoiseFactoryPreservesInformationOrderAndFallbackRoles) {
    Config config;
    config.odom_noise_rotation = 0.11;
    config.odom_noise_position = 0.22;
    config.loop_noise_rotation = 0.33;
    config.loop_noise_position = 0.44;
    config.use_robust_kernel = true;
    config.robust_kernel_type = "Huber";

    Eigen::Matrix<double, 6, 6> explicit_information =
        Eigen::Matrix<double, 6, 6>::Zero();
    explicit_information.diagonal() << 4.0, 9.0, 16.0,
        25.0, 36.0, 49.0;
    const auto explicit_noise = makeFullGraphFactorNoise(
        explicit_information, config, GraphFactorNoiseRole::ODOMETRY,
        false);
    ASSERT_TRUE(explicit_noise.model);
    EXPECT_TRUE(explicit_noise.explicit_information);
    EXPECT_FALSE(explicit_noise.fallback);
    EXPECT_FALSE(explicit_noise.robust);
    gtsam::Vector6 factor_order_residual;
    factor_order_residual << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    const double expected_mahalanobis =
        25.0 * 1.0 + 36.0 * 4.0 + 49.0 * 9.0 +
        4.0 * 16.0 + 9.0 * 25.0 + 16.0 * 36.0;
    EXPECT_NEAR(explicit_noise.model->squaredMahalanobisDistance(
                    factor_order_residual),
                expected_mahalanobis, 1e-9);

    const auto odometry_fallback = makeFullGraphFactorNoise(
        Eigen::Matrix<double, 6, 6>::Zero(), config,
        GraphFactorNoiseRole::ODOMETRY, false);
    ASSERT_TRUE(odometry_fallback.model);
    EXPECT_FALSE(odometry_fallback.explicit_information);
    EXPECT_TRUE(odometry_fallback.fallback);
    EXPECT_FALSE(odometry_fallback.robust);
    const gtsam::Vector odometry_sigmas =
        odometry_fallback.model->sigmas();
    ASSERT_EQ(odometry_sigmas.size(), 6);
    EXPECT_TRUE(odometry_sigmas.isApprox(
        (gtsam::Vector6() << 0.11, 0.11, 0.11,
                             0.22, 0.22, 0.22).finished(),
        1e-12));

    const auto loop_fallback = makeFullGraphFactorNoise(
        Eigen::Matrix<double, 6, 6>::Zero(), config,
        GraphFactorNoiseRole::ROBUST_GLOBAL_OR_SESSION_ODOMETRY,
        true);
    ASSERT_TRUE(loop_fallback.model);
    EXPECT_FALSE(loop_fallback.explicit_information);
    EXPECT_TRUE(loop_fallback.fallback);
    EXPECT_TRUE(loop_fallback.robust);
}

TEST(SubmapGraphTrialTest,
     IntraSubmapEdgeRemainsAReportedConstantTerm) {
    SubmapGraphSnapshot snapshot;
    snapshot.valid = true;
    snapshot.nodes.push_back(makeNode(
        7, 3, 70, makePose(Eigen::Vector3d(2.0, -1.0, 0.3),
                            0.02, -0.03, 0.4)));
    snapshot.nodes.front().keyframe_count = 2;
    snapshot.keyframe_ownership.emplace(70, 7);
    snapshot.keyframe_ownership.emplace(71, 7);

    SubmapGraphEdgeProjection intra;
    intra.source_edge_index = 0;
    intra.source_edge.from_id = 70;
    intra.source_edge.to_id = 71;
    intra.source_edge.measurement =
        makePose(Eigen::Vector3d(0.5, 0.0, 0.0));
    intra.source_edge.information = information(10.0, 10.0);
    intra.source_edge.type = EdgeType::ODOMETRY;
    intra.source_edge.constraint_mode = EdgeConstraintMode::FULL_6DOF;
    intra.classification = SubmapGraphEdgeClass::INTRA_SUBMAP;
    intra.from_submap_id = 7;
    intra.to_submap_id = 7;
    snapshot.edge_projections.push_back(intra);
    snapshot.source_edge_count = 1;
    snapshot.intra_submap_edge_count = 1;

    Config config;
    const auto trial = evaluateSubmapGraphOptimizationTrial(
        snapshot, config);

    ASSERT_TRUE(trial.valid) << trial.failure_reason;
    EXPECT_TRUE(trial.solved);
    EXPECT_EQ(trial.node_count, 1u);
    EXPECT_EQ(trial.gauge_anchor_count, 1u);
    EXPECT_EQ(trial.active_edge_factor_count, 0u);
    EXPECT_EQ(trial.intra_submap_constant_edge_count, 1u);
    ASSERT_EQ(trial.nodes.size(), 1u);
    EXPECT_TRUE(trial.nodes.front().gauge_anchor);
    EXPECT_TRUE(trial.nodes.front().optimized_pose.matrix().isApprox(
        snapshot.nodes.front().T_map_submap.matrix(), 1e-12));
    EXPECT_NEAR(trial.initial_nonlinear_error, 0.0, 1e-12);
    EXPECT_NEAR(trial.final_nonlinear_error, 0.0, 1e-12);
}

TEST(SubmapGraphTrialTest,
     DisconnectedTopologyFailsBeforeOptimizationAndIsDeterministic) {
    SubmapGraphSnapshot snapshot;
    snapshot.valid = true;
    snapshot.nodes.push_back(makeNode(
        5, 0, 50, Eigen::Isometry3d::Identity()));
    snapshot.nodes.push_back(makeNode(
        8, 0, 80, makePose(Eigen::Vector3d(2.0, 0.0, 0.0))));
    snapshot.keyframe_ownership.emplace(50, 5);
    snapshot.keyframe_ownership.emplace(80, 8);

    Config config;
    const auto first = evaluateSubmapGraphOptimizationTrial(snapshot, config);
    const auto second = evaluateSubmapGraphOptimizationTrial(snapshot, config);
    EXPECT_FALSE(first.valid);
    EXPECT_FALSE(first.attempted);
    EXPECT_FALSE(first.solved);
    EXPECT_EQ(first.failure_reason, "topology_not_ready");
    EXPECT_EQ(second.failure_reason, first.failure_reason);
    EXPECT_TRUE(snapshot.nodes[1].T_map_submap.matrix().isApprox(
        makePose(Eigen::Vector3d(2.0, 0.0, 0.0)).matrix(), 0.0));
}

TEST(SubmapGraphTrialTest, RepeatedIsolatedSolveIsDeterministic) {
    const Eigen::Isometry3d true_to_pose = makePose(
        Eigen::Vector3d(6.0, -2.0, 0.3), 0.04, -0.02, 0.25);
    const SubmapGraphSnapshot snapshot = makeTwoNodeFullSnapshot(
        true_to_pose * makePose(Eigen::Vector3d(-0.3, 0.2, 0.1),
                                -0.02, 0.01, -0.06),
        true_to_pose);
    Config config;
    config.use_robust_kernel = false;
    config.optimization_iterations = 30;

    const auto first = evaluateSubmapGraphOptimizationTrial(snapshot, config);
    const auto second = evaluateSubmapGraphOptimizationTrial(snapshot, config);
    ASSERT_TRUE(first.valid) << first.failure_reason;
    ASSERT_TRUE(second.valid) << second.failure_reason;
    ASSERT_EQ(first.nodes.size(), second.nodes.size());
    EXPECT_NEAR(first.initial_nonlinear_error,
                second.initial_nonlinear_error, 1e-12);
    EXPECT_NEAR(first.final_nonlinear_error,
                second.final_nonlinear_error, 1e-12);
    for (std::size_t index = 0; index < first.nodes.size(); ++index) {
        EXPECT_TRUE(first.nodes[index].optimized_pose.matrix().isApprox(
            second.nodes[index].optimized_pose.matrix(), 1e-12));
    }
}

}  // namespace
}  // namespace n3mapping
