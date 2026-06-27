#include "n3mapping/loop_consensus_verifier.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

Eigen::Isometry3d pose(double x, double y, double z, double yaw = 0.0)
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(x, y, z);
    transform.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return transform;
}

LoopConsensusPairEvidence pairEvidence(int offset, double translation, double rotation, bool converged = true)
{
    LoopConsensusPairEvidence pair;
    pair.offset = offset;
    pair.valid = true;
    pair.converged = converged;
    pair.fitness_score = 0.1;
    pair.inlier_ratio = 0.9;
    pair.delta_translation_norm = translation;
    pair.delta_rotation_norm = rotation;
    return pair;
}

LoopConsensusPairEvidence estimatedPair(int offset, const Eigen::Isometry3d& estimate)
{
    LoopConsensusPairEvidence pair = pairEvidence(offset, 0.1, 0.04);
    pair.has_estimated_measurement = true;
    pair.estimated_match_query = estimate;
    return pair;
}

Config consensusConfig()
{
    Config config;
    config.keyframe_distance_threshold = 1.0;
    config.keyframe_angle_threshold = 0.5;
    return config;
}

}  // namespace

TEST(LoopConsensusVerifierTest, PredictsNeighborTransformFromCentralLoop)
{
    const Eigen::Isometry3d T_w_q = pose(10.0, 0.0, 0.0);
    const Eigen::Isometry3d T_w_m = pose(1.0, 2.0, 0.0);
    const Eigen::Isometry3d T_w_qi = pose(12.0, 0.0, 0.0);
    const Eigen::Isometry3d T_w_mi = pose(3.0, 2.0, 0.0);
    const Eigen::Isometry3d T_m_q = T_w_m.inverse() * T_w_q;

    const Eigen::Isometry3d predicted = LoopConsensusVerifier::predictNeighborTransform(
        T_w_q, T_w_m, T_w_qi, T_w_mi, T_m_q);

    const Eigen::Isometry3d expected = T_w_mi.inverse() * T_w_qi;
    EXPECT_NEAR((predicted.translation() - expected.translation()).norm(), 0.0, 1.0e-9);
    EXPECT_NEAR(Eigen::AngleAxisd(predicted.rotation().transpose() * expected.rotation()).angle(), 0.0, 1.0e-9);
}

TEST(LoopConsensusVerifierTest, CommitsWhenNeighborDeltasFormTightTwoSidedCluster)
{
    const auto result = LoopConsensusVerifier::summarizePairs(consensusConfig(), {
        pairEvidence(-2, 0.10, 0.04),
        pairEvidence(-1, 0.12, 0.05),
        pairEvidence(1, 0.09, 0.03),
        pairEvidence(2, 0.11, 0.04),
    });

    EXPECT_EQ(result.decision, LoopConsensusDecision::Commit);
    EXPECT_EQ(result.reason, "neighborhood_consensus");
    EXPECT_EQ(result.valid_pair_count, 4);
    EXPECT_EQ(result.left_support_count, 2);
    EXPECT_EQ(result.right_support_count, 2);
    EXPECT_NEAR(result.median_translation_delta, 0.105, 1.0e-9);
}

TEST(LoopConsensusVerifierTest, DefersWhenProofSupportIsTooSmall)
{
    const auto result = LoopConsensusVerifier::summarizePairs(consensusConfig(), {
        pairEvidence(-1, 0.10, 0.04),
        pairEvidence(1, 0.09, 0.03),
    });

    EXPECT_EQ(result.decision, LoopConsensusDecision::Defer);
    EXPECT_EQ(result.reason, "insufficient_support");
    EXPECT_EQ(result.valid_pair_count, 2);
}

TEST(LoopConsensusVerifierTest, CommitsWithEnoughOneSidedSupportForOnlineLatestQuery)
{
    const auto result = LoopConsensusVerifier::summarizePairs(consensusConfig(), {
        pairEvidence(-5, 0.10, 0.04),
        pairEvidence(-4, 0.12, 0.05),
        pairEvidence(-3, 0.09, 0.03),
        pairEvidence(-2, 0.11, 0.04),
        pairEvidence(-1, 0.10, 0.04),
    });

    EXPECT_EQ(result.decision, LoopConsensusDecision::Commit);
    EXPECT_EQ(result.reason, "neighborhood_consensus");
    EXPECT_EQ(result.valid_pair_count, 5);
    EXPECT_EQ(result.left_support_count, 5);
    EXPECT_EQ(result.right_support_count, 0);
}

TEST(LoopConsensusVerifierTest, RejectsWhenMostValidPairsContradictCentralMeasurement)
{
    const auto result = LoopConsensusVerifier::summarizePairs(consensusConfig(), {
        pairEvidence(-2, 5.0, 0.1),
        pairEvidence(-1, 5.2, 0.1),
        pairEvidence(1, 5.1, 0.1),
        pairEvidence(2, 0.2, 0.1),
    });

    EXPECT_EQ(result.decision, LoopConsensusDecision::Reject);
    EXPECT_EQ(result.reason, "neighborhood_contradiction");
    EXPECT_EQ(result.contradiction_count, 3);
}

TEST(LoopConsensusVerifierTest, EstimatesConsensusMeasurementFromNeighborPairs)
{
    const auto result = LoopConsensusVerifier::summarizePairs(consensusConfig(), {
        estimatedPair(-2, pose(5.0, 1.0, 0.20, 0.10)),
        estimatedPair(-1, pose(5.1, 0.9, 0.25, 0.11)),
        estimatedPair(1, pose(4.9, 1.1, 0.15, 0.09)),
        estimatedPair(2, pose(20.0, 1.0, 4.00, 1.50)),
    }, pose(5.0, 1.0, 0.20, 0.10));

    EXPECT_TRUE(result.estimator_valid);
    EXPECT_EQ(result.estimator_pair_count, 4);
    EXPECT_EQ(result.estimator_inlier_count, 3);
    EXPECT_NEAR(result.estimator_inlier_ratio, 0.75, 1.0e-9);
    EXPECT_NEAR(result.estimator_z_median, 0.20, 1.0e-9);
    EXPECT_NEAR(result.estimator_yaw_median, 0.10, 1.0e-9);
    EXPECT_EQ(result.estimator_recommendation, "stable_consensus_measurement");
    EXPECT_LT(result.estimator_measurement_delta_translation, 0.2);
    EXPECT_LT(result.estimator_measurement_delta_rotation, 0.1);
}

}  // namespace n3mapping
