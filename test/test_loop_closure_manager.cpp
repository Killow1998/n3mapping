#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_referee.h"
#include "n3mapping/loop_segment_consistency.h"
#include "n3mapping/loop_verifier.h"

namespace n3mapping {
namespace {

Keyframe::PointCloudT::Ptr makeTinyCloud()
{
    auto cloud = std::make_shared<Keyframe::PointCloudT>();
    pcl::PointXYZI point;
    point.x = 0.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 1.0f;
    cloud->push_back(point);
    return cloud;
}

Eigen::Isometry3d poseAt(double x, double y = 0.0, double yaw = 0.0)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(x, y, 0.0);
    pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return pose;
}

}  // namespace

TEST(LoopRefereeTest, AcceptsOnlyConsistentFeatureBundles)
{
    LoopFeatures good;
    good.descriptor_score = 0.8;
    good.spatial_score = 0.5;
    good.geometric_overlap = 0.7;
    good.temporal_gap = 1.0;
    good.local_map_consistency = 0.9;
    good.segment_consistency = 1.0;
    good.segment_support = 1.0;
    EXPECT_EQ(LoopReferee::decide(good), LoopDecision::Accept);

    LoopFeatures bad;
    bad.descriptor_score = 0.1;
    bad.spatial_score = 0.1;
    bad.geometric_overlap = 0.0;
    bad.temporal_gap = 0.1;
    bad.local_map_consistency = 0.1;
    bad.segment_consistency = 0.0;
    bad.segment_support = 1.0;
    EXPECT_EQ(LoopReferee::decide(bad), LoopDecision::Reject);
}

TEST(LoopRefereeTest, RejectsSupportedButInconsistentSegments)
{
    LoopFeatures features;
    features.descriptor_score = 1.0;
    features.spatial_score = 1.0;
    features.geometric_overlap = 1.0;
    features.temporal_gap = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 1.0;
    features.segment_consistency = 0.25;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "segment_inconsistent");
    EXPECT_EQ(decision.risk_flags, "segment");
}

TEST(LoopRefereeTest, RejectsSpatialOnlyWithoutDescriptorSupport)
{
    LoopFeatures features;
    features.spatial_only = true;
    features.spatial_score = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 1.0;
    features.segment_consistency = 1.0;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "spatial_only_unconfirmed");
    EXPECT_EQ(decision.risk_flags, "source");
}

TEST(LoopRefereeTest, RejectsLargePredictedMotionWithWeakSegment)
{
    LoopFeatures features;
    features.descriptor_supported = true;
    features.descriptor_score = 1.0;
    features.geometric_overlap = 0.2;
    features.local_map_consistency = 1.0;
    features.segment_support = 0.5;
    features.segment_consistency = 0.5;
    features.predicted_translation_norm = LoopReferee::kLargePredictedTranslationM + 0.1;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "large_prediction_with_weak_segment");
    EXPECT_EQ(decision.risk_flags, "prediction_segment");
}

TEST(LoopRefereeTest, RejectsYawFlipWhenSegmentTranslationDisagrees)
{
    LoopFeatures features;
    features.descriptor_supported = true;
    features.descriptor_score = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 1.0;
    features.segment_consistency = 1.0;
    features.icp_correction_yaw_abs = LoopReferee::kYawFlipRad + 0.1;
    features.segment_translation_median = LoopReferee::kLargeSegmentTranslationM + 0.1;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "yaw_flip_with_segment_disagreement");
    EXPECT_EQ(decision.risk_flags, "yaw_segment");
}

TEST(LoopRefereeTest, AcceptsDescriptorBackedVerifiedGeometryWithLimitedSegmentEvidence)
{
    LoopFeatures features;
    features.descriptor_supported = true;
    features.descriptor_score = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 0.5;
    features.segment_consistency = 0.5;
    features.predicted_translation_norm = 2.0;
    features.icp_correction_yaw_abs = 0.1;
    features.segment_translation_median = 0.5;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Accept);
    EXPECT_EQ(decision.reason, "descriptor_geometry_consistent");
}

TEST(LoopRefereeTest, RejectsDescriptorBackedLoopWithNoSubmapOrSegmentSupport)
{
    LoopFeatures features;
    features.descriptor_supported = true;
    features.descriptor_score = 1.0;
    features.geometric_overlap = 0.0;
    features.local_map_consistency = 0.3;
    features.segment_support = 0.0;
    features.segment_consistency = 0.0;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "descriptor_submap_inconsistent");
    EXPECT_EQ(decision.risk_flags, "submap");
}

TEST(LoopVerifierEvidenceTest, MeasurementResidualUsesPredictedAndMeasuredTransforms)
{
    Eigen::Isometry3d predicted = Eigen::Isometry3d::Identity();
    predicted.translation() = Eigen::Vector3d(10.0, 0.0, 1.0);

    Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
    correction.translation() = Eigen::Vector3d(0.5, -0.25, 0.75);

    const Eigen::Isometry3d measured = correction * predicted;
    const Eigen::Isometry3d residual = LoopVerifier::measurementResidual(predicted, measured);

    EXPECT_TRUE(measured.isApprox(correction * predicted, 1e-12));
    EXPECT_TRUE(residual.isApprox(predicted.inverse() * measured, 1e-12));
}

TEST(PointCloudMatcherEvidenceTest, ClassifiesTermination)
{
    EXPECT_EQ(classifyMatchTermination(true, 2, 10, true), MatchTermination::Converged);
    EXPECT_EQ(classifyMatchTermination(false, 9, 10, true), MatchTermination::MaxIterations);
    EXPECT_EQ(classifyMatchTermination(false, 2, 10, true), MatchTermination::Stalled);
    EXPECT_EQ(classifyMatchTermination(false, 0, 10, false), MatchTermination::Invalid);
    EXPECT_STREQ(matchTerminationName(MatchTermination::MaxIterations), "max_iterations");
}

TEST(LoopSegmentConsistencyTest, ReportsConsistentSameDirectionSegments)
{
    Config config;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> keyframes;
    for (int id = 0; id <= 12; ++id) {
        auto keyframe = Keyframe::create(id, static_cast<double>(id), poseAt(id), makeTinyCloud());
        keyframe->pose_optimized = poseAt(id);
        keyframes.push_back(keyframe);
    }
    manager.loadKeyframes(keyframes);

    VerifiedLoop loop;
    loop.query_id = 10;
    loop.match_id = 2;
    loop.verified = true;
    const auto diagnostics = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(diagnostics.valid_pair_count, 4);
    EXPECT_EQ(diagnostics.consensus_inlier_count, 4);
    EXPECT_DOUBLE_EQ(diagnostics.consensus_ratio, 1.0);
    EXPECT_EQ(diagnostics.direction, "same");
    EXPECT_EQ(diagnostics.recommendation, "consistent");
}

TEST(LoopSegmentConsistencyTest, ChoosesReverseDirectionWhenTraversalIsOpposite)
{
    Config config;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> keyframes;
    for (int id = 0; id <= 12; ++id) {
        auto keyframe = Keyframe::create(id, static_cast<double>(id), poseAt(id), makeTinyCloud());
        keyframe->pose_optimized = poseAt(id);
        keyframes.push_back(keyframe);
    }
    // Make the match segment around id=2 run opposite to the query segment around id=10.
    keyframes[0]->pose_optimized = poseAt(2.0);
    keyframes[1]->pose_optimized = poseAt(1.0);
    keyframes[2]->pose_optimized = poseAt(0.0);
    keyframes[3]->pose_optimized = poseAt(-1.0);
    keyframes[4]->pose_optimized = poseAt(-2.0);
    manager.loadKeyframes(keyframes);

    VerifiedLoop loop;
    loop.query_id = 10;
    loop.match_id = 2;
    loop.verified = true;
    const auto diagnostics = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(diagnostics.valid_pair_count, 4);
    EXPECT_EQ(diagnostics.consensus_inlier_count, 4);
    EXPECT_DOUBLE_EQ(diagnostics.consensus_ratio, 1.0);
    EXPECT_EQ(diagnostics.direction, "reverse");
    EXPECT_EQ(diagnostics.recommendation, "consistent");
}

class MockLoopOptimizer : public LoopOptimizerInterface
{
  public:
    MOCK_METHOD(void, addLoopEdge, (const EdgeInfo& edge), (override));
    MOCK_METHOD(bool, incrementalOptimize, (), (override));
};

TEST(LoopClosureManagerTest, FilterValidLoopsOnlyChecksValidity)
{
    Config config;
    config.loop_min_inlier_ratio = 0.3;
    config.loop_fitness_threshold = 1.0;

    LoopClosureManager manager(config);

    VerifiedLoop valid;
    valid.query_id = 1;
    valid.match_id = 2;
    valid.verified = true;
    valid.inlier_ratio = 0.5;
    valid.fitness_score = 0.5;

    VerifiedLoop low_inlier = valid;
    low_inlier.query_id = 3;
    low_inlier.inlier_ratio = 0.1;

    VerifiedLoop not_verified = valid;
    not_verified.query_id = 4;
    not_verified.verified = false;

    auto result = manager.filterValidLoops({ valid, low_inlier, not_verified });
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.front().query_id, 1);
    EXPECT_EQ(result.front().match_id, 2);
}

TEST(LoopClosureManagerTest, FilterValidLoopsDoesNotRejectLargeMeasurementResidualZ)
{
    Config config;
    config.loop_min_inlier_ratio = 0.3;
    config.loop_fitness_threshold = 1.0;

    LoopClosureManager manager(config);

    VerifiedLoop valid;
    valid.query_id = 1;
    valid.match_id = 2;
    valid.verified = true;
    valid.inlier_ratio = 0.5;
    valid.fitness_score = 0.5;
    valid.T_measurement_residual = Eigen::Isometry3d::Identity();
    valid.T_measurement_residual.translation().z() = 4.9;

    VerifiedLoop bad_z = valid;
    bad_z.query_id = 3;
    bad_z.T_measurement_residual.translation().z() = 5.1;

    auto result = manager.filterValidLoops({valid, bad_z});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[1].query_id, 3);
}

TEST(LoopClosureManagerTest, FilterValidLoopsIgnoresMeasurementResidualZ)
{
    Config config;
    config.loop_min_inlier_ratio = 0.3;
    config.loop_fitness_threshold = 1.0;

    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.verified = true;
    loop.inlier_ratio = 0.5;
    loop.fitness_score = 0.5;
    loop.T_measurement_residual = Eigen::Isometry3d::Identity();
    loop.T_measurement_residual.translation().z() = 100.0;

    auto result = manager.filterValidLoops({loop});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().query_id, 1);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryChoosesHighestRefereeEnergy)
{
    Config config;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 10.0;

    LoopClosureManager manager(config);

    VerifiedLoop a;
    a.query_id = 1;
    a.match_id = 10;
    a.verified = true;
    a.inlier_ratio = 1.0;
    a.fitness_score = 0.8;
    a.loop_referee_energy = 0.7;

    VerifiedLoop b = a;
    b.match_id = 11;
    b.fitness_score = 0.2;
    b.loop_referee_energy = 0.9;

    VerifiedLoop c = a;
    c.query_id = 2;
    c.match_id = 20;
    c.fitness_score = 0.4;
    c.loop_referee_energy = 0.1;

    auto best = manager.selectBestPerQuery({ a, b, c });
    ASSERT_EQ(best.size(), 2u);

    auto it = std::find_if(best.begin(), best.end(), [](const VerifiedLoop& loop) { return loop.query_id == 1; });
    ASSERT_TRUE(it != best.end());
    EXPECT_EQ(it->match_id, 11);
    EXPECT_DOUBLE_EQ(it->fitness_score, 0.2);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryFallsBackToFitnessWithoutEnergy)
{
    Config config;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 10.0;

    LoopClosureManager manager(config);

    VerifiedLoop high_z;
    high_z.query_id = 1;
    high_z.match_id = 10;
    high_z.verified = true;
    high_z.inlier_ratio = 1.0;
    high_z.fitness_score = 0.20;
    high_z.T_measurement_residual = Eigen::Isometry3d::Identity();
    high_z.T_measurement_residual.translation().z() = 4.0;

    VerifiedLoop low_z = high_z;
    low_z.match_id = 11;
    low_z.fitness_score = 0.205;
    low_z.T_measurement_residual.translation().z() = 0.1;

    auto best = manager.selectBestPerQuery({high_z, low_z});
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.front().match_id, 10);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryKeepsClearlyBetterFitness)
{
    Config config;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 10.0;

    LoopClosureManager manager(config);

    VerifiedLoop strong_fitness;
    strong_fitness.query_id = 1;
    strong_fitness.match_id = 10;
    strong_fitness.verified = true;
    strong_fitness.inlier_ratio = 1.0;
    strong_fitness.fitness_score = 0.05;
    strong_fitness.T_measurement_residual = Eigen::Isometry3d::Identity();
    strong_fitness.T_measurement_residual.translation().z() = 4.0;

    VerifiedLoop low_z = strong_fitness;
    low_z.match_id = 11;
    low_z.fitness_score = 0.20;
    low_z.T_measurement_residual.translation().z() = 0.1;

    auto best = manager.selectBestPerQuery({strong_fitness, low_z});
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.front().match_id, 10);
}

TEST(LoopClosureManagerTest, BuildLoopEdgesRespectsDirection)
{
    Config config;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.verified = true;
    loop.inlier_ratio = 1.0;
    loop.fitness_score = 0.1;
    loop.T_match_query = Eigen::Isometry3d::Identity();
    loop.T_match_query.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);

    auto edges_qm = manager.buildLoopEdges({ loop }, LoopEdgeDirection::QueryToMatch);
    ASSERT_EQ(edges_qm.size(), 1u);
    EXPECT_EQ(edges_qm.front().from_id, 1);
    EXPECT_EQ(edges_qm.front().to_id, 2);
    EXPECT_TRUE(edges_qm.front().measurement.isApprox(loop.T_match_query.inverse(), 1e-9));

    auto edges_mq = manager.buildLoopEdges({ loop }, LoopEdgeDirection::MatchToQuery);
    ASSERT_EQ(edges_mq.size(), 1u);
    EXPECT_EQ(edges_mq.front().from_id, 2);
    EXPECT_EQ(edges_mq.front().to_id, 1);
    EXPECT_TRUE(edges_mq.front().measurement.isApprox(loop.T_match_query, 1e-9));
}

TEST(LoopClosureManagerTest, ApplyEdgeModelKeepsFull6DofWhenVerticalResidualIsObservable)
{
    Config config;
    config.loop_noise_position = 0.5;
    config.loop_noise_rotation = 0.5;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.information.diagonal() << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    loop.T_measurement_residual = Eigen::Isometry3d::Identity();
    loop.T_measurement_residual.translation().z() = 0.1;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::Full6Dof);
    EXPECT_FALSE(modeled.vertical_downweighted);
    EXPECT_TRUE(modeled.information.isApprox(loop.information, 1e-12));
}

TEST(LoopClosureManagerTest, ApplyEdgeModelDoesNotDownweightVerticalAxes)
{
    Config config;
    config.loop_noise_position = 0.5;
    config.loop_noise_rotation = 0.5;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.information.diagonal() << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    loop.T_measurement_residual = Eigen::Isometry3d::Identity();
    loop.T_measurement_residual.translation().z() = 3.0;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::Full6Dof);
    EXPECT_FALSE(modeled.vertical_downweighted);
    EXPECT_TRUE(modeled.information.isApprox(loop.information, 1e-12));
}

TEST(LoopClosureManagerTest, ApplyEdgeModelDoesNotRejectVerticalOutlier)
{
    Config config;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.T_measurement_residual = Eigen::Isometry3d::Identity();
    loop.T_measurement_residual.translation().z() = 4.5;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::Full6Dof);
    EXPECT_EQ(manager.buildLoopEdges({modeled}, LoopEdgeDirection::MatchToQuery).size(), 1u);
}

TEST(LoopClosureManagerTest, ApplyEdgeModelDoesNotRejectYawInconsistentOutlier)
{
    Config config;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.candidate_yaw_diff_rad = M_PI;
    loop.T_measurement_residual = Eigen::Isometry3d::Identity();

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::Full6Dof);
    EXPECT_EQ(manager.buildLoopEdges({modeled}, LoopEdgeDirection::MatchToQuery).size(), 1u);
}

TEST(LoopClosureManagerTest, BuildLoopEdgesSupportsExplicitXYYawLoopEdge)
{
    Config config;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 10;
    loop.match_id = 3;
    loop.edge_mode = LoopEdgeMode::XYYaw;
    loop.T_match_query = Eigen::Isometry3d::Identity();

    const auto edges = manager.buildLoopEdges({loop}, LoopEdgeDirection::MatchToQuery);
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].from_id, 3);
    EXPECT_EQ(edges[0].to_id, 10);
    EXPECT_EQ(edges[0].type, EdgeType::LOOP);
    EXPECT_EQ(edges[0].constraint_mode, EdgeConstraintMode::XY_YAW);
}

TEST(LoopClosureManagerTest, ApplyEdgesCallsOptimizer)
{
    Config config;
    LoopClosureManager manager(config);

    EdgeInfo e1;
    e1.from_id = 1;
    e1.to_id = 2;
    e1.type = EdgeType::LOOP;

    EdgeInfo e2;
    e2.from_id = 3;
    e2.to_id = 4;
    e2.type = EdgeType::LOOP;

    MockLoopOptimizer mock;
    EXPECT_CALL(mock, addLoopEdge(testing::_)).Times(2);
    EXPECT_CALL(mock, incrementalOptimize()).Times(1).WillOnce(testing::Return(true));

    bool optimized = manager.applyEdges({ e1, e2 }, mock);
    EXPECT_TRUE(optimized);
}

TEST(LoopClosureManagerTest, ApplyEdgesReturnsFalseWhenOptimizerRejects)
{
    Config config;
    LoopClosureManager manager(config);

    EdgeInfo edge;
    edge.from_id = 1;
    edge.to_id = 2;
    edge.type = EdgeType::LOOP;

    MockLoopOptimizer mock;
    EXPECT_CALL(mock, addLoopEdge(testing::_)).Times(1);
    EXPECT_CALL(mock, incrementalOptimize()).Times(1).WillOnce(testing::Return(false));

    const bool optimized = manager.applyEdges({edge}, mock);
    EXPECT_FALSE(optimized);
}

TEST(LoopClosureManagerTest, ApplyEdgesEmptyDoesNothing)
{
    Config config;
    LoopClosureManager manager(config);

    MockLoopOptimizer mock;
    EXPECT_CALL(mock, addLoopEdge(testing::_)).Times(0);
    EXPECT_CALL(mock, incrementalOptimize()).Times(0);

    bool optimized = manager.applyEdges({}, mock);
    EXPECT_FALSE(optimized);
}

} // namespace n3mapping
