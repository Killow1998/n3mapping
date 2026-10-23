#include "n3mapping/pcl_compat.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <pcl/common/transforms.h>

#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_referee.h"
#include "n3mapping/loop_segment_consistency.h"
#include "n3mapping/loop_verification_pipeline.h"
#include "n3mapping/loop_verifier.h"

namespace n3mapping {
namespace {

Keyframe::PointCloudT::Ptr makeTinyCloud()
{
    auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
    pcl::PointXYZI point;
    point.x = 0.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 1.0f;
    cloud->push_back(point);
    return cloud;
}

Keyframe::PointCloudT::Ptr makeRegistrationCloud()
{
    auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
    for (int a = -5; a <= 5; ++a) {
        for (int b = -5; b <= 5; ++b) {
            pcl::PointXYZI floor;
            floor.x = 0.2f * static_cast<float>(a);
            floor.y = 0.2f * static_cast<float>(b);
            floor.z = 0.03f * static_cast<float>((a + 2 * b) % 3);
            floor.intensity = 1.0f;
            cloud->push_back(floor);

            pcl::PointXYZI wall_x;
            wall_x.x = 1.5f;
            wall_x.y = floor.x;
            wall_x.z = 0.2f * static_cast<float>(b + 6);
            wall_x.intensity = 2.0f;
            cloud->push_back(wall_x);

            pcl::PointXYZI wall_y;
            wall_y.x = floor.x;
            wall_y.y = -1.2f;
            wall_y.z = 0.2f * static_cast<float>(b + 6);
            wall_y.intensity = 3.0f;
            cloud->push_back(wall_y);
        }
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
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

    // A descriptor hit is now enough on its own, so a bundle that is weak
    // everywhere else has to be weak there too before it can be rejected.
    LoopFeatures bad;
    bad.descriptor_score = 0.0;
    bad.spatial_score = 0.1;
    bad.geometric_overlap = 0.0;
    bad.temporal_gap = 0.1;
    bad.local_map_consistency = 0.1;
    bad.segment_consistency = 0.0;
    bad.segment_support = 1.0;
    EXPECT_EQ(LoopReferee::decide(bad), LoopDecision::Reject);

    // What the same bundle does once a descriptor backs it, stated here so the
    // loosening is visible rather than implied.
    LoopFeatures bad_but_recognised = bad;
    bad_but_recognised.descriptor_score = 0.8;
    EXPECT_EQ(LoopReferee::decide(bad_but_recognised), LoopDecision::Accept);
}

TEST(LoopRefereeTest, RejectsInconsistentSegmentsOnlyWithoutADescriptor)
{
    LoopFeatures features;
    features.spatial_score = 1.0;
    features.geometric_overlap = 1.0;
    features.temporal_gap = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 1.0;
    features.segment_consistency = 0.25;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "unconfirmed_weak_segment");
    EXPECT_EQ(decision.risk_flags, "segment");

    // The segment statistic rests on two neighbour pairs. With the window sized
    // to the drift it is the descriptor that carries the independent
    // confirmation, and requiring both discarded 86 candidates that registered
    // at a median fitness of 0.052.
    features.descriptor_score = 1.0;
    EXPECT_EQ(LoopReferee::evaluate(features).decision, LoopDecision::Accept);
}

TEST(LoopRefereeTest, RejectsSpatialOnlyWithoutDescriptorSupport)
{
    LoopFeatures features;
    features.spatial_only = true;
    features.spatial_score = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 1.0;
    features.segment_consistency = 0.75;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Reject);
    EXPECT_EQ(decision.reason, "spatial_only_unconfirmed");
    EXPECT_EQ(decision.risk_flags, "source");

    // A spatial candidate is proposed by the drifted poses, so it needs
    // confirmation from somewhere those poses cannot reach. Neighbouring
    // keyframes that all register against the same match are such a source, and
    // a stronger one than a descriptor hit; insisting on the descriptor
    // specifically discarded 26 of the session's 57 best-evidenced candidates.
    features.segment_consistency = 1.0;
    EXPECT_EQ(LoopReferee::evaluate(features).decision, LoopDecision::Accept);
}

TEST(LoopRefereeTest, DoesNotJudgeALoopByHowFarItAsksToMove)
{
    // predicted_translation_norm is the separation the drifted poses report,
    // which grows precisely for the loops that would repair the drift. Using it
    // as evidence against a candidate rejected the corrections worth making.
    LoopFeatures features;
    features.descriptor_supported = true;
    features.descriptor_score = 1.0;
    features.local_map_consistency = 1.0;
    features.segment_support = 0.5;
    features.segment_consistency = 0.5;
    features.predicted_translation_norm = LoopReferee::kLargePredictedTranslationM + 0.1;

    const auto decision = LoopReferee::evaluate(features);
    EXPECT_EQ(decision.decision, LoopDecision::Accept);
    EXPECT_EQ(decision.reason, "descriptor_geometry_consistent");

    // What still rejects it is the absence of any confirmation, at any distance.
    features.descriptor_supported = false;
    features.descriptor_score = 0.0;
    const auto unconfirmed = LoopReferee::evaluate(features);
    EXPECT_EQ(unconfirmed.decision, LoopDecision::Reject);
    EXPECT_EQ(unconfirmed.reason, "unconfirmed_weak_segment");

    features.predicted_translation_norm = 0.1;
    EXPECT_EQ(LoopReferee::evaluate(features).decision, LoopDecision::Reject);
}

TEST(LoopRefereeTest, RejectsAYawFlipWhateverTheSegmentSays)
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
    EXPECT_EQ(decision.reason, "yaw_flip");
    EXPECT_EQ(decision.risk_flags, "yaw");

    // A correction near half a turn is a flipped match on its own evidence, so
    // this rule keeps rejecting once the segment agrees -- the one place the
    // referee was made stricter rather than looser.
    features.segment_translation_median = 0.0;
    EXPECT_EQ(LoopReferee::evaluate(features).reason, "yaw_flip");
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

TEST(LoopVerifierEvidenceTest, PreparedPathUsesConfiguredSerializableInformation)
{
    Config config;
    config.num_threads = 1;
    config.gicp_downsampling_resolution = 0.05;
    config.gicp_max_correspondence_distance = 2.0;
    config.gicp_fitness_threshold = 1.0;
    config.reloc_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 1.0;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_noise_position = 0.2;
    config.loop_noise_position_z = 0.4;
    config.loop_noise_rotation = 0.5;
    config.loop_axis_weighting_enable = false;
    config.loop_use_icp_information = false;

    const auto cloud = makeRegistrationCloud();
    const auto target = Keyframe::create(
        3, 1.0, Eigen::Isometry3d::Identity(), cloud);
    const auto source = Keyframe::create(
        10, 2.0, Eigen::Isometry3d::Identity(), cloud);
    LoopCandidate candidate;
    candidate.query_id = source->id;
    candidate.match_id = target->id;

    PointCloudMatcher matcher(config);
    const auto registration = matcher.alignCloud(
        target->cloud, source->cloud, Eigen::Isometry3d::Identity());
    const LoopVerification verification =
        LoopVerifier(config).verifyQueryToMatchResult(
            candidate, source, target, source->cloud, target->cloud,
            registration);

    ASSERT_TRUE(verification.loop.verified);
    const auto& information = verification.loop.information;
    EXPECT_NEAR(information(0, 0), 25.0, 1e-9);
    EXPECT_NEAR(information(1, 1), 25.0, 1e-9);
    EXPECT_NEAR(information(2, 2), 6.25, 1e-9);
    EXPECT_NEAR(information(3, 3), 4.0, 1e-9);
    EXPECT_NEAR(information(4, 4), 4.0, 1e-9);
    EXPECT_NEAR(information(5, 5), 4.0, 1e-9);
    EXPECT_TRUE(information.isApprox(
        information.diagonal().asDiagonal().toDenseMatrix(), 1e-12));

    config.loop_use_icp_information = true;
    const LoopVerification icp_information =
        LoopVerifier(config).verifyQueryToMatchResult(
            candidate, source, target, source->cloud, target->cloud,
            registration);
    EXPECT_TRUE(icp_information.loop.information.isApprox(
        icp_information.loop.information.transpose(), 1e-12));
}

TEST(LoopVerificationPipelineTest,
     QueryScanCannotSupplyItsOwnTargetEvidence)
{
    Config config;
    config.num_threads = 1;
    config.gicp_submap_size = 5;
    const auto query_cloud = makeRegistrationCloud();
    auto unrelated = pcl::make_shared<Keyframe::PointCloudT>(*query_cloud);
    for (auto& point : *unrelated) point.x += 50.0f;
    KeyframeManager keyframes(config);
    keyframes.loadKeyframes({
        Keyframe::create(0, 1.0, Eigen::Isometry3d::Identity(), unrelated),
        Keyframe::create(5, 2.0, Eigen::Isometry3d::Identity(), query_cloud)});
    PointCloudMatcher matcher(config);
    GraphOptimizer optimizer(config);
    LoopClosureManager manager(config);
    LoopVerificationPipeline pipeline(
        config, keyframes, matcher, optimizer, manager);
    LoopCandidate candidate;
    candidate.query_id = 5;
    candidate.match_id = 0;
    candidate.source_flags = LoopCandidate::SOURCE_SPATIAL;
    candidate.spatial_score = 1.0;

    const auto result = pipeline.evaluate(candidate, {});
    EXPECT_TRUE(result.registration_attempted);
    EXPECT_FALSE(result.loop.verified);
    EXPECT_EQ(result.verification.match_result.num_inliers, 0u);
}

TEST(LoopVerificationPipelineTest,
     SmallRollCorrectionIsNotAHalfTurn)
{
    Config config;
    config.num_threads = 1;
    const auto cloud = makeRegistrationCloud();
    for (double roll : {-0.01, 0.01}) {
        KeyframeManager keyframes(config);
        auto predicted = Eigen::Isometry3d::Identity();
        predicted.linear() =
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        auto match = Keyframe::create(
            0, 1.0, Eigen::Isometry3d::Identity(), cloud);
        auto query = Keyframe::create(10, 2.0, predicted, cloud);
        keyframes.loadKeyframes({match, query});
        PointCloudMatcher matcher(config);
        GraphOptimizer optimizer(config);
        LoopClosureManager manager(config);
        LoopVerificationPipeline pipeline(
            config, keyframes, matcher, optimizer, manager);
        LoopCandidate candidate;
        candidate.query_id = query->id;
        candidate.match_id = match->id;
        candidate.source_flags = LoopCandidate::SOURCE_RHPD;
        candidate.descriptor_score = 1.0;

        const auto result = pipeline.evaluate(candidate, {});
        ASSERT_TRUE(result.registration_attempted);
        EXPECT_LT(result.verification.icp_rotation_norm, 0.02);
        EXPECT_TRUE(result.loop.verified) << result.reject_reason;
        EXPECT_TRUE(result.verification.T_measured_match_query.isApprox(
            Eigen::Isometry3d::Identity(), 0.005));
    }
}

TEST(LoopVerifierEvidenceTest, PretransformedSourceKeepsMeasurementUncertainty)
{
    Config config;
    config.num_threads = 1;
    config.loop_use_icp_information = true;
    auto query_pose = poseAt(3.0, -2.0, 0.4);
    const auto raw = makeRegistrationCloud();
    auto transformed = pcl::make_shared<Keyframe::PointCloudT>();
    pcl::transformPointCloud(*raw, *transformed, query_pose.matrix().cast<float>());
    const auto match = Keyframe::create(0, 1.0, Eigen::Isometry3d::Identity(), transformed);
    const auto query = Keyframe::create(10, 2.0, query_pose, raw);
    LoopCandidate candidate;
    candidate.query_id = query->id;
    candidate.match_id = match->id;
    PointCloudMatcher matcher(config);
    const auto result = LoopVerifier(config).verifyPreparedSubmaps(
        candidate, query, match, transformed, transformed, matcher);
    ASSERT_TRUE(result.loop.verified);
    const gtsam::Pose3 prediction(query_pose.matrix());
    const gtsam::Pose3 correction(result.match_result.T_target_source.matrix());
    const gtsam::Pose3 measurement(result.T_measured_match_query.matrix());
    gtsam::Vector6 perturbation;
    perturbation << 0.001, -0.002, 0.003, 0.01, 0.02, -0.03;
    const auto displaced = measurement * gtsam::Pose3::Expmap(perturbation);
    const gtsam::Vector6 correction_error =
        gtsam::Pose3::Logmap(correction.between(displaced * prediction.inverse()));
    Eigen::Matrix<double, 6, 1> measurement_error, raw_error;
    measurement_error << perturbation.tail<3>(), perturbation.head<3>();
    raw_error << correction_error.tail<3>(), correction_error.head<3>();
    const double expected =
        raw_error.dot(result.match_result.information * raw_error);
    EXPECT_NEAR(measurement_error.dot(result.loop.information * measurement_error),
                expected, expected * 1e-9);
}

TEST(LoopVerificationPipelineTest,
     CrossSessionRejectsNewSessionMatchBeforeRegistration)
{
    Config config;
    KeyframeManager keyframes(config);
    const auto cloud = makeRegistrationCloud();
    auto match = Keyframe::create(
        0, 1.0, Eigen::Isometry3d::Identity(), cloud);
    auto query = Keyframe::create(
        10, 2.0, Eigen::Isometry3d::Identity(), cloud);
    keyframes.loadKeyframes({match, query});
    match->is_from_loaded_map = false;
    query->is_from_loaded_map = false;

    PointCloudMatcher matcher(config);
    GraphOptimizer optimizer(config);
    LoopClosureManager loop_closure_manager(config);
    LoopVerificationPipeline pipeline(
        config, keyframes, matcher, optimizer, loop_closure_manager);

    LoopCandidate candidate;
    candidate.query_id = query->id;
    candidate.match_id = match->id;
    LoopVerificationContext context;
    context.cross_session = true;
    const auto result = pipeline.evaluate(candidate, context);

    EXPECT_FALSE(result.registration_attempted);
    EXPECT_FALSE(result.loop.verified);
    EXPECT_EQ(result.reject_stage, "session_boundary");
    EXPECT_EQ(result.reject_reason,
              "candidate_not_loaded_map_to_new_session");
}

TEST(LoopVerificationPipelineTest,
     DescriptorSeedDoesNotDependOnDriftedGraphTranslation)
{
    Config config;
    config.num_threads = 1;
    config.loop_max_range = 0.5;
    config.loop_fitness_threshold = 1.0;
    config.loop_min_inlier_ratio = 0.0;
    KeyframeManager keyframes(config);
    const auto cloud = makeRegistrationCloud();
    auto match = Keyframe::create(
        0, 1.0, Eigen::Isometry3d::Identity(), cloud);
    auto query = Keyframe::create(
        10, 2.0, poseAt(10.0), cloud);
    keyframes.loadKeyframes({match, query});
    query->is_from_loaded_map = false;

    PointCloudMatcher matcher(config);
    GraphOptimizer optimizer(config);
    LoopClosureManager loop_closure_manager(config);
    LoopVerificationPipeline pipeline(
        config, keyframes, matcher, optimizer, loop_closure_manager);

    LoopCandidate candidate;
    candidate.query_id = query->id;
    candidate.match_id = match->id;
    candidate.source_flags =
        LoopCandidate::SOURCE_RHPD | LoopCandidate::SOURCE_SC;
    candidate.sc_distance = 0.0;
    candidate.yaw_diff_rad = 0.0f;
    candidate.descriptor_score = 1.0;
    LoopVerificationContext context;
    context.cross_session = true;
    const auto result = pipeline.evaluate(candidate, context);

    EXPECT_TRUE(result.registration_attempted);
    EXPECT_TRUE(result.descriptor_seeded);
    EXPECT_GT(result.registration_hypothesis_count, 1);
    EXPECT_TRUE(result.loop.verified);
    EXPECT_LT(result.verification.match_result.fitness_score, 1.0e-3);
    EXPECT_TRUE(result.verification.T_measured_match_query.isApprox(
        Eigen::Isometry3d::Identity(), 1.0e-2));

    LoopVerificationContext guarded_context;
    guarded_context.cross_session = true;
    guarded_context.pose_visibility_evaluator =
        [](const Eigen::Isometry3d&) {
            VisibilityConsistencyResult visibility;
            visibility.valid = true;
            visibility.consistency_ratio = 0.4;
            visibility.evidence_log_odds = -0.1;
            return visibility;
        };
    const auto visibility_rejected =
        pipeline.evaluate(candidate, guarded_context);
    EXPECT_FALSE(visibility_rejected.loop.verified);
    EXPECT_EQ(visibility_rejected.reject_stage, "visibility");
    EXPECT_EQ(visibility_rejected.reject_reason,
              "loaded_map_visibility_nonpositive");
}

TEST(LoopVerificationPipelineTest,
     CrossSessionDefersConstraintWithoutNeighborConsensus)
{
    Config config;
    KeyframeManager keyframes(config);
    const auto cloud = makeTinyCloud();
    auto match = Keyframe::create(
        0, 1.0, Eigen::Isometry3d::Identity(), cloud);
    auto query = Keyframe::create(
        100, 2.0, Eigen::Isometry3d::Identity(), cloud);
    keyframes.loadKeyframes({match, query});
    query->is_from_loaded_map = false;

    PointCloudMatcher matcher(config);
    GraphOptimizer optimizer(config);
    LoopClosureManager loop_closure_manager(config);
    LoopVerificationPipeline pipeline(
        config, keyframes, matcher, optimizer, loop_closure_manager);

    VerifiedLoop loop;
    loop.query_id = query->id;
    loop.match_id = match->id;
    loop.verified = true;
    const auto result = pipeline.evaluateConstraints(
        {loop}, LoopEdgeDirection::MatchToQuery, {true}).front();

    EXPECT_FALSE(result.accepted);
    EXPECT_FALSE(result.has_edge);
    EXPECT_EQ(result.consensus.decision, LoopConsensusDecision::Defer);
    EXPECT_EQ(result.reject_stage, "consensus");
    EXPECT_EQ(result.reject_reason, "session_merge_consensus_deferred");
}

TEST(LoopVerificationPipelineTest, DebugTrialDoesNotChangeConstraintDecision)
{
    Config config;
    config.num_threads = 1;
    KeyframeManager keyframes(config);
    const auto cloud = makeRegistrationCloud();
    std::vector<Keyframe::Ptr> frames;
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    std::vector<EdgeInfo> edges;
    for (int64_t id = 0; id <= 80; ++id) {
        frames.push_back(Keyframe::create(
            id, static_cast<double>(id), Eigen::Isometry3d::Identity(), cloud));
        nodes.emplace_back(id, Eigen::Isometry3d::Identity());
        if (id > 0) edges.emplace_back(id - 1, id,
            Eigen::Isometry3d::Identity(), Eigen::Matrix<double, 6, 6>::Identity(),
            EdgeType::ODOMETRY);
    }
    ASSERT_TRUE(keyframes.loadKeyframes(frames));
    PointCloudMatcher matcher(config);
    GraphOptimizer optimizer(config);
    ASSERT_TRUE(optimizer.loadGraph(nodes, edges));
    LoopClosureManager manager(config);
    VerifiedLoop loop;
    loop.query_id = 80;
    loop.match_id = 10;
    loop.verified = true;
    loop.fitness_score = 0.0;
    loop.inlier_ratio = 1.0;
    const auto quiet = LoopVerificationPipeline(
        config, keyframes, matcher, optimizer, manager).evaluateConstraints(
            {loop}, LoopEdgeDirection::MatchToQuery).front();
    config.loop_debug_enable = true;
    const auto debug = LoopVerificationPipeline(
        config, keyframes, matcher, optimizer, manager).evaluateConstraints(
            {loop}, LoopEdgeDirection::MatchToQuery).front();
    ASSERT_TRUE(quiet.accepted);
    EXPECT_TRUE(debug.accepted);
    EXPECT_TRUE(quiet.graph_trial.success);
    EXPECT_FALSE(quiet.has_consensus_estimator_trial);
    EXPECT_TRUE(debug.has_consensus_estimator_trial);
    EXPECT_TRUE(debug.consensus_estimator_trial.success);
    EXPECT_TRUE(quiet.edge.measurement.isApprox(debug.edge.measurement, 1.0e-12));
    EXPECT_EQ(optimizer.getEdges().size(), edges.size());  // Trials do not commit.
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

TEST(LoopSegmentConsistencyTest, AssociatesPositionAndHeadingBeforeCheckingConsistency)
{
    Config config;
    config.loop_max_icp_translation = 2.0;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> frames;
    for (const auto& item : std::vector<std::pair<int64_t, Eigen::Isometry3d>>{
             {6, poseAt(-2.02)}, {7, poseAt(-1.02)},
             {8, poseAt(-2.0, 0.0, 1.0)}, {9, poseAt(-1.0, 0.0, 1.0)},
             {10, poseAt(0.0)}, {28, poseAt(98.0)},
             {29, poseAt(99.0)}, {30, poseAt(100.0)}}) {
        frames.push_back(Keyframe::create(item.first, item.first,
                                         item.second, makeTinyCloud()));
    }
    ASSERT_TRUE(manager.loadKeyframes(frames));
    VerifiedLoop loop;
    loop.query_id = 30;
    loop.match_id = 10;
    auto result = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(result.consensus_inlier_count, 2);
    EXPECT_EQ(result.recommendation, "consistent");
    // A small summed score can still violate one axis gate. Prefer a valid
    // alternative over a closer point whose heading is outside the same gate.
    frames[0]->pose_optimized = poseAt(-3.8, 0.0, 0.4);
    frames[1]->pose_optimized = poseAt(-2.8, 0.0, 0.4);
    frames[2]->pose_optimized = poseAt(-2.0, 0.0, 0.55);
    frames[3]->pose_optimized = poseAt(-1.0, 0.0, 0.55);
    ASSERT_TRUE(manager.loadKeyframes(frames));
    result = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(result.consensus_inlier_count, 2);
    // With no heading-consistent alternative, the same yaw gate still rejects.
    frames[0]->pose_optimized = poseAt(-2.02, 0.0, 1.0);
    frames[1]->pose_optimized = poseAt(-1.02, 0.0, 1.0);
    ASSERT_TRUE(manager.loadKeyframes(frames));
    result = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(result.consensus_inlier_count, 0);
    EXPECT_EQ(result.recommendation, "inconsistent");
}

TEST(LoopSegmentConsistencyTest, SupportedDirectionSurvivesAnAdditionalObservation)
{
    Config config;
    config.loop_max_icp_translation = 2.0;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> frames;
    for (const auto& item : std::vector<std::pair<int64_t, Eigen::Isometry3d>>{
             {8, poseAt(-3.0, 0.0, 0.4)}, {9, poseAt(1.0)},
             {10, poseAt(0.0)}, {11, poseAt(2.0, 0.0, 0.4)},
             {28, poseAt(98.0)}, {29, poseAt(99.0)},
             {30, poseAt(100.0)}, {31, poseAt(101.0)}}) {
        frames.push_back(Keyframe::create(item.first, item.first,
                                         item.second, makeTinyCloud()));
    }
    VerifiedLoop loop;
    loop.query_id = 30;
    loop.match_id = 10;
    ASSERT_TRUE(manager.loadKeyframes({frames.begin(), frames.end() - 1}));
    EXPECT_EQ(computeLoopSegmentConsistency(config, manager, loop, 2).recommendation,
              "consistent");
    ASSERT_TRUE(manager.loadKeyframes(frames));
    const auto observed = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(observed.pair_count, 3);
    EXPECT_EQ(observed.consensus_inlier_count, 3);
    EXPECT_EQ(observed.direction, "same");
    EXPECT_EQ(observed.recommendation, "consistent");

    // If neither direction explains the observed neighbor, it still counts
    // against support; selecting a direction cannot erase that contradiction.
    frames[3]->pose_optimized = poseAt(20.0, 0.0, 0.4);
    ASSERT_TRUE(manager.loadKeyframes(frames));
    const auto contradicting = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(contradicting.consensus_inlier_count, 2);
    EXPECT_EQ(contradicting.recommendation, "inconsistent");
}

TEST(LoopSegmentConsistencyTest, UnobservedFutureIsNotAContradictingMeasurement)
{
    Config config;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> keyframes;
    for (int id = 0; id <= 12; ++id) {
        keyframes.push_back(Keyframe::create(
            id, static_cast<double>(id), poseAt(id), makeTinyCloud()));
    }
    VerifiedLoop loop;
    loop.query_id = 10;
    loop.match_id = 2;
    manager.loadKeyframes({keyframes.begin(), keyframes.begin() + 11});
    const auto live = computeLoopSegmentConsistency(config, manager, loop, 2);
    ASSERT_EQ(live.pair_count, 2);
    EXPECT_EQ(live.valid_pair_count, 2);
    EXPECT_EQ(live.recommendation, "consistent");
    manager.loadKeyframes(keyframes);
    const auto complete = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(complete.pair_count, 4);
    EXPECT_EQ(complete.valid_pair_count, 4);
    EXPECT_DOUBLE_EQ(live.consensus_ratio, complete.consensus_ratio);

    // Once contradictory motion is observed it must count against the loop.
    keyframes[11]->pose_optimized = poseAt(100.0);
    keyframes[12]->pose_optimized = poseAt(200.0);
    manager.loadKeyframes(keyframes);
    const auto contradictory = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(contradictory.pair_count, 4);
    EXPECT_EQ(contradictory.valid_pair_count, 4);
    EXPECT_EQ(contradictory.recommendation, "inconsistent");
}

TEST(LoopSegmentConsistencyTest, MissingMatchesAndSingleObservationsAreInsufficient)
{
    Config config;
    KeyframeManager manager(config);
    std::vector<Keyframe::Ptr> keyframes;
    for (int id : {2, 8, 9, 10}) {
        keyframes.push_back(Keyframe::create(
            id, static_cast<double>(id), poseAt(id), makeTinyCloud()));
    }
    VerifiedLoop loop;
    loop.query_id = 10;
    loop.match_id = 2;
    manager.loadKeyframes(keyframes);
    const auto missing_match = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(missing_match.pair_count, 2);
    EXPECT_EQ(missing_match.valid_pair_count, 0);
    EXPECT_EQ(missing_match.recommendation, "insufficient_support");
    keyframes = {keyframes[0], keyframes[2], keyframes[3],
        Keyframe::create(1, 1.0, poseAt(1), makeTinyCloud())};
    manager.loadKeyframes(keyframes);
    const auto single = computeLoopSegmentConsistency(config, manager, loop, 2);
    EXPECT_EQ(single.pair_count, 1);
    EXPECT_EQ(single.valid_pair_count, 1);
    EXPECT_EQ(single.recommendation, "insufficient_support");
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

TEST(LoopClosureManagerTest, SameQueryConsensusRejectsOnePoseOutlier)
{
    Config config;
    config.loop_same_query_consensus_translation_m = 1.0;
    config.loop_same_query_consensus_rotation_rad = 0.2;
    LoopClosureManager manager(config);

    std::map<int64_t, Keyframe::Ptr> keyframes;
    keyframes[100] = Keyframe::create(100, 0.0, poseAt(9.8), makeTinyCloud());
    std::vector<VerifiedLoop> loops;
    for (int match_id = 0; match_id < 6; ++match_id) {
        const auto match_pose = poseAt(static_cast<double>(match_id));
        keyframes[match_id] =
            Keyframe::create(match_id, 0.0, match_pose, makeTinyCloud());
        VerifiedLoop loop;
        loop.query_id = 100;
        loop.match_id = match_id;
        loop.verified = true;
        loop.fitness_score = 0.1 + 0.01 * match_id;
        const double implied_x = match_id == 1
                                     ? 13.2
                                     : 10.0 + 0.02 * match_id;
        loop.T_match_query = match_pose.inverse() * poseAt(implied_x);
        loops.push_back(loop);
    }
    // Make the geometric outlier look best to the historical scalar ranking.
    loops[1].fitness_score = 0.001;

    const auto selection =
        manager.selectSameQueryConsensus(loops, keyframes);
    ASSERT_EQ(selection.selected.size(), 5u);
    EXPECT_TRUE(std::none_of(selection.selected.begin(),
                             selection.selected.end(),
                             [](const VerifiedLoop& loop) {
                                 return loop.match_id == 1;
                             }));
    const auto rejected = selection.rejected.find({100, 1});
    ASSERT_NE(rejected, selection.rejected.end());
    EXPECT_EQ(rejected->second, "same_query_consensus_outlier");
}

TEST(LoopClosureManagerTest, SameQueryConsensusFallsBackWhenNoCandidateAgrees)
{
    Config config;
    config.loop_same_query_consensus_translation_m = 1.0;
    config.loop_same_query_consensus_rotation_rad = 0.2;
    LoopClosureManager manager(config);

    std::map<int64_t, Keyframe::Ptr> keyframes;
    keyframes[100] = Keyframe::create(100, 0.0, poseAt(10.0), makeTinyCloud());
    keyframes[0] = Keyframe::create(0, 0.0, poseAt(0.0), makeTinyCloud());
    keyframes[1] = Keyframe::create(1, 0.0, poseAt(1.0), makeTinyCloud());

    VerifiedLoop worse;
    worse.query_id = 100;
    worse.match_id = 0;
    worse.verified = true;
    worse.fitness_score = 0.2;
    worse.T_match_query = poseAt(10.0);

    VerifiedLoop better = worse;
    better.match_id = 1;
    better.fitness_score = 0.1;
    better.T_match_query = poseAt(19.0);

    const auto selection =
        manager.selectSameQueryConsensus({worse, better}, keyframes);
    ASSERT_EQ(selection.selected.size(), 1u);
    EXPECT_EQ(selection.selected.front().match_id, 1);
    EXPECT_EQ(selection.rejected.at({100, 0}),
              "same_query_consensus_no_support");
}

TEST(LoopClosureManagerTest, SameQueryConsensusDoesNotChooseBetweenTiedClusters)
{
    Config config;
    config.loop_same_query_consensus_translation_m = 1.0;
    config.loop_same_query_consensus_rotation_rad = 0.2;
    LoopClosureManager manager(config);

    std::map<int64_t, Keyframe::Ptr> keyframes;
    keyframes[100] = Keyframe::create(100, 0.0, poseAt(10.0), makeTinyCloud());
    std::vector<VerifiedLoop> loops;
    for (int match_id = 0; match_id < 4; ++match_id) {
        const auto match_pose = poseAt(static_cast<double>(match_id));
        keyframes[match_id] =
            Keyframe::create(match_id, 0.0, match_pose, makeTinyCloud());
        VerifiedLoop loop;
        loop.query_id = 100;
        loop.match_id = match_id;
        loop.verified = true;
        loop.fitness_score = match_id == 3 ? 0.01 : 0.2 + match_id;
        const double implied_x = match_id < 2
                                     ? 10.0 + 0.1 * match_id
                                     : 20.0 + 0.1 * (match_id - 2);
        loop.T_match_query = match_pose.inverse() * poseAt(implied_x);
        loops.push_back(loop);
    }

    const auto selection =
        manager.selectSameQueryConsensus(loops, keyframes);
    ASSERT_EQ(selection.selected.size(), 1u);
    EXPECT_EQ(selection.selected.front().match_id, 3);
    EXPECT_EQ(selection.rejected.at({100, 0}),
              "same_query_consensus_ambiguous");
    EXPECT_EQ(selection.rejected.at({100, 1}),
              "same_query_consensus_ambiguous");
    EXPECT_EQ(selection.rejected.at({100, 2}),
              "same_query_consensus_ambiguous");
}

TEST(LoopClosureManagerTest, SameQueryConsensusFailsClosedWithoutMatchPose)
{
    Config config;
    LoopClosureManager manager(config);

    std::map<int64_t, Keyframe::Ptr> keyframes;
    keyframes[100] = Keyframe::create(100, 0.0, poseAt(10.0), makeTinyCloud());

    VerifiedLoop loop;
    loop.query_id = 100;
    loop.match_id = 7;
    loop.verified = true;

    const auto selection =
        manager.selectSameQueryConsensus({loop}, keyframes);
    EXPECT_TRUE(selection.selected.empty());
    EXPECT_EQ(selection.rejected.at({100, 7}),
              "same_query_consensus_missing_match_pose");
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

TEST(LoopClosureManagerTest, ReversingFullEdgePreservesWhitenedError)
{
    LoopClosureManager manager(Config{});
    VerifiedLoop loop;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.verified = true;
    loop.T_match_query = poseAt(3.0, -2.0, 0.4);
    loop.information.diagonal() << 100.0, 200.0, 300.0, 1000.0, 2000.0, 3000.0;
    const auto reverse =
        manager.buildLoopEdges({loop}, LoopEdgeDirection::QueryToMatch).front();
    const gtsam::Pose3 measurement(loop.T_match_query.matrix());
    gtsam::Vector6 perturbation;
    perturbation << 0.001, -0.002, 0.003, 0.01, 0.02, -0.03;
    const auto predicted = measurement * gtsam::Pose3::Expmap(perturbation);
    const gtsam::Vector6 inverse_error =
        gtsam::Pose3::Logmap(measurement.inverse().between(predicted.inverse()));
    Eigen::Matrix<double, 6, 1> forward_error, reverse_error;
    forward_error << perturbation.tail<3>(), perturbation.head<3>();
    reverse_error << inverse_error.tail<3>(), inverse_error.head<3>();
    EXPECT_NEAR(forward_error.dot(loop.information * forward_error),
                reverse_error.dot(reverse.information * reverse_error), 1e-10);
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
