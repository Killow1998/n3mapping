#include <gtest/gtest.h>
#include "n3mapping/loop_detector.h"
#include "n3mapping/relocalization_candidate_evaluator.h"
#include "support/registration_fixture.h"

namespace n3mapping::test {
class RelocalizationCandidateEvaluatorTest : public RegistrationFixture {
 protected:
    struct FixedTargetProvider : RelocTargetProvider {
        PreparedRelocTarget target;
        PreparedRelocTarget getTarget(const RelocTargetRequest&) override { return target; }
        void clear() override {}
        RelocTargetProviderDiagnostics diagnostics() const override { return {}; }
    };

    std::vector<RelocalizationCandidateEvaluation> evaluateVisibilityFallback(double correspondence_distance) {
        config_.icp_refine_use_gicp = false;
        config_.gicp_max_correspondence_distance = correspondence_distance;
        matcher_ = std::make_unique<PointCloudMatcher>(config_);
        auto query = createPlaneCloud(2000, 10.0, 0.0);
        for (auto& point : *query) point.z = 3.0f;
        Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
        offset.translation().z() = 0.6;
        const auto registration_cloud = transformCloud(query, offset);

        // Distinct surfaces force the fallback branch; registration is real.
        FixedTargetProvider provider;
        provider.target.visibility_target = query;
        provider.target.registration_target = std::make_shared<PointCloudMatcher::PreparedTarget>(
            matcher_->prepareTargetCloud(registration_cloud));
        KeyframeManager keyframes(config_);
        LoopCandidate candidate;
        candidate.match_id = keyframes.addKeyframe(1.0, Eigen::Isometry3d::Identity(), query);
        RelocalizationCandidateEvaluator evaluator(config_, keyframes, *matcher_, provider);
        return evaluator.evaluate(matcher_->prepareSourceCloud(query), candidate, {},
            prepareVisibilityObservation(*query, visibilityOptionsFromConfig(config_)),
            Eigen::Isometry3d::Identity());
    }

};

TEST_F(RelocalizationCandidateEvaluatorTest, CandidateVisibilityFallbackCarriesItsOwnQuality) {
    const auto evaluations = evaluateVisibilityFallback(2.0);
    ASSERT_FALSE(evaluations.empty());
    bool saw_fallback = false;
    for (const auto& evaluation : evaluations) {
        const auto& match = evaluation.match;
        if (match.converged) continue;
        saw_fallback = true;
        EXPECT_NEAR(match.T_target_source.translation().z(), 0.0, 1e-12);
        EXPECT_EQ(match.termination, MatchTermination::FixedPoseEvaluation);
        EXPECT_NEAR(match.fitness_score, 0.18, 1e-5);
    }
    EXPECT_TRUE(saw_fallback);
}

TEST_F(RelocalizationCandidateEvaluatorTest, ReturningSeedReusesProposalButRequiresCurrentGeometry) {
    config_.icp_refine_use_gicp = false;
    config_.rhpd_yaw_hypotheses = 1;
    config_.reloc_min_confidence = 0.5;
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
    auto map = createPlaneCloud(2000, 10.0, 0.0);
    for (auto& point : *map) point.z = 3.0F;
    FixedTargetProvider provider;
    provider.target.visibility_target = map;
    provider.target.registration_target = std::make_shared<PointCloudMatcher::PreparedTarget>(
        matcher_->prepareTargetCloud(map));
    KeyframeManager keyframes(config_);
    LoopCandidate candidate;
    candidate.match_id = keyframes.addKeyframe(1.0, Eigen::Isometry3d::Identity(), map);
    RelocalizationCandidateEvaluator evaluator(config_, keyframes, *matcher_, provider);
    evaluator.beginSeedPlan({{candidate.match_id, 0.0}}, keyframes.revision());
    const auto evaluate = [&](const Keyframe::PointCloudT::Ptr& cloud,
                              RelocalizationPerformance* cost = nullptr) {
        return evaluator.evaluate(matcher_->prepareSourceCloud(cloud), candidate, {},
            prepareVisibilityObservation(*cloud, visibilityOptionsFromConfig(config_)),
            Eigen::Isometry3d::Identity(), cost);
    };
    ASSERT_FALSE(evaluate(map).empty());
    // Ranking temporarily loses this seed; returning to the same place should
    // check current geometry without discarding the useful pose proposal.
    for (int i = 0; i < 3; ++i) evaluator.beginSeedPlan({}, keyframes.revision());
    evaluator.beginSeedPlan({{candidate.match_id, 0.0}}, keyframes.revision());
    RelocalizationPerformance returning;
    ASSERT_FALSE(evaluate(map, &returning).empty());
    EXPECT_EQ(returning.seed_continuations, 1U);
    EXPECT_EQ(returning.seed_initializations, 0U);
    // The unreported 0.6 m sensor motion leaves the old pose within the loose
    // fitness gate (0.18 < 0.3), but below the actual confidence requirement.
    Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
    motion.translation().z() = 0.6;
    RelocalizationPerformance changed;
    const auto recovered = evaluate(transformCloud(map, motion), &changed);
    ASSERT_FALSE(recovered.empty());
    EXPECT_NEAR(recovered.front().match.T_target_source.translation().z(), -0.6, 1e-3);
    EXPECT_LT(recovered.front().match.fitness_score, 0.01);
    EXPECT_EQ(changed.seed_retries, 1U);
    // History remains bounded even when searches keep introducing other seeds.
    for (int i = 0; i < 32; ++i) evaluator.beginSeedPlan({}, keyframes.revision());
    evaluator.beginSeedPlan({{candidate.match_id, 0.0}}, keyframes.revision());
    RelocalizationPerformance expired;
    ASSERT_FALSE(evaluate(map, &expired).empty());
    EXPECT_EQ(expired.seed_continuations, 0U);
    EXPECT_EQ(expired.seed_initializations, 1U);
}

TEST_F(RelocalizationCandidateEvaluatorTest, NeighboringHeadingSectorReusesItsSharedYawSeeds) {
    config_.icp_refine_use_gicp = false;
    config_.rhpd_yaw_hypotheses = 1;
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
    auto map = createPlaneCloud(2000, 10.0, 0.0);
    for (auto& point : *map) point.z = 3.0F;
    FixedTargetProvider provider;
    provider.target.visibility_target = map;
    provider.target.registration_target = std::make_shared<PointCloudMatcher::PreparedTarget>(
        matcher_->prepareTargetCloud(map));
    KeyframeManager keyframes(config_);
    LoopCandidate candidate;
    candidate.match_id = keyframes.addKeyframe(1.0, Eigen::Isometry3d::Identity(), map);
    candidate.source_flags = LoopCandidate::SOURCE_RHPD | LoopCandidate::SOURCE_SC;
    candidate.sc_distance = 0.0;
    const float sector = static_cast<float>(2.0 * M_PI / config_.sc_num_sectors);
    RelocalizationCandidateEvaluator evaluator(config_, keyframes, *matcher_, provider);
    const auto source = matcher_->prepareSourceCloud(map);
    const auto observation = prepareVisibilityObservation(*map, visibilityOptionsFromConfig(config_));
    const auto evaluate = [&](float yaw, RelocalizationPerformance& cost) {
        candidate.yaw_diff_rad = yaw;
        std::vector<std::pair<int64_t, double>> seeds;
        for (double angle : buildDescriptorYawHypotheses(candidate, config_))
            seeds.emplace_back(candidate.match_id, angle);
        std::sort(seeds.begin(), seeds.end());
        evaluator.beginSeedPlan(seeds, keyframes.revision());
        return evaluator.evaluate(source, candidate, {}, observation,
                                  Eigen::Isometry3d::Identity(), &cost);
    };
    RelocalizationPerformance first, adjacent;
    ASSERT_FALSE(evaluate(sector, first).empty());
    ASSERT_FALSE(evaluate(2.0F * sector, adjacent).empty());
    // Only the new outer heading needs registration. The two overlapping
    // measured headings still receive fresh geometric checks. The one-bin
    // global grid now coincides with the measured heading, not absolute zero.
    EXPECT_EQ(adjacent.seed_initializations, 1U);
    EXPECT_EQ(adjacent.seed_continuations, 2U);
}

TEST_F(RelocalizationCandidateEvaluatorTest, UnmeasurableFallbackDoesNotBorrowEndpointQuality) {
    const auto evaluations = evaluateVisibilityFallback(0.25);
    ASSERT_FALSE(evaluations.empty());
    for (const auto& evaluation : evaluations) {
        const auto& match = evaluation.match;
        EXPECT_NEAR(match.T_target_source.translation().z(), 0.0, 1e-12);
        EXPECT_FALSE(match.success);
        EXPECT_FALSE(match.converged);
        EXPECT_EQ(match.termination, MatchTermination::Invalid);
        EXPECT_EQ(match.num_inliers, 0u);
        EXPECT_TRUE(match.information.isZero());
    }
}
}  // namespace n3mapping::test
