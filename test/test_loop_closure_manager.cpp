#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "n3mapping/loop_closure_manager.h"

namespace n3mapping {

class MockLoopOptimizer : public LoopOptimizerInterface
{
  public:
    MOCK_METHOD(void, addLoopEdge, (const EdgeInfo& edge), (override));
    MOCK_METHOD(bool, incrementalOptimize, (), (override));
};

TEST(LoopClosureManagerTest, FilterValidLoopsAppliesThresholds)
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
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().query_id, 1);
    EXPECT_EQ(result.front().match_id, 2);
}

TEST(LoopClosureManagerTest, FilterValidLoopsRejectsLargeCandidateResidualZ)
{
    Config config;
    config.loop_min_inlier_ratio = 0.3;
    config.loop_fitness_threshold = 1.0;
    config.loop_max_candidate_residual_z = 5.0;

    LoopClosureManager manager(config);

    VerifiedLoop valid;
    valid.query_id = 1;
    valid.match_id = 2;
    valid.verified = true;
    valid.inlier_ratio = 0.5;
    valid.fitness_score = 0.5;
    valid.candidate_residual = Eigen::Isometry3d::Identity();
    valid.candidate_residual.translation().z() = 4.9;

    VerifiedLoop bad_z = valid;
    bad_z.query_id = 3;
    bad_z.candidate_residual.translation().z() = 5.1;

    auto result = manager.filterValidLoops({valid, bad_z});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().query_id, 1);
}

TEST(LoopClosureManagerTest, FilterValidLoopsAllowsResidualZGateDisabled)
{
    Config config;
    config.loop_min_inlier_ratio = 0.3;
    config.loop_fitness_threshold = 1.0;
    config.loop_max_candidate_residual_z = 0.0;

    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.verified = true;
    loop.inlier_ratio = 0.5;
    loop.fitness_score = 0.5;
    loop.candidate_residual = Eigen::Isometry3d::Identity();
    loop.candidate_residual.translation().z() = 100.0;

    auto result = manager.filterValidLoops({loop});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().query_id, 1);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryChoosesLowestFitness)
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

    VerifiedLoop b = a;
    b.match_id = 11;
    b.fitness_score = 0.2;

    VerifiedLoop c = a;
    c.query_id = 2;
    c.match_id = 20;
    c.fitness_score = 0.4;

    auto best = manager.selectBestPerQuery({ a, b, c });
    ASSERT_EQ(best.size(), 2u);

    auto it = std::find_if(best.begin(), best.end(), [](const VerifiedLoop& loop) { return loop.query_id == 1; });
    ASSERT_TRUE(it != best.end());
    EXPECT_EQ(it->match_id, 11);
    EXPECT_DOUBLE_EQ(it->fitness_score, 0.2);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryBreaksFitnessTieWithVerticalResidual)
{
    Config config;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 10.0;
    config.loop_max_candidate_residual_z = 5.0;

    LoopClosureManager manager(config);

    VerifiedLoop high_z;
    high_z.query_id = 1;
    high_z.match_id = 10;
    high_z.verified = true;
    high_z.inlier_ratio = 1.0;
    high_z.fitness_score = 0.20;
    high_z.candidate_residual = Eigen::Isometry3d::Identity();
    high_z.candidate_residual.translation().z() = 4.0;

    VerifiedLoop low_z = high_z;
    low_z.match_id = 11;
    low_z.fitness_score = 0.205;
    low_z.candidate_residual.translation().z() = 0.1;

    auto best = manager.selectBestPerQuery({high_z, low_z});
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.front().match_id, 11);
}

TEST(LoopClosureManagerTest, SelectBestPerQueryKeepsClearlyBetterFitness)
{
    Config config;
    config.loop_min_inlier_ratio = 0.0;
    config.loop_fitness_threshold = 10.0;
    config.loop_max_candidate_residual_z = 5.0;

    LoopClosureManager manager(config);

    VerifiedLoop strong_fitness;
    strong_fitness.query_id = 1;
    strong_fitness.match_id = 10;
    strong_fitness.verified = true;
    strong_fitness.inlier_ratio = 1.0;
    strong_fitness.fitness_score = 0.05;
    strong_fitness.candidate_residual = Eigen::Isometry3d::Identity();
    strong_fitness.candidate_residual.translation().z() = 4.0;

    VerifiedLoop low_z = strong_fitness;
    low_z.match_id = 11;
    low_z.fitness_score = 0.20;
    low_z.candidate_residual.translation().z() = 0.1;

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
    loop.candidate_residual = Eigen::Isometry3d::Identity();
    loop.candidate_residual.translation().z() = 0.1;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::Full6Dof);
    EXPECT_FALSE(modeled.vertical_downweighted);
    EXPECT_TRUE(modeled.information.isApprox(loop.information, 1e-12));
}

TEST(LoopClosureManagerTest, ApplyEdgeModelDownweightsVerticalAxesWhenResidualIsWeak)
{
    Config config;
    config.loop_noise_position = 0.5;
    config.loop_noise_rotation = 0.5;
    config.loop_planar_vertical_weight = 0.25;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.information.diagonal() << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    loop.candidate_residual = Eigen::Isometry3d::Identity();
    loop.candidate_residual.translation().z() = 3.0;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_TRUE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::PlanarXYYaw);
    EXPECT_TRUE(modeled.vertical_downweighted);
    EXPECT_DOUBLE_EQ(modeled.information(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(modeled.information(1, 1), 20.0);
    EXPECT_DOUBLE_EQ(modeled.information(2, 2), 7.5);
    EXPECT_DOUBLE_EQ(modeled.information(3, 3), 10.0);
    EXPECT_DOUBLE_EQ(modeled.information(4, 4), 12.5);
    EXPECT_DOUBLE_EQ(modeled.information(5, 5), 60.0);
    EXPECT_NEAR(modeled.vertical_observability_score, 0.375, 1e-12);
}

TEST(LoopClosureManagerTest, ApplyEdgeModelRejectsVerticalOutlier)
{
    Config config;
    config.loop_max_candidate_residual_z = 5.0;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.candidate_residual = Eigen::Isometry3d::Identity();
    loop.candidate_residual.translation().z() = 4.5;

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_FALSE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::RejectedVerticalInconsistent);
    EXPECT_TRUE(manager.buildLoopEdges({modeled}, LoopEdgeDirection::MatchToQuery).empty());
}

TEST(LoopClosureManagerTest, ApplyEdgeModelRejectsYawInconsistentOutlier)
{
    Config config;
    LoopClosureManager manager(config);

    VerifiedLoop loop;
    loop.verified = true;
    loop.query_id = 1;
    loop.match_id = 2;
    loop.candidate_yaw_diff_rad = M_PI;
    loop.candidate_residual = Eigen::Isometry3d::Identity();

    const auto modeled = manager.applyEdgeModel(loop);
    EXPECT_FALSE(modeled.verified);
    EXPECT_EQ(modeled.edge_mode, LoopEdgeMode::RejectedYawInconsistent);
    EXPECT_TRUE(manager.buildLoopEdges({modeled}, LoopEdgeDirection::MatchToQuery).empty());
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
