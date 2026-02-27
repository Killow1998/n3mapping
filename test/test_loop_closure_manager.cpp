#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "n3mapping/loop_closure_manager.h"

namespace n3mapping {

class MockLoopOptimizer : public LoopOptimizerInterface
{
  public:
    MOCK_METHOD(void, addLoopEdge, (const EdgeInfo& edge), (override));
    MOCK_METHOD(void, incrementalOptimize, (), (override));
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
    EXPECT_TRUE(edges_qm.front().measurement.isApprox(loop.T_match_query, 1e-9));

    auto edges_mq = manager.buildLoopEdges({ loop }, LoopEdgeDirection::MatchToQuery);
    ASSERT_EQ(edges_mq.size(), 1u);
    EXPECT_EQ(edges_mq.front().from_id, 2);
    EXPECT_EQ(edges_mq.front().to_id, 1);
    EXPECT_TRUE(edges_mq.front().measurement.isApprox(loop.T_match_query.inverse(), 1e-9));
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
    EXPECT_CALL(mock, incrementalOptimize()).Times(1);

    bool optimized = manager.applyEdges({ e1, e2 }, mock);
    EXPECT_TRUE(optimized);
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
