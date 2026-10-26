#include "../src/graph_solver_execution.h"
#include "n3mapping/graph_optimizer.h"

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

namespace n3mapping {
namespace {
using namespace std::chrono_literals;

TEST(GraphSolverExecution, NestedWorkAndExceptionsStayWithinLocalArena) {
    tbb::global_control budget(tbb::global_control::max_allowed_parallelism, 4);
    const auto owner = std::this_thread::get_id();
    std::atomic<int> calls{0}, foreign_threads{0};
    EXPECT_EQ(internal::runGraphSolve([&] {
        EXPECT_EQ(tbb::this_task_arena::max_concurrency(), 1);
        EXPECT_EQ(tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism), 4U);
        tbb::parallel_for(0, 32, [&](int) {
            ++calls;
            if (std::this_thread::get_id() != owner) ++foreign_threads;
        });
        return internal::runGraphSolve([] { return 17; });
    }), 17);
    EXPECT_EQ(calls, 32);
    EXPECT_EQ(foreign_threads, 0);
    EXPECT_THROW(internal::runGraphSolve([]() -> int {
        throw std::runtime_error("solver failed");
    }), std::runtime_error);
    EXPECT_EQ(tbb::global_control::active_value(
        tbb::global_control::max_allowed_parallelism), 4U);
}

TEST(GraphSolverExecution, IndependentOwnersAndUnrelatedTbbWorkRemainConcurrent) {
    tbb::global_control budget(tbb::global_control::max_allowed_parallelism, 4);
    std::atomic<int> entered{0};
    std::promise<void> release;
    const auto released = release.get_future().share();
    auto launch = [&] {
        return std::async(std::launch::async, [&] {
            return internal::runGraphSolve([&] {
                ++entered;
                const auto status = released.wait_for(5s);
                return status == std::future_status::ready &&
                    tbb::this_task_arena::max_concurrency() == 1;
            });
        });
    };
    auto first = launch();
    auto second = launch();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (entered < 2 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    EXPECT_EQ(entered, 2);
    EXPECT_EQ(tbb::global_control::active_value(
        tbb::global_control::max_allowed_parallelism), 4U);
    std::atomic<int> active{0}, peak{0};
    tbb::parallel_for(0, 16, [&](int) {
        const int current = ++active;
        int previous = peak.load();
        while (previous < current && !peak.compare_exchange_weak(previous, current)) {}
        std::this_thread::sleep_for(2ms);
        --active;
    });
    release.set_value();
    EXPECT_TRUE(first.get());
    EXPECT_TRUE(second.get());
    EXPECT_GE(peak, 2);
}

TEST(GraphSolverExecution, ConcurrentGraphSnapshotsMatchSerialNumerics) {
    tbb::global_control budget(tbb::global_control::max_allowed_parallelism, 4);
    Config config;
    GraphOptimizer source(config);
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    std::vector<EdgeInfo> edges;
    for (int id = 0; id < 120; ++id) {
        auto pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(.2 * (id % 12), .2 * (id / 12), 0.);
        nodes.emplace_back(id, pose);
        for (const int from : {id - 1, id - 12}) {
            if (from < 0) continue;
            EdgeInfo edge;
            edge.from_id = from;
            edge.to_id = id;
            edge.type = from == id - 1 ? EdgeType::ODOMETRY : EdgeType::LOOP;
            edge.measurement = nodes[from].second.inverse() * pose;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.;
            edges.push_back(edge);
        }
    }
    ASSERT_TRUE(source.loadGraph(nodes, edges));
    const auto unchanged = source.getOptimizedPoses();
    auto exercise = [](std::unique_ptr<GraphOptimizer> graph) {
        for (int repeat = 0; repeat < 4; ++repeat) {
            graph->optimize();
            EdgeInfo edge;
            edge.from_id = 119 + repeat;
            edge.to_id = 120 + repeat;
            edge.type = EdgeType::ODOMETRY;
            edge.measurement.translation().x() = .2;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.;
            graph->addOdometryEdge(edge);
            if (!graph->incrementalOptimize()) throw std::runtime_error("append failed");
        }
        return graph->getOptimizedPoses();
    };
    const auto expected = exercise(source.cloneCommitted());
    auto a = std::async(std::launch::async, exercise, source.cloneCommitted());
    auto b = std::async(std::launch::async, exercise, source.cloneCommitted());
    for (const auto& actual : {a.get(), b.get()}) {
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto& [id, pose] : expected)
            EXPECT_TRUE(actual.at(id).matrix().isApprox(pose.matrix(), 1e-10));
    }
    for (const auto& [id, pose] : unchanged)
        EXPECT_TRUE(source.getOptimizedPose(id).matrix().isApprox(pose.matrix(), 0.));
}

}  // namespace
}  // namespace n3mapping
