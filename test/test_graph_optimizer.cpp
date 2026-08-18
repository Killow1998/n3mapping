#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "n3mapping/graph_optimizer.h"

namespace n3mapping {
namespace test {

class GraphOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.optimization_iterations = 10;
        config_.prior_noise_position = 0.01;
        config_.prior_noise_rotation = 0.01;
        config_.odom_noise_position = 0.1;
        config_.odom_noise_rotation = 0.1;
        config_.loop_noise_position = 0.1;
        config_.loop_noise_rotation = 0.1;
        // Floor-factor lifecycle tests explicitly opt in to the experimental feature.
        config_.floor_attitude_enable = true;
        
        optimizer_ = std::make_unique<GraphOptimizer>(config_);
    }

    void TearDown() override {
        optimizer_.reset();
    }

    // 创建位姿
    Eigen::Isometry3d createPose(double x, double y, double z, 
                                  double roll = 0, double pitch = 0, double yaw = 0) {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(x, y, z);
        
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        
        pose.rotate(yawAngle * pitchAngle * rollAngle);
        return pose;
    }

    // 创建信息矩阵
    Eigen::Matrix<double, 6, 6> createInformationMatrix(double pos_weight = 1.0, 
                                                         double rot_weight = 1.0) {
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
        info.block<3, 3>(0, 0) *= pos_weight;  // translation
        info.block<3, 3>(3, 3) *= rot_weight;  // rotation
        return info;
    }

    // 比较两个位姿是否接近
    bool posesNear(const Eigen::Isometry3d& p1, const Eigen::Isometry3d& p2, 
                   double pos_tol = 1e-3, double rot_tol = 1e-3) {
        double pos_diff = (p1.translation() - p2.translation()).norm();
        Eigen::Quaterniond q1(p1.rotation());
        Eigen::Quaterniond q2(p2.rotation());
        double rot_diff = q1.angularDistance(q2);
        return pos_diff < pos_tol && rot_diff < rot_tol;
    }

    Config config_;
    std::unique_ptr<GraphOptimizer> optimizer_;
};

// ==================== 测试初始状态 ====================

TEST_F(GraphOptimizerTest, InitialState) {
    EXPECT_EQ(optimizer_->getNumNodes(), 0u);
    EXPECT_EQ(optimizer_->getNumEdges(), 0u);
    EXPECT_FALSE(optimizer_->hasLoopClosure());
}

// ==================== 测试添加先验因子 ====================

TEST_F(GraphOptimizerTest, AddPriorFactor) {
    auto pose = createPose(0, 0, 0);
    
    optimizer_->addPriorFactor(0, pose);
    
    EXPECT_EQ(optimizer_->getNumNodes(), 1u);
    EXPECT_TRUE(optimizer_->hasNode(0));
    
    optimizer_->incrementalOptimize();
    
    auto optimized_pose = optimizer_->getOptimizedPose(0);
    EXPECT_TRUE(posesNear(pose, optimized_pose, 1e-6, 1e-6));
}

// ==================== 测试添加里程计边 ====================

TEST_F(GraphOptimizerTest, AddOdometryEdge) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    auto relative_pose = createPose(1, 0, 0);  // 向前 1 米
    EdgeInfo edge;
    edge.from_id = 0;
    edge.to_id = 1;
    edge.measurement = relative_pose;
    edge.information = createInformationMatrix(10.0, 10.0);
    edge.type = EdgeType::ODOMETRY;
    
    optimizer_->addOdometryEdge(edge);
    
    EXPECT_EQ(optimizer_->getNumNodes(), 2u);
    EXPECT_EQ(optimizer_->getNumEdges(), 1u);
    
    optimizer_->incrementalOptimize();
    
    auto opt_pose0 = optimizer_->getOptimizedPose(0);
    auto opt_pose1 = optimizer_->getOptimizedPose(1);
    
    EXPECT_TRUE(posesNear(pose0, opt_pose0, 1e-3, 1e-3));
    
    auto expected_pose1 = createPose(1, 0, 0);
    EXPECT_TRUE(posesNear(expected_pose1, opt_pose1, 1e-3, 1e-3));
}

// ==================== [核心修复] 测试添加回环边 (走正方形) ====================

TEST_F(GraphOptimizerTest, AddLoopEdge) {
    // 1. 添加先验 (原点)
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    // 2. 模拟走一个正方形 (边长 1m)
    // 注意：为了简化，这里使用全局坐标下的相对位移模拟（假设无旋转或全向移动）
    // 0->1: x+1
    optimizer_->addOdometryEdge({0, 1, createPose(1, 0, 0), createInformationMatrix(100, 100), EdgeType::ODOMETRY});
    // 1->2: y+1
    optimizer_->addOdometryEdge({1, 2, createPose(0, 1, 0), createInformationMatrix(100, 100), EdgeType::ODOMETRY});
    // 2->3: x-1
    optimizer_->addOdometryEdge({2, 3, createPose(-1, 0, 0), createInformationMatrix(100, 100), EdgeType::ODOMETRY});
    // 3->4: y-1 (回到原点)
    optimizer_->addOdometryEdge({3, 4, createPose(0, -1, 0), createInformationMatrix(100, 100), EdgeType::ODOMETRY});

    optimizer_->incrementalOptimize();
    
    EXPECT_FALSE(optimizer_->hasLoopClosure());
    
    // 3. 添加回环边：4 -> 0
    // 此时 Node 4 和 Node 0 物理上应该重合，所以相对位姿为 Identity
    EdgeInfo loop_edge;
    loop_edge.from_id = 4;
    loop_edge.to_id = 0;
    loop_edge.measurement = createPose(0, 0, 0); // 测量为重合
    loop_edge.information = createInformationMatrix(1000.0, 1000.0);  // 强约束
    loop_edge.type = EdgeType::LOOP;
    
    optimizer_->addLoopEdge(loop_edge);
    
    EXPECT_TRUE(optimizer_->hasLoopClosure());
    
    // 4. 执行优化
    optimizer_->incrementalOptimize();
    
    // 5. 验证
    auto opt_pose0 = optimizer_->getOptimizedPose(0);
    auto opt_pose4 = optimizer_->getOptimizedPose(4);
    
    // 节点 4 应该非常接近节点 0
    double dist = (opt_pose4.translation() - opt_pose0.translation()).norm();
    EXPECT_LT(dist, 0.1); 
}

// ==================== 测试批量优化 ====================

TEST_F(GraphOptimizerTest, BatchOptimize) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    for (int i = 0; i < 5; ++i) {
        EdgeInfo edge;
        edge.from_id = i;
        edge.to_id = i + 1;
        edge.measurement = createPose(1, 0, 0);
        edge.information = createInformationMatrix(10.0, 10.0);
        edge.type = EdgeType::ODOMETRY;
        optimizer_->addOdometryEdge(edge);
    }
    
    optimizer_->optimize();
    
    EXPECT_EQ(optimizer_->getNumNodes(), 6u);
    auto poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(poses.size(), 6u);
    
    EXPECT_TRUE(posesNear(pose0, poses[0], 1e-3, 1e-3));
    auto expected_pose5 = createPose(5, 0, 0);
    EXPECT_TRUE(posesNear(expected_pose5, poses[5], 1e-3, 1e-3));
}

// ==================== 测试增量优化 ====================

TEST_F(GraphOptimizerTest, IncrementalOptimize) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    optimizer_->incrementalOptimize();
    
    for (int i = 0; i < 5; ++i) {
        EdgeInfo edge;
        edge.from_id = i;
        edge.to_id = i + 1;
        edge.measurement = createPose(1, 0, 0);
        edge.information = createInformationMatrix(10.0, 10.0);
        edge.type = EdgeType::ODOMETRY;
        optimizer_->addOdometryEdge(edge);
        optimizer_->incrementalOptimize();
    }
    
    EXPECT_EQ(optimizer_->getNumNodes(), 6u);
    auto poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(poses.size(), 6u);
}

TEST_F(GraphOptimizerTest, FailedIncrementalOptimizeRollsBackPendingEdgeAndAllowsFutureCommit) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    optimizer_->incrementalOptimize();

    ASSERT_EQ(optimizer_->getNumNodes(), 1u);
    ASSERT_EQ(optimizer_->getNumEdges(), 0u);
    const auto pose_before = optimizer_->getOptimizedPose(0);

    EdgeInfo bad_loop;
    bad_loop.from_id = 0;
    bad_loop.to_id = 999;
    bad_loop.measurement = Eigen::Isometry3d::Identity();
    bad_loop.information = createInformationMatrix(10.0, 10.0);
    bad_loop.type = EdgeType::LOOP;
    optimizer_->addLoopEdge(bad_loop);
    EXPECT_FALSE(optimizer_->incrementalOptimize());

    EXPECT_EQ(optimizer_->getNumEdges(), 0u);
    EXPECT_FALSE(optimizer_->hasNode(999));
    EXPECT_TRUE(posesNear(pose_before, optimizer_->getOptimizedPose(0), 1e-9, 1e-9));

    EdgeInfo good_odom;
    good_odom.from_id = 0;
    good_odom.to_id = 1;
    good_odom.measurement = createPose(1, 0, 0);
    good_odom.information = createInformationMatrix(10.0, 10.0);
    good_odom.type = EdgeType::ODOMETRY;
    optimizer_->addOdometryEdge(good_odom);
    EXPECT_TRUE(optimizer_->incrementalOptimize());

    EXPECT_EQ(optimizer_->getNumEdges(), 1u);
    EXPECT_TRUE(optimizer_->hasNode(1));
    EXPECT_TRUE(posesNear(createPose(1, 0, 0), optimizer_->getOptimizedPose(1), 1e-3, 1e-3));
}

TEST_F(GraphOptimizerTest, XYYawLoopEdgeDoesNotConstrainBadVerticalMeasurement) {
    config_.use_robust_kernel = false;
    optimizer_ = std::make_unique<GraphOptimizer>(config_);

    optimizer_->addPriorFactor(0, createPose(0, 0, 0));

    EdgeInfo odom;
    odom.from_id = 0;
    odom.to_id = 1;
    odom.measurement = createPose(5.0, 0.0, 2.0, 0.0, 0.0, 0.5);
    odom.information = createInformationMatrix(100.0, 100.0);
    odom.type = EdgeType::ODOMETRY;
    optimizer_->addOdometryEdge(odom);
    ASSERT_TRUE(optimizer_->incrementalOptimize());

    EdgeInfo loop;
    loop.from_id = 0;
    loop.to_id = 1;
    loop.measurement = createPose(1.0, 0.0, 100.0, 1.0, 1.0, 0.0);
    loop.information = createInformationMatrix(1000.0, 1000.0);
    loop.type = EdgeType::LOOP;
    loop.constraint_mode = EdgeConstraintMode::XY_YAW;
    optimizer_->addLoopEdge(loop);
    ASSERT_TRUE(optimizer_->incrementalOptimize());

    const auto pose1 = optimizer_->getOptimizedPose(1);
    EXPECT_NEAR(pose1.translation().z(), 2.0, 0.2);
    EXPECT_LT(pose1.translation().x(), 4.0);
}

// ==================== 测试获取优化后位姿 ====================

TEST_F(GraphOptimizerTest, GetOptimizedPoses) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    for (int i = 0; i < 3; ++i) {
        EdgeInfo edge;
        edge.from_id = i;
        edge.to_id = i + 1;
        edge.measurement = createPose(1, 0, 0);
        edge.information = createInformationMatrix(10.0, 10.0);
        edge.type = EdgeType::ODOMETRY;
        optimizer_->addOdometryEdge(edge);
    }
    
    optimizer_->incrementalOptimize();
    
    auto all_poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(all_poses.size(), 4u);
    
    for (int i = 0; i < 4; ++i) {
        auto pose = optimizer_->getOptimizedPose(i);
        EXPECT_TRUE(posesNear(pose, all_poses[i], 1e-9, 1e-9));
    }
    
    EXPECT_THROW(optimizer_->getOptimizedPose(100), std::out_of_range);
}

// ==================== 测试序列化支持 ====================

TEST_F(GraphOptimizerTest, LoadGraph) {
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    std::vector<EdgeInfo> edges;
    
    for (int i = 0; i < 4; ++i) {
        nodes.push_back({i, createPose(i, 0, 0)});
    }
    
    for (int i = 0; i < 3; ++i) {
        EdgeInfo edge;
        edge.from_id = i;
        edge.to_id = i + 1;
        edge.measurement = createPose(1, 0, 0);
        edge.information = createInformationMatrix(10.0, 10.0);
        edge.type = EdgeType::ODOMETRY;
        edges.push_back(edge);
    }
    
    ASSERT_TRUE(optimizer_->loadGraph(nodes, edges));
    
    EXPECT_EQ(optimizer_->getNumNodes(), 4u);
    EXPECT_EQ(optimizer_->getNumEdges(), 3u);
    
    auto poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(poses.size(), 4u);
    
    auto loaded_edges = optimizer_->getEdges();
    EXPECT_EQ(loaded_edges.size(), 3u);
}

TEST_F(GraphOptimizerTest, LoadGraphRestoresLargeCrossLinkedMapDeterministically) {
    constexpr int64_t kLastOldNode = 216;
    constexpr int64_t kFirstSessionNode = 217;
    constexpr int64_t kLastNode = 424;

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    nodes.reserve(kLastNode + 1);
    for (int64_t id = 0; id <= kLastNode; ++id) {
        nodes.emplace_back(id, createPose(0.1 * static_cast<double>(id), 0, 0));
    }

    std::vector<EdgeInfo> edges;
    edges.reserve(631);
    for (int64_t id = 0; id < kLastOldNode; ++id) {
        EdgeInfo odometry;
        odometry.from_id = id;
        odometry.to_id = id + 1;
        odometry.measurement = createPose(0.1, 0, 0);
        odometry.information = createInformationMatrix(10000.0, 10000.0);
        odometry.type = EdgeType::ODOMETRY;
        edges.push_back(odometry);
    }

    EdgeInfo anchor;
    anchor.from_id = 6;
    anchor.to_id = kFirstSessionNode;
    anchor.measurement = createPose(21.1, 0, 0);
    anchor.information = createInformationMatrix(400.0, 10000.0);
    anchor.type = EdgeType::SESSION_ANCHOR;
    edges.push_back(anchor);

    for (int64_t id = kFirstSessionNode; id < kLastNode; ++id) {
        EdgeInfo odometry;
        odometry.from_id = id;
        odometry.to_id = id + 1;
        odometry.measurement = createPose(0.1, 0, 0);
        odometry.information = createInformationMatrix(10000.0, 1000000.0);
        odometry.type = EdgeType::ODOMETRY;
        edges.push_back(odometry);
    }

    for (int64_t query = kFirstSessionNode + 1; query <= kLastNode; ++query) {
        const int64_t match = (query * 37) % (kLastOldNode + 1);
        EdgeInfo tracked;
        tracked.from_id = match;
        tracked.to_id = query;
        tracked.measurement = createPose(
            0.1 * static_cast<double>(query - match), 0, 0);
        tracked.information = createInformationMatrix(400.0, 10000.0);
        tracked.type = EdgeType::LOOP;
        edges.push_back(tracked);
    }
    ASSERT_EQ(edges.size(), 631u);

    // Repeat to protect against the former intermittent TBB Bayes-tree race,
    // while keeping this focused regression well below a second.
    for (int attempt = 0; attempt < 3; ++attempt) {
        GraphOptimizer loaded(config_);
        ASSERT_TRUE(loaded.loadGraph(nodes, edges)) << "attempt=" << attempt;
        EXPECT_EQ(loaded.getNumNodes(), nodes.size());
        EXPECT_EQ(loaded.getNumEdges(), edges.size());
    }
}

TEST_F(GraphOptimizerTest, LoadGraphFailureDoesNotPolluteExistingState) {
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes = {
        {0, createPose(0, 0, 0)},
        {1, createPose(1, 0, 0)},
    };
    std::vector<EdgeInfo> edges = {
        {0, 1, createPose(1, 0, 0), createInformationMatrix(10.0, 10.0), EdgeType::ODOMETRY},
    };
    ASSERT_TRUE(optimizer_->loadGraph(nodes, edges));
    ASSERT_EQ(optimizer_->getNumNodes(), 2u);
    ASSERT_EQ(optimizer_->getNumEdges(), 1u);
    const auto pose0 = optimizer_->getOptimizedPose(0);

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> bad_nodes = {
        {10, createPose(10, 0, 0)},
    };
    std::vector<EdgeInfo> bad_edges = {
        {10, 999, createPose(1, 0, 0), createInformationMatrix(10.0, 10.0), EdgeType::LOOP},
    };

    EXPECT_FALSE(optimizer_->loadGraph(bad_nodes, bad_edges));
    EXPECT_EQ(optimizer_->getNumNodes(), 2u);
    EXPECT_EQ(optimizer_->getNumEdges(), 1u);
    EXPECT_TRUE(optimizer_->hasNode(0));
    EXPECT_TRUE(optimizer_->hasNode(1));
    EXPECT_FALSE(optimizer_->hasNode(10));
    EXPECT_FALSE(optimizer_->hasNode(999));
    EXPECT_TRUE(posesNear(pose0, optimizer_->getOptimizedPose(0), 1e-9, 1e-9));
}

// ==================== 测试清空图 ====================

TEST_F(GraphOptimizerTest, Clear) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    EdgeInfo edge;
    edge.from_id = 0;
    edge.to_id = 1;
    edge.measurement = createPose(1, 0, 0);
    edge.information = createInformationMatrix(10.0, 10.0);
    edge.type = EdgeType::ODOMETRY;
    optimizer_->addOdometryEdge(edge);
    
    EXPECT_EQ(optimizer_->getNumNodes(), 2u);
    EXPECT_EQ(optimizer_->getNumEdges(), 1u);
    
    optimizer_->clear();
    
    EXPECT_EQ(optimizer_->getNumNodes(), 0u);
    EXPECT_EQ(optimizer_->getNumEdges(), 0u);
    EXPECT_FALSE(optimizer_->hasLoopClosure());
}

// ==================== [核心修复] 测试位姿一致性 (正方形闭环) ====================

TEST_F(GraphOptimizerTest, PoseConsistency) {
    auto pose0 = createPose(0, 0, 0);
    optimizer_->addPriorFactor(0, pose0);
    
    // 模拟正方形轨迹 0->1->2->3->4
    // 0->1: x+1
    optimizer_->addOdometryEdge({0, 1, createPose(1, 0, 0), createInformationMatrix(10.0, 10.0), EdgeType::ODOMETRY});
    // 1->2: y+1
    optimizer_->addOdometryEdge({1, 2, createPose(0, 1, 0), createInformationMatrix(10.0, 10.0), EdgeType::ODOMETRY});
    // 2->3: x-1
    optimizer_->addOdometryEdge({2, 3, createPose(-1, 0, 0), createInformationMatrix(10.0, 10.0), EdgeType::ODOMETRY});
    // 3->4: y-1 (此时积累了一些误差，假设里程计有点飘，但这里为了测试一致性，我们给完美的里程计)
    optimizer_->addOdometryEdge({3, 4, createPose(0, -1, 0), createInformationMatrix(10.0, 10.0), EdgeType::ODOMETRY});
    
    // 添加回环：4 和 0 重合
    EdgeInfo loop_edge;
    loop_edge.from_id = 4;
    loop_edge.to_id = 0;
    loop_edge.measurement = createPose(0, 0, 0); // 相对位姿为0
    loop_edge.information = createInformationMatrix(1000.0, 1000.0); 
    loop_edge.type = EdgeType::LOOP;
    optimizer_->addLoopEdge(loop_edge);
    
    optimizer_->incrementalOptimize();
    
    auto poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(poses.size(), 5u);
    
    double distance = (poses[0].translation() - poses[4].translation()).norm();
    
    // 应该非常接近 0
    EXPECT_LT(distance, 0.1);
}

TEST_F(GraphOptimizerTest, FloorConstraintCommittedOnlyAfterOptimization) {
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    optimizer_->addFloorAttitudeFactor(0, Eigen::Vector3d(0, 0, 1));
    // Transactional: pending until a successful optimization commits it.
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 0);
    EXPECT_TRUE(optimizer_->floorAttitudeConstraints().empty());
    optimizer_->incrementalOptimize();
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 1);
    ASSERT_EQ(optimizer_->floorAttitudeConstraints().size(), 1u);
    EXPECT_EQ(optimizer_->floorAttitudeConstraints()[0].node_id, 0);
}

TEST_F(GraphOptimizerTest, FloorConstraintNotCommittedOnRollback) {
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    optimizer_->incrementalOptimize();
    optimizer_->addFloorAttitudeFactor(0, Eigen::Vector3d(0, 0, 1));
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 0);
    optimizer_->rollbackToLastState();
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 0);
    EXPECT_TRUE(optimizer_->floorAttitudeConstraints().empty());
    // Re-add and commit: the rollback must not have poisoned anything.
    optimizer_->addFloorAttitudeFactor(0, Eigen::Vector3d(0, 0, 1));
    optimizer_->incrementalOptimize();
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 1);
}

TEST_F(GraphOptimizerTest, FloorConstraintClearResetsEverything) {
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    optimizer_->addFloorAttitudeFactor(0, Eigen::Vector3d(0, 0, 1));
    optimizer_->incrementalOptimize();
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 1);
    optimizer_->clear();
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 0);
    EXPECT_TRUE(optimizer_->floorAttitudeConstraints().empty());
}

TEST_F(GraphOptimizerTest, FloorConstraintSwapWithCarriesState) {
    GraphOptimizer other(config_);
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    optimizer_->addFloorAttitudeFactor(0, Eigen::Vector3d(0, 0, 1));
    optimizer_->incrementalOptimize();
    optimizer_->swapWith(other);
    EXPECT_EQ(optimizer_->floorAttitudeFactorCount(), 0);
    EXPECT_TRUE(optimizer_->floorAttitudeConstraints().empty());
    EXPECT_EQ(other.floorAttitudeFactorCount(), 1);
    ASSERT_EQ(other.floorAttitudeConstraints().size(), 1u);
    EXPECT_EQ(other.floorAttitudeConstraints()[0].node_id, 0);
}

TEST_F(GraphOptimizerTest, LoadGraphRestoresFloorConstraints) {
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    nodes.emplace_back(0, createPose(0, 0, 0));
    nodes.emplace_back(1, createPose(1, 0, 0));
    std::vector<FloorAttitudeConstraint> floors;
    floors.push_back(FloorAttitudeConstraint{0, Eigen::Vector3d(0, 0, 1), 0.02});
    floors.push_back(FloorAttitudeConstraint{1, Eigen::Vector3d(0, 0, 1), 0.03});
    GraphOptimizer loaded(config_);
    ASSERT_TRUE(loaded.loadGraph(nodes, {}, floors));
    EXPECT_EQ(loaded.floorAttitudeFactorCount(), 2);
    ASSERT_EQ(loaded.floorAttitudeConstraints().size(), 2u);
    EXPECT_EQ(loaded.floorAttitudeConstraints()[0].node_id, 0);
    EXPECT_DOUBLE_EQ(loaded.floorAttitudeConstraints()[0].sigma_rad, 0.02);
    EXPECT_DOUBLE_EQ(loaded.floorAttitudeConstraints()[1].sigma_rad, 0.03);
    // Old overload (empty floor list) stays compatible.
    GraphOptimizer legacy(config_);
    ASSERT_TRUE(legacy.loadGraph(nodes, {}));
    EXPECT_EQ(legacy.floorAttitudeFactorCount(), 0);
    EXPECT_TRUE(legacy.floorAttitudeConstraints().empty());
}

TEST_F(GraphOptimizerTest, SessionAnchorEdgeUsesRobustPathAndStaysSeparateFromLoop) {
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    optimizer_->incrementalOptimize();

    EdgeInfo anchor;
    anchor.from_id = 0;
    anchor.to_id = 1;
    anchor.measurement = Eigen::Isometry3d::Identity();
    anchor.measurement.translation().x() = 1.0;
    anchor.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    anchor.type = EdgeType::SESSION_ANCHOR;
    anchor.constraint_mode = EdgeConstraintMode::FULL_6DOF;
    ASSERT_TRUE(optimizer_->addSessionAnchorEdge(anchor));

    // The first anchor of a new session creates the target initial value.
    ASSERT_TRUE(optimizer_->hasNode(1));
    EXPECT_TRUE(posesNear(
        createPose(1, 0, 0), optimizer_->getOptimizedPose(1), 1e-9, 1e-9));

    // An anchor is a global constraint but must not be reported as a loop.
    EXPECT_TRUE(optimizer_->hasGlobalConstraint());
    EXPECT_FALSE(optimizer_->hasLoopClosure());

    ASSERT_TRUE(optimizer_->incrementalOptimize());
    const auto& edges = optimizer_->getEdges();
    bool found_anchor = false;
    for (const auto& edge : edges) {
        if (edge.type == EdgeType::SESSION_ANCHOR) {
            found_anchor = true;
        }
    }
    EXPECT_TRUE(found_anchor);
}

TEST_F(GraphOptimizerTest, SessionAnchorRejectsMissingSourceWithoutPendingState) {
    EdgeInfo anchor;
    anchor.from_id = 42;
    anchor.to_id = 43;
    anchor.measurement = createPose(1, 0, 0);
    anchor.information = createInformationMatrix(100.0, 100.0);
    anchor.type = EdgeType::SESSION_ANCHOR;

    EXPECT_FALSE(optimizer_->addSessionAnchorEdge(anchor));
    EXPECT_FALSE(optimizer_->hasNode(43));
    EXPECT_EQ(optimizer_->getNumEdges(), 0u);
    EXPECT_FALSE(optimizer_->hasGlobalConstraint());
    EXPECT_TRUE(optimizer_->incrementalOptimize());
}

TEST_F(GraphOptimizerTest, FailedUpdateRollsBackPendingSessionAnchorNodeAndFlag) {
    optimizer_->addPriorFactor(0, createPose(0, 0, 0));
    ASSERT_TRUE(optimizer_->incrementalOptimize());

    EdgeInfo anchor;
    anchor.from_id = 0;
    anchor.to_id = 1;
    anchor.measurement = createPose(1, 0, 0);
    anchor.information = createInformationMatrix(100.0, 100.0);
    anchor.type = EdgeType::SESSION_ANCHOR;
    ASSERT_TRUE(optimizer_->addSessionAnchorEdge(anchor));

    EdgeInfo invalid_loop;
    invalid_loop.from_id = 0;
    invalid_loop.to_id = 999;
    invalid_loop.measurement = Eigen::Isometry3d::Identity();
    invalid_loop.information = createInformationMatrix(10.0, 10.0);
    invalid_loop.type = EdgeType::LOOP;
    optimizer_->addLoopEdge(invalid_loop);

    EXPECT_FALSE(optimizer_->incrementalOptimize());
    EXPECT_FALSE(optimizer_->hasNode(1));
    EXPECT_EQ(optimizer_->getNumEdges(), 0u);
    EXPECT_FALSE(optimizer_->hasGlobalConstraint());

    // A failed transaction must leave the committed optimizer usable by the
    // next valid transaction.
    ASSERT_TRUE(optimizer_->addSessionAnchorEdge(anchor));
    ASSERT_TRUE(optimizer_->incrementalOptimize());
    EXPECT_TRUE(optimizer_->hasNode(1));
    EXPECT_EQ(optimizer_->getNumEdges(), 1u);
    EXPECT_TRUE(optimizer_->hasGlobalConstraint());
}

TEST_F(GraphOptimizerTest, LoadGraphRestoresSessionAnchorType) {
    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    nodes.emplace_back(0, createPose(0, 0, 0));
    nodes.emplace_back(1, createPose(1, 0, 0));

    std::vector<EdgeInfo> edges;
    EdgeInfo xy_yaw_anchor;
    xy_yaw_anchor.from_id = 0;
    xy_yaw_anchor.to_id = 1;
    xy_yaw_anchor.measurement = Eigen::Isometry3d::Identity();
    xy_yaw_anchor.measurement.translation().x() = 1.0;
    xy_yaw_anchor.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    xy_yaw_anchor.type = EdgeType::SESSION_ANCHOR;
    xy_yaw_anchor.constraint_mode = EdgeConstraintMode::XY_YAW;
    edges.push_back(xy_yaw_anchor);

    GraphOptimizer loaded(config_);
    ASSERT_TRUE(loaded.loadGraph(nodes, edges));
    EXPECT_TRUE(loaded.hasGlobalConstraint());
    EXPECT_FALSE(loaded.hasLoopClosure());
    const auto& loaded_edges = loaded.getEdges();
    ASSERT_EQ(loaded_edges.size(), 1u);
    EXPECT_EQ(loaded_edges[0].type, EdgeType::SESSION_ANCHOR);
}

TEST_F(GraphOptimizerTest, TrustedSessionLoopRebasesOutsideCauchyBasin) {
    config_.use_robust_kernel = true;
    config_.robust_kernel_type = "Cauchy";
    config_.robust_kernel_delta = 1.0;
    GraphOptimizer optimizer(config_);

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
    nodes.emplace_back(0, createPose(0, 0, 0));
    for (int64_t id = 100; id <= 105; ++id) {
        nodes.emplace_back(id, createPose(static_cast<double>(id - 100), 0, 0));
    }

    std::vector<EdgeInfo> edges;
    EdgeInfo anchor;
    anchor.from_id = 0;
    anchor.to_id = 100;
    anchor.measurement = Eigen::Isometry3d::Identity();
    anchor.information = createInformationMatrix(100.0, 100.0);
    anchor.type = EdgeType::SESSION_ANCHOR;
    edges.push_back(anchor);
    for (int64_t id = 100; id < 105; ++id) {
        EdgeInfo odometry;
        odometry.from_id = id;
        odometry.to_id = id + 1;
        odometry.measurement = createPose(1, 0, 0);
        odometry.information = createInformationMatrix(10000.0, 10000.0);
        odometry.type = EdgeType::ODOMETRY;
        edges.push_back(odometry);
    }
    ASSERT_TRUE(optimizer.loadGraph(nodes, edges));

    EdgeInfo loop;
    loop.from_id = 0;
    loop.to_id = 105;
    loop.measurement = createPose(15, 0, 0);
    loop.information = createInformationMatrix(400.0, 4.0);
    loop.type = EdgeType::LOOP;
    ASSERT_TRUE(optimizer.rebaseSessionAndAddLoopEdge(loop, 100));

    EXPECT_NEAR(optimizer.getOptimizedPose(105).translation().x(), 15.0, 0.1);
    ASSERT_EQ(optimizer.getEdges().size(), edges.size() + 1);
    EXPECT_EQ(optimizer.getEdges().back().type, EdgeType::LOOP);
    EXPECT_TRUE(optimizer.hasLoopClosure());
}

TEST_F(GraphOptimizerTest,
       SessionOdometryYieldsToTrackedPoseAndReloadKeepsSemantics) {
    config_.use_robust_kernel = true;
    config_.robust_kernel_type = "Cauchy";
    config_.robust_kernel_delta = 1.0;
    GraphOptimizer optimizer(config_);

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes = {
        {0, createPose(0, 0, 0)}, {100, createPose(0, 0, 0)}};
    EdgeInfo anchor;
    anchor.from_id = 0;
    anchor.to_id = 100;
    anchor.measurement = Eigen::Isometry3d::Identity();
    anchor.information = createInformationMatrix(400.0, 4.0);
    anchor.type = EdgeType::SESSION_ANCHOR;
    ASSERT_TRUE(optimizer.loadGraph(nodes, {anchor}));

    EdgeInfo odometry;
    odometry.from_id = 100;
    odometry.to_id = 101;
    odometry.measurement = createPose(1, 0, 0);
    odometry.information = createInformationMatrix(10000.0, 1000000.0);
    odometry.type = EdgeType::ODOMETRY;
    ASSERT_TRUE(optimizer.addSessionOdometryEdge(
        odometry, createPose(10, 0, 0)));

    EdgeInfo tracked;
    tracked.from_id = 0;
    tracked.to_id = 101;
    tracked.measurement = createPose(10, 0, 0);
    tracked.information = createInformationMatrix(400.0, 4.0);
    tracked.type = EdgeType::LOOP;
    optimizer.addLoopEdge(tracked);
    ASSERT_TRUE(optimizer.incrementalOptimize());
    EXPECT_NEAR(optimizer.getOptimizedPose(101).translation().x(), 10.0, 0.1);

    std::vector<std::pair<int64_t, Eigen::Isometry3d>> saved_nodes;
    for (const auto& [id, pose] : optimizer.getOptimizedPoses()) {
        saved_nodes.emplace_back(id, pose);
    }
    GraphOptimizer reloaded(config_);
    ASSERT_TRUE(reloaded.loadGraph(saved_nodes, optimizer.getEdges()));
    EXPECT_NEAR(reloaded.getOptimizedPose(101).translation().x(), 10.0, 0.1);
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
