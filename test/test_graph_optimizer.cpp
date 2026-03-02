#include <gtest/gtest.h>
#include <ros/ros.h>
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
    
    optimizer_->loadGraph(nodes, edges);
    
    EXPECT_EQ(optimizer_->getNumNodes(), 4u);
    EXPECT_EQ(optimizer_->getNumEdges(), 3u);
    
    auto poses = optimizer_->getOptimizedPoses();
    EXPECT_EQ(poses.size(), 4u);
    
    auto loaded_edges = optimizer_->getEdges();
    EXPECT_EQ(loaded_edges.size(), 3u);
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

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_graph_optimizer");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}