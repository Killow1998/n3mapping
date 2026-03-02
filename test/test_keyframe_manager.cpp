#include <gtest/gtest.h>
#include <ros/ros.h>
#include "n3mapping/keyframe_manager.h"

namespace n3mapping {
namespace test {

class KeyframeManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.keyframe_distance_threshold = 1.0;  // 1 米
        config_.keyframe_angle_threshold = 0.5;     // ~28.6 度
        
        manager_ = std::make_unique<KeyframeManager>(config_);
    }

    void TearDown() override {
        manager_.reset();
    }

    // 创建测试点云
    Keyframe::PointCloudT::Ptr createTestCloud(size_t num_points = 100) {
        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = static_cast<float>(i) * 0.1f;
            cloud->points[i].y = static_cast<float>(i) * 0.1f;
            cloud->points[i].z = 0.0f;
            cloud->points[i].intensity = 1.0f;
        }
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
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

    Config config_;
    std::unique_ptr<KeyframeManager> manager_;
};

// 测试初始状态
TEST_F(KeyframeManagerTest, InitialState) {
    EXPECT_TRUE(manager_->empty());
    EXPECT_EQ(manager_->size(), 0u);
    EXPECT_EQ(manager_->getNextKeyframeId(), 0);
    EXPECT_EQ(manager_->getLatestKeyframe(), nullptr);
}

// 测试添加第一个关键帧
TEST_F(KeyframeManagerTest, AddFirstKeyframe) {
    auto pose = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    // 第一帧应该总是被添加
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose));
    
    int64_t id = manager_->addKeyframe(0.0, pose, cloud);
    
    EXPECT_EQ(id, 0);
    EXPECT_EQ(manager_->size(), 1u);
    EXPECT_FALSE(manager_->empty());
    EXPECT_EQ(manager_->getNextKeyframeId(), 1);
    
    auto kf = manager_->getLatestKeyframe();
    ASSERT_NE(kf, nullptr);
    EXPECT_EQ(kf->id, 0);
    EXPECT_DOUBLE_EQ(kf->timestamp, 0.0);
}

// 测试距离阈值判断 - Requirements 2.2
TEST_F(KeyframeManagerTest, DistanceThreshold) {
    auto pose1 = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, pose1, cloud);
    
    // 距离小于阈值，不应该添加
    auto pose2 = createPose(0.5, 0, 0);  // 0.5m < 1.0m
    EXPECT_FALSE(manager_->shouldAddKeyframe(pose2));
    
    // 距离等于阈值，应该添加
    auto pose3 = createPose(1.0, 0, 0);  // 1.0m == 1.0m
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose3));
    
    // 距离大于阈值，应该添加
    auto pose4 = createPose(1.5, 0, 0);  // 1.5m > 1.0m
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose4));
}

// 测试角度阈值判断 - Requirements 2.3
TEST_F(KeyframeManagerTest, AngleThreshold) {
    auto pose1 = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, pose1, cloud);
    
    // 角度小于阈值，不应该添加
    auto pose2 = createPose(0, 0, 0, 0, 0, 0.3);  // 0.3 rad < 0.5 rad
    EXPECT_FALSE(manager_->shouldAddKeyframe(pose2));
    
    // 角度略大于阈值，应该添加 (避免浮点数精度问题)
    auto pose3 = createPose(0, 0, 0, 0, 0, 0.51);  // 0.51 rad > 0.5 rad
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose3));
    
    // 角度明显大于阈值，应该添加
    auto pose4 = createPose(0, 0, 0, 0, 0, 0.7);  // 0.7 rad > 0.5 rad
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose4));
}

// 测试组合条件 - 距离和角度都不满足
TEST_F(KeyframeManagerTest, CombinedConditionBothBelow) {
    auto pose1 = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, pose1, cloud);
    
    // 距离和角度都小于阈值
    auto pose2 = createPose(0.3, 0.3, 0, 0, 0, 0.2);
    EXPECT_FALSE(manager_->shouldAddKeyframe(pose2));
}

// 测试组合条件 - 只有距离满足
TEST_F(KeyframeManagerTest, CombinedConditionDistanceOnly) {
    auto pose1 = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, pose1, cloud);
    
    // 距离满足，角度不满足
    auto pose2 = createPose(1.5, 0, 0, 0, 0, 0.1);
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose2));
}

// 测试组合条件 - 只有角度满足
TEST_F(KeyframeManagerTest, CombinedConditionAngleOnly) {
    auto pose1 = createPose(0, 0, 0);
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, pose1, cloud);
    
    // 角度满足，距离不满足
    auto pose2 = createPose(0.1, 0, 0, 0, 0, 0.6);
    EXPECT_TRUE(manager_->shouldAddKeyframe(pose2));
}

// 测试多个关键帧添加
TEST_F(KeyframeManagerTest, MultipleKeyframes) {
    auto cloud = createTestCloud();
    
    for (int i = 0; i < 5; ++i) {
        auto pose = createPose(i * 2.0, 0, 0);  // 每帧间隔 2m
        int64_t id = manager_->addKeyframe(i * 0.1, pose, cloud);
        EXPECT_EQ(id, i);
    }
    
    EXPECT_EQ(manager_->size(), 5u);
    EXPECT_EQ(manager_->getNextKeyframeId(), 5);
    
    // 验证最新关键帧
    auto latest = manager_->getLatestKeyframe();
    ASSERT_NE(latest, nullptr);
    EXPECT_EQ(latest->id, 4);
}

// 测试按 ID 获取关键帧 - Requirements 2.6
TEST_F(KeyframeManagerTest, GetKeyframeById) {
    auto cloud = createTestCloud();
    
    for (int i = 0; i < 3; ++i) {
        auto pose = createPose(i, 0, 0);
        manager_->addKeyframe(i * 0.1, pose, cloud);
    }
    
    // 获取存在的关键帧
    auto kf1 = manager_->getKeyframe(1);
    ASSERT_NE(kf1, nullptr);
    EXPECT_EQ(kf1->id, 1);
    EXPECT_NEAR(kf1->pose_odom.translation().x(), 1.0, 1e-9);
    
    // 获取不存在的关键帧
    auto kf_invalid = manager_->getKeyframe(100);
    EXPECT_EQ(kf_invalid, nullptr);
}

// 测试获取所有关键帧
TEST_F(KeyframeManagerTest, GetAllKeyframes) {
    auto cloud = createTestCloud();
    
    for (int i = 0; i < 3; ++i) {
        auto pose = createPose(i, 0, 0);
        manager_->addKeyframe(i * 0.1, pose, cloud);
    }
    
    auto all_kfs = manager_->getAllKeyframes();
    EXPECT_EQ(all_kfs.size(), 3u);
}

// 测试更新优化后位姿
TEST_F(KeyframeManagerTest, UpdateOptimizedPoses) {
    auto cloud = createTestCloud();
    
    auto pose1 = createPose(0, 0, 0);
    auto pose2 = createPose(1, 0, 0);
    
    manager_->addKeyframe(0.0, pose1, cloud);
    manager_->addKeyframe(0.1, pose2, cloud);
    
    // 更新优化后位姿
    std::map<int64_t, Eigen::Isometry3d> optimized_poses;
    optimized_poses[0] = createPose(0.1, 0.1, 0);
    optimized_poses[1] = createPose(1.1, 0.1, 0);
    
    manager_->updateOptimizedPoses(optimized_poses);
    
    auto kf0 = manager_->getKeyframe(0);
    auto kf1 = manager_->getKeyframe(1);
    
    EXPECT_NEAR(kf0->pose_optimized.translation().x(), 0.1, 1e-9);
    EXPECT_NEAR(kf1->pose_optimized.translation().x(), 1.1, 1e-9);
}

// 测试加载关键帧
TEST_F(KeyframeManagerTest, LoadKeyframes) {
    auto cloud = createTestCloud();
    
    // 创建要加载的关键帧
    std::vector<Keyframe::Ptr> keyframes;
    for (int i = 0; i < 3; ++i) {
        auto kf = Keyframe::create(i + 10, i * 0.1, createPose(i, 0, 0), cloud);
        keyframes.push_back(kf);
    }
    
    manager_->loadKeyframes(keyframes);
    
    EXPECT_EQ(manager_->size(), 3u);
    EXPECT_EQ(manager_->getNextKeyframeId(), 13);  // 最大 ID (12) + 1
    
    // 验证加载的关键帧标记
    auto kf = manager_->getKeyframe(10);
    ASSERT_NE(kf, nullptr);
    EXPECT_TRUE(kf->is_from_loaded_map);
}

// 测试清空
TEST_F(KeyframeManagerTest, Clear) {
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, createPose(0, 0, 0), cloud);
    manager_->addKeyframe(0.1, createPose(1, 0, 0), cloud);
    
    EXPECT_EQ(manager_->size(), 2u);
    
    manager_->clear();
    
    EXPECT_TRUE(manager_->empty());
    EXPECT_EQ(manager_->size(), 0u);
    EXPECT_EQ(manager_->getNextKeyframeId(), 0);
    EXPECT_EQ(manager_->getLatestKeyframe(), nullptr);
}

// 测试按时间戳查找
TEST_F(KeyframeManagerTest, FindNearestByTimestamp) {
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, createPose(0, 0, 0), cloud);
    manager_->addKeyframe(1.0, createPose(1, 0, 0), cloud);
    manager_->addKeyframe(2.0, createPose(2, 0, 0), cloud);
    
    auto nearest = manager_->findNearestByTimestamp(0.8);
    ASSERT_NE(nearest, nullptr);
    EXPECT_EQ(nearest->id, 1);  // 最接近 1.0
    
    nearest = manager_->findNearestByTimestamp(1.6);
    ASSERT_NE(nearest, nullptr);
    EXPECT_EQ(nearest->id, 2);  // 最接近 2.0
}

// 测试按位置查找
TEST_F(KeyframeManagerTest, FindNearestByPosition) {
    auto cloud = createTestCloud();
    
    manager_->addKeyframe(0.0, createPose(0, 0, 0), cloud);
    manager_->addKeyframe(0.1, createPose(5, 0, 0), cloud);
    manager_->addKeyframe(0.2, createPose(10, 0, 0), cloud);
    
    auto nearest = manager_->findNearestByPosition(Eigen::Vector3d(4, 0, 0));
    ASSERT_NE(nearest, nullptr);
    EXPECT_EQ(nearest->id, 1);  // 最接近 (5, 0, 0)
    
    nearest = manager_->findNearestByPosition(Eigen::Vector3d(8, 0, 0));
    ASSERT_NE(nearest, nullptr);
    EXPECT_EQ(nearest->id, 2);  // 最接近 (10, 0, 0)
}

// 测试关键帧有效性检查
TEST_F(KeyframeManagerTest, KeyframeValidity) {
    auto cloud = createTestCloud();
    auto empty_cloud = std::make_shared<Keyframe::PointCloudT>();
    
    // 有效关键帧
    auto kf_valid = Keyframe::create(0, 0.0, createPose(0, 0, 0), cloud);
    EXPECT_TRUE(kf_valid->isValid());
    
    // 无效关键帧 - 空点云
    auto kf_empty = Keyframe::create(1, 0.0, createPose(0, 0, 0), empty_cloud);
    EXPECT_FALSE(kf_empty->isValid());
    
    // 无效关键帧 - nullptr 点云
    auto kf_null = Keyframe::create(2, 0.0, createPose(0, 0, 0), nullptr);
    EXPECT_FALSE(kf_null->isValid());
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    ros::init(argc, argv, "n3mapping_test_keyframe_manager");
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
