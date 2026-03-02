/**
 * @file test_n3mapping_node_pbt.cpp
 * @brief N3MappingNode 属性测试
 * 
 * Feature: n3mapping-backend
 * Property 6: 坐标变换正确性
 * 
 * *For any* 点云和位姿，Body_Frame 点云变换到 World_Frame 后应与直接在 World_Frame 中的点云一致。
 * 
 * **Validates: Requirements 6.2, 6.3**
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <random>
#include <Eigen/Dense>

namespace n3mapping {
namespace test {

/**
 * @brief 坐标变换属性测试
 */
class CoordinateTransformPBTTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(std::random_device{}());
    }

    // 生成随机点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr generateRandomCloud(size_t num_points = 100) {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->resize(num_points);
        
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> intensity_dist(0.0f, 255.0f);
        
        for (auto& pt : cloud->points) {
            pt.x = dist(rng_);
            pt.y = dist(rng_);
            pt.z = dist(rng_);
            pt.intensity = intensity_dist(rng_);
        }
        
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = true;
        
        return cloud;
    }

    // 生成随机位姿
    Eigen::Isometry3d generateRandomPose() {
        std::uniform_real_distribution<double> trans_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
        
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() << trans_dist(rng_), trans_dist(rng_), trans_dist(rng_);
        
        // 随机旋转 (使用欧拉角)
        double roll = angle_dist(rng_);
        double pitch = angle_dist(rng_);
        double yaw = angle_dist(rng_);
        
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        
        pose.linear() = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
        
        return pose;
    }

    std::mt19937 rng_;
};

/**
 * @brief Property 6: 坐标变换正确性
 * 
 * 测试 Body_Frame 点云变换到 World_Frame 的正确性
 * 
 * **Validates: Requirements 6.2, 6.3**
 */
TEST_F(CoordinateTransformPBTTest, Property6_CoordinateTransformCorrectness) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr double TOLERANCE = 1e-4;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 生成随机点云 (Body Frame)
        auto cloud_body = generateRandomCloud(50);
        
        // 生成随机位姿 T_world_body
        Eigen::Isometry3d T_world_body = generateRandomPose();
        
        // 变换点云到 World Frame
        auto cloud_world = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        Eigen::Matrix4f transform = T_world_body.matrix().cast<float>();
        pcl::transformPointCloud(*cloud_body, *cloud_world, transform);
        
        // 验证每个点的变换正确性
        ASSERT_EQ(cloud_body->size(), cloud_world->size())
            << "Iteration " << iter << ": Point cloud sizes don't match";
        
        for (size_t i = 0; i < cloud_body->size(); ++i) {
            const auto& pt_body = cloud_body->points[i];
            const auto& pt_world = cloud_world->points[i];
            
            // 手动计算期望的世界坐标
            Eigen::Vector3d pt_body_vec(pt_body.x, pt_body.y, pt_body.z);
            Eigen::Vector3d pt_world_expected = T_world_body * pt_body_vec;
            
            // 验证变换结果
            EXPECT_NEAR(pt_world.x, pt_world_expected.x(), TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": X mismatch";
            EXPECT_NEAR(pt_world.y, pt_world_expected.y(), TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Y mismatch";
            EXPECT_NEAR(pt_world.z, pt_world_expected.z(), TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Z mismatch";
            
            // 验证 intensity 保持不变
            EXPECT_FLOAT_EQ(pt_body.intensity, pt_world.intensity)
                << "Iteration " << iter << ", point " << i << ": Intensity changed";
        }
    }
}

/**
 * @brief 测试逆变换正确性 (往返一致性)
 * 
 * 变换到 World Frame 再变换回 Body Frame 应该得到原始点云
 */
TEST_F(CoordinateTransformPBTTest, InverseTransformRoundTrip) {
    constexpr int NUM_ITERATIONS = 50;
    constexpr double TOLERANCE = 1e-4;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto cloud_body = generateRandomCloud(30);
        Eigen::Isometry3d T_world_body = generateRandomPose();
        
        // Body -> World
        auto cloud_world = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        Eigen::Matrix4f transform_forward = T_world_body.matrix().cast<float>();
        pcl::transformPointCloud(*cloud_body, *cloud_world, transform_forward);
        
        // World -> Body (逆变换)
        auto cloud_body_recovered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        Eigen::Matrix4f transform_inverse = T_world_body.inverse().matrix().cast<float>();
        pcl::transformPointCloud(*cloud_world, *cloud_body_recovered, transform_inverse);
        
        // 验证往返一致性
        ASSERT_EQ(cloud_body->size(), cloud_body_recovered->size());
        
        for (size_t i = 0; i < cloud_body->size(); ++i) {
            const auto& pt_original = cloud_body->points[i];
            const auto& pt_recovered = cloud_body_recovered->points[i];
            
            EXPECT_NEAR(pt_original.x, pt_recovered.x, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": X round-trip error";
            EXPECT_NEAR(pt_original.y, pt_recovered.y, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Y round-trip error";
            EXPECT_NEAR(pt_original.z, pt_recovered.z, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Z round-trip error";
        }
    }
}

/**
 * @brief 测试位姿链式变换正确性
 * 
 * T_C_A = T_C_B * T_B_A 应该正确组合
 */
TEST_F(CoordinateTransformPBTTest, ChainedTransformCorrectness) {
    constexpr int NUM_ITERATIONS = 50;
    constexpr double TOLERANCE = 1e-4;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto cloud_A = generateRandomCloud(20);
        
        // 生成两个随机变换
        Eigen::Isometry3d T_B_A = generateRandomPose();
        Eigen::Isometry3d T_C_B = generateRandomPose();
        
        // 方法1: 分步变换 A -> B -> C
        auto cloud_B = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto cloud_C_step = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::transformPointCloud(*cloud_A, *cloud_B, T_B_A.matrix().cast<float>());
        pcl::transformPointCloud(*cloud_B, *cloud_C_step, T_C_B.matrix().cast<float>());
        
        // 方法2: 组合变换 A -> C
        Eigen::Isometry3d T_C_A = T_C_B * T_B_A;
        auto cloud_C_direct = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::transformPointCloud(*cloud_A, *cloud_C_direct, T_C_A.matrix().cast<float>());
        
        // 验证两种方法结果一致
        ASSERT_EQ(cloud_C_step->size(), cloud_C_direct->size());
        
        for (size_t i = 0; i < cloud_C_step->size(); ++i) {
            const auto& pt_step = cloud_C_step->points[i];
            const auto& pt_direct = cloud_C_direct->points[i];
            
            EXPECT_NEAR(pt_step.x, pt_direct.x, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Chained X mismatch";
            EXPECT_NEAR(pt_step.y, pt_direct.y, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Chained Y mismatch";
            EXPECT_NEAR(pt_step.z, pt_direct.z, TOLERANCE)
                << "Iteration " << iter << ", point " << i << ": Chained Z mismatch";
        }
    }
}

/**
 * @brief 测试单位变换不改变点云
 */
TEST_F(CoordinateTransformPBTTest, IdentityTransformPreservesCloud) {
    constexpr int NUM_ITERATIONS = 20;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto cloud_original = generateRandomCloud(50);
        
        // 应用单位变换
        Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
        auto cloud_transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::transformPointCloud(*cloud_original, *cloud_transformed, identity.matrix().cast<float>());
        
        // 验证点云不变
        ASSERT_EQ(cloud_original->size(), cloud_transformed->size());
        
        for (size_t i = 0; i < cloud_original->size(); ++i) {
            const auto& pt_orig = cloud_original->points[i];
            const auto& pt_trans = cloud_transformed->points[i];
            
            EXPECT_FLOAT_EQ(pt_orig.x, pt_trans.x);
            EXPECT_FLOAT_EQ(pt_orig.y, pt_trans.y);
            EXPECT_FLOAT_EQ(pt_orig.z, pt_trans.z);
            EXPECT_FLOAT_EQ(pt_orig.intensity, pt_trans.intensity);
        }
    }
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "n3mapping_test_node_pbt");
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
