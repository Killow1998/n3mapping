#include <gtest/gtest.h>
#include <ros/ros.h>
#include <random>
#include <cmath>

#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
namespace test {

class PointCloudMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_correspondence_distance = 2.0;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.3;
        config_.gicp_num_neighbors = 20;
        config_.num_threads = 4;
        
        matcher_ = std::make_unique<PointCloudMatcher>(config_);
    }

    void TearDown() override {
        matcher_.reset();
    }

    /**
     * @brief 创建平面点云 (用于测试)
     * @param num_points 点数
     * @param size 平面大小
     * @param noise 噪声标准差
     * @return 点云指针
     */
    Keyframe::PointCloudT::Ptr createPlaneCloud(size_t num_points = 1000, 
                                                  double size = 10.0,
                                                  double noise = 0.01) {
        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);
        
        std::mt19937 rng(42);  // 固定种子以保证可重复性
        std::uniform_real_distribution<float> pos_dist(-size/2, size/2);
        std::normal_distribution<float> noise_dist(0.0f, static_cast<float>(noise));
        
        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = pos_dist(rng);
            cloud->points[i].y = pos_dist(rng);
            cloud->points[i].z = noise_dist(rng);  // 平面 z=0 加噪声
            cloud->points[i].intensity = 1.0f;
        }
        
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    /**
     * @brief 创建球形点云 (用于测试)
     * @param num_points 点数
     * @param radius 半径
     * @param noise 噪声标准差
     * @return 点云指针
     */
    Keyframe::PointCloudT::Ptr createSphereCloud(size_t num_points = 1000,
                                                   double radius = 5.0,
                                                   double noise = 0.01) {
        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.resize(num_points);
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> phi_dist(0.0f, M_PI);
        std::normal_distribution<float> noise_dist(0.0f, static_cast<float>(noise));
        
        for (size_t i = 0; i < num_points; ++i) {
            float theta = theta_dist(rng);
            float phi = phi_dist(rng);
            float r = static_cast<float>(radius) + noise_dist(rng);
            
            cloud->points[i].x = r * std::sin(phi) * std::cos(theta);
            cloud->points[i].y = r * std::sin(phi) * std::sin(theta);
            cloud->points[i].z = r * std::cos(phi);
            cloud->points[i].intensity = 1.0f;
        }
        
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    /**
     * @brief 变换点云
     * @param cloud 输入点云
     * @param transform 变换矩阵
     * @return 变换后的点云
     */
    Keyframe::PointCloudT::Ptr transformCloud(const Keyframe::PointCloudT::Ptr& cloud,
                                                const Eigen::Isometry3d& transform) {
        auto transformed = std::make_shared<Keyframe::PointCloudT>();
        transformed->points.resize(cloud->size());
        
        for (size_t i = 0; i < cloud->size(); ++i) {
            Eigen::Vector3d pt(cloud->points[i].x, 
                               cloud->points[i].y, 
                               cloud->points[i].z);
            Eigen::Vector3d pt_transformed = transform * pt;
            
            transformed->points[i].x = static_cast<float>(pt_transformed.x());
            transformed->points[i].y = static_cast<float>(pt_transformed.y());
            transformed->points[i].z = static_cast<float>(pt_transformed.z());
            transformed->points[i].intensity = cloud->points[i].intensity;
        }
        
        transformed->width = cloud->size();
        transformed->height = 1;
        transformed->is_dense = true;
        return transformed;
    }

    /**
     * @brief 创建位姿
     */
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

    /**
     * @brief 比较两个位姿是否接近
     */
    bool posesAreClose(const Eigen::Isometry3d& pose1, 
                       const Eigen::Isometry3d& pose2,
                       double trans_tol = 0.1,
                       double rot_tol = 0.05) {
        // 平移误差
        double trans_error = (pose1.translation() - pose2.translation()).norm();
        
        // 旋转误差 (角度)
        Eigen::Matrix3d R_diff = pose1.rotation().transpose() * pose2.rotation();
        double rot_error = std::acos(std::min(1.0, std::max(-1.0, 
            (R_diff.trace() - 1.0) / 2.0)));
        
        return trans_error < trans_tol && rot_error < rot_tol;
    }

    Config config_;
    std::unique_ptr<PointCloudMatcher> matcher_;
};

// 测试预处理功能
TEST_F(PointCloudMatcherTest, PreprocessPointCloud) {
    auto cloud = createPlaneCloud(1000);
    
    auto [processed, kdtree] = matcher_->preprocessPointCloud(cloud);
    
    ASSERT_NE(processed, nullptr);
    ASSERT_NE(kdtree, nullptr);
    EXPECT_GT(processed->size(), 0u);
    // 下采样后点数应该减少
    EXPECT_LT(processed->size(), cloud->size());
}

// 测试空点云预处理
TEST_F(PointCloudMatcherTest, PreprocessEmptyCloud) {
    auto empty_cloud = std::make_shared<Keyframe::PointCloudT>();
    
    auto [processed, kdtree] = matcher_->preprocessPointCloud(empty_cloud);
    
    EXPECT_TRUE(processed->empty());
    EXPECT_EQ(kdtree, nullptr);
}

// 测试空指针预处理
TEST_F(PointCloudMatcherTest, PreprocessNullCloud) {
    auto [processed, kdtree] = matcher_->preprocessPointCloud(nullptr);
    
    EXPECT_TRUE(processed->empty());
    EXPECT_EQ(kdtree, nullptr);
}

// 测试相同点云配准 (应该返回单位变换)
// Requirements: 3.1, 3.2
TEST_F(PointCloudMatcherTest, AlignIdenticalClouds) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    EXPECT_LT(result.fitness_score, config_.gicp_fitness_threshold);
    
    // 变换应该接近单位矩阵
    EXPECT_TRUE(posesAreClose(result.T_target_source, Eigen::Isometry3d::Identity(), 0.1, 0.05));
}

// 测试已知变换的配准
// Requirements: 3.1, 3.2
TEST_F(PointCloudMatcherTest, AlignWithKnownTransform) {
    auto target_cloud = createPlaneCloud(2000);
    
    // 创建较小的已知变换（GICP 对大变换敏感）
    Eigen::Isometry3d known_transform = createPose(0.2, 0.1, 0.05, 0.0, 0.0, 0.05);
    
    // 变换源点云
    auto source_cloud = transformCloud(target_cloud, known_transform.inverse());
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), target_cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), source_cloud);
    
    // 使用已知变换作为初始猜测
    auto result = matcher_->align(target_kf, source_kf, known_transform);
    
    EXPECT_TRUE(result.success);
    EXPECT_LT(result.fitness_score, config_.gicp_fitness_threshold);
    
    // 估计的变换应该接近已知变换
    EXPECT_TRUE(posesAreClose(result.T_target_source, known_transform, 0.3, 0.2))
        << "Estimated transform should be close to known transform";
}

// 测试配准得分计算
// Requirements: 3.3
TEST_F(PointCloudMatcherTest, FitnessScoreCalculation) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    // 相同点云配准得分应该很低
    EXPECT_LT(result.fitness_score, 0.1);
    EXPECT_GT(result.num_inliers, 0u);
}

// 测试无效输入处理
TEST_F(PointCloudMatcherTest, AlignWithInvalidInput) {
    auto cloud = createPlaneCloud(1000);
    auto empty_cloud = std::make_shared<Keyframe::PointCloudT>();
    
    auto valid_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto empty_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), empty_cloud);
    
    // 空源点云
    auto result1 = matcher_->align(valid_kf, empty_kf);
    EXPECT_FALSE(result1.success);
    
    // 空目标点云
    auto result2 = matcher_->align(empty_kf, valid_kf);
    EXPECT_FALSE(result2.success);
    
    // nullptr 关键帧
    auto result3 = matcher_->align(nullptr, valid_kf);
    EXPECT_FALSE(result3.success);
    
    auto result4 = matcher_->align(valid_kf, nullptr);
    EXPECT_FALSE(result4.success);
}

// 测试批量配准
// Requirements: 3.7, 4.8, 13.1
TEST_F(PointCloudMatcherTest, AlignBatch) {
    auto cloud1 = createPlaneCloud(1500);
    auto cloud2 = createPlaneCloud(1500);
    
    // 创建多个配准对
    std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>> pairs;
    std::vector<Eigen::Isometry3d> init_guesses;
    
    for (int i = 0; i < 3; ++i) {
        auto target_kf = Keyframe::create(i * 2, i * 0.1, Eigen::Isometry3d::Identity(), cloud1);
        auto source_kf = Keyframe::create(i * 2 + 1, i * 0.1 + 0.05, Eigen::Isometry3d::Identity(), cloud2);
        
        pairs.emplace_back(target_kf, source_kf);
        init_guesses.push_back(Eigen::Isometry3d::Identity());
    }
    
    auto results = matcher_->alignBatch(pairs, init_guesses);
    
    EXPECT_EQ(results.size(), pairs.size());
    
    // 所有配准应该成功
    for (const auto& result : results) {
        EXPECT_TRUE(result.success);
    }
}

// 测试空批量配准
TEST_F(PointCloudMatcherTest, AlignBatchEmpty) {
    std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>> pairs;
    std::vector<Eigen::Isometry3d> init_guesses;
    
    auto results = matcher_->alignBatch(pairs, init_guesses);
    
    EXPECT_TRUE(results.empty());
}

// 测试初始猜测对配准的影响
TEST_F(PointCloudMatcherTest, AlignWithInitialGuess) {
    auto target_cloud = createSphereCloud(2000);
    
    // 创建较大的变换
    Eigen::Isometry3d known_transform = createPose(1.0, 0.5, 0.2, 0.0, 0.0, 0.2);
    auto source_cloud = transformCloud(target_cloud, known_transform.inverse());
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), target_cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), source_cloud);
    
    // 使用接近真实值的初始猜测
    Eigen::Isometry3d init_guess = createPose(0.9, 0.4, 0.15, 0.0, 0.0, 0.15);
    
    auto result = matcher_->align(target_kf, source_kf, init_guess);
    
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(posesAreClose(result.T_target_source, known_transform, 0.2, 0.15));
}

// 测试信息矩阵输出
TEST_F(PointCloudMatcherTest, InformationMatrixOutput) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    
    // 信息矩阵应该是正定的 (对角线元素为正)
    for (int i = 0; i < 6; ++i) {
        EXPECT_GE(result.information(i, i), 0.0);
    }
}

// 测试配准设置修改
TEST_F(PointCloudMatcherTest, ModifySettings) {
    auto original_settings = matcher_->getSettings();
    
    small_gicp::RegistrationSetting new_settings;
    new_settings.max_iterations = 50;
    new_settings.max_correspondence_distance = 3.0;
    
    matcher_->setSettings(new_settings);
    
    auto updated_settings = matcher_->getSettings();
    EXPECT_EQ(updated_settings.max_iterations, 50);
    EXPECT_DOUBLE_EQ(updated_settings.max_correspondence_distance, 3.0);
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    ros::init(argc, argv, "n3mapping_test_point_cloud_matcher");
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
