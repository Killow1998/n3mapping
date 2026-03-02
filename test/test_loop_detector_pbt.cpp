/**
 * @file test_loop_detector_pbt.cpp
 * @brief Property-Based Tests for LoopDetector
 * 
 * Feature: n3mapping-backend
 * Property 4: 回环检测排除近邻帧
 * Validates: Requirements 4.6
 * 
 * Property 7: ScanContext 描述子维度不变性
 * Validates: Requirements 4.1
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <random>
#include <chrono>
#include "n3mapping/loop_detector.h"

namespace n3mapping {
namespace test {

/**
 * @brief 属性测试基类
 */
class LoopDetectorPBTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用随机种子
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        rng_.seed(static_cast<unsigned int>(seed));
        
        // 记录种子以便复现
        std::cout << "Random seed: " << seed << std::endl;
    }

    void TearDown() override {
    }

    // 生成随机点云
    Keyframe::PointCloudT::Ptr generateRandomCloud(size_t min_points = 500,
                                                    size_t max_points = 2000) {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        std::uniform_real_distribution<double> pos_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> height_dist(-5.0, 5.0);
        
        size_t num_points = num_dist(rng_);
        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.reserve(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            pcl::PointXYZI pt;
            pt.x = static_cast<float>(pos_dist(rng_));
            pt.y = static_cast<float>(pos_dist(rng_));
            pt.z = static_cast<float>(height_dist(rng_));
            pt.intensity = 1.0f;
            cloud->points.push_back(pt);
        }
        
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    // 生成圆形点云 (模拟 LiDAR 扫描)
    Keyframe::PointCloudT::Ptr generateCircularCloud(double radius = 20.0) {
        std::uniform_int_distribution<size_t> num_dist(800, 1500);
        std::uniform_real_distribution<double> height_dist(-3.0, 3.0);
        std::uniform_real_distribution<double> radius_noise(-2.0, 2.0);
        
        size_t num_points = num_dist(rng_);
        auto cloud = std::make_shared<Keyframe::PointCloudT>();
        cloud->points.reserve(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            double angle = 2.0 * M_PI * static_cast<double>(i) / num_points;
            double r = radius + radius_noise(rng_);
            
            pcl::PointXYZI pt;
            pt.x = static_cast<float>(r * std::cos(angle));
            pt.y = static_cast<float>(r * std::sin(angle));
            pt.z = static_cast<float>(height_dist(rng_));
            pt.intensity = 1.0f;
            cloud->points.push_back(pt);
        }
        
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    // 生成随机配置
    Config generateRandomConfig() {
        Config config;
        
        std::uniform_int_distribution<int> exclude_dist(5, 100);
        std::uniform_int_distribution<int> candidates_dist(3, 20);
        std::uniform_real_distribution<double> threshold_dist(0.1, 0.5);
        
        config.sc_num_exclude_recent = exclude_dist(rng_);
        config.sc_num_candidates = candidates_dist(rng_);
        config.sc_dist_threshold = threshold_dist(rng_);
        config.num_threads = 2;
        
        return config;
    }

    std::mt19937 rng_;
};

/**
 * @brief Property 4: 回环检测排除近邻帧
 * 
 * For any 回环检测查询，返回的候选帧索引应全部小于 (当前帧索引 - NUM_EXCLUDE_RECENT)
 * 
 * Feature: n3mapping-backend, Property 4: 回环检测排除近邻帧
 * Validates: Requirements 4.6
 */
TEST_F(LoopDetectorPBTTest, Property4_ExcludeRecentFrames) {
    constexpr int NUM_ITERATIONS = 100;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 生成随机配置
        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);
        
        // 生成随机数量的关键帧
        std::uniform_int_distribution<int> num_frames_dist(
            config.sc_num_exclude_recent + 5,  // 至少比排除数量多 5 帧
            config.sc_num_exclude_recent + 50
        );
        int num_frames = num_frames_dist(rng_);
        
        // 添加关键帧
        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }
        
        // 随机选择一个查询帧 (必须有足够的历史帧)
        std::uniform_int_distribution<int> query_dist(
            config.sc_num_exclude_recent,
            num_frames - 1
        );
        int query_id = query_dist(rng_);
        
        // 执行回环检测
        auto candidates = detector->detectLoopCandidates(query_id);
        
        // 验证属性：所有候选帧索引都小于 (query_id - num_exclude_recent)
        int max_allowed_id = query_id - config.sc_num_exclude_recent;
        
        for (const auto& candidate : candidates) {
            EXPECT_LT(candidate.match_id, max_allowed_id)
                << "Iteration: " << iter
                << ", Query ID: " << query_id
                << ", Match ID: " << candidate.match_id
                << ", Max Allowed: " << max_allowed_id
                << ", Num Exclude: " << config.sc_num_exclude_recent;
            
            EXPECT_GE(candidate.match_id, 0)
                << "Match ID should be non-negative";
            
            EXPECT_EQ(candidate.query_id, query_id)
                << "Query ID in candidate should match";
        }
    }
}

/**
 * @brief Property 7: ScanContext 描述子维度不变性
 * 
 * For any 输入点云，生成的 ScanContext 描述子维度应始终为 (PC_NUM_RING × PC_NUM_SECTOR)
 * 
 * Feature: n3mapping-backend, Property 7: ScanContext 描述子维度不变性
 * Validates: Requirements 4.1
 */
TEST_F(LoopDetectorPBTTest, Property7_DescriptorDimensionInvariance) {
    constexpr int NUM_ITERATIONS = 100;
    
    Config config;
    auto detector = std::make_unique<LoopDetector>(config);
    
    // 获取期望的维度
    auto [expected_rows, expected_cols] = detector->getDescriptorDimensions();
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 生成随机点云 (不同大小、不同分布)
        Keyframe::PointCloudT::Ptr cloud;
        
        std::uniform_int_distribution<int> cloud_type_dist(0, 2);
        int cloud_type = cloud_type_dist(rng_);
        
        switch (cloud_type) {
            case 0:
                // 随机分布点云
                cloud = generateRandomCloud(100, 5000);
                break;
            case 1:
                // 圆形点云
                cloud = generateCircularCloud();
                break;
            case 2: {
                // 稀疏点云
                cloud = generateRandomCloud(50, 200);
                break;
            }
        }
        
        // 生成描述子
        auto descriptor = detector->makeScanContext(cloud);
        
        // 验证属性：描述子维度应该恒定
        EXPECT_EQ(descriptor.rows(), expected_rows)
            << "Iteration: " << iter
            << ", Cloud type: " << cloud_type
            << ", Cloud size: " << cloud->size();
        
        EXPECT_EQ(descriptor.cols(), expected_cols)
            << "Iteration: " << iter
            << ", Cloud type: " << cloud_type
            << ", Cloud size: " << cloud->size();
    }
}

/**
 * @brief 补充测试：历史帧不足时返回空列表
 * 
 * 当历史帧数量不足以排除近邻帧时，应返回空的候选列表
 */
TEST_F(LoopDetectorPBTTest, InsufficientHistoryReturnsEmpty) {
    constexpr int NUM_ITERATIONS = 50;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);
        
        // 添加少于排除数量的帧
        std::uniform_int_distribution<int> num_frames_dist(1, config.sc_num_exclude_recent - 1);
        int num_frames = num_frames_dist(rng_);
        
        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }
        
        // 查询最后一帧
        auto candidates = detector->detectLoopCandidates(num_frames - 1);
        
        // 验证：应该返回空列表
        EXPECT_TRUE(candidates.empty())
            << "Iteration: " << iter
            << ", Num frames: " << num_frames
            << ", Num exclude: " << config.sc_num_exclude_recent;
    }
}

/**
 * @brief 补充测试：候选帧数量限制
 * 
 * 返回的候选帧数量不应超过配置的最大值
 */
TEST_F(LoopDetectorPBTTest, CandidateCountLimit) {
    constexpr int NUM_ITERATIONS = 50;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);
        
        // 添加足够多的帧
        int num_frames = config.sc_num_exclude_recent + config.sc_num_candidates + 20;
        
        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }
        
        // 查询最后一帧
        auto candidates = detector->detectLoopCandidates(num_frames - 1);
        
        // 验证：候选数量不超过配置值
        EXPECT_LE(static_cast<int>(candidates.size()), config.sc_num_candidates)
            << "Iteration: " << iter
            << ", Candidates: " << candidates.size()
            << ", Max: " << config.sc_num_candidates;
    }
}

}  // namespace test
}  // namespace n3mapping

int main(int argc, char** argv) {
    ros::init(argc, argv, "n3mapping_test_loop_detector_pbt");
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ros::shutdown();
    return result;
}
