#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"
#include <gtest/gtest.h>
#include <random>
#include <rclcpp/rclcpp.hpp>

namespace n3mapping {
namespace test {

/**
 * @brief WorldLocalizing 属性测试
 *
 * Feature: n3mapping-backend
 * Property 10: 重定位位姿变换一致性
 *
 * *For any* 成功的重定位，T_map_odom 变换应满足：
 * 对于后续的里程计输入 pose_odom，pose_map = T_map_odom * pose_odom
 * 应与地图中的位置一致。
 *
 * **Validates: Requirements 9.5, 9.8**
 */
class WorldLocalizingPBTTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // 初始化配置
        config_.keyframe_distance_threshold = 1.0;
        config_.keyframe_angle_threshold = 0.5;
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.5;
        config_.sc_dist_threshold = 0.3;
        config_.sc_num_exclude_recent = 5;
        config_.sc_num_candidates = 3;
        config_.reloc_num_candidates = 5;
        config_.reloc_sc_dist_threshold = 0.5;
        config_.reloc_search_radius = 20.0;
        config_.reloc_max_track_failures = 5;

        // 初始化随机数生成器
        rng_.seed(std::random_device{}());
    }

    // 生成随机位姿
    Eigen::Isometry3d generateRandomPose()
    {
        std::uniform_real_distribution<double> pos_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);

        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(pos_dist(rng_), pos_dist(rng_), 0.0);
        pose.rotate(Eigen::AngleAxisd(angle_dist(rng_), Eigen::Vector3d::UnitZ()));

        return pose;
    }

    // 生成随机变换
    Eigen::Isometry3d generateRandomTransform()
    {
        std::uniform_real_distribution<double> pos_dist(-10.0, 10.0);
        std::uniform_real_distribution<double> angle_dist(-M_PI / 4, M_PI / 4);

        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(pos_dist(rng_), pos_dist(rng_), 0.0);
        T.rotate(Eigen::AngleAxisd(angle_dist(rng_), Eigen::Vector3d::UnitZ()));

        return T;
    }

    Config config_;
    std::mt19937 rng_;
};

/**
 * @brief Property 10: 重定位位姿变换一致性
 *
 * 测试 T_map_odom 变换的数学一致性：
 * pose_map = T_map_odom * pose_odom
 *
 * **Validates: Requirements 9.5, 9.8**
 */
TEST_F(WorldLocalizingPBTTest, Property10_PoseTransformConsistency)
{
    constexpr int NUM_ITERATIONS = 100;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 创建组件
        KeyframeManager keyframe_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        WorldLocalizing reloc(config_, keyframe_manager, loop_detector, matcher);

        // 生成随机 T_map_odom
        Eigen::Isometry3d T_map_odom = generateRandomTransform();
        reloc.setMapToOdomTransform(T_map_odom);

        // 验证设置成功
        ASSERT_TRUE(reloc.isRelocalized());

        // 获取存储的变换
        Eigen::Isometry3d stored_T = reloc.getMapToOdomTransform();

        // 验证存储的变换与设置的一致
        EXPECT_TRUE(stored_T.isApprox(T_map_odom, 1e-9)) << "Iteration " << iter << ": Stored transform differs from set transform";

        // 生成随机里程计位姿
        Eigen::Isometry3d pose_odom = generateRandomPose();

        // 计算地图坐标系中的位姿
        Eigen::Isometry3d pose_map = T_map_odom * pose_odom;

        // 验证逆变换
        // pose_odom = T_map_odom.inverse() * pose_map
        Eigen::Isometry3d recovered_odom = T_map_odom.inverse() * pose_map;

        EXPECT_TRUE(recovered_odom.isApprox(pose_odom, 1e-9)) << "Iteration " << iter << ": Inverse transform failed";

        // 验证变换的结合律
        // (T_map_odom * pose_odom).inverse() = pose_odom.inverse() * T_map_odom.inverse()
        Eigen::Isometry3d lhs = (T_map_odom * pose_odom).inverse();
        Eigen::Isometry3d rhs = pose_odom.inverse() * T_map_odom.inverse();

        EXPECT_TRUE(lhs.isApprox(rhs, 1e-9)) << "Iteration " << iter << ": Associativity property failed";
    }
}

/**
 * @brief 测试连续位姿变换的一致性
 *
 * 验证多次位姿变换后，相对位姿关系保持不变
 *
 * **Validates: Requirements 9.8**
 */
TEST_F(WorldLocalizingPBTTest, ConsecutivePoseTransformConsistency)
{
    constexpr int NUM_ITERATIONS = 50;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 创建组件
        KeyframeManager keyframe_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        WorldLocalizing reloc(config_, keyframe_manager, loop_detector, matcher);

        // 设置 T_map_odom
        Eigen::Isometry3d T_map_odom = generateRandomTransform();
        reloc.setMapToOdomTransform(T_map_odom);

        // 生成两个连续的里程计位姿
        Eigen::Isometry3d pose_odom_1 = generateRandomPose();

        // 生成相对变换
        Eigen::Isometry3d delta_odom = generateRandomTransform();
        Eigen::Isometry3d pose_odom_2 = pose_odom_1 * delta_odom;

        // 计算地图坐标系中的位姿
        Eigen::Isometry3d pose_map_1 = T_map_odom * pose_odom_1;
        Eigen::Isometry3d pose_map_2 = T_map_odom * pose_odom_2;

        // 验证相对位姿在地图坐标系中保持不变
        // delta_map = pose_map_1.inverse() * pose_map_2
        // 应该等于 delta_odom
        Eigen::Isometry3d delta_map = pose_map_1.inverse() * pose_map_2;

        EXPECT_TRUE(delta_map.isApprox(delta_odom, 1e-9)) << "Iteration " << iter << ": Relative pose not preserved in map frame";
    }
}

/**
 * @brief 测试重置后状态的一致性
 *
 * 验证重置后所有状态恢复到初始值
 */
TEST_F(WorldLocalizingPBTTest, ResetStateConsistency)
{
    constexpr int NUM_ITERATIONS = 50;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 创建组件
        KeyframeManager keyframe_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        WorldLocalizing reloc(config_, keyframe_manager, loop_detector, matcher);

        // 设置随机状态
        Eigen::Isometry3d T_map_odom = generateRandomTransform();
        reloc.setMapToOdomTransform(T_map_odom);

        ASSERT_TRUE(reloc.isRelocalized());

        // 重置
        reloc.reset();

        // 验证状态恢复
        EXPECT_FALSE(reloc.isRelocalized()) << "Iteration " << iter << ": isRelocalized should be false after reset";

        EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1) << "Iteration " << iter << ": lastMatchedKeyframeId should be -1 after reset";

        EXPECT_TRUE(reloc.getMapToOdomTransform().isApprox(Eigen::Isometry3d::Identity(), 1e-9))
          << "Iteration " << iter << ": T_map_odom should be identity after reset";
    }
}

/**
 * @brief 测试变换矩阵的正交性
 *
 * 验证存储的变换矩阵保持正交性（旋转部分）
 */
TEST_F(WorldLocalizingPBTTest, TransformOrthogonality)
{
    constexpr int NUM_ITERATIONS = 100;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // 创建组件
        KeyframeManager keyframe_manager(config_);
        LoopDetector loop_detector(config_);
        PointCloudMatcher matcher(config_);
        WorldLocalizing reloc(config_, keyframe_manager, loop_detector, matcher);

        // 设置随机变换
        Eigen::Isometry3d T_map_odom = generateRandomTransform();
        reloc.setMapToOdomTransform(T_map_odom);

        // 获取存储的变换
        Eigen::Isometry3d stored_T = reloc.getMapToOdomTransform();

        // 提取旋转矩阵
        Eigen::Matrix3d R = stored_T.rotation();

        // 验证正交性: R * R^T = I
        Eigen::Matrix3d RRT = R * R.transpose();
        EXPECT_TRUE(RRT.isApprox(Eigen::Matrix3d::Identity(), 1e-9)) << "Iteration " << iter << ": Rotation matrix not orthogonal";

        // 验证行列式为 1 (proper rotation)
        double det = R.determinant();
        EXPECT_NEAR(det, 1.0, 1e-9) << "Iteration " << iter << ": Rotation determinant is " << det;
    }
}

} // namespace test
} // namespace n3mapping

int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
