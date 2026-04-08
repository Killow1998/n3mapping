#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"
#include <cmath>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <random>
#include <rclcpp/rclcpp.hpp>

namespace n3mapping {
namespace test {

class WorldLocalizingTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        config_.keyframe_distance_threshold = 1.0;
        config_.keyframe_angle_threshold = 0.5;
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_correspondence_distance = 2.0;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.5;
        config_.sc_dist_threshold = 0.3;
        config_.sc_num_exclude_recent = 5;
        config_.sc_num_candidates = 3;
        config_.reloc_num_candidates = 5;
        config_.reloc_sc_dist_threshold = 0.4;
        config_.reloc_search_radius = 15.0;
        config_.reloc_max_track_failures = 5;
        config_.num_threads = 2;

        keyframe_manager_ = std::make_unique<KeyframeManager>(config_);
        loop_detector_ = std::make_unique<LoopDetector>(config_);
        matcher_ = std::make_unique<PointCloudMatcher>(config_);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr generateCorridorCloud(const Eigen::Isometry3d& pose, double corridor_width = 3.0)
    {

        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

        for (float x = -5.0f; x <= 5.0f; x += 0.2f) {
            for (float z = 0.0f; z <= 2.5f; z += 0.3f) {
                pcl::PointXYZI pt_left;
                pt_left.x = x + noise(rng);
                pt_left.y = -corridor_width / 2.0f + noise(rng);
                pt_left.z = z + noise(rng);
                pt_left.intensity = 50.0f;
                cloud->push_back(pt_left);
                pcl::PointXYZI pt_right;
                pt_right.x = x + noise(rng);
                pt_right.y = corridor_width / 2.0f + noise(rng);
                pt_right.z = z + noise(rng);
                pt_right.intensity = 50.0f;
                cloud->push_back(pt_right);
            }
        }
        for (float x = -5.0f; x <= 5.0f; x += 0.3f) {
            for (float y = -corridor_width / 2.0f; y <= corridor_width / 2.0f; y += 0.3f) {
                pcl::PointXYZI pt;
                pt.x = x + noise(rng);
                pt.y = y + noise(rng);
                pt.z = noise(rng);
                pt.intensity = 30.0f;
                cloud->push_back(pt);
            }
        }

        auto transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        Eigen::Matrix4f transform = pose.matrix().cast<float>();
        pcl::transformPointCloud(*cloud, *transformed, transform);

        transformed->width = transformed->size();
        transformed->height = 1;
        transformed->is_dense = true;
        return transformed;
    }

    void buildTestMap(int num_keyframes = 10, double spacing = 2.0)
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        for (int i = 0; i < num_keyframes; ++i) {
            auto cloud = generateCorridorCloud(pose);
            int64_t kf_id = keyframe_manager_->addKeyframe(i * 0.1, pose, cloud);
            loop_detector_->addDescriptor(kf_id, cloud);
            pose.translation().x() += spacing;
        }
    }

    Config config_;
    std::unique_ptr<KeyframeManager> keyframe_manager_;
    std::unique_ptr<LoopDetector> loop_detector_;
    std::unique_ptr<PointCloudMatcher> matcher_;
};

TEST_F(WorldLocalizingTest, BasicConstruction)
{
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    EXPECT_FALSE(reloc.isRelocalized());
    EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1);
    EXPECT_TRUE(reloc.getMapToOdomTransform().isApprox(Eigen::Isometry3d::Identity()));
}

TEST_F(WorldLocalizingTest, RelocalizationEmptyMap)
{
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    auto cloud = generateCorridorCloud(pose);

    RelocResult result = reloc.relocalize(cloud, pose);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(reloc.isRelocalized());
}

TEST_F(WorldLocalizingTest, GlobalRelocalizationSuccess)
{
    buildTestMap(10, 2.0);
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);

    Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
    query_pose.translation().x() = 8.0;
    auto cloud = generateCorridorCloud(query_pose);

    RelocResult result = reloc.relocalize(cloud, query_pose);

    if (result.success) {
        EXPECT_TRUE(reloc.isRelocalized());
        EXPECT_GE(result.matched_keyframe_id, 0);
        EXPECT_GT(result.confidence, 0.0);
        double position_error = (result.pose_in_map.translation() - query_pose.translation()).norm();
        EXPECT_LT(position_error, 3.0);
    }
}

TEST_F(WorldLocalizingTest, TrackLocalization)
{
    buildTestMap(10, 2.0);
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);

    Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
    initial_pose.translation().x() = 4.0;
    auto initial_cloud = generateCorridorCloud(initial_pose);

    RelocResult initial_result = reloc.relocalize(initial_cloud, initial_pose);

    if (!initial_result.success) {
        reloc.setMapToOdomTransform(Eigen::Isometry3d::Identity());
    }

    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.translation().x() = 6.0;
    auto track_cloud = generateCorridorCloud(odom_pose);

    RelocResult track_result = reloc.trackLocalization(track_cloud, odom_pose);
    EXPECT_TRUE(track_result.success);
}

TEST_F(WorldLocalizingTest, Reset)
{
    buildTestMap(5, 2.0);
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
    T_map_odom.translation().x() = 1.0;
    reloc.setMapToOdomTransform(T_map_odom);
    EXPECT_TRUE(reloc.isRelocalized());
    reloc.reset();
    EXPECT_FALSE(reloc.isRelocalized());
    EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1);
    EXPECT_TRUE(reloc.getMapToOdomTransform().isApprox(Eigen::Isometry3d::Identity()));
}

TEST_F(WorldLocalizingTest, SetMapToOdomTransform)
{
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
    T_map_odom.translation() = Eigen::Vector3d(1.0, 2.0, 0.0);
    T_map_odom.rotate(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
    reloc.setMapToOdomTransform(T_map_odom);
    EXPECT_TRUE(reloc.isRelocalized());
    EXPECT_TRUE(reloc.getMapToOdomTransform().isApprox(T_map_odom, 1e-6));
}

TEST_F(WorldLocalizingTest, EmptyCloudInput)
{
    buildTestMap(5, 2.0);
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    auto empty_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    RelocResult result = reloc.relocalize(empty_cloud, Eigen::Isometry3d::Identity());
    EXPECT_FALSE(result.success);

    RelocResult result2 = reloc.relocalize(nullptr, Eigen::Isometry3d::Identity());
    EXPECT_FALSE(result2.success);
}

TEST_F(WorldLocalizingTest, PoseTransformConsistency)
{
    WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_, *matcher_);
    Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
    T_map_odom.translation() = Eigen::Vector3d(5.0, 3.0, 0.0);
    T_map_odom.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
    reloc.setMapToOdomTransform(T_map_odom);

    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
    Eigen::Isometry3d expected_map_pose = T_map_odom * odom_pose;
    Eigen::Isometry3d actual_map_pose = reloc.getMapToOdomTransform() * odom_pose;
    EXPECT_TRUE(expected_map_pose.isApprox(actual_map_pose, 1e-9));
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