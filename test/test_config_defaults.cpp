#include <gtest/gtest.h>

#include "n3mapping/config.h"

namespace n3mapping {
namespace test {

TEST(ConfigDefaultsTest, KeepsExternalLioCompatibilityBaseline) {
    Config config;
    EXPECT_EQ(config.frontend_mode, "external");
    EXPECT_EQ(config.cloud_topic, "/cloud_registered_body");
    EXPECT_EQ(config.odom_topic, "/Odometry");
    EXPECT_EQ(config.raw_lidar_topic, "/points_raw");
    EXPECT_EQ(config.raw_lidar_msg_type, "pointcloud2");
    EXPECT_EQ(config.imu_topic, "/imu");
}

TEST(ConfigDefaultsTest, KeepsRHPDPrimaryAndScanContextAuxiliaryDefaults) {
    Config config;
    EXPECT_TRUE(config.rhpd_enabled);
    EXPECT_DOUBLE_EQ(config.rhpd_primary_weight, 1.0);
    EXPECT_GT(config.rhpd_num_candidates, 0);
    EXPECT_GT(config.rhpd_preselect_candidates, config.rhpd_num_candidates);
    EXPECT_GT(config.sc_aux_weight, 0.0);
    EXPECT_LT(config.sc_aux_weight, config.rhpd_primary_weight);
    EXPECT_TRUE(config.rhpd_use_sc_yaw);
}

TEST(ConfigDefaultsTest, DisablesBuiltinFrontendDebugPublishingByDefault) {
    Config config;
    EXPECT_FALSE(config.frontend_publish_debug);
    EXPECT_FALSE(config.frontend_debug_publish_odom);
    EXPECT_FALSE(config.frontend_debug_publish_deskewed_cloud);
    EXPECT_FALSE(config.frontend_debug_publish_local_map);
    EXPECT_FALSE(config.frontend_debug_publish_timing);
}

TEST(ConfigDefaultsTest, UsesIdentityFrontendExtrinsicsByDefault) {
    Config config;
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_tx, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_ty, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_tz, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_roll, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_pitch, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_lidar_to_body_yaw, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_tx, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_ty, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_tz, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_roll, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_pitch, 0.0);
    EXPECT_DOUBLE_EQ(config.frontend_imu_to_body_yaw, 0.0);
}

}  // namespace test
}  // namespace n3mapping
