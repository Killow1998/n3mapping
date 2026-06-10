#include <gtest/gtest.h>

#include "n3mapping/config.h"

namespace n3mapping {
namespace test {

TEST(ConfigTest, DefaultValuesRemainStable) {
    Config config;

    EXPECT_EQ(config.mode, "mapping");
    EXPECT_EQ(config.cloud_topic, "/cloud_registered_body");
    EXPECT_EQ(config.odom_topic, "/Odometry");
    EXPECT_TRUE(config.rhpd_enabled);
    EXPECT_DOUBLE_EQ(config.rhpd_primary_weight, 1.0);
    EXPECT_DOUBLE_EQ(config.sc_aux_weight, 0.15);
    EXPECT_EQ(config.rhpd_preselect_candidates, 100);
    EXPECT_EQ(config.reloc_lock_min_winner_streak, 3);
    EXPECT_DOUBLE_EQ(config.save_global_map_voxel_size, 0.1);
}

TEST(ConfigTest, ToStringContainsKeyFields) {
    Config config;
    config.mode = "localization";
    config.map_path = "/tmp/test.pbstream";
    config.rhpd_enabled = true;
    config.sc_aux_veto_enabled = false;

    const std::string summary = config.toString();

    EXPECT_NE(summary.find("Mode: localization"), std::string::npos);
    EXPECT_NE(summary.find("Map path: /tmp/test.pbstream"), std::string::npos);
    EXPECT_NE(summary.find("Loop candidate pipeline: RHPD primary retrieval"), std::string::npos);
    EXPECT_NE(summary.find("RHPD primary retrieval: weight="), std::string::npos);
    EXPECT_NE(summary.find("Reloc temporal: window="), std::string::npos);
}

TEST(ConfigTest, RejectsZeroNoiseAndNegativeVoxelParameters) {
    Config config;
    std::string error;

    EXPECT_TRUE(config.validate(&error));

    config.odom_noise_position = 0.0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("odom_noise_position"), std::string::npos);

    config = Config{};
    config.gicp_downsampling_resolution = 0.0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("gicp_downsampling_resolution"), std::string::npos);

    config = Config{};
    config.save_global_map_voxel_size = -0.1;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("save_global_map_voxel_size"), std::string::npos);

    config = Config{};
    config.num_threads = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("num_threads"), std::string::npos);

    config = Config{};
    config.rhpd_num_candidates = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("rhpd_num_candidates"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
