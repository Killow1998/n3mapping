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
    EXPECT_FALSE(config.reloc_debug_enable);
    EXPECT_TRUE(config.reloc_debug_path.empty());
    EXPECT_DOUBLE_EQ(config.loop_icp_prefilter_voxel_size, 0.2);
    EXPECT_EQ(config.loop_icp_max_points, 50000);
    EXPECT_FALSE(config.loop_spatial_candidates_enable);
    EXPECT_DOUBLE_EQ(config.loop_spatial_candidate_radius, 15.0);
    EXPECT_EQ(config.loop_spatial_candidate_min_id_gap, 50);
    EXPECT_EQ(config.loop_spatial_candidate_max_candidates, 5);
    EXPECT_DOUBLE_EQ(config.save_global_map_voxel_size, 0.1);
    EXPECT_EQ(config.sync_queue_size, 100);
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
    EXPECT_NE(summary.find("Loop spatial candidates: OFF"), std::string::npos);
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
    config.loop_icp_prefilter_voxel_size = -0.1;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("loop_icp_prefilter_voxel_size"), std::string::npos);

    config = Config{};
    config.loop_icp_max_points = -1;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("loop_icp_max_points"), std::string::npos);

    config = Config{};
    config.loop_spatial_candidate_radius = 0.0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("loop_spatial_candidate_radius"), std::string::npos);

    config = Config{};
    config.loop_spatial_candidate_min_id_gap = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("loop_spatial_candidate_min_id_gap"), std::string::npos);

    config = Config{};
    config.loop_spatial_candidate_max_candidates = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("loop_spatial_candidate_max_candidates"), std::string::npos);

    config = Config{};
    config.num_threads = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("num_threads"), std::string::npos);

    config = Config{};
    config.sync_queue_size = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("sync_queue_size"), std::string::npos);

    config = Config{};
    config.rhpd_num_candidates = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("rhpd_num_candidates"), std::string::npos);
}

TEST(ConfigTest, RejectsUnknownMode) {
    Config config;
    std::string error;

    config.mode = "localizaton";
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("mode"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
