#include <gtest/gtest.h>

#include "n3mapping/config.h"

namespace n3mapping {
namespace test {

TEST(ConfigTest, DefaultValuesRemainStable) {
    Config config;

    EXPECT_EQ(config.mode, "mapping");
    EXPECT_TRUE(config.map_path.empty());
    EXPECT_FALSE(config.map_save_path.empty());
    EXPECT_EQ(config.cloud_topic, "/cloud_registered_body");
    EXPECT_EQ(config.odom_topic, "/Odometry");
    EXPECT_TRUE(config.rhpd_enabled);
    EXPECT_DOUBLE_EQ(config.rhpd_primary_weight, 1.0);
    EXPECT_DOUBLE_EQ(config.sc_aux_weight, 0.15);
    EXPECT_EQ(config.rhpd_preselect_candidates, 100);
    EXPECT_EQ(config.reloc_lock_min_winner_streak, 3);
    EXPECT_FALSE(config.reloc_debug_enable);
    EXPECT_TRUE(config.reloc_debug_path.empty());
    EXPECT_FALSE(config.reloc_atlas_enable);
    EXPECT_TRUE(config.reloc_atlas_path.empty());
    EXPECT_EQ(config.reloc_target_mode, "legacy_global_atlas");
    EXPECT_EQ(config.reloc_target_cache_max_bytes, 0);
    EXPECT_EQ(config.reloc_target_cache_max_entries, 0);
    EXPECT_DOUBLE_EQ(config.loop_icp_prefilter_voxel_size, 0.2);
    EXPECT_EQ(config.loop_icp_max_points, 50000);
    EXPECT_TRUE(config.loop_spatial_candidates_enable);
    EXPECT_DOUBLE_EQ(config.loop_spatial_candidate_radius, 15.0);
    EXPECT_EQ(config.loop_spatial_candidate_max_candidates, 5);
    EXPECT_FALSE(config.floor_attitude_enable);
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
    EXPECT_NE(summary.find("Loop candidate pipeline: descriptor + spatial proposals"), std::string::npos);
    EXPECT_NE(summary.find("Loop spatial candidates: ON"), std::string::npos);
    EXPECT_NE(summary.find("Floor attitude (experimental): OFF"),
              std::string::npos);
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

TEST(ConfigTest, LoadedMapModesRequireExplicitMapPath) {
    Config config;
    std::string error;

    config.mode = "localization";
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("map_path"), std::string::npos);

    config.mode = "map_extension";
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("map_path"), std::string::npos);

    config.map_path = "/deployment/map.pbstream";
    EXPECT_TRUE(config.validate(&error)) << error;
}

TEST(ConfigTest, RelocTargetProviderValidation) {
    Config config;
    std::string error;

    for (const char* mode : {"legacy_global_atlas", "local_no_cache",
                             "local_lru", "shadow_local_lru"}) {
        config.reloc_target_mode = mode;
        EXPECT_TRUE(config.validate(&error)) << mode << ": " << error;
    }

    config.reloc_target_mode = "implicit_magic_cache";
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_target_mode"), std::string::npos);

    config = Config{};
    config.reloc_target_cache_max_bytes = -1;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_target_cache_max_bytes"), std::string::npos);

    config = Config{};
    config.reloc_target_cache_max_entries = -1;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_target_cache_max_entries"), std::string::npos);
}

TEST(ConfigTest, ProductFactoryFreezesLocalizationProfileAndPaths) {
    const Config product = makeProductLocalizationConfig(
        "/deployment/map.pbstream",
        "/deployment/map.pbstream.localization_atlas.pb");
    EXPECT_EQ(product.mode, "localization");
    EXPECT_EQ(product.map_path, "/deployment/map.pbstream");
    EXPECT_TRUE(product.reloc_atlas_enable);
    EXPECT_EQ(product.reloc_atlas_path,
              "/deployment/map.pbstream.localization_atlas.pb");
    EXPECT_FALSE(product.save_global_map_on_shutdown);
    std::string error;
    EXPECT_TRUE(product.validate(&error)) << error;
}

TEST(ConfigTest, FreeSpaceAndPersistenceDefaultsMatchEnvUnsetBehaviour) {
    // These were process-environment switches read inside WorldLocalizing.
    // The defaults must equal the env-unset behaviour exactly, or a default
    // YAML run changes on this refactor.
    const Config config;
    EXPECT_TRUE(config.reloc_free_space_enable);
    EXPECT_EQ(config.reloc_free_space_mode, "veto");
    EXPECT_DOUBLE_EQ(config.reloc_free_space_resolution, 0.20);
    EXPECT_DOUBLE_EQ(config.reloc_free_space_max_ray_length, 30.0);
    EXPECT_EQ(config.reloc_free_space_occupied_min_points, 2);
    EXPECT_DOUBLE_EQ(config.reloc_free_space_kill_sigmas, 5.0);
    EXPECT_TRUE(config.reloc_free_space_map_pcd.empty());
    EXPECT_FALSE(config.reloc_persist_hypotheses);
    EXPECT_EQ(config.reloc_persist_max_frames, 300);

    const Config product = makeProductLocalizationConfig("/m.pbstream", "/a.pb");
    EXPECT_TRUE(product.reloc_free_space_enable);
    EXPECT_EQ(product.reloc_free_space_mode, "veto");
    EXPECT_EQ(product.reloc_persist_max_frames, 300);
}

TEST(ConfigTest, FreeSpaceAndPersistenceValidation) {
    Config config;
    std::string error;

    EXPECT_TRUE(config.validate(&error)) << error;

    config.reloc_free_space_mode = "kill";
    EXPECT_TRUE(config.validate(&error)) << error;
    config.reloc_free_space_mode = "veto";
    EXPECT_TRUE(config.validate(&error)) << error;

    config.reloc_free_space_mode = "banana";
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_mode"), std::string::npos);

    config = Config{};
    config.reloc_free_space_occupied_min_points = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_occupied_min_points"), std::string::npos);

    config = Config{};
    config.reloc_free_space_occupied_min_points = 256;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_occupied_min_points"), std::string::npos);

    config = Config{};
    config.reloc_free_space_resolution = 0.0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_resolution"), std::string::npos);

    config = Config{};
    config.reloc_free_space_max_ray_length = -1.0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_max_ray_length"), std::string::npos);

    config = Config{};
    config.reloc_free_space_kill_sigmas = -0.5;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_free_space_kill_sigmas"), std::string::npos);

    config = Config{};
    config.reloc_persist_max_frames = 0;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("reloc_persist_max_frames"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
