// Does each config key arrive where it is meant to?
//
// test_config_reaches_behaviour.py asks whether anything reads a field at all.
// This asks the next question: that the value read is the right one. Five
// defects in this package were a field that looked wired and was not, and a
// sixth was a test named for a gate the production path had stopped consulting.
// Distinct values are used throughout so that copying a neighbouring field --
// the way those defects actually happen -- fails instead of passing by
// coincidence.

#include <gtest/gtest.h>

#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include "n3mapping/static_start_guard.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {
namespace {

TEST(ConfigWiringTest, StaticStartGuardOptionsCarryEveryKey) {
    Config config;
    config.mapping_static_voxel_m = 0.41;
    config.mapping_static_moved_overlap = 0.42;
    config.mapping_static_moved_consecutive = 43;
    config.mapping_static_max_wait_s = 44.0;

    const auto options = staticStartGuardOptionsFromConfig(config);
    EXPECT_DOUBLE_EQ(options.voxel_m, 0.41);
    EXPECT_DOUBLE_EQ(options.moved_overlap, 0.42);
    EXPECT_EQ(options.moved_consecutive, 43);
    EXPECT_DOUBLE_EQ(options.max_wait_s, 44.0);
}

TEST(ConfigWiringTest, StaticStartGuardDefaultsMatchTheConfigDefaults) {
    // The guard shipped with mapping-side defaults chosen on 0723; a test that
    // built Options directly would not notice them drifting apart from config.
    const auto options = staticStartGuardOptionsFromConfig(Config{});
    const Config defaults;
    EXPECT_DOUBLE_EQ(options.voxel_m, defaults.mapping_static_voxel_m);
    EXPECT_DOUBLE_EQ(options.moved_overlap, defaults.mapping_static_moved_overlap);
    EXPECT_EQ(options.moved_consecutive, defaults.mapping_static_moved_consecutive);
    EXPECT_DOUBLE_EQ(options.max_wait_s, defaults.mapping_static_max_wait_s);
}

TEST(ConfigWiringTest, VisibilityOptionsCarryEveryKey) {
    Config config;
    config.rhpd_max_range = 17.0;
    config.global_map_voxel_size = 0.11;
    config.gicp_downsampling_resolution = 0.23;   // the larger of the two
    config.reloc_visibility_occlusion_aware = true;

    const auto options = visibilityOptionsFromConfig(config);
    EXPECT_DOUBLE_EQ(options.range_max_m, 17.0);
    // The tolerance follows whichever resolution is coarser, three voxels of it.
    EXPECT_DOUBLE_EQ(options.range_tolerance_m, 3.0 * 0.23);
    EXPECT_TRUE(options.occlusion_aware);

    config.global_map_voxel_size = 0.31;          // now the coarser one
    EXPECT_DOUBLE_EQ(visibilityOptionsFromConfig(config).range_tolerance_m,
                     3.0 * 0.31);
}

TEST(ConfigWiringTest, VisibilityOptionsDefaultToThePublishedArithmetic) {
    // Occlusion awareness changes the quantity that decides every lock, so it
    // has to stay off until today20 says otherwise. A default that flipped by
    // accident would be invisible without this.
    EXPECT_FALSE(visibilityOptionsFromConfig(Config{}).occlusion_aware);
}

TEST(ConfigWiringTest, VisibilityRangeStaysUsableForAbsurdConfigurations) {
    Config config;
    config.rhpd_max_range = 0.0;
    EXPECT_GE(visibilityOptionsFromConfig(config).range_max_m, 1.0);
}

TEST(ConfigWiringTest, DescriptorParamsCarryEveryKey) {
    Config config;
    config.rhpd_max_range = 21.0;
    config.rhpd_z_min = -3.0;
    config.rhpd_z_max = 7.0;
    config.rhpd_v2_enable = false;
    config.rhpd_v3_enable = true;
    config.rhpd_enable_negative_space = false;
    config.rhpd_enable_vertical_tokens = false;
    config.rhpd_enable_pca_confidence = false;
    config.rhpd_part_a_scale = 0.25;
    config.rhpd_aux_scale = 0.75;

    const auto params = rhpdParamsFromConfig(config);
    EXPECT_DOUBLE_EQ(params.max_range, 21.0);
    EXPECT_DOUBLE_EQ(params.z_min, -3.0);
    EXPECT_DOUBLE_EQ(params.z_max, 7.0);
    EXPECT_FALSE(params.v2_enable);
    EXPECT_TRUE(params.v3_enable);
    EXPECT_FALSE(params.enable_negative_space);
    EXPECT_FALSE(params.enable_vertical_tokens);
    EXPECT_FALSE(params.enable_pca_confidence);
    EXPECT_DOUBLE_EQ(params.part_a_scale, 0.25);
    EXPECT_DOUBLE_EQ(params.aux_scale, 0.75);
}

TEST(ConfigWiringTest, DescriptorHeightRangeCannotCollapse) {
    Config config;
    config.rhpd_z_min = 2.0;
    config.rhpd_z_max = 1.0;   // inverted
    const auto params = rhpdParamsFromConfig(config);
    EXPECT_GT(params.z_max, params.z_min);
}

TEST(ConfigWiringTest, DescriptorBlockScalesDefaultToThePublishedDistance) {
    // distance() is shared with relocalization, which has a contract. Measured
    // on 0723, quartering these scales costs a correct lock, so one is not an
    // arbitrary default -- it is the published arithmetic.
    const auto params = rhpdParamsFromConfig(Config{});
    EXPECT_DOUBLE_EQ(params.part_a_scale, 1.0);
    EXPECT_DOUBLE_EQ(params.aux_scale, 1.0);
}

}  // namespace
}  // namespace n3mapping
