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

}  // namespace test
}  // namespace n3mapping
