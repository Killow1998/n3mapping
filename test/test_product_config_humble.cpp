#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "n3mapping/config.h"
#include "n3mapping/humble/config_humble.h"

namespace n3mapping {
namespace test {

TEST(ProductConfigHumbleTest, RosParameterFileMatchesGateFactory) {
    bool initialized_here = false;
    if (!rclcpp::ok()) {
        int argc = 0;
        char** argv = nullptr;
        rclcpp::init(argc, argv);
        initialized_here = true;
    }

    const std::string config_path =
        std::string(N3MAPPING_SOURCE_DIR) + "/config/product_v1.yaml";
    rclcpp::NodeOptions options;
    options.arguments(
        {"--ros-args", "--params-file", config_path});
    auto node = std::make_shared<rclcpp::Node>("n3mapping_node", options);

    Config loaded;
    loadConfigFromHumble(node.get(), &loaded);
    EXPECT_EQ(loaded.map_save_path, Config{}.map_save_path);
    EXPECT_FALSE(loaded.floor_attitude_enable);
    loaded.map_path = "/tmp/product_map.pbstream";
    loaded.reloc_atlas_path =
        "/tmp/product_map.pbstream.localization_atlas.pb";

    const Config gate = makeProductLocalizationConfig(
        loaded.map_path, loaded.reloc_atlas_path);
    EXPECT_EQ(runtimeConfigCanonical(loaded),
              runtimeConfigCanonical(gate));

    node.reset();

    const std::string mapping_config_path =
        std::string(N3MAPPING_SOURCE_DIR) + "/config/n3mapping.yaml";
    rclcpp::NodeOptions mapping_options;
    mapping_options.arguments(
        {"--ros-args", "--params-file", mapping_config_path});
    auto mapping_node =
        std::make_shared<rclcpp::Node>("n3mapping_node", mapping_options);
    Config mapping_config;
    loadConfigFromHumble(mapping_node.get(), &mapping_config);
    EXPECT_FALSE(mapping_config.floor_attitude_enable);
    mapping_node.reset();

    const std::string experimental_config_path =
        std::string(N3MAPPING_SOURCE_DIR) +
        "/config/mapping_stages/s13_current_best.yaml";
    rclcpp::NodeOptions experimental_options;
    experimental_options.arguments(
        {"--ros-args", "--params-file", experimental_config_path});
    auto experimental_node =
        std::make_shared<rclcpp::Node>("n3mapping_node", experimental_options);
    Config experimental_config;
    loadConfigFromHumble(experimental_node.get(), &experimental_config);
    EXPECT_TRUE(experimental_config.floor_attitude_enable);
    experimental_node.reset();

    rclcpp::NodeOptions override_options;
    override_options.append_parameter_override(
        "map_save_path", "/tmp/n3mapping_explicit_map");
    auto override_node = std::make_shared<rclcpp::Node>(
        "n3mapping_map_save_path_override", override_options);
    Config overridden;
    loadConfigFromHumble(override_node.get(), &overridden);
    EXPECT_EQ(overridden.map_save_path, "/tmp/n3mapping_explicit_map");

    override_node.reset();
    if (initialized_here) {
        rclcpp::shutdown();
    }
}

}  // namespace test
}  // namespace n3mapping
