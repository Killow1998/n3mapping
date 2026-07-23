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
    loaded.map_path = "/tmp/product_map.pbstream";
    loaded.reloc_atlas_path =
        "/tmp/product_map.pbstream.localization_atlas.pb";

    const Config gate = makeProductLocalizationConfig(
        loaded.map_path, loaded.reloc_atlas_path);
    EXPECT_EQ(runtimeConfigCanonical(loaded),
              runtimeConfigCanonical(gate));

    node.reset();
    if (initialized_here) {
        rclcpp::shutdown();
    }
}

}  // namespace test
}  // namespace n3mapping
