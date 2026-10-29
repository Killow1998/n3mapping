#include "n3mapping/humble/config_humble.h"
#include "n3mapping/config_fields.h"

namespace n3mapping {

void loadConfigFromHumble(rclcpp::Node* node, Config* config) {
    const std::string default_map_save_path = config->map_save_path;
    forEachConfigField(*config, [&](const char* name, auto& value) {
        node->declare_parameter(name, value);
        node->get_parameter(name, value);
    });
    // An empty map directory selects the caller's default; other strings may be empty.
    if (config->map_save_path.empty()) config->map_save_path = default_map_save_path;
}

}  // namespace n3mapping
