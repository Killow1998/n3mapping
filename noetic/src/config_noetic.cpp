#include "n3mapping_noetic/config_noetic.h"
#include "n3mapping/config_fields.h"

namespace n3mapping {

void loadConfigFromNoetic(ros::NodeHandle& node_handle, Config* config) {
    const std::string default_map_save_path = config->map_save_path;
    forEachConfigField(*config, [&](const char* name, auto& value) {
        const std::string key(name);
        node_handle.param("n3mapping_node/ros__parameters/" + key, value, value);
        node_handle.param("ros__parameters/" + key, value, value);
        node_handle.param(key, value, value);
    });
    // An empty map directory selects the caller's default; other strings may be empty.
    if (config->map_save_path.empty()) config->map_save_path = default_map_save_path;
}

}  // namespace n3mapping
