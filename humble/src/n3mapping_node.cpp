#include "n3mapping_node.h"
#include "n3mapping/async_log.h"
#include "n3mapping/product_build_identity.h"

#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>

void
signalHandler(int signum)
{
    (void)signum;
    n3mapping::g_shutdown_requested = true;
}

std::optional<std::string>
extractAuthorityProbeScenario(int* argc, char** argv)
{
    std::optional<std::string> scenario;
    int write_index = 1;
    for (int read_index = 1; read_index < *argc; ++read_index) {
        const std::string_view argument(argv[read_index]);
        if (argument == "--product-authority-probe") {
            if (scenario || read_index + 1 >= *argc) {
                throw std::runtime_error(
                  "--product-authority-probe requires exactly one scenario");
            }
            scenario = std::string(argv[++read_index]);
            continue;
        }
        argv[write_index++] = argv[read_index];
    }
    *argc = write_index;
    argv[write_index] = nullptr;
    return scenario;
}

int
main(int argc, char** argv)
{
    if (n3mapping::productBuildIdentityRequested(argc, argv)) {
        std::cout << n3mapping::productBuildIdentityJson() << '\n';
        return 0;
    }
    const auto authority_probe_scenario =
      extractAuthorityProbeScenario(&argc, argv);
    n3mapping::ProcessLogging logging(argv[0]);
    n3mapping::RosLogging ros_logging;
    rclcpp::InitOptions init_options;
    init_options.auto_initialize_logging(false);
    rclcpp::init(argc, argv, init_options);

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    auto node =
      std::make_shared<n3mapping::N3MappingNode>(authority_probe_scenario);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    node->startLiveOutput();

    while (rclcpp::ok() && !n3mapping::g_shutdown_requested) {
        node->spinOnce(executor);
    }

    node->shutdown();
    rclcpp::shutdown();
    return 0;
}
