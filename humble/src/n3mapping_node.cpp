#include "n3mapping_node.h"
#include "n3mapping/product_build_identity.h"

#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

// 信号处理
void
signalHandler(int signum)
{
    (void)signum; // 避免未使用参数警告
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
    rclcpp::init(argc, argv);

    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    auto node =
      std::make_shared<n3mapping::N3MappingNode>(authority_probe_scenario);

    // The backend remains serial. Live odometry and cloud groups are explicitly
    // owned by two independent executors, including message-filter subscribers.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    node->startLiveOutput();

    while (rclcpp::ok() && !n3mapping::g_shutdown_requested) {
        executor.spin_some();
        node->processPendingLocalizationFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    node->shutdown();
    rclcpp::shutdown();
    return 0;
}
