#include "n3mapping_node.h"
#include "n3mapping/async_log.h"
#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

namespace n3mapping {
namespace {
class Forwarder final : public ros::console::LogAppender {
    void log(ros::console::Level level, const char* message, const char* file,
             const char*, int) override {
        try { forwardRuntimeLog(10 + 10 * static_cast<int>(level), file, message); }
        catch (...) { }
    }
} forwarder;
}
RosLogging::RosLogging() {
    ros::console::initialize();
    log4cxx::Logger::getRootLogger()->removeAllAppenders();
    for (const auto& logger : log4cxx::LogManager::getCurrentLoggers()) logger->removeAllAppenders();
    ros::console::register_appender(&forwarder);
}
RosLogging::~RosLogging() {
    // ros::shutdown already removes the appender. Noetic's deregister_appender
    // dereferences that cleared pointer; console shutdown is safe to repeat.
    ros::console::shutdown();
}
} // namespace n3mapping
