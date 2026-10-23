#include "n3mapping_node.h"
#include "n3mapping/async_log.h"
#include <rcutils/logging.h>
#include <cstdio>
#include <stdexcept>

namespace n3mapping {
namespace {
rcutils_logging_output_handler_t previous_handler = nullptr;
void forward(const rcutils_log_location_t*, int severity, const char* name,
             rcutils_time_point_value_t, const char* format, va_list* args) {
    char buffer[16384];
    va_list copy; va_copy(copy, *args);
    const int size = std::vsnprintf(buffer, sizeof(buffer), format, copy);
    va_end(copy);
    try {
        if (size < 0) { AsyncLog::instance().count("ros_format_failure"); return; }
        if (size >= static_cast<int>(sizeof(buffer))) AsyncLog::instance().count("ros_message_truncated");
        forwardRuntimeLog(severity, name, buffer);
    } catch (...) { }
}
}
RosLogging::RosLogging() {
    if (rcutils_logging_initialize() != RCUTILS_RET_OK)
        throw std::runtime_error("rcutils logging initialization failed");
    previous_handler = rcutils_logging_get_output_handler();
    rcutils_logging_set_output_handler(forward);
}
RosLogging::~RosLogging() { rcutils_logging_set_output_handler(previous_handler); }
} // namespace n3mapping
