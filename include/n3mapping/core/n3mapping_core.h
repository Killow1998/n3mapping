// Public draft facade for the future ROS-independent n3mapping core API.
#pragma once

#include <memory>
#include <string>

#include "n3mapping/config.h"
#include "n3mapping/core/types.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {
namespace core {

class N3MappingCore {
public:
    using Ptr = std::shared_ptr<N3MappingCore>;

    explicit N3MappingCore(const Config& config);
    ~N3MappingCore();

    MappingOutput processLioFrame(const LioFrame& frame);
    RelocResult relocalize(const LioFrame& frame);
    bool saveMap(const std::string& path);
    bool loadMap(const std::string& path);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace n3mapping

