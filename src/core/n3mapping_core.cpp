#include "n3mapping/core/n3mapping_core.h"

#include "n3mapping/core/n3mapping_session.h"

namespace n3mapping {
namespace core {
namespace {

double toSeconds(const TimeStamp& stamp) { return static_cast<double>(stamp.nsec) * 1e-9; }

}  // namespace

class N3MappingCore::Impl {
public:
    explicit Impl(const Config& config)
        : session_(config) {}

    MappingOutput processLioFrame(const LioFrame& frame) {
        MappingOutput output;
        output.T_world_lidar = frame.T_world_lidar;
        if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
            return output;
        }
        const auto result = session_.mappingModeProcessor().process(
            toSeconds(frame.stamp),
            frame.T_world_lidar,
            frame.undistorted_cloud,
            frame.covariance_valid ? &frame.covariance : nullptr);
        output.accepted_keyframe = result.accepted_keyframe;
        output.keyframe_id = result.keyframe_id;
        output.T_world_lidar = result.publish_pose;
        return output;
    }

    RelocResult relocalize(const LioFrame& frame) {
        if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
            return RelocResult{};
        }
        return session_.worldLocalizing().relocalize(frame.undistorted_cloud, frame.T_world_lidar);
    }

    bool saveMap(const std::string& path) {
        return session_.saveCurrentMap(path);
    }

    bool loadMap(const std::string& path) {
        return session_.loadMapForLocalization(path);
    }

private:
    N3MappingSession session_;
};

N3MappingCore::N3MappingCore(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

N3MappingCore::~N3MappingCore() = default;

MappingOutput N3MappingCore::processLioFrame(const LioFrame& frame) {
    return impl_->processLioFrame(frame);
}

RelocResult N3MappingCore::relocalize(const LioFrame& frame) {
    return impl_->relocalize(frame);
}

bool N3MappingCore::saveMap(const std::string& path) {
    return impl_->saveMap(path);
}

bool N3MappingCore::loadMap(const std::string& path) {
    return impl_->loadMap(path);
}

}  // namespace core
}  // namespace n3mapping
