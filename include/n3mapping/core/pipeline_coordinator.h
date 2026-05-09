// ROS-independent in-process pipeline coordinator.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/core/types.h"
#include "n3mapping/lio/external_frontend.h"
#include "n3mapping/lio/frontend_factory.h"

namespace n3mapping {
namespace core {

class PipelineCoordinator {
public:
    enum class RunMode {
        Mapping,
        Localization,
        MapExtension,
    };

    struct Output {
        bool success = false;
        bool has_lio_frame = false;
        bool accepted_keyframe = false;
        bool relocalization_locked = false;
        int64_t keyframe_id = -1;
        Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
        std::string error;
    };

    explicit PipelineCoordinator(const Config& config);

    bool ready() const;
    const std::string& error() const;
    RunMode mode() const;
    lio::FrontendCapability frontendCapability() const;
    N3MappingSession& session();
    const N3MappingSession& session() const;

    void setLioDebugCallbacks(const lio::LioDebugCallbacks& callbacks);
    void addImu(const ImuSample& imu);
    Output addRawLidar(const RawLidarFrame& frame);
    Output addExternalFrame(const TimeStamp& stamp,
                            const Eigen::Isometry3d& T_world_lidar,
                            const LioFrame::PointCloud::Ptr& cloud,
                            const Eigen::Matrix<double, 6, 6>& covariance =
                                Eigen::Matrix<double, 6, 6>::Identity(),
                            bool covariance_valid = false);
    Output processLioFrame(const LioFrame& frame);

    bool loadMap(const std::string& path);
    bool saveMap(const std::string& path);

private:
    static RunMode parseRunMode(const std::string& mode);

    Config config_;
    RunMode mode_ = RunMode::Mapping;
    N3MappingSession session_;
    std::unique_ptr<lio::LioFrontend> frontend_;
    lio::ExternalLioFrontend* external_frontend_ = nullptr;
    bool map_loaded_ = false;
    std::string error_;
};

}  // namespace core
}  // namespace n3mapping
