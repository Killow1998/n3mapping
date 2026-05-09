#include "n3mapping/core/pipeline_coordinator.h"

namespace n3mapping {
namespace core {

PipelineCoordinator::PipelineCoordinator(const Config& config)
    : config_(config)
    , mode_(parseRunMode(config.mode))
    , session_(config) {
    auto frontend_result = lio::createLioFrontend(config_);
    if (!frontend_result.ok()) {
        error_ = frontend_result.error;
        return;
    }
    frontend_ = std::move(frontend_result.frontend);
    external_frontend_ = dynamic_cast<lio::ExternalLioFrontend*>(frontend_.get());
}

bool PipelineCoordinator::ready() const {
    return frontend_ != nullptr && error_.empty();
}

const std::string& PipelineCoordinator::error() const {
    return error_;
}

PipelineCoordinator::RunMode PipelineCoordinator::mode() const {
    return mode_;
}

N3MappingSession& PipelineCoordinator::session() {
    return session_;
}

const N3MappingSession& PipelineCoordinator::session() const {
    return session_;
}

void PipelineCoordinator::addImu(const ImuSample& imu) {
    if (frontend_) {
        frontend_->addImu(imu);
    }
}

PipelineCoordinator::Output PipelineCoordinator::addRawLidar(const RawLidarFrame& frame) {
    Output output;
    if (!ready()) {
        output.error = error_.empty() ? "LIO frontend is not ready" : error_;
        return output;
    }

    auto lio_frame = frontend_->addLidar(frame);
    if (!lio_frame) {
        return output;
    }
    return processLioFrame(*lio_frame);
}

PipelineCoordinator::Output PipelineCoordinator::addExternalFrame(
    const TimeStamp& stamp,
    const Eigen::Isometry3d& T_world_lidar,
    const LioFrame::PointCloud::Ptr& cloud,
    const Eigen::Matrix<double, 6, 6>& covariance) {
    Output output;
    if (!ready()) {
        output.error = error_.empty() ? "LIO frontend is not ready" : error_;
        return output;
    }
    if (!external_frontend_) {
        output.error = "addExternalFrame requires frontend_mode=external";
        return output;
    }

    auto lio_frame =
        external_frontend_->addSynchronizedFrame(stamp, T_world_lidar, cloud, covariance);
    if (!lio_frame) {
        output.error = "external LIO frame was rejected";
        return output;
    }
    return processLioFrame(*lio_frame);
}

PipelineCoordinator::Output PipelineCoordinator::processLioFrame(const LioFrame& frame) {
    Output output;
    output.T_world_lidar = frame.T_world_lidar;
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        output.error = "invalid LIO frame";
        return output;
    }
    output.has_lio_frame = true;

    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    switch (mode_) {
        case RunMode::Mapping: {
            auto result = session_.mappingModeProcessor().process(
                timestamp, frame.T_world_lidar, frame.undistorted_cloud);
            output.success = true;
            output.accepted_keyframe = result.accepted_keyframe;
            output.keyframe_id = result.keyframe_id;
            output.T_world_lidar = result.publish_pose;
            return output;
        }
        case RunMode::Localization: {
            auto result = session_.localizationModeProcessor().process(
                map_loaded_, frame.T_world_lidar, frame.undistorted_cloud);
            output.success = result.success;
            output.relocalization_locked = result.relocalization_locked;
            output.T_world_lidar = result.publish_pose;
            if (!result.map_loaded) output.error = "map is not loaded";
            return output;
        }
        case RunMode::MapExtension: {
            auto result = session_.mapExtensionModeProcessor().process(
                map_loaded_, timestamp, frame.T_world_lidar, frame.undistorted_cloud);
            output.success = result.should_publish || result.initial_relocalization_success;
            output.accepted_keyframe = result.accepted_keyframe;
            output.keyframe_id = result.keyframe_id;
            output.T_world_lidar = result.publish_pose;
            if (!result.map_loaded) output.error = "map is not loaded";
            return output;
        }
    }
    output.error = "unknown run mode";
    return output;
}

bool PipelineCoordinator::loadMap(const std::string& path) {
    if (mode_ == RunMode::MapExtension) {
        map_loaded_ = session_.loadMapForExtension(path);
    } else {
        map_loaded_ = session_.loadMapForLocalization(path);
    }
    return map_loaded_;
}

bool PipelineCoordinator::saveMap(const std::string& path) {
    if (mode_ == RunMode::MapExtension) {
        return session_.saveExtendedMap(path);
    }
    return session_.saveCurrentMap(path);
}

PipelineCoordinator::RunMode PipelineCoordinator::parseRunMode(const std::string& mode) {
    if (mode == "localization") return RunMode::Localization;
    if (mode == "map_extension") return RunMode::MapExtension;
    return RunMode::Mapping;
}

}  // namespace core
}  // namespace n3mapping
