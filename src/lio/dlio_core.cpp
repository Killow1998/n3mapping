#include "n3mapping/lio/dlio_core.h"

#include "n3mapping/lio/dlio_settings.h"

#include <cmath>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

MapAccumulator::Options makeDenseMapOptions(const LioFrontendConfig& config) {
    MapAccumulator::Options options;
    options.leaf_size = config.dlio_dense_map_leaf_size;
    options.dense_input_skip = config.dlio_dense_input_skip;
    return options;
}

double secondsFromNsec(int64_t nsec) {
    return static_cast<double>(nsec) * 1.0e-9;
}

core::TimeStamp nsecFromSeconds(double stamp_sec) {
    core::TimeStamp stamp;
    stamp.nsec = static_cast<int64_t>(std::llround(stamp_sec * 1.0e9));
    return stamp;
}

std::vector<ImuMeasurement> makeDlioImuMeasurements(
    const std::vector<core::ImuSample>& samples) {
    std::vector<ImuMeasurement> measurements;
    measurements.reserve(samples.size());
    for (const auto& sample : samples) {
        ImuMeasurement measurement;
        measurement.stamp = secondsFromNsec(sample.stamp.nsec);
        if (!measurements.empty() &&
            measurement.stamp <= measurements.back().stamp) {
            continue;
        }
        measurement.dt = measurements.empty()
                             ? 0.0
                             : measurement.stamp - measurements.back().stamp;
        measurement.angular_velocity =
            sample.angular_velocity.cast<float>();
        measurement.linear_acceleration =
            sample.linear_accel.cast<float>();
        measurements.push_back(measurement);
    }
    return measurements;
}

const IntegratedPose& referencePose(
    const std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>& states,
    double stamp) {
    auto it = std::lower_bound(
        states.begin(), states.end(), stamp,
        [](const IntegratedPose& state, double value) {
            return state.stamp < value;
        });
    if (it == states.begin()) {
        return states.front();
    }
    if (it == states.end()) {
        return states.back();
    }
    const auto prev = std::prev(it);
    return std::abs(prev->stamp - stamp) <= std::abs(it->stamp - stamp)
               ? *prev
               : *it;
}

ImuPropagationState propagationFromIntegratedPose(
    const IntegratedPose& state,
    size_t used_samples) {
    ImuPropagationState propagation;
    propagation.stamp = nsecFromSeconds(state.stamp);
    propagation.orientation =
        Eigen::Quaterniond(state.T.block<3, 3>(0, 0).cast<double>());
    propagation.position = state.T.block<3, 1>(0, 3).cast<double>();
    propagation.velocity = state.velocity.cast<double>();
    propagation.used_samples = used_samples;
    propagation.valid = true;
    return propagation;
}

}  // namespace

Core::Core(const LioFrontendConfig& config)
    : config_(config),
      imu_buffer_(config.imu_buffer_max_samples),
      dense_map_(makeDenseMapOptions(config)) {}

void Core::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> Core::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    last_input_packet_ = buildInputPacket(frame, imu_buffer_, cloudOptions());
    last_scan_timing_ = computeScanTiming(frame);
    std::vector<double> integration_timestamps;
    std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>
        integrated_states;
    if (last_input_packet_.has_complete_imu_window &&
        last_input_packet_.imu_samples.size() >= 2 &&
        last_scan_timing_.valid) {
        ImuIntegrationRequest request;
        request.start_time = last_scan_timing_.stamp_begin;
        integration_timestamps = last_scan_timing_.has_point_timing
                                     ? last_scan_timing_.unique_point_timestamps
                                     : std::vector<double>{last_scan_timing_.stamp_median};
        request.sorted_timestamps = integration_timestamps;
        request.gravity = config_.dlio_gravity;
        if (predicted_state_) {
            request.q_init =
                Eigen::Quaternionf(predicted_state_->T_world_lidar.linear().cast<float>());
            request.p_init =
                predicted_state_->T_world_lidar.translation().cast<float>();
            request.v_init = predicted_state_->velocity_world.cast<float>();
        }

        integrated_states = integrateImuStates(
            makeDlioImuMeasurements(last_input_packet_.imu_samples),
            request);
        if (!integrated_states.empty()) {
            const auto& state =
                referencePose(integrated_states, last_scan_timing_.stamp_median);
            const auto propagation = propagationFromIntegratedPose(
                state, last_input_packet_.imu_samples.size());
            last_imu_propagation_ = propagation;
            predicted_state_ = stateFromImuPropagation(propagation);
        } else {
            last_imu_propagation_.reset();
            predicted_state_.reset();
        }
    } else {
        last_imu_propagation_.reset();
        predicted_state_.reset();
    }
    if (config_.prediction_only_output && predicted_state_ && frame.points &&
        !frame.points->empty()) {
        last_alignment_stats_ =
            local_map_.estimateAlignmentCorrection(
                *predicted_state_,
                frame.points,
                config_.alignment_max_correspondence_distance);
        if (last_alignment_stats_.valid) {
            predicted_state_->T_world_lidar.translation() +=
                last_alignment_stats_.centroid_correction_world;
        }
        auto deskewed_cloud = frame.points;
        auto dense_map_cloud = frame.points;
        if (last_imu_propagation_) {
            const auto deskewed = deskewToReference(
                frame,
                last_scan_timing_,
                integrated_states,
                predicted_state_->T_world_lidar.matrix().cast<float>());
            if (deskewed.valid && deskewed.cloud && !deskewed.cloud->empty()) {
                deskewed_cloud = deskewed.cloud;
            }
            const auto world_deskewed =
                deskewToWorld(frame, last_scan_timing_, integrated_states);
            if (world_deskewed.valid && world_deskewed.cloud &&
                !world_deskewed.cloud->empty()) {
                dense_map_cloud = world_deskewed.cloud;
            }
        }
        local_map_.addFrame(*predicted_state_, deskewed_cloud);
        auto output = frameFromState(*predicted_state_);
        output.undistorted_cloud = deskewed_cloud;
        output.pose_valid = predicted_state_->initialized;
        last_dense_map_add_result_ = dense_map_.addKeyframe(dense_map_cloud);
        return output;
    }
    last_dense_map_add_result_ = MapAccumulator::AddResult{};
    return std::nullopt;
}

void Core::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_input_packet_ = InputPacket{};
    last_scan_timing_ = ScanTiming{};
    last_imu_propagation_.reset();
    predicted_state_.reset();
    local_map_.clear();
    dense_map_.clear();
    last_dense_map_add_result_ = MapAccumulator::AddResult{};
    last_alignment_stats_ = LioLocalMap::AlignmentStats{};
}

CloudAdapterOptions Core::cloudOptions() const {
    return makeCloudAdapterOptions(makeSettings(config_));
}

const char* coreStatus() {
    return "dlio_core input boundary ready: odometry algorithm extraction pending";
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
