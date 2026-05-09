#include "n3mapping/lio/imu_propagator.h"

#include <cmath>

namespace n3mapping {
namespace lio {
namespace {

Eigen::Quaterniond deltaRotation(const Eigen::Vector3d& angular_velocity,
                                 double dt) {
    const double angle = angular_velocity.norm() * dt;
    if (angle < 1e-12) {
        return Eigen::Quaterniond::Identity();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, angular_velocity.normalized()));
}

}  // namespace

ImuPropagationState propagateImu(const std::vector<core::ImuSample>& samples,
                                 const ImuPropagationState& initial_state,
                                 const ImuPropagationOptions& options) {
    ImuPropagationState state = initial_state;
    if (samples.empty()) {
        state.valid = false;
        state.used_samples = 0;
        return state;
    }

    state.valid = true;
    state.used_samples = 1;
    state.stamp = samples.front().stamp;
    if (samples.size() == 1) {
        return state;
    }

    for (size_t i = 1; i < samples.size(); ++i) {
        const double dt =
            static_cast<double>(samples[i].stamp.nsec - samples[i - 1].stamp.nsec) *
            1.0e-9;
        if (dt <= 0.0 || !std::isfinite(dt)) {
            continue;
        }

        const Eigen::Vector3d accel_world =
            state.orientation * samples[i - 1].linear_accel + options.gravity_world;
        state.position += state.velocity * dt + 0.5 * accel_world * dt * dt;
        state.velocity += accel_world * dt;
        state.orientation = (state.orientation *
                             deltaRotation(samples[i - 1].angular_velocity, dt))
                                .normalized();
        state.stamp = samples[i].stamp;
        ++state.used_samples;
    }

    return state;
}

}  // namespace lio
}  // namespace n3mapping
