#include "n3mapping/lio/core_state.h"

namespace n3mapping {
namespace lio {

LioCoreState stateFromImuPropagation(const ImuPropagationState& propagation) {
    LioCoreState state;
    state.stamp = propagation.stamp;
    state.T_world_lidar.translation() = propagation.position;
    state.T_world_lidar.linear() = propagation.orientation.toRotationMatrix();
    state.velocity_world = propagation.velocity;
    state.initialized = propagation.valid;
    return state;
}

core::LioFrame frameFromState(const LioCoreState& state) {
    core::LioFrame frame;
    frame.stamp = state.stamp;
    frame.T_world_lidar = state.T_world_lidar;
    frame.pose_valid = state.initialized;
    return frame;
}

}  // namespace lio
}  // namespace n3mapping
