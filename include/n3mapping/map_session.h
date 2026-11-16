#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Geometry>

namespace n3mapping {

using MapSessionId = std::uint64_t;
constexpr MapSessionId kInvalidMapSessionId =
    std::numeric_limits<MapSessionId>::max();

struct MapSessionInfo {
    MapSessionId id = 0;
    std::string source_frame_id;
    double start_timestamp = 0.0;
    Eigen::Isometry3d T_map_session_initial =
        Eigen::Isometry3d::Identity();

    bool loaded = false;
};

}
