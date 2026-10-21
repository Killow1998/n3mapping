#pragma once
#include "n3mapping/planar_geometry.h"
#include "n3map.pb.h"
namespace n3mapping {
void writeSharedPlanes(const std::vector<SharedPlane>& planes, N3Map& map);
bool readSharedPlanes(const N3Map& map, std::vector<SharedPlane>& planes);
}  // namespace n3mapping
