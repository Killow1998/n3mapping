#pragma once
#include "association.h"

namespace n3mapping::planar_geometry {
// Fit exactly the same signed frame and retain the original merge tolerances.
Plane planeFrame(const std::vector<Plane>& planes, const std::vector<int>& ids);
bool coherentPlane(const std::vector<Plane>& planes, const std::vector<int>& ids,
                   const Plane& frame);
std::vector<std::vector<int>> partitionPlane(
    const std::vector<Plane>& planes, const std::vector<std::vector<int>>& neighbors,
    const std::vector<int>& ids);
} // namespace n3mapping::planar_geometry
