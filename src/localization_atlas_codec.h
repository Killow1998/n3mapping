#pragma once
#include "n3mapping/free_space_grid.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
class LocalizationAtlasData;
namespace atlas_codec {
bool writeAtlas(std::ostream& output, const LocalizationAtlasData& metadata,
                const PointCloudMatcher::PointCloudT& global_map,
                const PointCloudMatcher::PreparedTarget& target,
                const FreeSpaceGrid& grid, std::string* error);
}
}
