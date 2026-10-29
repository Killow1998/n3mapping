#pragma once
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
class LocalizationAtlasPreparedLevel;
namespace atlas_codec {
bool encodeLevel(const PointCloudMatcher::PreparedTargetLevel& input,
                 LocalizationAtlasPreparedLevel& output, std::string* error);
bool decodeLevel(LocalizationAtlasPreparedLevel& input,
                 PointCloudMatcher::PreparedTargetLevel& output,
                 bool with_covariances, std::string* error);
}  // namespace atlas_codec
}  // namespace n3mapping
