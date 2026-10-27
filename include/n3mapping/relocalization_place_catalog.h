#pragma once
#include "n3mapping/keyframe.h"
#include "n3mapping/rhpd_place.h"
#include "n3mapping/rhpd_heading.h"
namespace n3mapping {
struct Config;
// Each entry is one physical viewpoint, with its distinct observation IDs.
using RelocalizationPlaceCatalog = std::vector<std::vector<int64_t>>;
using RHPDHeadingCache = std::map<int64_t, RHPDHeadingPtr>;
RelocalizationPlaceCatalog compileRelocalizationPlaceCatalog(
    const Config &config, const std::vector<Keyframe::Ptr> &keyframes,
    const RHPDPlaceCache &frames, const RHPDPlaceCache &submaps,
    const RHPDHeadingCache &headings);
}
