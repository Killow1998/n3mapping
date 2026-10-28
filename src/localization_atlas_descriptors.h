#pragma once
#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/relocalization_place_catalog.h"

namespace n3mapping {
class LocalizationAtlasData;
void encodeAtlasDescriptors(const Config &config,
                            const std::vector<Keyframe::Ptr> &keyframes,
                            LocalizationAtlasData &data);
bool decodeAtlasDescriptors(const Config &config,
                            const LocalizationAtlasData &data,
                            RHPDPlaceCache &frames,
                            RHPDPlaceCache &submaps,
                            RHPDHeadingCache &headings,
                            RelocalizationPlaceCatalog &catalog,
                            bool &catalog_ready, std::string *error);
} // namespace n3mapping
