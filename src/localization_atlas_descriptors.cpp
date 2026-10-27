#include "localization_atlas_descriptors.h"

#include "causal_submap.h"
#include "localization_atlas.pb.h"
#include "n3mapping/RHPDescriptor.h"
#include <iomanip>
#include <set>
#include <sstream>

namespace n3mapping {
namespace {
std::string descriptorConfig(const Config &config) {
  const auto p = rhpdParamsFromConfig(config);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17) << "rhpd-frame-v1 " << p.max_range << ' '
      << p.z_min << ' ' << p.z_max << ' ' << p.min_points << ' '
      << p.azimuth_bins << ' ' << p.v2_enable << ' ' << p.v3_enable << ' '
      << p.v3_visibility_bins << ' ' << p.v3_free_space_margin_m << ' '
      << p.enable_negative_space << ' ' << p.enable_pca_confidence;
  return out.str();
}
std::string catalogConfig(const Config &config) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17) << descriptorConfig(config)
      << " place-catalog-v1 " << config.rhpd_submap_kf_radius << ' '
      << config.rhpd_submap_voxel_size << ' ' << config.sc_num_rings << ' '
      << config.sc_num_sectors << ' ' << config.sc_max_radius;
  return out.str();
}
void encodeCells(const RHPDCells &cells, LocalizationAtlasCells &output) {
  for (size_t i = 0; i < cells.indices.size(); ++i) {
    output.add_indices(cells.indices[i]);
    for (int c = 0; c < 4; ++c)
      output.add_values(cells.values[i](c));
  }
}
bool decodeCells(const LocalizationAtlasCells &input, int limit,
                 RHPDCells &cells) {
  if (input.indices_size() > limit ||
      input.values_size() != 4 * input.indices_size())
    return false;
  for (int i = 0; i < input.indices_size(); ++i) {
    if (input.indices(i) >= static_cast<unsigned>(limit) ||
        (i > 0 && input.indices(i) <= input.indices(i - 1)))
      return false;
    Eigen::Vector4d value;
    for (int c = 0; c < 4; ++c)
      value(c) = input.values(4 * i + c);
    if (!value.allFinite())
      return false;
    cells.append(input.indices(i), value);
  }
  return true;
}
} // namespace

void encodeAtlasDescriptors(const Config &config,
                            const std::vector<Keyframe::Ptr> &keyframes,
                            LocalizationAtlasData &data) {
  if (!config.rhpd_enabled)
    return;
  RHPDescriptor descriptor(rhpdParamsFromConfig(config));
  data.set_frame_descriptor_config(descriptorConfig(config));
  RHPDPlaceCache frames, submaps;
  RHPDHeadingCache headings;
  std::map<int64_t, Keyframe::Ptr> snapshot;
  for (const auto &keyframe : keyframes)
    if (keyframe)
      snapshot.emplace(keyframe->id, keyframe);
  for (const auto &keyframe : keyframes) {
    if (!keyframe || !keyframe->cloud || keyframe->cloud->empty())
      continue;
    auto features = descriptor.computeWithHeading(keyframe->cloud);
    headings.emplace(keyframe->id, std::move(features.heading));
    const int radius = std::max(0, config.rhpd_submap_kf_radius);
    auto cloud = radius > 0 ? buildCausalSubmap(snapshot, keyframe->id, radius,
                                                keyframe->id, false)
                            : keyframe->cloud;
    if (!cloud || cloud->empty())
      cloud = keyframe->cloud;
    cloud = maybeVoxelizeRhpdCloud(cloud, config.rhpd_submap_voxel_size);
    submaps.emplace(keyframe->id, std::make_shared<const RHPDPlace>(
                                      descriptor.compute(cloud)));
    const auto &place = features.place;
    if (place.isZero())
      continue;
    frames.emplace(keyframe->id, std::make_shared<const RHPDPlace>(place));
    auto &entry = *data.add_frame_descriptors();
    entry.set_keyframe_id(keyframe->id);
    for (const auto &plane : place.planes)
      encodeCells(plane, *entry.add_planes());
    encodeCells(place.ring_height, *entry.mutable_ring_height());
    for (Eigen::Index i = 0; i < place.auxiliary.size(); ++i)
      entry.add_auxiliary(place.auxiliary(i));
  }
  const auto catalog = compileRelocalizationPlaceCatalog(
      config, keyframes, frames, submaps, headings);
  data.set_place_catalog_config(catalogConfig(config));
  for (const auto &members : catalog) {
    auto &entry = *data.add_places();
    for (const auto id : members)
      entry.add_view_ids(id);
  }
}

bool decodeAtlasDescriptors(const Config &config,
                            const LocalizationAtlasData &data,
                            RHPDPlaceCache &frames,
                            RelocalizationPlaceCatalog &catalog,
                            bool &catalog_ready, std::string *error) {
  // Older sidecars and changed feature settings rebuild from the source clouds.
  if (!config.rhpd_enabled ||
      data.frame_descriptor_config() != descriptorConfig(config))
    return true;
  RHPDPlaceCache decoded;
  bool valid = static_cast<uint64_t>(data.frame_descriptors_size()) <=
               data.keyframe_count();
  for (const auto &entry : data.frame_descriptors()) {
    if (!valid || entry.keyframe_id() < 0 || entry.planes_size() != 3 ||
        entry.auxiliary_size() != RHPD_AUX_DIM) {
      valid = false;
      break;
    }
    auto place = std::make_shared<RHPDPlace>();
    for (int p = 0; p < 3; ++p)
      valid &= decodeCells(entry.planes(p), RHPD_PLANE_BINS * RHPD_PLANE_BINS,
                           place->planes[p]);
    valid &= decodeCells(entry.ring_height(), RHPD_RINGS * RHPD_ZLAYERS,
                         place->ring_height);
    for (int i = 0; i < RHPD_AUX_DIM; ++i)
      place->auxiliary(i) = entry.auxiliary(i);
    valid &= place->auxiliary.allFinite();
    valid &= decoded.emplace(entry.keyframe_id(), std::move(place)).second;
    if (!valid)
      break;
  }
  if (!valid) {
    if (error)
      *error = "atlas frame descriptor cache is malformed";
    return false;
  }
  frames.swap(decoded);
  if (data.place_catalog_config() == catalogConfig(config)) {
    RelocalizationPlaceCatalog places;
    std::set<int64_t> ids;
    bool valid_catalog =
        static_cast<uint64_t>(data.places_size()) <= data.keyframe_count();
    for (const auto &entry : data.places()) {
      if (entry.view_ids().empty())
        valid_catalog = false;
      places.emplace_back();
      for (const auto id : entry.view_ids()) {
        if (id < 0 || !ids.insert(id).second)
          valid_catalog = false;
        places.back().push_back(id);
      }
      if (!valid_catalog || ids.size() > data.keyframe_count()) {
        if (error)
          *error = "atlas place catalog is malformed";
        return false;
      }
    }
    if (!valid_catalog) {
      if (error)
        *error = "atlas place catalog is malformed";
      return false;
    }
    catalog.swap(places);
    catalog_ready = true;
  }
  return true;
}
} // namespace n3mapping
