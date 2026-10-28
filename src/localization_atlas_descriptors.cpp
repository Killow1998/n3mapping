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
void encodePlace(int64_t id, const RHPDPlace& place,
                 LocalizationAtlasFrameDescriptor& entry) {
  entry.set_keyframe_id(id);
  for (const auto& plane : place.planes) encodeCells(plane, *entry.add_planes());
  encodeCells(place.ring_height, *entry.mutable_ring_height());
  for (Eigen::Index i = 0; i < place.auxiliary.size(); ++i)
    entry.add_auxiliary(place.auxiliary(i));
}
bool decodePlace(const LocalizationAtlasFrameDescriptor& entry, RHPDPlace& place) {
  if (entry.keyframe_id() < 0 || entry.planes_size() != 3 ||
      entry.auxiliary_size() != RHPD_AUX_DIM) return false;
  for (int p = 0; p < 3; ++p)
    if (!decodeCells(entry.planes(p), RHPD_PLANE_BINS * RHPD_PLANE_BINS,
                     place.planes[p])) return false;
  if (!decodeCells(entry.ring_height(), RHPD_RINGS * RHPD_ZLAYERS,
                   place.ring_height)) return false;
  for (int i = 0; i < RHPD_AUX_DIM; ++i) place.auxiliary(i) = entry.auxiliary(i);
  return place.auxiliary.allFinite();
}
} // namespace

void encodeAtlasDescriptors(const Config &config,
                            const std::vector<Keyframe::Ptr> &keyframes,
                            LocalizationAtlasData &data) {
  if (!config.rhpd_enabled)
    return;
  RHPDescriptor descriptor(rhpdParamsFromConfig(config));
  data.set_frame_descriptor_config(descriptorConfig(config));
  data.set_loop_descriptor_config(catalogConfig(config) + " loop-descriptors-v1");
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
    auto& cached = *data.add_loop_descriptors();
    encodePlace(keyframe->id, *submaps.at(keyframe->id), *cached.mutable_submap());
    const auto& heading = *headings.at(keyframe->id);
    for (int offset : heading.offsets) cached.add_heading_offsets(offset);
    for (int ring : heading.ring_indices) cached.add_heading_rings(ring);
    for (double value : heading.values) cached.add_heading_values(value);
    const auto &place = features.place;
    if (!place.isZero()) {
      frames.emplace(keyframe->id, std::make_shared<const RHPDPlace>(place));
      encodePlace(keyframe->id, place, *data.add_frame_descriptors());
    }
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
                            RHPDPlaceCache &submaps,
                            RHPDHeadingCache &headings,
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
    auto place = std::make_shared<RHPDPlace>();
    valid = valid && decodePlace(entry, *place) &&
            decoded.emplace(entry.keyframe_id(), std::move(place)).second;
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
  if (data.loop_descriptor_config() == catalogConfig(config) + " loop-descriptors-v1") {
    const auto params = rhpdParamsFromConfig(config).heading;
    const int columns = RHPDHeadingParams::channels * params.sectors;
    bool valid = static_cast<uint64_t>(data.loop_descriptors_size()) <= data.keyframe_count();
    for (const auto& entry : data.loop_descriptors()) {
      auto submap = std::make_shared<RHPDPlace>();
      valid = valid && decodePlace(entry.submap(), *submap) &&
          entry.heading_offsets_size() == columns + 1 &&
          entry.heading_offsets(0) == 0 &&
          entry.heading_offsets(columns) == static_cast<uint32_t>(entry.heading_values_size()) &&
          entry.heading_rings_size() == entry.heading_values_size() &&
          entry.heading_values_size() <= columns * params.rings;
      if (!valid) break;
      Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(params.rows(), params.sectors);
      for (int col = 0; col < columns && valid; ++col) {
        const auto begin = entry.heading_offsets(col), end = entry.heading_offsets(col + 1);
        valid = begin <= end && end <= static_cast<uint32_t>(entry.heading_values_size());
        for (auto i = begin; i < end && valid; ++i) {
          const auto ring = entry.heading_rings(i);
          const double value = entry.heading_values(i);
          valid = ring < static_cast<uint32_t>(params.rings) && std::isfinite(value) &&
                  (i == begin || ring > entry.heading_rings(i - 1));
          if (valid) dense((col / params.sectors) * params.rings + ring,
                           col % params.sectors) = value;
        }
      }
      if (!valid) break;
      const auto id = entry.submap().keyframe_id();
      valid = submaps.emplace(id, std::move(submap)).second &&
              headings.emplace(id, makeRHPDHeading(dense)).second;
      if (!valid) break;
    }
    if (!valid) {
      if (error) *error = "atlas loop descriptor cache is malformed";
      return false;
    }
  }
  return true;
}
} // namespace n3mapping
