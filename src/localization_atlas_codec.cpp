#include "localization_atlas_codec.h"

#include "atlas_container.h"
#include "localization_atlas.pb.h"
#include "mapped_target.h"

#include <ostream>

namespace n3mapping::atlas_codec {
namespace {
constexpr char kPadding[8] = {};
static_assert(sizeof(std::streamoff) >= 8);
static_assert(sizeof(float) == 4 && sizeof(double) == 8);

template<typename T, typename Fill>
bool writeArray(std::ostream& output, std::size_t count, Fill fill) {
  std::array<T, 1024> buffer;
  for (std::size_t first = 0; first < count;) {
    const auto size = std::min(buffer.size(), count - first);
    for (std::size_t i = 0; i < size; ++i)
      if (!fill(first + i, buffer[i])) return false;
    output.write(reinterpret_cast<const char*>(buffer.data()), size * sizeof(T));
    if (!output) return false;
    first += size;
  }
  return output.good();
}

bool writeLevel(std::ostream& output, const PointCloudMatcher::PreparedTargetLevel& level) {
  using namespace registration_detail;
  const auto& cloud = *level.cloud;
  const PointCloudMatcher::SmallGicpKdTree tree(level.cloud, small_gicp::KdTreeBuilder{});
  const auto points = cloud.size();
  const LevelHeader header{level.resolution, points, !cloud.covs.empty(),
      sizeof(AtlasSearchHeader) + std::uint64_t(tree.kdtree.nodes.size()) * sizeof(AtlasSearchNode)};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  const auto padding = align8(points * kAtlasVectorBytes) - points * kAtlasVectorBytes;
  const auto write_features = [&](bool normals) {
    const bool written = writeArray<std::array<float, 3>>(output, points, [&](auto i, auto& out) {
      const auto index = tree.kdtree.indices[i];
      const auto& v = normals ? cloud.normal(index) : cloud.point(index);
      out = {float(v.x()), float(v.y()), float(v.z())};
      return std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]);
    });
    return written && bool(output.write(kPadding, padding));
  };
  if (!write_features(false) || !write_features(true)) return false;
  if (header.explicit_covariance &&
      !writeArray<std::array<double, 9>>(output, points, [&](auto i, auto& out) {
        const auto& cov = cloud.cov(tree.kdtree.indices[i]);
        for (int row = 0; row < 3; ++row)
          for (int col = 0; col < 3; ++col)
            out[3 * row + col] = cov(row, col);
        return cov.allFinite();
      }))
    return false;
  return writeAtlasSearchIndex(output, tree);
}
}

bool writeAtlas(std::ostream& output, const LocalizationAtlasData& metadata,
                const PointCloudMatcher::PointCloudT& global_map,
                const PointCloudMatcher::PreparedTarget& target,
                const FreeSpaceGrid& grid, std::string* error) {
  const bool explicit_refine = target.has_refine_level && !target.sharesRefineLevel();
  const ContainerHeader header{kMagic, 4u + explicit_refine + grid.valid()};
  std::array<SectionEntry, kMaxSections> directory{};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(directory.data()), header.sections * sizeof(SectionEntry));

  std::size_t next = 0;
  const auto section = [&](SectionTag tag, std::uint32_t flags, auto write) {
    if (!output) return false;
    const auto end = static_cast<std::uint64_t>(output.tellp());
    const auto offset = align8(end);
    output.write(kPadding, offset - end);
    if (!write()) return false;
    directory[next++] = {tag, flags, offset, static_cast<std::uint64_t>(output.tellp()) - offset};
    return true;
  };
  const bool written =
      section(SectionTag::GlobalPoints, 0, [&] {
        return writeArray<std::array<float, 4>>(output, global_map.size(), [&](auto i, auto& out) {
          const auto& p = global_map[i];
          out = {p.x, p.y, p.z, p.intensity};
          return std::isfinite(p.x) && std::isfinite(p.y) &&
                 std::isfinite(p.z) && std::isfinite(p.intensity);
        });
      }) &&
      section(SectionTag::PlaneLevel, 0, [&] { return writeLevel(output, target.plane_levels[0]); }) &&
      section(SectionTag::PlaneLevel, 1, [&] { return writeLevel(output, target.plane_levels[1]); }) &&
      (!explicit_refine || section(SectionTag::RefineLevel, 0,
          [&] { return writeLevel(output, target.refine_level); })) &&
      (!grid.valid() || section(SectionTag::FreeSpace, 0, [&] { return grid.write(output); })) &&
      section(SectionTag::Metadata, 0, [&] { return metadata.SerializeToOstream(&output); });
  if (written) {
    output.seekp(sizeof(header));
    output.write(reinterpret_cast<const char*>(directory.data()), header.sections * sizeof(SectionEntry));
    output.flush();
    if (output) return true;
  }
  if (error) *error = "failed to write Atlas v5 sections";
  return false;
}
}
