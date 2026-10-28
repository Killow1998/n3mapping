// Immutable, map-bound prepared target sidecar for global relocalization.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/free_space_grid.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_place_catalog.h"

namespace n3mapping {

struct LocalizationAtlasStats {
    std::size_t keyframe_count = 0;
    std::size_t global_point_count = 0;
    std::size_t prepared_point_count = 0;
    std::uintmax_t sidecar_bytes = 0;
    double global_map_ms = 0.0;
    double prepare_ms = 0.0;
    double serialize_ms = 0.0;
    double load_ms = 0.0;
    double kdtree_ms = 0.0;
    double descriptor_prepare_ms = 0.0;
    double descriptor_load_ms = 0.0;
    double free_space_prepare_ms = 0.0;
    double free_space_load_ms = 0.0;
};

class LocalizationAtlas {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    LocalizationAtlas(const Config& config, PointCloudMatcher& matcher);
    // Prepare an immutable in-memory reference in the worker. The owner swaps
    // only its payload; the matcher reference never crosses ownership domains.
    bool prepare(const std::vector<Keyframe::Ptr>& keyframes,
                 LocalizationAtlasStats* stats = nullptr);
    void swapData(LocalizationAtlas& other);

    bool compileAndSave(const std::string& map_path,
                        const std::vector<Keyframe::Ptr>& keyframes,
                        const std::string& atlas_path,
                        bool overwrite,
                        LocalizationAtlasStats* stats = nullptr,
                        std::string* error = nullptr);
    bool load(const std::string& map_path,
              const std::string& atlas_path,
              LocalizationAtlasStats* stats = nullptr,
              std::string* error = nullptr,
              bool require_current_cache = false);
    void clear();
    FreeSpaceGrid takeFreeSpaceGrid();

    bool loaded() const { return loaded_; }
    const PointCloudT::Ptr& globalMap() const { return global_map_; }
    const PointCloudMatcher::PreparedTarget& preparedTarget() const {
        return prepared_target_;
    }
    const std::string& mapSha256() const { return map_sha256_; }
    const std::string& configSha256() const { return config_sha256_; }
    const RHPDPlaceCache& frameDescriptors() const { return frame_descriptors_; }
    const RHPDPlaceCache& submapDescriptors() const { return submap_descriptors_; }
    const RHPDHeadingCache& headingDescriptors() const { return heading_descriptors_; }
    const RelocalizationPlaceCatalog* placeCatalog() const {
        return place_catalog_ready_ ? &place_catalog_ : nullptr;
    }

    PointCloudT::Ptr cropVisibilityTarget(const Eigen::Vector3d& center) const;
    // Original point order and exact sphere predicate, sharing immutable map data.
    std::shared_ptr<const std::vector<std::uint32_t>> cropVisibilityIndices(
        const Eigen::Vector3d& center) const;

    static PointCloudT::Ptr buildGlobalMap(const Config& config,
                                           const std::vector<Keyframe::Ptr>& keyframes);
    static PointCloudT::Ptr cropGlobalMap(const Config& config,
                                          const PointCloudT::Ptr& global_map,
                                          const Eigen::Vector3d& center);
    static std::string defaultAtlasPath(const std::string& map_path);
    static std::string fileSha256(const std::string& path, std::string* error = nullptr);
    static std::string configSha256(const Config& config);

private:
    Config config_;
    PointCloudMatcher& matcher_;
    bool loaded_ = false;
    PointCloudT::Ptr global_map_;
    PointCloudMatcher::PreparedTarget prepared_target_;
    std::string map_sha256_;
    std::string config_sha256_;
    RHPDPlaceCache frame_descriptors_, submap_descriptors_;
    RHPDHeadingCache heading_descriptors_;
    RelocalizationPlaceCatalog place_catalog_;
    bool place_catalog_ready_ = false;
    FreeSpaceGrid free_space_grid_;
};

}  // namespace n3mapping
