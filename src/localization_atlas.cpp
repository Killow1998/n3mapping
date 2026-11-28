#include "localization_atlas_descriptors.h"
#include "localization_atlas_codec.h"
#include "mapped_target.h"
#include "spatial_point_blocks.h"

#include "n3mapping/localization_atlas.h"
#include "n3mapping/global_map_cache.h"

#include "localization_atlas.pb.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include <openssl/evp.h>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

constexpr const char* kAtlasFormatVersion = registration_detail::kMappedAtlasFormat;


using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void setError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

bool shareAtlasRefineLevel(const Config& config) {
    const double plane_resolution = std::max(1e-3, config.gicp_downsampling_resolution);
    return config.icp_refine_use_gicp &&
           config.icp_refine_downsampling_resolution <= plane_resolution &&
           std::max(1e-3, config.global_map_voxel_size) >= plane_resolution;
}

std::string freeSpaceConfig(const Config& config) {
    std::ostringstream signature;
    signature << std::setprecision(17) << "free_space_v1;"
              << config.reloc_free_space_resolution << ';'
              << config.reloc_free_space_max_ray_length << ';'
              << config.reloc_free_space_occupied_min_points;
    return signature.str();
}

std::string digestBytes(const void* data, std::size_t size) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return {};
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data, size) == 1 &&
                    EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return {};

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

}

LocalizationAtlas::LocalizationAtlas(const Config& config, PointCloudMatcher& matcher)
    : config_(config), matcher_(matcher), global_map_(pcl::make_shared<PointCloudT>()) {}

std::string LocalizationAtlas::defaultAtlasPath(const std::string& map_path) {
    return map_path + ".localization_atlas.pb";
}

std::string LocalizationAtlas::fileSha256(const std::string& path, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        setError(error, "cannot open file for SHA-256: " + path);
        return {};
    }
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
        if (context) EVP_MD_CTX_free(context);
        setError(error, "failed to initialize SHA-256");
        return {};
    }
    std::array<char, 1 << 16> buffer{};
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(context, buffer.data(), static_cast<std::size_t>(count)) != 1) {
            EVP_MD_CTX_free(context);
            setError(error, "failed to update SHA-256");
            return {};
        }
    }
    if (!input.eof()) {
        EVP_MD_CTX_free(context);
        setError(error, "failed while reading file for SHA-256");
        return {};
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &digest_size) != 1) {
        EVP_MD_CTX_free(context);
        setError(error, "failed to finalize SHA-256");
        return {};
    }
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

std::string LocalizationAtlas::configSha256(const Config& config) {
    std::ostringstream signature;
    signature << std::setprecision(17)
              << "format=" << kAtlasFormatVersion
              << ";global_map_voxel_size=" << config.global_map_voxel_size
              << ";rhpd_max_range=" << config.rhpd_max_range
              << ";gicp_max_correspondence_distance="
              << config.gicp_max_correspondence_distance
              << ";gicp_downsampling_resolution="
              << config.gicp_downsampling_resolution
              << ";gicp_num_neighbors=" << config.gicp_num_neighbors
              << ";icp_refine_use_gicp=" << config.icp_refine_use_gicp
              << ";icp_refine_downsampling_resolution="
              << config.icp_refine_downsampling_resolution;
    const std::string text = signature.str();
    return digestBytes(text.data(), text.size());
}

bool LocalizationAtlas::prepare(const std::vector<Keyframe::Ptr>& keyframes,
                                LocalizationAtlasStats* stats) {
    clear();
    LocalizationAtlasStats local_stats;
    local_stats.keyframe_count = keyframes.size();
    auto start = Clock::now();
    global_map_ = GlobalMapCache::build(keyframes, std::max(1e-3, config_.global_map_voxel_size));
    local_stats.global_map_ms = elapsedMs(start);
    if (!global_map_ || global_map_->empty()) {

        return false;
    }
    local_stats.global_point_count = global_map_->size();
    blocks_ = std::make_shared<SpatialPointBlocks>(*global_map_);
    start = Clock::now();
    prepared_target_ = matcher_.prepareTargetCloud(global_map_,
        shareAtlasRefineLevel(config_) ? PointCloudMatcher::TargetFeatures::SharedRefine
                                       : PointCloudMatcher::TargetFeatures::Eager);
    local_stats.prepare_ms = elapsedMs(start);
    if (prepared_target_.plane_levels.size() != 2 ||
        !prepared_target_.plane_levels[0].ready() || !prepared_target_.plane_levels[1].ready() ||
        (config_.icp_refine_use_gicp && !prepared_target_.refine_level.ready())) {

        return false;
    }

    config_sha256_ = configSha256(config_);
    loaded_ = true;
    if (stats) *stats = local_stats;
    return true;
}

void LocalizationAtlas::swapData(LocalizationAtlas& other) {
    using std::swap;
    swap(loaded_, other.loaded_);
    swap(global_map_, other.global_map_);
    swap(prepared_target_, other.prepared_target_);
    swap(map_sha256_, other.map_sha256_);
    swap(config_sha256_, other.config_sha256_);
    swap(frame_descriptors_, other.frame_descriptors_);
    swap(submap_descriptors_, other.submap_descriptors_);
    swap(heading_descriptors_, other.heading_descriptors_);
    swap(place_catalog_, other.place_catalog_);
    swap(place_catalog_ready_, other.place_catalog_ready_);
    swap(free_space_grid_, other.free_space_grid_);
    swap(mapping_, other.mapping_);
    swap(blocks_, other.blocks_);
}

FreeSpaceGrid LocalizationAtlas::takeFreeSpaceGrid() {
    return std::exchange(free_space_grid_, FreeSpaceGrid{});
}

bool LocalizationAtlas::compileAndSave(
    const std::string& map_path, const std::vector<Keyframe::Ptr>& keyframes,
    const std::string& atlas_path, bool overwrite,
    LocalizationAtlasStats* stats, std::string* error) {
    clear();
    if (keyframes.empty()) {
        setError(error, "cannot compile atlas without keyframes");
        return false;
    }
    if (atlas_path.empty()) {
        setError(error, "atlas output path is empty");
        return false;
    }
    std::error_code ec;
    if (!overwrite && std::filesystem::exists(atlas_path, ec)) {
        setError(error, "atlas output already exists");
        return false;
    }

    LocalizationAtlasStats local_stats;
    auto map_sha256 = fileSha256(map_path, error);
    if (map_sha256.empty()) return false;
    if (!prepare(keyframes, &local_stats)) {
        setError(error, "failed to prepare global atlas target");
        return false;
    }
    loaded_ = false;
    if (config_sha256_.empty()) {
        setError(error, "failed to hash atlas config");
        return false;
    }
    map_sha256_ = std::move(map_sha256);
    LocalizationAtlasData data;
    data.set_format_version(kAtlasFormatVersion);
    data.set_map_sha256(map_sha256_);
    data.set_config_sha256(config_sha256_);
    data.set_keyframe_count(keyframes.size());
    data.set_global_point_count(global_map_->size());
    for (const auto& level : prepared_target_.plane_levels)
        local_stats.prepared_point_count += level.size();
    if (prepared_target_.sharesRefineLevel()) {
        data.set_shared_refine_resolution(prepared_target_.refine_level.resolution);
    } else if (prepared_target_.has_refine_level) {
        local_stats.prepared_point_count += prepared_target_.refine_level.size();
    }

    const auto descriptor_start = Clock::now();
    encodeAtlasDescriptors(config_, keyframes, data);
    if (!decodeAtlasDescriptors(config_, data, frame_descriptors_, submap_descriptors_,
                               heading_descriptors_, place_catalog_, place_catalog_ready_, error))
        return false;
    local_stats.descriptor_prepare_ms = elapsedMs(descriptor_start);

    if (config_.reloc_free_space_enable && config_.reloc_free_space_map_pcd.empty()) {
        const auto grid_start = Clock::now();
        data.set_free_space_config(freeSpaceConfig(config_));
        std::vector<Eigen::Vector3d> origins;
        origins.reserve(keyframes.size());
        for (const auto& frame : keyframes)
            if (frame && frame->isValid()) origins.push_back(frame->pose_optimized.translation());
        free_space_grid_.build(*global_map_, origins, config_.reloc_free_space_resolution,
            config_.reloc_free_space_max_ray_length, config_.reloc_free_space_occupied_min_points);
        local_stats.free_space_prepare_ms = elapsedMs(grid_start);
    }

    const auto start = Clock::now();
    const std::filesystem::path output(atlas_path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path(), ec);
        if (ec) {
            setError(error, "failed to create atlas output directory: " + ec.message());
            return false;
        }
    }
    const std::filesystem::path temporary = output.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        const bool written = atlas_codec::writeAtlas(
            stream, data, *global_map_, prepared_target_, free_space_grid_, error);
        stream.close();
        if (!written || !stream) {
            std::filesystem::remove(temporary, ec);
            if (written) setError(error, "failed to close atlas sidecar");
            return false;
        }
    }
    local_stats.serialize_ms = elapsedMs(start);

    prepared_target_ = {};
    data.Clear();
    const auto mapped_started = Clock::now();
    if (!registration_detail::loadMappedAtlas(temporary.string(), data,
            prepared_target_, mapping_, error)) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    std::filesystem::rename(temporary, output, ec);
    if (ec) {
        setError(error, "failed to publish atlas sidecar: " + ec.message());
        std::filesystem::remove(temporary, ec);
        return false;
    }
    local_stats.sidecar_bytes = mapping_->size();
    mapping_->releasePages();
    local_stats.paged_geometry = true;
    local_stats.mapped_load_ms = elapsedMs(mapped_started);
    loaded_ = true;
    if (stats) *stats = local_stats;
    return true;
}

bool LocalizationAtlas::load(const std::string& map_path,
                             const std::string& atlas_path,
                             LocalizationAtlasStats* stats,
                             std::string* error,
                             bool require_current_cache) {
    clear();
    LocalizationAtlasStats local_stats;
    auto start = Clock::now();
    LocalizationAtlasData data;
    if (!registration_detail::loadMappedAtlas(atlas_path, data,
            prepared_target_, mapping_, error)) return false;
    local_stats.paged_geometry = true;
    local_stats.mapped_load_ms = elapsedMs(start);
    map_sha256_ = fileSha256(map_path, error);
    if (map_sha256_.empty()) return false;
    config_sha256_ = configSha256(config_);
    if (data.map_sha256() != map_sha256_) {
        setError(error, "atlas map SHA-256 mismatch");
        return false;
    }
    if (data.config_sha256() != config_sha256_) {
        setError(error, "atlas config SHA-256 mismatch");
        return false;
    }
    const double base_res = std::max(1e-3, config_.gicp_downsampling_resolution);
    const bool share_refine = config_.icp_refine_use_gicp &&
        (shareAtlasRefineLevel(config_) ||
         std::abs(config_.icp_refine_downsampling_resolution - base_res) <= 1e-9);
    if (require_current_cache &&
        ((share_refine && data.shared_refine_resolution() <= 0.0) ||
         !atlasDescriptorsCurrent(config_, data) ||
         (config_.reloc_free_space_enable && config_.reloc_free_space_map_pcd.empty() &&
          data.free_space_config() != freeSpaceConfig(config_)))) {
        setError(error, "atlas precomputation is outdated");
        return false;
    }
    if (config_.icp_refine_use_gicp && !prepared_target_.has_refine_level) {
        setError(error, "atlas is missing required refine level");
        return false;
    }
    for (const auto& level : prepared_target_.plane_levels)
        local_stats.prepared_point_count += level.size();
    if (prepared_target_.has_refine_level && !prepared_target_.sharesRefineLevel())
        local_stats.prepared_point_count += prepared_target_.refine_level.size();

    const auto points = mapping_->globalPoints();
    global_map_->resize(data.global_point_count());
    for (std::size_t i = 0; i < global_map_->size(); ++i) {
        float xyzi[4];
        std::memcpy(xyzi, points.data + i * sizeof(xyzi), sizeof(xyzi));
        if (!std::isfinite(xyzi[0]) || !std::isfinite(xyzi[1]) ||
            !std::isfinite(xyzi[2]) || !std::isfinite(xyzi[3])) {
            setError(error, "global atlas map contains non-finite data");
            return false;
        }
        auto& point = global_map_->points[i];
        point.x = xyzi[0];
        point.y = xyzi[1];
        point.z = xyzi[2];
        point.intensity = xyzi[3];
    }
    global_map_->width = static_cast<std::uint32_t>(global_map_->size());
    global_map_->height = 1;
    global_map_->is_dense = true;
    blocks_ = std::make_shared<SpatialPointBlocks>(*global_map_);

    const auto descriptor_start = Clock::now();
    if (!decodeAtlasDescriptors(config_, data, frame_descriptors_, submap_descriptors_, heading_descriptors_, place_catalog_, place_catalog_ready_, error)) return false;
    local_stats.descriptor_load_ms = elapsedMs(descriptor_start);
    const auto free_space = mapping_->freeSpace();
    if (config_.reloc_free_space_enable && config_.reloc_free_space_map_pcd.empty() &&
        free_space.data && data.free_space_config() == freeSpaceConfig(config_)) {
        const auto grid_start = Clock::now();
        if (!free_space_grid_.map(free_space.data, free_space.size, mapping_) ||
            free_space_grid_.resolution() != config_.reloc_free_space_resolution) {
            setError(error, "atlas free-space grid is malformed");
            return false;
        }
        local_stats.free_space_load_ms = elapsedMs(grid_start);
    }
    local_stats.load_ms = elapsedMs(start);
    local_stats.keyframe_count = static_cast<std::size_t>(data.keyframe_count());
    local_stats.global_point_count = global_map_->size();
    local_stats.sidecar_bytes = mapping_->size();
    loaded_ = true;
    mapping_->releasePages();
    if (stats) *stats = local_stats;
    return true;
}

void LocalizationAtlas::clear() {
    loaded_ = false;
    mapping_.reset();
    blocks_.reset();
    global_map_ = pcl::make_shared<PointCloudT>();
    prepared_target_ = {};
    frame_descriptors_.clear();
    submap_descriptors_.clear();
    heading_descriptors_.clear();
    place_catalog_.clear();
    place_catalog_ready_ = false;
    free_space_grid_ = FreeSpaceGrid{};
    map_sha256_.clear();
    config_sha256_.clear();
}

}
