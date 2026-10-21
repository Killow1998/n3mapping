// LocalizationAtlas: compile/load a map-bound global prepared target sidecar.
#include "n3mapping/localization_atlas.h"

#include "localization_atlas.pb.h"
#include "compact_kdtree_builder.h"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

#include <openssl/evp.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

constexpr const char* kAtlasFormatVersion = "n3mapping_localization_atlas_v2";
constexpr std::uintmax_t kMaxAtlasBytes = std::uintmax_t{1} << 30;

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void setError(std::string* error, const std::string& message) {
    if (error) *error = message;
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

bool encodeLevel(const PointCloudMatcher::PreparedTargetLevel& input,
                 LocalizationAtlasPreparedLevel* output,
                 std::string* error) {
    if (!output || !input.cloud || !input.kdtree || input.cloud->empty()) {
        setError(error, "invalid prepared target level");
        return false;
    }
    if (input.cloud->size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 9)) {
        setError(error, "prepared target level exceeds protobuf array limits");
        return false;
    }
    output->set_resolution(input.resolution);
    output->set_num_points(input.cloud->size());
    output->mutable_points()->Reserve(static_cast<int>(input.cloud->size() * 3));
    output->mutable_normals()->Reserve(static_cast<int>(input.cloud->size() * 3));
    output->mutable_covariances()->Reserve(static_cast<int>(input.cloud->size() * 9));
    for (std::size_t i = 0; i < input.cloud->size(); ++i) {
        const auto& point = input.cloud->point(i);
        const auto& normal = input.cloud->normal(i);
        const auto& covariance = input.cloud->cov(i);
        if (!point.allFinite() || !normal.allFinite() || !covariance.allFinite()) {
            setError(error, "prepared target contains non-finite data");
            return false;
        }
        // small_gicp uses homogeneous storage. Only the spatial components
        // carry information; reject unexpected padding instead of discarding it.
        if (point[3] != 1.0 || normal[3] != 0.0 ||
            !covariance.row(3).isZero(0.0) || !covariance.col(3).isZero(0.0)) {
            setError(error, "prepared target has non-canonical homogeneous padding");
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            output->add_points(point[axis]);
            output->add_normals(normal[axis]);
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                output->add_covariances(covariance(row, column));
            }
        }
    }
    return true;
}

bool decodeLevel(LocalizationAtlasPreparedLevel& input,
                 PointCloudMatcher::PreparedTargetLevel* output,
                 std::string* error) {
    if (!output || input.num_points() == 0 || !std::isfinite(input.resolution()) ||
        input.resolution() <= 0.0) {
        setError(error, "invalid atlas prepared level metadata");
        return false;
    }
    if (input.num_points() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 9) ||
        input.points_size() != static_cast<int>(input.num_points() * 3) ||
        input.normals_size() != static_cast<int>(input.num_points() * 3) ||
        input.covariances_size() != static_cast<int>(input.num_points() * 9)) {
        setError(error, "atlas prepared level array size mismatch");
        return false;
    }

    auto cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>();
    cloud->resize(static_cast<std::size_t>(input.num_points()));
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        cloud->point(i)[3] = 1.0;
        cloud->normal(i)[3] = 0.0;
        cloud->cov(i).setZero();
        for (int axis = 0; axis < 3; ++axis) {
            cloud->point(i)[axis] = input.points(static_cast<int>(i * 3 + axis));
            cloud->normal(i)[axis] = input.normals(static_cast<int>(i * 3 + axis));
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                cloud->cov(i)(row, column) =
                    input.covariances(static_cast<int>(i * 9 + row * 3 + column));
            }
        }
        if (!cloud->point(i).allFinite() || !cloud->normal(i).allFinite() ||
            !cloud->cov(i).allFinite()) {
            setError(error, "atlas prepared level contains non-finite data");
            return false;
        }
    }
    output->resolution = input.resolution();
    output->cloud = cloud;
    // The validated arrays now belong to cloud. Release the serialized copy
    // before building the tree or decoding the next level; Clear retains it.
    LocalizationAtlasPreparedLevel{}.Swap(&input);
    output->kdtree = std::make_shared<PointCloudMatcher::SmallGicpKdTree>(
        cloud, registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilder>(*cloud, true));
    return true;
}

}  // namespace

LocalizationAtlas::LocalizationAtlas(const Config& config, PointCloudMatcher& matcher)
    : config_(config), matcher_(matcher), global_map_(pcl::make_shared<PointCloudT>()) {}

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::buildGlobalMap(
    const Config& config, const std::vector<Keyframe::Ptr>& keyframes) {
    auto global_map = pcl::make_shared<PointCloudT>();
    for (const auto& keyframe : keyframes) {
        if (!keyframe || !keyframe->cloud || keyframe->cloud->empty()) continue;
        PointCloudT transformed;
        pcl::transformPointCloud(*keyframe->cloud, transformed,
                                 keyframe->pose_optimized.matrix().cast<float>());
        *global_map += transformed;
    }

    PointCloudT::Ptr downsampled;
    const double voxel_size = std::max(1e-3, config.global_map_voxel_size);
    if (!global_map->empty() &&
        safeVoxelGridFilter<PointT>(global_map, voxel_size, &downsampled) &&
        downsampled) {
        return downsampled;
    }
    return global_map;
}

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
    global_map_ = buildGlobalMap(config_, keyframes);
    local_stats.global_map_ms = elapsedMs(start);
    if (!global_map_ || global_map_->empty()) {

        return false;
    }
    local_stats.global_point_count = global_map_->size();
    if (global_map_->size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 4)) {

        return false;
    }

    start = Clock::now();
    prepared_target_ = matcher_.prepareTargetCloud(global_map_);
    local_stats.prepare_ms = elapsedMs(start);
    if (prepared_target_.plane_levels.size() != 2 ||
        (config_.icp_refine_use_gicp && !prepared_target_.has_refine_level)) {

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
    local_stats.keyframe_count = keyframes.size();
    map_sha256_ = fileSha256(map_path, error);
    if (map_sha256_.empty()) return false;
    config_sha256_ = configSha256(config_);
    if (config_sha256_.empty()) {
        setError(error, "failed to hash atlas config");
        return false;
    }

    if (!prepare(keyframes, &local_stats)) {
        setError(error, "failed to prepare global atlas target");
        return false;
    }
    loaded_ = false;
    map_sha256_ = fileSha256(map_path, error);
    if (map_sha256_.empty()) return false;
    auto start = Clock::now();

    LocalizationAtlasData data;
    data.set_format_version(kAtlasFormatVersion);
    data.set_map_sha256(map_sha256_);
    data.set_config_sha256(config_sha256_);
    data.set_keyframe_count(keyframes.size());
    data.set_global_point_count(global_map_->size());
    data.mutable_global_points()->Reserve(static_cast<int>(global_map_->size() * 4));
    for (const auto& point : *global_map_) {
        if (!pcl::isFinite(point) || !std::isfinite(point.intensity)) {
            setError(error, "global atlas map contains non-finite data");
            return false;
        }
        data.add_global_points(point.x);
        data.add_global_points(point.y);
        data.add_global_points(point.z);
        data.add_global_points(point.intensity);
    }
    for (const auto& level : prepared_target_.plane_levels) {
        if (!encodeLevel(level, data.add_plane_levels(), error)) return false;
        local_stats.prepared_point_count += level.cloud->size();
    }
    data.set_refine_level_present(prepared_target_.has_refine_level);
    if (prepared_target_.has_refine_level) {
        if (!encodeLevel(prepared_target_.refine_level,
                         data.mutable_refine_level(), error)) {
            return false;
        }
        local_stats.prepared_point_count += prepared_target_.refine_level.cloud->size();
    }

    // Loading deliberately rejects sidecars larger than 1 GiB. Refuse the
    // exact serialized message before opening a temporary file, so compilation
    // cannot publish an artifact that the same binary is guaranteed to reject.
    const auto serialized_bytes =
        static_cast<std::uintmax_t>(data.ByteSizeLong());
    if (serialized_bytes == 0 || serialized_bytes > kMaxAtlasBytes) {
        setError(error,
                 "compiled atlas exceeds the 1 GiB load limit; use a smaller "
                 "or more strongly downsampled map");
        return false;
    }

    start = Clock::now();
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
        if (!stream.is_open() || !data.SerializeToOstream(&stream) || !stream.good()) {
            stream.close();
            std::filesystem::remove(temporary, ec);
            setError(error, "failed to serialize atlas sidecar");
            return false;
        }
    }
    std::filesystem::rename(temporary, output, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        setError(error, "failed to publish atlas sidecar: " + ec.message());
        return false;
    }
    local_stats.serialize_ms = elapsedMs(start);
    local_stats.sidecar_bytes = std::filesystem::file_size(output, ec);
    loaded_ = true;
    if (stats) *stats = local_stats;
    return true;
}

bool LocalizationAtlas::load(const std::string& map_path,
                             const std::string& atlas_path,
                             LocalizationAtlasStats* stats,
                             std::string* error) {
    clear();
    LocalizationAtlasStats local_stats;
    auto start = Clock::now();
    LocalizationAtlasData data;
    std::error_code file_error;
    const auto atlas_bytes = std::filesystem::file_size(atlas_path, file_error);
    if (file_error || atlas_bytes == 0 || atlas_bytes > kMaxAtlasBytes) {
        setError(error, "atlas sidecar size is invalid or exceeds 1 GiB");
        return false;
    }
    std::ifstream stream(atlas_path, std::ios::binary);
    if (!stream.is_open()) {
        setError(error, "failed to open atlas sidecar");
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_stream(&stream);
    google::protobuf::io::CodedInputStream coded_stream(&zero_copy_stream);
    coded_stream.SetTotalBytesLimit(static_cast<int>(kMaxAtlasBytes));
    if (!data.ParseFromCodedStream(&coded_stream) ||
        !coded_stream.ConsumedEntireMessage()) {
        setError(error, "failed to parse atlas sidecar");
        return false;
    }
    if (data.format_version() != kAtlasFormatVersion) {
        setError(error, "atlas format version mismatch; rebuild from the source map "
                        "with n3mapping_localization_atlas_compile");
        return false;
    }
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
    if (data.global_point_count() == 0 ||
        data.global_point_count() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 4) ||
        data.global_points_size() != static_cast<int>(data.global_point_count() * 4) ||
        data.plane_levels_size() != 2) {
        setError(error, "atlas global or level metadata is malformed");
        return false;
    }

    global_map_ = pcl::make_shared<PointCloudT>();
    global_map_->resize(static_cast<std::size_t>(data.global_point_count()));
    for (std::size_t i = 0; i < global_map_->size(); ++i) {
        auto& point = global_map_->points[i];
        point.x = data.global_points(static_cast<int>(i * 4));
        point.y = data.global_points(static_cast<int>(i * 4 + 1));
        point.z = data.global_points(static_cast<int>(i * 4 + 2));
        point.intensity = data.global_points(static_cast<int>(i * 4 + 3));
        if (!pcl::isFinite(point) || !std::isfinite(point.intensity)) {
            setError(error, "atlas global map contains non-finite data");
            return false;
        }
    }
    global_map_->width = static_cast<std::uint32_t>(global_map_->size());
    global_map_->height = 1;
    global_map_->is_dense = true;

    // All global points have been validated and copied to their final cloud.
    google::protobuf::RepeatedField<float>{}.Swap(data.mutable_global_points());

    const auto kdtree_start = Clock::now();
    prepared_target_.plane_levels.resize(2);
    for (int i = 0; i < data.plane_levels_size(); ++i) {
        if (!decodeLevel(*data.mutable_plane_levels(i),
                         &prepared_target_.plane_levels[static_cast<std::size_t>(i)],
                         error)) {
            return false;
        }
        local_stats.prepared_point_count +=
            prepared_target_.plane_levels[static_cast<std::size_t>(i)].cloud->size();
    }
    prepared_target_.has_refine_level = data.refine_level_present();
    if (data.refine_level_present()) {
        if (!decodeLevel(*data.mutable_refine_level(), &prepared_target_.refine_level, error)) {
            return false;
        }
        local_stats.prepared_point_count += prepared_target_.refine_level.cloud->size();
    } else if (config_.icp_refine_use_gicp) {
        setError(error, "atlas is missing required refine level");
        return false;
    }
    local_stats.kdtree_ms = elapsedMs(kdtree_start);
    local_stats.load_ms = elapsedMs(start);
    local_stats.keyframe_count = static_cast<std::size_t>(data.keyframe_count());
    local_stats.global_point_count = global_map_->size();
    std::error_code ec;
    local_stats.sidecar_bytes = std::filesystem::file_size(atlas_path, ec);
    loaded_ = true;
    if (stats) *stats = local_stats;
    return true;
}

void LocalizationAtlas::clear() {
    loaded_ = false;
    global_map_ = pcl::make_shared<PointCloudT>();
    prepared_target_ = {};
    map_sha256_.clear();
    config_sha256_.clear();
}

}  // namespace n3mapping
