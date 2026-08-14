// LocalizationAtlas: compile/load a map-bound global prepared target sidecar.
#include "n3mapping/localization_atlas.h"

#include "localization_atlas.pb.h"

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

constexpr const char* kAtlasFormatVersion = "n3mapping_localization_atlas_v1";
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
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 16)) {
        setError(error, "prepared target level exceeds protobuf array limits");
        return false;
    }
    output->set_resolution(input.resolution);
    output->set_num_points(input.cloud->size());
    output->mutable_points()->Reserve(static_cast<int>(input.cloud->size() * 4));
    output->mutable_normals()->Reserve(static_cast<int>(input.cloud->size() * 4));
    output->mutable_covariances()->Reserve(static_cast<int>(input.cloud->size() * 16));
    for (std::size_t i = 0; i < input.cloud->size(); ++i) {
        const auto& point = input.cloud->point(i);
        const auto& normal = input.cloud->normal(i);
        const auto& covariance = input.cloud->cov(i);
        if (!point.allFinite() || !normal.allFinite() || !covariance.allFinite()) {
            setError(error, "prepared target contains non-finite data");
            return false;
        }
        for (int axis = 0; axis < 4; ++axis) {
            output->add_points(point[axis]);
            output->add_normals(normal[axis]);
        }
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                output->add_covariances(covariance(row, column));
            }
        }
    }
    return true;
}

bool decodeLevel(const LocalizationAtlasPreparedLevel& input,
                 PointCloudMatcher::PreparedTargetLevel* output,
                 std::string* error) {
    if (!output || input.num_points() == 0 || !std::isfinite(input.resolution()) ||
        input.resolution() <= 0.0) {
        setError(error, "invalid atlas prepared level metadata");
        return false;
    }
    if (input.num_points() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 16) ||
        input.points_size() != static_cast<int>(input.num_points() * 4) ||
        input.normals_size() != static_cast<int>(input.num_points() * 4) ||
        input.covariances_size() != static_cast<int>(input.num_points() * 16)) {
        setError(error, "atlas prepared level array size mismatch");
        return false;
    }

    auto cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>();
    cloud->resize(static_cast<std::size_t>(input.num_points()));
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        for (int axis = 0; axis < 4; ++axis) {
            cloud->point(i)[axis] = input.points(static_cast<int>(i * 4 + axis));
            cloud->normal(i)[axis] = input.normals(static_cast<int>(i * 4 + axis));
        }
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                cloud->cov(i)(row, column) =
                    input.covariances(static_cast<int>(i * 16 + row * 4 + column));
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
    output->kdtree = std::make_shared<PointCloudMatcher::SmallGicpKdTree>(cloud);
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

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropGlobalMap(
    const Config& config, const PointCloudT::Ptr& global_map,
    const Eigen::Vector3d& center) {
    auto target = pcl::make_shared<PointCloudT>();
    if (!global_map || global_map->empty() || !center.allFinite()) return target;
    const double radius = std::max(1.0, config.rhpd_max_range) +
                          std::max(0.0, config.gicp_max_correspondence_distance);
    const double squared_radius = radius * radius;
    target->reserve(global_map->size());
    for (const auto& point : *global_map) {
        if (!pcl::isFinite(point)) continue;
        const Eigen::Vector3d position(point.x, point.y, point.z);
        if ((position - center).squaredNorm() <= squared_radius) {
            target->push_back(point);
        }
    }
    target->width = static_cast<std::uint32_t>(target->size());
    target->height = 1;
    target->is_dense = true;
    return target;
}

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropVisibilityTarget(
    const Eigen::Vector3d& center) const {
    return cropGlobalMap(config_, global_map_, center);
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

    auto start = Clock::now();
    global_map_ = buildGlobalMap(config_, keyframes);
    local_stats.global_map_ms = elapsedMs(start);
    if (!global_map_ || global_map_->empty()) {
        setError(error, "compiled global map is empty");
        return false;
    }
    local_stats.global_point_count = global_map_->size();
    if (global_map_->size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 4)) {
        setError(error, "global atlas map exceeds protobuf array limits");
        return false;
    }

    start = Clock::now();
    prepared_target_ = matcher_.prepareTargetCloud(global_map_);
    local_stats.prepare_ms = elapsedMs(start);
    if (prepared_target_.plane_levels.size() != 2 ||
        (config_.icp_refine_use_gicp && !prepared_target_.has_refine_level)) {
        setError(error, "failed to prepare global atlas target");
        return false;
    }

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
        setError(error, "atlas format version mismatch");
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

    const auto kdtree_start = Clock::now();
    prepared_target_.plane_levels.resize(2);
    for (int i = 0; i < data.plane_levels_size(); ++i) {
        if (!decodeLevel(data.plane_levels(i),
                         &prepared_target_.plane_levels[static_cast<std::size_t>(i)],
                         error)) {
            return false;
        }
        local_stats.prepared_point_count +=
            prepared_target_.plane_levels[static_cast<std::size_t>(i)].cloud->size();
    }
    prepared_target_.has_refine_level = data.refine_level_present();
    if (data.refine_level_present()) {
        if (!decodeLevel(data.refine_level(), &prepared_target_.refine_level, error)) {
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
