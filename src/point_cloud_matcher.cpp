// Point-cloud preparation, bounded storage estimates and matcher orchestration.
#include "n3mapping/point_cloud_matcher.h"
#include "compact_kdtree_builder.h"
#include "registration_bounds.h"
#include "point_cloud_sampling.h"
#include "point_cloud_features.h"
#include "point_cloud_feature_reuse.h"
#include "lazy_target_normals.h"

#include <glog/logging.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace n3mapping {
namespace {

std::size_t saturatedAdd(std::size_t lhs, std::size_t rhs) {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    return rhs > maximum - lhs ? maximum : lhs + rhs;
}

std::size_t saturatedMultiply(std::size_t lhs, std::size_t rhs) {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (lhs == 0 || rhs == 0) return 0;
    return lhs > maximum / rhs ? maximum : lhs * rhs;
}

std::size_t preparedLevelMemoryBytes(
    const PointCloudMatcher::PreparedTargetLevel& level) {
    if (!level.cloud) return 0;
    constexpr std::size_t kTreeBytesPerPoint = sizeof(std::size_t) * 8;
    std::size_t bytes = sizeof(level) + sizeof(*level.cloud);
    bytes = saturatedAdd(bytes, saturatedMultiply(
        level.cloud->points.capacity(), sizeof(Eigen::Vector4d)));
    bytes = saturatedAdd(bytes, saturatedMultiply(
        level.cloud->normals.capacity(), sizeof(Eigen::Vector4d)));
    bytes = saturatedAdd(bytes, saturatedMultiply(
        level.cloud->covs.capacity(), sizeof(Eigen::Matrix4d)));
    if (level.search_bounds) bytes = saturatedAdd(bytes,
        saturatedAdd(sizeof(*level.search_bounds), saturatedMultiply(
            level.search_bounds->nodes.capacity(), sizeof(registration_detail::SearchBounds::Box))));
    if (level.lazy_normals) bytes = saturatedAdd(bytes, level.lazy_normals->memoryBytes());
    return saturatedAdd(bytes, saturatedMultiply(
        level.cloud->size(), kTreeBytesPerPoint));
}

bool isSmallGicpVoxelCoordSafe(const pcl::PointXYZI& pt, double leaf_size) {
    if (!std::isfinite(static_cast<double>(pt.x)) ||
        !std::isfinite(static_cast<double>(pt.y)) ||
        !std::isfinite(static_cast<double>(pt.z)) ||
        !std::isfinite(leaf_size) || leaf_size <= 0.0) {
        return false;
    }

    constexpr double kMinCoord = -1048576.0;
    constexpr double kMaxCoord = 1048575.0;
    const double inv_leaf = 1.0 / leaf_size;
    const auto coord_safe = [inv_leaf](float value) {
        const double coord = std::floor(static_cast<double>(value) * inv_leaf);
        return coord >= kMinCoord && coord <= kMaxCoord;
    };
    return coord_safe(pt.x) && coord_safe(pt.y) && coord_safe(pt.z);
}

}  // namespace

std::size_t estimatePreparedTargetMemoryBytes(
    const PointCloudMatcher::PreparedTarget& target) {
    std::size_t bytes = saturatedAdd(
        sizeof(target),
        saturatedMultiply(target.plane_levels.capacity(),
                          sizeof(PointCloudMatcher::PreparedTargetLevel)));
    for (const auto& level : target.plane_levels) {
        bytes = saturatedAdd(bytes, preparedLevelMemoryBytes(level));
    }
    if (target.has_refine_level) {
        bytes = saturatedAdd(bytes,
                             preparedLevelMemoryBytes(target.refine_level));
    }
    return bytes;
}

PointCloudMatcher::PointCloudMatcher(const Config& config) : config_(config) {
    std::string config_error;
    if (!config_.validate(&config_error)) {
        throw std::invalid_argument("Invalid N3Mapping point cloud matcher config: " + config_error);
    }
    setting_.type = small_gicp::RegistrationSetting::PLANE_ICP;
    setting_.downsampling_resolution = config_.gicp_downsampling_resolution;
    setting_.max_correspondence_distance = config_.gicp_max_correspondence_distance;
    setting_.max_iterations = config_.gicp_max_iterations;
    setting_.translation_eps = std::max(1e-3, config_.gicp_transformation_epsilon);
    setting_.rotation_eps = std::max(0.1, config_.gicp_rotation_epsilon_deg) * M_PI / 180.0;
    setting_.num_threads = config_.num_threads;
    setting_.verbose = false;
}

PointCloudMatcher::SmallGicpCloud::Ptr PointCloudMatcher::convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud,
                                                                              double downsampling_resolution) {
    auto cloud = std::make_shared<SmallGicpCloud>();
    if (!pcl_cloud || pcl_cloud->empty()) return cloud;

    cloud->points.resize(pcl_cloud->size());
    size_t out_index = 0;
    size_t dropped_points = 0;
    for (size_t i = 0; i < pcl_cloud->size(); ++i) {
        const auto& pt = pcl_cloud->points[i];
        if (!isSmallGicpVoxelCoordSafe(pt, downsampling_resolution)) {
            ++dropped_points;
            continue;
        }
        cloud->point(out_index++) = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
    }
    cloud->points.resize(out_index);

    if (out_index == 0) {
        LOG_EVERY_N(WARNING, 100) << "[PointCloudMatcher] Dropped all " << pcl_cloud->size()
                                  << " points before small_gicp preprocessing due to non-finite or out-of-range voxel coordinates.";
        return cloud;
    }

    if (dropped_points > 0) {
        LOG_EVERY_N(WARNING, 100) << "[PointCloudMatcher] Dropped " << dropped_points
                                  << "/" << pcl_cloud->size()
                                  << " points before small_gicp preprocessing due to non-finite or out-of-range voxel coordinates.";
    }
    return cloud;
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::prepareTargetGeometry(const PointCloudT::Ptr& cloud,
                                         double downsampling_resolution) {
    if (!cloud || cloud->empty()) return { std::make_shared<SmallGicpCloud>(), nullptr };
    auto raw = convertToSmallGicp(cloud, downsampling_resolution);
    if (raw->empty()) return { raw, nullptr };
    auto ds = registration_detail::sampleCoordinates(*raw, downsampling_resolution, config_.num_threads);
    if (ds->empty()) return { ds, nullptr };
    raw.reset();
    auto kdtree = std::make_shared<SmallGicpKdTree>(
        ds, registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilderOMP>(
                *ds, false, config_.num_threads));
    return { ds, kdtree };
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::preprocessPointCloud(const PointCloudT::Ptr& cloud) {
    auto [ds, kdtree] = prepareTargetGeometry(cloud, config_.gicp_downsampling_resolution);
    if (!kdtree) return { ds, kdtree };
    registration_detail::estimateFeatures<small_gicp::NormalCovarianceSetter<SmallGicpCloud>>(
        *ds, kdtree->kdtree, config_.gicp_num_neighbors, config_.num_threads);
    return { ds, kdtree };
}

PointCloudMatcher::SmallGicpCloud::Ptr PointCloudMatcher::preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution) {
    if (!cloud || cloud->empty()) return std::make_shared<SmallGicpCloud>();
    auto raw = convertToSmallGicp(cloud, downsampling_resolution);
    if (raw->empty()) return raw;
    auto ds = registration_detail::sampleCoordinates(
        *raw, downsampling_resolution, config_.num_threads);
    return ds;
}

PointCloudMatcher::PreparedTarget PointCloudMatcher::prepareTargetCloud(
    const PointCloudT::Ptr& cloud, TargetFeatures features) {
    PreparedTarget prepared;
    if (!cloud || cloud->empty()) return prepared;

    try {
        const double base_res = std::max(1e-3, config_.gicp_downsampling_resolution);
        const std::array<double, 2> resolutions = { base_res * 2.0, base_res };
        prepared.plane_levels.reserve(resolutions.size());
        for (double resolution : resolutions) {
            auto [points, kdtree] = prepareTargetGeometry(cloud, resolution);
            prepared.plane_levels.push_back({ resolution, points, kdtree,
                kdtree ? std::make_shared<registration_detail::SearchBounds>(*kdtree) : nullptr });
        }

        if (config_.icp_refine_use_gicp) {
            const double resolution = config_.icp_refine_downsampling_resolution;
            auto [points, kdtree] = prepareTargetGeometry(cloud, resolution);
            prepared.refine_level = { resolution, points, kdtree,
                kdtree ? std::make_shared<registration_detail::SearchBounds>(*kdtree) : nullptr };
            prepared.has_refine_level = true;
        }
        if (features == TargetFeatures::OnDemand) {
            const auto defer = [&](PreparedTargetLevel& level) {
                if (level.kdtree) level.lazy_normals =
                    std::make_shared<registration_detail::LazyTargetNormals>(
                        level.cloud, level.kdtree, config_.gicp_num_neighbors);
            };
            for (auto& level : prepared.plane_levels) defer(level);
            if (prepared.has_refine_level) defer(prepared.refine_level);
            return prepared;
        }
        const auto& plane = prepared.plane_levels.back();
        auto& refine = prepared.refine_level;
        // Avoid coordinate matching when sampling densities are very different.
        // This is only a cost decision; actual reuse requires exact supports.
        const bool try_reuse = plane.kdtree && refine.kdtree &&
            plane.cloud->size() >= .95 * refine.cloud->size() &&
            refine.cloud->size() >= .95 * plane.cloud->size();
        std::vector<double> support_radius;
        for (auto& level : prepared.plane_levels) {
            if (!level.kdtree) continue;
            registration_detail::estimateFeatures<small_gicp::NormalSetter<SmallGicpCloud>>(
                *level.cloud, level.kdtree->kdtree, config_.gicp_num_neighbors,
                config_.num_threads, try_reuse && &level == &plane ? &support_radius : nullptr);
        }
        if (try_reuse) {
            const auto reused = registration_detail::reuseTargetFeatures(
                *plane.cloud, *plane.kdtree, support_radius, *refine.cloud, *refine.kdtree,
                config_.gicp_num_neighbors, config_.num_threads);
            VLOG(1) << "[PointCloudMatcher] Reused exact feature supports: "
                    << reused << "/" << refine.cloud->size();
        } else if (refine.kdtree) {
            registration_detail::estimateFeatures<small_gicp::NormalSetter<SmallGicpCloud>>(
                *refine.cloud, refine.kdtree->kdtree, config_.gicp_num_neighbors, config_.num_threads);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] prepareTargetCloud exception: " << e.what();
        return {};
    }
    return prepared;
}

PointCloudMatcher::PreparedSource PointCloudMatcher::prepareSourceCloud(
    const PointCloudT::Ptr& cloud) {
    PreparedSource prepared;
    if (!cloud || cloud->empty()) return prepared;

    try {
        const double base_res = std::max(1e-3, config_.gicp_downsampling_resolution);
        const std::array<double, 2> resolutions = { base_res * 2.0, base_res };
        prepared.plane_levels.reserve(resolutions.size());
        for (double resolution : resolutions) {
            prepared.plane_levels.push_back(
                { resolution, preprocessSourcePointCloud(cloud, resolution) });
        }

        if (config_.icp_refine_use_gicp) {
            auto points = preprocessSourcePointCloud(
                cloud, config_.icp_refine_downsampling_resolution);
            if (!points->empty()) {
                SmallGicpKdTree tree(
                    points, small_gicp::KdTreeBuilderOMP(config_.num_threads));
                registration_detail::estimateFeatures<small_gicp::CovarianceSetter<SmallGicpCloud>>(*points, tree.kdtree,
                    config_.gicp_num_neighbors, config_.num_threads);
            }
            prepared.refine_cloud = points;
            prepared.has_refine_cloud = true;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] prepareSourceCloud exception: " << e.what();
        return {};
    }
    return prepared;
}

std::vector<MatchResult> PointCloudMatcher::alignBatch(
    const std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>>& pairs,
    const std::vector<Eigen::Isometry3d>& init_guesses) {
    std::vector<MatchResult> results(pairs.size());
    std::vector<Eigen::Isometry3d> guesses = init_guesses;
    if (guesses.size() != pairs.size()) guesses.resize(pairs.size(), Eigen::Isometry3d::Identity());
#pragma omp parallel for num_threads(config_.num_threads) schedule(dynamic)
    for (size_t i = 0; i < pairs.size(); ++i)
        results[i] = align(pairs[i].first, pairs[i].second, guesses[i]);
    return results;
}

MatchResult PointCloudMatcher::alignCloud(const PointCloudT::Ptr& target_cloud,
                                          const PointCloudT::Ptr& source_cloud,
                                          const Eigen::Isometry3d& init_guess) {
    return alignCloudWithSetting(target_cloud, source_cloud, init_guess, setting_);
}

MatchResult PointCloudMatcher::alignCloud(const PointCloudT::Ptr& target_cloud,
                                          const PointCloudT::Ptr& source_cloud,
                                          const Eigen::Isometry3d& init_guess,
                                          const small_gicp::RegistrationSetting& setting) {
    return alignCloudWithSetting(target_cloud, source_cloud, init_guess, setting);
}

MatchResult PointCloudMatcher::alignCloudWithSetting(const PointCloudT::Ptr& target_cloud,
                                                     const PointCloudT::Ptr& source_cloud,
                                                     const Eigen::Isometry3d& init_guess,
                                                     const small_gicp::RegistrationSetting& setting) {
    if (!target_cloud || target_cloud->empty() || !source_cloud || source_cloud->empty()) return {};
    const auto target = prepareTargetCloud(target_cloud);
    const auto source = prepareSourceCloud(source_cloud);
    return alignPreparedWithSetting(target, source, init_guess, setting);
}

MatchResult PointCloudMatcher::alignPrepared(
    const PreparedTarget& target,
    const PreparedSource& source,
    const Eigen::Isometry3d& init_guess) {
    return alignPreparedWithSetting(target, source, init_guess, setting_);
}

MatchResult PointCloudMatcher::alignPrepared(
    const PreparedTarget& target,
    const PreparedSource& source,
    const Eigen::Isometry3d& init_guess,
    const small_gicp::RegistrationSetting& setting) {
    return alignPreparedWithSetting(target, source, init_guess, setting);
}

MatchResult PointCloudMatcher::resumePrepared(
    const PreparedTarget& target, const PreparedSource& source,
    const Eigen::Isometry3d& init_guess, const MatchMetric& metric) {
    if (!init_guess.matrix().allFinite() ||
        !std::isfinite(metric.resolution) || metric.resolution <= 0.0 ||
        !std::isfinite(metric.max_correspondence_distance) ||
        metric.max_correspondence_distance <= 0.0 ||
        (metric.type != small_gicp::RegistrationSetting::PLANE_ICP &&
         metric.type != small_gicp::RegistrationSetting::GICP))
        return {};
    auto setting = setting_;
    setting.type = metric.type;
    setting.downsampling_resolution = metric.resolution;
    setting.max_correspondence_distance = metric.max_correspondence_distance;
    if (metric.type == small_gicp::RegistrationSetting::GICP)
        setting.max_iterations = config_.icp_refine_max_iterations;
    return alignPreparedWithSetting(target, source, init_guess, setting, metric.resolution);
}

} // namespace n3mapping
