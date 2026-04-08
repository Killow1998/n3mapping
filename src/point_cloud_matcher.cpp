// PointCloudMatcher: small_gicp registration — single, batch, and cloud-level alignment with multi-scale ICP + optional GICP refinement.
#include "n3mapping/point_cloud_matcher.h"

#include <glog/logging.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <limits>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace n3mapping {

PointCloudMatcher::PointCloudMatcher(const Config& config) : config_(config) {
    setting_.type = small_gicp::RegistrationSetting::PLANE_ICP;
    setting_.downsampling_resolution = config_.gicp_downsampling_resolution;
    setting_.max_correspondence_distance = config_.gicp_max_correspondence_distance;
    setting_.max_iterations = config_.gicp_max_iterations;
    setting_.translation_eps = std::max(1e-3, config_.gicp_transformation_epsilon);
    setting_.rotation_eps = std::max(0.1, config_.gicp_rotation_epsilon_deg) * M_PI / 180.0;
    setting_.num_threads = config_.num_threads;
    setting_.verbose = false;
}

PointCloudMatcher::SmallGicpCloud::Ptr PointCloudMatcher::convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud) {
    auto cloud = std::make_shared<SmallGicpCloud>();
    if (!pcl_cloud || pcl_cloud->empty()) return cloud;
    cloud->resize(pcl_cloud->size());
    for (size_t i = 0; i < pcl_cloud->size(); ++i) {
        const auto& pt = pcl_cloud->points[i];
        cloud->point(i) = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
    }
    return cloud;
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::preprocessTargetPointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution) {
    if (!cloud || cloud->empty()) return { std::make_shared<SmallGicpCloud>(), nullptr };
    auto raw = convertToSmallGicp(cloud);
    auto ds = small_gicp::voxelgrid_sampling_omp<SmallGicpCloud>(*raw, downsampling_resolution, config_.num_threads);
    if (ds->empty()) return { ds, nullptr };
    auto kdtree = std::make_shared<SmallGicpKdTree>(ds);
    small_gicp::estimate_normals_covariances_omp(*ds, kdtree->kdtree, config_.gicp_num_neighbors, config_.num_threads);
    return { ds, kdtree };
}

PointCloudMatcher::SmallGicpCloud::Ptr PointCloudMatcher::preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution) {
    if (!cloud || cloud->empty()) return std::make_shared<SmallGicpCloud>();
    auto raw = convertToSmallGicp(cloud);
    return small_gicp::voxelgrid_sampling_omp<SmallGicpCloud>(*raw, downsampling_resolution, config_.num_threads);
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::preprocessPointCloud(const PointCloudT::Ptr& cloud) {
    return preprocessTargetPointCloud(cloud, config_.gicp_downsampling_resolution);
}

MatchResult PointCloudMatcher::align(const Keyframe::Ptr& target, const Keyframe::Ptr& source, const Eigen::Isometry3d& init_guess) {
    MatchResult result;
    if (!target || !source || !target->cloud || !source->cloud || target->cloud->empty() || source->cloud->empty()) return result;
    try {
        auto [tp, tt] = preprocessTargetPointCloud(target->cloud, config_.gicp_downsampling_resolution);
        auto sp = preprocessSourcePointCloud(source->cloud, config_.gicp_downsampling_resolution);
        if (!tt || tp->size() < 10 || sp->size() < 10) return result;
        auto ls = setting_;
        ls.type = small_gicp::RegistrationSetting::GICP;
        auto rr = small_gicp::align(*tp, *sp, *tt, init_guess, ls);
        result.T_target_source = rr.T_target_source;
        result.converged = rr.converged;
        result.num_inliers = rr.num_inliers;
        result.inlier_ratio = sp->empty() ? 0.0 : static_cast<double>(rr.num_inliers) / static_cast<double>(sp->size());
        const auto& H = rr.H;
        result.information.block<3,3>(0,0)=H.block<3,3>(3,3); result.information.block<3,3>(0,3)=H.block<3,3>(3,0);
        result.information.block<3,3>(3,0)=H.block<3,3>(0,3); result.information.block<3,3>(3,3)=H.block<3,3>(0,0);
        result.fitness_score = (rr.num_inliers > 0) ? rr.error / static_cast<double>(rr.num_inliers) : std::numeric_limits<double>::max();
        const bool quality_passed =
            result.fitness_score < config_.gicp_fitness_threshold &&
            result.inlier_ratio >= config_.reloc_min_inlier_ratio;
        result.success = result.converged && quality_passed;
    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] align exception: " << e.what();
    }
    return result;
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

MatchResult PointCloudMatcher::alignCloud(const PointCloudT::Ptr& target_cloud, const PointCloudT::Ptr& source_cloud, const Eigen::Isometry3d& init_guess) {
    MatchResult result;
    if (!target_cloud || target_cloud->empty() || !source_cloud || source_cloud->empty()) return result;
    try {
        const double base_res = std::max(1e-3, config_.gicp_downsampling_resolution);
        const std::array<double, 2> resolutions = { base_res * 2.0, base_res };
        Eigen::Isometry3d T = init_guess;
        size_t coarse_inliers = 0; double coarse_inlier_ratio = 0.0;
        Eigen::Matrix<double, 6, 6> coarse_H = Eigen::Matrix<double, 6, 6>::Identity();
        bool coarse_converged = false; double coarse_fitness = std::numeric_limits<double>::max();

        for (double res : resolutions) {
            if (res < base_res * 0.99) continue;
            auto [tp, tt] = preprocessTargetPointCloud(target_cloud, res);
            auto sp = preprocessSourcePointCloud(source_cloud, res);
            if (!tt || tp->size() < 10 || sp->size() < 10) return result;
            auto ls = setting_;
            if (res > base_res * 1.01) ls.max_correspondence_distance = std::max(ls.max_correspondence_distance, res * 6.0);
            auto rr = small_gicp::align(*tp, *sp, *tt, T, ls);
            T = rr.T_target_source;
            coarse_converged = rr.converged;
            coarse_inliers = rr.num_inliers;
            coarse_inlier_ratio = sp->empty() ? 0.0 : static_cast<double>(rr.num_inliers) / static_cast<double>(sp->size());
            coarse_H = rr.H;
            coarse_fitness = (rr.num_inliers > 0) ? rr.error / static_cast<double>(rr.num_inliers) : std::numeric_limits<double>::max();
        }

        result.T_target_source = T;
        result.converged = coarse_converged;
        result.num_inliers = coarse_inliers;
        result.inlier_ratio = coarse_inlier_ratio;
        result.information.block<3,3>(0,0)=coarse_H.block<3,3>(3,3); result.information.block<3,3>(0,3)=coarse_H.block<3,3>(3,0);
        result.information.block<3,3>(3,0)=coarse_H.block<3,3>(0,3); result.information.block<3,3>(3,3)=coarse_H.block<3,3>(0,0);
        result.fitness_score = coarse_fitness;
        const bool coarse_quality_passed =
            coarse_fitness < config_.gicp_fitness_threshold &&
            coarse_inlier_ratio >= config_.reloc_min_inlier_ratio;
        result.success = result.converged && coarse_quality_passed;

        if (config_.icp_refine_use_gicp && coarse_fitness < config_.icp_refine_fitness_gate) {
            const Eigen::Isometry3d delta = init_guess.inverse() * T;
            if (delta.translation().norm() <= config_.icp_refine_delta_translation_gate &&
                Eigen::AngleAxisd(delta.rotation()).angle() <= config_.icp_refine_delta_rotation_gate) {
                auto [tr, trt] = preprocessTargetPointCloud(target_cloud, config_.icp_refine_downsampling_resolution);
                auto [sr, srt] = preprocessTargetPointCloud(source_cloud, config_.icp_refine_downsampling_resolution);
                if (trt && tr->size() >= 10 && sr->size() >= 10) {
                    auto rs = setting_;
                    rs.type = small_gicp::RegistrationSetting::GICP;
                    rs.max_iterations = config_.icp_refine_max_iterations;
                    rs.max_correspondence_distance = config_.icp_refine_max_correspondence_distance;
                    rs.downsampling_resolution = config_.icp_refine_downsampling_resolution;
                    auto rr = small_gicp::align(*tr, *sr, *trt, T, rs);
                    double rf = (rr.num_inliers > 0) ? rr.error / static_cast<double>(rr.num_inliers) : std::numeric_limits<double>::max();
                    if (rr.converged && rf < config_.gicp_fitness_threshold) {
                        result.T_target_source = rr.T_target_source;
                        result.converged = true; result.num_inliers = rr.num_inliers;
                        result.inlier_ratio = sr->empty() ? 0.0 : static_cast<double>(rr.num_inliers) / static_cast<double>(sr->size());
                        result.information.block<3,3>(0,0)=rr.H.block<3,3>(3,3); result.information.block<3,3>(0,3)=rr.H.block<3,3>(3,0);
                        result.information.block<3,3>(3,0)=rr.H.block<3,3>(0,3); result.information.block<3,3>(3,3)=rr.H.block<3,3>(0,0);
                        result.fitness_score = rf; result.success = true;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] alignCloud exception: " << e.what();
    }
    return result;
}

} // namespace n3mapping
