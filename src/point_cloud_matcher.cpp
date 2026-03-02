#include "n3mapping/point_cloud_matcher.h"

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <limits>

#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace n3mapping {

PointCloudMatcher::PointCloudMatcher(const Config& config)
  : config_(config)
{
    // 配置 small_gicp 配准设置
    // 重定位/scan-to-map 更稳定：使用 point-to-plane（target 提供法向约束）
    setting_.type = small_gicp::RegistrationSetting::PLANE_ICP;
    setting_.downsampling_resolution = config_.gicp_downsampling_resolution;
    setting_.max_correspondence_distance = config_.gicp_max_correspondence_distance;
    setting_.max_iterations = config_.gicp_max_iterations;
    // small_gicp 的 translation_eps 是“增量收敛阈值”（单位：米），过小会导致永远 converged=false。
    // 保留用户参数，但做合理下限保护。
    setting_.translation_eps = std::max(1e-3, config_.gicp_transformation_epsilon);
    setting_.rotation_eps = std::max(0.1, config_.gicp_rotation_epsilon_deg) * M_PI / 180.0;
    setting_.num_threads = config_.num_threads;
    setting_.verbose = false;
}

PointCloudMatcher::SmallGicpCloud::Ptr
PointCloudMatcher::convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud)
{
    if (!pcl_cloud || pcl_cloud->empty()) {
        return std::make_shared<SmallGicpCloud>();
    }

    auto cloud = std::make_shared<SmallGicpCloud>();
    cloud->resize(pcl_cloud->size());

    for (size_t i = 0; i < pcl_cloud->size(); ++i) {
        const auto& pt = pcl_cloud->points[i];
        cloud->point(i) = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
    }

    return cloud;
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::preprocessTargetPointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution)
{
    // 检查输入有效性
    if (!cloud || cloud->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Empty point cloud for preprocessing";
        return { std::make_shared<SmallGicpCloud>(), nullptr };
    }

    // 转换为 small_gicp 点云格式
    auto raw_cloud = convertToSmallGicp(cloud);

    // 下采样 (使用 OpenMP 加速)
    auto downsampled = small_gicp::voxelgrid_sampling_omp<SmallGicpCloud>(*raw_cloud, downsampling_resolution, config_.num_threads);

    if (downsampled->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Downsampled point cloud is empty";
        return { downsampled, nullptr };
    }

    // 构建 KD 树 (使用 shared_ptr 版本)
    auto kdtree = std::make_shared<SmallGicpKdTree>(downsampled);

    // 估计法向量和协方差 (使用 OpenMP 加速)
    // PLANE_ICP 只要求 target 有法向；这里沿用 small_gicp 的 normals/covariances 估计接口
    small_gicp::estimate_normals_covariances_omp(*downsampled, kdtree->kdtree, config_.gicp_num_neighbors, config_.num_threads);

    return { downsampled, kdtree };
}

PointCloudMatcher::SmallGicpCloud::Ptr
PointCloudMatcher::preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution)
{
    if (!cloud || cloud->empty()) {
        return std::make_shared<SmallGicpCloud>();
    }

    auto raw_cloud = convertToSmallGicp(cloud);
    auto downsampled = small_gicp::voxelgrid_sampling_omp<SmallGicpCloud>(*raw_cloud, downsampling_resolution, config_.num_threads);

    return downsampled;
}

std::pair<PointCloudMatcher::SmallGicpCloud::Ptr, std::shared_ptr<PointCloudMatcher::SmallGicpKdTree>>
PointCloudMatcher::preprocessPointCloud(const PointCloudT::Ptr& cloud)
{
    return preprocessTargetPointCloud(cloud, config_.gicp_downsampling_resolution);
}

MatchResult
PointCloudMatcher::align(const Keyframe::Ptr& target, const Keyframe::Ptr& source, const Eigen::Isometry3d& init_guess)
{
    MatchResult result;
    result.success = false;

    // 检查输入有效性
    if (!target || !source) {
        LOG(ERROR) << "[PointCloudMatcher] Invalid keyframe pointers for alignment";
        return result;
    }

    if (!target->cloud || target->cloud->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Target point cloud is empty or null";
        return result;
    }

    if (!source->cloud || source->cloud->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Source point cloud is empty or null";
        return result;
    }

    try {
        auto [target_processed, target_tree] = preprocessTargetPointCloud(target->cloud, config_.gicp_downsampling_resolution);
        auto source_processed = preprocessSourcePointCloud(source->cloud, config_.gicp_downsampling_resolution);

        if (!target_tree) {
            LOG(WARNING) << "[PointCloudMatcher] Failed to build target KD tree for alignment";
            return result;
        }

        if (target_processed->size() < 10 || source_processed->size() < 10) {
            LOG(WARNING) << "Preprocessed point clouds too small: target=" << target_processed->size()
                         << ", source=" << source_processed->size();
            return result;
        }

        // 执行配准
        auto local_setting = setting_;
        local_setting.type = small_gicp::RegistrationSetting::GICP;
        auto reg_result = small_gicp::align(*target_processed, *source_processed, *target_tree, init_guess, local_setting);

        VLOG(1) << "[PointCloudMatcher] align: target=" << target_processed->size() << ", source=" << source_processed->size()
                << ", converged=" << reg_result.converged << ", iters=" << reg_result.iterations << ", inliers=" << reg_result.num_inliers
                << ", error=" << reg_result.error;

        // 填充结果
        result.T_target_source = reg_result.T_target_source;
        result.converged = reg_result.converged;
        result.num_inliers = reg_result.num_inliers;
        result.inlier_ratio =
          source_processed->empty() ? 0.0 : (static_cast<double>(reg_result.num_inliers) / static_cast<double>(source_processed->size()));
        // small_gicp Hessian is (rotation, translation) order [rx,ry,rz, tx,ty,tz]
        // Convert to our internal (translation, rotation) convention
        {
            const auto& H = reg_result.H;
            result.information.block<3, 3>(0, 0) = H.block<3, 3>(3, 3);
            result.information.block<3, 3>(0, 3) = H.block<3, 3>(3, 0);
            result.information.block<3, 3>(3, 0) = H.block<3, 3>(0, 3);
            result.information.block<3, 3>(3, 3) = H.block<3, 3>(0, 0);
        }

        // 计算配准得分 (平均误差)
        if (reg_result.num_inliers > 0) {
            result.fitness_score = reg_result.error / static_cast<double>(reg_result.num_inliers);
        } else {
            result.fitness_score = std::numeric_limits<double>::max();
        }

        // 验证配准质量：fitness 好 + inlier_ratio 足够即可
        if (result.fitness_score < config_.gicp_fitness_threshold &&
            result.inlier_ratio >= config_.reloc_min_inlier_ratio) {
            result.converged = true;
            result.success = true;
        }
        VLOG(1) << "[PointCloudMatcher] align: converged=" << result.converged
                << " fitness=" << result.fitness_score << " inlier_ratio=" << result.inlier_ratio
                << " success=" << result.success;

    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] Exception in point cloud alignment: " << e.what();
    }

    return result;
}

std::vector<MatchResult>
PointCloudMatcher::alignBatch(const std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>>& pairs,
                              const std::vector<Eigen::Isometry3d>& init_guesses)
{

    std::vector<MatchResult> results(pairs.size());

    if (pairs.empty()) {
        return results;
    }

    // 确保初始猜测数量匹配
    std::vector<Eigen::Isometry3d> guesses = init_guesses;
    if (guesses.size() != pairs.size()) {
        guesses.resize(pairs.size(), Eigen::Isometry3d::Identity());
    }

// 使用 OpenMP 并行计算
#pragma omp parallel for num_threads(config_.num_threads) schedule(dynamic)
    for (size_t i = 0; i < pairs.size(); ++i) {
        results[i] = align(pairs[i].first, pairs[i].second, guesses[i]);
    }

    return results;
}

MatchResult
PointCloudMatcher::alignCloud(const PointCloudT::Ptr& target_cloud, const PointCloudT::Ptr& source_cloud, const Eigen::Isometry3d& init_guess)
{
    MatchResult result;
    result.success = false;

    // 检查输入有效性
    if (!target_cloud || target_cloud->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Target point cloud is empty or null";
        return result;
    }

    if (!source_cloud || source_cloud->empty()) {
        LOG(WARNING) << "[PointCloudMatcher] Source point cloud is empty or null";
        return result;
    }

    try {
        // Stage 1: PLANE_ICP 粗配准（多尺度）
        const double base_res = std::max(1e-3, config_.gicp_downsampling_resolution);
        const std::array<double, 2> resolutions = { base_res * 2.0, base_res };
        Eigen::Isometry3d T = init_guess;

        size_t coarse_num_inliers = 0;
        double coarse_inlier_ratio = 0.0;
        Eigen::Matrix<double, 6, 6> coarse_H = Eigen::Matrix<double, 6, 6>::Identity();
        bool coarse_converged = false;
        double coarse_fitness = std::numeric_limits<double>::max();

        for (double res : resolutions) {
            // 避免重复层
            if (res < base_res * 0.99) continue;

            auto [target_processed, target_tree] = preprocessTargetPointCloud(target_cloud, res);
            auto source_processed = preprocessSourcePointCloud(source_cloud, res);

            if (!target_tree) {
                LOG(WARNING) << "[PointCloudMatcher] Failed to build target KD tree for alignment";
                return result;
            }

            if (target_processed->size() < 10 || source_processed->size() < 10) {
                LOG(WARNING) << "Preprocessed point clouds too small: target=" << target_processed->size()
                             << ", source=" << source_processed->size();
                return result;
            }

            auto local_setting = setting_;
            if (res > base_res * 1.01) {
                local_setting.max_correspondence_distance = std::max(local_setting.max_correspondence_distance, res * 6.0);
            }

            auto reg_result = small_gicp::align(*target_processed, *source_processed, *target_tree, T, local_setting);
            T = reg_result.T_target_source;

            coarse_converged = reg_result.converged;
            coarse_num_inliers = reg_result.num_inliers;
            coarse_inlier_ratio =
              source_processed->empty() ? 0.0 : (static_cast<double>(reg_result.num_inliers) / static_cast<double>(source_processed->size()));
            coarse_H = reg_result.H;

            if (reg_result.num_inliers > 0) {
                coarse_fitness = reg_result.error / static_cast<double>(reg_result.num_inliers);
            } else {
                coarse_fitness = std::numeric_limits<double>::max();
            }

            VLOG(2) << "[PointCloudMatcher] PLANE_ICP res=" << res
                    << " converged=" << reg_result.converged << " iters=" << reg_result.iterations
                    << " fitness=" << coarse_fitness << " inlier_ratio=" << coarse_inlier_ratio;
            // 不因粗分辨率层未收敛就 break — 精细层仍可用粗层位姿作为初值继续优化
        }

        // 先用粗配准结果填充
        result.T_target_source = T;
        result.converged = coarse_converged;
        result.num_inliers = coarse_num_inliers;
        result.inlier_ratio = coarse_inlier_ratio;
        // small_gicp Hessian is (rotation, translation) order; convert to (translation, rotation)
        {
            const auto& H = coarse_H;
            result.information.block<3, 3>(0, 0) = H.block<3, 3>(3, 3);
            result.information.block<3, 3>(0, 3) = H.block<3, 3>(3, 0);
            result.information.block<3, 3>(3, 0) = H.block<3, 3>(0, 3);
            result.information.block<3, 3>(3, 3) = H.block<3, 3>(0, 0);
        }
        result.fitness_score = coarse_fitness;
        // 即使形式上 converged=0，如果 fitness 和 inlier_ratio 都很好，也认为配准成功
        if (coarse_fitness < config_.gicp_fitness_threshold &&
            coarse_inlier_ratio >= config_.reloc_min_inlier_ratio) {
            result.converged = true; // 视为实质收敛
            result.success = true;
        }

        // Stage 2: 可选 GICP 精修（fitness 在 gate 内即可触发，不再要求 converged）
        if (config_.icp_refine_use_gicp && coarse_fitness < config_.icp_refine_fitness_gate) {
            const Eigen::Isometry3d delta = init_guess.inverse() * T;
            const double delta_translation = delta.translation().norm();
            const double delta_rotation = Eigen::AngleAxisd(delta.rotation()).angle();

            if (delta_translation <= config_.icp_refine_delta_translation_gate && delta_rotation <= config_.icp_refine_delta_rotation_gate) {

                auto [target_refine, target_refine_tree] = preprocessTargetPointCloud(target_cloud, config_.icp_refine_downsampling_resolution);
                auto [source_refine, source_refine_tree] = preprocessTargetPointCloud(source_cloud, config_.icp_refine_downsampling_resolution);

                if (target_refine_tree && target_refine->size() >= 10 && source_refine->size() >= 10) {
                    auto refine_setting = setting_;
                    refine_setting.type = small_gicp::RegistrationSetting::GICP;
                    refine_setting.max_iterations = config_.icp_refine_max_iterations;
                    refine_setting.max_correspondence_distance = config_.icp_refine_max_correspondence_distance;
                    refine_setting.downsampling_resolution = config_.icp_refine_downsampling_resolution;

                    auto refine_result = small_gicp::align(*target_refine, *source_refine, *target_refine_tree, T, refine_setting);

                    double refine_fitness = std::numeric_limits<double>::max();
                    if (refine_result.num_inliers > 0) {
                        refine_fitness = refine_result.error / static_cast<double>(refine_result.num_inliers);
                    }

                    const bool refine_ok = refine_result.converged && refine_fitness < config_.gicp_fitness_threshold;
                    if (refine_ok) {
                        result.T_target_source = refine_result.T_target_source;
                        result.converged = refine_result.converged;
                        result.num_inliers = refine_result.num_inliers;
                        result.inlier_ratio = source_refine->empty()
                                                ? 0.0
                                                : (static_cast<double>(refine_result.num_inliers) / static_cast<double>(source_refine->size()));
                        // small_gicp Hessian is (rotation, translation) order; convert to (translation, rotation)
                        {
                            const auto& H = refine_result.H;
                            result.information.block<3, 3>(0, 0) = H.block<3, 3>(3, 3);
                            result.information.block<3, 3>(0, 3) = H.block<3, 3>(3, 0);
                            result.information.block<3, 3>(3, 0) = H.block<3, 3>(0, 3);
                            result.information.block<3, 3>(3, 3) = H.block<3, 3>(0, 0);
                        }
                        result.fitness_score = refine_fitness;
                        result.success = true;
                    }
                }
            }
        }

        VLOG(1) << "[PointCloudMatcher] alignCloud: converged=" << result.converged
                << " fitness=" << result.fitness_score << " inlier_ratio=" << result.inlier_ratio
                << " success=" << result.success;

    } catch (const std::exception& e) {
        LOG(ERROR) << "[PointCloudMatcher] Exception in point cloud alignment: " << e.what();
    }

    return result;
}

} // namespace n3mapping
