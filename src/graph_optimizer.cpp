#include "n3mapping/graph_optimizer.h"

#include <stdexcept>
#include <iostream>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>

namespace n3mapping {

GraphOptimizer::GraphOptimizer(const Config& config)
    : config_(config)
    , has_loop_closure_(false)
    , needs_optimization_(false)
{
    // 初始化 iSAM2 优化器
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    params.cacheLinearizedFactors = false;
    params.enableDetailedResults = false;
    params.enablePartialRelinearizationCheck = true;
    
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
}

// ==================== 添加因子 ====================

void GraphOptimizer::addPriorFactor(int64_t id, const Eigen::Isometry3d& pose) {
    gtsam::Key key = gtsam::Symbol('x', id);
    gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
    
    // 创建先验噪声模型
    auto noise_model = createPriorNoiseModel();
    
    // 添加先验因子
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise_model));
    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise_model));
    
    // 添加初始值
    if (!initial_values_.exists(key)) {
        initial_values_.insert(key, gtsam_pose);
        new_values_.insert(key, gtsam_pose);
        node_ids_.insert(id);
    }
    
    needs_optimization_ = true;
}

void GraphOptimizer::addOdometryEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
    
    // 创建噪声模型
    auto noise_model = createNoiseModel(edge.information);
    if (!noise_model) {
        noise_model = createOdomNoiseModel();
    }
    
    // 添加 Between 因子
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    
    // 如果目标节点不存在，添加初始值
    if (!initial_values_.exists(key_to)) {
        // 使用里程计测量值计算初始位姿
        gtsam::Pose3 pose_from;
        if (initial_values_.exists(key_from)) {
            pose_from = initial_values_.at<gtsam::Pose3>(key_from);
        } else if (current_estimate_.exists(key_from)) {
            pose_from = current_estimate_.at<gtsam::Pose3>(key_from);
        }
        gtsam::Pose3 pose_to = pose_from * measurement;
        initial_values_.insert(key_to, pose_to);
        new_values_.insert(key_to, pose_to);
        node_ids_.insert(edge.to_id);
    }
    
    // 存储边信息
    edges_.push_back(edge);
    needs_optimization_ = true;
}


void GraphOptimizer::addLoopEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
    
    // 创建带鲁棒核函数的噪声模型
    auto noise_model = createRobustNoiseModel(edge.information, config_.use_robust_kernel);
    
    // 添加 Between 因子
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    
    // 存储边信息
    EdgeInfo loop_edge = edge;
    loop_edge.type = EdgeType::LOOP;
    edges_.push_back(loop_edge);
    
    has_loop_closure_ = true;
    needs_optimization_ = true;
}

// ==================== 优化 ====================

void GraphOptimizer::optimize() {
    if (graph_.empty() || initial_values_.empty()) {
        return;
    }
    
    try {
        // 使用 Levenberg-Marquardt 优化器进行批量优化
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = config_.optimization_iterations;
        params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
        
        gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_values_, params);
        current_estimate_ = optimizer.optimize();
        
        needs_optimization_ = false;
    } catch (const std::exception& e) {
        std::cerr << "GraphOptimizer::optimize() failed: " << e.what() << std::endl;
    }
}

void GraphOptimizer::incrementalOptimize() {
    if (new_factors_.empty() && new_values_.empty()) {
        return;
    }
    
    try {
        // 保存当前状态用于可能的回滚
        if (!current_estimate_.empty()) {
            last_good_estimate_ = current_estimate_;
        }
        
        // 使用 iSAM2 进行增量优化
        isam2_->update(new_factors_, new_values_);
        
        // 如果有回环约束，多执行几次优化迭代以确保收敛
        if (has_loop_closure_) {
            for (int i = 0; i < config_.optimization_iterations; ++i) {
                isam2_->update();
            }
        } else {
            // 普通情况下也执行一次额外更新以确保收敛
            isam2_->update();
        }
        
        // 获取当前估计
        current_estimate_ = isam2_->calculateEstimate();
        
        // 检查优化健康度
        if (!isOptimizationHealthy()) {
            std::cerr << "Optimization unhealthy, rolling back to last state" << std::endl;
            rollbackToLastState();
            return;
        }
        
        // 清空新增因子和值
        new_factors_.resize(0);
        new_values_.clear();
        
        needs_optimization_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "GTSAM optimization exception: " << e.what() << std::endl;
        rollbackToLastState();
    }
}

// ==================== 获取结果 ====================

std::map<int64_t, Eigen::Isometry3d> GraphOptimizer::getOptimizedPoses() const {
    std::map<int64_t, Eigen::Isometry3d> poses;
    
    for (int64_t id : node_ids_) {
        gtsam::Key key = gtsam::Symbol('x', id);
        if (current_estimate_.exists(key)) {
            gtsam::Pose3 pose = current_estimate_.at<gtsam::Pose3>(key);
            poses[id] = gtsamToEigen(pose);
        } else if (initial_values_.exists(key)) {
            gtsam::Pose3 pose = initial_values_.at<gtsam::Pose3>(key);
            poses[id] = gtsamToEigen(pose);
        }
    }
    
    return poses;
}

Eigen::Isometry3d GraphOptimizer::getOptimizedPose(int64_t id) const {
    gtsam::Key key = gtsam::Symbol('x', id);
    
    if (current_estimate_.exists(key)) {
        return gtsamToEigen(current_estimate_.at<gtsam::Pose3>(key));
    } else if (initial_values_.exists(key)) {
        return gtsamToEigen(initial_values_.at<gtsam::Pose3>(key));
    }
    
    throw std::out_of_range("Node " + std::to_string(id) + " does not exist in graph");
}

bool GraphOptimizer::hasNode(int64_t id) const {
    return node_ids_.find(id) != node_ids_.end();
}

size_t GraphOptimizer::getNumNodes() const {
    return node_ids_.size();
}

size_t GraphOptimizer::getNumEdges() const {
    return edges_.size();
}

bool GraphOptimizer::hasLoopClosure() const {
    return has_loop_closure_;
}


// ==================== 序列化支持 ====================

void GraphOptimizer::loadGraph(
    const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
    const std::vector<EdgeInfo>& edges) {
    
    // 清空现有数据
    clear();
    
    // 添加节点
    for (const auto& [id, pose] : nodes) {
        gtsam::Key key = gtsam::Symbol('x', id);
        gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
        initial_values_.insert(key, gtsam_pose);
        node_ids_.insert(id);
    }
    
    // 添加第一个节点的先验因子
    if (!nodes.empty()) {
        int64_t first_id = nodes.front().first;
        gtsam::Key first_key = gtsam::Symbol('x', first_id);
        gtsam::Pose3 first_pose = initial_values_.at<gtsam::Pose3>(first_key);
        auto prior_noise = createPriorNoiseModel();
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(first_key, first_pose, prior_noise));
    }
    
    // 添加边
    for (const auto& edge : edges) {
        gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
        gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
        gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
        
        gtsam::noiseModel::Base::shared_ptr noise_model;
        if (edge.type == EdgeType::LOOP) {
            noise_model = createLoopNoiseModel();
            has_loop_closure_ = true;
        } else {
            noise_model = createOdomNoiseModel();
        }
        
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
        edges_.push_back(edge);
    }
    
    // 执行优化
    if (!graph_.empty() && !initial_values_.empty()) {
        optimize();
    }
    
    // 重新初始化 iSAM2
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
    
    // 将当前图添加到 iSAM2
    if (!current_estimate_.empty()) {
        isam2_->update(graph_, current_estimate_);
    }
}

std::vector<EdgeInfo> GraphOptimizer::getEdges() const {
    return edges_;
}

void GraphOptimizer::clear() {
    graph_.resize(0);
    new_factors_.resize(0);
    initial_values_.clear();
    new_values_.clear();
    current_estimate_.clear();
    last_good_estimate_.clear();
    edges_.clear();
    node_ids_.clear();
    has_loop_closure_ = false;
    needs_optimization_ = false;
    
    // 重新初始化 iSAM2
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
}

// ==================== 错误处理 ====================

void GraphOptimizer::rollbackToLastState() {
    if (!last_good_estimate_.empty()) {
        current_estimate_ = last_good_estimate_;
        new_factors_.resize(0);
        new_values_.clear();
        std::cerr << "Rolled back to last good optimization state" << std::endl;
    } else {
        std::cerr << "No previous state to rollback to" << std::endl;
    }
}

bool GraphOptimizer::isOptimizationHealthy() const {
    // 简化的健康度检查
    // TODO: 可以添加更复杂的检查，如误差激增检测
    return !current_estimate_.empty();
}

// ==================== 辅助函数 ====================

gtsam::Pose3 GraphOptimizer::eigenToGtsam(const Eigen::Isometry3d& pose) {
    Eigen::Matrix3d R = pose.rotation();
    Eigen::Vector3d t = pose.translation();
    return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

Eigen::Isometry3d GraphOptimizer::gtsamToEigen(const gtsam::Pose3& pose) {
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = pose.rotation().matrix();
    result.translation() = pose.translation();
    return result;
}

gtsam::noiseModel::Gaussian::shared_ptr GraphOptimizer::createNoiseModel(
    const Eigen::Matrix<double, 6, 6>& info) {
    
    // 检查信息矩阵是否有效
    if (info.isZero(1e-10)) {
        return nullptr;
    }
    
    // 信息矩阵是协方差矩阵的逆
    // GTSAM 使用 (rotation, translation) 顺序，而我们的信息矩阵是 (translation, rotation)
    // 需要重新排列
    Eigen::Matrix<double, 6, 6> gtsam_info;
    gtsam_info.block<3, 3>(0, 0) = info.block<3, 3>(3, 3);  // rotation
    gtsam_info.block<3, 3>(0, 3) = info.block<3, 3>(3, 0);  // rotation-translation
    gtsam_info.block<3, 3>(3, 0) = info.block<3, 3>(0, 3);  // translation-rotation
    gtsam_info.block<3, 3>(3, 3) = info.block<3, 3>(0, 0);  // translation
    
    try {
        return gtsam::noiseModel::Gaussian::Information(gtsam_info);
    } catch (...) {
        return nullptr;
    }
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createOdomNoiseModel() const {
    // GTSAM 使用 (rotation, translation) 顺序
    gtsam::Vector6 sigmas;
    sigmas << config_.odom_noise_rotation, config_.odom_noise_rotation, config_.odom_noise_rotation,
              config_.odom_noise_position, config_.odom_noise_position, config_.odom_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createLoopNoiseModel() const {
    // GTSAM 使用 (rotation, translation) 顺序
    gtsam::Vector6 sigmas;
    sigmas << config_.loop_noise_rotation, config_.loop_noise_rotation, config_.loop_noise_rotation,
              config_.loop_noise_position, config_.loop_noise_position, config_.loop_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createPriorNoiseModel() const {
    // GTSAM 使用 (rotation, translation) 顺序
    gtsam::Vector6 sigmas;
    sigmas << config_.prior_noise_rotation, config_.prior_noise_rotation, config_.prior_noise_rotation,
              config_.prior_noise_position, config_.prior_noise_position, config_.prior_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Base::shared_ptr GraphOptimizer::createRobustNoiseModel(
    const Eigen::Matrix<double, 6, 6>& information,
    bool use_robust) const {
    
    // 首先创建基础噪声模型
    gtsam::noiseModel::Gaussian::shared_ptr base_noise;
    
    if (!information.isZero(1e-10)) {
        // 使用提供的信息矩阵
        // GTSAM 使用 (rotation, translation) 顺序，需要重新排列
        Eigen::Matrix<double, 6, 6> gtsam_info;
        gtsam_info.block<3, 3>(0, 0) = information.block<3, 3>(3, 3);  // rotation
        gtsam_info.block<3, 3>(0, 3) = information.block<3, 3>(3, 0);  // rotation-translation
        gtsam_info.block<3, 3>(3, 0) = information.block<3, 3>(0, 3);  // translation-rotation
        gtsam_info.block<3, 3>(3, 3) = information.block<3, 3>(0, 0);  // translation
        
        try {
            base_noise = gtsam::noiseModel::Gaussian::Information(gtsam_info);
        } catch (...) {
            base_noise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << config_.loop_noise_rotation, config_.loop_noise_rotation, config_.loop_noise_rotation,
                                      config_.loop_noise_position, config_.loop_noise_position, config_.loop_noise_position).finished()
            );
        }
    } else {
        // 使用默认回环噪声模型
        gtsam::Vector6 sigmas;
        sigmas << config_.loop_noise_rotation, config_.loop_noise_rotation, config_.loop_noise_rotation,
                  config_.loop_noise_position, config_.loop_noise_position, config_.loop_noise_position;
        base_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }
    
    // 如果启用鲁棒核函数，包装基础噪声模型
    if (use_robust && config_.use_robust_kernel) {
        // GTSAM 4.0+ 使用 Robust::Create
        if (config_.robust_kernel_type == "Huber") {
            auto huber = gtsam::noiseModel::mEstimator::Huber::Create(config_.robust_kernel_delta);
            return gtsam::noiseModel::Robust::Create(huber, base_noise);
        } else if (config_.robust_kernel_type == "Cauchy") {
            auto cauchy = gtsam::noiseModel::mEstimator::Cauchy::Create(config_.robust_kernel_delta);
            return gtsam::noiseModel::Robust::Create(cauchy, base_noise);
        } else if (config_.robust_kernel_type == "DCS") {
            auto dcs = gtsam::noiseModel::mEstimator::DCS::Create(config_.robust_kernel_delta);
            return gtsam::noiseModel::Robust::Create(dcs, base_noise);
        }
    }
    
    return base_noise;
}

}  // namespace n3mapping
