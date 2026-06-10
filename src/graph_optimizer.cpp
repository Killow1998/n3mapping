#include "n3mapping/graph_optimizer.h"

#include <stdexcept>
#include <iostream>
#include <utility>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>

namespace n3mapping {

GraphOptimizer::GraphOptimizer(const Config& config)
    : config_(config)
    , has_loop_closure_(false)
    , pending_has_loop_closure_(false)
    , needs_optimization_(false)
{
    std::string config_error;
    if (!config_.validate(&config_error)) {
        throw std::invalid_argument("Invalid N3Mapping graph optimizer config: " + config_error);
    }
    isam2_ = createISAM2();
}

// ==================== 添加因子 ====================

void GraphOptimizer::addPriorFactor(int64_t id, const Eigen::Isometry3d& pose) {
    gtsam::Key key = gtsam::Symbol('x', id);
    gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
    
    // 创建先验噪声模型
    auto noise_model = createPriorNoiseModel();
    
    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise_model));
    
    // 添加待提交初始值
    if (!hasAnyNode(id)) {
        new_values_.insert(key, gtsam_pose);
        pending_node_ids_.insert(id);
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
    
    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    
    // 如果目标节点不存在，尝试从 from 节点推导待提交初始值。
    if (!hasAnyNode(edge.to_id)) {
        // 使用里程计测量值计算初始位姿
        gtsam::Pose3 pose_from;
        if (getAnyPose(key_from, &pose_from)) {
            gtsam::Pose3 pose_to = pose_from * measurement;
            new_values_.insert(key_to, pose_to);
            pending_node_ids_.insert(edge.to_id);
        }
    }
    
    // 存储待提交边信息
    pending_edges_.push_back(edge);
    needs_optimization_ = true;
}


void GraphOptimizer::addLoopEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
    
    // 创建带鲁棒核函数的噪声模型
    auto noise_model = createRobustNoiseModel(edge.information, config_.use_robust_kernel);
    
    // 添加到待提交因子；只有优化成功后才进入 committed graph。
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
    
    // 存储待提交边信息
    EdgeInfo loop_edge = edge;
    loop_edge.type = EdgeType::LOOP;
    pending_edges_.push_back(loop_edge);
    
    pending_has_loop_closure_ = true;
    needs_optimization_ = true;
}

// ==================== 优化 ====================

void GraphOptimizer::optimize() {
    gtsam::NonlinearFactorGraph candidate_graph = graph_;
    gtsam::Values candidate_values = initial_values_;
    appendPendingTo(&candidate_graph, &candidate_values);

    if (candidate_graph.empty() || candidate_values.empty()) {
        return;
    }
    
    gtsam::Values optimized_estimate;
    if (!optimizeCandidate(candidate_graph, candidate_values, &optimized_estimate)) {
        rollbackToLastState();
        return;
    }

    if (!new_factors_.empty() || !new_values_.empty() || !pending_edges_.empty()) {
        commitPending(optimized_estimate, nullptr);
    } else {
        current_estimate_ = optimized_estimate;
        last_good_estimate_ = optimized_estimate;
        rebuildISAM2FromCommitted();
        needs_optimization_ = false;
    }
}

bool GraphOptimizer::incrementalOptimize() {
    if (new_factors_.empty() && new_values_.empty() && pending_edges_.empty() && pending_node_ids_.empty()) {
        return true;
    }
    
    try {
        auto trial_isam2 = createISAM2();

        if (!graph_.empty()) {
            const gtsam::Values& committed_values = !current_estimate_.empty() ? current_estimate_ : initial_values_;
            trial_isam2->update(graph_, committed_values);
        }

        trial_isam2->update(new_factors_, new_values_);
        
        // 如果有回环约束，多执行几次优化迭代以确保收敛
        if (has_loop_closure_ || pending_has_loop_closure_) {
            for (int i = 0; i < config_.optimization_iterations; ++i) {
                trial_isam2->update();
            }
        } else {
            // 普通情况下也执行一次额外更新以确保收敛
            trial_isam2->update();
        }
        
        // 获取当前估计
        gtsam::Values optimized_estimate = trial_isam2->calculateEstimate();
        
        // 检查优化健康度
        if (optimized_estimate.empty()) {
            std::cerr << "Optimization unhealthy, rolling back to last state" << std::endl;
            rollbackToLastState();
            return false;
        }

        commitPending(optimized_estimate, std::move(trial_isam2));
        return true;

    } catch (const std::exception& e) {
        std::cerr << "GTSAM optimization exception: " << e.what() << std::endl;
        rollbackToLastState();
        return false;
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
    for (int64_t id : pending_node_ids_) {
        gtsam::Key key = gtsam::Symbol('x', id);
        if (new_values_.exists(key)) {
            poses[id] = gtsamToEigen(new_values_.at<gtsam::Pose3>(key));
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
    } else if (new_values_.exists(key)) {
        return gtsamToEigen(new_values_.at<gtsam::Pose3>(key));
    }
    
    throw std::out_of_range("Node " + std::to_string(id) + " does not exist in graph");
}

bool GraphOptimizer::hasNode(int64_t id) const {
    return hasAnyNode(id);
}

size_t GraphOptimizer::getNumNodes() const {
    size_t count = node_ids_.size();
    for (int64_t id : pending_node_ids_) {
        if (node_ids_.find(id) == node_ids_.end()) {
            ++count;
        }
    }
    return count;
}

size_t GraphOptimizer::getNumEdges() const {
    return edges_.size() + pending_edges_.size();
}

bool GraphOptimizer::hasLoopClosure() const {
    return has_loop_closure_ || pending_has_loop_closure_;
}


// ==================== 序列化支持 ====================

bool GraphOptimizer::loadGraph(
    const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
    const std::vector<EdgeInfo>& edges) {
    GraphOptimizer temp(config_);

    if (nodes.empty()) {
        swapWith(temp);
        return true;
    }
    
    // 添加节点
    for (const auto& [id, pose] : nodes) {
        gtsam::Key key = gtsam::Symbol('x', id);
        gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
        if (temp.initial_values_.exists(key)) {
            std::cerr << "GraphOptimizer::loadGraph() duplicate node id: " << id << std::endl;
            return false;
        }
        temp.initial_values_.insert(key, gtsam_pose);
        temp.node_ids_.insert(id);
    }
    
    // 添加第一个节点的先验因子
    int64_t first_id = nodes.front().first;
    gtsam::Key first_key = gtsam::Symbol('x', first_id);
    gtsam::Pose3 first_pose = temp.initial_values_.at<gtsam::Pose3>(first_key);
    auto prior_noise = temp.createPriorNoiseModel();
    temp.graph_.add(gtsam::PriorFactor<gtsam::Pose3>(first_key, first_pose, prior_noise));
    
    // 添加边
    for (const auto& edge : edges) {
        gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
        gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
        gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
        
        gtsam::noiseModel::Base::shared_ptr noise_model;
        if (edge.type == EdgeType::LOOP) {
            noise_model = temp.createRobustNoiseModel(edge.information, config_.use_robust_kernel);
            if (!noise_model) {
                noise_model = temp.createLoopNoiseModel();
            }
            temp.has_loop_closure_ = true;
        } else {
            noise_model = createNoiseModel(edge.information);
            if (!noise_model) {
                noise_model = temp.createOdomNoiseModel();
            }
        }
        
        temp.graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise_model));
        temp.edges_.push_back(edge);
    }
    
    // 执行优化
    if (!temp.graph_.empty() && !temp.initial_values_.empty()) {
        gtsam::Values optimized_estimate;
        if (!temp.optimizeCandidate(temp.graph_, temp.initial_values_, &optimized_estimate)) {
            std::cerr << "GraphOptimizer::loadGraph() failed; existing graph left unchanged" << std::endl;
            return false;
        }
        temp.current_estimate_ = optimized_estimate;
        temp.last_good_estimate_ = optimized_estimate;
    }

    temp.rebuildISAM2FromCommitted();
    temp.clearPending();
    temp.needs_optimization_ = false;
    swapWith(temp);
    return true;
}

void GraphOptimizer::swapWith(GraphOptimizer& other) {
    if (this == &other) return;
    using std::swap;
    swap(config_, other.config_);
    swap(isam2_, other.isam2_);
    swap(graph_, other.graph_);
    swap(new_factors_, other.new_factors_);
    swap(initial_values_, other.initial_values_);
    swap(new_values_, other.new_values_);
    swap(current_estimate_, other.current_estimate_);
    swap(last_good_estimate_, other.last_good_estimate_);
    swap(edges_, other.edges_);
    swap(pending_edges_, other.pending_edges_);
    swap(node_ids_, other.node_ids_);
    swap(pending_node_ids_, other.pending_node_ids_);
    swap(has_loop_closure_, other.has_loop_closure_);
    swap(pending_has_loop_closure_, other.pending_has_loop_closure_);
    swap(needs_optimization_, other.needs_optimization_);
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
    pending_edges_.clear();
    node_ids_.clear();
    pending_node_ids_.clear();
    has_loop_closure_ = false;
    pending_has_loop_closure_ = false;
    needs_optimization_ = false;
    isam2_ = createISAM2();
}

// ==================== 错误处理 ====================

void GraphOptimizer::rollbackToLastState() {
    clearPending();
    if (!last_good_estimate_.empty()) {
        current_estimate_ = last_good_estimate_;
        rebuildISAM2FromCommitted();
        std::cerr << "Rolled back to last good optimization state" << std::endl;
    } else {
        if (!current_estimate_.empty() || !initial_values_.empty()) {
            rebuildISAM2FromCommitted();
        }
        std::cerr << "No previous state to rollback to" << std::endl;
    }
    needs_optimization_ = false;
}

bool GraphOptimizer::isOptimizationHealthy() const {
    // 简化的健康度检查
    // TODO: 可以添加更复杂的检查，如误差激增检测
    return !current_estimate_.empty();
}

// ==================== 辅助函数 ====================

std::unique_ptr<gtsam::ISAM2> GraphOptimizer::createISAM2() const {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    params.cacheLinearizedFactors = false;
    params.enableDetailedResults = false;
    params.enablePartialRelinearizationCheck = true;
    return std::make_unique<gtsam::ISAM2>(params);
}

void GraphOptimizer::clearPending() {
    new_factors_.resize(0);
    new_values_.clear();
    pending_edges_.clear();
    pending_node_ids_.clear();
    pending_has_loop_closure_ = false;
}

bool GraphOptimizer::hasAnyNode(int64_t id) const {
    return node_ids_.find(id) != node_ids_.end() ||
           pending_node_ids_.find(id) != pending_node_ids_.end();
}

bool GraphOptimizer::getAnyPose(gtsam::Key key, gtsam::Pose3* pose) const {
    if (!pose) return false;
    if (new_values_.exists(key)) {
        *pose = new_values_.at<gtsam::Pose3>(key);
        return true;
    }
    if (current_estimate_.exists(key)) {
        *pose = current_estimate_.at<gtsam::Pose3>(key);
        return true;
    }
    if (initial_values_.exists(key)) {
        *pose = initial_values_.at<gtsam::Pose3>(key);
        return true;
    }
    return false;
}

void GraphOptimizer::commitPending(const gtsam::Values& optimized_estimate,
                                   std::unique_ptr<gtsam::ISAM2> optimized_isam2) {
    for (const auto& factor : new_factors_) {
        graph_.push_back(factor);
    }
    for (int64_t id : pending_node_ids_) {
        gtsam::Key key = gtsam::Symbol('x', id);
        if (!initial_values_.exists(key) && new_values_.exists(key)) {
            initial_values_.insert(key, new_values_.at<gtsam::Pose3>(key));
        }
        node_ids_.insert(id);
    }
    edges_.insert(edges_.end(), pending_edges_.begin(), pending_edges_.end());
    has_loop_closure_ = has_loop_closure_ || pending_has_loop_closure_;
    current_estimate_ = optimized_estimate;
    last_good_estimate_ = optimized_estimate;
    clearPending();
    if (optimized_isam2) {
        isam2_ = std::move(optimized_isam2);
    } else {
        rebuildISAM2FromCommitted();
    }
    needs_optimization_ = false;
}

void GraphOptimizer::rebuildISAM2FromCommitted() {
    isam2_ = createISAM2();
    if (graph_.empty()) {
        return;
    }
    const gtsam::Values& values = !current_estimate_.empty() ? current_estimate_ : initial_values_;
    if (values.empty()) {
        return;
    }
    try {
        isam2_->update(graph_, values);
    } catch (const std::exception& e) {
        std::cerr << "GraphOptimizer::rebuildISAM2FromCommitted() failed: " << e.what() << std::endl;
        isam2_ = createISAM2();
    }
}

bool GraphOptimizer::optimizeCandidate(const gtsam::NonlinearFactorGraph& candidate_graph,
                                       const gtsam::Values& candidate_values,
                                       gtsam::Values* optimized_estimate) const {
    if (!optimized_estimate || candidate_graph.empty() || candidate_values.empty()) {
        return false;
    }
    try {
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = config_.optimization_iterations;
        params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;

        gtsam::LevenbergMarquardtOptimizer optimizer(candidate_graph, candidate_values, params);
        *optimized_estimate = optimizer.optimize();
        return !optimized_estimate->empty();
    } catch (const std::exception& e) {
        std::cerr << "GraphOptimizer::optimizeCandidate() failed: " << e.what() << std::endl;
        return false;
    }
}

void GraphOptimizer::appendPendingTo(gtsam::NonlinearFactorGraph* candidate_graph,
                                     gtsam::Values* candidate_values) const {
    if (!candidate_graph || !candidate_values) {
        return;
    }
    for (const auto& factor : new_factors_) {
        candidate_graph->push_back(factor);
    }
    for (int64_t id : pending_node_ids_) {
        gtsam::Key key = gtsam::Symbol('x', id);
        if (!candidate_values->exists(key) && new_values_.exists(key)) {
            candidate_values->insert(key, new_values_.at<gtsam::Pose3>(key));
        }
    }
}

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
