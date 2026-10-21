#include "n3mapping/graph_optimizer.h"
#include "n3mapping/graph_factor_noise.h"
#include "graph_pose_factors.h"

#include <gtsam/navigation/AttitudeFactor.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <tbb/global_control.h>

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

void GraphOptimizer::addFloorAttitudeFactor(int64_t id,
                                            const Eigen::Vector3d& normal_body) {
    if (!config_.floor_attitude_enable) {
        return;
    }
    if (!normal_body.allFinite() || normal_body.norm() < 1e-6) {
        return;
    }
    // Pose3AttitudeFactor(key, nZ, model, bRef) asserts that bRef, taken through
    // the pose's rotation, lands on nZ. Here bRef is the floor normal in the
    // sensor frame and nZ is world up, so the factor constrains roll and pitch
    // and nothing else.
    const gtsam::Key key = gtsam::Symbol('x', id);
    const double sigma = config_.floor_attitude_noise_deg * M_PI / 180.0;
    auto noise = gtsam::noiseModel::Isotropic::Sigma(2, sigma);
    new_factors_.add(gtsam::Pose3AttitudeFactor(
        key, gtsam::Unit3(0.0, 0.0, 1.0), noise,
        gtsam::Unit3(normal_body.normalized())));
    // Transactional: the constraint enters the pending set and only becomes
    // committed (and counted) when commitPending() runs after a successful
    // optimization. rollbackToLastState()/clearPending() drop it without
    // touching committed state.
    pending_floor_attitude_constraints_.push_back(
        FloorAttitudeConstraint{id, normal_body, sigma});
    const int total_factors = floor_attitude_factor_count_ +
                              static_cast<int>(pending_floor_attitude_constraints_.size());
    if (total_factors == 1 || total_factors % 25 == 0) {
        std::cout << "[OPTIMIZATION] floor_attitude kf=" << id
                  << " normal_body=" << normal_body.transpose()
                  << " sigma_deg=" << config_.floor_attitude_noise_deg
                  << " total_factors=" << total_factors
                  << std::endl;
    }
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

    if (!shared_planes_.empty() ||
        (config_.loop_geometry_enable && !node_ids_.empty() &&
         (has_session_anchor_ || pending_has_session_anchor_)))
        return optimizeGeometryUpdate();

    try {
        auto trial_isam2 = createISAM2();

        if (!graph_.empty()) {
            const gtsam::Values& committed_values = !current_estimate_.empty() ? current_estimate_ : initial_values_;
            trial_isam2->update(graph_, committed_values);
        }

        trial_isam2->update(new_factors_, new_values_);
        
        // Robust global constraints (loop or session anchor) receive the same
        // additional relinearization updates.
        if (hasGlobalConstraint()) {
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
    swap(shared_planes_, other.shared_planes_);
    swap(pending_plane_observations_, other.pending_plane_observations_);
    swap(pending_edges_, other.pending_edges_);
    swap(node_ids_, other.node_ids_);
    swap(pending_node_ids_, other.pending_node_ids_);
    swap(has_loop_closure_, other.has_loop_closure_);
    swap(pending_has_loop_closure_, other.pending_has_loop_closure_);
    swap(has_session_anchor_, other.has_session_anchor_);
    swap(pending_has_session_anchor_, other.pending_has_session_anchor_);
    swap(floor_attitude_factor_count_, other.floor_attitude_factor_count_);
    swap(committed_floor_attitude_constraints_,
         other.committed_floor_attitude_constraints_);
    swap(pending_floor_attitude_constraints_,
         other.pending_floor_attitude_constraints_);
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
    shared_planes_.clear();
    pending_plane_observations_.clear();
    pending_edges_.clear();
    node_ids_.clear();
    pending_node_ids_.clear();
    has_loop_closure_ = false;
    pending_has_loop_closure_ = false;
    has_session_anchor_ = false;
    pending_has_session_anchor_ = false;
    floor_attitude_factor_count_ = 0;
    committed_floor_attitude_constraints_.clear();
    pending_floor_attitude_constraints_.clear();
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
    pending_plane_observations_.clear();
    new_factors_.resize(0);
    new_values_.clear();
    pending_edges_.clear();
    pending_node_ids_.clear();
    pending_has_loop_closure_ = false;
    pending_has_session_anchor_ = false;
    pending_floor_attitude_constraints_.clear();
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

bool GraphOptimizer::ensureTargetInitialValue(
    int64_t from_id,
    int64_t to_id,
    const Eigen::Isometry3d& measurement) {
    const gtsam::Key key_from = gtsam::Symbol('x', from_id);
    const gtsam::Key key_to = gtsam::Symbol('x', to_id);
    gtsam::Pose3 pose_from;
    if (!getAnyPose(key_from, &pose_from)) {
        return false;
    }
    if (hasAnyNode(to_id)) {
        return true;
    }
    new_values_.insert(key_to, pose_from * eigenToGtsam(measurement));
    pending_node_ids_.insert(to_id);
    return true;
}

void GraphOptimizer::commitPending(const gtsam::Values& optimized_estimate,
                                   std::unique_ptr<gtsam::ISAM2> optimized_isam2) {
    for (const auto& [index, observation] : pending_plane_observations_)
        shared_planes_.at(index).observations.push_back(observation);
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
    committed_floor_attitude_constraints_.insert(
        committed_floor_attitude_constraints_.end(),
        pending_floor_attitude_constraints_.begin(),
        pending_floor_attitude_constraints_.end());
    floor_attitude_factor_count_ +=
        static_cast<int>(pending_floor_attitude_constraints_.size());
    has_loop_closure_ = has_loop_closure_ || pending_has_loop_closure_;
    has_session_anchor_ = has_session_anchor_ || pending_has_session_anchor_;
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
    // Isometry::inverse uses transpose. Repeated snapshot corrections otherwise
    // amplify even 1e-15 rotation scale error geometrically. Canonicalize at the
    // graph boundary, including float-derived registration measurements.
    const Eigen::Quaterniond rotation = Eigen::Quaterniond(R).normalized();
    return gtsam::Pose3(gtsam::Rot3(rotation), gtsam::Point3(t));
}

Eigen::Isometry3d GraphOptimizer::gtsamToEigen(const gtsam::Pose3& pose) {
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = Eigen::Quaterniond(pose.rotation().matrix())
                          .normalized().toRotationMatrix();
    result.translation() = pose.translation();
    return result;
}

gtsam::noiseModel::Gaussian::shared_ptr GraphOptimizer::createNoiseModel(
    const Eigen::Matrix<double, 6, 6>& info) {
    return makeExplicitFullGraphFactorNoise(info);
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
    return makeFullGraphFactorNoise(
        information, config_,
        GraphFactorNoiseRole::ROBUST_GLOBAL_OR_SESSION_ODOMETRY,
        use_robust).model;
}

gtsam::noiseModel::Base::shared_ptr GraphOptimizer::createXYYawLoopNoiseModel(
    const Eigen::Matrix<double, 6, 6>& information,
    bool use_robust) const {
    return makeXYYawGraphFactorNoise(
        information, config_, use_robust).model;
}

}  // namespace n3mapping
