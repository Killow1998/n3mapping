// GraphOptimizer: GTSAM iSAM2 factor graph, prior/odom/loop factors, incremental and batch optimization.
#include "n3mapping/graph_optimizer.h"

#include <iostream>
#include <stdexcept>
#include <gtsam/linear/NoiseModel.h>

namespace n3mapping {

GraphOptimizer::GraphOptimizer(const Config& config)
    : config_(config), has_loop_closure_(false), needs_optimization_(false) {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    params.cacheLinearizedFactors = false;
    params.enableDetailedResults = false;
    params.enablePartialRelinearizationCheck = true;
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
}

void GraphOptimizer::addPriorFactor(int64_t id, const Eigen::Isometry3d& pose) {
    gtsam::Key key = gtsam::Symbol('x', id);
    gtsam::Pose3 gtsam_pose = eigenToGtsam(pose);
    auto noise = createPriorNoiseModel();
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise));
    new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, gtsam_pose, noise));
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
    auto noise = createNoiseModel(edge.information);
    if (!noise) noise = createOdomNoiseModel();
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise));
    if (!initial_values_.exists(key_to)) {
        gtsam::Pose3 pose_from;
        if (initial_values_.exists(key_from)) pose_from = initial_values_.at<gtsam::Pose3>(key_from);
        else if (current_estimate_.exists(key_from)) pose_from = current_estimate_.at<gtsam::Pose3>(key_from);
        initial_values_.insert(key_to, pose_from * measurement);
        new_values_.insert(key_to, pose_from * measurement);
        node_ids_.insert(edge.to_id);
    }
    edges_.push_back(edge);
    needs_optimization_ = true;
}

void GraphOptimizer::addLoopEdge(const EdgeInfo& edge) {
    gtsam::Key key_from = gtsam::Symbol('x', edge.from_id);
    gtsam::Key key_to = gtsam::Symbol('x', edge.to_id);
    gtsam::Pose3 measurement = eigenToGtsam(edge.measurement);
    auto noise = createRobustNoiseModel(edge.information, config_.use_robust_kernel);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise));
    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_from, key_to, measurement, noise));
    EdgeInfo loop_edge = edge;
    loop_edge.type = EdgeType::LOOP;
    edges_.push_back(loop_edge);
    has_loop_closure_ = true;
    needs_optimization_ = true;
}

void GraphOptimizer::optimize() {
    if (graph_.empty() || initial_values_.empty()) return;
    try {
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
    if (new_factors_.empty() && new_values_.empty()) return;
    try {
        if (!current_estimate_.empty()) last_good_estimate_ = current_estimate_;
        isam2_->update(new_factors_, new_values_);
        if (has_loop_closure_) {
            for (int i = 0; i < config_.optimization_iterations; ++i) isam2_->update();
            has_loop_closure_ = false;  // reset after extra iterations
        } else {
            isam2_->update();
        }
        current_estimate_ = isam2_->calculateEstimate();
        if (!isOptimizationHealthy()) { rollbackToLastState(); return; }
        new_factors_.resize(0);
        new_values_.clear();
        needs_optimization_ = false;
    } catch (const std::exception& e) {
        std::cerr << "GTSAM optimization exception: " << e.what() << std::endl;
        rollbackToLastState();
    }
}

std::map<int64_t, Eigen::Isometry3d> GraphOptimizer::getOptimizedPoses() const {
    std::map<int64_t, Eigen::Isometry3d> poses;
    for (int64_t id : node_ids_) {
        gtsam::Key key = gtsam::Symbol('x', id);
        if (current_estimate_.exists(key)) poses[id] = gtsamToEigen(current_estimate_.at<gtsam::Pose3>(key));
        else if (initial_values_.exists(key)) poses[id] = gtsamToEigen(initial_values_.at<gtsam::Pose3>(key));
    }
    return poses;
}

Eigen::Isometry3d GraphOptimizer::getOptimizedPose(int64_t id) const {
    gtsam::Key key = gtsam::Symbol('x', id);
    if (current_estimate_.exists(key)) return gtsamToEigen(current_estimate_.at<gtsam::Pose3>(key));
    if (initial_values_.exists(key)) return gtsamToEigen(initial_values_.at<gtsam::Pose3>(key));
    throw std::out_of_range("Node " + std::to_string(id) + " does not exist");
}

bool GraphOptimizer::hasNode(int64_t id) const { return node_ids_.count(id) > 0; }
size_t GraphOptimizer::getNumNodes() const { return node_ids_.size(); }
size_t GraphOptimizer::getNumEdges() const { return edges_.size(); }
bool GraphOptimizer::hasLoopClosure() const { return has_loop_closure_; }

void GraphOptimizer::loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
                               const std::vector<EdgeInfo>& edges) {
    clear();
    for (const auto& [id, pose] : nodes) {
        gtsam::Key key = gtsam::Symbol('x', id);
        initial_values_.insert(key, eigenToGtsam(pose));
        node_ids_.insert(id);
    }
    if (!nodes.empty()) {
        gtsam::Key first_key = gtsam::Symbol('x', nodes.front().first);
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(first_key, initial_values_.at<gtsam::Pose3>(first_key), createPriorNoiseModel()));
    }
    for (const auto& edge : edges) {
        gtsam::Key kf = gtsam::Symbol('x', edge.from_id), kt = gtsam::Symbol('x', edge.to_id);
        auto noise = (edge.type == EdgeType::LOOP) ?
            static_cast<gtsam::noiseModel::Base::shared_ptr>(createLoopNoiseModel()) :
            static_cast<gtsam::noiseModel::Base::shared_ptr>(createOdomNoiseModel());
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(kf, kt, eigenToGtsam(edge.measurement), noise));
        edges_.push_back(edge);
        if (edge.type == EdgeType::LOOP) has_loop_closure_ = true;
    }
    if (!graph_.empty() && !initial_values_.empty()) optimize();
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
    if (!current_estimate_.empty()) isam2_->update(graph_, current_estimate_);
}

std::vector<EdgeInfo> GraphOptimizer::getEdges() const { return edges_; }

void GraphOptimizer::clear() {
    graph_.resize(0); new_factors_.resize(0);
    initial_values_.clear(); new_values_.clear();
    current_estimate_.clear(); last_good_estimate_.clear();
    edges_.clear(); node_ids_.clear();
    has_loop_closure_ = false; needs_optimization_ = false;
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam2_ = std::make_unique<gtsam::ISAM2>(params);
}

void GraphOptimizer::rollbackToLastState() {
    if (!last_good_estimate_.empty()) {
        current_estimate_ = last_good_estimate_;
        new_factors_.resize(0); new_values_.clear();
    }
}

bool GraphOptimizer::isOptimizationHealthy() const { return !current_estimate_.empty(); }

gtsam::Pose3 GraphOptimizer::eigenToGtsam(const Eigen::Isometry3d& pose) {
    return gtsam::Pose3(gtsam::Rot3(pose.rotation()), gtsam::Point3(pose.translation()));
}

Eigen::Isometry3d GraphOptimizer::gtsamToEigen(const gtsam::Pose3& pose) {
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = pose.rotation().matrix();
    result.translation() = pose.translation();
    return result;
}

gtsam::noiseModel::Gaussian::shared_ptr GraphOptimizer::createNoiseModel(const Eigen::Matrix<double, 6, 6>& info) {
    if (info.isZero(1e-10)) return nullptr;
    Eigen::Matrix<double, 6, 6> gtsam_info;
    gtsam_info.block<3,3>(0,0) = info.block<3,3>(3,3);
    gtsam_info.block<3,3>(0,3) = info.block<3,3>(3,0);
    gtsam_info.block<3,3>(3,0) = info.block<3,3>(0,3);
    gtsam_info.block<3,3>(3,3) = info.block<3,3>(0,0);
    try { return gtsam::noiseModel::Gaussian::Information(gtsam_info); }
    catch (...) { return nullptr; }
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createOdomNoiseModel() const {
    gtsam::Vector6 sigmas;
    sigmas << config_.odom_noise_rotation, config_.odom_noise_rotation, config_.odom_noise_rotation,
              config_.odom_noise_position, config_.odom_noise_position, config_.odom_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createLoopNoiseModel() const {
    gtsam::Vector6 sigmas;
    sigmas << config_.loop_noise_rotation, config_.loop_noise_rotation, config_.loop_noise_rotation,
              config_.loop_noise_position, config_.loop_noise_position, config_.loop_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Diagonal::shared_ptr GraphOptimizer::createPriorNoiseModel() const {
    gtsam::Vector6 sigmas;
    sigmas << config_.prior_noise_rotation, config_.prior_noise_rotation, config_.prior_noise_rotation,
              config_.prior_noise_position, config_.prior_noise_position, config_.prior_noise_position;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::noiseModel::Base::shared_ptr GraphOptimizer::createRobustNoiseModel(
    const Eigen::Matrix<double, 6, 6>& information, bool use_robust) const {
    gtsam::noiseModel::Gaussian::shared_ptr base_noise;
    if (!information.isZero(1e-10)) {
        Eigen::Matrix<double, 6, 6> gtsam_info;
        gtsam_info.block<3,3>(0,0) = information.block<3,3>(3,3);
        gtsam_info.block<3,3>(0,3) = information.block<3,3>(3,0);
        gtsam_info.block<3,3>(3,0) = information.block<3,3>(0,3);
        gtsam_info.block<3,3>(3,3) = information.block<3,3>(0,0);
        try { base_noise = gtsam::noiseModel::Gaussian::Information(gtsam_info); }
        catch (...) { base_noise = createLoopNoiseModel(); }
    } else {
        base_noise = createLoopNoiseModel();
    }
    if (use_robust && config_.use_robust_kernel) {
        if (config_.robust_kernel_type == "Huber") {
            return gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(config_.robust_kernel_delta), base_noise);
        } else if (config_.robust_kernel_type == "Cauchy") {
            return gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Cauchy::Create(config_.robust_kernel_delta), base_noise);
        } else if (config_.robust_kernel_type == "DCS") {
            return gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::DCS::Create(config_.robust_kernel_delta), base_noise);
        }
    }
    return base_noise;
}

} // namespace n3mapping
