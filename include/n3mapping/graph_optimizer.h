// GraphOptimizer: GTSAM-based factor graph with iSAM2 incremental optimization.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "n3mapping/config.h"

namespace n3mapping {

enum class EdgeType { ODOMETRY, LOOP };

struct EdgeInfo {
    int64_t from_id = -1;
    int64_t to_id = -1;
    Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    EdgeType type = EdgeType::ODOMETRY;

    EdgeInfo() = default;
    EdgeInfo(int64_t from, int64_t to, const Eigen::Isometry3d& meas,
             const Eigen::Matrix<double, 6, 6>& info, EdgeType t)
        : from_id(from), to_id(to), measurement(meas), information(info), type(t) {}
};

class LoopOptimizerInterface {
public:
    virtual ~LoopOptimizerInterface() = default;
    virtual void addLoopEdge(const EdgeInfo& edge) = 0;
    virtual void incrementalOptimize() = 0;
};

class GraphOptimizer : public LoopOptimizerInterface {
public:
    explicit GraphOptimizer(const Config& config);
    ~GraphOptimizer() = default;

    void addPriorFactor(int64_t id, const Eigen::Isometry3d& pose);
    void addOdometryEdge(const EdgeInfo& edge);
    void addLoopEdge(const EdgeInfo& edge) override;

    void optimize();
    void incrementalOptimize() override;

    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;
    Eigen::Isometry3d getOptimizedPose(int64_t id) const;
    bool hasNode(int64_t id) const;
    size_t getNumNodes() const;
    size_t getNumEdges() const;
    bool hasLoopClosure() const;

    void loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
                   const std::vector<EdgeInfo>& edges);
    std::vector<EdgeInfo> getEdges() const;
    void clear();
    void rollbackToLastState();
    bool isOptimizationHealthy() const;

private:
    Config config_;
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values initial_values_;
    gtsam::Values new_values_;
    gtsam::Values current_estimate_;
    gtsam::Values last_good_estimate_;
    std::vector<EdgeInfo> edges_;
    std::set<int64_t> node_ids_;
    bool has_loop_closure_;
    bool needs_optimization_;

    static gtsam::Pose3 eigenToGtsam(const Eigen::Isometry3d& pose);
    static Eigen::Isometry3d gtsamToEigen(const gtsam::Pose3& pose);
    static gtsam::noiseModel::Gaussian::shared_ptr createNoiseModel(const Eigen::Matrix<double, 6, 6>& info);
    gtsam::noiseModel::Diagonal::shared_ptr createOdomNoiseModel() const;
    gtsam::noiseModel::Diagonal::shared_ptr createLoopNoiseModel() const;
    gtsam::noiseModel::Diagonal::shared_ptr createPriorNoiseModel() const;
    gtsam::noiseModel::Base::shared_ptr createRobustNoiseModel(
        const Eigen::Matrix<double, 6, 6>& information, bool use_robust = true) const;
};

} // namespace n3mapping
