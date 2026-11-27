#ifndef N3MAPPING_GRAPH_OPTIMIZER_H
#define N3MAPPING_GRAPH_OPTIMIZER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "n3mapping/graph_types.h"

#include "n3mapping/config.h"
#include "n3mapping/planar_geometry.h"

namespace gtsam {
class ISAM2;
class Pose3;
class Values;
class NonlinearFactorGraph;
class LevenbergMarquardtParams;
namespace noiseModel { class Base; class Gaussian; class Diagonal; }
}

namespace n3mapping {

class GraphOptimizer : public LoopOptimizerInterface
{
  public:
    explicit GraphOptimizer(const Config& config);

    ~GraphOptimizer();

    void addPriorFactor(int64_t id, const Eigen::Isometry3d& pose);

    void addFloorAttitudeFactor(int64_t id, const Eigen::Vector3d& normal_body);
    int floorAttitudeFactorCount() const { return floor_attitude_factor_count_; }
    const std::vector<FloorAttitudeConstraint>& floorAttitudeConstraints() const {
        return committed_floor_attitude_constraints_;
    }

    void addOdometryEdge(const EdgeInfo& edge);

    bool addSessionOdometryEdge(const EdgeInfo& edge,
                                const Eigen::Isometry3d& target_initial_pose);

    void addLoopEdge(const EdgeInfo& edge);

    bool rebaseSessionAndAddLoopEdge(const EdgeInfo& edge,
                                     int64_t first_session_node_id);

    bool addSessionAnchorEdge(const EdgeInfo& edge);
    static bool isRobustGlobalEdge(EdgeType type);
    bool hasGlobalConstraint() const;

    void setFixedNodes(const std::set<int64_t>& ids) { fixed_node_ids_ = ids; }
    const std::set<int64_t>& fixedNodes() const { return fixed_node_ids_; }

    bool optimizeLoopEdges(const std::vector<EdgeInfo>& edges);
    void optimize();

    bool incrementalOptimize();

    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;

    Eigen::Isometry3d getOptimizedPose(int64_t id) const;

    bool hasNode(int64_t id) const;

    size_t getNumNodes() const;

    size_t getNumEdges() const;

    bool hasLoopClosure() const;

    bool loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes, const std::vector<EdgeInfo>& edges);

    bool loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
                   const std::vector<EdgeInfo>& edges,
                   const std::vector<FloorAttitudeConstraint>& floor_constraints,
                   const std::vector<SharedPlane>& planes = {},
                   bool restore_estimates = false,
                   const Eigen::Isometry3d* prior_pose = nullptr);

    bool replaceSharedPlanes(const std::vector<SharedPlane>& planes,
                             bool optimize, GeometryOptimizationStats& stats);
    std::vector<SharedPlane> sharedPlanes() const;

    size_t appendPlaneObservations(int64_t id, const std::vector<PlaneMoment>& moments);
    Eigen::Isometry3d gaugePrior() const;

    bool appendCommittedTail(const GraphOptimizer& live,
                             const GraphOptimizer& snapshot,
                             const std::map<int64_t, std::vector<PlaneMoment>>& body_moments);
    void swapWith(GraphOptimizer& other);

    std::unique_ptr<GraphOptimizer> cloneCommitted() const;

    std::vector<EdgeInfo> getEdges() const;

    void clear();

    void rollbackToLastState();

    bool isOptimizationHealthy() const;

  private:
    bool optimizeGeometryUpdate(const gtsam::Values* rebased_poses = nullptr);
    gtsam::Values solveConditioned(
        const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
        const gtsam::LevenbergMarquardtParams& params) const;
    struct GtsamState;
    std::unique_ptr<GtsamState> gtsam_;
    Config config_;
    std::vector<EdgeInfo> edges_;
    std::vector<SharedPlane> shared_planes_;
    std::vector<std::pair<size_t, PlaneObservation>> pending_plane_observations_;
    std::vector<EdgeInfo> pending_edges_;
    std::set<int64_t> node_ids_;
    std::set<int64_t> fixed_node_ids_;
    std::set<int64_t> pending_node_ids_;
    bool has_loop_closure_;
    bool pending_has_loop_closure_;
    bool has_session_anchor_ = false;
    bool pending_has_session_anchor_ = false;
    int floor_attitude_factor_count_ = 0;
    std::vector<FloorAttitudeConstraint> committed_floor_attitude_constraints_;
    std::vector<FloorAttitudeConstraint> pending_floor_attitude_constraints_;
    bool needs_optimization_;

    static gtsam::Pose3 eigenToGtsam(const Eigen::Isometry3d& pose);

    static Eigen::Isometry3d gtsamToEigen(const gtsam::Pose3& pose);

    std::unique_ptr<gtsam::ISAM2> createISAM2() const;

    void clearPending();

    bool hasAnyNode(int64_t id) const;

    bool ensureTargetInitialValue(int64_t from_id,
                                  int64_t to_id,
                                  const Eigen::Isometry3d& measurement);

    void commitPending(const gtsam::Values& optimized_estimate, std::unique_ptr<gtsam::ISAM2> optimized_isam2);

    void rebuildISAM2FromCommitted();

    bool optimizeCandidate(const gtsam::NonlinearFactorGraph& candidate_graph,
                           const gtsam::Values& candidate_values,
                           gtsam::Values* optimized_estimate) const;

    void appendPendingTo(gtsam::NonlinearFactorGraph* candidate_graph, gtsam::Values* candidate_values) const;

    static boost::shared_ptr<gtsam::noiseModel::Gaussian> createNoiseModel(const Eigen::Matrix<double, 6, 6>& info);

    boost::shared_ptr<gtsam::noiseModel::Diagonal> createOdomNoiseModel() const;

    boost::shared_ptr<gtsam::noiseModel::Diagonal> createLoopNoiseModel() const;

    boost::shared_ptr<gtsam::noiseModel::Diagonal> createPriorNoiseModel() const;

    boost::shared_ptr<gtsam::noiseModel::Base> createRobustNoiseModel(const Eigen::Matrix<double, 6, 6>& information, bool use_robust = true) const;

    boost::shared_ptr<gtsam::noiseModel::Base> createXYYawLoopNoiseModel(const Eigen::Matrix<double, 6, 6>& information,
                                                                  bool use_robust = true) const;
};

}

#endif
