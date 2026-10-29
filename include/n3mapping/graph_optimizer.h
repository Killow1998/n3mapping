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
namespace noiseModel { class Base; class Gaussian; class Diagonal; }
}

namespace n3mapping {

class GraphOptimizer : public LoopOptimizerInterface
{
  public:
    explicit GraphOptimizer(const Config& config);

    ~GraphOptimizer();

    // ==================== 添加因子 ====================

    void addPriorFactor(int64_t id, const Eigen::Isometry3d& pose);

    // Absolute roll/pitch for one node, from the floor normal measured in that
    // keyframe's own sensor frame. Leaves yaw free.
    void addFloorAttitudeFactor(int64_t id, const Eigen::Vector3d& normal_body);
    int floorAttitudeFactorCount() const { return floor_attitude_factor_count_; }
    const std::vector<FloorAttitudeConstraint>& floorAttitudeConstraints() const {
        return committed_floor_attitude_constraints_;
    }

    void addOdometryEdge(const EdgeInfo& edge);
    // Adds an odometry edge inside an extension session. The raw measurement
    // is preserved, but the factor is robust because an independently
    // verified loaded-map pose may legitimately expose large session-LIO
    // drift. The explicit target pose keeps that geometric observation in the
    // optimizer's initial basin. Both the node and edge remain pending until
    // incrementalOptimize() succeeds.
    bool addSessionOdometryEdge(const EdgeInfo& edge,
                                const Eigen::Isometry3d& target_initial_pose);

    void addLoopEdge(const EdgeInfo& edge);
    // A trusted cross-session loop can begin far outside a redescending
    // robust kernel's basin. Reinitialize only the new session by the loop's
    // rigid correction, then transactionally rebuild with every original edge
    // unchanged and the loop appended. This changes estimates, not odometry
    // measurements or serialized edge semantics.
    bool rebaseSessionAndAddLoopEdge(const EdgeInfo& edge,
                                     int64_t first_session_node_id);
    // Adds a robust cross-session edge. If the target node is new, its
    // initial pose is derived from the source pose and measurement.
    // Returns false without staging anything when the source node is absent.
    bool addSessionAnchorEdge(const EdgeInfo& edge);
    static bool isRobustGlobalEdge(EdgeType type);
    bool hasGlobalConstraint() const;

    // ==================== 优化 ====================

    void optimize();

    bool incrementalOptimize();

    // ==================== 获取结果 ====================

    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;

    Eigen::Isometry3d getOptimizedPose(int64_t id) const;

    bool hasNode(int64_t id) const;

    size_t getNumNodes() const;

    size_t getNumEdges() const;

    bool hasLoopClosure() const;

    // ==================== 序列化支持 ====================

    bool loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes, const std::vector<EdgeInfo>& edges);
    // Loads nodes, edges and floor attitude constraints together. The old
    // overload delegates here with an empty constraint list.
    bool loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes,
                   const std::vector<EdgeInfo>& edges,
                   const std::vector<FloorAttitudeConstraint>& floor_constraints,
                   const std::vector<SharedPlane>& planes = {},
                   bool restore_estimates = false,
                   const Eigen::Isometry3d* prior_pose = nullptr);
    // Replace associations transactionally; retain every original graph factor.
    bool replaceSharedPlanes(const std::vector<SharedPlane>& planes,
                             bool optimize, GeometryOptimizationStats& stats);
    std::vector<SharedPlane> sharedPlanes() const;
    // Stage observations for a pending new node against committed geometry.
    // The append solve holds old poses/planes fixed; the worker later refines all.
    size_t appendPlaneObservations(int64_t id, const std::vector<PlaneMoment>& moments);
    Eigen::Isometry3d gaugePrior() const;
    // Catch up an isolated solve only when the live graph has appended nodes
    // without changing its snapshot's factors or poses. No registration repeats.
    // Supply unassociated body moments: old-map admission is not valid for a
    // newly solved reference. Missing geometry is represented by an empty map.
    bool appendCommittedTail(const GraphOptimizer& live,
                             const GraphOptimizer& snapshot,
                             const std::map<int64_t, std::vector<PlaneMoment>>& body_moments);
    void swapWith(GraphOptimizer& other);
    // Owner-thread snapshot for an isolated loop solve. Factors are immutable;
    // estimates and staging state belong exclusively to the returned optimizer.
    std::unique_ptr<GraphOptimizer> cloneCommitted() const;

    std::vector<EdgeInfo> getEdges() const;

    void clear();

    // ==================== 错误处理 ====================

    void rollbackToLastState();

    bool isOptimizationHealthy() const;

  private:
    bool optimizeGeometryUpdate();
    struct GtsamState;
    std::unique_ptr<GtsamState> gtsam_;
    Config config_;                           ///< 配置参数
    std::vector<EdgeInfo> edges_;             ///< 所有边
    std::vector<SharedPlane> shared_planes_;
    std::vector<std::pair<size_t, PlaneObservation>> pending_plane_observations_;
    std::vector<EdgeInfo> pending_edges_;     ///< 待提交边
    std::set<int64_t> node_ids_;              ///< 已添加的节点 ID
    std::set<int64_t> pending_node_ids_;      ///< 待提交节点 ID
    bool has_loop_closure_;                   ///< 是否有回环约束
    bool pending_has_loop_closure_;           ///< 待提交更新是否含回环约束
    bool has_session_anchor_ = false;
    bool pending_has_session_anchor_ = false;
    int floor_attitude_factor_count_ = 0;
    std::vector<FloorAttitudeConstraint> committed_floor_attitude_constraints_;
    std::vector<FloorAttitudeConstraint> pending_floor_attitude_constraints_;
    bool needs_optimization_;                 ///< 是否需要优化

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

} // namespace n3mapping

#endif // N3MAPPING_GRAPH_OPTIMIZER_H
