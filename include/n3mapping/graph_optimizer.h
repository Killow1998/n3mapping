#ifndef N3MAPPING_GRAPH_OPTIMIZER_H
#define N3MAPPING_GRAPH_OPTIMIZER_H

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

/**
 * @brief 约束边类型枚举
 */
enum class EdgeType
{
    ODOMETRY, ///< 里程计约束
    LOOP      ///< 回环约束
};

/**
 * @brief 约束边信息结构
 *
 * 存储因子图中边的信息，包括连接的节点、测量值和信息矩阵
 */
struct EdgeInfo
{
    int64_t from_id;                         ///< 起始节点 ID
    int64_t to_id;                           ///< 目标节点 ID
    Eigen::Isometry3d measurement;           ///< 相对位姿测量值
    Eigen::Matrix<double, 6, 6> information; ///< 信息矩阵 (协方差逆)
    EdgeType type;                           ///< 边类型

    EdgeInfo()
      : from_id(-1)
      , to_id(-1)
      , measurement(Eigen::Isometry3d::Identity())
      , information(Eigen::Matrix<double, 6, 6>::Identity())
      , type(EdgeType::ODOMETRY)
    {
    }

    EdgeInfo(int64_t from, int64_t to, const Eigen::Isometry3d& meas, const Eigen::Matrix<double, 6, 6>& info, EdgeType t)
      : from_id(from)
      , to_id(to)
      , measurement(meas)
      , information(info)
      , type(t)
    {
    }
};

/**
 * @brief 回环优化接口（用于测试与解耦）
 */
class LoopOptimizerInterface
{
  public:
    virtual ~LoopOptimizerInterface() = default;
    virtual void addLoopEdge(const EdgeInfo& edge) = 0;
    virtual void incrementalOptimize() = 0;
};

/**
 * @brief 图优化器类
 *
 * 基于 GTSAM 的因子图优化器，支持 iSAM2 增量优化
 * Requirements: 5.1, 5.2, 5.3, 5.6
 */
class GraphOptimizer : public LoopOptimizerInterface
{
  public:
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit GraphOptimizer(const Config& config);

    /**
     * @brief 析构函数
     */
    ~GraphOptimizer() = default;

    // ==================== 添加因子 ====================

    /**
     * @brief 添加先验因子
     *
     * 为第一个关键帧添加先验约束，固定初始位姿
     * Requirements: 5.1
     *
     * @param id 节点 ID
     * @param pose 先验位姿
     */
    void addPriorFactor(int64_t id, const Eigen::Isometry3d& pose);

    /**
     * @brief 添加里程计边
     *
     * 添加相邻关键帧之间的里程计约束
     * Requirements: 5.1
     *
     * @param edge 边信息
     */
    void addOdometryEdge(const EdgeInfo& edge);

    /**
     * @brief 添加回环边
     *
     * 添加回环检测到的约束边，默认启用鲁棒核函数 (Huber/Cauchy)
     * Requirements: 5.1
     *
     * @param edge 边信息
     */
    void addLoopEdge(const EdgeInfo& edge);

    // ==================== 优化 ====================

    /**
     * @brief 执行批量优化
     *
     * 使用 Levenberg-Marquardt 算法进行批量优化
     * Requirements: 5.2
     */
    void optimize();

    /**
     * @brief 执行增量优化
     *
     * 使用 iSAM2 进行增量优化，适用于实时场景
     * Requirements: 5.2
     */
    void incrementalOptimize();

    // ==================== 获取结果 ====================

    /**
     * @brief 获取所有优化后的位姿
     *
     * Requirements: 5.3
     *
     * @return 节点 ID 到优化后位姿的映射
     */
    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;

    /**
     * @brief 获取指定节点的优化后位姿
     *
     * Requirements: 5.3
     *
     * @param id 节点 ID
     * @return 优化后位姿
     * @throws std::out_of_range 如果节点不存在
     */
    Eigen::Isometry3d getOptimizedPose(int64_t id) const;

    /**
     * @brief 检查节点是否存在
     * @param id 节点 ID
     * @return true 如果节点存在
     */
    bool hasNode(int64_t id) const;

    /**
     * @brief 获取节点数量
     * @return 节点数量
     */
    size_t getNumNodes() const;

    /**
     * @brief 获取边数量
     * @return 边数量
     */
    size_t getNumEdges() const;

    /**
     * @brief 是否有回环约束
     * @return true 如果有回环约束
     */
    bool hasLoopClosure() const;

    // ==================== 序列化支持 ====================

    /**
     * @brief 加载图数据
     *
     * 从序列化数据恢复因子图
     *
     * @param nodes 节点列表 (ID, 位姿)
     * @param edges 边列表
     */
    void loadGraph(const std::vector<std::pair<int64_t, Eigen::Isometry3d>>& nodes, const std::vector<EdgeInfo>& edges);

    /**
     * @brief 获取所有边
     * @return 边列表
     */
    std::vector<EdgeInfo> getEdges() const;

    /**
     * @brief 清空图
     */
    void clear();

    // ==================== 错误处理 ====================

    /**
     * @brief 回滚到上一次优化状态
     *
     * 当优化失败或误差激增时，恢复到上一次成功的优化状态
     */
    void rollbackToLastState();

    /**
     * @brief 检查优化是否健康
     * @return true 如果优化状态正常
     */
    bool isOptimizationHealthy() const;

  private:
    Config config_;                           ///< 配置参数
    std::unique_ptr<gtsam::ISAM2> isam2_;     ///< iSAM2 优化器
    gtsam::NonlinearFactorGraph graph_;       ///< 因子图
    gtsam::NonlinearFactorGraph new_factors_; ///< 新增因子 (用于增量优化)
    gtsam::Values initial_values_;            ///< 初始值
    gtsam::Values new_values_;                ///< 新增值 (用于增量优化)
    gtsam::Values current_estimate_;          ///< 当前估计值
    gtsam::Values last_good_estimate_;        ///< 上一次成功的估计值 (用于回滚)
    std::vector<EdgeInfo> edges_;             ///< 所有边
    std::set<int64_t> node_ids_;              ///< 已添加的节点 ID
    bool has_loop_closure_;                   ///< 是否有回环约束
    bool needs_optimization_;                 ///< 是否需要优化

    /**
     * @brief 将 Eigen 位姿转换为 GTSAM Pose3
     * @param pose Eigen 位姿
     * @return GTSAM Pose3
     */
    static gtsam::Pose3 eigenToGtsam(const Eigen::Isometry3d& pose);

    /**
     * @brief 将 GTSAM Pose3 转换为 Eigen 位姿
     * @param pose GTSAM Pose3
     * @return Eigen 位姿
     */
    static Eigen::Isometry3d gtsamToEigen(const gtsam::Pose3& pose);

    /**
     * @brief 创建噪声模型
     * @param info 信息矩阵
     * @return GTSAM 噪声模型
     */
    static gtsam::noiseModel::Gaussian::shared_ptr createNoiseModel(const Eigen::Matrix<double, 6, 6>& info);

    /**
     * @brief 创建默认里程计噪声模型
     * @return GTSAM 噪声模型
     */
    gtsam::noiseModel::Diagonal::shared_ptr createOdomNoiseModel() const;

    /**
     * @brief 创建默认回环噪声模型
     * @return GTSAM 噪声模型
     */
    gtsam::noiseModel::Diagonal::shared_ptr createLoopNoiseModel() const;

    /**
     * @brief 创建先验噪声模型
     * @return GTSAM 噪声模型
     */
    gtsam::noiseModel::Diagonal::shared_ptr createPriorNoiseModel() const;

    /**
     * @brief 创建鲁棒噪声模型
     *
     * 为回环边创建带鲁棒核函数的噪声模型，防止误匹配导致地图变形
     *
     * @param information 信息矩阵
     * @param use_robust 是否使用鲁棒核函数
     * @return GTSAM 噪声模型
     */
    gtsam::noiseModel::Base::shared_ptr createRobustNoiseModel(const Eigen::Matrix<double, 6, 6>& information, bool use_robust = true) const;
};

} // namespace n3mapping

#endif // N3MAPPING_GRAPH_OPTIMIZER_H
