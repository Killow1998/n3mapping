#ifndef N3MAPPING_LOOP_DETECTOR_H
#define N3MAPPING_LOOP_DETECTOR_H

#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"
#include "Scancontext/Scancontext.h"

namespace n3mapping {

/**
 * @brief 回环候选结构
 * 
 * 存储回环检测的候选帧信息
 */
struct LoopCandidate {
    /// 查询帧 ID
    int64_t query_id = -1;
    
    /// 匹配帧 ID
    int64_t match_id = -1;
    
    /// ScanContext 距离 (越小越相似)
    double sc_distance = std::numeric_limits<double>::max();
    
    /// 估计的 yaw 差异 (弧度)
    float yaw_diff_rad = 0.0f;
    
    /// 是否有效
    bool isValid() const { return query_id >= 0 && match_id >= 0; }
};

/**
 * @brief 验证后的回环结构
 * 
 * 存储经过 ICP 验证的回环信息
 */
struct VerifiedLoop {
    /// 查询帧 ID
    int64_t query_id = -1;
    
    /// 匹配帧 ID
    int64_t match_id = -1;
    
    /// 相对位姿变换 (T_match_query)
    Eigen::Isometry3d T_match_query = Eigen::Isometry3d::Identity();
    
    /// 配准得分 (越小越好)
    double fitness_score = std::numeric_limits<double>::max();
    
    /// 信息矩阵 (6x6)
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    
    /// 是否验证通过
    bool verified = false;
    
    /// 是否有效
    bool isValid() const { return verified && query_id >= 0 && match_id >= 0; }
};

/**
 * @brief 回环检测器
 * 
 * 基于 ScanContext 的回环检测模块
 * Requirements: 4.1, 4.2, 4.3, 4.6, 4.7, 4.8, 4.9
 */
class LoopDetector {
public:
    using Ptr = std::shared_ptr<LoopDetector>;
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit LoopDetector(const Config& config);

    /**
     * @brief 析构函数
     */
    ~LoopDetector() = default;

    /**
     * @brief 从点云生成 ScanContext 描述子
     * 
     * Requirements: 4.1
     * 
     * @param cloud 输入点云
     * @return ScanContext 描述子矩阵
     */
    Eigen::MatrixXd makeScanContext(const PointCloudT::Ptr& cloud);

    /**
     * @brief 添加关键帧描述子
     * 
     * 生成 ScanContext 描述子并添加到索引中
     * Requirements: 4.1
     * 
     * @param keyframe_id 关键帧 ID
     * @param cloud 关键帧点云
     * @return 生成的描述子
     */
    Eigen::MatrixXd addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud);

    /**
     * @brief 添加已有的描述子
     * 
     * 用于地图加载时添加已有描述子
     * 
     * @param keyframe_id 关键帧 ID
     * @param descriptor ScanContext 描述子
     */
    void addDescriptor(int64_t keyframe_id, const Eigen::MatrixXd& descriptor);

    /**
     * @brief 检测回环候选帧
     * 
     * 在历史关键帧中搜索相似描述子，排除最近 N 帧
     * Requirements: 4.2, 4.3, 4.6
     * 
     * @param query_id 查询帧 ID
     * @return 候选回环帧列表
     */
    std::vector<LoopCandidate> detectLoopCandidates(int64_t query_id);

    /**
     * @brief 验证单个回环候选
     * 
     * 使用 PointCloudMatcher 进行 ICP 验证
     * Requirements: 4.4, 4.5
     * 
     * @param candidate 回环候选
     * @param query_keyframe 查询关键帧
     * @param match_keyframe 匹配关键帧
     * @param matcher 点云配准器
     * @return 验证后的回环结果
     */
    VerifiedLoop verifyLoopCandidate(const LoopCandidate& candidate,
                                      const Keyframe::Ptr& query_keyframe,
                                      const Keyframe::Ptr& match_keyframe,
                                      PointCloudMatcher& matcher);

    /**
     * @brief 并行验证多个回环候选
     * 
     * 使用 OpenMP 并行计算多个候选帧的配准得分
     * Requirements: 4.4, 4.5, 4.8
     * 
     * @param candidates 回环候选列表
     * @param keyframes 关键帧映射 (id -> keyframe)
     * @param matcher 点云配准器
     * @return 验证后的回环结果列表 (只包含验证通过的)
     */
    std::vector<VerifiedLoop> verifyLoopCandidatesBatch(
        const std::vector<LoopCandidate>& candidates,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        PointCloudMatcher& matcher);

    /**
     * @brief 重建 KD 树索引
     * 
     * 用于地图加载后重建索引
     */
    void rebuildTree();

    /**
     * @brief 获取所有描述子
     * 
     * 用于地图序列化
     * 
     * @return 描述子列表 (keyframe_id, descriptor)
     */
    std::vector<std::pair<int64_t, Eigen::MatrixXd>> getDescriptors() const;

    /**
     * @brief 加载描述子
     * 
     * 用于地图反序列化
     * 
     * @param descriptors 描述子列表 (keyframe_id, descriptor)
     */
    void loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors);

    /**
     * @brief 获取描述子数量
     * @return 描述子数量
     */
    size_t size() const;

    /**
     * @brief 清空所有描述子
     */
    void clear();

    /**
     * @brief 获取描述子维度
     * @return (rows, cols) 描述子维度
     */
    std::pair<int, int> getDescriptorDimensions() const;

    /**
     * @brief 根据 ID 获取描述子
     * @param keyframe_id 关键帧 ID
     * @return 描述子矩阵，如果不存在返回空矩阵
     */
    Eigen::MatrixXd getDescriptor(int64_t keyframe_id) const;

    /**
     * @brief 获取排除的最近帧数
     * @return 排除的最近帧数
     */
    int getNumExcludeRecent() const { return config_.sc_num_exclude_recent; }

    /**
     * @brief 计算两个描述子之间的距离
     * 
     * 公开此方法以便 RelocalizationModule 使用
     * 
     * @param sc1 描述子1
     * @param sc2 描述子2
     * @return (距离, yaw偏移索引)
     */
    std::pair<double, int> computeDistance(const Eigen::MatrixXd& sc1, 
                                           const Eigen::MatrixXd& sc2);

private:

    /// 配置参数
    Config config_;

    /// ScanContext 管理器
    SCManager sc_manager_;

    /// 关键帧 ID 到 SC 索引的映射
    std::map<int64_t, size_t> id_to_index_;

    /// SC 索引到关键帧 ID 的映射
    std::vector<int64_t> index_to_id_;

    /// 存储的描述子
    std::vector<Eigen::MatrixXd> descriptors_;

    /// 互斥锁
    mutable std::mutex mutex_;
};

}  // namespace n3mapping

#endif  // N3MAPPING_LOOP_DETECTOR_H
