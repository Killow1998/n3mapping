#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

/**
 * @brief 重定位结果
 */
struct RelocResult
{
    bool success = false;             // 重定位是否成功
    int64_t matched_keyframe_id = -1; // 匹配的关键帧 ID
    Eigen::Isometry3d pose_in_map;    // 在地图坐标系中的位姿
    double confidence = 0.0;          // 置信度 (基于配准得分)
    double fitness_score = 0.0;       // 配准得分

    RelocResult()
      : pose_in_map(Eigen::Isometry3d::Identity())
    {
    }
};

/**
 * @brief 世界定位模块
 *
 * 提供全局重定位和跟踪定位功能：
 * - 全局重定位：使用 ScanContext 搜索候选帧，small_gicp 精确配准
 * - 跟踪定位：融合里程计和地图约束，维护 T_map_odom 变换
 */
class WorldLocalizing
{
  public:
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    /**
     * @brief 构造函数
     * @param config 配置参数
     * @param keyframe_manager 关键帧管理器引用
     * @param loop_detector 回环检测器引用
     * @param matcher 点云配准器引用
     */
    WorldLocalizing(const Config& config, KeyframeManager& keyframe_manager, LoopDetector& loop_detector, PointCloudMatcher& matcher);

    /**
     * @brief 全局重定位
     *
     * 使用 ScanContext 在已加载的地图中搜索候选帧，
     * 然后使用 small_gicp 进行精确配准验证。
     *
     * @param cloud 当前点云
     * @return 重定位结果
     */
    RelocResult relocalize(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);

    /**
     * @brief 跟踪定位
     *
     * 在重定位成功后，使用里程计增量和地图约束进行跟踪定位。
     * 维护 T_map_odom 变换，将里程计位姿转换到地图坐标系。
     *
     * @param cloud 当前点云
     * @param odom_pose 当前里程计位姿
     * @return 重定位结果
     */
    RelocResult trackLocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);

    /**
     * @brief 检查是否已成功重定位
     * @return 是否已重定位
     */
    bool isRelocalized() const;

    /**
     * @brief 获取地图到里程计的变换
     * @return T_map_odom 变换
     */
    Eigen::Isometry3d getMapToOdomTransform() const;

    /**
     * @brief 重置重定位状态
     */
    void reset();

    /**
     * @brief 设置地图到里程计的变换 (用于地图续建)
     * @param T_map_odom 变换矩阵
     */
    void setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom);

    /**
     * @brief 获取上次匹配的关键帧 ID
     * @return 关键帧 ID，如果未重定位则返回 -1
     */
    int64_t getLastMatchedKeyframeId() const;

  private:
    /**
     * @brief 使用 ScanContext 搜索候选帧
     * @param cloud 当前点云
     * @return 候选帧列表 (按 SC 距离排序)
     */
    std::vector<LoopCandidate> searchCandidates(const PointCloudT::Ptr& cloud);

    /**
     * @brief 验证候选帧
     * @param cloud 当前点云
     * @param candidates 候选帧列表
     * @return 最佳匹配结果
     */
    RelocResult verifyCandidates(const PointCloudT::Ptr& cloud, const std::vector<LoopCandidate>& candidates);

    /**
     * @brief 查找最近的关键帧
     * @param pose 当前位姿
     * @return 最近关键帧的 ID，如果没有则返回 -1
     */
    int64_t findNearestKeyframe(const Eigen::Isometry3d& pose) const;

    Config config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    PointCloudMatcher& matcher_;

    bool is_relocalized_;              // 是否已重定位
    Eigen::Isometry3d T_map_odom_;     // 地图到里程计的变换
    int64_t last_matched_id_;          // 上次匹配的关键帧 ID
    Eigen::Isometry3d last_odom_pose_; // 上次里程计位姿
    int consecutive_track_failures_;   // 连续跟踪失败次数

    mutable std::mutex mutex_;
};

} // namespace n3mapping
