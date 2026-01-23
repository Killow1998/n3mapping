#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/relocalization_module.h"

namespace n3mapping {

/**
 * @brief 地图续建状态
 */
enum class MapExtensionState {
    NOT_INITIALIZED,    // 未初始化
    MAP_LOADED,         // 地图已加载
    RELOCALIZED,        // 已重定位
    EXTENDING           // 正在续建
};

/**
 * @brief 地图续建模块
 * 
 * 提供地图续建功能：
 * - 加载已有地图
 * - 执行重定位建立初始约束
 * - 在新旧关键帧之间检测回环
 * - 保存扩展后的地图
 * 
 * Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
 */
class MapExtensionModule {
public:
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    /**
     * @brief 构造函数
     * @param config 配置参数
     * @param keyframe_manager 关键帧管理器引用
     * @param loop_detector 回环检测器引用
     * @param matcher 点云配准器引用
     * @param optimizer 图优化器引用
     * @param serializer 地图序列化器引用
     * @param relocalization 重定位模块引用
     */
    MapExtensionModule(const Config& config,
                       KeyframeManager& keyframe_manager,
                       LoopDetector& loop_detector,
                       PointCloudMatcher& matcher,
                       GraphOptimizer& optimizer,
                       MapSerializer& serializer,
                       RelocalizationModule& relocalization);

    /**
     * @brief 加载已有地图
     * 
     * Requirements: 12.1
     * 
     * @param map_path 地图文件路径
     * @return 是否加载成功
     */
    bool loadExistingMap(const std::string& map_path);

    /**
     * @brief 执行初始重定位
     * 
     * 在已加载的地图中进行重定位，建立当前位姿与地图的约束
     * 
     * Requirements: 12.2
     * 
     * @param cloud 当前点云
     * @param odom_pose 当前里程计位姿
     * @return 是否重定位成功
     */
    bool performInitialRelocalization(const PointCloudT::Ptr& cloud,
                                      const Eigen::Isometry3d& odom_pose);

    /**
     * @brief 处理新的关键帧
     * 
     * 添加新关键帧并检测与旧地图的回环
     * 
     * Requirements: 12.3, 12.4
     * 
     * @param timestamp 时间戳
     * @param odom_pose 里程计位姿
     * @param cloud 点云数据
     * @return 新关键帧 ID，如果不需要添加则返回 -1
     */
    int64_t processNewKeyframe(double timestamp,
                               const Eigen::Isometry3d& odom_pose,
                               const PointCloudT::Ptr& cloud);

    /**
     * @brief 检测新旧关键帧之间的回环
     * 
     * Requirements: 12.4, 12.5
     * 
     * @param new_keyframe_id 新关键帧 ID
     * @return 检测到的回环数量
     */
    int detectCrossLoops(int64_t new_keyframe_id);

    /**
     * @brief 保存扩展后的地图
     * 
     * Requirements: 12.6, 12.7
     * 
     * @param map_path 保存路径
     * @return 是否保存成功
     */
    bool saveExtendedMap(const std::string& map_path);

    /**
     * @brief 获取当前状态
     * @return 地图续建状态
     */
    MapExtensionState getState() const;

    /**
     * @brief 获取原始地图关键帧数量
     * @return 原始地图关键帧数量
     */
    size_t getOriginalKeyframeCount() const;

    /**
     * @brief 获取新增关键帧数量
     * @return 新增关键帧数量
     */
    size_t getNewKeyframeCount() const;

    /**
     * @brief 获取检测到的跨地图回环数量
     * @return 跨地图回环数量
     */
    size_t getCrossLoopCount() const;

    /**
     * @brief 检查关键帧是否来自原始地图
     * @param keyframe_id 关键帧 ID
     * @return 是否来自原始地图
     */
    bool isFromOriginalMap(int64_t keyframe_id) const;

    /**
     * @brief 重置模块状态
     */
    void reset();

private:
    /**
     * @brief 添加重定位约束到因子图
     * @param new_keyframe_id 新关键帧 ID
     * @param matched_keyframe_id 匹配的原始地图关键帧 ID
     * @param T_match_new 相对位姿变换
     */
    void addRelocalizationConstraint(int64_t new_keyframe_id,
                                     int64_t matched_keyframe_id,
                                     const Eigen::Isometry3d& T_match_new);

    Config config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    PointCloudMatcher& matcher_;
    GraphOptimizer& optimizer_;
    MapSerializer& serializer_;
    RelocalizationModule& relocalization_;

    MapExtensionState state_;
    size_t original_keyframe_count_;     // 原始地图关键帧数量
    int64_t original_max_keyframe_id_;   // 原始地图最大关键帧 ID
    size_t cross_loop_count_;            // 跨地图回环数量
    
    Eigen::Isometry3d last_keyframe_pose_;  // 上一个关键帧位姿
    int64_t last_keyframe_id_;              // 上一个关键帧 ID

    mutable std::mutex mutex_;
};

}  // namespace n3mapping
