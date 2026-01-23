#ifndef N3MAPPING_KEYFRAME_MANAGER_H
#define N3MAPPING_KEYFRAME_MANAGER_H

#include <map>
#include <vector>
#include <mutex>
#include <cmath>

#include "n3mapping/keyframe.h"
#include "n3mapping/config.h"

namespace n3mapping {

/**
 * @brief 关键帧管理器
 * 
 * 负责关键帧的选择、存储和检索
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.6
 */
class KeyframeManager {
public:
    using Ptr = std::shared_ptr<KeyframeManager>;

    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit KeyframeManager(const Config& config);

    /**
     * @brief 判断是否应该添加新关键帧
     * 
     * 基于距离阈值和角度阈值判断当前位姿是否应该成为关键帧
     * Requirements: 2.1, 2.2, 2.3
     * 
     * @param current_pose 当前位姿
     * @return true 如果应该添加关键帧
     */
    bool shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const;

    /**
     * @brief 添加新关键帧
     * 
     * Requirements: 2.4
     * 
     * @param timestamp 时间戳
     * @param pose 里程计位姿
     * @param cloud 点云数据
     * @return 新关键帧的 ID
     */
    int64_t addKeyframe(double timestamp,
                        const Eigen::Isometry3d& pose,
                        const Keyframe::PointCloudT::Ptr& cloud);

    /**
     * @brief 根据 ID 获取关键帧
     * 
     * Requirements: 2.6
     * 
     * @param id 关键帧 ID
     * @return 关键帧指针，如果不存在返回 nullptr
     */
    Keyframe::Ptr getKeyframe(int64_t id) const;

    /**
     * @brief 获取最新的关键帧
     * 
     * Requirements: 2.6
     * 
     * @return 最新关键帧指针，如果没有关键帧返回 nullptr
     */
    Keyframe::Ptr getLatestKeyframe() const;

    /**
     * @brief 获取所有关键帧
     * @return 所有关键帧的向量
     */
    std::vector<Keyframe::Ptr> getAllKeyframes() const;

    /**
     * @brief 获取关键帧数量
     * @return 关键帧数量
     */
    size_t size() const;

    /**
     * @brief 检查是否为空
     * @return true 如果没有关键帧
     */
    bool empty() const;

    /**
     * @brief 更新优化后的位姿
     * @param poses 位姿映射 (id -> pose)
     */
    void updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses);

    /**
     * @brief 加载关键帧 (用于地图加载)
     * @param keyframes 关键帧向量
     */
    void loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes);

    /**
     * @brief 获取下一个关键帧 ID
     * @return 下一个可用的关键帧 ID
     */
    int64_t getNextKeyframeId() const;

    /**
     * @brief 清空所有关键帧
     */
    void clear();

    /**
     * @brief 根据时间戳查找最近的关键帧
     * @param timestamp 目标时间戳
     * @return 最近的关键帧指针
     */
    Keyframe::Ptr findNearestByTimestamp(double timestamp) const;

    /**
     * @brief 根据位置查找最近的关键帧
     * @param position 目标位置
     * @return 最近的关键帧指针
     */
    Keyframe::Ptr findNearestByPosition(const Eigen::Vector3d& position) const;

    /**
     * @brief 更新关键帧的 ScanContext 描述子
     * @param id 关键帧 ID
     * @param descriptor ScanContext 描述子
     * @return true 如果更新成功
     */
    bool updateDescriptor(int64_t id, const Eigen::MatrixXd& descriptor);

    /**
     * @brief 构建局部子图
     * 
     * 将指定关键帧前后 N 帧的点云叠加成一个局部子图
     * 所有点云变换到中心关键帧的坐标系下
     * 
     * @param center_id 中心关键帧 ID
     * @param submap_size 前后各取 N 帧 (共 2N+1 帧)
     * @return 叠加后的点云 (在中心关键帧坐标系下)
     */
    Keyframe::PointCloudT::Ptr buildLocalSubmap(int64_t center_id, int submap_size) const;

private:
    Config config_;
    std::map<int64_t, Keyframe::Ptr> keyframes_;
    int64_t next_id_;
    Keyframe::Ptr last_keyframe_;
    mutable std::mutex mutex_;

    /**
     * @brief 计算两个位姿之间的平移距离
     * @param pose1 位姿1
     * @param pose2 位姿2
     * @return 平移距离 (米)
     */
    static double computeTranslationDistance(const Eigen::Isometry3d& pose1,
                                             const Eigen::Isometry3d& pose2);

    /**
     * @brief 计算两个位姿之间的旋转角度
     * @param pose1 位姿1
     * @param pose2 位姿2
     * @return 旋转角度 (弧度)
     */
    static double computeRotationAngle(const Eigen::Isometry3d& pose1,
                                       const Eigen::Isometry3d& pose2);
};

}  // namespace n3mapping

#endif  // N3MAPPING_KEYFRAME_MANAGER_H
