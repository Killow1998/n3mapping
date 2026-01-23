#ifndef N3MAPPING_KEYFRAME_H
#define N3MAPPING_KEYFRAME_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

/**
 * @brief 关键帧数据结构
 * 
 * 存储关键帧的所有相关信息，包括位姿、点云和描述子
 * Requirements: 2.4
 */
struct Keyframe {
    using Ptr = std::shared_ptr<Keyframe>;
    using ConstPtr = std::shared_ptr<const Keyframe>;
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    /// 关键帧唯一标识符
    int64_t id = -1;

    /// 时间戳 (秒)
    double timestamp = 0.0;

    /// 里程计位姿 (原始，来自前端)
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();

    /// 优化后位姿 (经过图优化)
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();

    /// 原始点云 (用于 GICP 配准，确保精度)
    PointCloudT::Ptr cloud = nullptr;

    /// ScanContext 描述子 (用于回环检测)
    Eigen::MatrixXd sc_descriptor;

    /// 是否来自加载的地图 (用于地图续建)
    bool is_from_loaded_map = false;

    /**
     * @brief 默认构造函数
     */
    Keyframe() = default;

    /**
     * @brief 带参数的构造函数
     * @param id 关键帧 ID
     * @param timestamp 时间戳
     * @param pose 里程计位姿
     * @param cloud 点云数据
     */
    Keyframe(int64_t id, double timestamp, 
             const Eigen::Isometry3d& pose,
             const PointCloudT::Ptr& cloud)
        : id(id)
        , timestamp(timestamp)
        , pose_odom(pose)
        , pose_optimized(pose)
        , cloud(cloud)
        , is_from_loaded_map(false) {}

    /**
     * @brief 创建关键帧智能指针
     * @param id 关键帧 ID
     * @param timestamp 时间戳
     * @param pose 里程计位姿
     * @param cloud 点云数据
     * @return 关键帧智能指针
     */
    static Ptr create(int64_t id, double timestamp,
                      const Eigen::Isometry3d& pose,
                      const PointCloudT::Ptr& cloud) {
        return std::make_shared<Keyframe>(id, timestamp, pose, cloud);
    }

    /**
     * @brief 检查关键帧是否有效
     * @return true 如果关键帧有效
     */
    bool isValid() const {
        return id >= 0 && cloud != nullptr && !cloud->empty();
    }

    /**
     * @brief 获取位置 (优化后)
     * @return 3D 位置向量
     */
    Eigen::Vector3d getPosition() const {
        return pose_optimized.translation();
    }

    /**
     * @brief 获取旋转 (优化后)
     * @return 旋转矩阵
     */
    Eigen::Matrix3d getRotation() const {
        return pose_optimized.rotation();
    }

    /**
     * @brief 获取四元数 (优化后)
     * @return 四元数
     */
    Eigen::Quaterniond getQuaternion() const {
        return Eigen::Quaterniond(pose_optimized.rotation());
    }
};

}  // namespace n3mapping

#endif  // N3MAPPING_KEYFRAME_H
