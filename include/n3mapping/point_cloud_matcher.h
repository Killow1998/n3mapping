#ifndef N3MAPPING_POINT_CLOUD_MATCHER_H
#define N3MAPPING_POINT_CLOUD_MATCHER_H

#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"

namespace n3mapping {

/**
 * @brief 点云配准结果
 *
 * 包含配准变换、得分和信息矩阵
 */
struct MatchResult
{
    /// 配准是否成功
    bool success = false;

    /// 优化是否收敛（即使 success=false 也可能为 true）
    bool converged = false;

    /// 相对位姿变换 (T_target_source)
    Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();

    /// 配准得分 (越小越好)
    double fitness_score = std::numeric_limits<double>::max();

    /// 内点数量
    size_t num_inliers = 0;

    /// 内点比例（相对于 source downsampled 点数）
    double inlier_ratio = 0.0;

    /// 信息矩阵 (6x6)
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
};

/**
 * @brief 点云配准器
 *
 * 使用 small_gicp 进行点云配准，支持单次配准和并行批量配准
 * Requirements: 3.1, 3.2, 3.5, 3.6, 3.7, 4.8, 13.1
 */
class PointCloudMatcher
{
  public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;
    using SmallGicpCloud = small_gicp::PointCloud;
    using SmallGicpKdTree = small_gicp::KdTree<SmallGicpCloud>;

    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit PointCloudMatcher(const Config& config);

    /**
     * @brief 析构函数
     */
    ~PointCloudMatcher() = default;

    /**
     * @brief 单次点云配准
     *
     * 使用原始点云实时预处理，确保配准精度
     *
     * @param target 目标关键帧
     * @param source 源关键帧
     * @param init_guess 初始位姿猜测 (默认为单位矩阵)
     * @return 配准结果
     */
    MatchResult align(const Keyframe::Ptr& target,
                      const Keyframe::Ptr& source,
                      const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());

    /**
     * @brief 并行批量配准
     *
     * 使用 OpenMP 并行计算多个配准任务
     *
     * @param pairs 关键帧对列表 (target, source)
     * @param init_guesses 初始位姿猜测列表
     * @return 配准结果列表
     */
    std::vector<MatchResult> alignBatch(const std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>>& pairs,
                                        const std::vector<Eigen::Isometry3d>& init_guesses);

    /**
     * @brief 直接使用点云进行配准
     *
     * 用于将当前帧与子图进行配准
     * target_cloud 应该已经在世界坐标系下
     * source_cloud 在 body 坐标系下
     *
     * @param target_cloud 目标点云 (世界坐标系下的子图)
     * @param source_cloud 源点云 (body 坐标系下的当前帧)
     * @param init_guess 初始位姿猜测 (source 在世界坐标系下的位姿)
     * @return 配准结果，T_target_source 表示 source 相对于 target 的变换
     */
    MatchResult alignCloud(const PointCloudT::Ptr& target_cloud,
                           const PointCloudT::Ptr& source_cloud,
                           const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());

    /**
     * @brief 点云预处理
     *
     * 下采样 + 法向量/协方差估计 + KD 树构建
     * 配准时实时调用，不缓存结果
     * Requirements: 3.5, 3.6
     *
     * @param cloud PCL 点云
     * @return 预处理后的 small_gicp 点云和 KD 树
     */
    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>> preprocessPointCloud(const PointCloudT::Ptr& cloud);

    /**
     * @brief 获取配准设置
     * @return 配准设置引用
     */
    const small_gicp::RegistrationSetting& getSettings() const { return setting_; }

    /**
     * @brief 设置配准参数
     * @param setting 新的配准设置
     */
    void setSettings(const small_gicp::RegistrationSetting& setting) { setting_ = setting; }

  private:
    /**
     * @brief 将 PCL 点云转换为 small_gicp 点云
     * @param pcl_cloud PCL 点云
     * @return small_gicp 点云
     */
    SmallGicpCloud::Ptr convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud);

    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>>
    preprocessTargetPointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);

    SmallGicpCloud::Ptr preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);

    /// 配置参数
    Config config_;

    /// small_gicp 配准设置
    small_gicp::RegistrationSetting setting_;
};

} // namespace n3mapping

#endif // N3MAPPING_POINT_CLOUD_MATCHER_H
