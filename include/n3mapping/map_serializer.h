#ifndef N3MAPPING_MAP_SERIALIZER_H
#define N3MAPPING_MAP_SERIALIZER_H

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/graph_optimizer.h"
#include "n3map.pb.h"

namespace n3mapping {

/**
 * @brief 地图序列化器类
 * 
 * 负责地图数据的序列化和反序列化，使用 Protocol Buffers 格式
 * Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 11.2, 11.3, 11.4
 */
class MapSerializer {
public:
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit MapSerializer(const Config& config);

    /**
     * @brief 析构函数
     */
    ~MapSerializer() = default;

    // ==================== 地图保存 ====================

    /**
     * @brief 保存地图到文件
     * 
     * 序列化关键帧、约束边、ScanContext 描述子到 Protobuf 文件
     * Requirements: 7.6, 8.1
     * 
     * @param filepath 保存路径
     * @param keyframe_manager 关键帧管理器
     * @param loop_detector 回环检测器
     * @param optimizer 图优化器
     * @return true 如果保存成功
     */
    bool saveMap(const std::string& filepath,
                 const KeyframeManager& keyframe_manager,
                 const LoopDetector& loop_detector,
                 const GraphOptimizer& optimizer);

    /**
     * @brief 加载地图从文件
     * 
     * 反序列化 Protobuf 文件，恢复关键帧、约束边、ScanContext 描述子
     * Requirements: 8.2, 8.3, 8.4
     * 
     * @param filepath 文件路径
     * @param keyframe_manager 关键帧管理器
     * @param loop_detector 回环检测器
     * @param optimizer 图优化器
     * @return true 如果加载成功
     */
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer);

    /**
     * @brief 保存全局点云地图
     * 
     * 使用优化后位姿变换所有关键帧点云，下采样后保存为 PCD 文件
     * Requirements: 11.2, 11.3, 11.4
     * 
     * @param filepath 保存路径 (.pcd)
     * @param keyframe_manager 关键帧管理器
     * @param voxel_size 下采样体素大小 (0 表示不下采样)
     * @return true 如果保存成功
     */
    bool saveGlobalMap(const std::string& filepath,
                       const KeyframeManager& keyframe_manager,
                       double voxel_size = 0.1);

private:
    Config config_;  ///< 配置参数

    // ==================== Proto 转换辅助函数 ====================

    /**
     * @brief 将关键帧转换为 Proto 消息
     * Requirements: 7.2, 7.3
     * 
     * @param kf 关键帧
     * @param proto Proto 消息指针
     */
    void keyframeToProto(const Keyframe::Ptr& kf, n3mapping::KeyframeProto* proto);

    /**
     * @brief 将 Proto 消息转换为关键帧
     * Requirements: 7.4, 7.5
     * 
     * @param proto Proto 消息
     * @return 关键帧指针
     */
    Keyframe::Ptr protoToKeyframe(const n3mapping::KeyframeProto& proto);

    /**
     * @brief 将约束边转换为 Proto 消息
     * Requirements: 7.2, 7.3
     * 
     * @param edge 约束边
     * @param proto Proto 消息指针
     */
    void edgeToProto(const EdgeInfo& edge, n3mapping::EdgeProto* proto);

    /**
     * @brief 将 Proto 消息转换为约束边
     * Requirements: 7.4, 7.5
     * 
     * @param proto Proto 消息
     * @return 约束边
     */
    EdgeInfo protoToEdge(const n3mapping::EdgeProto& proto);

    /**
     * @brief 将 Eigen 位姿转换为 Proto 消息
     * 
     * @param pose Eigen 位姿
     * @param proto Proto 消息指针
     */
    void poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto);

    /**
     * @brief 将 Proto 消息转换为 Eigen 位姿
     * 
     * @param proto Proto 消息
     * @return Eigen 位姿
     */
    Eigen::Isometry3d protoToPose(const n3mapping::Pose3D& proto);

    /**
     * @brief 将点云转换为 Proto 消息
     * 
     * @param cloud 点云
     * @param proto Proto 消息指针
     */
    void pointCloudToProto(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                           n3mapping::PointCloudData* proto);

    /**
     * @brief 将 Proto 消息转换为点云
     * 
     * @param proto Proto 消息
     * @return 点云指针
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr protoToPointCloud(
        const n3mapping::PointCloudData& proto);

    /**
     * @brief 将 ScanContext 描述子转换为 Proto 消息
     * 
     * @param descriptor 描述子矩阵
     * @param proto Proto 消息指针
     */
    void descriptorToProto(const Eigen::MatrixXd& descriptor,
                           n3mapping::ScanContextDescriptor* proto);

    /**
     * @brief 将 Proto 消息转换为 ScanContext 描述子
     * 
     * @param proto Proto 消息
     * @return 描述子矩阵
     */
    Eigen::MatrixXd protoToDescriptor(const n3mapping::ScanContextDescriptor& proto);

    /**
     * @brief 将信息矩阵转换为 Proto 消息 (上三角存储)
     * 
     * @param info 信息矩阵
     * @param proto Proto 消息指针
     */
    void informationToProto(const Eigen::Matrix<double, 6, 6>& info,
                            n3mapping::InformationMatrix* proto);

    /**
     * @brief 将 Proto 消息转换为信息矩阵
     * 
     * @param proto Proto 消息
     * @return 信息矩阵
     */
    Eigen::Matrix<double, 6, 6> protoToInformation(
        const n3mapping::InformationMatrix& proto);
};

}  // namespace n3mapping

#endif  // N3MAPPING_MAP_SERIALIZER_H
