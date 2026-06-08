// MapSerializer: Protobuf-based map save/load and global PCD export.
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/core/types.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3map.pb.h"

namespace n3mapping {

class MapSerializer {
public:
    explicit MapSerializer(const Config& config);
    ~MapSerializer() = default;

    bool saveMap(const std::string& filepath,
                 const KeyframeManager& keyframe_manager,
                 const LoopDetector& loop_detector,
                 const GraphOptimizer& optimizer);
    bool saveMap(const std::string& filepath,
                 const KeyframeManager& keyframe_manager,
                 const LoopDetector& loop_detector,
                 const GraphOptimizer& optimizer,
                 const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory);
    bool saveMap(const std::string& filepath,
                 const KeyframeManager& keyframe_manager,
                 const LoopDetector& loop_detector,
                 const GraphOptimizer& optimizer,
                 const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory,
                 const core::DenseTrajectoryMetadata& dense_trajectory_metadata);

    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 const PbstreamLoadOptions& options);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                 const PbstreamLoadOptions& options);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                 core::DenseTrajectoryMetadata* dense_trajectory_metadata);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                 core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                 const PbstreamLoadOptions& options);

    bool saveGlobalMap(const std::string& filepath,
                       const KeyframeManager& keyframe_manager,
                       double voxel_size = 0.1);
    pcl::PointCloud<pcl::PointXYZI>::Ptr buildGlobalMap(const KeyframeManager& keyframe_manager,
                                                        double voxel_size = 0.1) const;

private:
    Config config_;

    void keyframeToProto(const Keyframe::Ptr& kf, n3mapping::KeyframeProto* proto);
    void edgeToProto(const EdgeInfo& edge, n3mapping::EdgeProto* proto);
    void denseTrajectoryToProto(const core::DenseTrajectoryPose& pose,
                                n3mapping::DenseTrajectoryPose* proto);
    void poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto);
    void pointCloudToProto(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, n3mapping::PointCloudData* proto);
    void descriptorToProto(const Eigen::MatrixXd& descriptor, n3mapping::ScanContextDescriptor* proto);
    void rhpdToProto(const Eigen::VectorXd& rhpd, n3mapping::RHPDDescriptor* proto);
    void informationToProto(const Eigen::Matrix<double, 6, 6>& info, n3mapping::InformationMatrix* proto);
};

} // namespace n3mapping
