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

namespace n3mapping {
class KeyframeManager;
class LoopDetector;
class GraphOptimizer;
class SubmapBuilder;
struct PbstreamLoadOptions;

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
                 const SubmapBuilder* submap_builder);
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
    bool saveMap(const std::string& filepath,
                 const KeyframeManager& keyframe_manager,
                 const LoopDetector& loop_detector,
                 const GraphOptimizer& optimizer,
                 const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory,
                 const core::DenseTrajectoryMetadata& dense_trajectory_metadata,
                 const SubmapBuilder* submap_builder);

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
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                 core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                 SubmapBuilder* submap_builder);
    bool loadMap(const std::string& filepath,
                 KeyframeManager& keyframe_manager,
                 LoopDetector& loop_detector,
                 GraphOptimizer& optimizer,
                 std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                 core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                 const PbstreamLoadOptions& options,
                 SubmapBuilder* submap_builder);

    bool saveGlobalMap(const std::string& filepath,
                       const KeyframeManager& keyframe_manager,
                       double voxel_size = 0.1);
    pcl::PointCloud<pcl::PointXYZI>::Ptr buildGlobalMap(const KeyframeManager& keyframe_manager,
                                                        double voxel_size = 0.1) const;

private:
    static constexpr const char* kMapVersion = "2.10.0";
    Config config_;

};

} // namespace n3mapping
