// ROS-free facade for n3mapping backend processing.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/core/types.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

enum class CoreRunMode {
    MAPPING,
    LOCALIZATION,
    MAP_EXTENSION,
};

CoreRunMode parseCoreRunMode(const std::string& mode);
const char* coreRunModeName(CoreRunMode mode);
bool coreRunModeLoadsMap(CoreRunMode mode);
bool coreRunModeSavesMap(CoreRunMode mode);
bool coreRunModeProcessesLoopClosures(CoreRunMode mode);

struct CoreLoopClosureResult {
    bool optimized = false;
    std::size_t edge_count = 0;
    std::size_t pose_update_count = 0;
    double loop_residual_translation_before = 0.0;
    double loop_residual_translation_after = 0.0;
    double loop_residual_rotation_before = 0.0;
    double loop_residual_rotation_after = 0.0;
    double mean_pose_update_translation = 0.0;
    double max_pose_update_translation = 0.0;
    double mean_pose_update_rotation = 0.0;
    double max_pose_update_rotation = 0.0;
    std::vector<VerifiedLoop> accepted_loops;
};

class N3MappingCore {
  public:
    explicit N3MappingCore(const Config& config);
    ~N3MappingCore();

    core::BackendOutput processMappingFrame(const core::LioFrame& frame);
    core::BackendOutput processLocalizationFrame(const core::LioFrame& frame);
    core::BackendOutput processMapExtensionFrame(const core::LioFrame& frame);
    core::BackendOutput processFrame(CoreRunMode mode, const core::LioFrame& frame);
    CoreLoopClosureResult processPendingLoopClosures();

    bool loadMap(const std::string& map_path);
    bool saveMap(const std::string& map_path);
    bool saveGlobalMap(const std::string& pcd_path);
    bool saveMapSnapshot(std::string* error = nullptr);
    core::LioFrame::PointCloud::Ptr buildGlobalMap() const;
    bool mapLoaded() const;

    Keyframe::Ptr getKeyframe(int64_t id) const;
    std::vector<Keyframe::Ptr> getAllKeyframes() const;
    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;

  private:
    using PointCloud = core::LioFrame::PointCloud;

    core::BackendOutput makeOutput(bool success,
                                   const Eigen::Isometry3d& pose,
                                   const PointCloud::Ptr& cloud) const;
    PointCloud::Ptr makeWorldCloud(const PointCloud::Ptr& cloud,
                                   const Eigen::Isometry3d& pose) const;
    void addRhpdDescriptorForKeyframe(int64_t keyframe_id, const PointCloud::Ptr& fallback_cloud);
    bool addOdometryConstraint(int64_t keyframe_id, const Eigen::Isometry3d& pose);
    void refreshOptimizedPoses();

    Config config_;
    std::unique_ptr<core::N3MappingSession> session_;
    mutable std::mutex loop_queue_mutex_;
    std::vector<int64_t> loop_detection_queue_;
    int64_t last_loop_check_id_ = -1000;
    std::size_t loop_count_ = 0;
    bool map_loaded_ = false;
};

}  // namespace n3mapping
