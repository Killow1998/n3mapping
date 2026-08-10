// ROS-free facade for n3mapping backend processing.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "n3mapping/odometry_sanity.h"
#include "n3mapping/static_start_guard.h"
#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/core/types.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/loop_debug_logger.h"
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
    std::size_t place_candidate_count = 0;
    std::size_t edge_count = 0;
    std::size_t graph_edge_count = 0;
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
    const StaticStartGuard& staticStartGuard() const { return static_start_guard_; }
    const OdometrySanityVerdict& odometrySanity() const {
      return odometry_sanity_.verdict();
    }

    RegistrationSeedProbeResult probeLocalizationRegistration(
        const core::LioFrame::PointCloud::Ptr& cloud,
        const Eigen::Isometry3d& odom_pose,
        const Eigen::Isometry3d& oracle_pose);

    Keyframe::Ptr getKeyframe(int64_t id) const;
    std::vector<Keyframe::Ptr> getAllKeyframes() const;
    KeyframeMapRevision mapRevision() const;
    std::map<int64_t, Eigen::Isometry3d> getOptimizedPoses() const;
    std::vector<core::DenseTrajectoryPose> getDenseOptimizedTrajectory() const;
    void setExternalDenseTrajectoryRecordingEnabled(bool enabled);
    void recordDenseTrajectoryPose(CoreRunMode mode,
                                   double timestamp,
                                   const Eigen::Isometry3d& pose_world_lidar);

  private:
    using PointCloud = core::LioFrame::PointCloud;

    core::BackendOutput makeOutput(bool success,
                                   const Eigen::Isometry3d& pose,
                                   const PointCloud::Ptr& cloud) const;
    PointCloud::Ptr makeWorldCloud(const PointCloud::Ptr& cloud,
                                   const Eigen::Isometry3d& pose) const;
    void appendDenseTrajectorySample(double timestamp,
                                     const Eigen::Isometry3d& raw_pose,
                                     int64_t anchor_keyframe_id,
                                     const Eigen::Isometry3d& anchor_raw_pose,
                                     bool use_bracketing_correction = true);
    void appendDenseTrajectorySampleWithLatestAnchor(double timestamp,
                                                     const Eigen::Isometry3d& raw_pose,
                                                     bool use_bracketing_correction = true);
    std::vector<core::DenseTrajectoryPose> buildDenseOptimizedTrajectory() const;
    Eigen::Isometry3d interpolateDenseCorrection(double timestamp) const;
    void addRhpdDescriptorForKeyframe(int64_t keyframe_id, const PointCloud::Ptr& fallback_cloud);
    bool addOdometryConstraint(int64_t keyframe_id, const Eigen::Isometry3d& pose);
    void refreshOptimizedPoses();
    void appendLoopDebugCandidate(const LoopDebugCandidateEvent& event) const;
    void appendLoopDebugOptimization(const LoopDebugOptimizationEvent& event) const;

    Config config_;
    std::unique_ptr<core::N3MappingSession> session_;
    mutable std::mutex loop_queue_mutex_;
    mutable std::mutex loop_debug_mutex_;
    std::vector<int64_t> loop_detection_queue_;
    std::vector<core::AnchoredDenseTrajectorySample> dense_trajectory_samples_;
    core::DenseTrajectoryMetadata dense_trajectory_metadata_;
    bool external_dense_trajectory_recording_enabled_ = false;
    int64_t last_loop_check_id_ = -1000;
    // How many keyframes contributed an absolute attitude observation. Reported
    // so a run that silently stops finding the floor is visible.
    OdometrySanity odometry_sanity_;
    StaticStartGuard static_start_guard_;
    bool static_start_guard_configured_ = false;
    bool static_start_reported_ = false;
    bool odometry_sanity_configured_ = false;
    int floor_attitude_accepted_ = 0;
    int floor_attitude_rejected_ = 0;
    std::size_t loop_count_ = 0;
    bool map_loaded_ = false;
};

}  // namespace n3mapping
