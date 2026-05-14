// ROS-free facade for n3mapping backend processing.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/core/types.h"
#include "n3mapping/keyframe.h"

namespace n3mapping {

class N3MappingCore {
  public:
    explicit N3MappingCore(const Config& config);
    ~N3MappingCore();

    core::BackendOutput processMappingFrame(const core::LioFrame& frame);
    core::BackendOutput processLocalizationFrame(const core::LioFrame& frame);
    core::BackendOutput processMapExtensionFrame(const core::LioFrame& frame);

    bool loadMap(const std::string& map_path);
    bool saveMap(const std::string& map_path);
    bool saveGlobalMap(const std::string& pcd_path);
    bool mapLoaded() const;

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
    bool map_loaded_ = false;
};

}  // namespace n3mapping
