#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include <Eigen/Geometry>

#include "n3mapping/core/realtime_odometry.h"
#include "n3mapping/core/run_mode.h"
#include "n3mapping/relocalization_output_authority.h"
#include "n3mapping/ros_output_contract.h"

namespace n3mapping::core {

struct OdometrySource {
  int64_t stamp_nsec = 0;
  std::string source_frame;
  std::string body_frame;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  bool pose_valid = true;
};

struct BackendPublication {
  RelocalizationPublicationPlan plan;
  bool live_authorized = false;
};

struct OdometryEmission {
  std::optional<RealtimeOdometryPose> pose;
  RealtimeOdometryDiagnostic diagnostic;
  std::string reason;
  std::string status_json;
};

struct CloudEmission {
  RealtimeOdometryPose projection;
  std::string status_json;
};

// One owner for backend authority, its matching observation snapshot, and the
// live output contract. Registration and cloud conversion run outside it.
class RealtimeOutput {
 public:
  RealtimeOutput(CoreRunMode mode, std::string world_frame, std::size_t retention);

  // Called by the existing single backend owner after processing a paired scan.
  BackendPublication acceptBackend(const OdometrySource& source,
                                   double observation_stamp, BackendOutput& output);

  // The adapter reports whether its standard twist conversion can be published;
  // rejected inputs, projection metadata and scalar clocks are decided here.
  OdometryEmission observeOdometry(const OdometrySource& source, int64_t now_nsec,
                                   bool twist_publishable);
  std::optional<CloudEmission> observeCloud(const OdometrySource& source,
                                           int64_t now_nsec,
                                           uint64_t cloud_stamp_nsec,
                                           const std::string& cloud_frame);
  bool isCurrent(const CloudEmission& cloud) const;

 private:
  ProjectionMetadata metadata(const RealtimeOdometryPose& pose,
                              const std::string& kind) const;

  const CoreRunMode mode_;
  const std::string world_frame_;
  const std::string session_id_;
  RelocalizationOutputAuthority authority_;
  RealtimeOdometry odometry_;
  // Lock order is snapshot -> odometry. No serialization, publication or cloud
  // processing holds the snapshot lock; scan projection never takes it nested.
  std::mutex snapshot_mutex_;
  BackendOutput backend_status_;
  double observation_stamp_ = 0.0;
  std::string scalar_status_json_;
};

}  // namespace n3mapping::core
