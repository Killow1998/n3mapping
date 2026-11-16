#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
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

class RealtimeOutput {
public:
  RealtimeOutput(CoreRunMode mode, std::string world_frame,
                 std::size_t retention);

  void setMode(CoreRunMode mode);

  void startInputMonitoring(double cloud_timeout_sec);
  void observeCloudInput(int64_t stamp_nsec);
  std::string checkInputHealth();

  BackendPublication acceptBackend(const OdometrySource &source,
                                   TimeStamp observation_stamp,
                                   BackendOutput &output);

  OdometryEmission observeOdometry(const OdometrySource &source,
                                   int64_t now_nsec, bool twist_publishable);
  std::optional<CloudEmission> observeCloud(const OdometrySource &source,
                                            int64_t now_nsec,
                                            uint64_t cloud_stamp_nsec,
                                            const std::string &cloud_frame);
  bool isCurrent(const CloudEmission &cloud) const;

private:
  std::string inputFailureReason();
  ProjectionMetadata metadata(const RealtimeOdometryPose &pose,
                              const std::string &kind) const;

  CoreRunMode mode_;
  const std::string world_frame_;
  const std::string session_id_;
  RelocalizationOutputAuthority authority_;
  RealtimeOdometry odometry_;

  std::mutex snapshot_mutex_;
  BackendOutput backend_status_;
  double observation_stamp_ = 0.0;
  std::string scalar_status_json_;
  std::optional<std::chrono::steady_clock::time_point> last_cloud_receive_;
  std::chrono::duration<double> cloud_timeout_{0.5};
  bool input_failed_ = false;
  std::optional<int64_t> recovery_cloud_stamp_nsec_;
  double estimate_stamp_ = 0.0;
  double applied_correction_stamp_ = 0.0;
};

}
