#include "n3mapping/core/realtime_output.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>
#include <utility>

namespace n3mapping::core {
namespace {

std::string projectionSessionId() {
  std::random_device random;
  std::ostringstream id;
  id << std::hex << std::setfill('0');
  for (int i = 0; i < 4; ++i) id << std::setw(8) << random();
  return id.str();
}

double sourceSeconds(uint64_t stamp_nsec) {
  return static_cast<double>(stamp_nsec / 1000000000ULL) +
         static_cast<double>(stamp_nsec % 1000000000ULL) * 1e-9;
}

}  // namespace

RealtimeOutput::RealtimeOutput(CoreRunMode mode, std::string world_frame,
                               std::size_t retention)
    : mode_(mode), world_frame_(std::move(world_frame)),
      session_id_(projectionSessionId()), odometry_(retention) {}

void RealtimeOutput::setMode(CoreRunMode mode) {
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  mode_ = mode;
  scalar_status_json_.clear();
}

BackendPublication RealtimeOutput::acceptBackend(
    const OdometrySource& source, double observation_stamp, BackendOutput& output) {
  const bool localization = mode_ == CoreRunMode::LOCALIZATION;
  const auto output_mode = localization
      ? RelocalizationOutputMode::LOCALIZATION
      : (mode_ == CoreRunMode::MAP_EXTENSION
             ? RelocalizationOutputMode::MAP_EXTENSION
             : RelocalizationOutputMode::OTHER);
  BackendPublication publication;
  publication.plan = authority_.process(output_mode, output);
  publication.live_authorized = publication.plan.publish_global_pose &&
      (localization ? hasAuthoritativeRelocalizationInitializationPose(
          output.relocalization_state, output.pose_source)
                    : (output.success || output.accepted_keyframe)) && source.pose_valid;
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  odometry_.updateCorrection(source.stamp_nsec, source.source_frame, source.body_frame,
                            source.pose, output.T_world_lidar, publication.live_authorized);
  if (output.mapping_block_reason == MappingBlockReason::OdometryDiverged) {
    odometry_.invalidateForOdometryDivergence(source.stamp_nsec);
  }
  backend_status_ = output;
  backend_status_.cloud_body.reset();
  backend_status_.cloud_world.reset();
  observation_stamp_ = observation_stamp;
  return publication;
}

OdometryEmission RealtimeOutput::observeOdometry(
    const OdometrySource& source, int64_t now_nsec, bool twist_publishable) {
  OdometryEmission emission;
  BackendOutput backend;
  CoreRunMode mode;
  RealtimeLocalizationStatus realtime;
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    emission.pose = odometry_.project(source.stamp_nsec, now_nsec,
        source.source_frame, source.body_frame, source.pose, source.pose_valid,
        &emission.diagnostic);
    backend = backend_status_;
    mode = mode_;
    realtime.observation_stamp = observation_stamp_;
  }
  auto& diagnostic = emission.diagnostic;
  emission.reason = realtimeOdometryReasonName(diagnostic.reason);
  realtime.estimate_attempted = diagnostic.correction_stamp_nsec > 0;
  if (emission.pose) {
    if (!twist_publishable) {
      emission.reason = "invalid_twist_or_body_frame";
      diagnostic.generation = odometry_.invalidateUnpublishableInput(
          std::min(diagnostic.input_stamp_nsec, diagnostic.now_nsec));
      emission.pose.reset();
    } else {
      realtime.estimate_stamp = sourceSeconds(source.stamp_nsec);
      realtime.projection = metadata(*emission.pose, "odom");
    }
  }
  if (diagnostic.reason == RealtimeOdometryReason::DuplicateInput ||
      diagnostic.reason == RealtimeOdometryReason::InputBeforeCorrection) {
    return emission;
  }
  if (!emission.pose && diagnostic.reason != RealtimeOdometryReason::ProjectionExpired) {
    ProjectionMetadata unavailable;
    unavailable.kind = "odom";
    unavailable.available = false;
    unavailable.session_id = session_id_;
    unavailable.generation = diagnostic.generation;
    unavailable.odom_stamp_nsec = source.stamp_nsec;
    unavailable.world_frame = world_frame_;
    unavailable.source_frame = source.source_frame;
    unavailable.body_frame = source.body_frame;
    realtime.projection = unavailable;
  }
  realtime.correction_stamp = diagnostic.correction_stamp_nsec * 1e-9;
  realtime.estimate_available = emission.pose.has_value();
  realtime.input_reason = emission.reason;
  emission.status_json = localizationStatusJson(
      mode, backend, sourceSeconds(source.stamp_nsec), world_frame_, &realtime);
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    if (mode == mode_) scalar_status_json_ = emission.status_json;
  }
  return emission;
}

std::optional<CloudEmission> RealtimeOutput::observeCloud(
    const OdometrySource& source, int64_t now_nsec, uint64_t cloud_stamp_nsec,
    const std::string& cloud_frame) {
  if (!source.pose_valid) return std::nullopt;
  const auto projected = odometry_.projectScan(source.stamp_nsec, now_nsec,
      source.source_frame, source.body_frame, source.pose);
  if (!projected) return std::nullopt;
  std::string scalar_status;
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    scalar_status = scalar_status_json_;
  }
  if (scalar_status.empty()) {
    RealtimeLocalizationStatus unavailable;
    BackendOutput backend;
    CoreRunMode mode;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      backend = backend_status_;
      mode = mode_;
      unavailable.observation_stamp = observation_stamp_;
    }
    scalar_status = localizationStatusJson(
        mode, backend, sourceSeconds(cloud_stamp_nsec), world_frame_, &unavailable);
  }
  auto projection = metadata(*projected, "cloud");
  projection.cloud_stamp_nsec = cloud_stamp_nsec;
  projection.cloud_frame = cloud_frame;
  if (!odometry_.isCurrent(*projected)) return std::nullopt;
  auto status = appendProjectionMetadata(scalar_status, projection);
  if (status.empty()) return std::nullopt;
  return CloudEmission{*projected, std::move(status)};
}

bool RealtimeOutput::isCurrent(const CloudEmission& cloud) const {
  return odometry_.isCurrent(cloud.projection);
}

ProjectionMetadata RealtimeOutput::metadata(
    const RealtimeOdometryPose& pose, const std::string& kind) const {
  ProjectionMetadata result;
  result.kind = kind;
  result.session_id = session_id_;
  result.generation = pose.generation;
  result.basis_revision = pose.basis_revision;
  result.world_frame = world_frame_;
  result.source_frame = pose.source_frame;
  result.body_frame = pose.body_frame;
  result.odom_stamp_nsec = pose.stamp_nsec;
  result.map_from_source = pose.map_from_source;
  result.source_from_body = pose.source_from_body;
  return result;
}

}  // namespace n3mapping::core
