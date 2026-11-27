#include "n3mapping/core/map_artifacts.h"
#include "n3mapping/async_log.h"
#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_core.h"
#include <filesystem>
#include "n3mapping/map_serializer.h"

namespace n3mapping {

MapArtifacts::MapArtifacts(const Config &config)
    : map_save_path_(config.map_save_path),
      save_global_map_(config.save_global_map_on_shutdown),
      mode_(parseCoreRunMode(config.mode)), cache_(config.global_map_voxel_size) {}

MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore &core) {
  if (auto cloud = core.compactLocalizationCloud())
    return {{}, core.mapRevision(), std::move(cloud)};
  const auto keyframes = core.getAllKeyframes();
  std::vector<Keyframe::Ptr> snapshots;
  snapshots.reserve(keyframes.size());
  for (const auto &keyframe : keyframes) {
    auto snapshot = std::make_shared<Keyframe>();
    snapshot->id = keyframe->id;
    snapshot->timestamp = keyframe->timestamp;
    snapshot->pose_odom = keyframe->pose_odom;
    snapshot->pose_optimized = keyframe->pose_optimized;
    snapshot->cloud = keyframe->cloud;
    snapshot->is_from_loaded_map = keyframe->is_from_loaded_map;
    snapshots.push_back(std::move(snapshot));
  }
  return {std::move(snapshots), core.mapRevision(), {}};
}

bool MapArtifacts::save(N3MappingCore &core, std::string &error,
                        GlobalMapCache::PointCloudConstPtr &saved_cloud) {
  if (!coreRunModeSavesMap(mode_)) {
    error = std::string("save_disabled_in_mode:") + coreRunModeName(mode_);
    return false;
  }
  if (core.getAllKeyframes().empty()) {
    error = "no_keyframes";
    return false;
  }
  std::error_code directory_error;
  std::filesystem::create_directories(map_save_path_, directory_error);
  if (directory_error) {
    error = "create_map_directory_failed";
    return false;
  }
  if (!core.saveMap(map_save_path_ + "/n3map.pbstream", save_global_map_ ? &saved_cloud : nullptr)) {
    error = "save_pbstream_failed";
    return false;
  }
  return true;
}

void MapArtifacts::saveDebug(const GlobalMapCache::PointCloud &saved_cloud,
                             std::string &warning) {
  warning.clear();
  LogOperation operation("map_save_pcd", map_save_path_);
  std::lock_guard<std::mutex> lock(mutex_);
  if (saved_cloud.empty()) {
    warning = "save_global_map_empty";
    return;
  }
  if (!MapSerializer::saveGlobalMap(map_save_path_ + "/global_map.pcd", saved_cloud)) {
    warning = "save_global_map_failed";
    return;
  }
  operation.succeed();
}

void MapArtifacts::withGlobalCloud(
    const MapKeyframeSnapshot &snapshot,
    const std::function<void(const GlobalCloud &)> &consume) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (snapshot.atlas_cloud) {
    consume({*snapshot.atlas_cloud, snapshot.revision.generation});
    return;
  }
  const auto cloud = cache_.update(snapshot.keyframes, snapshot.revision);
  consume({*cloud, cache_.revision()});
}

void MapArtifacts::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

void MapArtifacts::invalidate() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.markFullRebuildRequired();
}

}
