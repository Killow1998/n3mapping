#include "n3mapping/core/map_artifacts.h"
#include "n3mapping/async_log.h"
#include "n3mapping/config.h"
#include "n3mapping/core/n3mapping_core.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>

namespace n3mapping {

MapArtifacts::MapArtifacts(const Config &config)
    : map_save_path_(config.map_save_path),
      save_global_map_(config.save_global_map_on_shutdown),
      save_voxel_size_(config.save_global_map_voxel_size),
      mode_(parseCoreRunMode(config.mode)), cache_(config.global_map_voxel_size) {}

MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore &core) {
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
  return {std::move(snapshots), core.mapRevision()};
}

bool MapArtifacts::save(N3MappingCore &core, std::string &error,
                        MapKeyframeSnapshot &snapshot) {
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
  if (!core.saveMap(map_save_path_ + "/n3map.pbstream")) {
    error = "save_pbstream_failed";
    return false;
  }
  if (save_global_map_)
    snapshot = snapshotMapKeyframes(core);
  return true;
}

void MapArtifacts::saveDebug(const MapKeyframeSnapshot &snapshot,
                             std::string &warning) {
  warning.clear();
  if (snapshot.keyframes.empty()) {
    warning = "save_global_map_empty";
    return;
  }
  LogOperation operation("map_save_pcd", map_save_path_);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto cloud = save_voxel_size_ == cache_.voxelSize()
                         ? cache_.update(snapshot.keyframes, snapshot.revision)
                         : GlobalMapCache::build(snapshot.keyframes, save_voxel_size_);
  if (cloud->empty()) {
    warning = "save_global_map_empty";
    return;
  }
  if (pcl::io::savePCDFileBinary(map_save_path_ + "/global_map.pcd", *cloud) ==
      -1) {
    warning = "save_global_map_failed";
    return;
  }
  operation.succeed();
}

void MapArtifacts::withGlobalCloud(
    const MapKeyframeSnapshot &snapshot,
    const std::function<void(const GlobalCloud &)> &consume) {
  std::lock_guard<std::mutex> lock(mutex_);
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
