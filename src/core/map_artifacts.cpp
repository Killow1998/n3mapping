#include "n3mapping/core/map_artifacts.h"
#include "n3mapping/core/n3mapping_core.h"
#include <sys/stat.h>
#include <cerrno>
#include <pcl/io/pcd_io.h>
namespace n3mapping {
bool ensureMapDirectory(const std::string& path)
{
  if (path.empty()) {
    return false;
  }

  std::string current;
  std::size_t start = 0;
  if (path[0] == '/') {
    current = "/";
    start = 1;
  }

  while (start <= path.size()) {
    const std::size_t end = path.find('/', start);
    const std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!part.empty()) {
      if (!current.empty() && current.back() != '/') {
        current += "/";
      }
      current += part;
      if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore& core)
{
  std::vector<Keyframe::Ptr> snapshots;
  const auto keyframes = core.getAllKeyframes();
  snapshots.reserve(keyframes.size());
  for (const auto& keyframe : keyframes) {
    if (!keyframe) {
      continue;
    }
    auto snapshot = std::make_shared<Keyframe>();
    snapshot->id = keyframe->id;
    snapshot->timestamp = keyframe->timestamp;
    snapshot->pose_odom = keyframe->pose_odom;
    snapshot->pose_optimized = keyframe->pose_optimized;
    snapshot->cloud = keyframe->cloud;
    snapshot->is_from_loaded_map = keyframe->is_from_loaded_map;
    snapshots.push_back(snapshot);
  }
  return { std::move(snapshots), core.mapRevision() };
}

bool MapArtifacts::save(N3MappingCore* core, std::string* error, MapKeyframeSnapshot* snapshot)
{
  if (!coreRunModeSavesMap(mode_)) {
    if (error) {
      *error = std::string("save_disabled_in_mode:") + coreRunModeName(mode_);
    }
    return false;
  }
  if (!core) {
    if (error) {
      *error = "core_unavailable";
    }
    return false;
  }
  if (core->getAllKeyframes().empty()) {
    if (error) {
      *error = "no_keyframes";
    }
    return false;
  }
  if (!ensureMapDirectory(config_.map_save_path)) {
    if (error) {
      *error = "create_map_directory_failed";
    }
    return false;
  }

  const std::string map_file = config_.map_save_path + "/n3map.pbstream";
  if (!core->saveMap(map_file)) {
    if (error) {
      *error = "save_pbstream_failed";
    }
    return false;
  }

  if (config_.save_global_map_on_shutdown && snapshot) {
    *snapshot = snapshotMapKeyframes(*core);
  }
  return true;
}
void MapArtifacts::saveDebug(const MapKeyframeSnapshot& snapshot, std::string* warning)
{
  if (warning) {
    warning->clear();
  }
  if (snapshot.keyframes.empty()) {
    if (warning) {
      *warning = "save_global_map_empty";
    }
    return;
  }
  std::lock_guard<std::mutex> global_map_lock(mutex_);
  const auto cloud = cache_.update(snapshot.keyframes, snapshot.revision);
  if (!cloud || cloud->empty()) {
    if (warning) {
      *warning = "save_global_map_empty";
    }
    return;
  }

  const std::string global_map_file = config_.map_save_path + "/global_map.pcd";
  if (pcl::io::savePCDFileBinary(global_map_file, *cloud) == -1) {
    if (warning) {
      *warning = "save_global_map_failed";
    }
    return;
  }
}

void MapArtifacts::withGlobalCloud(const MapKeyframeSnapshot& snapshot,
                                   const std::function<void(const GlobalCloud&)>& consume)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto cloud = cache_.update(snapshot.keyframes, snapshot.revision);
  consume({ cloud, cache_.revision() });
}
void MapArtifacts::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}
void MapArtifacts::invalidate()
{
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.markFullRebuildRequired();
}
}  // namespace n3mapping
