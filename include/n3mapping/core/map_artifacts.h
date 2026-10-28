#pragma once
#include "n3mapping/config.h"
#include "n3mapping/core/run_mode.h"
#include "n3mapping/core/types.h"
#include "n3mapping/global_map_cache.h"
#include <functional>
#include <mutex>

namespace n3mapping {
class N3MappingCore;
struct MapKeyframeSnapshot {
  std::vector<Keyframe::Ptr> keyframes;
  KeyframeMapRevision revision;
};
// Capture while holding the backend lock; poses are frozen, clouds are immutable.
MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore* core);
bool ensureMapDirectory(const std::string& path);
class MapArtifacts {
public:
  explicit MapArtifacts(const Config& config)
    : config_(config), mode_(parseCoreRunMode(config.mode)), cache_()
  {
    cache_.setVoxelSize(config.global_map_voxel_size);
  }
  // Caller holds the backend lock through save and the post-commit snapshot.
  bool save(N3MappingCore* core, std::string* error, MapKeyframeSnapshot* snapshot);
  // Same backend owner as save(); retain the existing cloud cache.
  void setMode(CoreRunMode mode) { mode_ = mode; }
  void saveDebug(const MapKeyframeSnapshot& snapshot, std::string* warning);
  struct GlobalCloud {
    core::LioFrame::PointCloud::ConstPtr cloud;
    std::uint64_t revision;
  };
  // Cache clouds can be appended in place; consume under the cache lock.
  // The callback must not retain the cloud or re-enter MapArtifacts.
  void withGlobalCloud(const MapKeyframeSnapshot& snapshot,
                       const std::function<void(const GlobalCloud&)>& consume);
  std::uint64_t revision() const;
  void clear();
  void invalidate();
private:
  const Config config_;
  CoreRunMode mode_;
  mutable std::mutex mutex_;
  GlobalMapCache cache_;
};
}  // namespace n3mapping
