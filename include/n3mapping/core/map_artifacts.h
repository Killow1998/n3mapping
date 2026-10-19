#pragma once
#include "n3mapping/config.h"
#include "n3mapping/core/run_mode.h"
#include "n3mapping/core/types.h"
#include "n3mapping/global_map_cache.h"
#include <mutex>
#include <functional>
namespace n3mapping {
class N3MappingCore;
struct MapKeyframeSnapshot {
  std::vector<Keyframe::Ptr> keyframes;
  KeyframeMapRevision revision;
};
// Capture while holding the backend lock; poses are frozen, clouds are immutable.
MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore& core);
bool ensureMapDirectory(const std::string& path);
class MapArtifacts {
public:
  explicit MapArtifacts(const Config& config)
    : config_(config), mode_(parseCoreRunMode(config.mode)), cache_()
  {
    cache_.setVoxelSize(config.global_map_voxel_size);
  }
  bool save(N3MappingCore* core, std::string* error, MapKeyframeSnapshot* snapshot);
  void saveDebug(const MapKeyframeSnapshot& snapshot, std::string* warning);
  struct GlobalCloud {
    core::LioFrame::PointCloud::ConstPtr cloud;
    uint64_t revision;
  };
  void withGlobalCloud(const MapKeyframeSnapshot& snapshot,
                       const std::function<void(const GlobalCloud&)>& consume);
  void clear();
  void invalidate();
private:
  const Config config_;
  const CoreRunMode mode_;
  std::mutex mutex_;
  GlobalMapCache cache_;
};
}  // namespace n3mapping
