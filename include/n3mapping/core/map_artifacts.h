#pragma once
#include "n3mapping/core/run_mode.h"
#include "n3mapping/global_map_cache.h"
#include <functional>
#include <mutex>
#include <string>

namespace n3mapping {
struct Config;
class N3MappingCore;
struct MapKeyframeSnapshot {
  std::vector<Keyframe::Ptr> keyframes;
  KeyframeMapRevision revision;
};

MapKeyframeSnapshot snapshotMapKeyframes(const N3MappingCore &core);
class MapArtifacts {
public:
  explicit MapArtifacts(const Config &config);

  bool save(N3MappingCore &core, std::string &error,
            MapKeyframeSnapshot &snapshot);

  void setMode(CoreRunMode mode) { mode_ = mode; }
  void saveDebug(const MapKeyframeSnapshot &snapshot, std::string &warning);
  struct GlobalCloud {
    const GlobalMapCache::PointCloud &cloud;
    std::uint64_t revision;
  };

  void withGlobalCloud(const MapKeyframeSnapshot &snapshot,
                       const std::function<void(const GlobalCloud &)> &consume);
  void clear();
  void invalidate();
private:
  const std::string map_save_path_;
  const bool save_global_map_;
  const double save_voxel_size_;
  CoreRunMode mode_;
  std::mutex mutex_;
  GlobalMapCache cache_;
};
}
