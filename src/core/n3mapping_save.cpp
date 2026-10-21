#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <glog/logging.h>

namespace n3mapping {
namespace {
struct SaveDirectory {
  std::filesystem::path path;
  ~SaveDirectory() {
    if (!path.empty()) { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
  }
};
} // namespace
bool N3MappingCore::saveMap(const std::string &map_path) {
  if (async_loops_ || (map_loaded_ && config_.loop_geometry_enable)) processPendingLoopClosures();
  if (!refreshSubmapPoses("save_map")) {
    return false;
  }
  const auto dense_optimized_trajectory = buildDenseOptimizedTrajectory();
  core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
  if (!dense_optimized_trajectory.empty() &&
      (metadata.source.empty() || metadata.source == "none")) {
    metadata.source = "native";
    metadata.degraded = false;
  }
  const bool with_atlas = !session_->graphOptimizer().sharedPlanes().empty();
  // Prepare both files before publication. Atlas SHA binding makes even the
  // short interval between renames fail closed for concurrent readers.
  SaveDirectory stage;
  if (with_atlas) {
    auto parent = std::filesystem::path(map_path).parent_path();
    if (parent.empty()) parent = ".";
    const auto name = (parent / ".n3map-save-XXXXXX").string();
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back('\0');
    if (!mkdtemp(buffer.data())) return false;
    stage.path = buffer.data();
  }
  const std::string pending = with_atlas ? (stage.path / "map").string() : map_path;
  if (!session_->mapSerializer().saveMap(
      pending, session_->keyframeManager(), session_->loopDetector(),
      session_->graphOptimizer(), dense_optimized_trajectory, metadata,
      &session_->submapBuilder())) return false;
  if (!with_atlas) return true;
  const auto atlas_path = LocalizationAtlas::defaultAtlasPath(map_path);
  const auto pending_atlas = (stage.path / "atlas").string();
  const auto backup_atlas = (stage.path / "atlas.previous").string();
  LocalizationAtlas atlas(config_, session_->pointCloudMatcher());
  LocalizationAtlasStats stats;
  std::string error;
  if (!atlas.compileAndSave(pending, session_->keyframeManager().getAllKeyframes(),
                            pending_atlas, true, &stats, &error)) {
    LOG(ERROR) << "Map/Atlas save not published: " << error;
    std::filesystem::remove(pending);
    return false;
  }
  bool backed_up = false, published_atlas = false;
  try {
    if (std::filesystem::exists(atlas_path)) {
      std::filesystem::create_hard_link(atlas_path, backup_atlas);
      backed_up = true;
    }
    std::filesystem::rename(pending_atlas, atlas_path);
    published_atlas = true;
    std::filesystem::rename(pending, map_path);
  } catch (const std::exception& e) {
    std::error_code ignored;
    if (backed_up) std::filesystem::rename(backup_atlas, atlas_path, ignored);
    else if (published_atlas) std::filesystem::remove(atlas_path, ignored);
    std::filesystem::remove(pending, ignored);
    std::filesystem::remove(pending_atlas, ignored);
    LOG(ERROR) << "Map/Atlas publication failed: " << e.what();
    return false;
  }
  if (backed_up) std::filesystem::remove(backup_atlas);
  LOG(INFO) << "[JointGeometrySave] atlas_global_ms=" << stats.global_map_ms
            << " atlas_prepare_ms=" << stats.prepare_ms
            << " atlas_serialize_ms=" << stats.serialize_ms
            << " atlas_bytes=" << stats.sidecar_bytes;
  return true;
}

} // namespace n3mapping
