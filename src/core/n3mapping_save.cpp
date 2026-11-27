#include "n3mapping/async_log.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping_session.h"
#include "../continuation_cleaning.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <optional>
#include <set>
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
}
bool N3MappingCore::saveMap(const std::string &map_path,
                            core::LioFrame::PointCloud::ConstPtr *saved_cloud) {
  LogOperation operation("map_save", map_path);
  if (async_loops_ || map_loaded_) {
    LogOperation drain("map_save_drain", map_path);
    const auto loops = processPendingLoopClosures();
    logEvent("map_save_drained", "edges=" + std::to_string(loops.edge_count) +
        " worker_ms=" + std::to_string(loops.worker_wall_ms) +
        " geometry_ms=" + std::to_string(loops.geometry.solve_ms));
    drain.succeed();
  }
  if (!refreshSubmapPoses("save_map")) {
    return false;
  }
  const auto dense_optimized_trajectory = getDenseOptimizedTrajectory();
  core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
  if (!dense_optimized_trajectory.empty() &&
      (metadata.source.empty() || metadata.source == "none")) {
    metadata.source = "native";
    metadata.degraded = false;
  }
  const bool with_atlas = config_.reloc_atlas_enable ||
                         !session_->optimizer.sharedPlanes().empty();

  auto keyframes = session_->keyframes.getAllKeyframes();
  const auto sessions = session_->keyframes.getSessions();
  ContinuationCleaning cleaning;
  std::set<int64_t> changed_clouds;
  std::unique_ptr<GraphOptimizer> refreshed_graph;
  std::unique_ptr<SubmapBuilder> refreshed_submaps;
  std::optional<LocalizationAtlas> refreshed_reference;
  if (map_loaded_ && config_.continuation_penetration_cleaning_enable) {
    const auto start = std::chrono::steady_clock::now();
    cleaning = computeContinuationCleaning(config_, keyframes);
    const auto wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    LOG(INFO) << "[ContinuationCleaning] inspected_keyframes=" << cleaning.inspected_keyframes
              << " modified_keyframes=" << cleaning.clouds.size()
              << " skipped_empty_keyframes=" << cleaning.skipped_empty_keyframes
              << " removed_points=" << cleaning.removed_points
              << " kept_points=" << cleaning.kept_points
              << " wall_ms=" << wall_ms;
  }
  if (!cleaning.clouds.empty()) {
    std::vector<Keyframe::Ptr> loaded_keyframes;
    for (auto& frame : keyframes) {
      const auto replacement = cleaning.clouds.find(frame->id);
      if (replacement != cleaning.clouds.end()) {
        frame = std::make_shared<Keyframe>(*frame);
        frame->cloud = replacement->second;
        changed_clouds.insert(frame->id);
      }
      if (frame->is_from_loaded_map) loaded_keyframes.push_back(frame);
    }
    if (config_.loop_geometry_enable && !session_->optimizer.sharedPlanes().empty()) {
      refreshed_graph = session_->optimizer.cloneCommitted();
      GeometryOptimizationStats stats;
      const auto planes = buildSharedPlanes(keyframes, stats);
      if (!refreshed_graph->replaceSharedPlanes(planes, false, stats)) return false;
    }
    const auto submaps = session_->submaps.getSubmaps();
    if (!submaps.empty()) {
      refreshed_submaps = std::make_unique<SubmapBuilder>(config_);
      if (!refreshed_submaps->loadSubmaps(submaps, keyframes, sessions, changed_clouds))
        return false;
    }
    refreshed_reference.emplace(config_, session_->matcher);
    if (session_->localizer.localizationAtlasLoaded() &&
        !refreshed_reference->prepare(loaded_keyframes)) {
      refreshed_reference->clear();
      LOG(WARNING) << "[ContinuationCleaning] cleaned loaded reference cannot form an Atlas target";
    }
  }
  const auto commit_cleaning = [&] {
    if (changed_clouds.empty()) return;
    session_->keyframes.replaceKeyframeClouds(cleaning.clouds);
    if (refreshed_graph) session_->optimizer.swapWith(*refreshed_graph);
    if (refreshed_submaps) session_->submaps.swapWith(*refreshed_submaps);
    session_->detector.refreshMapDescriptors(session_->keyframes, changed_clouds);
    session_->localizer.installOptimizedReference(
        *refreshed_reference, Eigen::Isometry3d::Identity());
  };

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
  {
    LogOperation serialize("map_save_serialize", map_path);
    if (!session_->serializer.saveMap(
        pending, keyframes, sessions,
        refreshed_graph ? *refreshed_graph : session_->optimizer,
        dense_optimized_trajectory, metadata,
        refreshed_submaps ? refreshed_submaps.get() : &session_->submaps)) return false;
    serialize.succeed();
  }
  if (!with_atlas) {
    if (saved_cloud)
      *saved_cloud = GlobalMapCache::build(keyframes, config_.save_global_map_voxel_size);
    commit_cleaning();
    operation.succeed();
    return true;
  }
  const auto atlas_path = LocalizationAtlas::defaultAtlasPath(map_path);
  const auto pending_atlas = (stage.path / "atlas").string();
  const auto backup_atlas = (stage.path / "atlas.previous").string();
  LocalizationAtlas atlas(config_, session_->matcher);
  LocalizationAtlasStats stats;
  std::string error;
  {
    LogOperation compile("map_save_atlas", map_path);
    if (!atlas.compileAndSave(pending, keyframes,
                            pending_atlas, true, &stats, &error)) {
      LOG(ERROR) << "Map/Atlas save not published: " << error;
      return false;
    }
    compile.succeed();
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
    LOG(ERROR) << "Map/Atlas publication failed: " << e.what();
    return false;
  }
  LOG(INFO) << "[JointGeometrySave] atlas_global_ms=" << stats.global_map_ms
            << " atlas_prepare_ms=" << stats.prepare_ms
            << " atlas_serialize_ms=" << stats.serialize_ms
            << " atlas_bytes=" << stats.sidecar_bytes;
  if (saved_cloud) {
    const double atlas_voxel_size = std::max(1e-3, config_.global_map_voxel_size);
    *saved_cloud = std::abs(config_.save_global_map_voxel_size - atlas_voxel_size) <= 1e-9
        ? atlas.globalMap()
        : GlobalMapCache::build(keyframes, config_.save_global_map_voxel_size);
  }
  commit_cleaning();
  operation.succeed();
  return true;
}

bool N3MappingCore::saveGlobalMap(const std::string &pcd_path) {
  LogOperation operation("map_save_pcd", pcd_path);
  const bool saved = session_->serializer.saveGlobalMap(
      pcd_path, session_->keyframes,
      config_.save_global_map_voxel_size);
  if (saved) operation.succeed();
  return saved;
}

bool N3MappingCore::saveMapSnapshot(std::string *error) {
  if (session_->keyframes.size() < 1) {
    if (error)
      *error = "no_keyframes";
    return false;
  }

  const std::string map_file = config_.map_save_path + "/n3map.pbstream";
  core::LioFrame::PointCloud::ConstPtr saved_cloud;
  if (!saveMap(map_file, config_.save_global_map_on_shutdown ? &saved_cloud : nullptr)) {
    if (error)
      *error = "save_pbstream_failed";
    return false;
  }

  if (config_.save_global_map_on_shutdown) {
    const std::string global_map_file =
        config_.map_save_path + "/global_map.pcd";
    LogOperation pcd("map_save_pcd", global_map_file);
    if (!MapSerializer::saveGlobalMap(global_map_file, *saved_cloud)) {
      if (error)
        *error = "save_global_map_failed";
      return false;
    }
    pcd.succeed();
  }

  return true;
}

core::LioFrame::PointCloud::Ptr N3MappingCore::buildGlobalMap() const {
  if (const auto cloud = compactLocalizationCloud())
    return pcl::make_shared<core::LioFrame::PointCloud>(*cloud);
  return session_->serializer.buildGlobalMap(
      session_->keyframes, config_.global_map_voxel_size);
}

}
