#include "n3mapping/local_map_selector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace n3mapping {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

} // namespace

KeyframeSpatialSearchResult
KeyframeSpatialIndex::radiusSearch(const std::vector<Keyframe::Ptr> &keyframes,
                                   const KeyframeMapRevision &map_revision,
                                   const Eigen::Vector3d &center, double radius,
                                   int max_keyframes) {
  KeyframeSpatialSearchResult result;
  result.map_revision = map_revision;
  if (!center.allFinite() || !std::isfinite(radius) || radius <= 0.0 ||
      max_keyframes <= 0) {
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!have_indexed_revision_ || indexed_revision_ != map_revision) {
    rebuildLocked(keyframes, map_revision, &result.index_build_ms);
    result.index_rebuilt = true;
  }
  if (positions_->empty()) {
    return result;
  }

  const auto query_start = Clock::now();
  pcl::PointXYZ query;
  query.x = static_cast<float>(center.x());
  query.y = static_cast<float>(center.y());
  query.z = static_cast<float>(center.z());
  std::vector<int> indices;
  std::vector<float> squared_distances;
  tree_.radiusSearch(query, radius, indices, squared_distances);
  result.matches.reserve(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const int index = indices[i];
    if (index < 0 || static_cast<std::size_t>(index) >= keyframe_ids_.size()) {
      continue;
    }
    result.matches.push_back({keyframe_ids_[static_cast<std::size_t>(index)],
                              static_cast<double>(squared_distances[i])});
  }
  std::sort(
      result.matches.begin(), result.matches.end(),
      [](const SpatialKeyframeMatch &lhs, const SpatialKeyframeMatch &rhs) {
        if (lhs.squared_distance != rhs.squared_distance) {
          return lhs.squared_distance < rhs.squared_distance;
        }
        return lhs.keyframe_id < rhs.keyframe_id;
      });
  if (result.matches.size() > static_cast<std::size_t>(max_keyframes)) {
    result.matches.resize(static_cast<std::size_t>(max_keyframes));
  }
  result.query_ms = elapsedMs(query_start);
  return result;
}

void KeyframeSpatialIndex::rebuildLocked(
    const std::vector<Keyframe::Ptr> &keyframes,
    const KeyframeMapRevision &map_revision, double *build_ms) {
  const auto start = Clock::now();
  positions_->clear();
  keyframe_ids_.clear();
  positions_->reserve(keyframes.size());
  keyframe_ids_.reserve(keyframes.size());
  for (const auto &keyframe : keyframes) {
    if (!keyframe || !keyframe->pose_optimized.matrix().allFinite()) {
      continue;
    }
    const Eigen::Vector3d position = keyframe->pose_optimized.translation();
    positions_->push_back(pcl::PointXYZ(static_cast<float>(position.x()),
                                        static_cast<float>(position.y()),
                                        static_cast<float>(position.z())));
    keyframe_ids_.push_back(keyframe->id);
  }
  positions_->width = static_cast<std::uint32_t>(positions_->size());
  positions_->height = 1;
  positions_->is_dense = true;
  if (!positions_->empty()) {
    tree_.setInputCloud(positions_);
  }
  indexed_revision_ = map_revision;
  have_indexed_revision_ = true;
  if (build_ms) {
    *build_ms = elapsedMs(start);
  }
}

LocalMapSelection
LocalMapSelector::select(const LocalMapSelectionRequest &request) {
  LocalMapSelection selection;
  if (!request.predicted_pose.matrix().allFinite() ||
      !std::isfinite(request.spatial_radius) || request.spatial_radius <= 0.0 ||
      request.max_spatial_keyframes <= 0 || request.recent_tail_count < 0) {
    selection.reason = "invalid_request";
    return selection;
  }

  KeyframeMapRevision revision_before;
  KeyframeMapRevision revision_after;
  std::vector<Keyframe::Ptr> keyframes;
  for (int attempt = 0; attempt < 2; ++attempt) {
    revision_before = keyframe_manager_.revision();
    keyframes = keyframe_manager_.getAllKeyframes();
    revision_after = keyframe_manager_.revision();
    if (revision_before == revision_after) {
      break;
    }
  }
  selection.map_revision = revision_after;
  selection.requested_revision_matched = request.map_revision == revision_after;

  const KeyframeSpatialSearchResult spatial = spatial_index_.radiusSearch(
      keyframes, revision_after, request.predicted_pose.translation(),
      request.spatial_radius, request.max_spatial_keyframes);
  selection.index_rebuilt = spatial.index_rebuilt;
  selection.index_build_ms = spatial.index_build_ms;
  selection.query_ms = spatial.query_ms;

  std::map<int64_t, Keyframe::Ptr> by_id;
  for (const auto &keyframe : keyframes) {
    if (keyframe) {
      by_id[keyframe->id] = keyframe;
    }
  }

  std::set<int64_t> selected_ids;
  const auto append_id = [&](int64_t id) {
    if (by_id.find(id) != by_id.end() && selected_ids.insert(id).second) {
      selection.keyframe_ids.push_back(id);
    }
  };
  for (const auto &match : spatial.matches) {
    append_id(match.keyframe_id);
  }
  append_id(request.anchor_id);
  int recent_added = 0;
  for (auto it = by_id.rbegin();
       it != by_id.rend() && recent_added < request.recent_tail_count; ++it) {
    const std::size_t size_before = selection.keyframe_ids.size();
    append_id(it->first);
    if (selection.keyframe_ids.size() > size_before) {
      ++recent_added;
    }
  }

  selection.reason = request.recent_tail_count > 0 ? "spatial_anchor_recent"
                                                   : "spatial_anchor";
  double z_min = std::numeric_limits<double>::infinity();
  double z_max = -std::numeric_limits<double>::infinity();
  for (const int64_t id : selection.keyframe_ids) {
    const auto found = by_id.find(id);
    if (found == by_id.end() || !found->second) {
      continue;
    }
    const auto &keyframe = found->second;
    if (keyframe->cloud) {
      selection.selected_points += keyframe->cloud->size();
    }
    if (keyframe->is_from_loaded_map) {
      ++selection.loaded_keyframes;
    } else {
      ++selection.current_keyframes;
    }
    const double z = keyframe->pose_optimized.translation().z();
    z_min = std::min(z_min, z);
    z_max = std::max(z_max, z);
  }
  if (std::isfinite(z_min) && std::isfinite(z_max)) {
    selection.z_min = z_min;
    selection.z_max = z_max;
  }
  return selection;
}

} // namespace n3mapping
