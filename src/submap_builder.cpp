#include "n3mapping/submap_builder.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <utility>

#include <glog/logging.h>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

bool finiteKeyframe(const Keyframe::Ptr& keyframe) {
    if (!keyframe || keyframe->id < 0 || !keyframe->cloud ||
        keyframe->cloud->empty() ||
        !keyframe->pose_odom.matrix().allFinite() ||
        !keyframe->pose_optimized.matrix().allFinite()) {
        return false;
    }
    for (const auto& point : keyframe->cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z) || !std::isfinite(point.intensity)) {
            return false;
        }
    }
    return true;
}

bool containsKeyframe(const Submap& submap, int64_t keyframe_id) {
    return std::find(submap.keyframe_ids.begin(), submap.keyframe_ids.end(),
                     keyframe_id) != submap.keyframe_ids.end();
}

struct KeyframePoseSnapshot {
    MapSessionId session_id = kInvalidMapSessionId;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();
};

bool finiteRigidPose(const Eigen::Isometry3d& pose,
                     double tolerance = 1e-6) {
    if (!pose.matrix().allFinite()) return false;
    const Eigen::Matrix3d rotation = pose.linear();
    return (rotation.transpose() * rotation)
               .isApprox(Eigen::Matrix3d::Identity(), tolerance) &&
           std::abs(rotation.determinant() - 1.0) <= tolerance &&
           pose.matrix().row(3).isApprox(
               Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), tolerance);
}

bool snapshotKeyframePoses(
    const std::vector<Keyframe::Ptr>& keyframes,
    std::map<int64_t, KeyframePoseSnapshot>* snapshots,
    std::string* failure_reason) {
    if (!snapshots || !failure_reason) return false;
    snapshots->clear();
    for (const auto& keyframe : keyframes) {
        if (!keyframe || keyframe->id < 0 ||
            keyframe->session_id == kInvalidMapSessionId ||
            !finiteRigidPose(keyframe->pose_odom) ||
            !finiteRigidPose(keyframe->pose_optimized)) {
            *failure_reason = "invalid_keyframe_pose";
            return false;
        }
        KeyframePoseSnapshot snapshot;
        snapshot.session_id = keyframe->session_id;
        snapshot.pose_odom = keyframe->pose_odom;
        snapshot.pose_optimized = keyframe->pose_optimized;
        if (!snapshots->emplace(keyframe->id, std::move(snapshot)).second) {
            *failure_reason = "duplicate_keyframe_id";
            return false;
        }
    }
    return true;
}

SubmapPoseProjectionDiagnostics computePoseProjection(
    const std::vector<Submap>& submaps,
    const std::map<int64_t, KeyframePoseSnapshot>& keyframes,
    bool reanchor,
    std::vector<Eigen::Isometry3d>* refreshed_origins) {
    SubmapPoseProjectionDiagnostics diagnostics;
    diagnostics.submap_count = submaps.size();
    if (reanchor && !refreshed_origins) {
        diagnostics.failure_reason = "missing_refresh_output";
        return diagnostics;
    }
    if (refreshed_origins) {
        refreshed_origins->clear();
        refreshed_origins->reserve(submaps.size());
    }

    std::set<int64_t> assigned_keyframes;
    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    for (const auto& submap : submaps) {
        if (submap.id == kInvalidSubmapId ||
            submap.session_id == kInvalidMapSessionId ||
            submap.keyframe_ids.empty() ||
            !finiteRigidPose(submap.T_session_submap) ||
            !finiteRigidPose(submap.T_map_submap)) {
            diagnostics.failure_reason = "invalid_submap_pose";
            return diagnostics;
        }

        const int64_t anchor_id = submap.descriptor_keyframe_id >= 0
            ? submap.descriptor_keyframe_id
            : submap.keyframe_ids.front();
        if (!containsKeyframe(submap, anchor_id)) {
            diagnostics.failure_reason = "invalid_submap_anchor";
            return diagnostics;
        }
        const auto anchor = keyframes.find(anchor_id);
        if (anchor == keyframes.end() ||
            anchor->second.session_id != submap.session_id) {
            diagnostics.failure_reason = "missing_submap_anchor";
            return diagnostics;
        }

        Eigen::Isometry3d T_map_submap = submap.T_map_submap;
        if (reanchor) {
            const Eigen::Isometry3d T_submap_anchor =
                submap.T_session_submap.inverse() *
                anchor->second.pose_odom;
            T_map_submap = anchor->second.pose_optimized *
                           T_submap_anchor.inverse();
            if (!finiteRigidPose(T_map_submap)) {
                diagnostics.failure_reason = "invalid_refreshed_origin";
                return diagnostics;
            }
            refreshed_origins->push_back(T_map_submap);
            if (!T_map_submap.matrix().isApprox(
                    submap.T_map_submap.matrix(), 1e-12)) {
                ++diagnostics.refreshed_submap_count;
            }
        }

        for (const int64_t keyframe_id : submap.keyframe_ids) {
            const auto keyframe = keyframes.find(keyframe_id);
            if (keyframe == keyframes.end() ||
                keyframe->second.session_id != submap.session_id) {
                diagnostics.failure_reason = "missing_submap_keyframe";
                return diagnostics;
            }
            if (!assigned_keyframes.insert(keyframe_id).second) {
                diagnostics.failure_reason = "duplicate_submap_membership";
                return diagnostics;
            }

            const Eigen::Isometry3d T_submap_keyframe =
                submap.T_session_submap.inverse() *
                keyframe->second.pose_odom;
            const Eigen::Isometry3d projected_pose =
                T_map_submap * T_submap_keyframe;
            const Eigen::Isometry3d residual =
                keyframe->second.pose_optimized.inverse() * projected_pose;
            if (!finiteRigidPose(T_submap_keyframe) ||
                !finiteRigidPose(projected_pose) ||
                !finiteRigidPose(residual)) {
                diagnostics.failure_reason = "invalid_projected_pose";
                return diagnostics;
            }
            const double translation_residual =
                residual.translation().norm();
            const double rotation_residual = std::abs(
                Eigen::AngleAxisd(residual.rotation()).angle());
            if (!std::isfinite(translation_residual) ||
                !std::isfinite(rotation_residual)) {
                diagnostics.failure_reason = "nonfinite_projection_residual";
                return diagnostics;
            }
            translation_sum += translation_residual;
            rotation_sum += rotation_residual;
            diagnostics.max_translation_residual_m = std::max(
                diagnostics.max_translation_residual_m,
                translation_residual);
            diagnostics.max_rotation_residual_rad = std::max(
                diagnostics.max_rotation_residual_rad,
                rotation_residual);
            ++diagnostics.projected_keyframe_count;
        }
    }

    diagnostics.unassigned_keyframe_count =
        keyframes.size() - assigned_keyframes.size();
    if (diagnostics.projected_keyframe_count > 0) {
        const double denominator = static_cast<double>(
            diagnostics.projected_keyframe_count);
        diagnostics.mean_translation_residual_m =
            translation_sum / denominator;
        diagnostics.mean_rotation_residual_rad = rotation_sum / denominator;
    }
    diagnostics.valid = true;
    return diagnostics;
}

}  // namespace

std::uint64_t Submap::materializedCloudBytes() const {
    if (!registration_cloud) return 0;
    // registration_cloud and visualization_cloud intentionally alias in this
    // scaffold, so count the allocation once.
    return static_cast<std::uint64_t>(registration_cloud->size()) *
           sizeof(PointCloudT::PointType);
}

SubmapBuilder::SubmapBuilder(const Config& config)
    : SubmapBuilder(SubmapBuilderOptions{
          config.submap_shadow_enable,
          static_cast<std::size_t>(std::max(1, config.submap_max_keyframes)),
          static_cast<std::uint64_t>(
              std::max(0, config.submap_cloud_max_bytes))}) {}

SubmapBuilder::SubmapBuilder(SubmapBuilderOptions options)
    : options_(std::move(options)) {}

bool SubmapBuilder::appendKeyframe(const Keyframe::Ptr& keyframe) {
    if (!options_.enable || !finiteKeyframe(keyframe) ||
        options_.max_keyframes == 0 ||
        options_.cloud_max_bytes < sizeof(pcl::PointXYZI)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& submap : submaps_) {
        if (containsKeyframe(submap, keyframe->id)) return false;
    }

    if (!submaps_.empty() && !submaps_.back().closed &&
        submaps_.back().session_id != keyframe->session_id) {
        closeActiveSubmapNoLock();
    }
    if (submaps_.empty() || submaps_.back().closed) {
        if (next_id_ == kInvalidSubmapId) return false;
        Submap submap;
        submap.id = next_id_++;
        submap.session_id = keyframe->session_id;
        submap.T_session_submap = keyframe->pose_odom;
        submap.T_map_submap = keyframe->pose_optimized;
        submap.descriptor_keyframe_id = keyframe->id;
        submap.cloud_byte_budget = options_.cloud_max_bytes;
        submap.registration_cloud = pcl::make_shared<Submap::PointCloudT>();
        submap.visualization_cloud = submap.registration_cloud;
        submaps_.push_back(std::move(submap));
    }

    Submap& active = submaps_.back();
    if (active.closed || active.session_id != keyframe->session_id ||
        !appendCloudNoLock(&active, keyframe)) {
        return false;
    }
    active.keyframe_ids.push_back(keyframe->id);
    ++active.content_revision;
    if (active.keyframe_ids.size() >= options_.max_keyframes) {
        closeActiveSubmapNoLock();
    }

    VLOG(1) << "[SubmapShadow] id=" << active.id
            << " session=" << active.session_id
            << " keyframes=" << active.keyframe_ids.size()
            << " points=" << active.cloud_point_count
            << " bytes=" << active.materializedCloudBytes()
            << " budget=" << active.cloud_byte_budget
            << " truncated=" << active.cloud_truncated
            << " closed=" << active.closed
            << " revision=" << active.content_revision;
    return true;
}

bool SubmapBuilder::appendCloudNoLock(Submap* submap,
                                      const Keyframe::Ptr& keyframe) {
    if (!submap || !finiteKeyframe(keyframe) || submap->closed) return false;
    if (!submap->registration_cloud) {
        submap->registration_cloud = pcl::make_shared<Submap::PointCloudT>();
        submap->visualization_cloud = submap->registration_cloud;
    }
    const std::uint64_t max_points =
        submap->cloud_byte_budget / sizeof(pcl::PointXYZI);
    const Eigen::Matrix4f T_submap_keyframe =
        (submap->T_session_submap.inverse() * keyframe->pose_odom)
            .matrix().cast<float>();
    auto& cloud = *submap->registration_cloud;
    for (const auto& point : keyframe->cloud->points) {
        if (cloud.size() >= max_points) {
            submap->cloud_truncated = true;
            break;
        }
        const Eigen::Vector4f transformed =
            T_submap_keyframe * Eigen::Vector4f(point.x, point.y, point.z, 1.0f);
        pcl::PointXYZI output;
        output.x = transformed.x();
        output.y = transformed.y();
        output.z = transformed.z();
        output.intensity = point.intensity;
        cloud.push_back(output);
    }
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;
    submap->visualization_cloud = submap->registration_cloud;
    submap->cloud_point_count = cloud.size();
    return true;
}

bool SubmapBuilder::closeActiveSubmap() {
    std::lock_guard<std::mutex> lock(mutex_);
    return closeActiveSubmapNoLock();
}

bool SubmapBuilder::closeActiveSubmapNoLock() {
    if (submaps_.empty() || submaps_.back().closed) return false;
    submaps_.back().closed = true;
    ++submaps_.back().content_revision;
    return true;
}

bool SubmapBuilder::loadSubmaps(
    const std::vector<Submap>& submaps,
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::vector<MapSessionInfo>& sessions) {
    std::map<int64_t, Keyframe::Ptr> keyframe_by_id;
    for (const auto& keyframe : keyframes) {
        if (keyframe) keyframe_by_id[keyframe->id] = keyframe;
    }
    std::set<MapSessionId> session_ids;
    for (const auto& session : sessions) session_ids.insert(session.id);

    std::set<SubmapId> submap_ids;
    std::set<int64_t> assigned_keyframes;
    bool saw_open = false;
    SubmapId previous_id = 0;
    bool have_previous = false;
    for (std::size_t index = 0; index < submaps.size(); ++index) {
        const auto& submap = submaps[index];
        if (submap.id == kInvalidSubmapId ||
            !submap_ids.insert(submap.id).second ||
            (have_previous && submap.id <= previous_id) ||
            session_ids.find(submap.session_id) == session_ids.end() ||
            submap.keyframe_ids.empty() ||
            !submap.T_session_submap.matrix().allFinite() ||
            !submap.T_map_submap.matrix().allFinite() ||
            submap.cloud_byte_budget < sizeof(pcl::PointXYZI) ||
            submap.cloud_point_count >
                submap.cloud_byte_budget / sizeof(pcl::PointXYZI) ||
            submap.content_revision == 0) {
            return false;
        }
        if (!submap.closed) {
            if (saw_open || index + 1 != submaps.size()) return false;
            saw_open = true;
        }
        for (const int64_t keyframe_id : submap.keyframe_ids) {
            const auto keyframe = keyframe_by_id.find(keyframe_id);
            if (keyframe == keyframe_by_id.end() || !keyframe->second ||
                keyframe->second->session_id != submap.session_id ||
                !assigned_keyframes.insert(keyframe_id).second) {
                return false;
            }
        }
        if (submap.descriptor_keyframe_id >= 0 &&
            !containsKeyframe(submap, submap.descriptor_keyframe_id)) {
            return false;
        }
        previous_id = submap.id;
        have_previous = true;
    }
    if (!submaps.empty() &&
        submaps.back().id == kInvalidSubmapId - 1u) {
        return false;
    }

    // Build and validate the replacement off to the side. A failed load must
    // leave the currently active builder untouched.
    std::vector<Submap> loaded = submaps;
    if (options_.enable) {
        std::map<int64_t, KeyframePoseSnapshot> pose_snapshots;
        std::string failure_reason;
        if (!snapshotKeyframePoses(
                keyframes, &pose_snapshots, &failure_reason)) {
            return false;
        }
        std::vector<Eigen::Isometry3d> refreshed_origins;
        const auto diagnostics = computePoseProjection(
            loaded, pose_snapshots, true, &refreshed_origins);
        if (!diagnostics.valid ||
            refreshed_origins.size() != loaded.size()) {
            return false;
        }
        for (std::size_t index = 0; index < loaded.size(); ++index) {
            loaded[index].T_map_submap = refreshed_origins[index];
        }
    }
    for (auto& submap : loaded) {
        submap.registration_cloud.reset();
        submap.visualization_cloud.reset();
        if (options_.enable &&
            !materializeNoLock(&submap, keyframes, true)) {
            return false;
        }
    }
    const SubmapId next_id = loaded.empty() ? 0 : loaded.back().id + 1u;

    std::lock_guard<std::mutex> lock(mutex_);
    submaps_ = std::move(loaded);
    next_id_ = next_id;
    return true;
}

bool SubmapBuilder::materialize(
    SubmapId id, const std::vector<Keyframe::Ptr>& keyframes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto submap = std::find_if(
        submaps_.begin(), submaps_.end(),
        [id](const Submap& candidate) { return candidate.id == id; });
    return submap != submaps_.end() &&
           materializeNoLock(&*submap, keyframes, true);
}

bool SubmapBuilder::materializeNoLock(
    Submap* submap, const std::vector<Keyframe::Ptr>& keyframes,
    bool verify_serialized_metadata) {
    if (!submap) return false;
    std::map<int64_t, Keyframe::Ptr> keyframe_by_id;
    for (const auto& keyframe : keyframes) {
        if (keyframe) keyframe_by_id[keyframe->id] = keyframe;
    }
    const std::uint64_t expected_points = submap->cloud_point_count;
    const bool expected_truncated = submap->cloud_truncated;
    Submap materialized = *submap;
    const bool was_closed = materialized.closed;
    materialized.registration_cloud =
        pcl::make_shared<Submap::PointCloudT>();
    materialized.visualization_cloud = materialized.registration_cloud;
    materialized.cloud_point_count = 0;
    materialized.cloud_truncated = false;
    // Rebuilding the derived cache does not change the logical closed state or
    // content revision. The original object remains untouched until success.
    materialized.closed = false;
    for (const int64_t keyframe_id : materialized.keyframe_ids) {
        const auto keyframe = keyframe_by_id.find(keyframe_id);
        if (keyframe == keyframe_by_id.end() ||
            !appendCloudNoLock(&materialized, keyframe->second)) {
            return false;
        }
    }
    materialized.closed = was_closed;
    if (verify_serialized_metadata &&
        (materialized.cloud_point_count != expected_points ||
         materialized.cloud_truncated != expected_truncated)) {
        return false;
    }
    *submap = std::move(materialized);
    return true;
}

SubmapPoseProjectionDiagnostics SubmapBuilder::refreshMapPoses(
    const std::vector<Keyframe::Ptr>& keyframes) {
    std::map<int64_t, KeyframePoseSnapshot> pose_snapshots;
    SubmapPoseProjectionDiagnostics diagnostics;
    if (!snapshotKeyframePoses(
            keyframes, &pose_snapshots, &diagnostics.failure_reason)) {
        return diagnostics;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Eigen::Isometry3d> refreshed_origins;
    diagnostics = computePoseProjection(
        submaps_, pose_snapshots, true, &refreshed_origins);
    if (!diagnostics.valid ||
        refreshed_origins.size() != submaps_.size()) {
        return diagnostics;
    }
    for (std::size_t index = 0; index < submaps_.size(); ++index) {
        submaps_[index].T_map_submap = refreshed_origins[index];
    }
    return diagnostics;
}

SubmapPoseProjectionDiagnostics SubmapBuilder::evaluatePoseProjection(
    const std::vector<Keyframe::Ptr>& keyframes) const {
    std::map<int64_t, KeyframePoseSnapshot> pose_snapshots;
    SubmapPoseProjectionDiagnostics diagnostics;
    if (!snapshotKeyframePoses(
            keyframes, &pose_snapshots, &diagnostics.failure_reason)) {
        return diagnostics;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    return computePoseProjection(submaps_, pose_snapshots, false, nullptr);
}

std::vector<Submap> SubmapBuilder::getSubmaps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return submaps_;
}

void SubmapBuilder::swapWith(SubmapBuilder& other) {
    if (this == &other) return;
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> this_lock(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> other_lock(other.mutex_, std::adopt_lock);
    std::swap(options_, other.options_);
    std::swap(submaps_, other.submaps_);
    std::swap(next_id_, other.next_id_);
}

void SubmapBuilder::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    submaps_.clear();
    next_id_ = 0;
}

}  // namespace n3mapping
