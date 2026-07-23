// Validation and row conversion for the native dense-trajectory CSV exporter.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/n3map_nav_resource_reader.h"

namespace n3mapping {
namespace tools {

struct DenseTrajectoryCsvRow {
    int64_t stamp_ns = 0;
    uint64_t seq = 0;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

inline bool prepareNativeDenseTrajectoryCsvRows(
    const N3NavResource& resource,
    std::vector<DenseTrajectoryCsvRow>* rows,
    std::string* error) {
    const auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!rows) return fail("null dense trajectory CSV row output");
    rows->clear();
    if (!resource.has_native_dense_trajectory ||
        resource.dense_trajectory_source != "native" ||
        resource.dense_trajectory_degraded) {
        return fail("pbstream does not contain a native non-degraded dense trajectory");
    }
    if (resource.dense_optimized_trajectory.empty()) {
        return fail("native dense trajectory is empty");
    }

    rows->reserve(resource.dense_optimized_trajectory.size());
    double previous_timestamp = 0.0;
    int64_t previous_stamp_ns = 0;
    bool have_previous = false;
    for (const auto& sample : resource.dense_optimized_trajectory) {
        if (!std::isfinite(sample.timestamp)) {
            return fail("dense trajectory contains a non-finite timestamp");
        }
        if (!sample.pose_world_lidar.matrix().array().isFinite().all()) {
            return fail("dense trajectory contains a non-finite pose");
        }
        if (have_previous && sample.timestamp <= previous_timestamp) {
            return fail(
                sample.timestamp == previous_timestamp
                    ? "dense trajectory contains a duplicate timestamp"
                    : "dense trajectory timestamps are not strictly increasing");
        }

        const long double scaled =
            static_cast<long double>(sample.timestamp) * 1000000000.0L;
        if (!std::isfinite(scaled) ||
            scaled < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
            scaled > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
            return fail("dense trajectory timestamp is outside int64 nanosecond range");
        }
        const int64_t stamp_ns = std::llround(scaled);
        if (have_previous && stamp_ns <= previous_stamp_ns) {
            return fail(
                stamp_ns == previous_stamp_ns
                    ? "dense trajectory timestamps collide after llround to nanoseconds"
                    : "dense trajectory nanosecond stamps are not strictly increasing");
        }

        Eigen::Quaterniond orientation(sample.pose_world_lidar.rotation());
        if (!orientation.coeffs().array().isFinite().all() ||
            !std::isfinite(orientation.norm()) ||
            orientation.norm() <= 1e-12) {
            return fail("dense trajectory contains an invalid orientation");
        }
        orientation.normalize();

        DenseTrajectoryCsvRow row;
        row.stamp_ns = stamp_ns;
        row.seq = sample.seq;
        row.translation = sample.pose_world_lidar.translation();
        row.orientation = orientation;
        rows->push_back(row);
        previous_timestamp = sample.timestamp;
        previous_stamp_ns = stamp_ns;
        have_previous = true;
    }
    return true;
}

}  // namespace tools
}  // namespace n3mapping
