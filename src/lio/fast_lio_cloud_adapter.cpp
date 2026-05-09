#include "n3mapping/lio/fast_lio_cloud_adapter.h"

#include <algorithm>
#include <cmath>

namespace n3mapping {
namespace lio {
namespace fast_lio {
namespace {

bool isFinitePoint(const pcl::PointXYZI& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && std::isfinite(point.intensity);
}

bool isWithinCoordinateRange(const pcl::PointXYZI& point, double max_abs_coordinate) {
    return std::abs(point.x) <= max_abs_coordinate &&
           std::abs(point.y) <= max_abs_coordinate &&
           std::abs(point.z) <= max_abs_coordinate;
}

double squaredRange(const pcl::PointXYZI& point) {
    return static_cast<double>(point.x) * point.x +
           static_cast<double>(point.y) * point.y +
           static_cast<double>(point.z) * point.z;
}

}  // namespace

CloudAdapterResult cloudFromRawLidar(const core::RawLidarFrame& frame,
                                     const CloudAdapterOptions& options) {
    CloudAdapterResult result;
    result.cloud = pcl::make_shared<PointCloud>();
    if (!frame.points) {
        return result;
    }

    const size_t filter_num = std::max<size_t>(1, options.point_filter_num);
    const double blind_sq = options.blind * options.blind;
    result.stats.input_points = frame.points->size();
    result.cloud->reserve(frame.points->size() / filter_num + 1);

    for (size_t i = 0; i < frame.points->size(); ++i) {
        if (filter_num > 1 && i % filter_num != 0) {
            ++result.stats.skipped_by_filter;
            continue;
        }

        const auto& raw = frame.points->at(i);
        if (!isFinitePoint(raw) ||
            !isWithinCoordinateRange(raw, options.max_abs_coordinate)) {
            ++result.stats.skipped_non_finite;
            continue;
        }

        if (options.blind > 0.0 && squaredRange(raw) < blind_sq) {
            ++result.stats.skipped_blind;
            continue;
        }

        if (!frame.point_lines.empty() && i < frame.point_lines.size() &&
            frame.point_lines[i] >= options.scan_lines) {
            ++result.stats.skipped_invalid_line;
            continue;
        }

        PointType point;
        point.x = raw.x;
        point.y = raw.y;
        point.z = raw.z;
        point.intensity = raw.intensity;
        point.normal_x = 0.0f;
        point.normal_y = 0.0f;
        point.normal_z = 0.0f;
        point.curvature =
            i < frame.point_time_offsets_ns.size()
                ? static_cast<float>(frame.point_time_offsets_ns[i]) / 1000000.0f
                : 0.0f;
        result.cloud->push_back(point);
    }

    result.stats.output_points = result.cloud->size();
    result.cloud->width = static_cast<uint32_t>(result.cloud->size());
    result.cloud->height = 1;
    result.cloud->is_dense = true;
    return result;
}

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
