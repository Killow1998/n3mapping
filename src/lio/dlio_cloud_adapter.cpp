#include "n3mapping/lio/dlio_cloud_adapter.h"

#include <cmath>

namespace n3mapping {
namespace lio {
namespace dlio {
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

TimeEncoding resolveTimeEncoding(const core::RawLidarFrame& frame,
                                 TimeEncoding requested) {
    if (requested != TimeEncoding::Auto) {
        return requested;
    }
    if (frame.source_format == "livox_custom") {
        return TimeEncoding::LivoxOffsetNs;
    }
    if (frame.source_format == "velodyne" || frame.source_format == "velodyne_ros") {
        return TimeEncoding::VelodyneOffsetSeconds;
    }
    return TimeEncoding::OusterOffsetNs;
}

uint32_t pointOffsetNs(const core::RawLidarFrame& frame, size_t index) {
    return index < frame.point_time_offsets_ns.size()
               ? frame.point_time_offsets_ns[index]
               : 0u;
}

void setPointTime(Point& point, uint32_t offset_ns, TimeEncoding encoding) {
    switch (encoding) {
        case TimeEncoding::VelodyneOffsetSeconds:
            point.time = static_cast<float>(offset_ns) * 1.0e-9f;
            break;
        case TimeEncoding::LivoxOffsetNs:
            point.timestamp = static_cast<double>(offset_ns);
            break;
        case TimeEncoding::Auto:
        case TimeEncoding::OusterOffsetNs:
            point.t = offset_ns;
            break;
    }
}

}  // namespace

CloudAdapterResult cloudFromRawLidar(const core::RawLidarFrame& frame,
                                     const CloudAdapterOptions& options) {
    CloudAdapterResult result;
    result.cloud = pcl::make_shared<PointCloud>();
    result.resolved_time_encoding =
        resolveTimeEncoding(frame, options.time_encoding);
    if (!frame.points) {
        return result;
    }

    const double blind_sq = options.blind * options.blind;
    result.stats.input_points = frame.points->size();
    result.cloud->reserve(frame.points->size());

    for (size_t i = 0; i < frame.points->size(); ++i) {
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

        Point point;
        point.x = raw.x;
        point.y = raw.y;
        point.z = raw.z;
        point.intensity = raw.intensity;
        setPointTime(point, pointOffsetNs(frame, i), result.resolved_time_encoding);
        result.cloud->push_back(point);
    }

    result.stats.output_points = result.cloud->size();
    result.cloud->width = static_cast<uint32_t>(result.cloud->size());
    result.cloud->height = 1;
    result.cloud->is_dense = true;
    return result;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
