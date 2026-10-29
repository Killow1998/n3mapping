#include "localization_atlas_codec.h"
#include "localization_atlas.pb.h"
#include "compact_kdtree_builder.h"
#include "registration_bounds.h"
#include "point_cloud_target_view.h"
#include <cmath>
#include <limits>

namespace n3mapping::atlas_codec {
namespace {
void setError(std::string* error, const std::string& message) {
    if (error) *error = message;
}
}  // namespace

bool encodeLevel(const PointCloudMatcher::PreparedTargetLevel& input,
                 LocalizationAtlasPreparedLevel& output,
                 std::string* error) {
    if (!input.cloud || !input.kdtree || input.cloud->empty()) {
        setError(error, "invalid prepared target level");
        return false;
    }
    if (input.cloud->size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 9)) {
        setError(error, "prepared target level exceeds protobuf array limits");
        return false;
    }
    output.set_resolution(input.resolution);
    output.set_num_points(input.cloud->size());
    output.mutable_points()->Reserve(static_cast<int>(input.cloud->size() * 3));
    output.mutable_normals()->Reserve(static_cast<int>(input.cloud->size() * 3));
    const bool derived_covariance = input.cloud->covs.empty();
    output.set_covariance_from_normal(derived_covariance);
    if (!derived_covariance)
        output.mutable_covariances()->Reserve(static_cast<int>(input.cloud->size() * 9));
    for (std::size_t i = 0; i < input.cloud->size(); ++i) {
        const auto& point = input.cloud->point(i);
        const auto& normal = input.cloud->normal(i);
        const Eigen::Matrix4d covariance = derived_covariance
            ? Eigen::Matrix4d::Zero() : input.cloud->cov(i);
        if (!point.allFinite() || !normal.allFinite() || !covariance.allFinite()) {
            setError(error, "prepared target contains non-finite data");
            return false;
        }
        // small_gicp uses homogeneous storage. Only the spatial components
        // carry information; reject unexpected padding instead of discarding it.
        if (point[3] != 1.0 || normal[3] != 0.0 ||
            !covariance.row(3).isZero(0.0) || !covariance.col(3).isZero(0.0)) {
            setError(error, "prepared target has non-canonical homogeneous padding");
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            output.add_points(point[axis]);
            output.add_normals(normal[axis]);
        }
        for (int row = 0; !derived_covariance && row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                output.add_covariances(covariance(row, column));
            }
        }
    }
    return true;
}

namespace {
Eigen::Matrix4d decodedCovariance(const LocalizationAtlasPreparedLevel& input,
                                  std::size_t index) {
    Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            covariance(row, column) = input.covariances(
                static_cast<int>(index * 9 + row * 3 + column));
    return covariance;
}

}  // namespace

bool decodeLevel(LocalizationAtlasPreparedLevel& input,
                 PointCloudMatcher::PreparedTargetLevel& output,
                 bool with_covariances,
                 std::string* error) {
    if (input.num_points() == 0 || !std::isfinite(input.resolution()) ||
        input.resolution() <= 0.0) {
        setError(error, "invalid atlas prepared level metadata");
        return false;
    }
    if (input.num_points() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 9) ||
        input.points_size() != static_cast<int>(input.num_points() * 3) ||
        input.normals_size() != static_cast<int>(input.num_points() * 3) ||
        input.covariances_size() != (input.covariance_from_normal() ? 0 :
            static_cast<int>(input.num_points() * 9))) {
        setError(error, "atlas prepared level array size mismatch");
        return false;
    }

    auto cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>();
    cloud->points.resize(static_cast<std::size_t>(input.num_points()));
    cloud->normals.resize(cloud->size());
    bool compact_covariances = with_covariances;
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        cloud->point(i)[3] = 1.0;
        cloud->normal(i)[3] = 0.0;
        const Eigen::Matrix4d covariance = input.covariance_from_normal()
            ? Eigen::Matrix4d::Zero() : decodedCovariance(input, i);
        for (int axis = 0; axis < 3; ++axis) {
            cloud->point(i)[axis] = input.points(static_cast<int>(i * 3 + axis));
            cloud->normal(i)[axis] = input.normals(static_cast<int>(i * 3 + axis));
        }
        if (!cloud->point(i).allFinite() || !cloud->normal(i).allFinite() ||
            !covariance.allFinite()) {
            setError(error, "atlas prepared level contains non-finite data");
            return false;
        }
        if (with_covariances && !input.covariance_from_normal()) {
            if (compact_covariances && !covariance.isApprox(
                    registration_detail::covarianceFromNormal(cloud->normal(i)), 1e-12)) {
                // Only a nonstandard covariance needs storage. Backfill the
                // exact serialized matrices, including any earlier entries.
                compact_covariances = false;
                cloud->covs.resize(cloud->size());
                for (std::size_t j = 0; j < i; ++j)
                    cloud->cov(j) = decodedCovariance(input, j);
            }
            if (!compact_covariances) cloud->cov(i) = covariance;
        }
    }
    output.resolution = input.resolution();
    output.cloud = cloud;
    // The validated arrays now belong to cloud. Release the serialized copy
    // before building the tree or decoding the next level; Clear retains it.
    LocalizationAtlasPreparedLevel{}.Swap(&input);
    output.kdtree = std::make_shared<PointCloudMatcher::SmallGicpKdTree>(
        cloud, registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilder>(*cloud, true));
    output.search_bounds = std::make_shared<registration_detail::SearchBounds>(*output.kdtree);
    return true;
}

}  // namespace n3mapping::atlas_codec
