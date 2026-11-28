#include "n3mapping/relocalizability_analyzer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>

#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/pcl_compat.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/rotation_utils.h"
#include "registration_planar_geometry.h"
#include "third_party/nanoflann.hpp"

namespace n3mapping {
namespace {

constexpr double kLocalRadius = 8.0;
constexpr double kLocalVertical = 1.8;
constexpr std::size_t kSubmapFrameLimit = 8;
constexpr std::size_t kMinAdaptiveInlierNeighbors = 3;

struct DescriptorView {
  std::shared_ptr<const RHPDPlace> place;
  RHPDHeadingPtr heading;
  double yaw = 0.0;
};

struct KeyframePositions {
  const std::vector<Keyframe::Ptr>& frames;
  std::size_t kdtree_get_point_count() const { return frames.size(); }
  double kdtree_get_pt(std::size_t index, std::size_t axis) const {
    return frames[index]->pose_optimized.translation()[axis];
  }
  template <class Box> bool kdtree_get_bbox(Box&) const { return false; }
};

bool localPeer(const Eigen::Vector3d& position, const Eigen::Vector3d& other,
               const FreeSpaceGrid& free_space) {
  const Eigen::Vector3d delta = other - position;
  return delta.head<2>().squaredNorm() <= kLocalRadius * kLocalRadius &&
      std::abs(delta.z()) <= kLocalVertical && !free_space.hasOccupiedBarrier(position, other);
}

void evaluateGeometry(RelocalizabilityKeyframe& result, const Keyframe& query,
                      const std::vector<Keyframe::Ptr>& frames,
                      std::vector<std::pair<double, std::size_t>>& neighbors,
                      PointCloudMatcher& matcher, double resolution) {
  const auto retained = std::min(neighbors.size(), kSubmapFrameLimit);
  result.local_submap_keyframes = retained;
  if (retained == 0 || !query.cloud || query.cloud->empty()) return;
  std::partial_sort(neighbors.begin(), neighbors.begin() + retained, neighbors.end());

  auto target = pcl::make_shared<LocalizationAtlas::PointCloudT>();
  const auto T_query_map = query.pose_optimized.inverse();
  for (std::size_t index = 0; index < retained; ++index) {
    const auto& frame = *frames[neighbors[index].second];
    const Eigen::Matrix4f transform =
        (T_query_map * frame.pose_optimized).matrix().cast<float>();
    LocalizationAtlas::PointCloudT aligned;
    pcl::transformPointCloud(*frame.cloud, aligned, transform);
    *target += aligned;
  }

  const auto geometry = matcher.evaluatePreparedPlaneGeometry(
      matcher.prepareTargetCloud(target, PointCloudMatcher::TargetFeatures::OnDemand),
      matcher.prepareSourceCloud(query.cloud), Eigen::Isometry3d::Identity(),
      {small_gicp::RegistrationSetting::PLANE_ICP, resolution, resolution});
  result.num_inliers = geometry.num_inliers;
  result.inlier_ratio = geometry.inlier_ratio;
  if (geometry.termination != MatchTermination::FixedPoseEvaluation ||
      geometry.num_inliers == 0 || !geometry.information.allFinite()) return;

  const Eigen::Matrix3d world_in_body = query.pose_optimized.rotation().transpose();
  Eigen::Matrix<double, 6, 3> basis = Eigen::Matrix<double, 6, 3>::Zero();
  basis.block<3, 2>(0, 0) = world_in_body.leftCols<2>();
  basis.block<3, 1>(3, 2) = world_in_body.col(2);
  const Eigen::Matrix3d raw = basis.transpose() * geometry.information * basis;
  const Eigen::Matrix3d planar = (raw + raw.transpose()) / (2.0 * geometry.num_inliers);
  const double planar_minimum = minimumPlanarRegistrationInformation(
      geometry.information, world_in_body.col(2));
  const double tolerance = 6.0 * std::numeric_limits<double>::epsilon() *
                           geometry.information.cwiseAbs().maxCoeff();
  if (!std::isfinite(planar_minimum) || planar_minimum < -tolerance) return;
  result.planar_information = planar;
  result.minimum_planar_information_per_inlier =
      std::max(0.0, planar_minimum) / geometry.num_inliers;
  Eigen::Matrix2d marginal = planar.topLeftCorner<2, 2>();
  if (planar(2, 2) > std::numeric_limits<double>::epsilon()) {
    const Eigen::Vector2d xy_yaw = planar.topRightCorner<2, 1>();
    marginal.noalias() -= xy_yaw * xy_yaw.transpose() / planar(2, 2);
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> spectrum(marginal);
  if (spectrum.info() != Eigen::Success ||
      spectrum.eigenvalues()[0] < -tolerance / geometry.num_inliers) {
    result.planar_information.reset();
    result.minimum_planar_information_per_inlier.reset();
    return;
  }
  const auto eigenvalues = spectrum.eigenvalues();
  result.planar_observability = std::max(0.0, eigenvalues[0]);
  result.yaw_unobservable = planar_minimum <= tolerance &&
      eigenvalues[0] > tolerance / geometry.num_inliers;
  if (eigenvalues[0] > std::numeric_limits<double>::epsilon())
    result.xy_condition_number = eigenvalues[1] / eigenvalues[0];
  const auto axis = spectrum.eigenvectors().col(0);
  result.weakest_axis_yaw_rad = std::remainder(std::atan2(axis.y(), axis.x()), M_PI);
}

void classify(RelocalizabilityKeyframe& result, const RelocalizabilityThresholds& thresholds,
              double min_inlier_ratio, bool descriptor_available) {
  const auto mark = [&](RelocalizabilityColor color, const char* reason) {
    if (color > result.color) {
      result.color = color;
      result.reasons.insert(result.reasons.begin(), reason);
    } else {
      result.reasons.emplace_back(reason);
    }
  };
  if (result.distinctiveness_margin && result.top_confuser->fused_score <= thresholds.max_alias_score &&
      *result.distinctiveness_margin <= thresholds.yellow_alias_margin) {
    const bool strong_alias = result.top_confuser->fused_score <= thresholds.red_alias_score &&
        *result.distinctiveness_margin <= thresholds.red_alias_margin;
    mark(strong_alias ? RelocalizabilityColor::Red : RelocalizabilityColor::Yellow,
         std::abs(result.top_confuser->dz_m) > kLocalVertical
             ? "remote_floor_alias" : "remote_same_floor_alias");
  }
  if (!result.planar_observability || result.num_inliers < static_cast<std::size_t>(thresholds.min_inliers)) {
    mark(RelocalizabilityColor::Yellow, "low_inlier_support");
  } else {
    const bool below_abs_floor = result.inlier_ratio < min_inlier_ratio;
    const bool below_local_env = result.relative_inlier_ratio &&
        *result.relative_inlier_ratio < thresholds.min_relative_inlier_ratio;
    if (below_abs_floor || below_local_env)
      mark(RelocalizabilityColor::Yellow, "low_inlier_support");
    if (*result.planar_observability <= thresholds.yellow_observability) {
      mark(*result.planar_observability <= thresholds.red_observability
               ? RelocalizabilityColor::Red : RelocalizabilityColor::Yellow,
           "corridor_degeneracy");
    }
  }
  if (!descriptor_available || !result.local_neighbor_fused_score)
    mark(RelocalizabilityColor::Yellow, "low_descriptor_support");
  if (result.yaw_unobservable)
    mark(RelocalizabilityColor::Red, "yaw_unobservable");
  if (result.reasons.empty()) result.reasons.emplace_back("good");
}

}

std::vector<RelocalizabilityKeyframe> analyzeRelocalizability(
    const Config& config, const std::vector<Keyframe::Ptr>& frames,
    const FreeSpaceGrid& free_space, const RelocalizabilityThresholds& thresholds,
    const LocalizationAtlas* atlas) {
  const RHPDescriptor descriptor(rhpdParamsFromConfig(config));
  const auto& heading_params = descriptor.params().heading;
  const double place_weight =
      config.rhpd_primary_weight / std::max(1e-6, config.rhpd_dist_threshold);
  const double heading_weight =
      config.heading_aux_weight / std::max(1e-6, config.heading_aux_veto_threshold);
  std::vector<DescriptorView> views(frames.size());
  const auto count = static_cast<std::ptrdiff_t>(frames.size());

#pragma omp parallel for num_threads(config.num_threads) schedule(dynamic)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const auto& frame = *frames[index];
    auto& view = views[index];
    view.yaw = canonicalRollPitchYaw(frame.pose_optimized).z();
    if (atlas) {
      const auto place = atlas->frameDescriptors().find(frame.id);
      const auto heading = atlas->headingDescriptors().find(frame.id);
      if (place != atlas->frameDescriptors().end() && heading != atlas->headingDescriptors().end()) {
        view.place = place->second;
        view.heading = heading->second;
        continue;
      }
    }
    if (!frame.cloud || frame.cloud->empty()) continue;
    auto features = descriptor.computeWithHeading(frame.cloud);
    if (!features.place.isZero() && features.heading) {
      view.place = std::make_shared<const RHPDPlace>(std::move(features.place));
      view.heading = std::move(features.heading);
    }
  }

  Config geometry_config = config;
  geometry_config.num_threads = 1;
  geometry_config.icp_refine_use_gicp = false;
  const double resolution = std::max(1e-3, geometry_config.gicp_downsampling_resolution);
  std::vector<RelocalizabilityKeyframe> results(frames.size());
  std::vector<std::ptrdiff_t> session_order(frames.size());
  {
    std::vector<std::size_t> capture_order(frames.size());
    std::iota(capture_order.begin(), capture_order.end(), 0);
    std::stable_sort(capture_order.begin(), capture_order.end(), [&](auto a, auto b) {
      return std::tie(frames[a]->session_id, frames[a]->timestamp) <
             std::tie(frames[b]->session_id, frames[b]->timestamp);
    });
    for (std::size_t index = 0; index < capture_order.size(); ++index)
      session_order[capture_order[index]] = static_cast<std::ptrdiff_t>(index);
  }

#pragma omp parallel num_threads(config.num_threads)
  {
    PointCloudMatcher matcher(geometry_config);
#pragma omp for schedule(dynamic)
    for (std::ptrdiff_t index = 0; index < count; ++index) {
      const auto& query = *frames[index];
      const auto& view = views[index];
      auto& result = results[index];
      result.keyframe_id = query.id;
      result.session_id = query.session_id;
      result.position = query.getPosition();

      std::vector<std::pair<double, std::size_t>> neighbors;
      std::optional<RHPDescriptor::PreparedQuery> prepared_query;
      if (view.place) prepared_query.emplace(descriptor.prepareQuery(*view.place));
      const double infinity = std::numeric_limits<double>::infinity();

      for (std::ptrdiff_t candidate = 0; candidate < count; ++candidate) {
        if (candidate == index) continue;
        const auto& match = *frames[candidate];
        const Eigen::Vector3d delta = match.getPosition() - result.position;
        const double xy_distance = delta.head<2>().norm();
        const bool local = localPeer(result.position, match.getPosition(), free_space);
        if (local && match.cloud && !match.cloud->empty() &&
            (match.session_id != query.session_id ||
             std::abs(session_order[candidate] - session_order[index]) > 1))
          neighbors.emplace_back(delta.squaredNorm(), candidate);

        const auto& other = views[candidate];
        if (!prepared_query || !other.place) continue;
        const double rhpd = descriptor.distance(*prepared_query, *other.place);
        const double lower_bound = place_weight * rhpd;
        if (local) {
          if (lower_bound >= result.local_neighbor_fused_score.value_or(infinity)) continue;
        } else if (std::all_of(result.directional_confusers.begin(),
                               result.directional_confusers.end(), [&](const auto& existing) {
              return existing && lower_bound >= existing->fused_score;
            })) {
          continue;
        }

        const auto matches = matchRHPDHeadingDirections(
            *view.heading, *other.heading, heading_params, other.yaw - view.yaw);
        for (int direction = 0; direction < 2; ++direction) {
          const auto [heading, shift] = matches[direction];
          if (!std::isfinite(heading)) continue;
          const double score = lower_bound + heading_weight * std::max(0.0, heading);
          if (local) {
            if (score < result.local_neighbor_fused_score.value_or(infinity))
              result.local_neighbor_fused_score = score;
            continue;
          }
          auto& best = result.directional_confusers[direction];
          if (!best || score < best->fused_score) {
            const double yaw_shift = shift * (2.0 * M_PI / heading_params.sectors);
            best = RelocalizabilityConfuser{
                match.id, match.session_id, match.getPosition(), xy_distance, delta.z(),
                std::remainder(yaw_shift, 2.0 * M_PI), direction == 1, rhpd, heading, score};
          }
        }
      }

      for (const auto& confuser : result.directional_confusers)
        if (confuser && (!result.top_confuser || confuser->fused_score < result.top_confuser->fused_score))
          result.top_confuser = confuser;
      if (result.top_confuser && result.local_neighbor_fused_score)
        result.distinctiveness_margin =
            result.top_confuser->fused_score - *result.local_neighbor_fused_score;

      evaluateGeometry(result, query, frames, neighbors, matcher, resolution);
    }
  }

  const double min_inlier_ratio =
      thresholds.min_inlier_ratio.value_or(config.reloc_min_inlier_ratio);
  const KeyframePositions positions{frames};
  using PositionMetric = nanoflann::L2_Simple_Adaptor<double, KeyframePositions>;
  nanoflann::KDTreeSingleIndexAdaptor<PositionMetric, KeyframePositions, 3, std::size_t>
      tree(3, positions, nanoflann::KDTreeSingleIndexAdaptorParams(12));
  tree.buildIndex();
  const double radius_squared = std::nextafter(
      kLocalRadius * kLocalRadius + kLocalVertical * kLocalVertical,
      std::numeric_limits<double>::infinity());
  std::vector<std::pair<std::size_t, double>> nearby;
  std::vector<double> peer_ratios;
  for (std::size_t index = 0; index < results.size(); ++index) {
    auto& result = results[index];
    peer_ratios.clear();
    tree.radiusSearch(result.position.data(), radius_squared, nearby, nanoflann::SearchParams{});
    for (const auto& entry : nearby) {
      const auto peer = entry.first;
      if (peer != index && results[peer].planar_observability && results[peer].inlier_ratio > 0.0 &&
          localPeer(result.position, results[peer].position, free_space))
        peer_ratios.push_back(results[peer].inlier_ratio);
    }
    if (peer_ratios.size() >= kMinAdaptiveInlierNeighbors) {
      const std::size_t mid = peer_ratios.size() / 2;
      std::nth_element(peer_ratios.begin(), peer_ratios.begin() + mid, peer_ratios.end());
      double median = peer_ratios[mid];
      if ((peer_ratios.size() & 1u) == 0u) {
        const double lower = *std::max_element(peer_ratios.begin(), peer_ratios.begin() + mid);
        median = 0.5 * (lower + median);
      }
      if (median > std::numeric_limits<double>::epsilon()) {
        result.local_median_inlier_ratio = median;
        result.relative_inlier_ratio = result.inlier_ratio / median;
      }
    }
    classify(result, thresholds, min_inlier_ratio, static_cast<bool>(views[index].place));
  }
  return results;
}

}
