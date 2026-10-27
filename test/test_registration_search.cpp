#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include "../src/registration_search.h"
#include "../src/compact_kdtree_builder.h"
#include "../src/point_cloud_feature_reuse.h"
#include "../src/point_cloud_target_view.h"

namespace n3mapping::test {
namespace {
using Search = registration_detail::CorrespondenceSearch;
using Cloud = small_gicp::PointCloud;

Cloud::Ptr randomCloud() {
  auto cloud = std::make_shared<Cloud>();
  cloud->resize(800);
  std::mt19937 generator(3588);
  std::uniform_real_distribution<double> coordinate(-10.0, 10.0);
  for (std::size_t i = 0; i < cloud->size(); ++i)
    cloud->point(i) = Eigen::Vector4d(coordinate(generator), coordinate(generator),
                                     coordinate(generator), 1.0);
  return cloud;
}

void expectSameCorrespondence(const Search::Tree& tree,
                              const Eigen::Vector4d& query, double gate_squared) {
  std::size_t reference_index = 0, actual_index = 0;
  double reference_distance = 0.0, actual_distance = 0.0;
  const bool reference = tree.nearest_neighbor_search(
      query, &reference_index, &reference_distance) && reference_distance <= gate_squared;
  const registration_detail::SearchBounds bounds(tree);
  // Exercise both useful and stale hints, including a different equal-distance
  // point and an out-of-range index. A hint must never choose the answer.
  for (const std::size_t hint : {std::numeric_limits<std::size_t>::max(),
                                 std::size_t{0}, tree.points->size() - 1,
                                 reference_index, tree.points->size()}) {
    for (const auto* occupied : {static_cast<const registration_detail::SearchBounds*>(nullptr), &bounds}) {
      const bool actual = Search(tree, gate_squared, occupied).withHint(hint).nearest(
          query, &actual_index, &actual_distance);
      ASSERT_EQ(actual, reference);
      if (reference) {
        EXPECT_EQ(actual_index, reference_index);
        EXPECT_DOUBLE_EQ(actual_distance, reference_distance);
      }
    }
  }
}

TEST(RegistrationSearch, PreservesExactAcceptedNeighborsForNearAndDistantQueries) {
  const auto cloud = randomCloud();
  const Search::Tree tree(cloud);
  std::mt19937 generator(20261017);
  std::uniform_real_distribution<double> coordinate(-30.0, 30.0);
  for (int i = 0; i < 2000; ++i) {
    const Eigen::Vector4d query(coordinate(generator), coordinate(generator),
                                coordinate(generator), 1.0);
    for (const double radius : {0.0, 0.05, 0.3, 1.0, 2.0, 20.0})
      expectSameCorrespondence(tree, query, radius * radius);
  }
}

TEST(RegistrationSearch, IncludesDistanceBoundaryAndPreservesEqualDistanceChoice) {
  auto cloud = randomCloud();
  // Multiple equal-distance points, including duplicates, across tree leaves.
  for (std::size_t i = 0; i < cloud->size(); ++i) {
    const double sign = i % 2 ? 1.0 : -1.0;
    cloud->point(i) = Eigen::Vector4d::Zero();
    cloud->point(i)[i % 3] = sign;
    cloud->point(i).w() = 1.0;
  }
  const Search::Tree tree(cloud);
  const Eigen::Vector4d origin(0.0, 0.0, 0.0, 1.0);
  for (const double gate : {std::nextafter(1.0, 0.0), 1.0,
                            std::nextafter(1.0, 2.0)})
    expectSameCorrespondence(tree, origin, gate);
  for (std::size_t i = 0; i < 6; ++i)
    expectSameCorrespondence(tree, cloud->point(i), 0.0);
}

TEST(RegistrationSearch, KeepsUnboundedAndNonFiniteQueryFallback) {
  const auto cloud = randomCloud();
  const Search::Tree tree(cloud);
  expectSameCorrespondence(tree, Eigen::Vector4d(100, 100, 100, 1),
                           std::numeric_limits<double>::infinity());
  Eigen::Vector4d query(0, 0, 0, 1);
  query.x() = std::numeric_limits<double>::quiet_NaN();
  std::size_t expected_index = 0, actual_index = 0;
  double expected_distance = 0.0, actual_distance = 0.0;
  EXPECT_EQ(tree.nearest_neighbor_search(query, &expected_index, &expected_distance),
            Search(tree, 1.0).nearest(query, &actual_index, &actual_distance));
  EXPECT_EQ(actual_index, expected_index);
  EXPECT_EQ(std::isnan(actual_distance), std::isnan(expected_distance));
}

TEST(RegistrationSearch, PreservesNeighborsAtFloatingPointSplitBoundaries) {
  const auto cloud = randomCloud();
  const Search::Tree tree(cloud);
  for (const auto& node : tree.kdtree.nodes) {
    if (node.left == small_gicp::INVALID_NODE) continue;
    for (const double direction : {-std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()}) {
      Eigen::Vector4d query(0.1, -0.2, 0.3, 1.0);
      query[node.node_type.sub.proj.axis] =
          std::nextafter(node.node_type.sub.thresh, direction);
      std::size_t index;
      double distance;
      ASSERT_TRUE(tree.nearest_neighbor_search(query, &index, &distance));
      for (const double gate : {std::nextafter(distance, 0.0), distance,
                                std::nextafter(distance, direction)})
        expectSameCorrespondence(tree, query, gate);
    }
  }
}

TEST(RegistrationSearch, PreservesOptimizerPoseQualityAndInformation) {
  const auto target = randomCloud();
  const Search::Tree tree(target);
  const registration_detail::SearchBounds bounds(tree);
  small_gicp::estimate_normals_covariances_omp(*target, tree, 20, 1);
  const Cloud source = *target;
  for (const int threads : {1, 4}) {
    for (const auto type : {small_gicp::RegistrationSetting::ICP,
                            small_gicp::RegistrationSetting::PLANE_ICP,
                            small_gicp::RegistrationSetting::GICP}) {
      for (const double offset : {0.0, 0.1, 2.0, 40.0}) {
        small_gicp::RegistrationSetting setting;
        setting.type = type;
        setting.num_threads = threads;
        setting.max_correspondence_distance = 1.0;
        Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
        initial.translation() = Eigen::Vector3d(offset, -0.5 * offset, 0.2 * offset);
        const auto expected = small_gicp::align(*target, source, tree, initial, setting);
        const auto actual = registration_detail::align(*target, source, tree, initial, setting, &bounds);
        EXPECT_EQ(actual.converged, expected.converged);
        EXPECT_EQ(actual.iterations, expected.iterations);
        EXPECT_EQ(actual.num_inliers, expected.num_inliers);
        if (threads == 1) {
          EXPECT_DOUBLE_EQ(actual.error, expected.error);
          EXPECT_EQ(actual.T_target_source.matrix(), expected.T_target_source.matrix());
          EXPECT_EQ(actual.H, expected.H);
          EXPECT_EQ(actual.b, expected.b);
        } else {
          // Guided scheduling already changes summation order between runs.
          EXPECT_NEAR(actual.error, expected.error, 1e-9);
          EXPECT_TRUE(actual.T_target_source.matrix().isApprox(
              expected.T_target_source.matrix(), 1e-10));
          EXPECT_LE((actual.H - expected.H).norm(),
                    1e-12 * std::max(1.0, expected.H.norm()));
          EXPECT_LE((actual.b - expected.b).norm(),
                    std::max(1e-9, 1e-12 * expected.b.norm()));
        }
      }
    }
  }
}

TEST(RegistrationSearch, PackedCovarianceOnlySourcePreservesEveryPointAndFeature) {
  const auto source = randomCloud();
  source->normals.clear();
  source->point(1) = source->point(0);
  for (std::size_t i = 0; i < source->size(); ++i)
    source->cov(i) = Eigen::Matrix4d::Identity() * double(i + 1);
  const Cloud original = *source;
  const Search::Tree tree(source,
      registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilderOMP>(
          *source, true, 4));
  ASSERT_TRUE(source->normals.empty());
  ASSERT_EQ(source->size(), original.size());
  ASSERT_EQ(source->covs.size(), original.covs.size());
  std::vector<int> seen(source->size(), 0);
  for (std::size_t i = 0; i < source->size(); ++i) {
    // The feature value identifies each original tuple, even at duplicate XYZ.
    const auto before = static_cast<std::size_t>(source->cov(i)(0, 0)) - 1;
    ASSERT_LT(before, original.size());
    ++seen[before];
    EXPECT_EQ(source->point(i), original.point(before));
    EXPECT_EQ(source->cov(i), original.cov(before));
  }
  EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int count) { return count == 1; }));
}

TEST(RegistrationSearch, PackedTargetKeepsNeighborGeometryAndFeatures) {
  const auto original = randomCloud();
  original->point(1) = original->point(0);
  for (std::size_t i = 0; i < original->size(); ++i) {
    original->normal(i) = Eigen::Vector4d(double(i), -double(i), 2.0, 0.0);
    original->cov(i) = Eigen::Matrix4d::Identity() * double(i + 1);
  }
  const Search::Tree reference(original);
  const auto packed = std::make_shared<Cloud>(*original);
  const Search::Tree actual(packed,
      registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilder>(*packed, true));
  std::mt19937 generator(20261018);
  std::normal_distribution<double> noise(0.0, 0.5);
  for (int i = 0; i < 10000; ++i) {
    Eigen::Vector4d query = original->point(i % original->size());
    if (i >= static_cast<int>(original->size()))
      query.head<3>() += Eigen::Vector3d(noise(generator), noise(generator), noise(generator));
    std::size_t expected_index, actual_index;
    double expected_distance, actual_distance;
    ASSERT_EQ(actual.nearest_neighbor_search(query, &actual_index, &actual_distance),
              reference.nearest_neighbor_search(query, &expected_index, &expected_distance));
    ASSERT_DOUBLE_EQ(actual_distance, expected_distance);
    ASSERT_EQ(packed->point(actual_index), original->point(expected_index));
    ASSERT_EQ(packed->normal(actual_index), original->normal(expected_index));
    ASSERT_EQ(packed->cov(actual_index), original->cov(expected_index));
  }
}

TEST(RegistrationSearch, FeatureReuseInvalidatesChangedSupportsAndKeepsUnchangedGeometry) {
  // Insert, delete, duplicate and move points; a changed point can invalidate a
  // neighbor's normal even when that neighbor's own coordinate did not move.
  for (const int neighbors : {5, 20, 31}) {
    const auto source = randomCloud();
    const Search::Tree source_tree(source);
    std::vector<double> radius;
    registration_detail::estimateFeatures<small_gicp::NormalSetter<Cloud>>(
        *source, source_tree.kdtree, neighbors, 4, &radius);
    auto target = std::make_shared<Cloud>(*source);
    target->points.erase(target->points.begin() + 30);
    target->point(5).z() += .1;
    target->points.push_back(target->point(50));
    target->points.push_back(Eigen::Vector4d(0, 0, 0, 1));
    std::reverse(target->points.begin(), target->points.end());
    const Search::Tree target_tree(target);
    const auto reused = registration_detail::reuseTargetFeatures(
        *source, source_tree, radius, *target, target_tree, neighbors, 4);
    EXPECT_GT(reused, target->size() / 2);
    EXPECT_LT(reused, target->size());
    const auto normals = target->normals;
    std::vector<Eigen::Matrix4d> covariances;
    for (size_t i = 0; i < target->size(); ++i)
      covariances.push_back(registration_detail::TargetView{*target}.covariance(i));
    small_gicp::estimate_normals_covariances_omp(*target, target_tree, neighbors, 4);
    for (size_t i = 0; i < target->size(); ++i) {
      EXPECT_TRUE(normals[i].isApprox(target->normal(i), 1e-10));
      EXPECT_TRUE(covariances[i].isApprox(target->cov(i), 1e-10));
    }
  }
}

TEST(RegistrationSearch, FeatureReuseRecomputesTiedAndUndersizedNeighborhoods) {
  for (const int neighbors : {5, 20}) {
    auto source = std::make_shared<Cloud>();
    for (int i = 0; i < 18; ++i) {
      Eigen::Vector4d p(0, 0, 0, 1);
      p[i % 3] = i % 2 ? 1 : -1;
      source->points.push_back(p);
    }
    const Search::Tree source_tree(source);
    std::vector<double> radius;
    registration_detail::estimateFeatures<small_gicp::NormalSetter<Cloud>>(
        *source, source_tree.kdtree, neighbors, 1, &radius);
    const auto target = std::make_shared<Cloud>(*source);
    std::reverse(target->points.begin(), target->points.end());
    const Search::Tree target_tree(target);
    EXPECT_EQ(registration_detail::reuseTargetFeatures(
        *source, source_tree, radius, *target, target_tree, neighbors, 1), 0u);
    std::vector<Eigen::Matrix4d> covariances;
    for (size_t i = 0; i < target->size(); ++i)
      covariances.push_back(registration_detail::TargetView{*target}.covariance(i));
    small_gicp::estimate_normals_covariances_omp(*target, target_tree, neighbors, 1);
    for (size_t i = 0; i < target->size(); ++i)
      EXPECT_TRUE(covariances[i].isApprox(target->cov(i), 1e-12));
  }
}

TEST(RegistrationSearch, CompactCovariancePreservesGicpObjectivePoseAndInformation) {
  const auto full = randomCloud();
  const Search::Tree tree(full);
  small_gicp::estimate_normals_covariances_omp(*full, tree, 20, 1);
  auto compact = *full;
  compact.covs.clear();
  for (size_t i = 0; i < full->size(); ++i)
    ASSERT_TRUE(registration_detail::TargetView{compact}.covariance(i).isApprox(full->cov(i), 1e-12));
  small_gicp::RegistrationSetting setting;
  setting.type = small_gicp::RegistrationSetting::GICP;
  setting.num_threads = 1;
  for (double offset : {0.0, .1, .5}) {
    auto seed = Eigen::Isometry3d::Identity();
    seed.translation() = Eigen::Vector3d(offset, -.4 * offset, .2 * offset);
    const auto expected = registration_detail::align(*full, *full, tree, seed, setting);
    const auto actual = registration_detail::align(compact, *full, tree, seed, setting);
    EXPECT_EQ(actual.converged, expected.converged);
    EXPECT_EQ(actual.num_inliers, expected.num_inliers);
    EXPECT_TRUE(actual.T_target_source.matrix().isApprox(expected.T_target_source.matrix(), 1e-10));
    EXPECT_NEAR(actual.error, expected.error, 1e-10);
    EXPECT_LE((actual.H - expected.H).norm(), 1e-10 * expected.H.norm());
  }
  full->cov(0) *= 2;
  EXPECT_EQ(registration_detail::TargetView{*full}.covariance(0), full->cov(0));
}
}  // namespace
}  // namespace n3mapping::test
