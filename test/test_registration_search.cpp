#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include "../src/registration_search.h"

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
  const bool actual = Search(tree, gate_squared).nearest(query, &actual_index, &actual_distance);
  ASSERT_EQ(actual, reference);
  if (reference) {
    EXPECT_EQ(actual_index, reference_index);
    EXPECT_DOUBLE_EQ(actual_distance, reference_distance);
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
  small_gicp::estimate_normals_covariances_omp(*target, tree, 20, 1);
  const Cloud source = *target;
  for (const auto type : {small_gicp::RegistrationSetting::ICP,
                          small_gicp::RegistrationSetting::PLANE_ICP,
                          small_gicp::RegistrationSetting::GICP}) {
    for (const double offset : {0.0, 0.1, 2.0, 40.0}) {
      small_gicp::RegistrationSetting setting;
      setting.type = type;
      setting.num_threads = 1;
      setting.max_correspondence_distance = 1.0;
      Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
      initial.translation() = Eigen::Vector3d(offset, -0.5 * offset, 0.2 * offset);
      const auto expected = small_gicp::align(*target, source, tree, initial, setting);
      const auto actual = registration_detail::align(*target, source, tree, initial, setting);
      EXPECT_EQ(actual.converged, expected.converged);
      EXPECT_EQ(actual.iterations, expected.iterations);
      EXPECT_EQ(actual.num_inliers, expected.num_inliers);
      EXPECT_DOUBLE_EQ(actual.error, expected.error);
      EXPECT_EQ(actual.T_target_source.matrix(), expected.T_target_source.matrix());
      EXPECT_EQ(actual.H, expected.H);
      EXPECT_EQ(actual.b, expected.b);
    }
  }
}
}  // namespace
}  // namespace n3mapping::test
