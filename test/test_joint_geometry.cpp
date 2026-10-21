#include <gtest/gtest.h>
#include "n3mapping/map_serializer.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/tracking_target_cache.h"
#include "n3mapping/world_localizing.h"
#include "../src/planar_geometry/association.h"
#include "../src/core/loop_closure_async.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace n3mapping {
namespace {
TEST(PlaneAssociationTest, LocalBridgesDoNotExplodeOrFuseDistinctSurfaces) {
  using namespace planar_geometry;
  std::vector<Plane> cells;
  std::vector<gtsam::Pose3> poses(3, gtsam::Pose3(gtsam::Rot3(), {0., 0., 2.}));
  int input_points = 0;
  for (int x = 0; x < 45; ++x) {
    const double height = x < 20 ? 0. : x < 25 ? .06 * (x - 19) : .3;
    Plane plane{Eigen::Vector3d::UnitZ(), Eigen::Vector3d(x + .5, .5, .5),
                height - .5, .01, {}};
    for (int scan = 0; scan < 3; ++scan) {
      Moment m;
      for (int u = 0; u < 5; ++u) for (int v = 0; v < 5; ++v)
        m.add(Eigen::Vector3d(x + .2 + .1*u, .2 + .1*v, height - 2.));
      plane.patches.push_back({scan, m, poses[scan].transformFrom(m.mean()),
                               Eigen::Vector3d::UnitZ(), 0.});
      input_points += m.count;
    }
    cells.push_back(std::move(plane));
  }
  Json::Value audit;
  const auto merged = mergePlanes(cells, poses, audit);
  EXPECT_GT(audit["merged_components_refused"].asInt(), 0);
  EXPECT_LT(merged.size(), cells.size() / 2);
  int output_points = 0;
  for (const auto& plane : merged) {
    for (const auto& patch : plane.patches) {
      output_points += patch.moment.count;
      // Distinct horizontal floors may not become a tilted compromise.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(patch.moment.covariance());
      EXPECT_LT(eigen.eigenvalues()[0], .0004 + 1e-10);
      EXPECT_LT(patch.moment.covariance()(2, 2), .01 + 1e-10);
      EXPECT_GT(plane.normal.dot(Eigen::Vector3d::UnitZ()), .999);
    }
  }
  EXPECT_EQ(output_points, input_points);
}

Eigen::Isometry3d pose(double x, double z = 0) {
  Eigen::Isometry3d value = Eigen::Isometry3d::Identity();
  value.translation() = Eigen::Vector3d(x, 0, z);
  return value;
}
class JointGeometryTest : public ::testing::Test {
protected:
  Config config;
  GraphOptimizer graph{config};
  KeyframeManager frames{config};
  LoopDetector detector{config};
  MapSerializer serializer{config};
  std::filesystem::path directory;
  void SetUp() override {
    char path[] = "/tmp/n3m-joint-XXXXXX";
    directory = mkdtemp(path);
    graph.addPriorFactor(0, pose(0));
    for (int i = 1; i < 3; ++i)
      graph.addOdometryEdge({i-1, i, pose(.2, .03),
          Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::ODOMETRY});
    ASSERT_TRUE(graph.incrementalOptimize());
    std::vector<Keyframe::Ptr> keyframes;
    for (int id = 0; id < 3; ++id) {
      auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
      for (int x = 0; x < 21; ++x) for (int y = 0; y < 21; ++y) {
        pcl::PointXYZI p;
        p.x = .1*x; p.y = .1*y; p.z = 1.; p.intensity = 1;
        cloud->push_back(p);
      }
      keyframes.push_back(Keyframe::create(id, id, graph.getOptimizedPose(id), cloud));
    }
    ASSERT_TRUE(frames.loadKeyframes(keyframes));
    GeometryOptimizationStats stats;
    auto planes = buildSharedPlanes(frames.getAllKeyframes(), stats);
    ASSERT_FALSE(planes.empty());
    ASSERT_TRUE(graph.replaceSharedPlanes(planes, true, stats));
    EXPECT_LT(stats.final_cost, stats.initial_cost);
    EXPECT_LT(std::abs(graph.getOptimizedPose(2).translation().z()), .02);
    frames.updateOptimizedPoses(graph.getOptimizedPoses());
  }
  void TearDown() override { std::filesystem::remove_all(directory); }
};

TEST_F(JointGeometryTest, MapRoundTripRetainsGeometryGaugeAndExactPoses) {
  const auto path = (directory / "map.pbstream").string();
  ASSERT_TRUE(serializer.saveMap(path, frames, detector, graph));
  GraphOptimizer restored(config);
  KeyframeManager restored_frames(config);
  LoopDetector restored_detector(config);
  ASSERT_TRUE(serializer.loadMap(path, restored_frames, restored_detector, restored));
  ASSERT_EQ(restored.sharedPlanes().size(), graph.sharedPlanes().size());
  EXPECT_TRUE(restored.gaugePrior().matrix().isApprox(graph.gaugePrior().matrix(), 1e-14));
  for (const auto& [id, value] : graph.getOptimizedPoses()) {
    EXPECT_LT((value.matrix()-restored.getOptimizedPose(id).matrix()).norm(), 1e-13);
    EXPECT_LT((value.matrix()-restored_frames.getKeyframe(id)->pose_optimized.matrix()).norm(), 1e-13);
  }
  const auto old = restored.getOptimizedPoses();
  restored.addOdometryEdge({2, 3, pose(.2),
      Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::ODOMETRY});
  ASSERT_TRUE(restored.incrementalOptimize());
  for (const auto& [id, value] : old)
    EXPECT_EQ((value.matrix()-restored.getOptimizedPose(id).matrix()).norm(), 0);
  EXPECT_LT((restored.getOptimizedPose(3).matrix() -
             (old.at(2)*pose(.2)).matrix()).norm(), 1e-12);
}

TEST_F(JointGeometryTest, WorkerCatchesUpNewNodesWithoutChangingItsSolvedPoses) {
  const auto snapshot = graph.cloneCommitted();
  const auto worker = graph.cloneCommitted();
  GeometryOptimizationStats stats;
  ASSERT_TRUE(worker->replaceSharedPlanes(worker->sharedPlanes(), true, stats));
  const auto solved = worker->getOptimizedPoses();
  graph.addOdometryEdge({2, 3, pose(.2),
      Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::ODOMETRY});
  ASSERT_TRUE(graph.incrementalOptimize());
  ASSERT_TRUE(worker->appendCommittedTail(graph, *snapshot));
  EXPECT_EQ(worker->getNumNodes(), 4u);
  EXPECT_EQ(worker->getNumEdges(), graph.getNumEdges());
  EXPECT_EQ(worker->sharedPlanes().size(), graph.sharedPlanes().size());
  for (const auto& [id, value] : solved)
    EXPECT_EQ((value.matrix()-worker->getOptimizedPose(id).matrix()).norm(), 0);
  EXPECT_LT((worker->getOptimizedPose(3).matrix() - (solved.at(2)*pose(.2)).matrix()).norm(), 1e-12);
}

TEST_F(JointGeometryTest, SharedGeometryRefinesNewFrameWithoutLoopAdmission) {
  graph.addOdometryEdge({2, 3, pose(.2, .04),
      Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::ODOMETRY});
  ASSERT_TRUE(graph.incrementalOptimize());
  const auto before = graph.getOptimizedPose(3);
  auto keyframes = frames.getAllKeyframes();
  const auto cloud = pcl::make_shared<Keyframe::PointCloudT>(*frames.getKeyframe(2)->cloud);
  keyframes.push_back(Keyframe::create(3, 3, before, cloud));
  ASSERT_TRUE(frames.loadKeyframes(keyframes));
  DetectedLoopQuery query;
  query.id = 3;
  AsyncLoopJob worker(config, frames, graph.cloneCommitted(), query);
  worker.run();
  EXPECT_TRUE(worker.outcome.summary.geometry.optimized);
  EXPECT_TRUE(worker.outcome.summary.accepted_loops.empty());
  EXPECT_EQ(worker.graph->getNumEdges(), graph.getNumEdges());
  EXPECT_LT(std::abs(worker.graph->getOptimizedPose(3).translation().z()),
            .5 * std::abs(before.translation().z()));
  // A background refinement cannot modify the live graph before commit.
  EXPECT_EQ(graph.getOptimizedPose(3).matrix(), before.matrix());
  auto disabled = config;
  disabled.loop_geometry_enable = false;
  AsyncLoopJob unchanged(disabled, frames, graph.cloneCommitted(), query);
  unchanged.run();
  EXPECT_FALSE(unchanged.outcome.summary.optimized);
  EXPECT_EQ(unchanged.graph->getOptimizedPose(3).matrix(), before.matrix());
}

TEST_F(JointGeometryTest, WorkerRejectsChangedOldGraphTransactionally) {
  const auto snapshot = graph.cloneCommitted();
  const auto worker = graph.cloneCommitted();
  graph.addLoopEdge({0, 2, pose(.4, .005),
      Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::LOOP});
  ASSERT_TRUE(graph.incrementalOptimize());
  const auto before = worker->getOptimizedPoses();
  EXPECT_FALSE(worker->appendCommittedTail(graph, *snapshot));
  EXPECT_EQ(worker->getNumEdges(), snapshot->getNumEdges());
  for (const auto& [id, value] : before)
    EXPECT_EQ((value.matrix()-worker->getOptimizedPose(id).matrix()).norm(), 0);
}

TEST_F(JointGeometryTest, RepeatedAsyncCatchUpPreservesRigidMotion) {
  Eigen::Isometry3d step = pose(.2, .001);
  step.linear() = Eigen::AngleAxisd(.03, Eigen::Vector3d(.3, .5, 1).normalized()).toRotationMatrix();
  auto expected = graph.getOptimizedPose(2);
  for (int64_t id = 3; id < 83; ++id) {
    const auto snapshot = graph.cloneCommitted();
    auto worker = graph.cloneCommitted();
    graph.addOdometryEdge({id-1, id, step,
        Eigen::Matrix<double, 6, 6>::Identity()*100, EdgeType::ODOMETRY});
    ASSERT_TRUE(graph.incrementalOptimize());
    ASSERT_TRUE(worker->appendCommittedTail(graph, *snapshot));
    graph.swapWith(*worker);
    expected = expected * step;
    const auto actual = graph.getOptimizedPose(id);
    ASSERT_NEAR(actual.linear().determinant(), 1., 1e-10) << "cycle " << id;
    ASSERT_LT((actual.linear().transpose()*actual.linear()-Eigen::Matrix3d::Identity()).norm(), 1e-10);
    ASSERT_LT((actual.translation()-expected.translation()).norm(), 1e-8) << "cycle " << id;
  }
}

TEST_F(JointGeometryTest, MalformedSavedPlaneCannotReplaceLiveMap) {
  const auto path = (directory / "map.pbstream").string();
  ASSERT_TRUE(serializer.saveMap(path, frames, detector, graph));
  N3Map map;
  std::string error;
  ASSERT_TRUE(readN3MapProtoFromFile(path, &map, &error));
  ASSERT_GT(map.shared_planes_size(), 0);
  map.mutable_shared_planes(0)->mutable_observations(0)->set_node_id(9999);
  { std::ofstream file(path, std::ios::binary); ASSERT_TRUE(map.SerializeToOstream(&file)); }
  const auto before = graph.getOptimizedPoses();
  EXPECT_FALSE(serializer.loadMap(path, frames, detector, graph));
  for (const auto& [id, value] : before)
    EXPECT_EQ((value.matrix()-graph.getOptimizedPose(id).matrix()).norm(), 0);
}

TEST_F(JointGeometryTest, AtlasTrackingAndReloadUseTheSameOptimizedGeometry) {
  const auto path = (directory / "map.pbstream").string();
  ASSERT_TRUE(serializer.saveMap(path, frames, detector, graph));
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher), loaded(config, matcher);
  const auto sidecar = LocalizationAtlas::defaultAtlasPath(path);
  ASSERT_TRUE(atlas.compileAndSave(path, frames.getAllKeyframes(), sidecar, false));
  ASSERT_TRUE(loaded.load(path, sidecar));
  ASSERT_EQ(atlas.globalMap()->size(), loaded.globalMap()->size());
  for (size_t i = 0; i < atlas.globalMap()->size(); ++i)
    EXPECT_EQ((atlas.globalMap()->at(i).getVector3fMap() -
               loaded.globalMap()->at(i).getVector3fMap()).norm(), 0);
  TrackingTargetCache targets(config, frames, matcher);
  targets.installLoadedReference(loaded.globalMap(), loaded.preparedTarget());
  const auto target = targets.acquireLoaded(0);
  ASSERT_TRUE(target.target);
  EXPECT_EQ(target.target->plane_levels.front().cloud,
            loaded.preparedTarget().plane_levels.front().cloud);
  WorldLocalizing localizer(config, frames, detector, matcher);
  localizer.setMapToOdomTransform(pose(1));
  localizer.installOptimizedReference(loaded, pose(.2));
  EXPECT_TRUE(localizer.isRelocalized());
  EXPECT_TRUE(localizer.localizationAtlasLoaded());
  EXPECT_LT((localizer.getMapToOdomTransform().matrix()-pose(1.2).matrix()).norm(), 1e-14);
}
} // namespace
} // namespace n3mapping
