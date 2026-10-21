#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include "n3mapping/core/n3mapping_core.h"
#include "n3map.pb.h"

namespace n3mapping {
namespace {
void pose(Pose3D* p, double x) { p->set_tx(x); p->set_qw(1); }
N3Map checkpoint(bool interpolate) {
  N3Map map;
  map.mutable_metadata()->set_version("2.9.0");
  map.mutable_metadata()->set_dense_trajectory_source("native");
  map.mutable_metadata()->set_num_keyframes(4);
  map.mutable_metadata()->set_num_odometry_edges(2);
  map.mutable_metadata()->set_num_session_anchor_edges(1);
  for (int session = 0; session < 2; ++session) {
    auto* s = map.add_sessions();
    s->set_id(session); s->set_source_frame_id("session" + std::to_string(session));
    s->set_start_timestamp(1); pose(s->mutable_t_map_session_initial(), 0);
    for (int frame = 0; frame < 2; ++frame) {
      auto* k = map.add_keyframes();
      k->set_id(2*session+frame); k->set_session_id(session); k->set_timestamp(1+frame);
      pose(k->mutable_pose_odom(), 10*session+frame);
      pose(k->mutable_pose_optimized(), 10*session+frame);
      auto* cloud = k->mutable_cloud(); cloud->set_num_points(20);
      for (int i=0; i<20; ++i) {
        cloud->add_points(.1*i); cloud->add_points(.2*(i%3));
        cloud->add_points(.1*(i%5)); cloud->add_points(1);
      }
    }
  }
  for(int id=1;id<4;++id) {
    auto* edge=map.add_edges();edge->set_from_id(id-1);edge->set_to_id(id);
    pose(edge->mutable_measurement(),id==2?9:1);
    edge->set_type(id==2?EdgeProto::SESSION_ANCHOR:EdgeProto::ODOMETRY);
    for(int r=0;r<6;++r)for(int c=r;c<6;++c)edge->mutable_information()->add_values(r==c?100:0);
  }
  pose(map.mutable_graph_gauge_prior(),0);
  auto* dense = map.add_dense_optimized_trajectory();
  dense->set_timestamp(1.5); pose(dense->mutable_pose_world_lidar(), .5);
  auto* source = dense->mutable_correction_source();
  pose(source->mutable_pose_world_lidar_raw(), .5);
  pose(source->mutable_anchor_pose_world_lidar_raw(), 0);
  source->set_has_anchor(true); source->set_anchor_keyframe_id(0);
  source->set_use_bracketing_correction(interpolate);
  return map;
}
void write(const std::filesystem::path& path, const N3Map& map) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(map.SerializeToOstream(&file));
}
TEST(TrajectoryCheckpoint, OverlappingSessionClocksRetainTheOriginalCorrection) {
  char temporary[] = "/tmp/n3m-trajectory-XXXXXX";
  const std::filesystem::path dir = mkdtemp(temporary);
  for (bool interpolate : {false, true}) {
    SCOPED_TRACE(interpolate);
    const auto input = dir / (interpolate ? "bracket.pbstream" : "anchor.pbstream");
    const auto output = dir / (interpolate ? "bracket_saved.pbstream" : "anchor_saved.pbstream");
    auto map = checkpoint(interpolate);
    write(input, map);
    Config config;
    N3MappingCore original(config);
    ASSERT_TRUE(original.loadMap(input.string()));
    ASSERT_TRUE(original.saveMap(output.string()));
    N3MappingCore restored(config);
    ASSERT_TRUE(restored.loadMap(output.string()));
    restored.getKeyframe(0)->pose_optimized.translation().y() = 1;
    restored.getKeyframe(1)->pose_optimized.translation().y() = 3;
    restored.getKeyframe(2)->pose_optimized.translation().y() = 10;
    restored.getKeyframe(3)->pose_optimized.translation().y() = 30;
    const auto dense = restored.getDenseOptimizedTrajectory();
    ASSERT_EQ(dense.size(), 1u);
    EXPECT_NEAR(dense[0].pose_world_lidar.translation().x(), .5, 1e-12);
    EXPECT_NEAR(dense[0].pose_world_lidar.translation().y(), interpolate ? 2 : 1, 1e-12);
    map.mutable_dense_optimized_trajectory(0)->mutable_correction_source()->set_anchor_keyframe_id(999);
    write(input,map);
    EXPECT_FALSE(restored.loadMap(input.string()));
    EXPECT_EQ(restored.getKeyframe(0)->pose_optimized.translation().y(), 1);
  }
  std::filesystem::remove_all(dir);
}
} // namespace
} // namespace n3mapping
