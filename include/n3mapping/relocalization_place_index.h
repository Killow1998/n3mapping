// Descriptor retrieval and its map-revision-bound frame-level RHPD cache.
// Candidate evaluation and lock policy deliberately remain outside this class.
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

class RelocalizationPlaceIndex {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  RelocalizationPlaceIndex(const Config &config,
                           KeyframeManager &keyframe_manager,
                           LoopDetector &loop_detector);

  std::vector<LoopCandidate> search(const PointCloudT::Ptr &cloud);
  void resetMapDerivedState();
  std::size_t indexedKeyframes() const;

private:
  void rebuildFrameRHPDIndexIfNeeded();
  void appendFrameRHPDCandidates(const Eigen::VectorXd &query_rhpd,
                                 const Eigen::MatrixXd &query_sc,
                                 std::vector<LoopCandidate> &candidates);

  Config config_;
  KeyframeManager &keyframe_manager_;
  LoopDetector &loop_detector_;
  RHPDManager frame_rhpd_manager_;
  std::size_t frame_rhpd_indexed_keyframes_ = 0;
  KeyframeMapRevision frame_rhpd_revision_;
};

} // namespace n3mapping
