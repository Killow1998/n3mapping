#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/relocalization_place_catalog.h"

namespace n3mapping {

class RelocalizationPlaceIndex {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  RelocalizationPlaceIndex(const Config &config,
                           KeyframeManager &keyframe_manager,
                           LoopDetector &loop_detector);

  std::vector<LoopCandidate> search(const PointCloudT::Ptr &cloud,
      DescriptorSearchPerformance *performance = nullptr, std::size_t offset = 0,
      bool apply_initial_prior = false);
  void prepare(const RHPDPlaceCache &cached_frames = {},
               DescriptorSearchPerformance *performance = nullptr,
               const RelocalizationPlaceCatalog *cached_catalog = nullptr);
  void resetMapDerivedState();
  std::size_t indexedKeyframes() const;
  std::size_t indexedViews() const { return retrieval_view_count_; }
  bool matchesInitialPriorPose(const Eigen::Isometry3d& pose) const;

private:
  struct View {
    int64_t id;
    Eigen::Vector3d position;
    std::shared_ptr<const RHPDPlace> frame, submap;
    RHPDHeadingPtr heading;
    RHPDManager::VecD frame_coarse, submap_coarse;
  };
  struct Place { std::vector<View> views; };
  std::vector<LoopCandidate> searchRHPDCandidates(
      const RHPDPlace &query, const RHPDHeadingPtr &heading,
      DescriptorSearchPerformance *performance, std::size_t offset, bool apply_initial_prior);
  bool matchesInitialPrior(const Eigen::Vector3d& position) const;

  Config config_;
  KeyframeManager &keyframe_manager_;
  LoopDetector &loop_detector_;
  RHPDescriptor descriptor_;
  RHPDPlaceCache frames_;
  std::vector<Place> places_;
  std::size_t retrieval_view_count_ = 0;
  KeyframeMapRevision view_revision_;
  std::size_t frame_rhpd_indexed_keyframes_ = 0;
  KeyframeMapRevision frame_rhpd_revision_;
};

}
