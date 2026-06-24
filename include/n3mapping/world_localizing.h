// WorldLocalizing: global relocalization via RHPD + ICP, and tracking localization with T_map_odom.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct RelocResult {
    bool success = false;
    int64_t matched_keyframe_id = -1;
    Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
    double confidence = 0.0;
    double fitness_score = 0.0;
};

class WorldLocalizing {
public:
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    WorldLocalizing(const Config& config, KeyframeManager& keyframe_manager,
                    LoopDetector& loop_detector, PointCloudMatcher& matcher);
    ~WorldLocalizing();

    RelocResult relocalize(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    RelocResult trackLocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    bool isRelocalized() const;
    Eigen::Isometry3d getMapToOdomTransform() const;
    void reset();
    void setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom);
    int64_t getLastMatchedKeyframeId() const;

private:
    struct RelocHypothesis {
        int64_t seed_match_id = -1;
        int64_t last_match_id = -1;
        Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
        double cumulative_log_likelihood = 0.0;
        int num_updates = 0;
        int converged_updates = 0;
        bool alive = true;
    };

    struct QueryFrame {
        PointCloudT::Ptr cloud;
        Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    };

    struct TrackingTargetCache {
        int64_t center_keyframe_id = -1;
        int submap_range = -1;
        PointCloudT::Ptr submap;
        std::vector<PointCloudMatcher::PreparedTarget> targets;
    };

    struct TrackingTargetRequest {
        int64_t center_keyframe_id = -1;
        int submap_range = -1;
        uint64_t generation = 0;
    };

    std::vector<LoopCandidate> searchCandidates(const PointCloudT::Ptr& cloud);
    RelocResult verifyCandidates(const PointCloudT::Ptr& cloud, const std::vector<LoopCandidate>& candidates);
    bool evaluateSingleCandidate(const PointCloudT::Ptr& cloud,
                                 const LoopCandidate& candidate,
                                 MatchResult& best_match,
                                 int64_t& matched_kf_id);
    void rebuildFrameRHPDIndexIfNeeded();
    void appendFrameRHPDCandidates(const Eigen::VectorXd& query_rhpd,
                                   const Eigen::MatrixXd& query_sc,
                                   std::vector<LoopCandidate>& candidates);
    double computeRelocLogLikelihood(const LoopCandidate& candidate, const MatchResult& match_result) const;
    double computeTrackLogLikelihood(const MatchResult& match_result,
                                     const Eigen::Isometry3d& predicted_pose) const;
    PointCloudT::Ptr buildRelocQueryCloud(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    void clearRelocHypotheses();
    int64_t findNearestKeyframe(const Eigen::Isometry3d& pose) const;
    bool getTrackingTargetCache(int64_t center_keyframe_id,
                                int submap_range,
                                TrackingTargetCache* cache,
                                bool* cache_hit,
                                std::size_t* cache_entries);
    bool buildTrackingTargetCache(int64_t center_keyframe_id,
                                  int submap_range,
                                  TrackingTargetCache* cache,
                                  double* build_ms,
                                  double* prepare_ms);
    void requestTrackingTargetPrefetch(int64_t center_keyframe_id, int submap_range);
    void requestTrackingTargetPrefetchNeighborhood(int64_t center_keyframe_id, int submap_range);
    void trackingTargetPrefetchLoop();
    std::size_t trackingTargetCacheLimit() const;
    std::size_t trackingTargetCacheEstimatedBytesLocked() const;
    void warnIfTrackingTargetCacheMemoryHighLocked();
    bool hasTrackingTargetCacheLocked(int64_t center_keyframe_id, int submap_range) const;
    void appendTrackingPrefetchLog(const TrackingTargetRequest& request,
                                   bool built,
                                   double build_ms,
                                   double prepare_ms,
                                   std::size_t target_points,
                                   std::size_t cache_entries,
                                   double cache_estimated_mb);
    void clearTrackingTargetCache();

    Config config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    PointCloudMatcher& matcher_;
    RHPDManager frame_rhpd_manager_;
    size_t frame_rhpd_indexed_keyframes_;

    bool is_relocalized_;
    Eigen::Isometry3d T_map_odom_;
    int64_t last_matched_id_;
    Eigen::Isometry3d last_odom_pose_;
    int consecutive_track_failures_;
    std::vector<RelocHypothesis> pending_hypotheses_;
    std::deque<QueryFrame> query_frame_buffer_;
    std::deque<TrackingTargetCache> tracking_target_cache_;
    std::deque<TrackingTargetRequest> tracking_prefetch_queue_;
    std::set<std::pair<int64_t, int>> tracking_prefetch_pending_;
    mutable std::mutex tracking_cache_mutex_;
    std::condition_variable tracking_prefetch_cv_;
    std::thread tracking_prefetch_thread_;
    bool tracking_prefetch_stop_ = false;
    bool tracking_target_cache_memory_warning_logged_ = false;
    uint64_t tracking_cache_generation_ = 0;
    uint64_t tracking_perf_count_ = 0;
    int hypothesis_window_count_;
    int64_t last_window_winner_seed_id_;
    int winner_streak_;
    mutable std::mutex mutex_;
};

} // namespace n3mapping
