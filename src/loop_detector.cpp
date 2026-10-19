// LoopDetector: RHPD place/heading retrieval and spatial candidates.
#include <map>

#include "n3mapping/loop_detector.h"
#include "Scancontext/KDTreeVectorOfVectorsAdaptor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <glog/logging.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;
double wrapYaw(double yaw) {
    while (yaw >= M_PI) yaw -= 2.0 * M_PI;
    while (yaw < -M_PI) yaw += 2.0 * M_PI;
    return yaw;
}
} // anonymous namespace

namespace n3mapping {

std::vector<double> buildDescriptorYawHypotheses(
    const LoopCandidate& candidate, const Config& config) {
    std::vector<double> yaws;
    const auto append_unique = [&](double yaw) {
        const double wrapped = wrapYaw(yaw);
        const bool duplicate = std::any_of(
            yaws.begin(), yaws.end(), [&](double existing) {
                return std::abs(wrapYaw(existing - wrapped)) < kDescriptorYawMergeTolerance;
            });
        if (!duplicate) yaws.push_back(wrapped);
    };

    double heading_phase = 0.0;
    if (config.rhpd_use_sc_yaw && candidate.fromSC() &&
        std::isfinite(candidate.sc_distance)) {
        const double sector_rad =
            2.0 * M_PI / static_cast<double>(std::max(4, config.sc_num_sectors));
        // Matching shifts the reference's polar image into the query image.
        // A pose seed transforms query points into the reference frame, so
        // it needs the inverse rotation. Uniform yaw seeds can mask this sign
        // error, but then ICP starts far from the descriptor's actual heading.
        const double base_yaw = -static_cast<double>(candidate.yaw_diff_rad);
        heading_phase = base_yaw;
        append_unique(base_yaw);
        append_unique(base_yaw - sector_rad);
        append_unique(base_yaw + sector_rad);
    }
    if (candidate.fromRHPD() || yaws.empty()) {
        const int count = std::max(1, config.rhpd_yaw_hypotheses);
        // Keep the global grid uniform, but phase it with the measured yaw.
        // A half-turn-symmetric observation then gets equally precise opposite
        // seeds instead of one good seed and one distant local-optimizer guess.
        for (int i = 0; i < count; ++i) {
            append_unique(heading_phase + 2.0 * M_PI * static_cast<double>(i) /
                          static_cast<double>(count));
        }
    }
    return yaws;
}

LoopDetector::LoopDetector(const Config& config)
    : config_(config), rhpd_manager_(rhpdParamsFromConfig(config)) {}

RHPDHeadingPtr LoopDetector::computeHeading(const PointCloudT::Ptr& cloud) {
    return rhpd_manager_.computeHeading(cloud);
}

RHPDescriptor::Features LoopDetector::computeFeatures(const PointCloudT::Ptr& cloud) const {
    return rhpd_manager_.computeWithHeading(cloud);
}

RHPDHeadingPtr LoopDetector::addDescriptor(int64_t id, const PointCloudT::Ptr& cloud) {
    if (!cloud || cloud->empty()) return RHPDHeadingPtr();
    std::lock_guard<std::mutex> lock(mutex_);
    const RHPDHeadingPtr descriptor = computeHeading(cloud);
    if (!isHeadingShapeValid(descriptor)) return RHPDHeadingPtr();
    addDescriptorUnlocked(id, descriptor);
    return descriptors_[id_to_index_.at(id)];
}

void LoopDetector::addDescriptor(int64_t id, const RHPDHeadingPtr& descriptor) {
    if (!descriptor) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isHeadingShapeValid(descriptor)) {
        LOG(WARNING) << "[LoopDetector] Skip incompatible RHPD heading for keyframe "
                     << id << ": got " << descriptor->rings * RHPDHeadingParams::channels << "x" << descriptor->sectors
                     << ", expected " << getRHPDParams().heading.rows() << "x" << getRHPDParams().heading.sectors;
        return;
    }
    addDescriptorUnlocked(id, descriptor);
}

void LoopDetector::addDescriptorUnlocked(int64_t id, const RHPDHeadingPtr& descriptor) {
    if (id_to_index_.count(id) > 0) {
        LOG(WARNING) << "[LoopDetector] Duplicate RHPD heading id " << id
                     << ", keeping existing descriptor.";
        return;
    }
    id_to_index_[id] = descriptors_.size();
    index_to_id_.push_back(id);
    descriptors_.push_back(descriptor);
}

std::vector<LoopCandidate> LoopDetector::detectLoopCandidates(
    int64_t query_id, const std::map<int64_t, Keyframe::Ptr>& keyframes,
    const MatchFilter& accept_match) {
    std::vector<LoopCandidate> candidates;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = id_to_index_.find(query_id);
    if (it == id_to_index_.end()) return candidates;
    size_t query_index = it->second;

    // Scan Context searches a contiguous prefix of its key matrix, so it still
    // needs an index cutoff; RHPD does not, and is excluded by travelled path
    // below. A query too early for the prefix therefore skips Scan Context but
    // still gets RHPD candidates -- previously it returned nothing at all,
    // which is why no keyframe below 50 was ever examined.
    const int num_exclude = config_.sc_num_exclude_recent;
    const size_t search_end = static_cast<int>(query_index) >= num_exclude
                                  ? query_index - num_exclude
                                  : 0;

    // Cumulative odometry path length per keyframe index, so the distance
    // travelled between any two of them is one subtraction.
    std::vector<double> path_at(index_to_id_.size(), 0.0);
    {
        const Eigen::Vector3d *prev = nullptr;
        Eigen::Vector3d prev_store;
        double acc = 0.0;
        for (size_t i = 0; i < index_to_id_.size(); ++i) {
            const auto kit = keyframes.find(index_to_id_[i]);
            if (kit != keyframes.end() && kit->second) {
                const Eigen::Vector3d t =
                    kit->second->pose_odom.translation();
                if (prev)
                    acc += (t - *prev).norm();
                prev_store = t;
                prev = &prev_store;
            }
            path_at[i] = acc;
        }
    }
    const double min_path = std::max(0.0, config_.loop_min_path_length_m);

    const RHPDHeadingPtr& query_desc = descriptors_[query_index];
    const bool sc_query_available = static_cast<bool>(query_desc);

    if (config_.rhpd_enabled && rhpd_manager_.size() > 0) {
        RHPDPlace query_rhpd;
        if (rhpd_manager_.get(query_id, &query_rhpd) && !query_rhpd.isZero()) {
            const int preselect = std::max(config_.rhpd_num_candidates * 3, config_.sc_num_candidates * 4);
            // Informative rather than merely old: the pair must be separated by
            // enough travel that the odometry no longer pins it down.
            auto accept_old_enough = [&](int64_t id) {
                if (accept_match && !accept_match(id))
                    return false;
                const auto mit = id_to_index_.find(id);
                if (mit == id_to_index_.end() || mit->second >= query_index)
                    return false;
                return path_at[query_index] - path_at[mit->second] >= min_path;
            };
            auto rhpd_hits = rhpd_manager_.searchFiltered(
                query_rhpd, std::max(1, preselect), config_.rhpd_preselect_candidates, accept_old_enough);
            std::vector<LoopCandidate> ranked;
            ranked.reserve(rhpd_hits.size());

            int rhpd_rank = 0;
            for (const auto& [match_id, rhpd_dist] : rhpd_hits) {
                auto mit = id_to_index_.find(match_id);
                if (mit == id_to_index_.end()) continue;
                const size_t match_index = mit->second;
                LoopCandidate candidate;
                candidate.query_id = query_id;
                candidate.match_id = match_id;
                candidate.rhpd_distance = rhpd_dist;
                candidate.source_flags = LoopCandidate::SOURCE_RHPD;
                candidate.candidate_source = LoopCandidate::Source::RhpdPrimary;
                candidate.rhpd_rank = rhpd_rank++;

                const bool sc_match_available = match_index < descriptors_.size() && static_cast<bool>(descriptors_[match_index]);
                if (config_.rhpd_use_sc_yaw && sc_query_available && sc_match_available) {
                    auto [sc_dist, yaw_shift] = computeDistance(query_desc, descriptors_[match_index]);
                    candidate.sc_distance = sc_dist;
                    candidate.yaw_diff_rad = static_cast<float>(yaw_shift) *
                        static_cast<float>(getRHPDParams().heading.sectorAngleDeg()) * static_cast<float>(M_PI / 180.0);
                    candidate.source_flags |= LoopCandidate::SOURCE_SC;
                    candidate.sc_rank = -1;
                    if (config_.sc_aux_veto_enabled && sc_dist > config_.sc_aux_veto_threshold) {
                        continue;
                    }
                }

                const double rhpd_norm = rhpd_dist / std::max(1e-6, config_.rhpd_dist_threshold);
                const double sc_norm = candidate.fromSC() && std::isfinite(candidate.sc_distance)
                    ? candidate.sc_distance / std::max(1e-6, config_.sc_aux_veto_threshold)
                    : 0.0;
                candidate.fused_score =
                    config_.rhpd_primary_weight * rhpd_norm + config_.sc_aux_weight * sc_norm;
                candidate.descriptor_score = 1.0 / (1.0 + candidate.fused_score);
                ranked.push_back(candidate);
            }

            std::sort(ranked.begin(), ranked.end(),
                      [](const LoopCandidate& a, const LoopCandidate& b) {
                          if (a.fused_score != b.fused_score) return a.fused_score < b.fused_score;
                          return a.rhpd_distance < b.rhpd_distance;
                      });

            const int keep = std::min(config_.rhpd_num_candidates, static_cast<int>(ranked.size()));
            for (int i = 0; i < keep; ++i) {
                ranked[i].fused_rank = i;
                candidates.push_back(ranked[i]);
            }
            if (!candidates.empty()) {
                VLOG(1) << "[Loop/RHPDPrimary] query=" << query_id
                        << " kept=" << candidates.size()
                        << " top_match=" << candidates.front().match_id
                        << " rhpd=" << candidates.front().rhpd_distance
                        << " sc=" << candidates.front().sc_distance
                        << " yaw=" << candidates.front().yaw_diff_rad;
                return candidates;
            }
        }
    }

    // Place retrieval disabled or unavailable: use the direction-preserving channels.
    if (search_end == 0) return candidates;
    const auto& query_ringkey_vec = descriptors_[query_index]->ring_key;

    // Build a temporary eligible search set. Cross-session callers must apply
    // their loaded-map boundary before top-k selection; filtering the returned
    // top-k afterwards lets new-session descriptors crowd out every candidate
    // that is actually legal to commit.
    KeyMat search_keys;
    std::vector<size_t> search_indices;
    search_keys.reserve(search_end);
    search_indices.reserve(search_end);
    for (size_t i = 0; i < search_end; ++i) {
        if (accept_match && !accept_match(index_to_id_[i])) {
            continue;
        }
        search_keys.push_back(descriptors_[i]->ring_key);
        search_indices.push_back(i);
    }
    if (search_keys.empty()) return candidates;
    int tree_dim = getRHPDParams().heading.rows();
    InvKeyTree search_tree(tree_dim, search_keys, 10 /* max leaf */);

    // KNN 搜索: 取 NUM_CANDIDATES_FROM_TREE 个初筛候选
    constexpr size_t NUM_CANDIDATES_FROM_TREE = 20;  // 比 sc_num_candidates 大，给精排留余量
    size_t num_search = std::min(NUM_CANDIDATES_FROM_TREE, search_keys.size());
    std::vector<size_t> knn_indices(num_search);
    std::vector<float>  knn_dists(num_search);
    nanoflann::KNNResultSet<float> result_set(num_search);
    result_set.init(knn_indices.data(), knn_dists.data());
    search_tree.index->findNeighbors(result_set, query_ringkey_vec.data(),
                                     nanoflann::SearchParams(10));

    if (num_search > 0) {
        VLOG(1) << "[SC] query=" << query_id
                << " search_end=" << search_end
                << " eligible=" << search_keys.size()
                << " kd_top1_idx=" << index_to_id_[search_indices[knn_indices[0]]]
                << " kd_top1_L2=" << knn_dists[0];
    }

    // -------- 阶段 2: 对 KD-tree 初筛候选做完整多通道距离精排 --------
    std::vector<std::tuple<double, int, size_t>> refined_candidates;
    refined_candidates.reserve(num_search);

    for (size_t k = 0; k < num_search; ++k) {
        if (knn_indices[k] >= search_indices.size()) continue;
        size_t match_index = search_indices[knn_indices[k]];
        if (match_index >= search_end) continue;  // safety check
        auto [dist, yaw_shift] = computeDistance(query_desc, descriptors_[match_index]);
        refined_candidates.emplace_back(dist, yaw_shift, match_index);
    }

    // 按多通道距离排序
    std::sort(refined_candidates.begin(), refined_candidates.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    if (!refined_candidates.empty()) {
        VLOG(1) << "[SC] query=" << query_id
                << " refined_top1_match=" << index_to_id_[std::get<2>(refined_candidates[0])]
                << " refined_top1_dist=" << std::get<0>(refined_candidates[0])
                << " threshold=" << config_.sc_dist_threshold;
    }

    // 输出最终候选
    int num_candidates = std::min(config_.sc_num_candidates,
                                  static_cast<int>(refined_candidates.size()));
    for (int i = 0; i < num_candidates; ++i) {
        double dist = std::get<0>(refined_candidates[i]);
        int yaw_shift = std::get<1>(refined_candidates[i]);
        size_t match_index = std::get<2>(refined_candidates[i]);

        LoopCandidate candidate;
        candidate.query_id = query_id;
        candidate.match_id = index_to_id_[match_index];
        candidate.sc_distance = dist;
        candidate.rhpd_distance = std::numeric_limits<double>::max();
        candidate.yaw_diff_rad = static_cast<float>(yaw_shift) *
            static_cast<float>(getRHPDParams().heading.sectorAngleDeg()) * static_cast<float>(M_PI / 180.0);
        candidate.source_flags = LoopCandidate::SOURCE_SC;
        candidate.candidate_source = LoopCandidate::Source::ScanContextFallback;
        candidate.fused_score = dist;
        candidate.descriptor_score = 1.0 / (1.0 + dist / std::max(1e-6, config_.sc_dist_threshold));
        candidate.fused_rank = i;
        candidate.sc_rank = i;
        candidates.push_back(candidate);
    }
    return candidates;
}

std::vector<LoopCandidate> LoopDetector::detectSpatialCandidates(
    int64_t query_id,
    const std::map<int64_t, Keyframe::Ptr>& keyframes,
    const MatchFilter& accept_match) const {
    std::vector<LoopCandidate> candidates;
    if (!config_.loop_spatial_candidates_enable ||
        config_.loop_spatial_candidate_max_candidates <= 0) {
        return candidates;
    }

    auto query_it = keyframes.find(query_id);
    if (query_it == keyframes.end() || !query_it->second) {
        return candidates;
    }
    const auto& query_pose = query_it->second->pose_optimized;
    const double radius = std::max(1e-6, config_.loop_spatial_candidate_radius);
    const double min_path = std::max(0.0, config_.loop_min_path_length_m);

    // What makes a revisit worth registering is how far the platform walked
    // between the two visits, not how many keyframes it laid down. Asking for
    // fifty keyframes of index discarded the out-and-back at 95 to 104 m of
    // path on 0723 -- nine keyframes apart, nine metres of travel, and the
    // worst revisit disagreement in the map.
    std::map<int64_t, double> path_at;
    {
        double travelled = 0.0;
        const Eigen::Vector3d* previous = nullptr;
        Eigen::Vector3d previous_store;
        for (const auto& [id, keyframe] : keyframes) {
            if (keyframe) {
                const Eigen::Vector3d t = keyframe->pose_odom.translation();
                if (previous) {
                    travelled += (t - *previous).norm();
                }
                previous_store = t;
                previous = &previous_store;
            }
            path_at[id] = travelled;
        }
    }
    const auto query_path_it = path_at.find(query_id);
    const double query_path =
        query_path_it != path_at.end() ? query_path_it->second : 0.0;

    std::vector<std::pair<double, int64_t>> ranked;
    std::size_t spatial_considered = 0;
    std::size_t spatial_outside_radius = 0;
    for (const auto& [match_id, keyframe] : keyframes) {
        if (!keyframe || match_id >= query_id) {
            continue;
        }
        if (accept_match && !accept_match(match_id)) {
            continue;
        }
        const auto match_path_it = path_at.find(match_id);
        if (match_path_it == path_at.end() ||
            query_path - match_path_it->second < min_path) {
            continue;
        }
        const double squared_distance =
            (query_pose.translation() - keyframe->pose_optimized.translation()).squaredNorm();
        if (!std::isfinite(squared_distance)) {
            continue;
        }
        const double distance = std::sqrt(squared_distance);
        ++spatial_considered;
        // The radius is a hard bound, not a score normalizer: a candidate
        // farther than loop_spatial_candidate_radius must never enter the
        // returned set, however few candidates remain. Equal stays in.
        if (distance > radius) {
            ++spatial_outside_radius;
            continue;
        }
        ranked.emplace_back(distance, match_id);
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    const int keep = std::min(config_.loop_spatial_candidate_max_candidates,
                              static_cast<int>(ranked.size()));
    candidates.reserve(keep);
    for (int i = 0; i < keep; ++i) {
        LoopCandidate candidate;
        candidate.query_id = query_id;
        candidate.match_id = ranked[i].second;
        candidate.source_flags = LoopCandidate::SOURCE_SPATIAL;
        candidate.candidate_source = LoopCandidate::Source::SpatialRadius;
        candidate.fused_score = ranked[i].first / std::max(1e-6, radius);
        candidate.spatial_score = 1.0 / (1.0 + candidate.fused_score);
        candidate.fused_rank = i;
        candidates.push_back(candidate);
    }
    VLOG(1) << "[Loop/SpatialCandidates] considered=" << spatial_considered
            << " outside_radius=" << spatial_outside_radius
            << " kept=" << candidates.size() << " radius=" << radius;
    return candidates;
}

std::vector<std::pair<int64_t, RHPDHeadingPtr>> LoopDetector::getDescriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<int64_t, RHPDHeadingPtr>> result;
    result.reserve(descriptors_.size());
    for (size_t i = 0; i < descriptors_.size(); ++i) result.emplace_back(index_to_id_[i], descriptors_[i]);
    return result;
}

bool LoopDetector::isHeadingShapeValid(const RHPDHeadingPtr& descriptor) const {
    return descriptor &&
           descriptor->rings * RHPDHeadingParams::channels == getRHPDParams().heading.rows() &&
           descriptor->sectors == getRHPDParams().heading.sectors;
}

void LoopDetector::swapWith(LoopDetector& other) {
    if (this == &other) return;
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lock_this(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
    using std::swap;
    swap(config_, other.config_);
    rhpd_manager_.swapWith(other.rhpd_manager_);
    swap(id_to_index_, other.id_to_index_);
    swap(index_to_id_, other.index_to_id_);
    swap(descriptors_, other.descriptors_);
}

size_t LoopDetector::size() const { std::lock_guard<std::mutex> lock(mutex_); return descriptors_.size(); }

void LoopDetector::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    id_to_index_.clear(); index_to_id_.clear(); descriptors_.clear();
    rhpd_manager_.clear();
}

std::pair<int, int> LoopDetector::getDescriptorDimensions() const {
    return { getRHPDParams().heading.rows(), getRHPDParams().heading.sectors };
}

RHPDHeadingPtr LoopDetector::getDescriptor(int64_t keyframe_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = id_to_index_.find(keyframe_id);
    return (it != id_to_index_.end()) ? descriptors_[it->second] : RHPDHeadingPtr();
}

std::pair<double, int> LoopDetector::computeDistance(const RHPDHeadingPtr& sc1, const RHPDHeadingPtr& sc2) {
    if (!isHeadingShapeValid(sc1) || !isHeadingShapeValid(sc2)) return {1.0, 0};
    return matchRHPDHeading(*sc1, *sc2, getRHPDParams().heading);
}

// ---- RHPD methods ----

RHPDPlace LoopDetector::computeRHPD(const PointCloudT::Ptr& cloud) const {
    return rhpd_manager_.compute(cloud);
}

RHPDPlace LoopDetector::addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    return rhpd_manager_.addCloud(kf_id, cloud);
}

} // namespace n3mapping
