// LoopDetector: ScanContext descriptor generation, candidate search, and ICP verification.
#include "n3mapping/loop_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <glog/logging.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
// Eigen column vector → std::vector<float> (for KD-tree)
inline std::vector<float> eig2stdvec(const Eigen::MatrixXd& mat) {
    std::vector<float> vec(mat.size());
    for (int i = 0; i < static_cast<int>(mat.size()); ++i)
        vec[i] = static_cast<float>(mat(i));
    return vec;
}

void swapScanContextConfig(HybridSCManager& lhs, HybridSCManager& rhs) {
    using std::swap;
    swap(lhs.PC_NUM_RING, rhs.PC_NUM_RING);
    swap(lhs.PC_NUM_SECTOR, rhs.PC_NUM_SECTOR);
    swap(lhs.PC_MAX_RADIUS, rhs.PC_MAX_RADIUS);
    swap(lhs.PC_MIN_RADIUS, rhs.PC_MIN_RADIUS);
    swap(lhs.LIDAR_HEIGHT, rhs.LIDAR_HEIGHT);
    swap(lhs.USE_LOG_POLAR, rhs.USE_LOG_POLAR);
    swap(lhs.OCCUPY_Z_MIN, rhs.OCCUPY_Z_MIN);
    swap(lhs.OCCUPY_Z_MAX, rhs.OCCUPY_Z_MAX);
    swap(lhs.SEARCH_RATIO, rhs.SEARCH_RATIO);
    swap(lhs.W_HEIGHT, rhs.W_HEIGHT);
    swap(lhs.W_HEIGHT_VAR, rhs.W_HEIGHT_VAR);
    swap(lhs.W_DENSITY, rhs.W_DENSITY);
    swap(lhs.W_OCCUPY_L0, rhs.W_OCCUPY_L0);
    swap(lhs.W_OCCUPY_L1, rhs.W_OCCUPY_L1);
    swap(lhs.W_OCCUPY_L2, rhs.W_OCCUPY_L2);
    swap(lhs.W_OCCUPY_L3, rhs.W_OCCUPY_L3);
    swap(lhs.PC_UNIT_SECTORANGLE, rhs.PC_UNIT_SECTORANGLE);
}
} // anonymous namespace

namespace n3mapping {

namespace {
RHPDescriptor::Params makeRHPDParams(const Config& config) {
    RHPDescriptor::Params rhpd_params;
    rhpd_params.max_range = std::max(1.0, config.rhpd_max_range);
    rhpd_params.z_min = config.rhpd_z_min;
    rhpd_params.z_max = std::max(config.rhpd_z_max, config.rhpd_z_min + 1e-3);
    rhpd_params.v2_enable = config.rhpd_v2_enable;
    rhpd_params.v3_enable = config.rhpd_v3_enable;
    rhpd_params.enable_negative_space = config.rhpd_enable_negative_space;
    rhpd_params.enable_vertical_tokens = config.rhpd_enable_vertical_tokens;
    rhpd_params.enable_pca_confidence = config.rhpd_enable_pca_confidence;
    return rhpd_params;
}
}  // namespace

LoopDetector::LoopDetector(const Config& config)
    : config_(config), rhpd_manager_(makeRHPDParams(config)) {
    sc_manager_.PC_NUM_RING = std::max(1, config_.sc_num_rings);
    sc_manager_.PC_NUM_SECTOR = std::max(4, config_.sc_num_sectors);
    sc_manager_.PC_MAX_RADIUS = std::max(config_.sc_max_radius, sc_manager_.PC_MIN_RADIUS + 1e-3);
    sc_manager_.PC_UNIT_SECTORANGLE = 360.0 / static_cast<double>(sc_manager_.PC_NUM_SECTOR);
}

Eigen::MatrixXd LoopDetector::makeScanContext(const PointCloudT::Ptr& cloud) {
    if (!cloud || cloud->empty()) return Eigen::MatrixXd();
    pcl::PointCloud<SCPointType> cloud_copy = *cloud;
    return sc_manager_.makeScancontext(cloud_copy);
}

Eigen::MatrixXd LoopDetector::addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud) {
    if (!cloud || cloud->empty()) return Eigen::MatrixXd();
    std::lock_guard<std::mutex> lock(mutex_);
    pcl::PointCloud<SCPointType> cloud_copy = *cloud;
    Eigen::MatrixXd descriptor = sc_manager_.makeScancontext(cloud_copy);
    if (descriptor.size() == 0) return Eigen::MatrixXd();
    if (id_to_index_.count(keyframe_id) > 0) {
        LOG(WARNING) << "[LoopDetector] Duplicate ScanContext descriptor id " << keyframe_id
                     << ", keeping existing descriptor.";
        return descriptors_[id_to_index_[keyframe_id]];
    }
    size_t index = descriptors_.size();
    id_to_index_[keyframe_id] = index;
    index_to_id_.push_back(keyframe_id);
    descriptors_.push_back(descriptor);
    sc_manager_.makeAndSaveScancontextAndKeys(cloud_copy);
    return descriptor;
}

void LoopDetector::addDescriptor(int64_t keyframe_id, const Eigen::MatrixXd& descriptor) {
    if (descriptor.size() == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isScanContextDescriptorCompatible(descriptor)) {
        LOG(WARNING) << "[LoopDetector] Skip incompatible ScanContext descriptor for keyframe "
                     << keyframe_id << ": got " << descriptor.rows() << "x" << descriptor.cols()
                     << ", expected " << sc_manager_.totalRows() << "x" << sc_manager_.PC_NUM_SECTOR;
        return;
    }
    if (id_to_index_.count(keyframe_id) > 0) {
        LOG(WARNING) << "[LoopDetector] Duplicate ScanContext descriptor id " << keyframe_id
                     << ", keeping existing descriptor.";
        return;
    }
    size_t index = descriptors_.size();
    id_to_index_[keyframe_id] = index;
    index_to_id_.push_back(keyframe_id);
    descriptors_.push_back(descriptor);
    sc_manager_.polarcontexts_.push_back(descriptor);
    Eigen::MatrixXd ringkey = sc_manager_.makeRingkeyFromScancontext(const_cast<Eigen::MatrixXd&>(descriptor));
    Eigen::MatrixXd sectorkey = sc_manager_.makeSectorkeyFromScancontext(const_cast<Eigen::MatrixXd&>(descriptor));
    sc_manager_.polarcontext_invkeys_.push_back(ringkey);
    sc_manager_.polarcontext_vkeys_.push_back(sectorkey);
    sc_manager_.polarcontext_invkeys_mat_.push_back(eig2stdvec(ringkey));
}

std::vector<LoopCandidate> LoopDetector::detectLoopCandidates(int64_t query_id) {
    std::vector<LoopCandidate> candidates;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = id_to_index_.find(query_id);
    if (it == id_to_index_.end()) return candidates;
    size_t query_index = it->second;

    int num_exclude = config_.sc_num_exclude_recent;
    if (static_cast<int>(query_index) < num_exclude) return candidates;

    size_t search_end = query_index - num_exclude;
    if (search_end == 0) return candidates;

    const Eigen::MatrixXd& query_desc = descriptors_[query_index];
    const bool sc_query_available = query_desc.size() > 0;

    if (config_.rhpd_enabled && rhpd_manager_.size() > 0) {
        Eigen::VectorXd query_rhpd;
        if (rhpd_manager_.get(query_id, &query_rhpd) && query_rhpd.size() == RHPD_DIM && !query_rhpd.isZero()) {
            const int preselect = std::max(config_.rhpd_num_candidates * 3, config_.sc_num_candidates * 4);
            auto accept_old_enough = [&](int64_t id) {
                auto mit = id_to_index_.find(id);
                return mit != id_to_index_.end() && mit->second < search_end;
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

                const bool sc_match_available = match_index < descriptors_.size() && descriptors_[match_index].size() > 0;
                if (config_.rhpd_use_sc_yaw && sc_query_available && sc_match_available) {
                    auto [sc_dist, yaw_shift] = computeDistance(query_desc, descriptors_[match_index]);
                    candidate.sc_distance = sc_dist;
                    candidate.yaw_diff_rad = static_cast<float>(yaw_shift) *
                        static_cast<float>(sc_manager_.PC_UNIT_SECTORANGLE) * static_cast<float>(M_PI / 180.0);
                    candidate.source_flags |= LoopCandidate::SOURCE_SC;
                    candidate.sc_rank = -1;
                    if (config_.sc_aux_veto_enabled && sc_dist > config_.sc_aux_veto_threshold) {
                        continue;
                    }
                }

                const double rhpd_norm = rhpd_dist / std::max(1e-6, config_.rhpd_dist_threshold);
                const double sc_norm = std::isfinite(candidate.sc_distance)
                    ? candidate.sc_distance / std::max(1e-6, config_.sc_aux_veto_threshold)
                    : 1.0;
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

    // RHPD disabled or unavailable: fallback to ScanContext for A/B testing and legacy maps.
    const auto& query_ringkey_vec = sc_manager_.polarcontext_invkeys_mat_[query_index];

    // 构建临时搜索集合（仅前 search_end 帧），避免匹配到最近帧
    KeyMat search_keys(sc_manager_.polarcontext_invkeys_mat_.begin(),
                       sc_manager_.polarcontext_invkeys_mat_.begin() + search_end);
    int tree_dim = sc_manager_.totalRows();
    InvKeyTree search_tree(tree_dim, search_keys, 10 /* max leaf */);

    // KNN 搜索: 取 NUM_CANDIDATES_FROM_TREE 个初筛候选
    constexpr size_t NUM_CANDIDATES_FROM_TREE = 20;  // 比 sc_num_candidates 大，给精排留余量
    size_t num_search = std::min(NUM_CANDIDATES_FROM_TREE, search_end);
    std::vector<size_t> knn_indices(num_search);
    std::vector<float>  knn_dists(num_search);
    nanoflann::KNNResultSet<float> result_set(num_search);
    result_set.init(knn_indices.data(), knn_dists.data());
    search_tree.index->findNeighbors(result_set, query_ringkey_vec.data(),
                                     nanoflann::SearchParams(10));

    if (num_search > 0) {
        VLOG(1) << "[SC] query=" << query_id
                << " search_end=" << search_end
                << " kd_top1_idx=" << index_to_id_[knn_indices[0]]
                << " kd_top1_L2=" << knn_dists[0];
    }

    // -------- 阶段 2: 对 KD-tree 初筛候选做完整多通道距离精排 --------
    std::vector<std::tuple<double, int, size_t>> refined_candidates;
    refined_candidates.reserve(num_search);

    for (size_t k = 0; k < num_search; ++k) {
        size_t match_index = knn_indices[k];
        if (match_index >= search_end) continue;  // 安全检查
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
            static_cast<float>(sc_manager_.PC_UNIT_SECTORANGLE) * static_cast<float>(M_PI / 180.0);
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
    const std::map<int64_t, Keyframe::Ptr>& keyframes) const {
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
    const int min_gap = std::max(1, config_.loop_spatial_candidate_min_id_gap);
    const double radius = std::max(1e-6, config_.loop_spatial_candidate_radius);

    std::vector<std::pair<double, int64_t>> ranked;
    for (const auto& [match_id, keyframe] : keyframes) {
        if (!keyframe || match_id >= query_id) {
            continue;
        }
        if (query_id - match_id < min_gap) {
            continue;
        }
        const double squared_distance =
            (query_pose.translation() - keyframe->pose_optimized.translation()).squaredNorm();
        if (!std::isfinite(squared_distance)) {
            continue;
        }
        ranked.emplace_back(std::sqrt(squared_distance), match_id);
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
    return candidates;
}

VerifiedLoop LoopDetector::verifyLoopCandidate(const LoopCandidate& candidate,
                                                const Keyframe::Ptr& query_keyframe,
                                                const Keyframe::Ptr& match_keyframe,
                                                PointCloudMatcher& matcher) {
    VerifiedLoop result;
    result.query_id = candidate.query_id;
    result.match_id = candidate.match_id;
    result.candidate_yaw_diff_rad = static_cast<double>(candidate.yaw_diff_rad);
    if (!query_keyframe || !match_keyframe) return result;
    Eigen::Isometry3d init_guess = match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    Eigen::AngleAxisd yaw_correction(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ());
    init_guess.linear() = init_guess.linear() * yaw_correction.toRotationMatrix();
    MatchResult match_result = matcher.align(match_keyframe, query_keyframe, init_guess);
    result.fitness_score = match_result.fitness_score;
    result.inlier_ratio = match_result.inlier_ratio;
    result.information = match_result.information;
    result.candidate_residual = init_guess.inverse() * match_result.T_target_source;
    result.verified = match_result.success;
    if (match_result.success) result.T_match_query = match_result.T_target_source;
    return result;
}

std::vector<VerifiedLoop> LoopDetector::verifyLoopCandidatesBatch(
    const std::vector<LoopCandidate>& candidates,
    const std::map<int64_t, Keyframe::Ptr>& keyframes,
    PointCloudMatcher& matcher) {
    std::vector<VerifiedLoop> results(candidates.size());
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto qit = keyframes.find(candidates[i].query_id);
        auto mit = keyframes.find(candidates[i].match_id);
        if (qit != keyframes.end() && mit != keyframes.end())
            results[i] = verifyLoopCandidate(candidates[i], qit->second, mit->second, matcher);
    }
    std::vector<VerifiedLoop> valid;
    for (auto& r : results) if (r.verified) valid.push_back(std::move(r));
    return valid;
}

void LoopDetector::rebuildTree() {
    std::lock_guard<std::mutex> lock(mutex_);
    rebuildTreeUnlocked();
}

void LoopDetector::rebuildTreeUnlocked() {
    sc_manager_.polarcontexts_.clear();
    sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear();
    sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_invkeys_to_search_.clear();
    sc_manager_.polarcontext_tree_.reset();
    for (size_t i = 0; i < descriptors_.size(); ++i) {
        Eigen::MatrixXd desc = descriptors_[i];
        sc_manager_.polarcontexts_.push_back(desc);
        sc_manager_.polarcontext_invkeys_.push_back(sc_manager_.makeRingkeyFromScancontext(desc));
        sc_manager_.polarcontext_vkeys_.push_back(sc_manager_.makeSectorkeyFromScancontext(desc));
        sc_manager_.polarcontext_invkeys_mat_.push_back(eig2stdvec(sc_manager_.polarcontext_invkeys_.back()));
    }
    if (!sc_manager_.polarcontext_invkeys_mat_.empty()) {
        sc_manager_.polarcontext_invkeys_to_search_ = sc_manager_.polarcontext_invkeys_mat_;
        sc_manager_.polarcontext_tree_.reset(
            new InvKeyTree(sc_manager_.totalRows(), sc_manager_.polarcontext_invkeys_to_search_, 10));
    }
}

std::vector<std::pair<int64_t, Eigen::MatrixXd>> LoopDetector::getDescriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<int64_t, Eigen::MatrixXd>> result;
    result.reserve(descriptors_.size());
    for (size_t i = 0; i < descriptors_.size(); ++i) result.emplace_back(index_to_id_[i], descriptors_[i]);
    return result;
}

bool LoopDetector::isScanContextDescriptorCompatible(const Eigen::MatrixXd& descriptor) const {
    return descriptor.size() > 0 &&
           descriptor.rows() == sc_manager_.totalRows() &&
           descriptor.cols() == sc_manager_.PC_NUM_SECTOR;
}

void LoopDetector::loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors) {
    std::lock_guard<std::mutex> lock(mutex_);
    id_to_index_.clear(); index_to_id_.clear(); descriptors_.clear();
    sc_manager_.polarcontexts_.clear(); sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear(); sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_invkeys_to_search_.clear();
    sc_manager_.polarcontext_tree_.reset();
    for (const auto& [id, desc] : descriptors) {
        if (id_to_index_.count(id) > 0) {
            LOG(WARNING) << "[LoopDetector] Skip duplicate ScanContext descriptor id " << id;
            continue;
        }
        if (!isScanContextDescriptorCompatible(desc)) {
            LOG(WARNING) << "[LoopDetector] Skip incompatible ScanContext descriptor for keyframe "
                         << id << ": got " << desc.rows() << "x" << desc.cols()
                         << ", expected " << sc_manager_.totalRows() << "x" << sc_manager_.PC_NUM_SECTOR;
            continue;
        }
        size_t index = descriptors_.size();
        id_to_index_[id] = index;
        index_to_id_.push_back(id);
        descriptors_.push_back(desc);
        sc_manager_.polarcontexts_.push_back(desc);
        Eigen::MatrixXd desc_copy = desc;
        sc_manager_.polarcontext_invkeys_.push_back(sc_manager_.makeRingkeyFromScancontext(desc_copy));
        sc_manager_.polarcontext_vkeys_.push_back(sc_manager_.makeSectorkeyFromScancontext(desc_copy));
        sc_manager_.polarcontext_invkeys_mat_.push_back(eig2stdvec(sc_manager_.polarcontext_invkeys_.back()));
    }
    if (!sc_manager_.polarcontext_invkeys_mat_.empty()) {
        sc_manager_.polarcontext_invkeys_to_search_ = sc_manager_.polarcontext_invkeys_mat_;
        sc_manager_.polarcontext_tree_.reset(
            new InvKeyTree(sc_manager_.totalRows(), sc_manager_.polarcontext_invkeys_to_search_, 10));
    }
}

void LoopDetector::swapWith(LoopDetector& other) {
    if (this == &other) return;
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lock_this(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
    using std::swap;
    swap(config_, other.config_);
    swapScanContextConfig(sc_manager_, other.sc_manager_);
    rhpd_manager_.swapWith(other.rhpd_manager_);
    swap(id_to_index_, other.id_to_index_);
    swap(index_to_id_, other.index_to_id_);
    swap(descriptors_, other.descriptors_);
    rebuildTreeUnlocked();
    other.rebuildTreeUnlocked();
}

size_t LoopDetector::size() const { std::lock_guard<std::mutex> lock(mutex_); return descriptors_.size(); }

void LoopDetector::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    id_to_index_.clear(); index_to_id_.clear(); descriptors_.clear();
    sc_manager_.polarcontexts_.clear(); sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear(); sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_invkeys_to_search_.clear();
    sc_manager_.polarcontext_tree_.reset();
    rhpd_manager_.clear();
}

std::pair<int, int> LoopDetector::getDescriptorDimensions() const {
    return { sc_manager_.totalRows(), sc_manager_.PC_NUM_SECTOR };
}

Eigen::MatrixXd LoopDetector::getDescriptor(int64_t keyframe_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = id_to_index_.find(keyframe_id);
    return (it != id_to_index_.end()) ? descriptors_[it->second] : Eigen::MatrixXd();
}

std::pair<double, int> LoopDetector::computeDistance(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
    Eigen::MatrixXd a = sc1, b = sc2;
    return sc_manager_.distanceBtnScanContext(a, b);
}

// ---- RHPD methods ----

Eigen::VectorXd LoopDetector::computeRHPD(const PointCloudT::Ptr& cloud) const {
    return rhpd_manager_.compute(cloud);
}

Eigen::VectorXd LoopDetector::addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    return rhpd_manager_.addCloud(kf_id, cloud);
}

void LoopDetector::loadRHPDDescriptors(const std::vector<std::pair<int64_t, Eigen::VectorXd>>& descriptors) {
    std::lock_guard<std::mutex> lock(mutex_);
    rhpd_manager_.loadAll(descriptors);
}

void LoopDetector::clearRHPD() {
    std::lock_guard<std::mutex> lock(mutex_);
    rhpd_manager_.clear();
}

} // namespace n3mapping
