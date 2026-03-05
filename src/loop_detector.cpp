// LoopDetector: ScanContext descriptor generation, candidate search, and ICP verification.
#include "n3mapping/loop_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
} // anonymous namespace

namespace n3mapping {

LoopDetector::LoopDetector(const Config& config) : config_(config) {}

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

    // -------- 阶段 1: KD-tree 用 ringkey 快速初筛 --------
    // 构建仅包含可搜索帧 (排除最近帧) 的 KD-tree
    size_t search_end = query_index - num_exclude;
    if (search_end == 0) return candidates;

    // 提取 query 的 ringkey
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

    // Diagnostic: log KD-tree top-1
    if (num_search > 0) {
        LOG(INFO) << "[SC] query=" << query_id
                  << " search_end=" << search_end
                  << " kd_top1_idx=" << index_to_id_[knn_indices[0]]
                  << " kd_top1_L2=" << knn_dists[0];
    }

    // -------- 阶段 2: 对 KD-tree 初筛候选做完整多通道距离精排 --------
    const Eigen::MatrixXd& query_desc = descriptors_[query_index];
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

    // Diagnostic: log refined top-1
    if (!refined_candidates.empty()) {
        LOG(INFO) << "[SC] query=" << query_id
                  << " refined_top1_match=" << index_to_id_[std::get<2>(refined_candidates[0])]
                  << " refined_top1_dist=" << std::get<0>(refined_candidates[0])
                  << " threshold=" << config_.sc_dist_threshold;
    }

    // 输出最终候选
    int num_candidates = std::min(config_.sc_num_candidates,
                                  static_cast<int>(refined_candidates.size()));
    for (int i = 0; i < num_candidates; ++i) {
        double dist = std::get<0>(refined_candidates[i]);
        if (dist >= config_.sc_dist_threshold) break;
        int yaw_shift = std::get<1>(refined_candidates[i]);
        size_t match_index = std::get<2>(refined_candidates[i]);

        LoopCandidate candidate;
        candidate.query_id = query_id;
        candidate.match_id = index_to_id_[match_index];
        candidate.sc_distance = dist;
        candidate.yaw_diff_rad = static_cast<float>(yaw_shift) *
            static_cast<float>(sc_manager_.PC_UNIT_SECTORANGLE) * static_cast<float>(M_PI / 180.0);
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
    if (!query_keyframe || !match_keyframe) return result;
    Eigen::Isometry3d init_guess = match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    Eigen::AngleAxisd yaw_correction(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ());
    init_guess.prerotate(yaw_correction);
    MatchResult match_result = matcher.align(match_keyframe, query_keyframe, init_guess);
    result.fitness_score = match_result.fitness_score;
    result.inlier_ratio = match_result.inlier_ratio;
    result.information = match_result.information;
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
    sc_manager_.polarcontexts_.clear();
    sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear();
    sc_manager_.polarcontext_invkeys_mat_.clear();
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

void LoopDetector::loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors) {
    std::lock_guard<std::mutex> lock(mutex_);
    id_to_index_.clear(); index_to_id_.clear(); descriptors_.clear();
    sc_manager_.polarcontexts_.clear(); sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear(); sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_tree_.reset();
    for (const auto& [id, desc] : descriptors) {
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

size_t LoopDetector::size() const { std::lock_guard<std::mutex> lock(mutex_); return descriptors_.size(); }

void LoopDetector::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    id_to_index_.clear(); index_to_id_.clear(); descriptors_.clear();
    sc_manager_.polarcontexts_.clear(); sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear(); sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_tree_.reset();
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
    return rhpd_manager_.addCloud(kf_id, cloud);
}

} // namespace n3mapping
