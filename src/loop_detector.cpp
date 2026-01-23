#include "n3mapping/loop_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace n3mapping {

LoopDetector::LoopDetector(const Config& config)
  : config_(config)
{
    // ScanContext 参数已在 SCManager 中硬编码
    // 如果需要可配置，需要修改 SCManager 类
}

Eigen::MatrixXd
LoopDetector::makeScanContext(const PointCloudT::Ptr& cloud)
{
    if (!cloud || cloud->empty()) {
        return Eigen::MatrixXd();
    }

    // 使用 SCManager 生成描述子
    pcl::PointCloud<SCPointType> cloud_copy = *cloud;
    return sc_manager_.makeScancontext(cloud_copy);
}

Eigen::MatrixXd
LoopDetector::addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud)
{
    if (!cloud || cloud->empty()) {
        return Eigen::MatrixXd();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 生成描述子
    pcl::PointCloud<SCPointType> cloud_copy = *cloud;
    Eigen::MatrixXd descriptor = sc_manager_.makeScancontext(cloud_copy);

    if (descriptor.size() == 0) {
        return Eigen::MatrixXd();
    }

    // 存储映射关系
    size_t index = descriptors_.size();
    id_to_index_[keyframe_id] = index;
    index_to_id_.push_back(keyframe_id);
    descriptors_.push_back(descriptor);

    // 同时添加到 SCManager 用于 KNN 搜索
    sc_manager_.makeAndSaveScancontextAndKeys(cloud_copy);

    return descriptor;
}

void
LoopDetector::addDescriptor(int64_t keyframe_id, const Eigen::MatrixXd& descriptor)
{
    if (descriptor.size() == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 存储映射关系
    size_t index = descriptors_.size();
    id_to_index_[keyframe_id] = index;
    index_to_id_.push_back(keyframe_id);
    descriptors_.push_back(descriptor);

    // 添加到 SCManager 的内部存储
    sc_manager_.polarcontexts_.push_back(descriptor);

    // 生成 ring key 和 sector key
    Eigen::MatrixXd ringkey = sc_manager_.makeRingkeyFromScancontext(const_cast<Eigen::MatrixXd&>(descriptor));
    Eigen::MatrixXd sectorkey = sc_manager_.makeSectorkeyFromScancontext(const_cast<Eigen::MatrixXd&>(descriptor));

    sc_manager_.polarcontext_invkeys_.push_back(ringkey);
    sc_manager_.polarcontext_vkeys_.push_back(sectorkey);

    // 更新 KNN 搜索用的矩阵
    std::vector<float> ringkey_vec = eig2stdvec(ringkey);
    sc_manager_.polarcontext_invkeys_mat_.push_back(ringkey_vec);
}

std::vector<LoopCandidate>
LoopDetector::detectLoopCandidates(int64_t query_id)
{
    std::vector<LoopCandidate> candidates;

    std::lock_guard<std::mutex> lock(mutex_);

    // 检查查询帧是否存在
    auto it = id_to_index_.find(query_id);
    if (it == id_to_index_.end()) {
        return candidates;
    }

    size_t query_index = it->second;

    // 检查是否有足够的历史帧
    int num_exclude = config_.sc_num_exclude_recent;
    if (static_cast<int>(query_index) < num_exclude) {
        return candidates; // 历史帧不足
    }

    const Eigen::MatrixXd& query_desc = descriptors_[query_index];

    // 搜索候选帧
    // 只在 [0, query_index - num_exclude) 范围内搜索
    size_t search_end = query_index - num_exclude;

    // 收集所有候选及其距离
    std::vector<std::tuple<double, int, size_t>> all_candidates; // (distance, yaw_shift, index)

    for (size_t i = 0; i < search_end; ++i) {
        auto [dist, yaw_shift] = computeDistance(query_desc, descriptors_[i]);
        all_candidates.emplace_back(dist, yaw_shift, i);
    }

    // 按距离排序
    std::sort(all_candidates.begin(), all_candidates.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    // 取 Top-K 候选
    int num_candidates = std::min(config_.sc_num_candidates, static_cast<int>(all_candidates.size()));

    for (int i = 0; i < num_candidates; ++i) {
        double dist = std::get<0>(all_candidates[i]);
        int yaw_shift = std::get<1>(all_candidates[i]);
        size_t match_index = std::get<2>(all_candidates[i]);

        // 只返回距离小于阈值的候选
        if (dist < config_.sc_dist_threshold) {
            LoopCandidate candidate;
            candidate.query_id = query_id;
            candidate.match_id = index_to_id_[match_index];
            candidate.sc_distance = dist;
            // 计算 yaw 差异 (弧度)
            candidate.yaw_diff_rad =
              static_cast<float>(yaw_shift) * static_cast<float>(sc_manager_.PC_UNIT_SECTORANGLE) * static_cast<float>(M_PI / 180.0);
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

void
LoopDetector::rebuildTree()
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 清空并重建 SCManager 的内部数据
    sc_manager_.polarcontexts_.clear();
    sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear();
    sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_tree_.reset();

    // 重新添加所有描述子
    for (size_t i = 0; i < descriptors_.size(); ++i) {
        Eigen::MatrixXd desc = descriptors_[i];
        sc_manager_.polarcontexts_.push_back(desc);

        Eigen::MatrixXd ringkey = sc_manager_.makeRingkeyFromScancontext(desc);
        Eigen::MatrixXd sectorkey = sc_manager_.makeSectorkeyFromScancontext(desc);

        sc_manager_.polarcontext_invkeys_.push_back(ringkey);
        sc_manager_.polarcontext_vkeys_.push_back(sectorkey);

        std::vector<float> ringkey_vec = eig2stdvec(ringkey);
        sc_manager_.polarcontext_invkeys_mat_.push_back(ringkey_vec);
    }

    // 重建 KD 树
    if (!sc_manager_.polarcontext_invkeys_mat_.empty()) {
        sc_manager_.polarcontext_invkeys_to_search_ = sc_manager_.polarcontext_invkeys_mat_;
        sc_manager_.polarcontext_tree_.reset(
          new InvKeyTree(sc_manager_.PC_NUM_RING, sc_manager_.polarcontext_invkeys_to_search_, 10 /* max leaf */));
    }
}

std::vector<std::pair<int64_t, Eigen::MatrixXd>>
LoopDetector::getDescriptors() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::pair<int64_t, Eigen::MatrixXd>> result;
    result.reserve(descriptors_.size());

    for (size_t i = 0; i < descriptors_.size(); ++i) {
        result.emplace_back(index_to_id_[i], descriptors_[i]);
    }

    return result;
}

void
LoopDetector::loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors)
{

    std::lock_guard<std::mutex> lock(mutex_);

    // 清空现有数据
    id_to_index_.clear();
    index_to_id_.clear();
    descriptors_.clear();

    // 清空 SCManager
    sc_manager_.polarcontexts_.clear();
    sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear();
    sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_tree_.reset();

    // 加载描述子并添加到 SCManager
    for (const auto& [id, desc] : descriptors) {
        size_t index = descriptors_.size();
        id_to_index_[id] = index;
        index_to_id_.push_back(id);
        descriptors_.push_back(desc);

        // 将描述子添加到 SCManager
        sc_manager_.polarcontexts_.push_back(desc);

        // 生成 ring key 和 sector key
        Eigen::MatrixXd desc_copy = desc; // makeRingkeyFromScancontext 需要非 const 引用
        Eigen::MatrixXd ringkey = sc_manager_.makeRingkeyFromScancontext(desc_copy);
        Eigen::MatrixXd sectorkey = sc_manager_.makeSectorkeyFromScancontext(desc_copy);

        sc_manager_.polarcontext_invkeys_.push_back(ringkey);
        sc_manager_.polarcontext_vkeys_.push_back(sectorkey);

        // 转换 ringkey 为 vector 并添加到 mat
        std::vector<float> ringkey_vec = eig2stdvec(ringkey);
        sc_manager_.polarcontext_invkeys_mat_.push_back(ringkey_vec);
    }

    // 重建 KD 树
    if (!sc_manager_.polarcontext_invkeys_mat_.empty()) {
        sc_manager_.polarcontext_invkeys_to_search_ = sc_manager_.polarcontext_invkeys_mat_;
        sc_manager_.polarcontext_tree_.reset(new InvKeyTree(sc_manager_.PC_NUM_RING, sc_manager_.polarcontext_invkeys_to_search_, 10));
    }
}

size_t
LoopDetector::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_.size();
}

void
LoopDetector::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);

    id_to_index_.clear();
    index_to_id_.clear();
    descriptors_.clear();

    // 清空 SCManager
    sc_manager_.polarcontexts_.clear();
    sc_manager_.polarcontext_invkeys_.clear();
    sc_manager_.polarcontext_vkeys_.clear();
    sc_manager_.polarcontext_invkeys_mat_.clear();
    sc_manager_.polarcontext_tree_.reset();
}

std::pair<int, int>
LoopDetector::getDescriptorDimensions() const
{
    return { sc_manager_.PC_NUM_RING, sc_manager_.PC_NUM_SECTOR };
}

Eigen::MatrixXd
LoopDetector::getDescriptor(int64_t keyframe_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = id_to_index_.find(keyframe_id);
    if (it == id_to_index_.end()) {
        return Eigen::MatrixXd();
    }

    return descriptors_[it->second];
}

std::pair<double, int>
LoopDetector::computeDistance(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2)
{
    // 使用 SCManager 的距离计算函数
    Eigen::MatrixXd sc1_copy = sc1;
    Eigen::MatrixXd sc2_copy = sc2;
    return sc_manager_.distanceBtnScanContext(sc1_copy, sc2_copy);
}

VerifiedLoop
LoopDetector::verifyLoopCandidate(const LoopCandidate& candidate,
                                  const Keyframe::Ptr& query_keyframe,
                                  const Keyframe::Ptr& match_keyframe,
                                  PointCloudMatcher& matcher)
{
    VerifiedLoop result;
    result.query_id = candidate.query_id;
    result.match_id = candidate.match_id;
    result.verified = false;

    // 检查输入有效性
    if (!query_keyframe || !match_keyframe) {
        return result;
    }

    if (!query_keyframe->cloud || !match_keyframe->cloud) {
        return result;
    }

    if (query_keyframe->cloud->empty() || match_keyframe->cloud->empty()) {
        return result;
    }

    // 构建初始位姿猜测
    // 使用 ScanContext 估计的 yaw 差异作为初始猜测
    Eigen::Isometry3d init_guess = Eigen::Isometry3d::Identity();
    init_guess.rotate(Eigen::AngleAxisd(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ()));

    // 执行 ICP 配准
    MatchResult match_result = matcher.align(match_keyframe, query_keyframe, init_guess);

    result.fitness_score = match_result.fitness_score;
    result.T_match_query = match_result.T_target_source;
    result.information = match_result.information;

    // 验证配准结果
    if (match_result.success) {
        result.verified = true;
    }

    return result;
}

std::vector<VerifiedLoop>
LoopDetector::verifyLoopCandidatesBatch(const std::vector<LoopCandidate>& candidates,
                                        const std::map<int64_t, Keyframe::Ptr>& keyframes,
                                        PointCloudMatcher& matcher)
{

    std::vector<VerifiedLoop> results;

    if (candidates.empty()) {
        return results;
    }

    // 预分配结果空间
    results.resize(candidates.size());

// 使用 OpenMP 并行验证
// Requirements: 4.8
#ifdef _OPENMP
#pragma omp parallel for num_threads(config_.num_threads)
#endif
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];

        // 获取关键帧
        auto query_it = keyframes.find(candidate.query_id);
        auto match_it = keyframes.find(candidate.match_id);

        if (query_it == keyframes.end() || match_it == keyframes.end()) {
            results[i].query_id = candidate.query_id;
            results[i].match_id = candidate.match_id;
            results[i].verified = false;
            continue;
        }

        // 验证候选
        results[i] = verifyLoopCandidate(candidate, query_it->second, match_it->second, matcher);
    }

    // 过滤出验证通过的结果
    std::vector<VerifiedLoop> verified_results;
    verified_results.reserve(results.size());

    for (const auto& result : results) {
        if (result.verified) {
            verified_results.push_back(result);
        }
    }

    return verified_results;
}

} // namespace n3mapping
