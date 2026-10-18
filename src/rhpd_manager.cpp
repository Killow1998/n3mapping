#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include <algorithm>
#include <utility>
#include <glog/logging.h>

namespace n3mapping {
void RHPDManager::add(int64_t kf_id, const VecD& descriptor) {
    if (descriptor.size() != RHPD_DIM) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(ids_.begin(), ids_.end(), kf_id) != ids_.end()) {
        LOG(WARNING) << "[RHPDManager] Duplicate RHPD descriptor id " << kf_id
                     << ", keeping existing descriptor.";
        return;
    }
    ids_.push_back(kf_id);
    descriptors_.push_back(descriptor);
    coarse_keys_.push_back(makeCoarseKey(descriptor));
}

RHPDManager::VecD RHPDManager::addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    VecD descriptor = desc_.compute(cloud);
    add(kf_id, descriptor);
    return descriptor;
}

std::vector<std::pair<int64_t, double>> RHPDManager::search(const VecD& query, int top_k) const {
    return search(query, top_k, 0);
}

std::vector<std::pair<int64_t, double>> RHPDManager::search(const VecD& query, int top_k, int preselect) const {
    return searchFiltered(query, top_k, preselect, nullptr);
}

std::vector<std::pair<int64_t, double>> RHPDManager::searchFiltered(
    const VecD& query,
    int top_k,
    int preselect,
    const std::function<bool(int64_t)>& accept) const {
    std::vector<std::pair<int64_t, double>> results;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ids_.empty() || query.size() != RHPD_DIM) return results;

    const VecD query_key = makeCoarseKey(query);
    const int safe_top_k = std::max(1, top_k);
    int candidate_count = static_cast<int>(ids_.size());
    if (preselect > 0) {
        candidate_count = std::min(candidate_count, std::max(safe_top_k, preselect));
    }

    std::vector<std::pair<double, size_t>> coarse_ranked;
    coarse_ranked.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (accept && !accept(ids_[i])) continue;
        if (descriptors_[i].size() != RHPD_DIM || coarse_keys_[i].size() != RHPD_COARSE_KEY_DIM) continue;
        coarse_ranked.emplace_back((query_key - coarse_keys_[i]).squaredNorm(), i);
    }

    if (candidate_count < static_cast<int>(coarse_ranked.size())) {
        std::nth_element(coarse_ranked.begin(),
                         coarse_ranked.begin() + candidate_count,
                         coarse_ranked.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        coarse_ranked.resize(candidate_count);
    }

    results.reserve(coarse_ranked.size());
    for (const auto& item : coarse_ranked) {
        const size_t i = item.second;
        const double d = desc_.distance(query, descriptors_[i]);
        results.emplace_back(ids_[i], d);
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (static_cast<int>(results.size()) > safe_top_k) {
        results.resize(safe_top_k);
    }
    return results;
}

bool RHPDManager::get(int64_t kf_id, VecD* descriptor) const {
    if (!descriptor) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (ids_[i] == kf_id) {
            *descriptor = descriptors_[i];
            return true;
        }
    }
    return false;
}

void RHPDManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.clear();
    descriptors_.clear();
    coarse_keys_.clear();
}

void RHPDManager::swapWith(RHPDManager& other) {
    if (this == &other) return;
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lock_this(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
    std::swap(desc_, other.desc_);
    std::swap(ids_, other.ids_);
    std::swap(descriptors_, other.descriptors_);
    std::swap(coarse_keys_, other.coarse_keys_);
}

std::vector<std::pair<int64_t, RHPDManager::VecD>> RHPDManager::getAll() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<int64_t, VecD>> out;
    out.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i) {
        out.emplace_back(ids_[i], descriptors_[i]);
    }
    return out;
}

void RHPDManager::loadAll(const std::vector<std::pair<int64_t, VecD>>& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.clear();
    descriptors_.clear();
    coarse_keys_.clear();
    for (const auto& [id, desc] : data) {
        if (std::find(ids_.begin(), ids_.end(), id) != ids_.end()) {
            LOG(WARNING) << "[RHPDManager] Skip duplicate RHPD descriptor id " << id;
            continue;
        }
        if (desc.size() == RHPD_DIM) {
            ids_.push_back(id);
            descriptors_.push_back(desc);
            coarse_keys_.push_back(makeCoarseKey(desc));
        }
    }
}

size_t RHPDManager::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.size();
}

RHPDManager::VecD RHPDManager::makeCoarseKey(const VecD& descriptor) const {
    VecD key = VecD::Zero(RHPD_COARSE_KEY_DIM);
    if (descriptor.size() != RHPD_DIM) return key;

    int out = 0;
    const int part_b_base = RHPD_PART_A_DIM;
    for (int r = 0; r < RHPD_RINGS; ++r) {
        double density_sum = 0.0;
        for (int z = 0; z < RHPD_ZLAYERS; ++z) {
            const int idx = part_b_base + (r * RHPD_ZLAYERS + z) * RHPD_RH_CHANS;
            density_sum += descriptor(idx);
        }
        key(out++) = density_sum;
    }

    const int neg_base = RHPD_PART_A_DIM + RHPD_PART_B_DIM;
    for (int s = 0; s < RHPD_NEG_SPACE_SECTORS; ++s) {
        const int idx = neg_base + s * RHPD_NEG_SPACE_CHANS;
        key(out++) = 0.6 * descriptor(idx) + 0.25 * descriptor(idx + 1) + 0.15 * descriptor(idx + 2);
    }

    const int token_base = neg_base + RHPD_NEG_SPACE_DIM;
    key.segment(out, RHPD_VERTICAL_TOKEN_DIM) = descriptor.segment(token_base, RHPD_VERTICAL_TOKEN_DIM);
    out += RHPD_VERTICAL_TOKEN_DIM;

    key(out) = descriptor(RHPD_DIM - 1);
    return key;
}

RHPDescriptor::Params rhpdParamsFromConfig(const Config& config) {
  RHPDescriptor::Params params;
  params.max_range = std::max(1.0, config.rhpd_max_range);
  params.z_min = config.rhpd_z_min;
  params.z_max = std::max(config.rhpd_z_max, config.rhpd_z_min + 1e-3);
  params.v2_enable = config.rhpd_v2_enable;
  params.v3_enable = config.rhpd_v3_enable;
  params.enable_negative_space = config.rhpd_enable_negative_space;
  params.enable_vertical_tokens = config.rhpd_enable_vertical_tokens;
  params.enable_pca_confidence = config.rhpd_enable_pca_confidence;
  params.part_a_scale = config.rhpd_part_a_scale;
  params.aux_scale = config.rhpd_aux_scale;
  return params;
}


} // namespace n3mapping
