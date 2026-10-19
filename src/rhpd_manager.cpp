#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include <algorithm>
#include <utility>
#include <glog/logging.h>

namespace n3mapping {
void RHPDManager::add(int64_t kf_id, const Place& descriptor) {
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

RHPDManager::Place RHPDManager::addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    Place descriptor = desc_.compute(cloud);
    add(kf_id, descriptor);
    return descriptor;
}

std::vector<std::pair<int64_t, double>> RHPDManager::search(const Place& query, int top_k) const {
    return search(query, top_k, 0);
}

std::vector<std::pair<int64_t, double>> RHPDManager::search(const Place& query, int top_k, int preselect,
    DescriptorSearchPerformance* performance) const {
    return searchFiltered(query, top_k, preselect, nullptr, performance);
}

std::vector<std::pair<int64_t, double>> RHPDManager::searchFiltered(
    const Place& query,
    int top_k,
    int preselect,
    const std::function<bool(int64_t)>& accept,
    DescriptorSearchPerformance* performance) const {
    std::vector<std::pair<int64_t, double>> results;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ids_.empty()) return results;

    DescriptorPhaseTimer coarse_timer(performance ? &performance->coarse : nullptr);
    const VecD query_key = makeCoarseKey(query);
    // Keep the same half-turn symmetry as the full descriptor distance.
    // Otherwise the shortlist can discard the correct reverse-facing place
    // before its heading and geometry are ever evaluated.
    VecD flipped_key = query_key;
    for (int sector = 0; sector < RHPD_NEG_SPACE_SECTORS; ++sector) {
        flipped_key(RHPD_RINGS + sector) = query_key(
            RHPD_RINGS + (sector + RHPD_NEG_SPACE_SECTORS / 2) % RHPD_NEG_SPACE_SECTORS);
    }
    const int safe_top_k = std::max(1, top_k);
    int candidate_count = static_cast<int>(ids_.size());
    if (preselect > 0) {
        candidate_count = std::min(candidate_count, std::max(safe_top_k, preselect));
    }

    std::vector<std::pair<double, size_t>> coarse_ranked;
    coarse_ranked.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i) {
        if (accept && !accept(ids_[i])) continue;
        if (coarse_keys_[i].size() != RHPD_COARSE_KEY_DIM) continue;
        if (performance) ++performance->coarse_comparisons;
        const double distance = std::min(
            (query_key - coarse_keys_[i]).squaredNorm(),
            (flipped_key - coarse_keys_[i]).squaredNorm());
        coarse_ranked.emplace_back(distance, i);
    }

    if (candidate_count < static_cast<int>(coarse_ranked.size())) {
        std::nth_element(coarse_ranked.begin(),
                         coarse_ranked.begin() + candidate_count,
                         coarse_ranked.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        coarse_ranked.resize(candidate_count);
    }

    coarse_timer.finish();
    DescriptorPhaseTimer place_timer(performance ? &performance->place : nullptr);
    if (performance) performance->place_comparisons += coarse_ranked.size();
    const auto prepared = desc_.prepareQuery(query);
    results.reserve(coarse_ranked.size());
    for (const auto& item : coarse_ranked) {
        const size_t i = item.second;
        const double d = desc_.distance(prepared, descriptors_[i]);
        results.emplace_back(ids_[i], d);
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (static_cast<int>(results.size()) > safe_top_k) {
        results.resize(safe_top_k);
    }
    return results;
}

bool RHPDManager::get(int64_t kf_id, Place* descriptor) const {
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

size_t RHPDManager::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.size();
}

RHPDManager::VecD RHPDManager::makeCoarseKey(const Place& descriptor) const {
    VecD key = VecD::Zero(RHPD_COARSE_KEY_DIM);
    for (size_t i = 0; i < descriptor.ring_height.indices.size(); ++i) {
        const int ring = descriptor.ring_height.indices[i] / RHPD_ZLAYERS;
        key(ring) += descriptor.ring_height.values[i](0);
    }
    int out = RHPD_RINGS;
    for (int sector = 0; sector < RHPD_NEG_SPACE_SECTORS; ++sector) {
        const int i = sector * RHPD_NEG_SPACE_CHANS;
        const auto& aux = descriptor.auxiliary;
        key(out++) = 0.6 * aux(i) + 0.25 * aux(i + 1) + 0.15 * aux(i + 2);
    }
    key(out) = descriptor.auxiliary(RHPD_AUX_DIM - 1);
    return key;
}

RHPDescriptor::Params rhpdParamsFromConfig(const Config& config) {
  RHPDescriptor::Params params;
  params.heading.rings = std::max(1, config.sc_num_rings);
  params.heading.sectors = std::max(4, config.sc_num_sectors);
  params.heading.max_radius = std::max(config.sc_max_radius, params.heading.min_radius + 1e-3);
  params.max_range = std::max(1.0, config.rhpd_max_range);
  params.z_min = config.rhpd_z_min;
  params.z_max = std::max(config.rhpd_z_max, config.rhpd_z_min + 1e-3);
  params.v2_enable = config.rhpd_v2_enable;
  params.v3_enable = config.rhpd_v3_enable;
  params.enable_negative_space = config.rhpd_enable_negative_space;
  params.enable_pca_confidence = config.rhpd_enable_pca_confidence;
  params.part_a_scale = config.rhpd_part_a_scale;
  params.aux_scale = config.rhpd_aux_scale;
  return params;
}


} // namespace n3mapping
