#include "n3mapping/RHPDescriptor.h"
#include "rhpd_heading_accumulator.h"
#include <algorithm>
#include <cmath>
#include <optional>

namespace n3mapping {
void RHPDescriptor::computePolar(const CloudT::Ptr& cloud,
                                               double pca_confidence,
                                               Place& place,
                                               RHPDHeadingPtr* heading) const {
    const int NR = RHPD_RINGS;
    const int NZ = RHPD_ZLAYERS;
    const int AB = params_.azimuth_bins;
    const double R  = params_.max_range;
    const double Zn = params_.z_min;
    const double Zx = params_.z_max;
    const double dz = (Zx - Zn) / NZ;

    constexpr int cells = RHPD_RINGS * RHPD_ZLAYERS;
    std::vector<double> azimuth_hist(cells * AB, 0.0);
    std::array<double, cells> cell_count{}, cell_r_sum{}, cell_r_sq_sum{};

    const int S = RHPD_NEG_SPACE_SECTORS;
    std::array<double, S> nearest, farthest{};
    nearest.fill(params_.max_range);
    std::array<int, S> neg_counts{};
    std::optional<detail::HeadingAccumulator> heading_accumulator;
    if (heading) heading_accumulator.emplace(params_.heading);
    for (const auto& pt : cloud->points) {
        // Ring-height and heading share sqrt(float) and atan2(float).
        // Auxiliary ranges retain hypot(float), including its boundary rounding.
        const double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        const float azimuth = std::atan2(pt.y, pt.x);
        if (heading_accumulator) heading_accumulator->add(pt, r, azimuth);
        if (r > 0.1 && r < R && pt.z >= Zn && pt.z < Zx) {
            int ri = static_cast<int>(r / R * NR);
            ri = std::clamp(ri, 0, NR - 1);

            int zi = static_cast<int>((pt.z - Zn) / dz);
            zi = std::clamp(zi, 0, NZ - 1);

            const double az = azimuth;
            int ai = static_cast<int>((az + M_PI) / (2.0 * M_PI) * AB);
            ai = std::clamp(ai, 0, AB - 1);

            azimuth_hist[(ri * NZ + zi) * AB + ai] += 1.0;
            cell_count[ri * NZ + zi] += 1.0;
            cell_r_sum[ri * NZ + zi] += r;
            cell_r_sq_sum[ri * NZ + zi] += r * r;
        }
        if (!params_.enable_negative_space) continue;
        const double aux_range = std::hypot(pt.x, pt.y);
        if (aux_range <= 0.1 || aux_range >= R) continue;
        const double az = azimuth;
        const int sector = std::clamp(static_cast<int>((az + M_PI) / (2.0 * M_PI) * S), 0, S - 1);
        nearest[sector] = std::min(nearest[sector], aux_range);
        farthest[sector] = std::max(farthest[sector], aux_range);
        ++neg_counts[sector];
    }

    double total = 0.0;
    size_t active = 0;
    for (double count : cell_count) {
        total += count;
        active += count > 0.0;
    }
    place.ring_height.indices.reserve(active);
    place.ring_height.values.reserve(active);
    double inv_total = total > 0 ? 1.0 / total : 0.0;

    auto& out = place.auxiliary;
    for (int cell = 0; cell < cells; ++cell) {
        double density = cell_count[cell] * inv_total;

        double az_var = 0.0;
        if (cell_count[cell] > 0) {
            double mean_count = cell_count[cell] / AB;
            double sum_sq = 0.0;
            for (int a = 0; a < AB; ++a) {
                double diff = azimuth_hist[cell * AB + a] - mean_count;
                sum_sq += diff * diff;
            }
            az_var = (mean_count > 0) ? (sum_sq / AB) / (mean_count * mean_count) : 0.0;
            az_var = std::min(az_var, 10.0) / 10.0;
        }

        double mean_range = 0.0;
        double range_var = 0.0;
        if (cell_count[cell] > 0) {
            mean_range = cell_r_sum[cell] / cell_count[cell];
            double mean_r_sq = cell_r_sq_sum[cell] / cell_count[cell];
            range_var = mean_r_sq - mean_range * mean_range;
            if (range_var < 0.0) range_var = 0.0;
            mean_range /= R;
            range_var = std::sqrt(range_var) / R;
        }

        place.ring_height.append(cell,
            Eigen::Vector4d(density, az_var, mean_range, range_var));
    }

    int offset = 0;
    if (params_.enable_negative_space) {
        for (int i = 0; i < S; ++i) {
            const bool hit = neg_counts[i] > 0;
            const bool left_hit = neg_counts[(i + S - 1) % S] > 0;
            const bool right_hit = neg_counts[(i + 1) % S] > 0;
            const bool neighbor_observed = left_hit || right_hit;
            const double nearest_norm = hit ? std::clamp(nearest[i] / params_.max_range, 0.0, 1.0) : 1.0;
            const double frontier_span_norm = hit
                ? std::clamp((farthest[i] - nearest[i]) / params_.max_range, 0.0, 1.0)
                : 0.0;
            double open_unknown_state = 0.5;  // no hit and no neighboring support: unknown
            if (hit) {
                open_unknown_state = (farthest[i] >= 0.75 * params_.max_range) ? 1.0 : 0.0;
            } else if (neighbor_observed) {
                open_unknown_state = 0.75;  // likely open gap between observed sectors
            }
            out(offset + i * RHPD_NEG_SPACE_CHANS + 0) = nearest_norm;
            out(offset + i * RHPD_NEG_SPACE_CHANS + 1) = frontier_span_norm;
            out(offset + i * RHPD_NEG_SPACE_CHANS + 2) = open_unknown_state;
        }
    }
    offset += RHPD_NEG_SPACE_DIM;
    out(offset) = params_.enable_pca_confidence ? std::clamp(pca_confidence, 0.0, 1.0) : 1.0;
    if (heading) *heading = heading_accumulator->finish();
}
} // namespace n3mapping
