#include "n3mapping/RHPDescriptor.h"
#include <algorithm>
#include <cmath>

namespace n3mapping {
RHPDescriptor::VecD RHPDescriptor::computePartB(const CloudT::Ptr& cloud) const {
    const int NR = RHPD_RINGS;
    const int NZ = RHPD_ZLAYERS;
    const int AB = params_.azimuth_bins;
    const double R  = params_.max_range;
    const double Zn = params_.z_min;
    const double Zx = params_.z_max;
    const double dz = (Zx - Zn) / NZ;

    std::vector<std::vector<std::vector<double>>> azimuth_hist(
        NR, std::vector<std::vector<double>>(NZ, std::vector<double>(AB, 0.0)));
    std::vector<std::vector<double>> cell_count(NR, std::vector<double>(NZ, 0.0));
    std::vector<std::vector<double>> cell_r_sum(NR, std::vector<double>(NZ, 0.0));
    std::vector<std::vector<double>> cell_r_sq_sum(NR, std::vector<double>(NZ, 0.0));

    for (const auto& pt : cloud->points) {
        double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        if (r <= 0.1 || r >= R) continue;
        if (pt.z < Zn || pt.z >= Zx) continue;

        int ri = static_cast<int>(r / R * NR);
        ri = std::clamp(ri, 0, NR - 1);

        int zi = static_cast<int>((pt.z - Zn) / dz);
        zi = std::clamp(zi, 0, NZ - 1);

        double az = std::atan2(pt.y, pt.x);
        int ai = static_cast<int>((az + M_PI) / (2.0 * M_PI) * AB);
        ai = std::clamp(ai, 0, AB - 1);

        azimuth_hist[ri][zi][ai] += 1.0;
        cell_count[ri][zi] += 1.0;
        cell_r_sum[ri][zi] += r;
        cell_r_sq_sum[ri][zi] += r * r;
    }

    double total = 0.0;
    for (int r = 0; r < NR; ++r)
        for (int z = 0; z < NZ; ++z)
            total += cell_count[r][z];
    double inv_total = total > 0 ? 1.0 / total : 0.0;

    VecD out(RHPD_PART_B_DIM);
    int idx = 0;
    for (int r = 0; r < NR; ++r) {
        for (int z = 0; z < NZ; ++z) {
            double density = cell_count[r][z] * inv_total;

            double az_var = 0.0;
            if (cell_count[r][z] > 0) {
                double mean_count = cell_count[r][z] / AB;
                double sum_sq = 0.0;
                for (int a = 0; a < AB; ++a) {
                    double diff = azimuth_hist[r][z][a] - mean_count;
                    sum_sq += diff * diff;
                }
                az_var = (mean_count > 0) ? (sum_sq / AB) / (mean_count * mean_count) : 0.0;
                az_var = std::min(az_var, 10.0) / 10.0;
            }

            double mean_range = 0.0;
            double range_var = 0.0;
            if (cell_count[r][z] > 0) {
                mean_range = cell_r_sum[r][z] / cell_count[r][z];
                double mean_r_sq = cell_r_sq_sum[r][z] / cell_count[r][z];
                range_var = mean_r_sq - mean_range * mean_range;
                if (range_var < 0.0) range_var = 0.0;
                mean_range /= R;
                range_var = std::sqrt(range_var) / R;
            }

            out(idx++) = density;
            out(idx++) = az_var;
            out(idx++) = mean_range;
            out(idx++) = range_var;
        }
    }
    return out;
}

RHPDescriptor::VecD RHPDescriptor::computeAux(const CloudT::Ptr& cloud, double pca_confidence) const {
    VecD out = VecD::Zero(RHPD_AUX_DIM);
    if (!cloud || cloud->empty()) {
        return out;
    }

    int offset = 0;
    if (params_.enable_negative_space) {
        const int S = RHPD_NEG_SPACE_SECTORS;
        std::vector<double> nearest(S, params_.max_range);
        std::vector<double> farthest(S, 0.0);
        std::vector<int> counts(S, 0);
        for (const auto& pt : cloud->points) {
            const double r = std::hypot(pt.x, pt.y);
            if (r <= 0.1 || r >= params_.max_range) continue;
            const double az = std::atan2(pt.y, pt.x);
            int si = static_cast<int>((az + M_PI) / (2.0 * M_PI) * S);
            si = std::clamp(si, 0, S - 1);
            nearest[si] = std::min(nearest[si], r);
            farthest[si] = std::max(farthest[si], r);
            ++counts[si];
        }

        for (int i = 0; i < S; ++i) {
            const bool hit = counts[i] > 0;
            const bool left_hit = counts[(i + S - 1) % S] > 0;
            const bool right_hit = counts[(i + 1) % S] > 0;
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

    if (params_.enable_vertical_tokens) {
        const int CR = RHPD_VERTICAL_TOKEN_RINGS;
        const double dz = (params_.z_max - params_.z_min) / 4.0;
        std::vector<std::vector<int>> ring_sector_tokens(
            CR, std::vector<int>(RHPD_NEG_SPACE_SECTORS, 0));
        for (const auto& pt : cloud->points) {
            const double r = std::hypot(pt.x, pt.y);
            if (r <= 0.1 || r >= params_.max_range) continue;
            if (pt.z < params_.z_min || pt.z >= params_.z_max) continue;
            int ri = static_cast<int>(r / params_.max_range * CR);
            ri = std::clamp(ri, 0, CR - 1);
            const double az = std::atan2(pt.y, pt.x);
            int si = static_cast<int>((az + M_PI) / (2.0 * M_PI) * RHPD_NEG_SPACE_SECTORS);
            si = std::clamp(si, 0, RHPD_NEG_SPACE_SECTORS - 1);
            int zi = static_cast<int>((pt.z - params_.z_min) / dz);
            zi = std::clamp(zi, 0, 3);
            ring_sector_tokens[ri][si] |= (1 << zi);
        }
        for (int r = 0; r < CR; ++r) {
            double active = 0.0;
            const int ring_base = offset + r * RHPD_VERTICAL_TOKEN_BINS;
            for (int token : ring_sector_tokens[r]) {
                if (token == 0) continue;
                out(ring_base + token) += 1.0;
                active += 1.0;
            }
            if (active > 0.0) {
                out.segment(ring_base, RHPD_VERTICAL_TOKEN_BINS) /= active;
            }
        }
    }
    offset += RHPD_VERTICAL_TOKEN_DIM;

    out(offset) = params_.enable_pca_confidence ? std::clamp(pca_confidence, 0.0, 1.0) : 1.0;
    return out;
}


} // namespace n3mapping
