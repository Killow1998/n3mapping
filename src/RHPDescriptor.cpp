// RHPDescriptor: Ring-Height + Planar descriptor with visibility-aware planar cues.
#include "n3mapping/RHPDescriptor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include "n3mapping/pcl_compat.h"
#include <memory>

namespace n3mapping {
namespace {
void flipPlaneAxis(Eigen::VectorXd& v, int base, bool flip_axis0, bool flip_axis1) {
    const int B = RHPD_PLANE_BINS;
    Eigen::VectorXd plane = v.segment(base, RHPD_PLANE_DIM);
    for (int row = 0; row < B; ++row) {
        const int src_row = flip_axis0 ? (B - 1 - row) : row;
        for (int col = 0; col < B; ++col) {
            const int src_col = flip_axis1 ? (B - 1 - col) : col;
            const int dst = base + (row * B + col) * RHPD_PLANE_CHANS;
            const int src = (src_row * B + src_col) * RHPD_PLANE_CHANS;
            for (int c = 0; c < RHPD_PLANE_CHANS; ++c) {
                v(dst + c) = plane(src + c);
            }
        }
    }
}

void shiftNegativeSpace180(Eigen::VectorXd& v) {
    const int base = RHPD_PART_A_DIM + RHPD_PART_B_DIM;
    Eigen::VectorXd neg = v.segment(base, RHPD_NEG_SPACE_DIM);
    const int half = RHPD_NEG_SPACE_SECTORS / 2;
    for (int i = 0; i < RHPD_NEG_SPACE_SECTORS; ++i) {
        const int src_sector = (i + half) % RHPD_NEG_SPACE_SECTORS;
        const int dst = base + i * RHPD_NEG_SPACE_CHANS;
        const int src = src_sector * RHPD_NEG_SPACE_CHANS;
        for (int c = 0; c < RHPD_NEG_SPACE_CHANS; ++c) {
            v(dst + c) = neg(src + c);
        }
    }
}

void shiftVerticalTokens180(Eigen::VectorXd&) {
    // Vertical token histogram is coarse radial, not azimuthal; no yaw shift needed.
}
} // namespace

// ============================================================
//  RHPDescriptor
// ============================================================

RHPDescriptor::RHPDescriptor(const Params& params) : params_(params) {}

// ---- PCA alignment ----

Eigen::Matrix3d RHPDescriptor::computePCAAlignment(const CloudT::Ptr& cloud) const {
    if (!cloud || static_cast<int>(cloud->size()) < params_.min_points) {
        return Eigen::Matrix3d::Identity();
    }

    // Compute mean (XY only for alignment)
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& pt : cloud->points) {
        mean += Eigen::Vector2d(pt.x, pt.y);
    }
    mean /= static_cast<double>(cloud->size());

    // 2x2 covariance matrix of XY
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& pt : cloud->points) {
        Eigen::Vector2d d(pt.x - mean.x(), pt.y - mean.y());
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(cloud->size());

    // Eigendecomposition — largest eigenvector = principal axis
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    // eigenvalues sorted ascending, so index 1 = largest
    Eigen::Vector2d principal = solver.eigenvectors().col(1);

    // Construct rotation matrix to align principal axis to +X
    double angle = std::atan2(principal.y(), principal.x());

    // 180° disambiguation: the half with more points becomes +X
    // Count points on positive principal axis side
    int pos_count = 0;
    Eigen::Vector2d perp(-principal.y(), principal.x());  // perpendicular (unused, just need sign)
    for (const auto& pt : cloud->points) {
        Eigen::Vector2d d(pt.x - mean.x(), pt.y - mean.y());
        if (d.dot(principal) > 0) ++pos_count;
    }
    int neg_count = static_cast<int>(cloud->size()) - pos_count;

    // Convention: the denser side (closer objects, more points from sensor) → negative principal
    // (machine facing +X means points behind it accumulate). Flip if positive side has more points.
    // Simple rule: keep the principal axis orientation that gives MORE points on positive side
    // We want +X to be the direction with fewer points (front of robot usually less dense in corridors)
    // Actually use a neutral rule: just align, accept 180° and handle in distance() with two comparisons.
    (void)pos_count; (void)neg_count;  // disambiguation handled in distance()

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    double c = std::cos(-angle), s = std::sin(-angle);
    R(0, 0) =  c;  R(0, 1) = -s;
    R(1, 0) =  s;  R(1, 1) =  c;
    return R;
}

double RHPDescriptor::computePCAConfidence(const CloudT::Ptr& cloud) const {
    if (!params_.enable_pca_confidence || !cloud || static_cast<int>(cloud->size()) < params_.min_points) {
        return params_.enable_pca_confidence ? 0.0 : 1.0;
    }

    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& pt : cloud->points) {
        mean += Eigen::Vector2d(pt.x, pt.y);
    }
    mean /= static_cast<double>(cloud->size());

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& pt : cloud->points) {
        const Eigen::Vector2d d(pt.x - mean.x(), pt.y - mean.y());
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(cloud->size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    if (solver.info() != Eigen::Success) return 0.0;
    const double l1 = std::max(0.0, solver.eigenvalues()(1));
    const double l2 = std::max(0.0, solver.eigenvalues()(0));
    const double denom = l1 + l2;
    if (denom <= 1e-9) return 0.0;
    return std::clamp((l1 - l2) / denom, 0.0, 1.0);
}

// ---- Single plane projection ----

RHPDescriptor::VecD RHPDescriptor::projectToPlane(
    const CloudT::Ptr& cloud,
    int axis0, double r0_min, double r0_max,
    int axis1, double r1_min, double r1_max,
    int height_axis) const
{
    const int B = RHPD_PLANE_BINS;
    // Per bin channels:
    // [0] density_norm      = count / total_points
    // [1] height_mean_norm  = mean normalized height
    //
    // Legacy(V1/V2):
    // [2] hit_count_norm    = count / max_count_among_bins
    // [3] height_extent_norm= (h_max - h_min) / height_range
    //
    // V3:
    // [2] visibility_state  = 1.0(occupied), 0.5(observed-free), 0.0(unknown)
    // [3] boundary_strength = transition strength between occupied/free/unknown cells
    const int B2 = B * B;
    std::vector<double> counts(B2, 0.0);
    std::vector<double> h_sum(B2, 0.0);
    std::vector<double> h_min(B2,  std::numeric_limits<double>::max());
    std::vector<double> h_max(B2, -std::numeric_limits<double>::max());
    std::vector<double> sector_max_r;
    if (params_.v3_enable) {
        const int sector_n = std::max(8, params_.v3_visibility_bins);
        sector_max_r.assign(sector_n, 0.0);
    }

    auto getCoord = [](const pcl::PointXYZI& pt, int axis) -> double {
        if (axis == 0) return pt.x;
        if (axis == 1) return pt.y;
        return pt.z;
    };

    double r0_range = r0_max - r0_min;
    double r1_range = r1_max - r1_min;
    if (r0_range <= 0 || r1_range <= 0) return VecD::Zero(B2 * RHPD_PLANE_CHANS);

    double h_range = (height_axis == 2) ? (params_.z_max - params_.z_min)
                   : params_.max_range * 2.0;
    double h_min_v = (height_axis == 2) ? params_.z_min : -params_.max_range;

    int total = 0;
    for (const auto& pt : cloud->points) {
        double v0 = getCoord(pt, axis0);
        double v1 = getCoord(pt, axis1);
        double vh = getCoord(pt, height_axis);

        if (v0 < r0_min || v0 >= r0_max) continue;
        if (v1 < r1_min || v1 >= r1_max) continue;

        int i0 = static_cast<int>((v0 - r0_min) / r0_range * B);
        int i1 = static_cast<int>((v1 - r1_min) / r1_range * B);
        i0 = std::clamp(i0, 0, B - 1);
        i1 = std::clamp(i1, 0, B - 1);

        int idx = i0 * B + i1;
        counts[idx] += 1.0;
        double h_norm = (vh - h_min_v) / h_range;
        h_norm = std::clamp(h_norm, 0.0, 1.0);
        h_sum[idx] += h_norm;
        h_min[idx] = std::min(h_min[idx], h_norm);
        h_max[idx] = std::max(h_max[idx], h_norm);
        if (params_.v3_enable) {
            const int sector_n = static_cast<int>(sector_max_r.size());
            const double rr = std::hypot(v0, v1);
            const double az = std::atan2(v1, v0);
            int si = static_cast<int>((az + M_PI) / (2.0 * M_PI) * sector_n);
            si = std::clamp(si, 0, sector_n - 1);
            sector_max_r[si] = std::max(sector_max_r[si], rr);
        }
        ++total;
    }

    VecD out(B2 * RHPD_PLANE_CHANS);
    double inv_total = total > 0 ? 1.0 / static_cast<double>(total) : 0.0;
    const double max_count = *std::max_element(counts.begin(), counts.end());
    const double inv_max_count = max_count > 0.0 ? 1.0 / max_count : 0.0;
    for (int i = 0; i < B2; ++i) {
        const double density_norm = counts[i] * inv_total;
        const double mean_height_norm = counts[i] > 0 ? h_sum[i] / counts[i] : 0.0;

        out(i * RHPD_PLANE_CHANS + 0) = density_norm;
        out(i * RHPD_PLANE_CHANS + 1) = mean_height_norm;

        if (params_.v3_enable) {
            const int row = i / B;
            const int col = i % B;
            const double c0 = r0_min + (static_cast<double>(row) + 0.5) / B * r0_range;
            const double c1 = r1_min + (static_cast<double>(col) + 0.5) / B * r1_range;
            const double rc = std::hypot(c0, c1);
            const int sector_n = static_cast<int>(sector_max_r.size());
            const double az = std::atan2(c1, c0);
            int si = static_cast<int>((az + M_PI) / (2.0 * M_PI) * sector_n);
            si = std::clamp(si, 0, sector_n - 1);
            const double frontier_r = sector_max_r[si];
            double visibility_state = 0.0;
            if (counts[i] > 0.0) {
                visibility_state = 1.0;  // occupied
            } else if (frontier_r > 1e-6 && rc + params_.v3_free_space_margin_m < frontier_r) {
                visibility_state = 0.5;  // observed free-space before measured frontier
            } else {
                visibility_state = 0.0;  // unknown / unobserved
            }

            auto class_of = [](double s) -> int {
                if (s >= 0.75) return 2;   // occupied
                if (s >= 0.25) return 1;   // free
                return 0;                  // unknown
            };
            const int cls = class_of(visibility_state);
            double transition = 0.0;
            int n_neighbors = 0;
            const int drow[4] = {-1, 1, 0, 0};
            const int dcol[4] = {0, 0, -1, 1};
            for (int n = 0; n < 4; ++n) {
                const int rr = row + drow[n];
                const int cc = col + dcol[n];
                if (rr < 0 || rr >= B || cc < 0 || cc >= B) continue;
                const int ni = rr * B + cc;
                double n_state = 0.0;
                if (counts[ni] > 0.0) {
                    n_state = 1.0;
                } else {
                    const double nc0 = r0_min + (static_cast<double>(rr) + 0.5) / B * r0_range;
                    const double nc1 = r1_min + (static_cast<double>(cc) + 0.5) / B * r1_range;
                    const double nrc = std::hypot(nc0, nc1);
                    const double naz = std::atan2(nc1, nc0);
                    int nsi = static_cast<int>((naz + M_PI) / (2.0 * M_PI) * sector_n);
                    nsi = std::clamp(nsi, 0, sector_n - 1);
                    const double nfrontier_r = sector_max_r[nsi];
                    if (nfrontier_r > 1e-6 && nrc + params_.v3_free_space_margin_m < nfrontier_r) {
                        n_state = 0.5;
                    } else {
                        n_state = 0.0;
                    }
                }
                const int ncls = class_of(n_state);
                transition += 0.5 * std::abs(static_cast<double>(cls - ncls));  // normalized to [0,1]
                ++n_neighbors;
            }
            const double boundary_strength = (n_neighbors > 0) ? (transition / n_neighbors) : 0.0;

            out(i * RHPD_PLANE_CHANS + 2) = visibility_state;
            out(i * RHPD_PLANE_CHANS + 3) = std::clamp(boundary_strength, 0.0, 1.0);
        } else if (params_.v2_enable) {
            const double hit_count_norm = counts[i] * inv_max_count;
            const double height_extent_norm =
                counts[i] > 0 ? std::max(0.0, h_max[i] - h_min[i]) : 0.0;
            out(i * RHPD_PLANE_CHANS + 2) = hit_count_norm;
            out(i * RHPD_PLANE_CHANS + 3) = height_extent_norm;
        } else {
            out(i * RHPD_PLANE_CHANS + 2) = 0.0;
            out(i * RHPD_PLANE_CHANS + 3) = 0.0;
        }
    }
    return out;
}

// ---- Part A: triple-plane projection ----

RHPDescriptor::VecD RHPDescriptor::computePartA(const CloudT::Ptr& aligned_cloud) const {
    const double R = params_.max_range;
    const double Zn = params_.z_min;
    const double Zx = params_.z_max;

    // XY plane: axis0=X [-R,+R], axis1=Y [-R,+R], height=Z
    VecD xy = projectToPlane(aligned_cloud, 0, -R, R, 1, -R, R, 2);

    // XZ plane: axis0=X [-R,+R], axis1=Z [Zn,Zx], height=Y (lateral)
    VecD xz = projectToPlane(aligned_cloud, 0, -R, R, 2, Zn, Zx, 1);

    // YZ plane: axis0=Y [-R,+R], axis1=Z [Zn,Zx], height=X (forward)
    VecD yz = projectToPlane(aligned_cloud, 1, -R, R, 2, Zn, Zx, 0);

    VecD out(RHPD_PART_A_DIM);
    out.segment(0,                   RHPD_PLANE_DIM) = xy;
    out.segment(RHPD_PLANE_DIM,      RHPD_PLANE_DIM) = xz;
    out.segment(RHPD_PLANE_DIM * 2,  RHPD_PLANE_DIM) = yz;
    return out;
}

// ---- Part B: ring-height rotation-invariant statistics ----

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

// ---- Main compute ----

RHPDescriptor::VecD RHPDescriptor::compute(const CloudT::Ptr& cloud) const {
    VecD zero = VecD::Zero(RHPD_DIM);
    if (!cloud || static_cast<int>(cloud->size()) < params_.min_points) {
        return zero;
    }

    // PCA alignment
    Eigen::Matrix3d R_align = computePCAAlignment(cloud);

    // Build aligned cloud (only rotate around Z, preserve heights)
    CloudT::Ptr aligned = pcl::make_shared<CloudT>();
    aligned->reserve(cloud->size());
    for (const auto& pt : cloud->points) {
        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d pa = R_align * p;
        pcl::PointXYZI apt;
        apt.x = static_cast<float>(pa.x());
        apt.y = static_cast<float>(pa.y());
        apt.z = static_cast<float>(pa.z());
        apt.intensity = pt.intensity;
        aligned->push_back(apt);
    }

    const double pca_confidence = computePCAConfidence(cloud);
    VecD part_a = computePartA(aligned);
    VecD part_b = computePartB(cloud);   // Part B is rotation-invariant, use original cloud
    VecD aux = computeAux(cloud, pca_confidence);

    VecD out(RHPD_DIM);
    out.segment(0, RHPD_PART_A_DIM) = part_a;
    out.segment(RHPD_PART_A_DIM, RHPD_PART_B_DIM) = part_b;
    out.segment(RHPD_PART_A_DIM + RHPD_PART_B_DIM, RHPD_AUX_DIM) = aux;
    return out;
}

// ---- Distance (with 180° handling) ----

double RHPDescriptor::distance(const VecD& a, const VecD& b) const {
    if (a.size() != RHPD_DIM || b.size() != RHPD_DIM) {
        return std::numeric_limits<double>::max();
    }

    VecD a_flip = a;
    flipPlaneAxis(a_flip, 0, true, true);                        // XY: X and Y flip
    flipPlaneAxis(a_flip, RHPD_PLANE_DIM, true, false);          // XZ: X flips only
    flipPlaneAxis(a_flip, RHPD_PLANE_DIM * 2, true, false);      // YZ: Y flips only
    if (params_.enable_negative_space) {
        shiftNegativeSpace180(a_flip);
    }
    shiftVerticalTokens180(a_flip);

    const double conf_a = params_.enable_pca_confidence ? a(RHPD_DIM - 1) : 1.0;
    const double conf_b = params_.enable_pca_confidence ? b(RHPD_DIM - 1) : 1.0;
    const double part_a_weight = params_.enable_pca_confidence
        ? (0.35 + 0.65 * std::min(std::clamp(conf_a, 0.0, 1.0), std::clamp(conf_b, 0.0, 1.0)))
        : 1.0;

    auto weightedDistance = [&](const VecD& lhs, const VecD& rhs) {
        const double da = (lhs.segment(0, RHPD_PART_A_DIM) - rhs.segment(0, RHPD_PART_A_DIM)).squaredNorm();
        const double db = (lhs.segment(RHPD_PART_A_DIM, RHPD_PART_B_DIM) -
                           rhs.segment(RHPD_PART_A_DIM, RHPD_PART_B_DIM)).squaredNorm();
        const double daux = (lhs.segment(RHPD_PART_A_DIM + RHPD_PART_B_DIM, RHPD_AUX_DIM) -
                             rhs.segment(RHPD_PART_A_DIM + RHPD_PART_B_DIM, RHPD_AUX_DIM)).squaredNorm();
        return std::sqrt(part_a_weight * da + db + 0.5 * daux);
    };

    return std::min(weightedDistance(a, b), weightedDistance(a_flip, b));
}

// ============================================================
//  RHPDManager
// ============================================================

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

} // namespace n3mapping
