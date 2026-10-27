// RHPDescriptor: Ring-Height + Planar descriptor with visibility-aware planar cues.
#include "n3mapping/RHPDescriptor.h"


#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
// ============================================================
//  RHPDescriptor
// ============================================================

RHPDescriptor::RHPDescriptor(const Params& params) : params_(params) {}

// ---- PCA alignment ----

Eigen::Matrix3d RHPDescriptor::computePCAAlignment(const CloudT& cloud, double* confidence) const {
    // Compute mean (XY only for alignment)
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& pt : cloud.points) {
        mean += Eigen::Vector2d(pt.x, pt.y);
    }
    mean /= static_cast<double>(cloud.size());

    // 2x2 covariance matrix of XY
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& pt : cloud.points) {
        Eigen::Vector2d d(pt.x - mean.x(), pt.y - mean.y());
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(cloud.size());

    // Eigendecomposition — largest eigenvector = principal axis
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    // eigenvalues sorted ascending, so index 1 = largest
    Eigen::Vector2d principal = solver.eigenvectors().col(1);

    // Construct rotation matrix to align principal axis to +X
    double angle = std::atan2(principal.y(), principal.x());

    if (confidence) {
        const double l1 = std::max(0.0, solver.eigenvalues()(1));
        const double l2 = std::max(0.0, solver.eigenvalues()(0));
        *confidence = solver.info() == Eigen::Success && l1 + l2 > 1e-9
            ? std::clamp((l1 - l2) / (l1 + l2), 0.0, 1.0) : 0.0;
    }

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    double c = std::cos(-angle), s = std::sin(-angle);
    R(0, 0) =  c;  R(0, 1) = -s;
    R(1, 0) =  s;  R(1, 1) =  c;
    return R;
}

// ---- Single plane projection ----

RHPDCells RHPDescriptor::projectToPlane(
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
    std::array<double, B2> counts{};
    std::array<double, B2> h_sum{};
    std::array<double, B2> h_min, h_max;
    h_min.fill(std::numeric_limits<double>::max());
    h_max.fill(-std::numeric_limits<double>::max());
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
    if (r0_range <= 0 || r1_range <= 0) return {};

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

    Eigen::Matrix<double, RHPD_PLANE_DIM, 1> out;
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
    RHPDCells cells;
    size_t active = 0;
    for (int i = 0; i < B2; ++i) active += !out.segment<4>(i * 4).isZero(0.0);
    cells.indices.reserve(active);
    cells.values.reserve(active);
    for (int i = 0; i < B2; ++i) cells.append(i, out.segment<4>(i * 4));
    return cells;
}

// ---- Part A: triple-plane projection ----

std::array<RHPDCells, 3> RHPDescriptor::computePartA(const CloudT::Ptr& cloud) const {
    const double r = params_.max_range, z0 = params_.z_min, z1 = params_.z_max;
    return {{projectToPlane(cloud, 0, -r, r, 1, -r, r, 2),
             projectToPlane(cloud, 0, -r, r, 2, z0, z1, 1),
             projectToPlane(cloud, 1, -r, r, 2, z0, z1, 0)}};
}

// ---- Main compute ----

RHPDescriptor::Place RHPDescriptor::compute(const CloudT::Ptr& cloud) const {
    return computeImpl(cloud, nullptr);
}

RHPDescriptor::Features RHPDescriptor::computeWithHeading(const CloudT::Ptr& cloud) const {
    Features result;
    result.place = computeImpl(cloud, &result.heading);
    return result;
}

RHPDHeadingPtr RHPDescriptor::computeHeading(const CloudT::Ptr& cloud) const {
    if (!cloud || cloud->empty()) return {};
    return computeRHPDHeading(*cloud, params_.heading);
}

RHPDescriptor::Place RHPDescriptor::computeImpl(const CloudT::Ptr& cloud, RHPDHeadingPtr* heading) const {
    if (!cloud || static_cast<int>(cloud->size()) < params_.min_points) {
        if (heading) *heading = computeHeading(cloud);
        return {};
    }

    // PCA alignment
    double confidence = 1.0;
    Eigen::Matrix3d R_align = computePCAAlignment(*cloud, params_.enable_pca_confidence ? &confidence : nullptr);

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

    Place out;
    out.planes = computePartA(aligned);
    computePolar(cloud, confidence, out, heading);
    return out;
}

} // namespace n3mapping
