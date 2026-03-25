// RHPDescriptor: Ring-Height + Planar descriptor with range profiling for corridor robustness.
#include "n3mapping/RHPDescriptor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/memory.h>
#include <memory>

namespace n3mapping {

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

// ---- Single plane projection ----

RHPDescriptor::VecD RHPDescriptor::projectToPlane(
    const CloudT::Ptr& cloud,
    int axis0, double r0_min, double r0_max,
    int axis1, double r1_min, double r1_max,
    int height_axis) const
{
    const int B = RHPD_PLANE_BINS;
    // Each bin stores: [density_sum, height_sum, count]
    // We compute density = count / total_points (normalized)
    //          height_mean = height_sum / count (normalized to [0,1])
    const int B2 = B * B;
    std::vector<double> counts(B2, 0.0);
    std::vector<double> h_sum(B2, 0.0);
    std::vector<double> h_min_global(B2,  std::numeric_limits<double>::max());
    std::vector<double> h_max_global(B2, -std::numeric_limits<double>::max());

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
        ++total;
    }

    VecD out(B2 * RHPD_PLANE_CHANS);
    double inv_total = total > 0 ? 1.0 / static_cast<double>(total) : 0.0;
    for (int i = 0; i < B2; ++i) {
        out(i * RHPD_PLANE_CHANS + 0) = counts[i] * inv_total;  // normalized density
        out(i * RHPD_PLANE_CHANS + 1) = counts[i] > 0 ? h_sum[i] / counts[i] : 0.0;  // mean height
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

    VecD part_a = computePartA(aligned);
    VecD part_b = computePartB(cloud);   // Part B is rotation-invariant, use original cloud

    VecD out(RHPD_DIM);
    out.head(RHPD_PART_A_DIM) = part_a;
    out.tail(RHPD_PART_B_DIM) = part_b;
    return out;
}

// ---- Distance (with 180° handling) ----

double RHPDescriptor::distance(const VecD& a, const VecD& b) const {
    if (a.size() != RHPD_DIM || b.size() != RHPD_DIM) {
        return std::numeric_limits<double>::max();
    }

    // Normal orientation
    double d0 = (a - b).norm();

    // 180°-flipped version of a:
    // Flip = negate X in aligned cloud means XY rows are reversed, XZ rows are reversed,
    // YZ Y-axis is reversed. For the flat vector, we rebuild a flipped Part A.
    //
    // Practical approximation: flip the XY and XZ sub-descriptors axis-0 order,
    // and the YZ sub-descriptor axis-0 order.
    // This is equivalent to reversing the order of rows in each 14x14 grid.
    const int B = RHPD_PLANE_BINS;
    const int B2C = RHPD_PLANE_BINS * RHPD_PLANE_BINS * RHPD_PLANE_CHANS;  // per-plane dim

    VecD a_flip = a;
    // Flip each of the 3 planes: reverse row order (axis0 corresponds to X)
    for (int p = 0; p < 3; ++p) {
        int base = p * B2C;
        for (int row = 0; row < B / 2; ++row) {
            int row_flip = B - 1 - row;
            for (int col = 0; col < B; ++col) {
                int idx1 = base + (row      * B + col) * RHPD_PLANE_CHANS;
                int idx2 = base + (row_flip * B + col) * RHPD_PLANE_CHANS;
                for (int c = 0; c < RHPD_PLANE_CHANS; ++c) {
                    std::swap(a_flip(idx1 + c), a_flip(idx2 + c));
                }
            }
        }
    }
    // Part B is rotation-invariant, no flip needed
    double d180 = (a_flip - b).norm();

    return std::min(d0, d180);
}

// ============================================================
//  RHPDManager
// ============================================================

void RHPDManager::add(int64_t kf_id, const VecD& descriptor) {
    ids_.push_back(kf_id);
    descriptors_.push_back(descriptor);
}

RHPDManager::VecD RHPDManager::addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    VecD descriptor = desc_.compute(cloud);
    add(kf_id, descriptor);
    return descriptor;
}

std::vector<std::pair<int64_t, double>> RHPDManager::search(const VecD& query, int top_k) const {
    std::vector<std::pair<int64_t, double>> results;
    if (ids_.empty() || query.size() != RHPD_DIM) return results;

    results.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i) {
        double d = desc_.distance(query, descriptors_[i]);
        results.emplace_back(ids_[i], d);
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }
    return results;
}

void RHPDManager::clear() {
    ids_.clear();
    descriptors_.clear();
}

std::vector<std::pair<int64_t, RHPDManager::VecD>> RHPDManager::getAll() const {
    std::vector<std::pair<int64_t, VecD>> out;
    out.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i) {
        out.emplace_back(ids_[i], descriptors_[i]);
    }
    return out;
}

void RHPDManager::loadAll(const std::vector<std::pair<int64_t, VecD>>& data) {
    clear();
    for (const auto& [id, desc] : data) {
        add(id, desc);
    }
}

} // namespace n3mapping
