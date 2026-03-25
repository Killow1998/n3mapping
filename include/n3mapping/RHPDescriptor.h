// RHPDescriptor: Ring-Height + Planar descriptor for place recognition.
//
// Part A (1176-dim): PCA-aligned triple-plane projection (XY + XZ + YZ),
//                    14x14 bins, 2 ch (density, height_mean).
// Part B (512-dim):  Rotation-invariant ring-height statistics,
//                    16 rings x 8 z-layers x 4 ch (density, azimuth_variance,
//                    mean_range, range_variance).
//
// Total: 1688 dim, stored as a flat Eigen::VectorXd.
// Matching: L2 distance with 180-degree flip handling.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <limits>
#include <cmath>

namespace n3mapping {

// ----------- Descriptor parameters (compile-time constants) -----------
constexpr int RHPD_PLANE_BINS  = 14;       // grid size for each planar projection
constexpr int RHPD_PLANE_CHANS = 2;        // channels per bin: density, height_mean
constexpr int RHPD_PLANE_DIM   = RHPD_PLANE_BINS * RHPD_PLANE_BINS * RHPD_PLANE_CHANS;
constexpr int RHPD_PART_A_DIM  = RHPD_PLANE_DIM * 3;  // XY + XZ + YZ = 588*2 = 1176? actually 14*14*2*3=1176

constexpr int RHPD_RINGS       = 16;       // radial rings for Part B
constexpr int RHPD_ZLAYERS     = 8;        // height layers for Part B
constexpr int RHPD_RH_CHANS    = 4;        // density, azimuth_variance, mean_range, range_variance
constexpr int RHPD_PART_B_DIM  = RHPD_RINGS * RHPD_ZLAYERS * RHPD_RH_CHANS;  // 512

constexpr int RHPD_DIM         = RHPD_PART_A_DIM + RHPD_PART_B_DIM;  // 1688

// ----------- RHPDescriptor class -----------
class RHPDescriptor {
public:
    using PointT  = pcl::PointXYZI;
    using CloudT  = pcl::PointCloud<PointT>;
    using VecD    = Eigen::VectorXd;

    struct Params {
        double max_range    = 30.0;   // max radial distance to consider [m]
        double z_min        = -2.0;   // min height relative to sensor [m]
        double z_max        =  6.0;   // max height relative to sensor [m]
        double min_points   = 20;     // min points per PCA for alignment
        int    azimuth_bins = 36;     // azimuth bins for computing variance in Part B
    };

    explicit RHPDescriptor(const Params& params);

    // Compute RHPD from a body-frame point cloud.
    // Returns a zero vector if cloud is too sparse.
    VecD compute(const CloudT::Ptr& cloud) const;

    // L2 distance between two descriptors (lower = more similar).
    // Handles 180-degree ambiguity by comparing both orientations and taking the minimum.
    double distance(const VecD& a, const VecD& b) const;

    int dim() const { return RHPD_DIM; }

private:
    // PCA-align: rotate cloud so XY-plane PCA principal axis → +X axis.
    // Returns the 2D rotation angle applied. Also performs 180° disambiguation.
    Eigen::Matrix3d computePCAAlignment(const CloudT::Ptr& cloud) const;

    // Part A: triple-plane projection (XY, XZ, YZ) on PCA-aligned cloud.
    // Output: RHPD_PART_A_DIM-dim vector.
    VecD computePartA(const CloudT::Ptr& aligned_cloud) const;

    // Part B: rotation-invariant ring-height statistics on original cloud.
    // Output: RHPD_PART_B_DIM-dim vector.
    VecD computePartB(const CloudT::Ptr& cloud) const;

    // Single plane projection: project points onto a plane, fill BINSxBINS grid.
    // axis0, axis1: which coordinate axes to use (0=x,1=y,2=z)
    // range0, range1: physical range [min, max] for each axis
    // Output: BINS*BINS*CHANS vector [density, height_mean] per cell.
    VecD projectToPlane(const CloudT::Ptr& cloud,
                        int axis0, double range0_min, double range0_max,
                        int axis1, double range1_min, double range1_max,
                        int height_axis) const;

    Params params_;
};

// ----------- RHPDManager: database + search -----------
class RHPDManager {
public:
    using VecD = Eigen::VectorXd;

    explicit RHPDManager(const RHPDescriptor::Params& params)
        : desc_(params) {}
    RHPDManager() : desc_(RHPDescriptor::Params{}) {}

    // Add a descriptor for a keyframe.
    void add(int64_t kf_id, const VecD& descriptor);

    // Add and compute descriptor from cloud. Returns the computed descriptor.
    VecD addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    // Search for the top-k nearest neighbors. Returns (kf_id, distance) pairs sorted by distance.
    std::vector<std::pair<int64_t, double>> search(const VecD& query, int top_k) const;

    // Compute descriptor for a cloud (for use before searching).
    VecD compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
        return desc_.compute(cloud);
    }

    size_t size() const { return ids_.size(); }
    void clear();

    // Serialization helpers.
    std::vector<std::pair<int64_t, VecD>> getAll() const;
    void loadAll(const std::vector<std::pair<int64_t, VecD>>& data);

private:
    RHPDescriptor desc_;
    std::vector<int64_t> ids_;
    std::vector<VecD>    descriptors_;
};

} // namespace n3mapping
