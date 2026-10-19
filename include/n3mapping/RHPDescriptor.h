// RHPDescriptor: Ring-Height + Planar descriptor for place recognition.
//
// Part A (2352-dim): PCA-aligned triple-plane projection (XY + XZ + YZ),
//                    14x14 bins, 4 channels.
//                    - legacy(V1/V2): density_norm, height_mean_norm,
//                                      hit_count_norm, height_extent_norm.
//                    - V3:           density_norm, height_mean_norm,
//                                      visibility_state, boundary_strength.
// Part B (512-dim):  Yaw-tolerant ring-height statistics,
//                    16 rings x 8 z-layers x 4 ch (density, azimuth_variance,
//                    mean_range, range_variance).
//
// Augmentation:
//   Negative-space signature: per-azimuth nearest hit, frontier span, open/unknown state.
//   PCA anisotropy confidence: one scalar used to down-weight unstable Part A.
//
// Place summary: sparse observed cells and a compact auxiliary block.
// Heading: 7 channels on a ring/sector grid, sharing the polar traversal.
// Heading matching estimates yaw by circular sector alignment.
// Matching: L2 distance with 180-degree flip handling.
#pragma once

#include "n3mapping/rhpd_heading.h"
#include "n3mapping/rhpd_place.h"
#include "n3mapping/descriptor_performance.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <limits>
#include <cmath>
#include <functional>
#include <mutex>

namespace n3mapping {

struct Config;

// ----------- RHPDescriptor class -----------
class RHPDescriptor {
public:
    using PointT  = pcl::PointXYZI;
    using CloudT  = pcl::PointCloud<PointT>;
    using VecD    = Eigen::VectorXd;
    using Place   = RHPDPlace;

    struct Features {
        Place place;
        RHPDHeadingPtr heading;
    };

    struct Params {
        RHPDHeadingParams heading;
        double max_range    = 30.0;   // max radial distance to consider [m]
        double z_min        = -2.0;   // min height relative to sensor [m]
        double z_max        =  6.0;   // max height relative to sensor [m]
        double min_points   = 20;     // min points per PCA for alignment
        int    azimuth_bins = 36;     // azimuth bins for computing variance in Part B
        bool   v2_enable    = true;   // if false, V2 extra channels are zeroed for old-ablation behavior
        bool   v3_enable    = false;  // if true, Part-A ch2/ch3 switch to visibility/boundary semantics
        int    v3_visibility_bins = 72;          // angular sectors for visibility projection
        double v3_free_space_margin_m = 0.15;    // margin before measured frontier to mark free space
        bool   enable_negative_space = true;
        bool   enable_pca_confidence = true;
        // Scales on the plane and auxiliary blocks, applied on top of the
        // weights already in distance(). One leaves the shipped behaviour
        // untouched; zero drops the block. Only the ring-height block, which
        // has no scale because it is the reference, was measured to separate
        // same-place from elsewhere at all.
        double part_a_scale = 1.0;
        double aux_scale    = 1.0;
    };

    explicit RHPDescriptor(const Params& params);

    // Compute RHPD from a body-frame point cloud.
    // Returns an empty observation summary if the cloud is too sparse.
    Place compute(const CloudT::Ptr& cloud) const;
    Features computeWithHeading(const CloudT::Ptr& cloud) const;
    RHPDHeadingPtr computeHeading(const CloudT::Ptr& cloud) const;

    // L2 distance between two descriptors (lower = more similar).
    // Handles 180-degree ambiguity by comparing both orientations and taking the minimum.
    double distance(const Place& a, const Place& b) const;

    const Params& params() const { return params_; }

private:
    // PCA-align: rotate cloud so XY-plane PCA principal axis → +X axis.
    // Returns the XY alignment rotation; distance() handles its 180-degree ambiguity.
    Eigen::Matrix3d computePCAAlignment(const CloudT::Ptr& cloud, double* confidence) const;

    // Part A: triple-plane projection (XY, XZ, YZ) on PCA-aligned cloud.
    // Stores only cells with nonzero channels.
    std::array<RHPDCells, 3> computePartA(const CloudT::Ptr& aligned_cloud) const;

    // Ring-height, negative-space and optional heading share
    // a polar traversal. Preserve each channel's original support and rounding.
    void computePolar(const CloudT::Ptr& cloud, double confidence, Place& place, RHPDHeadingPtr* heading) const;
    Place computeImpl(const CloudT::Ptr& cloud, RHPDHeadingPtr* heading) const;

    // Single plane projection: project points onto a plane, fill BINSxBINS grid.
    // axis0, axis1: which coordinate axes to use (0=x,1=y,2=z)
    // range0, range1: physical range [min, max] for each axis
    // Output: sorted cells with four channels per observed cell.
    // - legacy(V1/V2): [density_norm, height_mean_norm, hit_count_norm, height_extent_norm]
    // - V3:           [density_norm, height_mean_norm, visibility_state, boundary_strength]
    RHPDCells projectToPlane(const CloudT::Ptr& cloud,
                        int axis0, double range0_min, double range0_max,
                        int axis1, double range1_min, double range1_max,
                        int height_axis) const;

    friend class RHPDManager;
    struct PreparedQuery {
        const Place& place;
        std::array<RHPDCells, 3> flipped_planes;
        Eigen::Matrix<double, RHPD_AUX_DIM, 1> flipped_auxiliary;
    };
    PreparedQuery prepareQuery(const Place& place) const;
    double distance(const PreparedQuery& query, const Place& reference) const;
    Params params_;
};

// ----------- RHPDManager: database + search -----------
class RHPDManager {
public:
    using VecD = Eigen::VectorXd;
    using Place = RHPDPlace;

    explicit RHPDManager(const RHPDescriptor::Params& params)
        : desc_(params) {}
    RHPDManager() : desc_(RHPDescriptor::Params{}) {}

    // Add a descriptor for a keyframe.
    void add(int64_t kf_id, const Place& descriptor);

    // Add and compute descriptor from cloud. Returns the computed descriptor.
    Place addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    // Search for the top-k nearest neighbors. Returns (kf_id, distance) pairs sorted by distance.
    std::vector<std::pair<int64_t, double>> search(const Place& query, int top_k) const;
    std::vector<std::pair<int64_t, double>> search(const Place& query, int top_k, int preselect,
        DescriptorSearchPerformance* performance = nullptr) const;
    std::vector<std::pair<int64_t, double>> searchFiltered(
        const Place& query,
        int top_k,
        int preselect,
        const std::function<bool(int64_t)>& accept,
        DescriptorSearchPerformance* performance = nullptr) const;
    bool get(int64_t kf_id, Place* descriptor) const;
    VecD makeCoarseKey(const Place& descriptor) const;

    // Compute descriptor for a cloud (for use before searching).
    Place compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
        return desc_.compute(cloud);
    }

    RHPDescriptor::Features computeWithHeading(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
        return desc_.computeWithHeading(cloud);
    }
    RHPDHeadingPtr computeHeading(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
        return desc_.computeHeading(cloud);
    }

    size_t size() const;
    void clear();
    void swapWith(RHPDManager& other);

    const RHPDescriptor::Params& getDescriptorParams() const { return desc_.params(); }

private:
    RHPDescriptor desc_;
    std::vector<int64_t> ids_;
    std::vector<Place>   descriptors_;
    std::vector<VecD>    coarse_keys_;
    mutable std::mutex mutex_;
};

// The descriptor parameters as configuration sets them. It was already a
// function, in an anonymous namespace inside loop_detector.cpp, which is
// indistinguishable from inline for anything trying to test it.
RHPDescriptor::Params rhpdParamsFromConfig(const Config& config);

} // namespace n3mapping
