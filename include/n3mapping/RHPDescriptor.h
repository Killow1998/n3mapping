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
        double max_range = 30.0;
        double z_min = -2.0;
        double z_max = 6.0;
        double min_points = 20;
        int azimuth_bins = 36;
        bool v2_enable = true;
        bool v3_enable = false;
        int v3_visibility_bins = 72;
        double v3_free_space_margin_m = 0.15;
        bool   enable_negative_space = true;
        bool   enable_pca_confidence = true;

        double part_a_scale = 1.0;
        double aux_scale    = 1.0;
    };

    explicit RHPDescriptor(const Params& params);

    Place compute(const CloudT::Ptr& cloud) const;
    Features computeWithHeading(const CloudT::Ptr& cloud) const;
    RHPDHeadingPtr computeHeading(const CloudT::Ptr& cloud) const;

    double distance(const Place& a, const Place& b) const;

    struct PreparedQuery {
        const Place& place;
        std::array<RHPDCells, 3> flipped_planes;
        Eigen::Matrix<double, RHPD_AUX_DIM, 1> flipped_auxiliary;
    };
    PreparedQuery prepareQuery(const Place& place) const;
    double distance(const PreparedQuery& query, const Place& reference) const;

    const Params& params() const { return params_; }

private:
    Eigen::Matrix3d computePCAAlignment(const CloudT& cloud, double* confidence) const;

    std::array<RHPDCells, 3> computePartA(const CloudT::Ptr& aligned_cloud) const;

    void computePolar(const CloudT::Ptr& cloud, double confidence, Place& place, RHPDHeadingPtr* heading) const;
    Place computeImpl(const CloudT::Ptr& cloud, RHPDHeadingPtr* heading) const;

    RHPDCells projectToPlane(const CloudT::Ptr& cloud,
                        int axis0, double range0_min, double range0_max,
                        int axis1, double range1_min, double range1_max,
                        int height_axis) const;

    Params params_;
};

class RHPDManager {
public:
    using VecD = Eigen::VectorXd;
    using Place = RHPDPlace;

    explicit RHPDManager(const RHPDescriptor::Params& params)
        : desc_(params) {}
    RHPDManager() : desc_(RHPDescriptor::Params{}) {}

    void add(int64_t kf_id, const Place& descriptor, bool replace = false);

    Place addCloud(int64_t kf_id, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

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
    RHPDPlaceCache snapshot() const;
    VecD makeCoarseKey(const Place& descriptor) const;

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
    std::vector<std::shared_ptr<const Place>> descriptors_;
    std::vector<VecD>    coarse_keys_;
    mutable std::mutex mutex_;
};

RHPDescriptor::Params rhpdParamsFromConfig(const Config& config);

}
