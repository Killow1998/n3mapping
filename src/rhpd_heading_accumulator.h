#pragma once
#include "n3mapping/rhpd_heading.h"
#include <algorithm>
#include <cmath>

namespace n3mapping::detail {
// Also used by the joint polar pass: range/azimuth are computed only once
// for the RHPD ring-height and heading channels. Keep legacy float rounding.
class HeadingAccumulator {
public:
    explicit HeadingAccumulator(const RHPDHeadingParams& params)
        : p_(params), count_(Eigen::MatrixXd::Zero(p_.rings, p_.sectors)),
          sum_(count_), sum2_(count_), max_(Eigen::MatrixXd::Constant(p_.rings, p_.sectors, -1e6)) {
        for (auto& layer : layers_) layer = count_;
    }
    void add(const pcl::PointXYZI& point, double range, float azimuth) {
        if (range < p_.min_radius || range > p_.max_radius) return;
        int ring;
        if (p_.log_polar) {
            const double ratio = std::log(range / p_.min_radius) / std::log(p_.max_radius / p_.min_radius);
            ring = static_cast<int>(std::floor(p_.rings * ratio));
        } else {
            ring = static_cast<int>(std::ceil((range / p_.max_radius) * p_.rings)) - 1;
        }
        ring = std::clamp(ring, 0, p_.rings - 1);
        float angle = azimuth * 180.0f / static_cast<float>(M_PI);
        if (angle < 0.0f) angle += 360.0f;
        const int sector = std::clamp(static_cast<int>(std::floor(angle / 360.0f * p_.sectors)), 0, p_.sectors - 1);
        const float z = point.z + static_cast<float>(p_.lidar_height);
        sum_(ring, sector) += z;
        sum2_(ring, sector) += z * z;
        count_(ring, sector) += 1.0;
        max_(ring, sector) = std::max(max_(ring, sector), static_cast<double>(z));
        if (point.z >= p_.z_min && point.z < p_.z_max) {
            const double dz = (p_.z_max - p_.z_min) / 4.0;
            const int layer = std::min(static_cast<int>((point.z - p_.z_min) / dz), 3);
            layers_[layer](ring, sector) += 1.0;
        }
    }
    RHPDHeadingPtr finish() const {
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(p_.rows(), p_.sectors);
        const double max_count = std::max(1.0, count_.maxCoeff());
        for (int r = 0; r < p_.rings; ++r) for (int s = 0; s < p_.sectors; ++s) {
            const double n = count_(r, s);
            if (n < 1.0) continue;
            const double mean = sum_(r, s) / n;
            const double variance = std::max(0.0, sum2_(r, s) / n - mean * mean);
            out(r, s) = max_(r, s) < -999.0 ? 0.0 : max_(r, s);
            out(p_.rings + r, s) = std::sqrt(variance);
            out(2 * p_.rings + r, s) = n / max_count;
            for (int layer = 0; layer < 4; ++layer)
                out((3 + layer) * p_.rings + r, s) = layers_[layer](r, s) / n;
        }
        return makeRHPDHeading(out);
    }
private:
    const RHPDHeadingParams& p_;
    Eigen::MatrixXd count_, sum_, sum2_, max_;
    std::array<Eigen::MatrixXd, 4> layers_;
};
} // namespace n3mapping::detail
