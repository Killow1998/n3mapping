#include "n3mapping/rhpd_heading.h"
#include "rhpd_heading_accumulator.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace n3mapping {
RHPDHeadingPtr makeRHPDHeading(const Eigen::MatrixXd& values) {
    constexpr int channels = RHPDHeadingParams::channels;
    if (values.size() == 0 || values.rows() % channels != 0) return {};
    auto heading = std::make_shared<RHPDHeading>();
    heading->rings = values.rows() / channels;
    heading->sectors = values.cols();
    heading->column_norms.resize(channels, values.cols());
    heading->sector_key.resize(1, values.cols());
    heading->ring_key.reserve(values.rows());
    heading->offsets.reserve(channels * values.cols() + 1);
    const size_t nonzero = (values.array() != 0.0).count();
    heading->values.reserve(nonzero);
    heading->ring_indices.reserve(nonzero);
    for (int row = 0; row < values.rows(); ++row)
        heading->ring_key.push_back(static_cast<float>(values.row(row).mean()));
    for (int sector = 0; sector < values.cols(); ++sector)
        heading->sector_key(0, sector) = values.col(sector).mean();
    for (int channel = 0; channel < channels; ++channel) {
        for (int sector = 0; sector < values.cols(); ++sector) {
            heading->offsets.push_back(heading->values.size());
            heading->column_norms(channel, sector) =
                values.col(sector).segment(channel * heading->rings, heading->rings).norm();
            for (int ring = 0; ring < heading->rings; ++ring) {
                const double value = values(channel * heading->rings + ring, sector);
                if (value == 0.0) continue;
                heading->ring_indices.push_back(ring);
                heading->values.push_back(value);
            }
        }
    }
    heading->offsets.push_back(heading->values.size());
    return heading;
}

Eigen::MatrixXd rhpdHeadingDenseValues(const RHPDHeading& heading) {
    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(
        RHPDHeadingParams::channels * heading.rings, heading.sectors);
    for (int channel = 0; channel < RHPDHeadingParams::channels; ++channel)
        for (int sector = 0; sector < heading.sectors; ++sector) {
            const int col = channel * heading.sectors + sector;
            for (int i = heading.offsets[col]; i < heading.offsets[col + 1]; ++i)
                values(channel * heading.rings + heading.ring_indices[i], sector) = heading.values[i];
        }
    return values;
}

RHPDHeadingPtr computeRHPDHeading(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                                 const RHPDHeadingParams& params) {
    if (cloud.empty()) return {};
    detail::HeadingAccumulator accumulator(params);
    for (const auto& point : cloud.points)
        accumulator.add(point, std::sqrt(point.x * point.x + point.y * point.y),
                        std::atan2(point.y, point.x));
    return accumulator.finish();
}

namespace {
double sparseDot(const RHPDHeading& a, int col_a, const RHPDHeading& b, int col_b) {
    int i = a.offsets[col_a], j = b.offsets[col_b];
    double dot = 0.0;
    while (i < a.offsets[col_a + 1] && j < b.offsets[col_b + 1]) {
        if (a.ring_indices[i] == b.ring_indices[j]) {
            dot += a.values[i++] * b.values[j++];
        } else if (a.ring_indices[i] < b.ring_indices[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return dot;
}

double distanceAtShift(const RHPDHeading& query, const RHPDHeading& reference,
                       int shift, const double* weights) {
    double distance = 0.0, weight_sum = 0.0;
    const int sectors = query.sectors;
    for (int channel = 0; channel < RHPDHeadingParams::channels; ++channel) {
        int effective = 0;
        double similarity = 0.0;
        for (int sector = 0; sector < sectors; ++sector) {
            const int matched = (sector + sectors - shift) % sectors;
            const double n1 = query.column_norms(channel, sector);
            const double n2 = reference.column_norms(channel, matched);
            if (n1 < 1e-6 || n2 < 1e-6) continue;
            similarity += sparseDot(query, channel * sectors + sector,
                                    reference, channel * sectors + matched) / (n1 * n2);
            ++effective;
        }
        if (effective > 0) {
            distance += weights[channel] * (1.0 - similarity / effective);
            weight_sum += weights[channel];
        }
    }
    return weight_sum < 1e-6 ? 1.0 : distance / weight_sum;
}

int alignSectors(const Eigen::MatrixXd& query, const Eigen::MatrixXd& reference) {
    int best_shift = 0;
    double min_diff = std::numeric_limits<double>::max();
    Eigen::MatrixXd shifted(1, reference.cols());
    for (int shift = 0; shift < query.cols(); ++shift) {
        shifted.leftCols(shift) = reference.rightCols(shift);
        shifted.rightCols(reference.cols() - shift) = reference.leftCols(reference.cols() - shift);
        const double diff = (query - shifted).norm();
        if (diff < min_diff) {
            min_diff = diff;
            best_shift = shift;
        }
    }
    return best_shift;
}
} // namespace

std::pair<double, int> matchRHPDHeading(const RHPDHeading& query,
                                      const RHPDHeading& reference,
                                      const RHPDHeadingParams& params) {
    if (query.rings != reference.rings || query.sectors != reference.sectors || query.sectors <= 0)
        return {1.0, 0};
    const int coarse = alignSectors(query.sector_key, reference.sector_key);
    const int radius = std::max(1, static_cast<int>(std::round(0.5 * params.search_ratio * query.sectors)));
    double min_distance = std::numeric_limits<double>::max();
    int best_shift = 0;
    const auto consider = [&](int shift) {
        const double distance = distanceAtShift(query, reference, shift, params.weights.data());
        if (distance < min_distance) {
            min_distance = distance;
            best_shift = shift;
        }
    };
    consider(coarse);
    for (int i = 1; i <= radius; ++i) {
        consider((coarse + i) % query.sectors);
        consider(((coarse - i) % query.sectors + query.sectors) % query.sectors);
    }
    return {min_distance, best_shift};
}
} // namespace n3mapping
