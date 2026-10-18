#include "n3mapping/RHPDescriptor.h"
#include <algorithm>
#include <cmath>

namespace n3mapping {
void RHPDCells::append(uint16_t index, const Eigen::Vector4d& value) {
    if (value.isZero(0.0)) return;
    indices.push_back(index);
    values.push_back(value);
}
bool RHPDCells::isZero(double tolerance) const {
    return std::all_of(values.begin(), values.end(), [&](const auto& value) {
        return value.isZero(tolerance);
    });
}
bool RHPDPlace::isZero(double tolerance) const {
    return ring_height.isZero(tolerance) && auxiliary.isZero(tolerance) &&
        std::all_of(planes.begin(), planes.end(), [&](const auto& plane) {
            return plane.isZero(tolerance);
        });
}
size_t RHPDPlace::storedValues() const {
    size_t count = RHPD_AUX_DIM + 4 * ring_height.values.size();
    for (const auto& plane : planes) count += 4 * plane.values.size();
    return count;
}
Eigen::VectorXd rhpdDenseValues(const RHPDPlace& place) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(RHPD_DIM);
    auto expand = [&](const RHPDCells& cells, int base) {
        for (size_t i = 0; i < cells.indices.size(); ++i)
            out.segment<4>(base + 4 * cells.indices[i]) = cells.values[i];
    };
    for (int p = 0; p < 3; ++p) expand(place.planes[p], p * RHPD_PLANE_DIM);
    expand(place.ring_height, RHPD_PART_A_DIM);
    out.tail(RHPD_AUX_DIM) = place.auxiliary;
    return out;
}
namespace {
RHPDCells flipPlane(const RHPDCells& input, bool flip_columns) {
    RHPDCells out;
    out.indices.reserve(input.indices.size());
    out.values.reserve(input.values.size());
    constexpr int bins = RHPD_PLANE_BINS;
    if (flip_columns) {
        for (size_t i = input.indices.size(); i-- > 0;) {
            out.indices.push_back(bins * bins - 1 - input.indices[i]);
            out.values.push_back(input.values[i]);
        }
    } else {
        // Reverse row groups, preserving the sorted columns within each row.
        for (size_t end = input.indices.size(); end > 0;) {
            const int row = input.indices[end - 1] / bins;
            size_t begin = end - 1;
            while (begin > 0 && input.indices[begin - 1] / bins == row) --begin;
            for (size_t i = begin; i < end; ++i) {
                out.indices.push_back((bins - 1 - row) * bins + input.indices[i] % bins);
                out.values.push_back(input.values[i]);
            }
            end = begin;
        }
    }
    return out;
}
// Accumulate four channel lanes without expanding or visiting absent cells.
double squaredDistance(const RHPDCells& a, const RHPDCells& b) {
    size_t i = 0, j = 0;
    Eigen::Vector4d sum = Eigen::Vector4d::Zero();
    while (i < a.indices.size() && j < b.indices.size()) {
        if (a.indices[i] == b.indices[j]) {
            sum.array() += (a.values[i++] - b.values[j++]).array().square();
        } else if (a.indices[i] < b.indices[j]) {
            sum.array() += a.values[i++].array().square();
        } else {
            sum.array() += b.values[j++].array().square();
        }
    }
    while (i < a.indices.size()) sum.array() += a.values[i++].array().square();
    while (j < b.indices.size()) sum.array() += b.values[j++].array().square();
    return sum.sum();
}
} // namespace

RHPDescriptor::PreparedQuery RHPDescriptor::prepareQuery(const Place& place) const {
    PreparedQuery query{place, {}, place.auxiliary};
    for (int p = 0; p < 3; ++p) query.flipped_planes[p] = flipPlane(place.planes[p], p == 0);
    if (params_.enable_negative_space) {
        constexpr int sectors = RHPD_NEG_SPACE_SECTORS;
        for (int s = 0; s < sectors; ++s)
            query.flipped_auxiliary.segment<3>(s * 3) =
                place.auxiliary.segment<3>(((s + sectors / 2) % sectors) * 3);
    }
    return query;
}

double RHPDescriptor::distance(const Place& a, const Place& b) const {
    return distance(prepareQuery(a), b);
}
double RHPDescriptor::distance(const PreparedQuery& query, const Place& b) const {
    const auto& a = query.place;
    double direct = 0.0, flipped = 0.0;
    for (int p = 0; p < 3; ++p) {
        direct += squaredDistance(a.planes[p], b.planes[p]);
        flipped += squaredDistance(query.flipped_planes[p], b.planes[p]);
    }
    const double confidence = std::min(std::clamp(a.auxiliary(RHPD_AUX_DIM - 1), 0.0, 1.0),
                                       std::clamp(b.auxiliary(RHPD_AUX_DIM - 1), 0.0, 1.0));
    const double weight = params_.enable_pca_confidence ? 0.35 + 0.65 * confidence : 1.0;
    const double ring_height = squaredDistance(a.ring_height, b.ring_height);
    const double aux = (a.auxiliary - b.auxiliary).squaredNorm();
    const double flipped_aux = (query.flipped_auxiliary - b.auxiliary).squaredNorm();
    return std::sqrt(std::min(params_.part_a_scale * weight * direct + ring_height + params_.aux_scale * 0.5 * aux,
                              params_.part_a_scale * weight * flipped + ring_height + params_.aux_scale * 0.5 * flipped_aux));
}
} // namespace n3mapping
