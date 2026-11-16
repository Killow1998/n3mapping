#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <Eigen/Core>
namespace n3mapping {

constexpr int RHPD_PLANE_BINS = 14;
constexpr int RHPD_PLANE_CHANS = 4;
constexpr int RHPD_PLANE_DIM   = RHPD_PLANE_BINS * RHPD_PLANE_BINS * RHPD_PLANE_CHANS;
constexpr int RHPD_PART_A_DIM = RHPD_PLANE_DIM * 3;

constexpr int RHPD_RINGS = 16;
constexpr int RHPD_ZLAYERS = 8;
constexpr int RHPD_RH_CHANS = 4;
constexpr int RHPD_PART_B_DIM = RHPD_RINGS * RHPD_ZLAYERS * RHPD_RH_CHANS;

constexpr int RHPD_NEG_SPACE_SECTORS = 36;
constexpr int RHPD_NEG_SPACE_CHANS = 3;
constexpr int RHPD_NEG_SPACE_DIM     = RHPD_NEG_SPACE_SECTORS * RHPD_NEG_SPACE_CHANS;
constexpr int RHPD_PCA_CONF_DIM       = 1;
constexpr int RHPD_AUX_DIM = RHPD_NEG_SPACE_DIM + RHPD_PCA_CONF_DIM;

constexpr int RHPD_DIM = RHPD_PART_A_DIM + RHPD_PART_B_DIM + RHPD_AUX_DIM;
constexpr int RHPD_COARSE_KEY_DIM = RHPD_RINGS + RHPD_NEG_SPACE_SECTORS + RHPD_PCA_CONF_DIM;

struct RHPDCells {
    std::vector<uint16_t> indices;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> values;
    void append(uint16_t index, const Eigen::Vector4d& value);
    bool isZero(double tolerance = 0.0) const;
};
struct RHPDPlace {
    std::array<RHPDCells, 3> planes;
    RHPDCells ring_height;
    Eigen::Matrix<double, RHPD_AUX_DIM, 1> auxiliary =
        Eigen::Matrix<double, RHPD_AUX_DIM, 1>::Zero();
    bool isZero(double tolerance = 0.0) const;
    size_t storedValues() const;
};
using RHPDPlaceCache = std::map<int64_t, std::shared_ptr<const RHPDPlace>>;

Eigen::VectorXd rhpdDenseValues(const RHPDPlace& place);
}
