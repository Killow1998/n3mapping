#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <Eigen/Core>
namespace n3mapping {
// ----------- Descriptor parameters (compile-time constants) -----------
constexpr int RHPD_PLANE_BINS  = 14;       // grid size for each planar projection
constexpr int RHPD_PLANE_CHANS = 4;        // 4 channels per bin (semantics depend on V3 flag)
constexpr int RHPD_PLANE_DIM   = RHPD_PLANE_BINS * RHPD_PLANE_BINS * RHPD_PLANE_CHANS;
constexpr int RHPD_PART_A_DIM  = RHPD_PLANE_DIM * 3;  // XY + XZ + YZ

constexpr int RHPD_RINGS       = 16;       // radial rings for Part B
constexpr int RHPD_ZLAYERS     = 8;        // height layers for Part B
constexpr int RHPD_RH_CHANS    = 4;        // density, azimuth_variance, mean_range, range_variance
constexpr int RHPD_PART_B_DIM  = RHPD_RINGS * RHPD_ZLAYERS * RHPD_RH_CHANS;  // 512

constexpr int RHPD_NEG_SPACE_SECTORS = 36;
constexpr int RHPD_NEG_SPACE_CHANS   = 3;   // nearest hit, frontier span, open/unknown state
constexpr int RHPD_NEG_SPACE_DIM     = RHPD_NEG_SPACE_SECTORS * RHPD_NEG_SPACE_CHANS;
constexpr int RHPD_PCA_CONF_DIM       = 1;
constexpr int RHPD_AUX_DIM = RHPD_NEG_SPACE_DIM + RHPD_PCA_CONF_DIM;

constexpr int RHPD_DIM = RHPD_PART_A_DIM + RHPD_PART_B_DIM + RHPD_AUX_DIM;  // 2973
constexpr int RHPD_COARSE_KEY_DIM = RHPD_RINGS + RHPD_NEG_SPACE_SECTORS + RHPD_PCA_CONF_DIM;


// Sorted occupied cells. An absent cell has four zero channels. Visibility
// cells with no returns still count as occupied storage when V3 is enabled.
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
// Dense expansion is for offline analysis and numerical comparisons only.
Eigen::VectorXd rhpdDenseValues(const RHPDPlace& place);
} // namespace n3mapping
