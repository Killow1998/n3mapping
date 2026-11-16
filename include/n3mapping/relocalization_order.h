#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace n3mapping {

inline int compareRelocalizationScore(double lhs, double rhs) {
  const bool lhs_finite = std::isfinite(lhs);
  const bool rhs_finite = std::isfinite(rhs);
  if (lhs_finite != rhs_finite) return lhs_finite ? -1 : 1;
  if (!lhs_finite || lhs == rhs) return 0;
  return lhs > rhs ? -1 : 1;
}

inline bool relocalizationPoseLess(const Eigen::Isometry3d &lhs,
                                   const Eigen::Isometry3d &rhs) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      const int order = compareRelocalizationScore(lhs(row, col), rhs(row, col));
      if (order != 0) return order < 0;
    }
  }
  return false;
}

}
