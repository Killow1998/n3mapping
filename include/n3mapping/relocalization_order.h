#pragma once

#include <cmath>
#include <Eigen/Geometry>

namespace n3mapping {

// Exact score ordering. A pairwise epsilon is not an equivalence relation
// and cannot be used inside std::sort. Non-finite scores all rank last.
inline int compareRelocalizationScore(double lhs, double rhs) {
  const bool lhs_finite = std::isfinite(lhs);
  const bool rhs_finite = std::isfinite(rhs);
  if (lhs_finite != rhs_finite) return lhs_finite ? -1 : 1;
  if (!lhs_finite || lhs == rhs) return 0;
  return lhs > rhs ? -1 : 1;
}

// Break equal-score, equal-place ties by pose, not insertion order. The
// invalid-value handling also keeps diagnostic/synthetic inputs ordered.
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

} // namespace n3mapping
