#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace n3mapping::visibility_detail {

inline int originalAngularBin(double y, double x, double offset, double resolution,
                              int bins, bool wrap) {
  double angle = std::atan2(y, x) * 180.0 / M_PI + offset;
  if (wrap) {
    if (angle >= 360.0) angle -= 360.0;
    if (angle < 0.0) angle += 360.0;
  }
  return std::clamp(static_cast<int>(std::floor(angle / resolution)), 0, bins - 1);
}


// A table entry is usable only if the entire angle interval belongs to one
// output bin. Intervals touching bin boundaries use the original atan2 formula.
// The 1e-9 degree guard conservatively covers double-precision rounding in
// ratios, quadrant transforms and the elevation square root. Only the bin is
// accelerated: distances and occlusion evidence retain their original values.
class AngularBinLookup {
  static constexpr int kRatioIntervals = 1024;

 public:
  AngularBinLookup(double offset, double resolution, int bins, bool wrap)
      : offset_(offset), resolution_(resolution), bins_(bins), wrap_(wrap) {
    static const auto angles = [] {
      std::array<double, kRatioIntervals + 1> values{};
      for (int i = 0; i <= kRatioIntervals; ++i) values[i] = std::atan2(static_cast<double>(i) / kRatioIntervals, 1.0);
      return values;
    }();
    for (int octant = 0; octant < 8; ++octant) {
      for (int i = 0; i < kRatioIntervals; ++i) {
        double lower = angles[i];
        double upper = angles[i + 1];
        if (octant & 1) {
          const auto old = lower;
          lower = M_PI_2 - upper;
          upper = M_PI_2 - old;
        }
        if (octant & 2) {
          const auto old = lower;
          lower = M_PI - upper;
          upper = M_PI - old;
        }
        if (octant & 4) {
          const auto old = lower;
          lower = -upper;
          upper = -old;
        }
        lower = lower * 180.0 / M_PI + offset - 1e-9;
        upper = upper * 180.0 / M_PI + offset + 1e-9;
        int bin = -1;
        if (lower >= 0 && upper < (wrap ? 360.0 : 180.0)) {
          const int first = static_cast<int>(std::floor(lower / resolution));
          const int last = static_cast<int>(std::floor(upper / resolution));
          if (first == last) bin = std::clamp(first, 0, bins - 1);
        }
        table_[octant * kRatioIntervals + i] = bin;
      }
    }
  }

  int operator()(double y, double x) const {
    const int bin = lookup(y, x);
    return bin >= 0 ? bin : originalAngularBin(y, x, offset_, resolution_, bins_, wrap_);
  }

  int elevation(double z, double x, double y) const {
    const double horizontal_squared = x * x + y * y;
    if (horizontal_squared > 16.0 * std::numeric_limits<double>::min() &&
        std::isfinite(horizontal_squared)) {
      const int bin = lookup(z, std::sqrt(horizontal_squared));
      if (bin >= 0) return bin;
    }
    // Near a bin boundary (or outside normal arithmetic), preserve the exact
    // original hypot + atan2 path. The interval guard covers sqrt rounding.
    return originalAngularBin(z, std::hypot(x, y), offset_, resolution_, bins_, wrap_);
  }

 private:
  int lookup(double y, double x) const {
    const double ax = std::abs(x);
    const double ay = std::abs(y);
    const double maximum = std::max(ax, ay);
    if (maximum > 0 && std::isfinite(maximum)) {
      const int octant = (ay > ax) + 2 * (x < 0) + 4 * (y < 0);
      const int index = std::min(kRatioIntervals - 1,
          static_cast<int>(std::min(ax, ay) / maximum * kRatioIntervals));
      const int bin = table_[octant * kRatioIntervals + index];
      if (bin >= 0) return bin;
    }
    return -1;
  }

  double offset_;
  double resolution_;
  int bins_;
  bool wrap_;
  std::array<int, kRatioIntervals * 8> table_;
};

}  // namespace n3mapping::visibility_detail
