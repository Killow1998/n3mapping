#include "mapped_target_state.h"
#include "point_cloud_target_view.h"
#include "n3mapping/async_log.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sys/mman.h>
#include <unistd.h>

namespace n3mapping {
std::size_t PointCloudMatcher::PreparedTargetLevel::size() const {
  return mapped ? mapped->header.points : cloud ? cloud->size() : 0;
}
bool PointCloudMatcher::PreparedTargetLevel::ready() const {
  return mapped || (cloud && kdtree);
}

namespace registration_detail {
Eigen::Matrix4d MappedTargetLevel::covariance(std::size_t index) const {
  if (covariance_from_normal) return covarianceFromNormal(normal(index));
  Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
  double values[9];
  std::memcpy(values, covariances + index * 72, sizeof(values));
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) result(row, col) = values[row * 3 + col];
  return result;
}

AtlasMapping::AtlasMapping() : impl_(std::make_unique<Impl>()) {}
AtlasMapping::~AtlasMapping() = default;
const char* AtlasMapping::data() const {
  return static_cast<const char*>(impl_->region.get_address());
}
std::size_t AtlasMapping::size() const { return impl_->region.get_size(); }

std::shared_ptr<AtlasMapping> AtlasMapping::open(const std::string& path,
                                               std::string* error) {
  try {
    auto mapping = std::shared_ptr<AtlasMapping>(new AtlasMapping);
    mapping->impl_->file = boost::interprocess::file_mapping(
        path.c_str(), boost::interprocess::read_only);
    mapping->impl_->region = boost::interprocess::mapped_region(
        mapping->impl_->file, boost::interprocess::read_only);
    return mapping;
  } catch (const std::exception& failure) {
    if (error) *error = failure.what();
    return {};
  }
}

void AtlasMapping::releasePages() {
  if (madvise(const_cast<char*>(data()), size(), MADV_DONTNEED) != 0)
    logEvent("atlas_page_release_failed", "bytes=" + std::to_string(size()));
}

void AtlasMapping::prefetch(const std::vector<Eigen::AlignedBox3d>& requested) {
  if (impl_->levels.empty()) return;
  using namespace std::chrono_literals;
  if (impl_->pending.valid()) {
    if (impl_->pending.wait_for(0s) != std::future_status::ready) return;
    impl_->pending.get();
    impl_->active_key = std::move(impl_->pending_key);
  }
  std::vector<std::array<double, 6>> key;
  for (const auto& box : requested) {
    std::array<double, 6> extent;
    for (int axis = 0; axis < 3; ++axis) {
      extent[axis] = std::floor(box.min()[axis] / 4.0) * 4.0;
      extent[axis + 3] = std::ceil(box.max()[axis] / 4.0) * 4.0;
    }
    key.push_back(extent);
  }
  std::sort(key.begin(), key.end());
  key.erase(std::unique(key.begin(), key.end()), key.end());
  const bool covered = std::all_of(key.begin(), key.end(), [&](const auto& box) {
    return std::any_of(impl_->active_key.begin(), impl_->active_key.end(), [&](const auto& active) {
      for (int axis = 0; axis < 3; ++axis)
        if (box[axis] < active[axis] || box[axis + 3] > active[axis + 3]) return false;
      return true;
    });
  });
  if (key.empty() || covered) return;
  // One extra cell retains overlap while the observed scan boundary moves.
  for (auto& box : key) for (int axis = 0; axis < 3; ++axis) {
    box[axis] -= 4;
    box[axis + 3] += 4;
  }
  impl_->pending_key = key;
  impl_->pending = impl_->worker.submit([this, key = std::move(key)] {
    const auto started = std::chrono::steady_clock::now();
    const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    std::vector<std::uint8_t> active((size() + page_size - 1) / page_size);
    const auto mark = [&](const char* begin, std::size_t bytes) {
      if (!bytes) return;
      const auto offset = static_cast<std::size_t>(begin - data());
      const auto last = (offset + bytes - 1) / page_size;
      std::fill(active.begin() + offset / page_size, active.begin() + last + 1, 1);
    };
    for (const auto& level : impl_->levels) {
      const auto visit = [&](auto&& self, std::uint32_t index) -> void {
        const auto& node = level.node(index);
        const auto& bounds = level.search_index->bounds.nodes[index];
        const bool intersects = std::any_of(key.begin(), key.end(), [&](const auto& box) {
          for (int axis = 0; axis < 3; ++axis)
            if (bounds.lower[axis] > box[axis + 3] || bounds.upper[axis] < box[axis])
              return false;
          return true;
        });
        if (!intersects) return;
        if (node.left != UINT32_MAX) {
          self(self, node.left);
          self(self, node.right);
        } else {
          const auto first = node.node_type.lr.first;
          const auto count = node.node_type.lr.last - first;
          mark(level.points + first * 32ULL, count * 32ULL);
          mark(level.normals + first * 32ULL, count * 32ULL);
          if (!level.covariance_from_normal)
            mark(level.covariances + first * 72ULL, count * 72ULL);
        }
      };
      visit(visit, level.header.root);
    }
    std::size_t active_pages = 0, failures = 0;
    volatile unsigned char touched = 0;
    for (std::size_t first = 0; first < active.size();) {
      const bool selected = active[first];
      auto last = first + 1;
      while (last < active.size() && bool(active[last]) == selected) ++last;
      auto* address = const_cast<char*>(data()) + first * page_size;
      const auto bytes = std::min(size(), last * page_size) - first * page_size;
      if (selected) {
        failures += madvise(address, bytes, MADV_WILLNEED) != 0;
        for (auto page = first; page < last; ++page)
          touched ^= static_cast<unsigned char>(data()[page * page_size]);
        active_pages += last - first;
      } else {
        failures += madvise(address, bytes, MADV_DONTNEED) != 0;
      }
      first = last;
    }
    (void)touched;
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::size_t index_bytes = 0;
    for (const auto& level : impl_->levels)
      index_bytes += level.search_index->memoryBytes();
    logEvent("atlas_active_region_ready", "regions=" + std::to_string(key.size()) +
        " active_geometry_bytes=" + std::to_string(active_pages * page_size) +
        " index_bytes=" + std::to_string(index_bytes) +
        " prepare_ms=" + std::to_string(wall_ms) +
        " advice_failures=" + std::to_string(failures));
  });
}
} // namespace registration_detail
} // namespace n3mapping
