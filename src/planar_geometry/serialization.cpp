#include "serialization.h"
#include <limits>
namespace n3mapping {
void writeSharedPlanes(const std::vector<SharedPlane>& planes, N3Map& map) {
  for (const auto& plane : planes) {
    auto* p = map.add_shared_planes();
    for (int a = 0; a < 3; ++a) {
      p->add_normal(plane.normal[a]);
      p->add_origin(plane.origin[a]);
    }
    p->set_offset(plane.offset);
    p->set_sigma(plane.sigma);
    for (const auto& observation : plane.observations) {
      auto* o = p->add_observations();
      o->set_node_id(observation.node_id);
      o->set_count(observation.moment.count);
      for (int a = 0; a < 3; ++a) o->add_sum(observation.moment.sum[a]);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) o->add_square(observation.moment.square(r, c));
    }
  }
}
bool readSharedPlanes(const N3Map& map, std::vector<SharedPlane>& planes) {
  std::vector<SharedPlane> parsed;
  size_t observations = 0;
  for (const auto& p : map.shared_planes()) {
    if (p.normal_size() != 3 || p.origin_size() != 3 ||
        (observations += p.observations_size()) > 1000000) return false;
    SharedPlane plane;
    for (int a = 0; a < 3; ++a) {
      plane.normal[a] = p.normal(a);
      plane.origin[a] = p.origin(a);
    }
    plane.offset = p.offset();
    plane.sigma = p.sigma();
    for (const auto& o : p.observations()) {
      if (o.sum_size() != 3 || o.square_size() != 9 ||
          o.count() > uint32_t(std::numeric_limits<int>::max())) return false;
      PlaneObservation observation;
      observation.node_id = o.node_id();
      observation.moment.count = o.count();
      for (int a = 0; a < 3; ++a) observation.moment.sum[a] = o.sum(a);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) observation.moment.square(r, c) = o.square(3*r+c);
      plane.observations.push_back(std::move(observation));
    }
    parsed.push_back(std::move(plane));
  }
  planes.swap(parsed);
  return true; // Semantic validation belongs to the transactional graph load.
}
}  // namespace n3mapping
