#pragma once

#include <cstdint>

namespace n3mapping {

struct KeyframeMapRevision {
  std::uint64_t generation = 0;
  std::uint64_t structure_revision = 0;
  std::uint64_t pose_revision = 0;

  bool operator==(const KeyframeMapRevision &other) const {
    return generation == other.generation &&
           structure_revision == other.structure_revision &&
           pose_revision == other.pose_revision;
  }

  bool operator!=(const KeyframeMapRevision &other) const {
    return !(*this == other);
  }
};

}
