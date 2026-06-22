#pragma once

#include <algorithm>
#include <cmath>

namespace n3mapping {

struct LoopFeatures {
    double descriptor_score = 0.0;     // higher is better
    double spatial_score = 0.0;        // higher is better
    double geometric_overlap = 0.0;    // higher is better
    double temporal_gap = 0.0;         // normalized, higher means older
    double local_map_consistency = 0.0;// higher is better
};

enum class LoopDecision {
    Accept,
    Reject
};

class LoopReferee {
public:
    // ponytail: fixed first-principles weights; tune only after matrix evidence says this model is right.
    static double energy(const LoopFeatures& f)
    {
        return 0.35 * clamp01(f.descriptor_score) +
               0.20 * clamp01(f.spatial_score) +
               0.35 * clamp01(f.geometric_overlap) +
               0.20 * clamp01(f.local_map_consistency) -
               0.10 * (1.0 - clamp01(f.temporal_gap));
    }

    static LoopDecision decide(const LoopFeatures& f)
    {
        return energy(f) > 0.45 ? LoopDecision::Accept : LoopDecision::Reject;
    }

private:
    static double clamp01(double v)
    {
        return std::isfinite(v) ? std::clamp(v, 0.0, 1.0) : 0.0;
    }
};

} // namespace n3mapping
