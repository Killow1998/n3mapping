#pragma once

#include <algorithm>
#include <cmath>
#include <string>

namespace n3mapping {

struct LoopFeatures {
    double descriptor_score = 0.0;     // higher is better
    double spatial_score = 0.0;        // higher is better
    double geometric_overlap = 0.0;    // higher is better
    double temporal_gap = 0.0;         // normalized, higher means older
    double local_map_consistency = 0.0;// higher is better
    double segment_consistency = 0.0;  // higher is better
    double segment_support = 0.0;      // normalized, higher means more neighbors agreed
};

enum class LoopDecision {
    Accept,
    Reject
};

struct LoopRefereeDecision {
    LoopDecision decision = LoopDecision::Reject;
    double energy = 0.0;
    std::string reason = "not_evaluated";
    std::string risk_flags = "not_available";
};

class LoopReferee {
public:
    // ponytail: fixed first-principles weights; tune only after matrix evidence says this model is right.
    static double energy(const LoopFeatures& f)
    {
        const double segment = clamp01(f.segment_support) * clamp01(f.segment_consistency);
        return 0.20 * clamp01(f.descriptor_score) +
               0.10 * clamp01(f.spatial_score) +
               0.20 * clamp01(f.geometric_overlap) +
               0.20 * clamp01(f.local_map_consistency) +
               0.35 * segment -
               0.05 * (1.0 - clamp01(f.temporal_gap));
    }

    static LoopRefereeDecision evaluate(const LoopFeatures& f)
    {
        LoopRefereeDecision result;
        result.energy = energy(f);
        const double segment_support = clamp01(f.segment_support);
        const double segment_consistency = clamp01(f.segment_consistency);

        if (segment_support >= 0.5 && segment_consistency < 0.5) {
            result.decision = LoopDecision::Reject;
            result.reason = "segment_inconsistent";
            result.risk_flags = "segment";
            return result;
        }

        result.decision = result.energy > 0.55 ? LoopDecision::Accept : LoopDecision::Reject;
        result.reason = "segment_referee_energy";
        result.risk_flags = segment_support < 0.5 ? "segment_insufficient" : "none";
        return result;
    }

    static LoopDecision decide(const LoopFeatures& f)
    {
        return evaluate(f).decision;
    }

private:
    static double clamp01(double v)
    {
        return std::isfinite(v) ? std::clamp(v, 0.0, 1.0) : 0.0;
    }
};

} // namespace n3mapping
