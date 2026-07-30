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
    bool descriptor_supported = false;
    bool spatial_only = false;
    double predicted_translation_norm = 0.0;
    double icp_correction_yaw_abs = 0.0;
    double segment_translation_median = 0.0;
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
    static constexpr double kLargePredictedTranslationM = 5.0;
    static constexpr double kYawFlipRad = 2.8;
    static constexpr double kLargeSegmentTranslationM = 2.0;

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
        const bool has_descriptor = f.descriptor_supported || clamp01(f.descriptor_score) > 0.0;

        // A spatial candidate is proposed by the drifted poses, so it does
        // need confirmation from somewhere the poses cannot reach. A
        // descriptor hit is one such source; neighbouring keyframes that all
        // register consistently against the same match are another, and a
        // stronger one. Requiring the descriptor specifically discarded 26 of
        // the 57 best-evidenced candidates in the session.
        const bool segment_confirmed = segment_consistency >= 1.0 - 1e-9 &&
                                       segment_support > 0.0;
        if (f.spatial_only && !has_descriptor && !segment_confirmed) {
            result.decision = LoopDecision::Reject;
            result.reason = "spatial_only_unconfirmed";
            result.risk_flags = "source";
            return result;
        }

        // Every accepted loop needs at least one confirmation the drifted
        // poses could not have produced: a descriptor hit, or neighbouring
        // keyframes registering consistently against the same match. This
        // replaces a conjunction whose first clause was predicted_translation_
        // norm -- the separation the drifted poses report, which grows
        // precisely for the loops that would repair the drift, and which
        // rejected 86 candidates registering at a median fitness of 0.052.
        if (segment_consistency <= 0.5 && !has_descriptor) {
            result.decision = LoopDecision::Reject;
            result.reason = "unconfirmed_weak_segment";
            result.risk_flags = "segment";
            return result;
        }

        // This also subsumes what used to be a separate rule for a supported
        // but inconsistent segment: it asked for consistency < 0.5 with no
        // descriptor, which the line above has already caught at <= 0.5, so
        // the branch could never be reached.

        // A yaw correction near half a turn is a flipped match whatever the
        // segment says, so this one keeps its own evidence and drops the
        // segment clause.
        if (std::isfinite(f.icp_correction_yaw_abs) &&
            f.icp_correction_yaw_abs > kYawFlipRad) {
            result.decision = LoopDecision::Reject;
            result.reason = "yaw_flip";
            result.risk_flags = "yaw";
            return result;
        }

        if (has_descriptor) {
            result.decision = LoopDecision::Accept;
            result.reason = "descriptor_geometry_consistent";
            result.risk_flags = segment_support < 0.5 ? "segment_insufficient" : "none";
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
