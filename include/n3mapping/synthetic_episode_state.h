#pragma once

namespace n3mapping::synthetic {

enum class EpisodeFrameStage {
    INITIAL_LOCALIZATION,
    INITIAL_LOCK,
    REENTRY_LOCALIZATION,
    REENTRY_LOCK,
    TRACKING,
    TRACKING_LOSS,
};

struct EpisodeFrameTransition {
    bool locked_before_frame = false;
    bool locked_after_frame = false;
    bool ever_locked_before_frame = false;
    bool ever_locked_after_frame = false;
    bool localization_attempted = false;
    bool tracking_processed = false;
    bool lock_event = false;
    bool tracking_loss_event = false;
    EpisodeFrameStage stage = EpisodeFrameStage::INITIAL_LOCALIZATION;
};

inline const char* episodeFrameStageName(EpisodeFrameStage stage)
{
    switch (stage) {
    case EpisodeFrameStage::INITIAL_LOCALIZATION: return "initial_localization";
    case EpisodeFrameStage::INITIAL_LOCK: return "initial_lock";
    case EpisodeFrameStage::REENTRY_LOCALIZATION: return "reentry_localization";
    case EpisodeFrameStage::REENTRY_LOCK: return "reentry_lock";
    case EpisodeFrameStage::TRACKING: return "tracking";
    case EpisodeFrameStage::TRACKING_LOSS: return "tracking_loss";
    }
    return "unknown";
}

inline const char* episodeLockEventTypeName(EpisodeFrameStage stage)
{
    if (stage == EpisodeFrameStage::INITIAL_LOCK) return "initial_lock";
    if (stage == EpisodeFrameStage::REENTRY_LOCK) return "reentry_lock";
    return "none";
}

// Audit-only wrapper state for processLocalizationFrame(). The runtime localizer
// remains the source of output_success/output_relocalization_locked; this helper
// only makes the episode transition and its scoring/denominator semantics explicit.
inline EpisodeFrameTransition advanceEpisodeFrameState(
    bool current_locked,
    bool ever_locked,
    bool output_success,
    bool output_relocalization_locked)
{
    EpisodeFrameTransition transition;
    transition.locked_before_frame = current_locked;
    transition.locked_after_frame = current_locked;
    transition.ever_locked_before_frame = ever_locked;
    transition.ever_locked_after_frame = ever_locked;
    transition.localization_attempted = !current_locked;
    transition.tracking_processed = current_locked;

    if (output_relocalization_locked) {
        transition.localization_attempted = true;
        transition.lock_event = true;
        transition.locked_after_frame = true;
        transition.ever_locked_after_frame = true;
        transition.stage = ever_locked
            ? EpisodeFrameStage::REENTRY_LOCK
            : EpisodeFrameStage::INITIAL_LOCK;
        return transition;
    }

    if (current_locked) {
        if (output_success) {
            transition.stage = EpisodeFrameStage::TRACKING;
        } else {
            transition.locked_after_frame = false;
            transition.tracking_loss_event = true;
            transition.stage = EpisodeFrameStage::TRACKING_LOSS;
        }
        return transition;
    }

    transition.stage = ever_locked
        ? EpisodeFrameStage::REENTRY_LOCALIZATION
        : EpisodeFrameStage::INITIAL_LOCALIZATION;
    return transition;
}

} // namespace n3mapping::synthetic
