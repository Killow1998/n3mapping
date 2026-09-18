#include "n3mapping/ros_output_contract.h"
#include "n3mapping/core/n3mapping_core.h"

#include <json/json.h>

namespace n3mapping {

std::string localizationStatusJson(CoreRunMode mode,
                                   const core::BackendOutput& output,
                                   double stamp,
                                   const std::string& frame_id,
                                   const RealtimeLocalizationStatus* realtime) {
    const char* state = "localizing";
    const char* reason = "awaiting_localization";
    const bool valid_pose = isFiniteRigidPose(output.T_world_lidar);
    if (!std::isfinite(stamp) || stamp <= 0.0 || frame_id.empty()) {
        state = "error";
        reason = "invalid_data_header";
    } else if (!valid_pose) {
        state = "error";
        reason = "invalid_pose";
    } else if (mode != CoreRunMode::LOCALIZATION) {
        state = "initializing";
        reason = "awaiting_mapping_output";
        if (output.mapping_block_reason == core::MappingBlockReason::StaticStartGuard) {
            reason = "waiting_for_static_start_guard";
        } else if (output.mapping_block_reason == core::MappingBlockReason::OdometryDiverged) {
            state = "error";
            reason = "odometry_diverged";
        } else if (output.mapping_block_reason == core::MappingBlockReason::InvalidInput) {
            state = "error";
            reason = "invalid_mapping_input";
        } else if (output.success || output.accepted_keyframe) {
            state = "tracking";
            reason = "";
        }
    } else if (hasAuthoritativeRelocalizationInitializationPose(
                   output.relocalization_state, output.pose_source)) {
        state = "tracking";
        reason = "";
    } else if (output.relocalization_state == RelocalizationState::RECENTLY_LOST ||
               output.relocalization_state == RelocalizationState::DEGRADED_TRACKING) {
        state = "degraded";
        reason = "localization_not_authoritative";
    } else if (output.relocalization_state == RelocalizationState::LOST) {
        state = "lost";
        reason = "localization_lost";
    }

    if (realtime && !realtime->estimate_available && realtime->correction_stamp > 0.0) {
        state = "error";
        reason = realtime->input_reason.c_str();
    }
    Json::Value status(Json::objectValue);
    status["mode"] = coreRunModeName(mode);
    status["state"] = state;
    status["stamp"] = std::isfinite(stamp) ? stamp : 0.0;
    status["observation_stamp"] = realtime ? realtime->observation_stamp : status["stamp"].asDouble();
    status["correction_stamp"] = realtime ? realtime->correction_stamp :
        (std::string(state) == "tracking" ? status["stamp"].asDouble() : 0.0);
    status["frame_id"] = frame_id;
    status["reason"] = reason;
    const auto &tracking = output.tracking_performance;
    if (tracking.available) {
        auto &performance = status["tracking_performance"];
        const auto timing = [&](const char *name, double value) {
            if (std::isfinite(value) && value >= 0.0) performance[name] = value;
        };
        performance["strict_loaded_map"] = tracking.strict_loaded_map;
        timing("total_ms", tracking.tracking_total_ms);
        timing("lock_wait_ms", tracking.lock_wait_ms);
        timing("nearest_keyframe_ms", tracking.nearest_keyframe_ms);
        timing("loaded_map_cache_ms", tracking.loaded_map_cache_ms);
        timing("submap_build_ms", tracking.submap_build_ms);
        timing("target_prepare_ms", tracking.target_prepare_ms);
        timing("source_prepare_ms", tracking.source_prepare_ms);
        timing("registration_ms", tracking.registration_ms);
        timing("retry_registration_ms", tracking.retry_registration_ms);
        timing("visibility_ms", tracking.visibility_ms);
        performance["target_cache_hit"] = tracking.strict_loaded_map ?
            tracking.loaded_map_target_cache_hit : tracking.localization_target_cache_hit;
        performance["target_cache_miss"] = tracking.strict_loaded_map ?
            tracking.loaded_map_target_cache_miss : tracking.localization_target_cache_miss;
    }
    const auto &cost = output.relocalization_performance;
    if (cost.available) {
        auto &performance = status["relocalization_performance"];
        const auto timing = [&](const char *name, double value) {
            if (std::isfinite(value) && value >= 0.0) performance[name] = value;
        };
        timing("total_ms", cost.total_ms);
        timing("source_prepare_ms", cost.source_prepare_ms);
        timing("visibility_prepare_ms", cost.visibility_prepare_ms);
        timing("target_build_ms", cost.target_build_ms);
        timing("target_prepare_ms", cost.target_prepare_ms);
        timing("align_ms", cost.align_ms);
        timing("visibility_ms", cost.visibility_ms);
        performance["source_preparations"] = Json::UInt64(cost.source_preparations);
        performance["visibility_preparations"] = Json::UInt64(cost.visibility_preparations);
        performance["target_builds"] = Json::UInt64(cost.target_builds);
        performance["target_preparations"] = Json::UInt64(cost.target_preparations);
        performance["alignments"] = Json::UInt64(cost.alignments);
        performance["visibility_evaluations"] = Json::UInt64(cost.visibility_evaluations);
        performance["candidates"] = Json::UInt64(cost.candidates);
        performance["hypotheses"] = Json::UInt64(cost.hypotheses);
        performance["cache_hits"] = Json::UInt64(cost.cache_hits);
        performance["cache_misses"] = Json::UInt64(cost.cache_misses);
        performance["cache_bytes"] = Json::UInt64(cost.cache_bytes);
        performance["cache_entries"] = Json::UInt64(cost.cache_entries);
        performance["effective_cache_max_bytes"] = Json::UInt64(cost.effective_cache_max_bytes);
        performance["effective_cache_max_entries"] = Json::UInt64(cost.effective_cache_max_entries);
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, status);
}

}  // namespace n3mapping
