#include "n3mapping/submap_graph_trial_runtime.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

#include <Eigen/Geometry>

#include "n3mapping/product_build_identity.h"

namespace n3mapping {
namespace {

std::mutex& outputMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
                break;
        }
    }
    return output.str();
}

void comma(std::ostream& output, bool* first) {
    if (*first) {
        *first = false;
    } else {
        output << ',';
    }
}

void appendString(std::ostream& output, bool* first, const char* key,
                  const std::string& value) {
    comma(output, first);
    output << '"' << key << "\":\"" << jsonEscape(value) << '"';
}

void appendBool(std::ostream& output, bool* first, const char* key,
                bool value) {
    comma(output, first);
    output << '"' << key << "\":" << (value ? "true" : "false");
}

void appendSize(std::ostream& output, bool* first, const char* key,
                std::size_t value) {
    comma(output, first);
    output << '"' << key << "\":" << value;
}

void appendNumber(std::ostream& output, bool* first, const char* key,
                  double value) {
    comma(output, first);
    output << '"' << key << "\":";
    if (std::isfinite(value)) {
        output << std::setprecision(17) << value;
    } else {
        output << "null";
    }
}

void appendFiniteNumber(std::ostream& output, double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(17) << value;
    } else {
        output << "null";
    }
}

void appendPose(std::ostream& output, const char* key,
                const Eigen::Isometry3d& pose) {
    const Eigen::Quaterniond quaternion(pose.rotation());
    output << '"' << key << "\":{\"translation\":[";
    appendFiniteNumber(output, pose.translation().x());
    output << ',';
    appendFiniteNumber(output, pose.translation().y());
    output << ',';
    appendFiniteNumber(output, pose.translation().z());
    output << "],\"quaternion_xyzw\":[";
    appendFiniteNumber(output, quaternion.x());
    output << ',';
    appendFiniteNumber(output, quaternion.y());
    output << ',';
    appendFiniteNumber(output, quaternion.z());
    output << ',';
    appendFiniteNumber(output, quaternion.w());
    output << "]}";
}

std::string serializeRecord(
    const SubmapGraphSnapshot& snapshot,
    const Config& config,
    const std::string& runtime_source,
    const std::string& context,
    const SubmapGraphTrialDiagnostics& trial) {
    std::ostringstream output;
    bool first = true;
    const auto build_identity = productBuildIdentity();
    output << '{';
    appendString(output, &first, "schema",
                 "n3mapping_submap_graph_trial_v1");
    appendString(output, &first, "record_type", "checkpoint");
    appendString(output, &first, "runtime_source", runtime_source);
    appendString(output, &first, "context", context);
    appendString(output, &first, "mode", config.mode);
    appendString(output, &first, "product_commit", build_identity.commit);
    appendString(output, &first, "product_profile_sha256",
                 build_identity.product_profile_sha256);
    appendString(output, &first, "product_build_type",
                 build_identity.build_type);
    appendString(output, &first, "product_research_tools",
                 build_identity.research_tools);
    appendBool(output, &first, "product_verified",
               build_identity.verified);
    appendBool(output, &first, "no_writeback", true);
    appendBool(output, &first, "snapshot_valid", snapshot.valid);
    appendString(output, &first, "snapshot_failure_reason",
                 snapshot.failure_reason);
    appendSize(output, &first, "snapshot_node_count", snapshot.nodes.size());
    appendSize(output, &first, "snapshot_source_edge_count",
               snapshot.source_edge_count);
    appendSize(output, &first, "snapshot_intra_edge_count",
               snapshot.intra_submap_edge_count);
    appendSize(output, &first, "snapshot_cross_edge_count",
               snapshot.cross_submap_edge_count);
    appendSize(output, &first, "snapshot_unassigned_keyframe_count",
               snapshot.unassigned_keyframe_ids.size());
    appendSize(output, &first, "snapshot_unassigned_edge_count",
               snapshot.unassigned_endpoint_edge_count);
    appendSize(output, &first, "snapshot_floor_count",
               snapshot.source_floor_constraint_count);
    appendSize(output, &first, "snapshot_assigned_floor_count",
               snapshot.assigned_floor_constraint_count);
    appendSize(output, &first, "snapshot_unassigned_floor_count",
               snapshot.unassigned_floor_constraint_count);
    appendNumber(output, &first, "snapshot_mean_cross_translation_residual_m",
                 snapshot.mean_cross_edge_translation_residual_m);
    appendNumber(output, &first, "snapshot_max_cross_translation_residual_m",
                 snapshot.max_cross_edge_translation_residual_m);
    appendNumber(output, &first, "snapshot_mean_cross_rotation_residual_rad",
                 snapshot.mean_cross_edge_rotation_residual_rad);
    appendNumber(output, &first, "snapshot_max_cross_rotation_residual_rad",
                 snapshot.max_cross_edge_rotation_residual_rad);
    appendBool(output, &first, "valid", trial.valid);
    appendBool(output, &first, "attempted", trial.attempted);
    appendBool(output, &first, "solved", trial.solved);
    appendString(output, &first, "failure_reason", trial.failure_reason);
    appendSize(output, &first, "node_count", trial.node_count);
    appendSize(output, &first, "gauge_anchor_count",
               trial.gauge_anchor_count);
    appendSize(output, &first, "active_edge_factor_count",
               trial.active_edge_factor_count);
    appendSize(output, &first, "intra_submap_constant_edge_count",
               trial.intra_submap_constant_edge_count);
    appendSize(output, &first, "full_6d_factor_count",
               trial.full_6d_factor_count);
    appendSize(output, &first, "xy_yaw_lifted_factor_count",
               trial.xy_yaw_lifted_factor_count);
    appendSize(output, &first, "robust_factor_count",
               trial.robust_factor_count);
    appendSize(output, &first, "session_odometry_factor_count",
               trial.session_odometry_factor_count);
    appendSize(output, &first, "explicit_information_factor_count",
               trial.explicit_information_factor_count);
    appendSize(output, &first, "fallback_noise_factor_count",
               trial.fallback_noise_factor_count);
    appendSize(output, &first, "floor_factor_count",
               trial.floor_factor_count);
    appendNumber(output, &first, "initial_nonlinear_error",
                 trial.initial_nonlinear_error);
    appendNumber(output, &first, "final_nonlinear_error",
                 trial.final_nonlinear_error);
    appendNumber(output, &first, "nonlinear_error_reduction",
                 trial.nonlinear_error_reduction);
    appendNumber(output, &first, "max_translation_delta_m",
                 trial.max_translation_delta_m);
    appendNumber(output, &first, "max_rotation_delta_rad",
                 trial.max_rotation_delta_rad);

    comma(output, &first);
    output << "\"nodes\":[";
    for (std::size_t index = 0; index < trial.nodes.size(); ++index) {
        if (index > 0) output << ',';
        const auto& node = trial.nodes[index];
        output << "{\"submap_id\":" << node.submap_id
               << ",\"gauge_anchor\":"
               << (node.gauge_anchor ? "true" : "false") << ',';
        appendPose(output, "initial_pose", node.initial_pose);
        output << ',';
        appendPose(output, "optimized_pose", node.optimized_pose);
        output << ",\"translation_delta_m\":";
        appendFiniteNumber(output, node.translation_delta_m);
        output << ",\"rotation_delta_rad\":";
        appendFiniteNumber(output, node.rotation_delta_rad);
        output << '}';
    }
    output << "]}";
    return output.str();
}

bool appendLine(const std::string& path, const std::string& line) {
    std::lock_guard<std::mutex> lock(outputMutex());
    try {
        const std::filesystem::path filesystem_path(path);
        const auto parent = filesystem_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        std::ofstream file(path, std::ios::out | std::ios::app);
        if (!file.is_open()) return false;
        file << line << '\n';
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

bool isSubmapGraphTrialCheckpointContext(const std::string& context) {
    return context == "save_map" || context == "save_extended_map" ||
           context == "loop_commit" || context == "cross_session_loop";
}

std::string resolveSubmapGraphTrialPath(const Config& config) {
    if (config.map_save_path.empty()) {
        return "submap_graph_trial.jsonl";
    }
    return (std::filesystem::path(config.map_save_path) /
            "submap_graph_trial.jsonl").string();
}

SubmapGraphTrialRuntimeResult runSubmapGraphTrialCheckpoint(
    const SubmapGraphSnapshot& snapshot,
    const Config& config,
    const std::string& runtime_source,
    const std::string& context) {
    SubmapGraphTrialRuntimeResult result;
    if (!config.submap_shadow_enable ||
        !isSubmapGraphTrialCheckpointContext(context)) {
        return result;
    }
    result.checkpoint = true;
    result.output_path = resolveSubmapGraphTrialPath(config);
    result.diagnostics = evaluateSubmapGraphOptimizationTrial(
        snapshot, config);
    result.persisted = appendLine(
        result.output_path,
        serializeRecord(snapshot, config, runtime_source, context,
                        result.diagnostics));
    return result;
}

}  // namespace n3mapping
