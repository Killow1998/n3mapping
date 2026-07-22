#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/world_localizing.h"

namespace n3mapping::eval {

struct RegistrationProbeFrame {
    std::size_t frame_index = 0;
    std::string frame_token;
    Eigen::Isometry3d oracle_pose = Eigen::Isometry3d::Identity();
    RegistrationSeedProbeResult result;
};

inline double probeYaw(const Eigen::Isometry3d& pose)
{
    return std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
}

inline double probeAngleDiff(double left, double right)
{
    double delta = left - right;
    while (delta > M_PI) delta -= 2.0 * M_PI;
    while (delta < -M_PI) delta += 2.0 * M_PI;
    return delta;
}

inline double probeTranslationError(const Eigen::Isometry3d& pose,
                                    const Eigen::Isometry3d& oracle)
{
    return (pose.translation() - oracle.translation()).norm();
}

inline double probeYawErrorDeg(const Eigen::Isometry3d& pose,
                               const Eigen::Isometry3d& oracle)
{
    return std::abs(probeAngleDiff(probeYaw(pose), probeYaw(oracle))) *
           180.0 / M_PI;
}

inline bool probeWithinGate(const RegistrationSeedProbeAttempt& attempt,
                            const Eigen::Isometry3d& oracle,
                            double translation_gate_m,
                            double yaw_gate_deg)
{
    return probeTranslationError(attempt.match.T_target_source, oracle) <=
               translation_gate_m &&
           probeYawErrorDeg(attempt.match.T_target_source, oracle) <=
               yaw_gate_deg;
}

inline void writeRegistrationProbeArtifacts(
    const std::filesystem::path& output_dir,
    const std::string& dataset,
    const std::string& sequence,
    const std::vector<RegistrationProbeFrame>& frames,
    double translation_gate_m = 1.0,
    double yaw_gate_deg = 10.0)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream attempts(output_dir / "registration_probe_attempts.csv");
    std::ofstream stages(output_dir / "registration_probe_stages.csv");
    if (!attempts.is_open() || !stages.is_open()) {
        throw std::runtime_error("failed to open registration probe output");
    }
    attempts << "frame_index,frame_token,probe_valid,probe_error,seed_kind,"
                "seed_keyframe_id,yaw_offset_deg,initial_translation_error_m,"
                "initial_yaw_error_deg,converged,quality_success,termination,"
                "final_translation_error_m,final_yaw_error_deg,within_pose_gate,"
                "fitness_score,fitness_pass,inlier_ratio,inlier_pass,"
                "derived_confidence,confidence_pass,production_quality_pass,"
                "initial_visibility_valid,initial_visibility_resolution_deg,"
                "initial_visibility_observed_bins,initial_visibility_common_bins,"
                "initial_visibility_consistent_bins,initial_visibility_conflict_bins,"
                "initial_visibility_coverage,initial_visibility_consistency,"
                "initial_visibility_conflict_ratio,initial_visibility_evidence,"
                "refined_visibility_valid,refined_visibility_resolution_deg,"
                "refined_visibility_observed_bins,refined_visibility_common_bins,"
                "refined_visibility_consistent_bins,refined_visibility_conflict_bins,"
                "refined_visibility_coverage,refined_visibility_consistency,"
                "refined_visibility_conflict_ratio,refined_visibility_evidence,"
                "production_kept_initial_pose,production_translation_error_m,"
                "production_yaw_error_deg,production_within_pose_gate,"
                "iterations,optimizer_error\n";
    stages << "frame_index,frame_token,seed_kind,seed_keyframe_id,stage_index,"
              "stage,resolution,converged,termination,fitness_score,inlier_ratio,"
              "num_inliers,iterations,optimizer_error\n";

    std::size_t valid_frames = 0;
    std::size_t oracle_attempt_frames = 0;
    std::size_t oracle_converged_frames = 0;
    std::size_t oracle_within_gate_frames = 0;
    std::size_t descriptor_attempt_frames = 0;
    std::size_t descriptor_any_converged_frames = 0;
    std::size_t descriptor_any_within_gate_frames = 0;
    std::size_t oracle_only_within_gate_frames = 0;
    std::size_t descriptor_only_within_gate_frames = 0;
    std::size_t both_within_gate_frames = 0;
    std::size_t neither_within_gate_frames = 0;

    for (const auto& frame : frames) {
        if (frame.result.valid) ++valid_frames;
        bool oracle_present = false;
        bool oracle_converged = false;
        bool oracle_within = false;
        bool descriptor_present = false;
        bool descriptor_converged = false;
        bool descriptor_within = false;
        for (const auto& attempt : frame.result.attempts) {
            const bool within = probeWithinGate(
                attempt, frame.oracle_pose, translation_gate_m, yaw_gate_deg);
            const double initial_translation =
                probeTranslationError(attempt.initial_pose, frame.oracle_pose);
            const double initial_yaw =
                probeYawErrorDeg(attempt.initial_pose, frame.oracle_pose);
            const double final_translation = probeTranslationError(
                attempt.match.T_target_source, frame.oracle_pose);
            const double final_yaw = probeYawErrorDeg(
                attempt.match.T_target_source, frame.oracle_pose);
            const double production_translation = probeTranslationError(
                attempt.production_pose, frame.oracle_pose);
            const double production_yaw = probeYawErrorDeg(
                attempt.production_pose, frame.oracle_pose);
            const bool production_within =
                production_translation <= translation_gate_m &&
                production_yaw <= yaw_gate_deg;
            attempts << frame.frame_index << ',' << frame.frame_token << ','
                     << (frame.result.valid ? "true" : "false") << ','
                     << frame.result.error << ',' << attempt.seed_kind << ','
                     << attempt.seed_keyframe_id << ',' << std::setprecision(12)
                     << attempt.yaw_offset_rad * 180.0 / M_PI << ','
                     << initial_translation << ',' << initial_yaw << ','
                     << (attempt.match.converged ? "true" : "false") << ','
                     << (attempt.match.success ? "true" : "false") << ','
                     << matchTerminationName(attempt.match.termination) << ','
                     << final_translation << ',' << final_yaw << ','
                     << (within ? "true" : "false") << ','
                     << attempt.match.fitness_score << ','
                     << (attempt.fitness_pass ? "true" : "false") << ','
                     << attempt.match.inlier_ratio << ','
                     << (attempt.inlier_pass ? "true" : "false") << ','
                     << attempt.derived_confidence << ','
                     << (attempt.confidence_pass ? "true" : "false") << ','
                     << (attempt.production_quality_pass ? "true" : "false")
                     << ','
                     << (attempt.initial_visibility.valid ? "true" : "false")
                     << ',' << attempt.initial_visibility.angular_resolution_deg
                     << ',' << attempt.initial_visibility.observed_bins
                     << ',' << attempt.initial_visibility.common_bins
                     << ',' << attempt.initial_visibility.consistent_bins
                     << ',' << attempt.initial_visibility.foreground_conflict_bins
                     << ',' << attempt.initial_visibility.observed_coverage
                     << ',' << attempt.initial_visibility.consistency_ratio
                     << ',' << attempt.initial_visibility.foreground_conflict_ratio
                     << ',' << attempt.initial_visibility.evidence_log_odds
                     << ','
                     << (attempt.refined_visibility.valid ? "true" : "false")
                     << ',' << attempt.refined_visibility.angular_resolution_deg
                     << ',' << attempt.refined_visibility.observed_bins
                     << ',' << attempt.refined_visibility.common_bins
                     << ',' << attempt.refined_visibility.consistent_bins
                     << ',' << attempt.refined_visibility.foreground_conflict_bins
                     << ',' << attempt.refined_visibility.observed_coverage
                     << ',' << attempt.refined_visibility.consistency_ratio
                     << ',' << attempt.refined_visibility.foreground_conflict_ratio
                     << ',' << attempt.refined_visibility.evidence_log_odds
                     << ','
                     << (attempt.production_kept_initial_pose ? "true" : "false")
                     << ',' << production_translation << ',' << production_yaw
                     << ',' << (production_within ? "true" : "false") << ','
                     << attempt.match.iterations << ','
                     << attempt.match.optimizer_error << '\n';
            for (std::size_t stage_index = 0;
                 stage_index < attempt.match.stages.size(); ++stage_index) {
                const auto& stage = attempt.match.stages[stage_index];
                stages << frame.frame_index << ',' << frame.frame_token << ','
                       << attempt.seed_kind << ',' << attempt.seed_keyframe_id
                       << ',' << stage_index << ',' << stage.stage << ','
                       << std::setprecision(12) << stage.resolution << ','
                       << (stage.converged ? "true" : "false") << ','
                       << matchTerminationName(stage.termination) << ','
                       << stage.fitness_score << ',' << stage.inlier_ratio << ','
                       << stage.num_inliers << ',' << stage.iterations << ','
                       << stage.optimizer_error << '\n';
            }
            if (attempt.seed_kind == "oracle_gt") {
                oracle_present = true;
                oracle_converged = oracle_converged || attempt.match.converged;
                oracle_within = oracle_within || within;
            } else if (attempt.seed_kind == "descriptor") {
                descriptor_present = true;
                descriptor_converged =
                    descriptor_converged || attempt.match.converged;
                descriptor_within = descriptor_within || within;
            }
        }
        if (oracle_present) ++oracle_attempt_frames;
        if (oracle_converged) ++oracle_converged_frames;
        if (oracle_within) ++oracle_within_gate_frames;
        if (descriptor_present) ++descriptor_attempt_frames;
        if (descriptor_converged) ++descriptor_any_converged_frames;
        if (descriptor_within) ++descriptor_any_within_gate_frames;
        if (oracle_within && descriptor_within) {
            ++both_within_gate_frames;
        } else if (oracle_within) {
            ++oracle_only_within_gate_frames;
        } else if (descriptor_within) {
            ++descriptor_only_within_gate_frames;
        } else {
            ++neither_within_gate_frames;
        }
    }

    std::ofstream summary(output_dir / "registration_probe_summary.json");
    if (!summary.is_open()) {
        throw std::runtime_error("failed to open registration probe summary");
    }
    summary << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"dataset\": \"" << dataset << "\",\n"
            << "  \"sequence\": \"" << sequence << "\",\n"
            << "  \"frame_count\": " << frames.size() << ",\n"
            << "  \"valid_frame_count\": " << valid_frames << ",\n"
            << "  \"translation_gate_m\": " << translation_gate_m << ",\n"
            << "  \"yaw_gate_deg\": " << yaw_gate_deg << ",\n"
            << "  \"oracle_attempt_frame_count\": " << oracle_attempt_frames << ",\n"
            << "  \"oracle_converged_frame_count\": " << oracle_converged_frames << ",\n"
            << "  \"oracle_within_gate_frame_count\": " << oracle_within_gate_frames << ",\n"
            << "  \"descriptor_attempt_frame_count\": " << descriptor_attempt_frames << ",\n"
            << "  \"descriptor_any_converged_frame_count\": " << descriptor_any_converged_frames << ",\n"
            << "  \"descriptor_any_within_gate_frame_count\": " << descriptor_any_within_gate_frames << ",\n"
            << "  \"oracle_only_within_gate_frame_count\": " << oracle_only_within_gate_frames << ",\n"
            << "  \"descriptor_only_within_gate_frame_count\": " << descriptor_only_within_gate_frames << ",\n"
            << "  \"both_within_gate_frame_count\": " << both_within_gate_frames << ",\n"
            << "  \"neither_within_gate_frame_count\": " << neither_within_gate_frames << "\n"
            << "}\n";
}

}  // namespace n3mapping::eval
