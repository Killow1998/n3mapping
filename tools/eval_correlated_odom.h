#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace n3mapping::eval {

constexpr double kPi = 3.14159265358979323846;

struct CorrelatedOdomConfig {
    bool enabled = false;
    uint64_t seed = 0;
    double translation_scale_error_fraction = 0.0;
    double yaw_bias_deg_per_meter = 0.0;
    double translation_rw_std_m_per_sqrt_meter = 0.0;
    double rotation_rw_std_deg_per_sqrt_meter = 0.0;
};

struct CorrelatedOdomStats {
    std::size_t pose_count = 0;
    std::size_t increment_count = 0;
    double traveled_distance_m = 0.0;
    double translation_perturbation_squared_sum_m2 = 0.0;
    double rotation_perturbation_squared_sum_deg2 = 0.0;
    double final_translation_drift_m = 0.0;
    double final_rotation_drift_deg = 0.0;
};

inline bool hasNonzeroPerturbation(const CorrelatedOdomConfig& config)
{
    return config.translation_scale_error_fraction != 0.0 ||
        config.yaw_bias_deg_per_meter != 0.0 ||
        config.translation_rw_std_m_per_sqrt_meter != 0.0 ||
        config.rotation_rw_std_deg_per_sqrt_meter != 0.0;
}

inline void validateCorrelatedOdomConfig(const CorrelatedOdomConfig& config)
{
    const auto requireFinite = [](double value, const char* name) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(name) + " must be finite");
        }
    };
    requireFinite(config.translation_scale_error_fraction,
                  "translation scale error fraction");
    requireFinite(config.yaw_bias_deg_per_meter, "yaw bias deg per meter");
    requireFinite(config.translation_rw_std_m_per_sqrt_meter,
                  "translation random-walk std");
    requireFinite(config.rotation_rw_std_deg_per_sqrt_meter,
                  "rotation random-walk std");
    if (config.translation_scale_error_fraction <= -1.0) {
        throw std::runtime_error(
            "translation scale error fraction must be greater than -1");
    }
    if (config.translation_rw_std_m_per_sqrt_meter < 0.0) {
        throw std::runtime_error(
            "translation random-walk std must be non-negative");
    }
    if (config.rotation_rw_std_deg_per_sqrt_meter < 0.0) {
        throw std::runtime_error(
            "rotation random-walk std must be non-negative");
    }
    if (!config.enabled && hasNonzeroPerturbation(config)) {
        throw std::runtime_error(
            "nonzero odometry drift parameters require "
            "--enable_correlated_odom_drift");
    }
}

inline double rotationAngleRad(const Eigen::Matrix3d& rotation)
{
    Eigen::Quaterniond quaternion(rotation);
    quaternion.normalize();
    return Eigen::AngleAxisd(quaternion).angle();
}

inline Eigen::Matrix3d rotationVectorToMatrix(const Eigen::Vector3d& vector)
{
    const double angle = vector.norm();
    if (angle <= std::numeric_limits<double>::epsilon()) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

// Produces a pose stream by perturbing and integrating consecutive GT increments.
// Noise is sampled in each increment's local frame. Its standard deviation grows
// with sqrt(translational distance), so the accumulated error is correlated over
// time instead of being independent absolute-pose jitter.
class CorrelatedOdomGenerator {
public:
    explicit CorrelatedOdomGenerator(const CorrelatedOdomConfig& config)
        : config_(config), rng_(config.seed)
    {
        validateCorrelatedOdomConfig(config_);
    }

    Eigen::Isometry3d next(const Eigen::Isometry3d& gt_pose)
    {
        if (!gt_pose.matrix().allFinite()) {
            throw std::runtime_error("ground-truth pose contains non-finite values");
        }
        if (!initialized_) {
            initialized_ = true;
            previous_gt_ = normalizedPose(gt_pose);
            previous_odom_ = previous_gt_;
            ++stats_.pose_count;
            return previous_odom_;
        }

        const Eigen::Isometry3d normalized_gt = normalizedPose(gt_pose);
        const Eigen::Isometry3d gt_increment = previous_gt_.inverse() * normalized_gt;
        const double distance_m = gt_increment.translation().norm();
        const double sqrt_distance_m = std::sqrt(distance_m);

        Eigen::Isometry3d odom_increment = gt_increment;
        if (config_.enabled) {
            const double translation_std =
                config_.translation_rw_std_m_per_sqrt_meter * sqrt_distance_m;
            const double rotation_std_rad =
                config_.rotation_rw_std_deg_per_sqrt_meter * kPi / 180.0 *
                sqrt_distance_m;
            const Eigen::Vector3d translation_noise(
                sampleNormal(translation_std),
                sampleNormal(translation_std),
                sampleNormal(translation_std));
            Eigen::Vector3d rotation_noise(
                sampleNormal(rotation_std_rad),
                sampleNormal(rotation_std_rad),
                sampleNormal(rotation_std_rad));
            rotation_noise.z() +=
                config_.yaw_bias_deg_per_meter * kPi / 180.0 * distance_m;

            odom_increment.translation() =
                gt_increment.translation() *
                    (1.0 + config_.translation_scale_error_fraction) +
                translation_noise;
            odom_increment.linear() =
                gt_increment.rotation() * rotationVectorToMatrix(rotation_noise);

            const Eigen::Vector3d translation_perturbation =
                odom_increment.translation() - gt_increment.translation();
            stats_.translation_perturbation_squared_sum_m2 +=
                translation_perturbation.squaredNorm();
            const Eigen::Matrix3d rotation_perturbation =
                gt_increment.rotation().transpose() * odom_increment.rotation();
            const double rotation_perturbation_deg =
                rotationAngleRad(rotation_perturbation) * 180.0 / kPi;
            stats_.rotation_perturbation_squared_sum_deg2 +=
                rotation_perturbation_deg * rotation_perturbation_deg;
        }

        previous_odom_ = normalizedPose(previous_odom_ * odom_increment);
        previous_gt_ = normalized_gt;
        ++stats_.pose_count;
        ++stats_.increment_count;
        stats_.traveled_distance_m += distance_m;

        const Eigen::Isometry3d final_error = normalized_gt.inverse() * previous_odom_;
        stats_.final_translation_drift_m = final_error.translation().norm();
        stats_.final_rotation_drift_deg =
            rotationAngleRad(final_error.rotation()) * 180.0 / kPi;
        return previous_odom_;
    }

    const CorrelatedOdomStats& stats() const { return stats_; }

private:
    static Eigen::Isometry3d normalizedPose(const Eigen::Isometry3d& pose)
    {
        Eigen::Isometry3d normalized = pose;
        Eigen::Quaterniond quaternion(pose.rotation());
        if (!std::isfinite(quaternion.norm()) ||
            quaternion.norm() <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("pose has invalid rotation");
        }
        quaternion.normalize();
        normalized.linear() = quaternion.toRotationMatrix();
        return normalized;
    }

    double sampleNormal(double standard_deviation)
    {
        if (standard_deviation == 0.0) return 0.0;
        return standard_normal_(rng_) * standard_deviation;
    }

    CorrelatedOdomConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> standard_normal_{0.0, 1.0};
    bool initialized_ = false;
    Eigen::Isometry3d previous_gt_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d previous_odom_ = Eigen::Isometry3d::Identity();
    CorrelatedOdomStats stats_;
};

}  // namespace n3mapping::eval
