#include "n3mapping/graph_factor_noise.h"

#include <cmath>

namespace n3mapping {
namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

Matrix6d informationToGtsamOrder(const Matrix6d& information) {
    Matrix6d reordered;
    reordered.block<3, 3>(0, 0) = information.block<3, 3>(3, 3);
    reordered.block<3, 3>(0, 3) = information.block<3, 3>(3, 0);
    reordered.block<3, 3>(3, 0) = information.block<3, 3>(0, 3);
    reordered.block<3, 3>(3, 3) = information.block<3, 3>(0, 0);
    return reordered;
}

gtsam::noiseModel::Base::shared_ptr wrapRobust(
    const gtsam::noiseModel::Base::shared_ptr& base,
    const Config& config,
    bool use_robust,
    bool* wrapped) {
    if (wrapped) *wrapped = false;
    if (!base || !use_robust || !config.use_robust_kernel) return base;

    gtsam::noiseModel::mEstimator::Base::shared_ptr estimator;
    if (config.robust_kernel_type == "Huber") {
        estimator = gtsam::noiseModel::mEstimator::Huber::Create(
            config.robust_kernel_delta);
    } else if (config.robust_kernel_type == "Cauchy") {
        estimator = gtsam::noiseModel::mEstimator::Cauchy::Create(
            config.robust_kernel_delta);
    } else if (config.robust_kernel_type == "DCS") {
        estimator = gtsam::noiseModel::mEstimator::DCS::Create(
            config.robust_kernel_delta);
    }
    if (!estimator) return base;
    if (wrapped) *wrapped = true;
    return gtsam::noiseModel::Robust::Create(estimator, base);
}

}  // namespace

gtsam::noiseModel::Gaussian::shared_ptr makeExplicitFullGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information) {
    if (information.isZero(1e-10)) return nullptr;
    try {
        return gtsam::noiseModel::Gaussian::Information(
            informationToGtsamOrder(information));
    } catch (...) {
        return nullptr;
    }
}

GraphFactorNoiseSelection makeFullGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information,
    const Config& config,
    GraphFactorNoiseRole role,
    bool use_robust) {
    GraphFactorNoiseSelection selection;
    auto gaussian = makeExplicitFullGraphFactorNoise(information);
    selection.explicit_information = static_cast<bool>(gaussian);
    if (!gaussian) {
        gtsam::Vector6 sigmas;
        if (role ==
            GraphFactorNoiseRole::ROBUST_GLOBAL_OR_SESSION_ODOMETRY) {
            sigmas << config.loop_noise_rotation,
                config.loop_noise_rotation,
                config.loop_noise_rotation,
                config.loop_noise_position,
                config.loop_noise_position,
                config.loop_noise_position;
        } else {
            sigmas << config.odom_noise_rotation,
                config.odom_noise_rotation,
                config.odom_noise_rotation,
                config.odom_noise_position,
                config.odom_noise_position,
                config.odom_noise_position;
        }
        gaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        selection.fallback = true;
    }
    selection.model = wrapRobust(
        gaussian, config, use_robust, &selection.robust);
    return selection;
}

GraphFactorNoiseSelection makeXYYawGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information,
    const Config& config,
    bool use_robust) {
    GraphFactorNoiseSelection selection;
    auto sigmaFromInfo = [](double value, double fallback,
                            bool* used_explicit) {
        if (std::isfinite(value) && value > 1e-12) {
            if (used_explicit) *used_explicit = true;
            return 1.0 / std::sqrt(value);
        }
        return fallback;
    };
    bool x_explicit = false;
    bool y_explicit = false;
    bool yaw_explicit = false;
    gtsam::Vector3 sigmas;
    sigmas << sigmaFromInfo(information(0, 0),
                           config.loop_noise_position, &x_explicit),
        sigmaFromInfo(information(1, 1),
                      config.loop_noise_position, &y_explicit),
        sigmaFromInfo(information(5, 5),
                      config.loop_noise_rotation, &yaw_explicit);
    selection.explicit_information =
        x_explicit && y_explicit && yaw_explicit;
    selection.fallback = !selection.explicit_information;
    selection.model = wrapRobust(
        gtsam::noiseModel::Diagonal::Sigmas(sigmas), config,
        use_robust, &selection.robust);
    return selection;
}

}  // namespace n3mapping
