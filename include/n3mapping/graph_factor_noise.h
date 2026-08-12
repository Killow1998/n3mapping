// Shared noise-model selection for reference and shadow graph factors.
#pragma once

#include <Eigen/Core>
#include <gtsam/linear/NoiseModel.h>

#include "n3mapping/config.h"

namespace n3mapping {

enum class GraphFactorNoiseRole {
    ODOMETRY,
    ROBUST_GLOBAL_OR_SESSION_ODOMETRY,
};

struct GraphFactorNoiseSelection {
    gtsam::noiseModel::Base::shared_ptr model;
    bool explicit_information = false;
    bool fallback = false;
    bool robust = false;
};

gtsam::noiseModel::Gaussian::shared_ptr makeExplicitFullGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information);

GraphFactorNoiseSelection makeFullGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information,
    const Config& config,
    GraphFactorNoiseRole role,
    bool use_robust);

GraphFactorNoiseSelection makeXYYawGraphFactorNoise(
    const Eigen::Matrix<double, 6, 6>& information,
    const Config& config,
    bool use_robust);

}  // namespace n3mapping
