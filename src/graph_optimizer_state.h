#pragma once
#include "n3mapping/graph_optimizer.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace n3mapping {
struct GraphOptimizer::GtsamState {
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values initial_values_;
    gtsam::Values new_values_;
    gtsam::Values current_estimate_;
    gtsam::Values last_good_estimate_;
};
}  // namespace n3mapping
