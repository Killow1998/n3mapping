// Based on small_gicp's ParallelReductionOMP.
// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>
#include <Eigen/Geometry>
#include <omp.h>
#include <small_gicp/registration/reduction_omp.hpp>

namespace n3mapping::registration_detail {

// Keep each worker's sums on its own stack. Updating adjacent entries of the
// shared error vector for every point otherwise makes workers contend for the
// same cache line. Factors, guided scheduling and final reduction order stay
// the same as ParallelReductionOMP; its error evaluation is reused unchanged.
struct RegistrationReduction : small_gicp::ParallelReductionOMP {
  template <typename Target, typename Source, typename Search,
            typename Rejector, typename Factor>
  auto linearize(const Target& target, const Source& source,
                 const Search& search, const Rejector& rejector,
                 const Eigen::Isometry3d& pose,
                 std::vector<Factor>& factors) const {
    struct Sum {
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
      double error = 0.0;
    };
    std::vector<Sum> sums(num_threads);
#pragma omp parallel num_threads(num_threads)
    {
      Sum local;
#pragma omp for schedule(guided, 8) nowait
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(factors.size()); ++i) {
        Eigen::Matrix<double, 6, 6> H;
        Eigen::Matrix<double, 6, 1> b;
        double error;
        if (factors[i].linearize(target, source, search, pose, i, rejector,
                                 &H, &b, &error)) {
          local.H += H;
          local.b += b;
          local.error += error;
        }
      }
      sums[omp_get_thread_num()] = local;
    }
    for (int i = 1; i < num_threads; ++i) {
      sums[0].H += sums[i].H;
      sums[0].b += sums[i].b;
      sums[0].error += sums[i].error;
    }
    return std::make_tuple(sums[0].H, sums[0].b, sums[0].error);
  }
};

}  // namespace n3mapping::registration_detail
