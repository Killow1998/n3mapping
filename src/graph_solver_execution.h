#pragma once

#include "graph_solver_dependency.h"
#include <tbb/task_arena.h>
#include <utility>

namespace n3mapping::internal {

// Checks the loaded dependency once; no file I/O in the solve path.
int graphSolveConcurrency(int num_threads);

// Unpatched GTSAM reallocates childFactors while child tasks write into it.
// Keep that dependency serial; fixed builds use the configured local budget.
// This is not a lock for graph state or a process-wide concurrency control.
template <class Function>
decltype(auto) runGraphSolve(int num_threads, Function&& solve) {
    tbb::task_arena arena(graphSolveConcurrency(num_threads));
    return arena.execute(std::forward<Function>(solve));
}

}  // namespace n3mapping::internal
