#pragma once

#include "graph_solver_dependency.h"
#include <tbb/task_arena.h>
#include <utility>

namespace n3mapping::internal {

int graphSolveConcurrency(int num_threads);

template <class Function>
decltype(auto) runGraphSolve(int num_threads, Function&& solve) {
    tbb::task_arena arena(graphSolveConcurrency(num_threads));
    return arena.execute(std::forward<Function>(solve));
}

}
