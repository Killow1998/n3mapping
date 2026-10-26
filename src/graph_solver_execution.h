#pragma once

#include <tbb/task_arena.h>
#include <utility>

namespace n3mapping::internal {

// The installed GTSAM parallel traversal crashes on recorded large graphs.
// Confine each synchronous solve without throttling unrelated TBB work or
// serializing independent graph owners. This is not a lock for graph state.
template <class Function>
decltype(auto) runGraphSolve(Function&& solve) {
    tbb::task_arena arena(1);
    return arena.execute(std::forward<Function>(solve));
}

}  // namespace n3mapping::internal
