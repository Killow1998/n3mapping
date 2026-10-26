# Graph solver ownership and execution

The ROS adapter serializes mutations of its live `N3MappingCore`, including
mapping, loop-result commits and map save/load, with `data_mutex_`. A
`GraphOptimizer` is owned by one caller; it is not a concurrent mutable API.

`cloneCommitted()` copies estimates and graph metadata and creates a separate
ISAM2. Factor pointers can be shared because published factors are immutable.
An asynchronous loop job owns its graph snapshots and copies keyframe metadata;
point clouds remain shared read-only observations. The job's future is consumed
before commit/rebase/release. `AsyncLoopState::worker` is declared last so its
destructor joins before the snapshots are destroyed. The worker does not acquire
the live owner's mutex during shutdown.

All GTSAM solves enter `internal::runGraphSolve`: incremental ISAM2, reconstruction
of committed ISAM2, trial LM, geometry append LM, and shared-plane LM. Map restore
delegates to the same trial boundary. A synchronous `tbb::task_arena(1)` confines
the solve's nested tasks; return or exception completes that execution before
the arena is released. This is an execution boundary, **not a graph-state lock**.
Independent graph owners and unrelated TBB workloads remain concurrent. There
is no N3M process-wide `global_control` in these paths.

This boundary contains crashes observed with the installed TBB-enabled GTSAM
4.1.1. It does not repair GTSAM internals or establish the precise cause of those
crashes. Keep the boundary until an independently validated dependency revision
can safely run the recorded large graphs in parallel. An upgrade must compare
complete replay, saved/reloaded state, numerical results, latency, CPU and RAM;
removing the boundary based only on a small graph test is insufficient.

The [oneTBB arena specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneTBB/source/task_scheduler/task_arena/task_arena_cls.rst)
defines the local concurrency and synchronous `execute` contract.

Regression coverage is in `test_graph_solver_execution`: nested tasks and
exception recovery; independent owners alongside unrelated TBB work; and
concurrent real graph snapshots compared with serial numerical results.
Map persistence, asynchronous mapping and resuming retain their existing E2Es.

Incident and release evidence: [large mapping crash, 2026-10-25](evidence/large_mapping_crash_20261025.md).
