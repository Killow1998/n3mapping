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
of committed ISAM2, trial LM, geometry append LM, shared-plane LM, and the
optional shadow-submap diagnostic LM. Map restore delegates to the same trial
boundary. A synchronous `tbb::task_arena` confines
the solve's nested tasks; return or exception completes that execution before
the arena is released. This is an execution boundary, **not a graph-state lock**.
Independent graph owners and unrelated TBB workloads remain concurrent. There
is no N3M process-wide `global_control` in these paths.

Unpatched GTSAM 4.1.1 uses concurrency 1. Its elimination traversal can grow a
parent's `childFactors` while a child publishes into that vector. A probe with
instrumented oneTBB reproduces this race; the GTSAM-free control passes.
[Upstream PR #1158](https://github.com/borglab/gtsam/pull/1158) protects both
accesses with the parent's mutex. The same instrumented probe passes with that
patch. This establishes a dependency defect, but does not uniquely attribute
the original `init_bucket` SIGSEGV to it.

The build probes the selected GTSAM headers for this locking implementation and
records the selected library's GNU build ID. The first solve checks the loaded
library's build ID directly from its mapped ELF notes. Only matching, patched
builds use the existing `num_threads` budget, clamped to at least one. Missing
identity, mixed libraries or old headers retain local serial containment. The
check does no file I/O and runs once; `graph_solver_dependency` records its
decision. An old runtime library cannot silently inherit the permission to run
in parallel from newer build headers.

The probe identifies this implementation; it is not blanket qualification of
every future GTSAM revision. Deploy the tested core and dependency together.
Each dependency revision still requires complete replay, saved/reloaded state,
numerical results, latency, CPU and RAM comparisons. No installed or frozen
navigation dependency is replaced by the N3M build.

The [oneTBB arena specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneTBB/source/task_scheduler/task_arena/task_arena_cls.rst)
defines the local concurrency and synchronous `execute` contract.

Regression coverage is in `test_graph_solver_execution`: nested tasks and
exception recovery; independent owners alongside unrelated TBB work; and
concurrent real graph snapshots compared with serial numerical results.
Map persistence, asynchronous mapping and resuming retain their existing E2Es.

Save timing uses the existing bounded asynchronous logger. `map_save_drain`
measures waiting for pending loop work; `map_save_serialize`, `map_save_atlas`
and `map_save_pcd` separate artifact work. `map_save` retains the total core-save
duration. The drained job's worker time includes work before the save request,
so it must not be added to the drain duration. Save transaction and loop
admission behavior are unchanged.

Incident and release evidence: [large mapping crash, 2026-10-25](evidence/large_mapping_crash_20261025.md).
