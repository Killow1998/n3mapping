# TODO: Loop Closure Z Drift Diagnostics

## Observed Issue

Some real-world loop closures can correct XY/yaw while leaving visible drift or discontinuity in Z. This affects map consistency and downstream navigation surfaces, but it is not solved by Phase 1 pbstream/nav filtering work.

## Why Z Drift Can Happen

- Loop candidates may be descriptor-correct but geometrically weak in vertical structure.
- ICP verification can accept a loop with insufficient vertical constraints.
- Loop information matrices may not reflect anisotropic confidence.
- Submaps can contain rear/body artifacts or sparse floor/ceiling observations that bias vertical alignment.

## Why This Is Out Of Scope For Phase 1

Phase 1 is limited to serialization consistency, dense trajectory export, lightweight nav reading, nav pbstream filtering, and safe validation. It must not tune loop parameters, port another loop closure stack, add Z multi-hypothesis ICP, or change the mapping/relocalization algorithm.

## Required Future Diagnostics

Before changing loop behavior, add loop diagnostics that capture:

- query and match keyframe ids
- descriptor scores and yaw hypotheses
- ICP initial guess and final transform
- translation residual split into x, y, z
- rotation residual split into roll, pitch, yaw
- fitness score and inlier ratio
- accepted/rejected gate reason
- information matrix used for the loop edge
- optimized pose delta before and after applying loop edges

`loop_debug.jsonl` is the preferred artifact format because it can be appended during long bag runs and reviewed after shutdown.

## Future Fix Directions

Possible future fixes include anisotropic loop information, stricter vertical residual diagnostics, better loop candidate debugging, or Z-aware ICP validation. These should be evaluated only after the diagnostics above identify the failure mode.
