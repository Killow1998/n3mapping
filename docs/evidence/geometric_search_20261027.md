# Geometric search and dependency controls — 2026-10-26/27

Baseline runtime: `ab8e92a0b308558ab8d958acb5c369e0e2e71ee1`, repository
head at the start: `5fa4fc3`. This follows the
[prepared-place qualification](place_retrieval_20261026.md).
Grid preparation runtime source: **`6b21624`**; bounded relocalization proposal
follow-up: **`68d8ad6`**; confirmation/worker repair: **`9c45973`**; latest
source-locality follow-up: **`2d1917d`** (qualifications below). The
[machine-readable qualification](geometric_search_20261027.json) records source,
artifact hashes and numerical comparisons. No remote push, tag or installed
runtime replacement occurred.
The objective remains reliable stationary startup, shorter latency and lower
total CPU. General difficult-search operation at half a core is not accepted.

Evidence root, relative to the ROS workspace:
`build/n3mapping/geometric_search_20261026/`. Native Release builds use ccache,
CPUs **0,2,4,6**, four algorithm workers, `OMP_WAIT_POLICY=PASSIVE` and one
OpenBLAS worker. Replays use the existing frozen frontend clouds and odometry,
at `rate=0`. Truth remains an observer; query trajectories are not fitted to it.
Wall time excludes map loading unless explicitly stated. Host clocks and other
load are not controlled, so sub-percent changes are not established speedups.

## Grid preparation candidate

`FreeSpaceGrid` retains its existing occupancy, ray construction and scoring.
Atlas compilation now optionally stores its two bit layers, origin, dimensions
and resolution. Its separate configuration signature covers resolution, ray
length and occupied-point threshold. The existing map SHA binds it to the map.
No geometry solver, candidate count, confirmation count or acceptance gate changes.

The decoder validates external payloads once, including dimensions, exact packed
lengths, finite metadata, mutually exclusive occupied/free cells and padding bits.
Missing caches, changed parameters, disabled free-space scoring or an external
PCD override use the existing runtime construction path. Malformed matching
caches reject the Atlas load. The existing Atlas size limit still applies.

Ownership matters: the first LIO input resets odometry-dependent search state.
The first prototype handed the grid over at map load; that reset immediately
discarded it, and original14 replay still rebuilt it. The corrected candidate leaves
the grid in the Atlas until first use and checks the keyframe revision captured
at load. It then transfers ownership without copying. Normal release after lock
and map-revision invalidation remain unchanged. A later recovery may rebuild
the grid after that release; permanent retention is not introduced.

The serializer is separate from the grid algorithm. Both ROS adapters continue
using the shared core. Asynchronous loop `Atlas::prepare` does not build this
new cache; the work belongs to the existing map-artifact compilation stage.
The compiler and Atlas load report preparation/decode timing. There is no new
per-frame logging, worker or synchronization mechanism.

The corrected candidate passes **67/67 CTest
entries in 42.59 s**, including ROS E2Es. Added/extended coverage checks exact
score preservation after serialization, parameter/override misses, malformed
payloads and map replacement after an odometry reset. The 03 smoke retains
frame-4 lock and 41 outer alignments with no runtime grid-build event.

The full `grid_cache02` comparison uses the same newly compiled Atlas for both
cores; the baseline ignores the added optional fields and builds the grid as
before. Canonically matched prepared points, normals and covariance values have
zero differences from the old Atlas. Global point order, descriptor cache and
place catalog also match. The map SHA and geometry configuration are unchanged.

| Original14 | Baseline | Precomputed grid |
| --- | ---: | ---: |
| Locks / published poses | 14 / 4,088 | 14 / 4,088 |
| Mean first-lock computation | 2.350 s | 1.750 s |
| First-lock CPU sum | 74.356 s | 65.960 s |
| Entire replay CPU sum, excluding load | 158.155 s | 151.609 s |
| Mean load | 4.859 s | 4.926 s |
| Mean load plus first lock | 7.209 s | 6.676 s |
| Peak RSS, same newly compiled Atlas | 1,300.89 MiB | 1,300.60 MiB |
| Maximum position / rotation error | 2.481 cm / 0.08384° | 2.481 cm / 0.08384° |

First-lock wall falls **25.5%**, its CPU **11.3%**, and total load-plus-lock wall
**7.4%**. All 14 decision sequences, selected identities and alignment counts
match; maximum published matrix-element difference is 3.56e-15. Selected wall
times are 03: **3.197→2.559 s**, 08: **2.457→1.812 s**, 14: **2.981→2.381 s**.
This does not meet the half-core active-search target: 03 still uses about
7.314 CPU seconds during 2.559 seconds of first-lock computation, or 2.86 cores.

| Complete route | Outputs / lock frames | Maximum position error | CPU old→new | Wall old→new |
| --- | --- | ---: | ---: | ---: |
| `reentry_20261019/query` | 2,232 / 4 | 7.933 cm | 52.776→52.527 s | 18.715→18.115 s |
| `independent_v2` | 3,662 / 4,1938 | 3.247 cm | 115.338→115.829 s | 41.481→41.274 s |

Both retain decision/source counts and the second route's existing 59 coast
outputs. Matrix-element differences are below 5.69e-14. The new route's single-run
CPU is **0.43% higher**; no whole-route CPU speedup is claimed. This optimization
principally removes first-use map preparation, not repeated recovery geometry.

Offline grid construction/packing costs **650.289 ms**; decoding costs about
**16 ms**. The sidecar grows 625,350,425→626,874,615 bytes, **1.454 MiB**.
Full standalone compilation and verify-load take 12.21 s and 1,606.3 MiB maximum
RSS in this run, including all existing target/descriptor preparation.
The same-Atlas comparison is not evidence of a RAM reduction. A later 03
old/new-core × old/new-Atlas control measured 1,299.8–1,302.1 MiB for all four
combinations, including the untouched old core/cache. Thus the earlier 1,277 MiB
runs do not establish a cache-induced 24 MiB increase. The cause of this run-level
variation is not proven. Control timing varied substantially and is not used to
claim another speedup. Its fallback logs confirm runtime construction with the
old cache and no construction with the new cache.

Results: `fixed14_grid_cache02/results.json`, `motion_*_grid_cache02/score.json`,
`comparison.json`, `grid_atlas_geometry.json`, `grid_controls_*_atlas/results.json`.
No fresh hardware/ROS bag was acquired in this stage, and Noetic was not rebuilt.
The core APIs and scoring behavior are shared, but that is not a Noetic runtime
qualification. Production build metadata remains unverified in this independent
test build; it is not a new deployed or tagged release.

## Where the geometry work goes

`profile.cpp` is an offline interposition probe, not production logging. It
records solver stage calls and visibility checks and writes once on process exit.
`profile03.json` covers the first five 03 frames with the unmodified baseline:

| Operation | Calls | Sum of call wall times |
| --- | ---: | ---: |
| Coarse plane ICP | 33 | 858.6 ms |
| Fine plane ICP | 38 | 837.4 ms |
| GICP refinement | 32 | 709.7 ms |
| Visibility evaluation | 113 | 959.3 ms |

These call times overlap across workers. Their sum is neither elapsed time nor
process CPU. There are 41 outer alignment calls and 103 actual solver stages.
The first frame has 24 outer calls; the next two frames each admit four calls
for newly retrieved regions. These are not repeated initialization of unchanged
seeds. There are no seed resumptions/retries in this five-frame trace.

Only one exactly repeated visibility input was found, about 8 ms. No identical
alignment initial poses were found. Treating merely close poses as identical
would change the computation. Descriptor work on the first frame is about
34 ms; a separate serial free-space build costs about 655 ms near first lock.
This supports moving invariant map preparation out of startup, while geometry
proposal/verification remains the larger unresolved search cost.

## Ablations not retained

- Nearest-neighbor certificates based on the first/second-neighbor gap or pruned
  subtree bounds did not pay for their maintenance. Matched `-O3` controls were
  about 3.345 s / 8.408 CPU s versus 3.45 s / 8.75 CPU s for a certificate-plus-hint
  prototype. An earlier `-O2` versus `-O3` comparison is excluded.
- Removing occupied-node bounds made search slower. Removing orthogonal-cell
  bookkeeping and skipping the forced root descent improved a component test,
  but the full `candidate01` control was not a useful global improvement:
  first-lock mean 2.356→2.335 s, whole-original14 CPU 158.135→158.060 s; route CPU
  52.644→53.139 and 115.762→116.153 s. Locks, decisions and accuracy were retained.
  That source change was restored. Its 67-test pass is not a performance acceptance.
- Direct CPU [3D-BBS](https://github.com/KOKIAOKI/3d_bbs/tree/41529a34a2fb9618b5ff560fb3c2363f1615666d)
  was evaluated on the actual 03 first cloud and 24 retrieved seeds: 0.5 m voxels,
  translation ±4/±4/±1 m, yaw ±45 degrees and a 1 s timeout per seed. At depth 3,
  21/24 queries timed out; aggregate query wall/CPU were 22.325/85.334 s. Depth 1
  still timed out 21/24. Pruning equal-best branches also timed out 21/24, taking
  22.010/83.538 s. Many timed-out searches had not reached a leaf: loose upper
  bounds, not just enumeration of tied final answers, dominate this fixture.
  This coarse occupancy objective and search domain are not equivalent to N3M's
  ICP acceptance, so a BBS timeout cannot safely reject an existing candidate.

No certificate, KD pruning or BBS prototype is in production source. Raw inputs,
prototype sources and failures remain under the evidence root for reproducibility.
The previous unqualified descriptor-fusion experiment stays in its existing
stash; it is not applied by this stage.

## Bounded sparse relocalization proposals — 2026-10-27

Source **`68d8ad6`**, baseline **`6b21624`**. Evidence is under
`build/n3mapping/proposal_screen_20261027/`; the retained variant is `candidate03`.
The same map and compiled Atlas, frozen frontend inputs, four CPUs/workers and
all acceptance settings are used. The shared matcher, mapping-loop preparation,
normal tracking and original test assertions are unchanged.

`RelocalizationObservation` resamples only the broad proposal source once per
search observation. With the current settings, proposal spacing is 0.4 m instead
of 0.2 m; fine and covariance refinement retain 0.1 m and 0.05 m. Extra sampling
is limited so the new cell diagonal does not exceed half the correspondence
radius. Already coarse inputs keep their density, as do samples that would leave
fewer than the solver's ten-point minimum. Diagnostic seed probes now reuse the
same observation preparation. There is no new public API, parameter, dependency,
worker, synchronization, log stream or candidate rejection gate.

This follows the separation of a cheaper pose proposal and subsequent full
refinement also used in [Cartographer's constraint builder](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/constraints/constraint_builder_3d.cc).
It does not import its branch-and-bound matcher. The 03 audit reduces proposal
points by 37–49%, while retaining the original fine point counts. Full geometry,
visibility and ambiguity checks still decide acceptance. Eleven 03 alignments
have a nonconverged broad stage followed by passing final geometry; their poses
are over 1.55 m from the selected pose in this fixture. This observation alone
is not a cross-scene justification for rejecting every nonconverged proposal.

An unbounded shared-matcher prototype failed the existing moving-keyframe
recovery test: its 1 m proposal became 2 m. A bounded shared version passed its
subsequent controls but affected mapping and normal tracking too. It is not
retained. The final change belongs to the existing relocalization module;
the matcher and a temporary near-zero residual assertion adjustment are restored.
Final CTest passes **67/67 in 41.26 s**, including ROS E2Es, with original assertions.
An earlier test invocation loaded an installed old core by mistake; its linkage
failure is retained separately, not counted as a product result.

| Original14, full replays | Baseline | Sparse proposal |
| --- | ---: | ---: |
| Locks / outputs | 14 / 4,088 | 14 / 4,088 |
| Mean first-lock computation, excluding map load | 1.749 s | 1.671 s |
| First-lock CPU sum | 66.138 s | 61.272 s |
| Entire replay CPU sum, excluding load | 152.562 s | 147.843 s |
| Mean map load | 4.918 s | 4.967 s |
| Mean load plus first lock | 6.667 s | 6.638 s |
| Peak RSS | 1,300.602 MiB | 1,300.598 MiB |
| Maximum position / rotation error | 2.481 cm / 0.083842° | 2.481 cm / 0.083842° |

First-lock CPU falls **7.36%**, computation wall **4.49%**, and whole-original14
CPU **3.09%**. All positions still lock at frame 4. The largest output translation
change is **0.0061 mm**. RAM and load-inclusive startup have no established
improvement. Selected computation wall times are 03: **2.559→2.405 s**,
08: **1.824→1.583 s**, and 14: **2.346→2.096 s**.

This is not a uniform per-position speedup. Initial CPU rises 4.3% at 01 and
12.1% at 07; a separate five-frame repeat gives 1.6% and 16.6%, respectively.
07 still initializes the same 28 seeds. At final confirmation it performs
**12 instead of 9** solves, including three extra broad fallbacks; no seed is
reinitialized in those confirmation frames. A separate audit identifies the
triggers: resumed corrections of 0.594 m and 0.583 m exceed the existing 0.3 m
limit (the second also fails convergence); a third resumed result fails the
geometric quality gate. These are individual fallbacks, not a whole-group
restart. Keep them until their trigger can be addressed without excluding a
valid competing pose. This is a
remaining cost, not an accuracy or lock-time-in-frames regression.

| Complete route | Outputs / lock frames | Maximum position error | CPU baseline→candidate | Wall baseline→candidate |
| --- | --- | ---: | ---: | ---: |
| `reentry_20261019/query` | 2,232 / 4 | 7.933 cm | 52.409→52.329 s | 18.155→17.990 s |
| `independent_v2` | 3,662 / 4,1938 | 3.247 cm | 115.759→113.771 s | 40.911→40.511 s |

Both retain output counts, pose sources and lock/recovery frames, with no output
above 10 cm. Output matrix-element differences are below 4.20e-11. The new route
retains 59 coast outputs; ten pending-search decision labels change and total
alignments rise 410→419. Its measured CPU falls 1.72%; the old route's 0.15%
change is not an established speedup. Do not describe all internal decisions
or alignment counts as identical.

03 still spends 6.799 CPU seconds in 2.405 seconds of first-lock computation,
about **2.83 active cores**. Whole-route CPU divided by sensor duration is a
different measure and cannot establish difficult-search half-core operation.
The four-core ceiling is unchanged. The frozen navigation/rc3/FAST-LIO/GTSAM
libraries and rc1/rc2/rc3 tags are unchanged. No dependency update, deployment,
push or new release tag occurred; there is no fresh hardware qualification.
The independent binary still carries unverified production build metadata.

Evidence: `ctest03.log`, `fixed14_candidate03/results.json`,
`motion_*_candidate03/score.json`, `comparison.json`, `audit_summary.json`,
`slow_case_control/results.json`, `audit07_confirmation.json`,
`qualification.json`, and `protected_check.json`.
Core SHA-256: `365ccdba00a57d5c69147ecd1ceca51186839aad2b45c24f076b19c106158514`.

## Confirmation proposals and shared worker budget — 2026-10-27

Runtime source **`9c45973`**, compared with `68d8ad6`. Evidence lives in
`build/n3mapping/confirmation_cost_20261027/`. The same map, Atlas, observations,
acceptance gates and four-CPU affinity are retained. The primary comparison uses
four workers; an additional two-worker control uses the same four allowed CPUs.

### Trigger and retained change

The previous sparse proposal altered three competing initial solutions at 07.
During confirmation, two resumed fits required about 0.59 m of correction or
failed convergence; another failed the existing quality check. These correctly
triggered broad retries. Removing those checks would hide the cost and weaken
reliability.

The second voxel sampling now selects the **existing coarse point nearest each
voxel centre**, using coordinate order to break ties. It does not create another
centroid between surfaces. These are already prepared coarse points, not original
sensor returns. Source voxel spacing remains bounded by the existing correspondence
radius; fine and GICP layers retain their original input and resolution. The
sampler groups complete cells rather than emitting partial means when a cell
crosses a parallel sampling block. The upstream sampler's 1,024-point blocks and
possible extra partial-cell points were checked in its installed implementation.
A new property check covers input-order invariance, negative cells, equal-distance
selection and retaining a point from the input set.

The serial-centroid control removes two of 07's extra retries; the representative
control removes all three on the repeated fixture. This localizes the regression
to coarse proposal construction. It does not establish that one particular
cross-surface centroid explains every restart in other scenes. The mature
[PCL UniformSampling](https://pointclouds.org/documentation/classpcl_1_1_uniform_sampling.html)
pattern was considered; this private helper works directly on the existing double
coordinates and avoids an extra PCL cloud conversion or public abstraction.

Atlas basin tasks and yaw tasks now share **one OpenMP team** sized by the existing
`num_threads`. Nested point-level parallel regions do not create further workers.
Targets are prepared once per candidate; a task group finishes before its captures
leave scope. Each yaw writes its own result and cost slot; original-order merging
preserves tie-breaking. Existing seed locks and lazy-feature publication remain.
The sequential path remains available without Atlas/multiple basins. No new
configuration, thread-pool service, process-wide serial limit or acceptance gate
is added. Per-task `align_ms` values are summed task wall time, not elapsed search
latency or process CPU; the probe measures those independently.

### Ablations and full comparison

| Control | Finding |
| --- | --- |
| Exact serial centroid at unchanged sparse spacing | 07 confirmation calls 12→10; useful partial repair. |
| Denser 0.3 m centroid | 07 improves, but 14 adds a solve and CPU; not retained. |
| Representative point at unchanged 0.4 m spacing | 07 confirmation calls 12→9, preserving all gates; retained. |
| Capped basin-only/candidate-only asynchronous jobs | Uneven work leaves a long tail; 08 slows; not retained. |
| One shared team with basin and yaw tasks | Bounds worker multiplication and avoids the long basin tail; retained. CPU reduction from scheduling alone is not established. |
| One-pass fixed-pose factor accumulation | No stable complete-case benefit; reverted, with the shared geometric evaluator unchanged. |

Three alternating-order 07 A/B pairs consistently use **40→37 total solves**:
28 initial solves in both, **12→9** in confirmation. Median first-lock CPU is
**4.1649→3.5718 s (−14.2%)**; median computation time is
**1.4161→1.2709 s (−10.3%)**. The complete14 run measured 4.3179→3.7461 CPU s
and 1.6194→1.2968 wall s for 07. Use the repeated control for the speed claim;
host-load variation affects the one-shot wall measurements.

| Four-worker original14 | Baseline | Candidate |
| --- | ---: | ---: |
| Locks / outputs | 14 / 4,088 | 14 / 4,088 |
| Maximum position error | 2.481 cm | 2.481 cm |
| Mean first-lock computation | 1.8030 s | 1.7252 s |
| First-lock CPU sum | 64.0505 s | 63.1533 s |
| Entire replay CPU sum | 155.0185 s | 154.5422 s |
| Mean map load | 5.1031 s | 5.1190 s |
| Peak RSS | 1,300.60 MiB | 1,300.59 MiB |

All acceptance, output and lock decisions are retained; maximum static output
matrix-element difference is 4.64e-6. Total solve count falls 519→517. 04 and 12
each add one solve, while 13 removes one. Thus this is not a universal reduction in fallback
work. The entire-replay CPU change is only 0.3%; do not claim a general speedup
from it. Original14 first-lock CPU decreases 1.4% in this complete run.

| Full route | Outputs / lock frames | Maximum position error | CPU baseline→candidate | Wall baseline→candidate |
| --- | --- | ---: | ---: | ---: |
| `reentry_20261019/query` | 2,232 / 4 | 7.933 cm | 56.694→52.713 s | 22.682→18.228 s |
| `independent_v2` | 3,662 / 4,1938 | 3.247 cm | 114.125→113.345 s | 40.822→41.199 s |

Both routes retain acceptance, pose sources, output count and recovery, including
59 coast outputs in the new route. No published output exceeds 10 cm on these
fixtures. Maximum output matrix-element difference is 5.69e-14. New-route solve
count changes 418→419; seven per-frame counts differ. Old-route geometry/counts
are unchanged, so its observed timing drop is not an established algorithmic gain.

### Peak resource control

The 03 process was sampled every 50 ms using its process CPU clock. This measures
the core replay process, including its logger, not the entire ROS/robot stack.
Sub-window accounting jitter can produce a value slightly above the affinity
ceiling; report approximately four/two cores, not a literal 4.054-core maximum.

| Configured workers | Version | Maximum process threads | Observed CPU peak | First-lock CPU / wall |
| --- | --- | ---: | ---: | ---: |
| 4 | baseline | 17 | about 4 cores | 6.785 / 2.370 s |
| 4 | candidate | 5 | about 4 cores | 6.728 / 2.356 s |
| 2 | baseline | 9 | about 3.8 cores | 6.646 / 3.220 s |
| 2 | candidate | 3 | about 2 cores | 6.546 / 3.741 s |

Worker limits now control the search team. They do not remove the required
geometry or provide a hard CPU quota for other node threads. The default remains
four workers. The additional two-worker run passes **all original14 and both
complete routes**, preserving all compared decisions, counts and recovery;
original14 maximum error stays 2.481 cm and route maxima stay 7.933/3.247 cm.
Its original14 mean first-lock computation is 2.388 s, with 1,300.75 MiB peak RSS.
Lowering the worker budget trades latency for CPU headroom. In the isolated 03
control, active search still averages about **2.86 cores with four workers** and
**1.75 cores with two workers**. Difficult-search half-core acceptance remains open.

### Qualification and remaining work

The native Release suite passes **67/67 in 43.32 s**, including ROS E2Es, using an
isolated logger directory. The first test invocation had six logging-read failures
because another running instance held the default log directory's instance lock.
No assertion or product code was changed to make the suite pass, and that other
instance was left running. The first resource sampler attempt used a nonexistent
Python clock API; the corrected sampler uses libc's process-clock lookup. Its
failed attempt is not included in timing results.

The final comment-only rebuild is byte-identical to the tested core:
`5fa09cc05376fc024e7eadcb40541b08aaf24dffcda32ca29195b2c8f15cd02d`.
Frozen navigation, rc3, FAST-LIO, GTSAM, installed small_gicp and release tags are
unchanged. The independent core retains **UNVERIFIED** production metadata;
there is no new deployed release or target-hardware qualification. No remote
push occurred. Raw evidence: `comparison.json`, `repeat07_*/results.json`,
`resource_comparison.json`, `two_thread_comparison.json`, `ctest_isolated.log`,
`final_source_build_check.json`, `qualification.json` and `protected_check.json`.

Next, reduce necessary fine registration/verification work while preserving
competing poses and fallback gates. Map loading still costs about five seconds;
RAM remains about 1.27 GiB. These are separate remaining costs. Removing worker
oversubscription prevents one source of peak contention but does not establish
general half-core operation or a hard real-time bound on low-power hardware.

## Source locality for full geometry — 2026-10-27

Runtime source **`2d1917d`**, compared with `9c45973`. Evidence root:
`build/n3mapping/fine_geometry_20261027/`. The four-CPU affinity, four workers,
map, Atlas and input observations match the preceding qualification.

### What changed and why

Prepared relocalization source points are now packed in KD-tree leaf order once
per observation. Every candidate and confirmation then uses that same spatial
ordering. The existing `CompactKdTreeBuilder` performs a permutation: each layer
keeps its complete **already prepared** point set, and the GICP covariance moves
with its point. Covariances are computed through the unchanged preparation path
before this permutation. The builder now supports a covariance-only cloud without
requiring a normals array; the added test covers tuple preservation, duplicate
XYZ coordinates and that absent channel.

The change is limited to `RelocalizationObservation`. Public source preparation,
tracking and loop/reciprocal inputs keep their previous ordering. The short-lived
query trees are released after packing. There is no persistent map index, new
configuration, point thinning, solver change or acceptance-gate change.
Source preparation timing includes the added ordering work. Reordering can change
floating-point summation order; the complete replay comparison checks its observed
numerical effect. Better spatial locality is the reason for this experiment;
hardware cache-miss counters were unavailable, so no measured cache-miss rate is
claimed.

A temporary, buffered offline probe counted **1,604 linearizations and 1,604 LM
trial-error evaluations** on 03. Their summed call wall times were 3,676.99 and
197.34 ms; 40 fixed-pose linearizations totalled 154.84 ms. These overlapping call
sums are neither elapsed time nor process CPU. This fixture has one trial per
linearization, not repeated damping retries. The profiling code was removed from
production source. Kernel `perf_event_paranoid=4` blocked perf sampling; no system
permission was changed.

### Controls rejected or retained

| Control | Result |
| --- | --- |
| GICP 3D arithmetic and explicit shared `JᵀM` | 03 CPU 7.249→7.265 s; no stable complete-case gain. General-covariance oracle passes, while existing serial bit-exact optimizer assertions differ in rounding. Reverted without relaxing those assertions. |
| Precomputed KD partition entry from a previous match | Exact-neighbor tests pass, but 03 CPU 7.149→7.474 s. Added per-point routing storage did not pay for its query work; reverted. |
| Previous source point as a fallback search bound | 03 CPU 7.233→7.206 s; other cases mixed. No established benefit; reverted. |
| Sparse fine-target warmup for unfinished coarse seeds | 03 does not improve; 07 calls rise 37→39 and 08 41→43. The optional source field and extra solver stage were removed. |
| Only GICP source ordering | 03 CPU 7.601 s versus 6.894 s with all source layers ordered. Plane ordering contributes materially. |
| Only plane source ordering | 03 CPU 7.102 s versus 6.938 s with all layers; 07/14 also favour all layers slightly, 08 is effectively tied. Retain all-layer ordering for relocalization only. |

The first packed-source smoke improved 03 CPU 7.789→7.176 s. The complete run and
repeated controls below give the qualification; host frequency and unrelated load
are not controlled. Earlier first/second-neighbor and pruned-subtree certificate
failures were reviewed rather than repeated unchanged.

### Complete four-core comparison

| Original14 | Baseline | Candidate |
| --- | ---: | ---: |
| Locks / outputs | 14 / 4,088 | 14 / 4,088 |
| Maximum position error | 2.481 cm | 2.481 cm |
| Mean first-lock computation | 1.6923 s | 1.6307 s |
| First-lock CPU sum | 62.6747 s | 59.5303 s |
| Entire replay CPU sum | 153.5581 s | 149.4206 s |
| Mean map load | 5.2378 s | 5.2280 s |
| Peak RSS | 1,300.55 MiB | 1,300.59 MiB |

Original14 first-lock CPU falls **5.0%**. All compared decisions, outputs, support,
matched candidates and outer alignment counts remain equal. Maximum published
matrix-element difference is 7.11e-15. In the complete 03 run, first-lock CPU is
7.1692→6.9246 s and computation time 2.6048→2.4177 s, with 41 outer solves in both.
The 07 confirmation count stays at nine; no previous retry regression returns.

| Complete route | Outputs / locks | Maximum position error | CPU baseline→candidate | Wall baseline→candidate |
| --- | --- | ---: | ---: | ---: |
| `reentry_20261019/query` | 2,232 / 4 | 7.933 cm | 57.132→56.769 s | 21.507→21.514 s |
| `independent_v2` | 3,662 / 4,1938 | 3.247 cm | 121.483→119.507 s | 46.530→45.890 s |

Both retain all compared decisions, recovery and solve counts; the new route
retains 59 coast outputs. No published output exceeds 10 cm. Published
matrix-element differences are at most 7.11e-14. All 19,964 published poses across
both variants' original14/routes were checked finite. Routine tracking is outside
this source-ordering change, so small whole-route timing differences do not
establish a tracking speedup.

Three alternating-order 03 pairs confirm a median first-lock CPU reduction
**7.3675→6.9446 s (−5.7%)** and computation reduction
**2.6122→2.4967 s (−4.4%)**. These exclude map load. At 50 ms sampling, process
thread count remains five and peak CPU remains approximately four cores. The
median sampled time above 3.5 cores decreases approximately 0.905→0.806 s. This
coarse sample is not a hard real-time bound; short-window clock accounting can
slightly exceed four. Active-search average remains about **2.78 cores**, so
**half-core difficult search is not accepted**. Map load is still about 5.2 s,
and RAM remains about 1.27 GiB.

The final native Release suite passes **67/67 in 44.58 s**, including ROS E2Es,
with an isolated logger directory and the matching independent core/dependencies.
Core SHA-256:
`295e1e6c8d87c4eb210316948fab9cb4ccb6e800aaea1fa6e490426273b98550`.
Build metadata remains **UNVERIFIED**. Frozen navigation, rc3, FAST-LIO, GTSAM,
installed small_gicp and release tags are unchanged. No push or deployment occurred;
there is no new target-hardware or separate Noetic runtime qualification.

Raw evidence: `kernel03.csv`, `smoke_*/results.json`, `fixed14_final/results.json`,
`motion_*/score.json`, `comparison.json`, `resource_comparison.json`,
`ctest.log`, `qualification.json` and `protected_check.json`. Rejected source
variants are retained beside their independent binaries for reproducibility.

## Latest small_gicp control

The installed source is `78f2e7a221720625eb95271ad9da21a04fb77f86`.
Upstream main was `fa0cfc983417c044b6036f71dfa448038bb9c820`, dated 2026-09-29,
eight commits ahead when checked. See the
[upstream comparison](https://github.com/koide3/small_gicp/compare/78f2e7a221720625eb95271ad9da21a04fb77f86...fa0cfc983417c044b6036f71dfa448038bb9c820).
Changes include colored ICP, rotation normalization at solver entry, a VGICP
correspondence-distance repair and packaging/documentation work. The advertised
speed comparison against fast_gicp is not a comparison with our existing small_gicp.
The supplied Zhihu article returned HTTP 403 and was not used as evidence.

The candidate dependency and N3M were fully rebuilt together in isolated paths;
the test did not swap an incompatible library under old headers. Upstream tests
passed **93/93**; N3M passed **67/67**. Upstream enables PCL/TBB when building its
tests; the runtime helper was then rebuilt with those options OFF, matching the
installed dependency. All replay probes loaded their matching core/helper and
the already repaired GTSAM explicitly.

Earlier harness attempts used invalid ROS domain 234 and, separately, an
unintended old GTSAM runtime. They are retained as failed runs, not product
regressions; final suites use domain 228 and the explicit repaired dependency.

| Original14 | Existing small_gicp | Latest main |
| --- | ---: | ---: |
| Static locks / outputs | 14 / 4,088 | 14 / 4,088 |
| Mean first-lock wall | 2.35349 s | 2.35679 s |
| First-lock CPU sum | 74.3712 s | 74.6412 s |
| Entire replay CPU sum | 158.2492 s | 158.8499 s |
| Peak RSS | 1,277.26 MiB | 1,277.16 MiB |
| Maximum position error | 2.481 cm | 2.481 cm |

Old-route CPU was 52.491→52.652 s; new-route CPU was 115.185→116.159 s. Both
retain all outputs, decisions and lock frames, including recovery at frame 1938
and the existing 59 coast outputs in the new route. Maximum position errors
remain 7.933/3.247 cm. Published matrix-element differences are at most
7.11e-14. These runs demonstrate compatibility on the tested inputs, not an
observable speed or accuracy gain. Added color buffers are not allocated on
N3M's main coordinate-only preparation path; no material RSS growth was measured.
The installed dependency is unchanged. Rotation normalization remains a useful
upstream reliability change, but these inputs do not demonstrate a benefit from it.

## Review of proposed external optimizations

| Suggestion | Finding for this code and workload |
| --- | --- |
| Replace maximum clique with k-core or triplets | N3M's current startup path has no maximum-clique solver. A new robust proposal generator can be compared, but this is not removal of an existing bottleneck. |
| ROBIN costs 1–2 ms and rejects over 95% of outliers | Not established on these data. k-core decomposition is linear in vertices plus edges; building a dense compatibility graph still costs quadratic work. High degree alone does not certify the correct rigid match. |
| TurboReg is reliably 5–10 ms on our four cores | Its [README](https://github.com/Laka-3DV/TurboReg/blob/main/README.md) reports GPU headline FPS and consumes precomputed correspondences; CPU instructions are marked TBD. Feature extraction, matching and verification must be included before any N3M speed claim. |
| Pair/triplet hashing makes all matching linear | Hash lookup can be cheap, but generating all pairs/triplets and visiting large repeated-geometry buckets is not linear. It is a candidate proposal mechanism, not proof of uniqueness. |
| Replace repeatedly rebuilt KD trees with iVox | The Atlas target tree is already prepared once and shared across these seeds. Replacing its exact search changes the correspondence model; bounded-neighborhood hash search needs a recall and end-to-end comparison. |
| Incremental plane statistics and closed-form PCA | Plane moments already accumulate count/sum/second moment. Hot registration features already call Eigen `computeDirect`; no new 5–8× gain follows from requesting it again. Welford may improve numerical stability of moment accumulation, but is not itself a speed proof. |
| SmartFactor removes fill-in | Eliminating a shared landmark still couples its observing poses through a Schur complement. A different factor representation does not erase that coupling; cutting observations can discard useful constraints. N3M's mapping graph does have shared-plane landmarks, so this warrants a separate mapping profile, not a startup-speed claim. |
| Replace PCL GICP, add asynchronous loop processing | small_gicp, per-worker scratch and asynchronous snapshot/commit already exist. One global map-to-odom transform cannot represent all nonuniform trajectory/map corrections. |
| Enable `-march=native -ffast-math` for 30–50% | Release already uses `-O3`. Native tuning is not a portable deployment guarantee. GCC fast-math permits assuming away NaN/Inf and changing floating-point rules used by the existing validity gates; it is not enabled. |

Primary implementation references:
[KISS-Matcher pipeline](https://github.com/MIT-SPARK/KISS-Matcher/blob/main/cpp/kiss_matcher/core/kiss_matcher/KISSMatcher.cpp),
[ROBIN](https://github.com/MIT-SPARK/ROBIN),
[Eigen direct solver](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1SelfAdjointEigenSolver.html),
[GCC optimization rules](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).
The [earlier local survey](framework_survey_20261017/README.md) measured KISS
FasterPFH extraction alone at 26.60 CPU ms on 1,694 points; it did not qualify a
complete KISS/ROBIN pose pipeline.

The bounded proposal follow-up above establishes a modest aggregate improvement.
The confirmation follow-up resolves the three additional 07 retries on the
repeated fixture. Source locality further reduces complete original14 first-lock
CPU by 5.0%. Remaining work is fine registration/verification cost and the
roughly five-second map load. Difficult-search half-core acceptance remains open.
Any early rejection must retain a conservative justification and the competing
modes needed for reliable stationary startup. Microbenchmarks and published
timing ratios cannot substitute for the full comparison.
