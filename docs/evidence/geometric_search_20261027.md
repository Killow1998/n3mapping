# Geometric search and dependency controls — 2026-10-26/27

Baseline runtime: `ab8e92a0b308558ab8d958acb5c369e0e2e71ee1`, repository
head at the start: `5fa4fc3`. This follows the
[prepared-place qualification](place_retrieval_20261026.md).
Grid preparation runtime source: **`6b21624`**; bounded relocalization proposal
follow-up: **`68d8ad6`** (qualification below). The
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
Next, explain and reduce 07's additional confirmation fallbacks, and profile the
remaining fine registration, visibility and roughly five-second map-load cost.
Any early rejection must retain a conservative justification and the competing
modes needed for reliable stationary startup. Microbenchmarks and published
timing ratios cannot substitute for the full comparison.
