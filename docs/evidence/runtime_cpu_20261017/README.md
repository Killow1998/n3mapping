# Runtime CPU and relocalization measurements — 2026-10-17

Initial-stage baseline: `45f4a171fcab1d9742ab7d575f41a632c7d75073`, branch
`dev/n3mapping-realtime-pipeline`. In that stage the Product V1 configuration remained unchanged
(SHA-256 `1b603a0a180330857aac150e0450cb5e94e284a84934e9b6ee387000f3f254d4`).
These are Release/Humble development builds with **UNVERIFIED** build identity;
they do not constitute a Product Bundle or hardware qualification.

## Measurement boundaries

CPU is process CPU time divided by elapsed time: 1.0 means one fully occupied
logical CPU. All threads are included. Map loading precedes the measurement;
initial relocalization is included. First-path latency starts with the first
input, after map loading. RSS is the process high-water mark, not live retained
memory. The adopted runtime optimizations use no CPU quota, input reduction,
frequency reduction, or acceptance changes. The isolated reseeding experiments
at the end of this record do change candidate selection and are not adopted.
OpenMP has four configured threads and the existing launch default
`OMP_WAIT_POLICY=PASSIVE`; an isolated comparison also tests the unset policy.

The ROS probe puts the production node, replay publisher, and output subscribers
in one process. DDS subscriptions, exact message synchronization, the normal
backend loop, both live callback workers, cloud conversion, TF/output publication,
and complete path publication all execute. CPU includes the replay and subscriber
helpers as well. The initial runs used same-process Fast DDS with restricted
UDP/interface access; the latest stage uses authorized host execution.
**Cross-process DDS, physical MID360 input,
RK3588, and RViz rendering were not measured.** The independent LIO recordings
have no ground truth and do not independently establish their sensor model.

The older direct-callback/headless measurements are retained in the evidence but
are not used as the full ROS result. That driver can admit fewer startup frames;
the actual DDS runs processed every input pair.

## Data and results

`floor7_710toMeeting.bag` is an independent historical LIO scan/odometry recording:
1,848 paired scans, 184.8 seconds, approximately 97.6 m of odometry travel, with a
50 × 59 m map containing 152,116 global points. Four additional independent room
recordings contribute 1,796 frames. Their poses, states, identifiers and decisions
are compared with the frozen baseline; this is a behavioral regression check,
not a new absolute-accuracy claim.

Xian's `map_091426_20261010_212025.zip` contains a 66.9 × 95.0 m map with 266,304
global points and 250 keyframes. No matching independent recording was found.
The cost probe preserves saved scans and reconstructs 1,800 views along 180 seconds
of the saved trajectory, beginning at offset 360 seconds (about 118.7 m of motion).
These same-map views do **not** validate accuracy, visibility realism or recall.
Original files were read without modification; raw maps/scans are not published.

| Same-process ROS, 10 Hz | Baseline | Visibility/sorting/path candidate |
| --- | ---: | ---: |
| Independent floor7: average CPU cores | 0.271 | 0.267 |
| Independent floor7: first path | 1.192 s | 0.934 s |
| Independent floor7: peak RSS | 470.1 MiB | 473.3 MiB |
| Saved-map moving workload: average CPU cores | 0.259 | 0.256 |
| Saved-map moving workload: first path | 1.380 s | 1.076 s |
| Saved-map moving workload: peak RSS | 620.7 MiB | 631.1 MiB |

All 1,848 and 1,800 pairs were processed respectively; all compared backend
behavior fields agree. RSS did not improve consistently. Average CPU below half
a core is not a hard ceiling on short bursts, especially during relocalization.

The Product launcher previously omitted the wait-policy default already applied
by the ROS launch files. With the **same candidate algorithm**, a 600-frame,
59.93-second ROS run used 0.674 cores when the policy was unset and 0.340 cores
with `PASSIVE`. After the first ten seconds, these were 0.614 and 0.294 cores.
Both processed 600 frames with identical decisions and received output counts;
first-path latency was 0.946 and 0.925 seconds. The launcher now supplies the
existing default before exec and preserves explicit user overrides, including
`GOMP_SPINCOUNT`. This removes waiting CPU work; it is not a CPU limit.

## Changes and ablations

- Visibility bin lookup uses conservative angular intervals. A lookup result is
  used only when the whole interval lies in one original output bin; boundary
  cases fall back to the original `atan2`/`hypot` expression. Range, occlusion,
  point order and evidence calculations remain unchanged. No approximate bin is
  accepted. Random/extreme coordinates and every tested bin boundary are checked.
- Residual percentiles share one sort, and the query median no longer copies its
  input. This removes work without changing the interpolation formula.
- The Humble node retains the complete native path message, avoiding a history
  copy before each publication. Three alternating runs at 180,000 poses (five
  hours at 10 Hz) reduced CPU per publication from a median 9.22 to 7.41 ms.
  Message contents, timestamps, cadence and historical snapshots are preserved.
- The path timer previously accumulated into an initial NaN. Assignment now
  records the actual duration. This diagnostic does not affect localization.
- Fixed-size feature-neighbor buffers improved a microbenchmark by about 7%, but
  full replay CPU increased about 0.5–0.7%; the candidate was removed.
- Additional spatial bounds in feature search were slower and were rejected.
- Worker-local registration reduction saved about 1.4% on a prefix replay. It
  was not adopted: the limited gain did not justify copying upstream reduction
  machinery into production.

With four P-cores pinned only for controlled comparisons, visibility optimization
reduced first-lock processing by about 26% on both main datasets. The additional
room runs preserve the original lock frame (4, 4, 9 and 54); the hardest run fell
from 18.635 to 14.233 seconds of unpaced processing to lock. This is not a claim
that the physical evidence arrives sooner. Production CPU affinity is unchanged.

The full CTest suite passed 61/61. The updated launcher separately passed its six
end-to-end cases, including actual child environment inheritance. Frozen
six-scenario localization episodes preserve all 52 non-timing rows exactly in
legacy, atlas, and atlas-plus-kill configurations. The four-thread episode variant
also passes. Detailed samples, hashes, exclusions and scope are in [results.json](results.json).

## Atlas memory and difficult-recording follow-up

The follow-up starts from published commit
`bcba7f2167c49ba0135c1c872794c77473f4a4e2`. Atlas loading now releases each
serialized array after its values have been validated and copied into the final
cloud, before building its KD-tree or decoding the next level. Map files,
prepared geometry, feature values, runtime configuration and acceptance rules
remain unchanged. The existing round-trip test compares every stored point,
normal and covariance exactly.

| Peak process RSS | Before | After |
| --- | ---: | ---: |
| Floor7 loading plus 30 frames, median of three alternating runs | 287.5 MiB | 268.5 MiB |
| Saved-map loading plus 30 frames, median of three alternating runs | 493.0 MiB | 429.2 MiB |
| Full floor7 ROS replay, 1,848 pairs | 473.5 MiB | 464.9 MiB |
| Full saved-map ROS workload, 1,800 pairs | 626.0 MiB | 622.6 MiB |

Startup savings are larger than the full-run savings, where caches and allocator
reuse also determine the peak. Both full ROS runs preserve every processed pair,
backend decision and received output count. CPU was 0.268 cores before/after on
floor7 and 0.256/0.258 on the saved-map workload; the latter difference is 0.30
CPU seconds over 180 wall seconds. This change establishes a memory benefit,
without claiming an additional runtime CPU improvement.

The independent `floor7_outside_713_room.bag` also ran through actual same-process
DDS, comparing the original `45f4a17` baseline with the final combination. It
preserves all 448 input pairs and backend behavior fields. Both accept the 55th
observation after the same ambiguity checks, but the first path arrives in
**13.820 / 10.524 seconds**. CPU after that first path averages **0.174 / 0.162
cores**. The entire 44.7-second recording, including prolonged initial search,
averages **1.286 / 1.193 cores**. Thus the half-core result applies to normal
tracking; relocalization search and short bursts can exceed it. Earlier locking
also starts authoritative live output sooner (309 / 342 odometry messages).

Two scheduling experiments were rejected. Parallelizing only hypotheses with
the same target increased first-lock latency about 12%. Parallelizing across
targets saved about 4% on the easy prefix and 2% on the difficult recording,
while increasing the latter's peak RSS by 27–39 MiB. The original scheduler is
retained. Samples and rejected source/library hashes are in
[followup_results.json](followup_results.json).

The final full suite passes **61/61 CTest targets**. Frozen complete localization
episodes remain identical in all 52 non-timing rows for legacy, atlas and
atlas-plus-kill configurations.

The existing out-of-map/reentry episode was also run three times per version and
mode, alternating the original `45f4a17` baseline and final library. Every run
requires the same three return observations and preserves all non-timing fields.
Median backend processing across those observations falls from 504.8 to 472.9 ms
in legacy mode, 494.6 to 462.6 ms with atlas, and 398.9 to 377.3 ms with
atlas-plus-kill (5.4–6.5%). These are synthetic recovery computation measurements,
with one registration thread per call and four P-cores pinned for both versions;
they exclude the physical time required to acquire the observations.

## RHPD heading ablation and equivalent descriptor work reduction

This historical stage compares against `682f5af6b7848d7959d1e9e66f32e313a5ce8ce6`
and was published as `4c9c436`.
**That fusion prototype was rejected; this stage retained independent SC.**
It retained the 3037-value RHPD, map format, Product configuration and acceptance
thresholds. The subsequent fusion and sparse-storage stage is described below.

The prototype shared RHPD polar statistics with a heading histogram. There were
22 settings across five independent recordings, each using the first 100 frames:
[110 observations](rhpd_heading_ablation.csv). They cover histogram resolution,
linear/log radial bins, global/sector normalization, heading peaks, uniform and
cardinal coverage, explicit opposite headings, and angular reranking. These
are exploratory ablations without independent ground truth. Some runs overlapped
compilation; their timings are not deployment measurements.

The four-ring/four-layer/60-sector peak-only version accepted the difficult
recording at observation 10, versus 55 previously. Adding back the existing
cardinal coverage exposed a competing registration mode and moved acceptance
to observation 60. Other settings delayed the outside710 recording from 10 to
30 observations. A missing competitor cannot establish better disambiguation;
none of these changes qualified for the requested no-regression default switch.
The prototype and its map-format change were absent from `4c9c436`.

The retained changes remove redundant work while preserving descriptor scores:

- SC generation runs once per added keyframe. The detector stores one descriptor
  matrix and the ring keys used by its eligible-prefix search; its unused second
  descriptor store, sector-key cache and persistent tree are no longer built.
- Pair matching reuses column norms across shifts and reads circularly indexed
  column views. It avoids shifted descriptor matrices, copied column vectors and
  the two input-matrix copies. Coarse search order and weighted scoring stay the
  same.
- Explicitly disabling SC yaw no longer copies SC matrices for RHPD candidates.
  Missing SC scores contribute zero rather than overflowing the finite sentinel
  to infinity. The default option stays enabled.
- RHPD planar projection, polar statistics and index management have separate
  implementation files (392, 169 and 191 lines); moved method bodies are unchanged.

The new E2E cases check four unknown initial headings and rejection of a known
180-degree symmetric scene. Existing out-of-map/reentry, repeated-place, map
lifecycle and extension scenarios remain part of the suite. These checks compare
behavior and known synthetic poses; they do not establish real-world accuracy.

Measured against the frozen baseline, 8,640 SC pairs repeated three times have
**identical scores and yaw shifts**. Median matching thread CPU falls from
**769.123 to 143.513 ms (81.3%)**. This is the SC pair-matching component, not an
81% reduction in localization CPU.

Final full core replays preserve all decision, candidate, alignment, lock and
output-availability fields. Maximum pose-matrix component difference is
`2.78e-13`. Each workload below is one paired, unpaced run; CPU differences of
about 0.4–1.6% do not establish a stable whole-process speedup. RSS includes
startup, allocator high-water marks and caches.

| Workload | Frames | First accepted observation, before / after | Peak RSS before / after |
| --- | ---: | ---: | ---: |
| Floor7 moving recording | 1,848 | 5 / 5 | 413.7 / 389.6 MiB |
| Outside713 difficult recording | 448 | 55 / 55 | 288.0 / 268.9 MiB |
| Inside710 recording | 452 | 5 / 5 | 265.9 / 246.9 MiB |
| Outside707 recording | 448 | 5 / 5 | 264.2 / 245.3 MiB |
| Outside710 recording | 448 | 10 / 10 | 289.1 / 270.0 MiB |
| Saved-map moving cost fixture | 1,800 | 5 / 5 | 548.2 / 510.5 MiB |

The final build passes **61/61 CTest targets, zero skipped**. All eight E2E
scenarios also pass separately against both libraries in legacy, atlas and
atlas-plus-kill modes. Each mode preserves all 70 non-timing trace rows exactly,
including poses. Source/library hashes, raw aggregate samples and comparison
fields are in [descriptor_results.json](descriptor_results.json).
At this stage, relocalization latency and repeated initialization remained
unresolved, and SC removal had not qualified for the default path.

## RHPD fusion, channel ablation and sparse storage — 2026-10-18

Baseline: `4c9c43664c219016d66f6d4c0570e9e70b30c87c`. RHPD now owns both place
and heading features. The seven angular channels and circular matching formerly
implemented by Hybrid Scan Context are integrated into RHPD. Query generation
shares the polar traversal and PCA eigensolve; planar and polar temporary grids
use fixed or contiguous buffers. Independent SC classes, duplicate keyframe
descriptor storage and unused cache APIs have been removed. Place generation,
place matching, heading matching and index management have separate implementation
files. ROS topics and the existing heading-related configuration names remain.

The 64-value vertical-token block and its configuration have been removed after
[20 channel ablations across five recording prefixes](rhpd_channel_ablation.csv).
The remaining place grid has 2973 logical values and a 53-value coarse key.
[17 further grid/channel variants](rhpd_compaction_ablation.csv) did not qualify:
10/8/6-bin planar grids failed to lock the difficult recording within 100 frames;
the four-bin grid delayed acceptance and increased registrations. Area reweighting
could preserve lock frames but increased total registrations. Several smaller
heading grids either failed or changed the resulting pose; no independent GT was
available to establish their accuracy. Removing a planar channel that worked
alone also failed when combined with token removal. Ablations must be validated
in combination. Their exploratory timings are not deployment measurements.

Instead of reducing resolution further, absent zero cells are omitted from
storage and matching. The 60 default-query samples retain an average **505.5
place values** (range 325–637), while the logical grid remains 2973. Combined
place/heading storage averages **24.6 KiB**, including cell indices, object sizes,
vector capacity and cached norms/keys, versus **88.9 KiB** for the equivalent
2973+8400 dense doubles alone. Allocator headers and shared-pointer control blocks
are excluded; these samples are not a whole-map memory estimate. Heading objects
are immutable and shared by the index and its readers.

New pbstream files save geometry and use version **2.8.0**. Loading always rebuilds
place features from causal submaps with saved optimized poses, and heading from
each saved scan, using the active configuration. Old descriptor caches are ignored;
their protobuf fields remain only for resource limits and to prevent tag reuse.
This removes descriptor-schema compatibility paths. Older readers reject new
2.8.0 maps. Rebuilding has a startup cost: the saved-map fixture's map/Atlas setup
was 0.290 / 0.622 seconds before/after; the floor7 pair was 0.354 / 0.352 seconds.
These setup times precede the runtime measurements below.

Generation preserves all **2,588,400 surviving values exactly** across 240 samples
(15 recorded clouds, four yaw angles, V1/V2/V3 and a custom grid/range setting).
Three alternating runs, with ten repetitions per sample, reduce median generation
thread CPU from **1475.45 to 967.02 ms (34.5%)**. This is descriptor generation,
not whole-process CPU. Against the dense fused intermediate version, 3600 pair
comparisons have maximum place/heading score differences of `1.78e-15`/`2.22e-16`
and zero changed yaw selections. Place-pair CPU falls from 6.227 to 3.123 ms;
heading-pair CPU rises from 59.80 to 61.00 ms. The production heading API did not
reproduce an earlier prototype's speedup, and that prototype result is not counted.

All **61 CTest targets pass**. Eight E2E scenarios pass for both libraries in
legacy, Atlas and Atlas-plus-kill modes: 48 scenario passes, with all 70 non-timing
trace rows per mode identical. The suite covers unknown headings, symmetric-scene
rejection, out-of-map/reentry and map lifecycle. Map tests verify cache-free save,
ignored old caches, active descriptor configuration and optimized causal geometry.
No native Noetic build was run; its existing static wrapper check passed.

The six final complete core replays preserve lock frames, decisions, states,
matched/seed/support identifiers and output availability. Maximum pose-matrix
component difference is `2.42e-13`. Token removal changes some candidate and
registration counts; sparse storage alone preserves those counts against the
dense fused intermediate version. Each row is one paired unpaced run, including
all process threads but excluding map loading. RSS is the process high-water mark.

| Workload | Frames | First accepted observation | Process CPU before / after | Peak RSS before / after |
| --- | ---: | ---: | ---: | ---: |
| Floor7 moving recording | 1848 | 5 / 5 | 44.43 / 44.45 s | 391.6 / 366.8 MiB |
| Outside713 difficult recording | 448 | 55 / 55 | 46.81 / 43.19 s | 268.9 / 247.1 MiB |
| Inside710 recording | 452 | 5 / 5 | 8.93 / 8.57 s | 246.7 / 223.4 MiB |
| Outside707 recording | 448 | 5 / 5 | 9.64 / 9.58 s | 245.1 / 222.9 MiB |
| Outside710 recording | 448 | 10 / 10 | 16.15 / 14.50 s | 270.1 / 235.6 MiB |
| Saved-map moving cost fixture | 1800 | 5 / 5 | 41.06 / 41.01 s | 507.0 / 467.5 MiB |

Normal whole-core CPU is essentially unchanged. Difficult-recording CPU falls
7.7%; CPU through first acceptance falls from 42.12 to 38.48 seconds, while the
required observation remains 55. This does not resolve repeated initialization
or demonstrate earlier disambiguation. Independent real-world GT, physical MID360
and RK3588 validation remain outstanding. Raw aggregates, ablation proxy recall,
comparison fields and source/library fingerprints are in
[rhpd_compaction_results.json](rhpd_compaction_results.json).

Host ROS/DDS validation also completed, using the production callbacks and
publication workers at the recorded 10 Hz cadence, with all helper threads
included. The normal 600-frame prefix ran twice per version in alternating order;
the full difficult recording and saved-map 600-frame cost prefix each ran once
per version. Both wrapper executables use the unchanged `4c9c436` node source,
built with the matching version's headers and core library.

| Same-process DDS workload | Average CPU cores before / after | First path before / after | Peak RSS before / after |
| --- | ---: | ---: | ---: |
| Normal, 600 frames, median of two | 0.347 / 0.349 | 0.911 / 0.903 s | 452.3 / 428.2 MiB |
| Difficult, all 448 frames | 1.181 / 1.087 | 10.351 / 9.617 s | 330.9 / 318.9 MiB |
| Saved-map cost fixture, 600 frames | 0.282 / 0.275 | 1.049 / 0.920 s | 585.4 / 551.9 MiB |

Every input pair was processed. All 2248 compared backend rows agree on input
timestamp/point count, state, decision, matched identifier, lock, tracking and
backend publication flags. Normal CPU remains essentially unchanged; after ten
seconds it is 0.300 / 0.302 cores. The difficult recording's CPU after first path
is 0.160 / 0.158 cores, but its initial search still takes more than half a core.
Earlier first path starts authoritative live output sooner on that run (344 / 351
odometry outputs). Live callback counts depend on wall-clock activation and
scheduling; raw received counts are retained separately from backend equivalence.
These results support the local average-half-core normal-operation target on
the measured inputs, not a hard burst limit or RK3588/physical-sensor guarantee.

## Reproduce

Build the normal Release/Humble workspace first with `BUILD_TESTING=ON`; research
tools need not be on.
Build the evidence probes against its generated include/link configuration:

```bash
python3 docs/evidence/runtime_cpu_20261017/build_probes.py \
  /path/to/workspace/build/n3mapping /tmp/n3m_probes
```

Build each probe with its version's matching headers, library and wrapper source;
this stage changes internal C++ types and old/new probes are not ABI interchangeable.
Freeze each compared `libn3mapping_core.so` in its own directory and select it via
`LD_LIBRARY_PATH`, ahead of the normal dependency paths. Use `--node-source` when
building the ROS probe against a frozen wrapper source. The core probe arguments
are `MAP ATLAS MANIFEST OUTPUT_CSV RATE MAX_FRAMES`; rate `1` preserves recorded
timing, `0` runs as fast as possible, and max frames `0` means all frames.
Extract independent scan/odometry pairs with the existing
`tools/extract_ros1_relocalization_manifest.py`; use a compatible compiled Atlas.

For the actual DDS path, convert the original ROS1 bag to ROS2 sqlite/CDR without
changing its messages (the measured conversion used the Humble typestore):

```bash
N3M_DDS_REPLAY=1 OMP_WAIT_POLICY=PASSIVE ROS_LOCALHOST_ONLY=1 \
ROS_DOMAIN_ID=186 ROS_LOG_DIR=/tmp/n3m_ros_logs \
/tmp/n3m_probes/humble_replay_probe \
  /path/to/converted/bag.db3 /tmp/n3m_run 0 --ros-args \
  -p product_profile_v1:=true -p map_path:=/path/to/n3map.pbstream \
  -p reloc_atlas_path:=/path/to/atlas.pb --log-level warn
```

`N3M_PATH_PROBE=1` selects the isolated path-publication experiment. It creates
no external subscriber. Without either flag, inputs are directly injected into
the existing test seam; this mode must not be reported as DDS reception.

For the same-map cost input, generate `n3map_pb2.py` from `proto/n3map.proto`, put
it on `PYTHONPATH`, and run `make_saved_map_replay.py MAP OUTPUT --offset 360
--frames 1800`. Then run `make_callback_database.py MANIFEST OUTPUT_DB
TEMPLATE_ROS2_BAG_DIRECTORY`. The template supplies the original 48-byte
XYZINormal message layout; unused normals/curvature are zero. The database is
explicitly a cost fixture, not an independent sensor recording.

For the descriptor-only comparison, run the new probe with the same saved map
and PCD inputs for both frozen libraries:

```bash
/tmp/n3m_probes/sc_equivalence_probe MAP OUTPUT_CSV QUERY_PCD...
```

It rotates each query by 0, 42, 137 and 180 degrees and compares it with every
loaded heading descriptor. Compare `(query, keyframe, distance, shift)` rows;
`MATCH_CPU_MS` excludes generation, loading and CSV output. The recorded corpus
uses observations 1, 50 and 100 from each of the five recordings.

For generation, `rhpd_generation_probe OUTPUT_BIN REPETITIONS QUERY_PCD...`
emits the surviving place values followed by heading values for four rotations
and four parameter settings per input; the accompanying CSV reports actual
storage. Generation timing excludes dense expansion and file output. The current
builder defines `N3M_FUSED`; the `4c9c436` reference is compiled without it and
with that commit's headers/library. With the same 15 input PCDs, the binary has
240 samples and can be passed to `rhpd_pair_probe INPUT_BIN OUTPUT_CSV`, which
compares the 60 default-setting descriptors in all 3600 pairs. The dense fused
intermediate reference was a local experiment, identified by its library hash
in the results; it was not a separately published product version.

## Reseeding experiments — not enabled in Product (2026-10-18)

Baseline runtime: `114937a`. These experiments change candidate selection while
keeping the numeric acceptance thresholds, input clouds and input frequency.
**Production code and Product defaults remain unchanged.** Their performance
numbers describe isolated prototypes, not the version currently deployed.
Sources, input/output fingerprints and individual results are recorded in
[reseed_results.json](reseed_results.json).

**Promotion rejected on independent SpRobo input:** the guarded/occlusion
combination locks approximately 32 m from truth in the corridor counterexample
below, while baseline abstains. The historical speedups do not meet the
accuracy-preservation requirement. The experiment sources remain available to
reproduce this failure.

The difficult recording spends its first 55 observations in eleven independent
five-observation windows. A failed window clears its hypotheses. Initial seed
registration accounts for 462 of its 773 alignments; the remaining 311 update
existing hypotheses. Retaining hypotheses alone fails to lock within 100
observations: a plausible reverse-heading competitor survives. Odometry-based
warm starts move acceptance from observation 55 to 65 and increase CPU. Limiting
OpenMP to one thread saves only about 5% process CPU and increases processing to
first lock to 36.68 seconds. None of these alternatives qualifies.

Unconditional free-space elimination accepts at observation five and saves most
of that work, but outside710 then differs from baseline by up to 14.45 metres
and consumes more CPU. Rebuilding free rays from their actual saved keyframe
origins, instead of the nearest-origin approximation, does not remove that
counterexample. Neither variant is accepted.

The guarded prototype waits for the complete temporal window, the previous
winner streak, sufficient converged updates, log likelihood and moving-visibility
checks. Visibility and free-space evidence must favour the same physical pose
before elimination; final acceptance still retains the free-space veto. The
previous streak makes this deliberately conservative for windows as short as
the streak requirement. This is evidence-based candidate elimination, not exact
reuse of an unchanged computation, and changes acceptance behavior.

| Complete core replay | Baseline | Guarded prototype |
| --- | ---: | ---: |
| Difficult recording: first accepted observation | 55 | 5 |
| Initializations / registration calls | 11 / 773 | 1 / 68 |
| Total process CPU | 43.19 s | 7.97 s |
| Maximum common-output position / rotation difference | — | 0.01359 m / 0.06930° |

The other five complete inputs retain their lock observations, decisions and
output availability; pose differences are floating-point scale. The unchanged
outside710 acceptance at observation ten is an important counterexample to the
unconditional elimination variant. These recordings do not have independent GT.

New synthetic E2E probes expose a separate confidence problem. Two copies of a
scene are 20 metres apart. One map copy retains 50%, 75% or 90% of its samples;
the query is the same subset observed at the other copy. Baseline and guarded
prototypes abstain at 50%, but choose the wrong copy at 75% and 90%, approximately
20 metres from the supplied synthetic truth. This is an existing failure too,
not evidence that the guarded prototype alone introduced it. Five identical-copy
separations of 20.05–20.30 metres remain ambiguous in both versions.

A subsequent combination uses the existing occlusion-aware range matcher and
refuses elimination when the two visibility scores tie under the existing score
comparator. It passes all ten E2E scenarios, including all three sparse-copy
cases and all five fractional-grid cases. Across the six full core replays it
retains acceptance observations 5/5/5/5/10/5, with one initialization on the hard
recording. However, its real trajectories change: maximum differences from the
baseline across those recordings are 0.01359 metres and 0.16082 degrees. They are
regression measurements, not independently established accuracy errors. The
observation model now permits a matching accumulated return behind a nearer map
return; the synthetic improvement does not establish safety for every occlusion
or wrong-map case. Historical raw-sensor/twenty-query qualification was not rerun.

Authorized same-process DDS measurements include every input pair. The
guarded-only prototype uses 0.222 average cores and produces its first path in
0.896 seconds on the difficult recording. The final occlusion combination was
measured separately:

| Same-process DDS | Baseline | Final experimental combination |
| --- | ---: | ---: |
| Normal 600 frames: average cores | 0.348 | 0.345 |
| Normal: first path | 0.893 s | 0.899 s |
| Difficult 448 frames: average cores | 1.113 | 0.219 |
| Difficult: first path | 9.966 s | 0.919 s |
| Difficult: processed inputs | 448 | 448 |

The difficult average includes the entire 44.7-second recording. Initial search
still exceeds half a core; there is no hard CPU ceiling or throttling. These
single paired runs are evidence for the specific workloads, not deployment
qualification or a universal performance guarantee.

The reproducible experiment sources are [reseed_guard_probe.cpp](reseed_guard_probe.cpp),
[reseed_occlusion_probe.cpp](reseed_occlusion_probe.cpp), and
[reseed_alias_probe.cpp](reseed_alias_probe.cpp). Build them with
[build_reseed_probes.py](build_reseed_probes.py) against the baseline runtime.
They are outside the production build. The baseline alias probe deliberately
fails its two sparse-copy assertions; the final combination passes them.
`guarded_product.so` and `guard_occlusion_product.so` explicitly override the
frozen Product factory for DDS measurement and print their active settings.
Ordinary ROS parameter overrides do not modify that frozen factory; an initial
trial that did only that is excluded from candidate performance claims.

Further optimization must preserve competing evidence instead of deriving
certainty from its deletion. Reduced
reseed count on one recording is not proof that repetitive initialization is
solved for all ambiguous observations, nor that difficult search always uses
at most half a core.

## SpRobo simulation integration (2026-10-18)

The local `simple_rbt` project provides the SpRobo console and a MuJoCo C++
simulator. Its existing scene format supports a 100 x 100 m building. The new
[scene generator](../../../tools/n3mapping_sprobo_scene.py) creates 24 repeated
offices, continuous corridors, a floor and ceiling, and distinguishing obstacles
(178 objects in the original recorded fixture; the varied v2 reference below has
220). The original `changed` fixture moves furniture and partially blocks selected doors;
`repetitive` removes the distinguishing obstacles. Both variants were loaded by
the actual CPU simulator; they have not yet undergone localization qualification.

The [smoke runner](../../../tools/n3mapping_sprobo_smoke.py) uses separate host
processes for simulation, the existing Humble adapter, FAST_LIO and n3m. It uses
the standard n3m mapping/localization configuration, not the frozen Product
profile or reseeding prototype. No estimator runtime or SpRobo source was changed.
The simulator checkout had existing uncommitted development; executable hashes,
configuration, logs, bags and poses identify the actual run, not its HEAD alone.

The CPU sensor retains 20,000 rays per 10 Hz rolling scan and 200 Hz IMU. The
actual valid-return count is about 19,900 in this scene. Camera rendering is
disabled. `/livox/imu` uses the simulator's driver-compatible g units, while
FAST_LIO's gravity magnitude is set to 9.80665 m/s². Its existing MID360 extrinsics
are retained. Simulator odometry goes only to `/ground_truth/odom`, with simulator
dynamic odometry TF disabled; n3m receives actual FAST_LIO outputs, never truth.

A 35-second mapping session saved a 10-keyframe pbstream after roughly 7.5 m of
travel. An independent session started 4 m away with 180-degree heading and
successfully localized against that map. Each accepted acquisition contains 350
raw scans, 7,001 IMUs and 347 exact-stamp LIO cloud/odometry pairs. The simulator
generated the same raw counts. **The scene covers 100 x 100 m; the saved map does
not yet cover the whole building.** Results and identities are in
[sprobo_results.json](sprobo_results.json).

The [truth scorer](../../../tools/n3mapping_sprobo_score.py) composes the simulated
base pose with the IMU extrinsic. It fixes the map-to-simulation transform from
the mapping run's first backend pose and never fits the query trajectory. The
two complete query reruns had position RMSE 0.131–0.133 m and rotation RMSE
0.391–0.392 degrees;
these include frontend and mapping errors and are not a hardware accuracy claim.
Injecting a 20 m error into the query estimates remained visible as a 20.04 m
median error, checking that alignment cannot hide a wrong-location lock.

One initial attempt selected an older installed n3m core and failed at startup;
the runner now pairs the node with its build-directory libraries. A preliminary
query recorded only 6,827/7,001 IMUs with best-effort observation. It is excluded
from complete-acquisition evidence. The final recorder requests reliable sensor
delivery and checks received counts against the simulator summary; `/clock`
retains its publisher's best-effort QoS. No input rate or acceptance gate was
relaxed. Raw input completeness does not prove every ROS consumer received every
sample; the recorded LIO pairing is checked separately.

From the repository root, with Humble and the workspace sourced, reproduce on
the authorized host (choose new output directories and an unused ROS domain):

```bash
cmake -S /path/to/simple_rbt/wrappers -B /tmp/n3m-sprobo-adapter \
  -DSIMPLE_RBT_ROS=ROS2 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/n3m-sprobo-adapter --target simple-rbt-ros-session -j 4
python3 tools/n3mapping_sprobo_scene.py --output /tmp/n3m-indoor.json
# Optional: open the same scene in the existing desktop console.
sprobo --config /tmp/n3m-indoor.json
python3 tools/n3mapping_sprobo_smoke.py \
  --sim-root /path/to/simple_rbt --workspace /path/to/to_migrate_ws \
  --adapter /tmp/n3m-sprobo-adapter/simple-rbt-ros-session \
  --output /tmp/n3m-sim-mapping --domain-id 181
python3 tools/n3mapping_sprobo_smoke.py \
  --sim-root /path/to/simple_rbt --workspace /path/to/to_migrate_ws \
  --adapter /tmp/n3m-sprobo-adapter/simple-rbt-ros-session \
  --output /tmp/n3m-sim-query --domain-id 181 \
  --map /tmp/n3m-sim-mapping/map/n3map.pbstream --spawn -40 0 0 --yaw 180
python3 tools/n3mapping_sprobo_score.py --mapping /tmp/n3m-sim-mapping \
  --query /tmp/n3m-sim-query --output /tmp/n3m-sim-query/score.json
```

The acquisition runner supports `--route coverage` (default 1,800 s ceiling).
The original recorded route had 79 waypoints and approximately 1,232 m; the
current varied scene uses 133 waypoints and approximately 1,281 m through all
24 rooms and back to the start, stopping two seconds after completion. It stops on stale truth or 30 s
without progress and reports actual distance. The route controller consumes truth only to drive the test vehicle;
estimators still consume actual sensors/LIO. `--input-only` records independent
queries without launching n3m; `--route stationary` permits repeated-room and
occlusion observations. Six separate 35-second queries were acquired, each with
350 raw scans, 7,001 IMUs and 347 exact-stamp LIO pairs. They are inputs for
evaluation, not six successful localizations.

**The full mapping attempt was rejected.** After three of 79 waypoints and
145.09 simulation seconds, frontend and backend estimates had drifted badly
while truth stayed on the floor. The acquired data are complete (1,450 scans,
29,019 IMUs and 1,447 LIO pairs), but the backend's fixed-origin maximum position
error is 16.53 m. The run was stopped and preserved as a negative case. Its
pbstream must not become a localization reference map.

Coverage acquisition now requires both route completion and a gross-drift guard
on frontend and backend poses: maximum fixed-origin error at most 1 m and 5
degrees. These are test-map sanity limits, not Product localization acceptance
thresholds. The guard rejects the bad recorded map and accepts the original
short map. An actual 20-second E2E rerun preserved every raw input and passed
the geometry guard, but correctly failed overall because the route was
incomplete. This rerun does not demonstrate that long-route drift is repaired.
The scorer's `--topic /Odometry` option audits the frontend separately.

Recorded-data diagnosis found a numerical inconsistency in the simulated IMU:
after a native accelerate/stop fixture, velocity returns nearly to zero while
integrated accelerometer velocity retains about 0.304 m/s error at a 5 ms physics
step. The error falls to about 0.035 m/s at 0.5 ms. Sparse ray-to-authored-box
checks agree closely during steady motion. Neither these checks nor the IMU
finding establish the complete cause of frontend drift. Same-input 40-second
replays reproduced it; skipping startup, changing accelerometer/gyro covariance,
reducing voxel size, and a diagnostic acceleration replacement derived from
10 Hz truth did not repair it. The truth-derived replay is diagnostic only and
is excluded from end-to-end qualification. No simulator or FAST_LIO production
source/configuration has been changed to mask the failure.

Next: resolve the reproducible frontend/simulation discrepancy, qualify complete
mapping coverage, and test replacement computation-reuse optimizations against
the independent counterexamples below before expanding to the remaining rooms,
changed furniture and occlusion cases. Exact
symmetry should remain ambiguous until observations distinguish locations.
Measure CPU and RSS by replaying identical recorded inputs with the simulator
stopped; sensor generation and frontend cost must remain separate. Noise/bias,
dynamic objects, outside-map behavior, full-building coverage and RK3588
performance remain unqualified. No prototype has been promoted by these tests.

### Independent corridor counterexample and ablation

Against the accepted short map, a 347-frame independent reverse-heading query
gives 343 outputs in both baseline and the guarded/occlusion prototype. Both
first accept observation five; fixed-map position RMSE is 0.13528/0.13538 m.
This positive control alone would miss the regression.

The independent central-corridor query starts at world x=-10 m, well away from
the mapping trajectory near x=-44..-36.5 m. On its identical first 100 LIO inputs:

| Experiment | Output poses | Registration calls | Result |
| --- | ---: | ---: | --- |
| Product baseline | 0 | 1,227 | Remains ambiguous |
| Guarded + occlusion + elimination | 61 | 492 | Wrong position, RMSE 31.64 m |
| Occlusion + veto (elimination disabled) | 0 | 1,227 | Remains ambiguous |
| Earlier guarded elimination, without occlusion | 0 | 1,227 | Remains ambiguous |
| Occlusion + veto + existing persistence | 0 | 631 | Remains ambiguous in this prefix |

The wrong lock occurs at input stamp 4.299989997 s, before the test vehicle
starts moving. Trace logging shows five of six hypotheses eliminated at that
decision, leaving an infinite margin and no competing basin. The retained
candidate has free-space score 0.398618; a distinct viable candidate has score
0.229653. Their sampling deviations are only 0.004694/0.005452, but the relative
score does not establish that either candidate is the true location. Repeated,
correlated returns and partial map coverage make the five-sigma heuristic
insufficient. Disabling elimination prevents this observed failure without
altering numeric acceptance thresholds. The combination is rejected.

Persistence's lower registration count here is not an accepted fix: the earlier
real difficult recording fails to lock within 100 observations under persistence,
where baseline locks at 55. Work reduction and successful-localization latency
must both be checked. These runs are ROS-free, maximum-speed decision screens;
they do not qualify CPU consumption. The full baseline corridor run timed out
at 240 wall seconds after 314 frames. Only its completed first 100 frames are
used in the table. Additional screening was stopped after finding the concrete
counterexample; six acquired queries must not be described as six completed
baseline/prototype comparisons.

The 100-frame failure fixture is retained locally in the repository's existing
ignored map storage, `map/sprobo_corridor_regression` (about 18 MB), including
the map, body-cloud/odometry manifest, independent truth, output traces and
file fingerprints. It contains no truth pose in the estimator's input manifest.
Its hashes and the actual executable identities are in
[sprobo_results.json](sprobo_results.json). Rebuild the Atlas from `map.pbstream`,
then use the existing core probe and rejected experiment builder with this
manifest. The fixed-map scorer also accepts probe output:

```bash
python3 tools/n3mapping_sprobo_score.py \
  --mapping map/sprobo_corridor_regression/mapping \
  --query map/sprobo_corridor_regression/query \
  --probe-csv map/sprobo_corridor_regression/results/inputs_central_prototype.csv \
  --output /tmp/n3m-corridor-score.json
```

It reports 61 poses with approximately 32 m error; it does not fit the query to
truth. A probe with no output has no accuracy score. A coasted output cannot be
scored from the backend pose alone and is rejected by this scorer.

### Complete coverage, visible drift, and veto-only ablation

The subsequent complete acquisition reached **79/79 waypoints**, travelled
**1,224.86 m**, and stopped at 1,236.085 simulation seconds. It recorded all
12,360 raw scans and 247,218 IMUs, with 12,357 exact-stamp LIO pairs. All acquisition
checks passed; **the geometry check failed**. The saved map has 1,208 keyframes.
Frontend maximum horizontal/Z error is 54.93/17.57 m, with 33.55 degrees maximum
attitude error; backend maximum position error is 56.95 m. This map is a failure
fixture and visualization, not an accepted localization reference.

Local PCL Viewer artifacts are in `map/sprobo_view/full_route/`:

- `n3mapping_map_unqualified.pcd`: 3,898,619 points from optimized n3m keyframes,
  with 10 cm display voxel selection. It visibly retains the drift.
- `sprobo_authored_scene_ceiling_hidden.pcd`: 1,474,373 colored surface samples
  from the original 178-object scene JSON, independent of estimation. Walls are
  blue and furniture yellow; only the ceiling is omitted for viewing. The
  simulation still includes its ceiling at z=3.5 m.
- `simulation_truth_reference.pcd`: 4,559,858 measured returns positioned using
  interpolated simulation truth. This is another reference visualization, not
  an n3m map or a frontend-qualified map. Interpolation introduces small surface
  artifacts during motion.

An earlier 332-second acquisition was interrupted by an audit reading its live
SQLite recorder; the recorder reported `database is locked`. That incomplete
run is excluded from acquisition qualification. Monitor progress logs while a
recorder is active and inspect its database after shutdown. Its closed data are
still useful for diagnosis: maximum frontend horizontal/Z error was 0.255/3.415 m.
A diagnostic whole-trajectory rigid fit reduced position RMSE to 0.129 m,
suggesting a large constant-tilt component **in that run only**. This fit does
not replace the fixed-origin gate or fit independent query estimates.

The ceiling is present in the actual sensor inputs: nine sampled scans contain
4,096–11,609 ceiling returns each. A logging-only FAST_LIO build replayed the
first 40 seconds of the closed recording and reproduced all 397 output positions
exactly. At 40 s, 266 of 526 accepted plane normals have |nz| > 0.8; the unweighted
translation information diagonal is approximately (197, 56, 272). This block
does not establish full coupled LIO observability, but it contradicts an absent
vertical-plane-constraint explanation for this segment. Ground and ceiling
normals constrain the same translation direction; a ground plane alone can
penalize both upward and downward displacement. Attitude, inertial/scan timing
consistency and map deformation still require diagnosis. The later full run
also develops substantial XY/yaw error, so it cannot be described as Z drift only.

Separately, disabling candidate elimination while retaining the existing
occlusion-aware visibility model preserves the earlier real-recording benefit:
the difficult 448-frame case first accepts at observation **5 instead of 55**,
with **68 rather than 773** relocalization registrations. The other five complete
core recordings retain their acceptance observations. Maximum common-pose
position/rotation differences are 0.01359 m / 0.16082 degrees; these recordings
do not provide independent accuracy truth. The ten alias/episode scenarios pass
with occlusion + veto, including the two sparse-copy failures of the baseline.

All seven independent SpRobo queries completed with this ablation: the six
partial-map/repeated-room/changed-scene queries emit no poses over 347 inputs
each; the reverse-heading positive emits 343 poses, with fixed-map position
RMSE 0.13538 m. Thus the earlier 32 m false lock is avoided without eliminating
competing hypotheses. Rejection on these queries is not proof of useful coverage
or general localization success.

The twenty historical raw LiDAR/IMU recordings were also replayed through one
FAST_LIO and paired baseline/candidate Product nodes: 3,956 scans and 79,116 IMUs
were sent completely, and both nodes processed every generated LIO pair.
Both variants lock in all twenty. The previously unaccepted `11-56-10` now locks
in both current variants; its independent location label remains unresolved.
Asynchronous published poses differ transiently by up to 1.050 m around a
relocalization correction, despite much smaller differences at the lock itself
and convergence afterward. This is not a twenty-case absolute-accuracy pass.
One scratch replay timed out because a computed spin timeout could become
negative (an indefinite wait); nonnegative waits repaired the replay and the
incomplete attempt is excluded. No production ROS scheduling code was changed.

The pure Product-factory ablation is reproducible without overriding ranking or
elimination methods:

```bash
python3 docs/evidence/runtime_cpu_20261017/build_reseed_probes.py \
  /path/to/build/n3mapping /tmp/n3m-occlusion-veto --occlusion-veto-only
# LD_PRELOAD=/tmp/n3m-occlusion-veto/occlusion_veto_product.so
```

Its complete difficult replay matches the earlier veto ablation's decisions;
maximum pose-matrix element difference is 4.11e-15. **Production remains
unchanged.** These concurrent runs do not provide a new CPU/RSS qualification.
Next: isolate the first frontend/simulation divergence on the frozen full-route
recording, separate backend corrections from asynchronous output timing, and
resolve the unlabeled raw case before promoting visibility defaults. See the
`coverage_and_veto_followup` entry in [sprobo_results.json](sprobo_results.json)
for identities and measured results.

### Varied reference geometry (v2)

Visual review showed that the original 24 nearly identical rooms were a useful
ambiguity stress case but an inadequate representative indoor scene. The default
`reference` generator now varies room dimensions and offsets, rectangular/L-shaped/
chamfered outlines, 3–5 m door openings and their positions, and office, shelving,
meeting and column layouts (220 objects). `changed` shares this structure and
moves/adds furniture (228 objects). `repetitive` retains repeated room geometry
and furniture as a negative case (174 objects). These are geometric changes;
object IDs and display colors are not localization cues. Original sessions and
the frozen false-lock fixture retain their original JSON and are unchanged.

Scene construction and route generation share `room_layout()`, so door edits no
longer require separately rewriting acquisition waypoints. All 133 route segments
were checked against oriented obstacles inflated by a 0.4 m vehicle radius in
all three variants, with no intersections. This is a static path check, not a
completed dynamic coverage run on v2.

A real 35 s CPU-simulator/ROS run started at (43, -24), facing -90 degrees, and
crossed an offset door into a room. All 350 scans and 7,001 IMUs were recorded,
347 exact-stamp LIO pairs were processed, and the map saved successfully. Maximum
frontend/backend position error was 0.274/0.241 m on this short route. It does
not establish that full-route drift or relocalization ambiguity is repaired.
The current reference PCD is
`map/sprobo_view/varied_scene/sprobo_authored_scene_ceiling_hidden.pcd`
(1,533,889 colored points), shown in PCL Viewer with the title
`SpRobo reference v2 - varied geometry (ceiling hidden)`. It comes directly from
authored scene surfaces; the simulation retains the ceiling. Renderer colors do
not enter the MID360 estimator input. Exact run and source identities are in
`varied_scene_v2` in [sprobo_results.json](sprobo_results.json).

### Serial cost and sensor-consistency follow-up

The veto-only Product-factory ablation was measured through real same-process
DDS with no concurrent simulator/replay job. CPU is process CPU time divided by
wall time, not a cap or a four-worker speedup. Map loading is reported separately.

| Recording / variant | Pairs processed | Average cores | First path (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Difficult / baseline | 448/448 | 1.087 | 9.659 | 307.6 |
| Difficult / closest-range + veto | 448/448 | 0.212 | 0.885 | 294.8 |
| Normal prefix / baseline | 600/600 | 0.340 | 0.864 | 426.5 |
| Normal prefix / closest-range + veto | 600/600 | 0.338 | 0.901 | 428.0 |

These are one serial run per variant, on this host, with `OMP_WAIT_POLICY=PASSIVE`.
The normal recording is a 59.93 s prefix; the difficult recording is complete
(44.71 s). The candidate's difficult-run gain comes from reaching the existing
lock gate earlier, not from making every registration five times faster.
**The candidate remains disabled:** choosing a matching background return can
hide a nearer opaque surface. The header, source comments and existing
characterization-test name now state this limitation explicitly. No arithmetic,
threshold, assertion or Product default changed.

Two raw recordings with the largest asynchronous output differences were
replayed once to freeze the LIO cloud/odometry pairs, then both backends consumed
those identical pairs without scheduling differences. On `11-49-55` (194 pairs),
the maximum successful-backend position difference was 0.0928 m / 0.346 degrees;
the second lock occurred at zero-based frames 100 / 102. On `11-52-00` (195 pairs),
the corresponding differences were 0.0198 m / 0.145 degrees, and frames 73 / 74.
The earlier published-pose maxima (1.050 / 0.738 m) used different correction
ages: status metadata shows 3.000 vs 1.500 s and 5.000 vs 3.600 s at those stamps.
Thus asynchronous correction delivery amplifies a smaller backend difference;
the algorithms are not identical, and neither comparison establishes absolute
accuracy on these recordings.

Additional ablations were rejected rather than added to production:

- A one-entry per-thread angular-table cache preserved all difficult-recording
  decisions and pose matrix elements within 3.22e-15. Serial DDS used 1.1097 /
  1.1103 cores before/after, with the same 448 processed inputs and 394 paths.
  There is no measured overall saving to justify the extra cache state.
- Retaining nearest-return scoring but reducing the maximum angular bin width
  to 3, 2, 1 and 0.5 degrees did not improve the difficult 100-frame prefix:
  first locks were observation 55, none, 65 and 55. Registration counts were
  773, 1,441, 937 and 797. These unpaced prefix trials are decision ablations,
  not a controlled CPU comparison or independent accuracy validation.
- Registration worker counts of 1, 2 and 4 preserved all decisions on the same
  100-frame prefix (773 registrations, first lock at observation 55; maximum
  pose matrix element difference 4.67e-15). Elapsed time to lock was 25.59,
  14.46 and 9.81 s, while total process CPU time was 39.73, 38.35 and 44.43 s.
  More workers shorten elapsed time without removing the repeated work. These
  single unpaced runs overlapped the simulator owner's work, so they do not
  qualify an isolated performance gain or a half-core result. The worker count
  is not a process-wide CPU cap; the default remains four.

The simulator investigation now has a native-rate reproduction. Replaying the
frozen `command.applied` events through the same Scout Mini physics reproduces
all 1,400 truth poses and 28,001 raw IMU accelerations over the first 140 s with
zero element difference. It therefore avoids differentiating interpolated
10 Hz truth. At 5 ms `implicitfast`, integrating world-frame specific force after
gravity compensation disagrees with the actual IMU-site velocity increment by
`[-0.312476, -0.000571, -0.387392] m/s` over 7–10 s and
`[0.087318, 0.350173, -0.468243] m/s` over 7–40 s; the stationary 2–7 s interval
is consistent to about 1e-8 m/s. MuJoCo documents that forward acceleration and
implicit velocity integration are separate stages; the exact simulator fix
still needs validation ([computation documentation](https://mujoco.readthedocs.io/en/stable/computation/)).

Replacing only acceleration with the central difference of the reproduced
native velocity still leaves 16.61 m maximum frontend position error; delaying
startup to 6 s still leaves 3.61 m. These are failed diagnostic interventions,
not qualified sensor implementations or an identified sole root cause. The
logging-only frontend binary also diverges numerically from the recorded
trajectory on this longer input, so its results must not be presented as a
strict same-binary accuracy A/B. All transforms use the first pose only, with
no whole-trajectory or query fitting. The complete mapping run remains invalid.

Per Xian's request, the simulator defect and reproducer were sent to the original
SpRobo/simple_rbt development session. This repository keeps the n3m experiments
and consumes the eventual repaired simulator as an independently verified input.
See `sensor_and_serial_followup` in [sprobo_results.json](sprobo_results.json) for
measurements, local artifact paths, source hashes and remaining limitations.
