# Runtime CPU and relocalization measurements — 2026-10-17

Initial-stage baseline: `45f4a171fcab1d9742ab7d575f41a632c7d75073`, branch
`dev/n3mapping-realtime-pipeline`. In that stage the Product V1 configuration remained unchanged
(SHA-256 `1b603a0a180330857aac150e0450cb5e94e284a84934e9b6ee387000f3f254d4`).
These are Release/Humble development builds with **UNVERIFIED** build identity;
they do not constitute a Product Bundle or hardware qualification.

Latest stage: [exact correspondence bounds and map-error diagnosis](#exact-correspondence-bounds-and-map-error-diagnosis--2026-10-19)
reduces complete repetitive-negative CPU by 9.9% in a four-core pair and separates
dominant local map bias from registration residuals. All 14 initial searches
retain their behavior; full replay this stage covers 09/14 and the moving negative.
The earlier [complete 14-position result](#exact-fused-retrieval-and-corrected-heading-seeds--2026-10-19)
and its scope remain below. Difficult-search average half-core is still unmet.

## Measurement boundaries

CPU is process CPU time divided by elapsed time: 1.0 means one fully occupied
logical CPU. All threads are included. Map loading precedes the measurement;
initial relocalization is included. First-path latency starts with the first
input, after map loading. Unless separately labeled as current RSS, memory is the process high-water
mark, not live retained memory. The cost-only runtime optimizations use no CPU quota, input reduction,
frequency reduction, or acceptance changes. The separate October 19 reliability
repair adds an absolute directional witness before accepting a lock. The earlier occlusion-aware scoring and hypothesis-elimination prototypes
remain unadopted. The adopted half-turn shortlist correction and its separate
October 18 measurements are documented below.
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

### Frontend local-map window defect (2026-10-18)

The SpRobo owner delivered an isolated CPU sensor-consistency build. Its source
manifest and binary hash were verified before use. Its closed `live_centred140`
recording was replayed through the unchanged installed FAST_LIO: all 1,396 pose
timestamps and seven pose components exactly reproduce the reported failure.
A logging-only frontend also reproduces the first 396 poses exactly and receives
20 IMUs per processed scan. Delaying initialization to 6 s still leaves 43.51 m
maximum error. This failure is not explained by missing IMU packets or merely
waiting longer before starting the estimator.

The simulation harness starts the frontend executable directly with the MID360
YAML. That path used `mapping.det_range=100` with the node's default
`cube_side_length=200`; it missed the standard MID360 launch file's explicit
1,000 m override. This was a harness integration omission, not evidence of a
defect in recordings made through that standard launch. The local-map
implementation moves a window whenever a
boundary is within `MOV_THRESHOLD * det_range`, where `MOV_THRESHOLD=1.5`.
There is no stationary interior when the full side length is at most 300 m.
Runtime logging confirms the consequence: in a fixed-scan/fixed-IMU fixture,
the window shifts repeatedly; at 0.5 s it deletes 1,646 points, leaving 100,
and lowers its Z upper boundary to 0.044 m in the estimator frame. Across 97
static frames, 48 frames delete a total of 71,194 point instances (including
points subsequently reinserted). Scene ceilings exist, but the frontend removes
their map support. This is distinct from an absent simulated ceiling.

`FAST_LIO/config/mid360.yaml` now explicitly sets the existing window parameter
to 500 m, exceeding both boundary margins
([FAST_LIO e0a02d5](https://github.com/Killow1998/FAST_LIO/commit/e0a02d53701c3bac72c952fe399f024d3ebb2987)).
No frontend binary, input samples,
registration thresholds or truth inputs change. Each row below uses one fixed
first-pose alignment, with no trajectory fit:

| Frozen 140 s input / window | Poses | Position RMSE (m) | Maximum position (m) | Maximum rotation (deg) |
| --- | ---: | ---: | ---: | ---: |
| Original sensor / recorded 200 m baseline | 1,397 | 9.543 | 20.417 | 16.333 |
| Original sensor / 500 m | 1,397 | 0.174 | 0.375 | 0.543 |
| Corrected sensor / 200 m | 1,396 | 21.320 | 46.047 | 37.370 |
| Corrected sensor / 500 m | 1,396 | 0.147 | 0.317 | 0.338 |

Compare window sizes within each recording. The two sensor recordings have
different live command arrival times, so these are not a controlled accuracy
comparison between sensor implementations. Both 500 m frontend runs pass the
existing 1 m / 5 degree gross mapping guard; they do not yet qualify a complete
route, n3m relocalization, real hardware or the half-core target.

The acquisition tool accepts `--sim-executable` to select the isolated candidate
without replacing the desktop installation. It also allows a successful adapter
exit to precede simulator process exit; the existing native message-count checks
still reject truncated streams. One fully received query exposed this lifecycle
race, and its failed acquisition record is retained.

Four independent 35 s queries subsequently passed acquisition with 350 scans,
7,000 IMUs and 346 matched LIO outputs each: a reverse-facing central corridor,
a changed-furniture south corridor, a north corridor and a repetitive central
corridor. Their respective frontend maximum position errors are 0.0375, 0.0275,
0.0323 and 0.0356 m, with maximum rotation errors 0.0984, 0.0894, 0.0799 and
0.0583 degrees. These are input-quality checks,
not yet loaded-map localization results. The corrected IMU stream starts at
2.5 ms, and its last sample precedes the last scan end; the finite run therefore
does not synthesize an extra IMU to process that final scan.

The other launch overrides were checked: this fixture continues using two-point
subsampling and four iterations (the node defaults used in all preceding
experiments), now stated explicitly in its command. Standard launch uses three
and three respectively. Voxel sizes remain 0.5 m and feature extraction remains
disabled in both paths. The fixture is not claimed to reproduce the standard
launch's complete configuration.

See `frontend_window_followup` in [sprobo_results.json](sprobo_results.json) for
the paired measurements, runtime window trace and artifact fingerprints.

### Corrected full-route mapping (2026-10-18)

The isolated corrected sensor build and explicit 500 m frontend window completed
all 133 route waypoints: 1,272.64 m in 1,379.97 simulated seconds. The closed
recording contains 13,799 scans, 275,994 IMUs, 13,799 truth poses and 13,796
frontend outputs. n3m emits 13,724 poses; the initial difference is its existing
static-start guard (first output at 7.60 s). Acquisition, route completion,
saved-map and mapping-quality checks all pass.

| Output | Position RMSE (m) | Maximum position (m) | Maximum rotation (deg) |
| --- | ---: | ---: | ---: |
| Unchanged FAST_LIO binary | 0.2930 | 0.8123 | 1.2561 |
| n3m mapping output | 0.2640 | 0.8402 | 1.2030 |

Both use a single first-pose alignment and the original 1 m / 5 degree gross
quality guard. No trajectory fitting or estimator truth input is used. The
result qualifies this simulated mapping route; it does not qualify hardware,
independent relocalization or the half-core target. The saved map contains
1,276 keyframes. Original failed recordings remain separate and invalid.

The initial full-map Atlas compilation exceeded its existing 1 GiB load limit.
The subsequent cache-format investigation must preserve all effective prepared
target values and retain that limit; merely loosening it or downsampling the
map would not establish an equal-accuracy optimization.

### Atlas packing and RHPD shortlist symmetry (2026-10-18)

Atlas v2 writes 15 effective doubles per prepared point (xyz, normal xyz and
all nine covariance entries), instead of 24 homogeneous-storage doubles. It
validates the omitted constants and reconstructs them on load. The bit patterns
of all effective components survive the roundtrip; runtime cloud layout and
point counts are unchanged. The full-route cache now occupies 699,390,435 bytes
(667 MiB), with 2,400,842 global points and 5,508,139 prepared points, and reloads
successfully under the unchanged 1 GiB limit. This is a serialized-size saving,
not a corresponding resident-memory reduction or a universal large-map solution.
Old Atlas caches must be rebuilt from their original maps; see the root README.

The valid full-route map exposed a separate retrieval defect. In a reverse-facing
central-corridor query, the baseline accepted an approximately 95 m / 180 degree
alias at observation 225. Closest-range visibility with the existing veto also
accepted that alias, at observation 185; it remains disabled. Faster acceptance
of this wrong answer is not an improvement.

An offline descriptor audit of the first input cloud finds the correct region
ranked first by full place distance and by heading distance. The original coarse
key ranks the first correct region 199th, outside its 100-entry shortlist.
Coarse distance ignored the half-turn sector permutation already supported by
the full descriptor distance. Comparing both orientations puts the correct
region first without enlarging the shortlist. The shared RHPD search now applies
that symmetry in its coarse comparison; descriptor contents, registration and
acceptance thresholds remain unchanged. This also affects RHPD loop retrieval.

| Independent 346-frame query | Baseline first lock | Corrected first lock | Corrected maximum position (m) | Corrected maximum rotation (deg) |
| --- | ---: | ---: | ---: | ---: |
| Central corridor, reverse heading | 225, wrong region | 5 | 0.0296 | 0.1861 |
| South corridor, changed furniture | 15 | 5 | 0.0628 | 0.1594 |
| North corridor | 5 | 5 | 0.5128 | 0.4988 |

These scores use the mapping session's fixed first-pose alignment, without query
fitting. They include frontend and map error. Observation counts are not realtime
latency measurements. The central search performs 60 registrations instead of
2,724. The production library reproduces the prototype's decisions on all three
queries, with pose-matrix differences below 1.5e-14. The old implementation fails
the new shortlist regression; the corrected implementation passes it, all 39
loop-detector cases, and the full 61-test suite. Six existing full replay corpora
retain their first-lock observations, output counts and states; their largest
common-pose differences are 0.075 mm and 0.00166 degrees. Those recordings lack
truth and establish regression behavior, not absolute accuracy.

The repetitive-geometry query exceeded the initial 300 s offline timeout in
both the baseline and the first corrected trial; those incomplete records have
no outputs. A longer production run completes all 346 frames with no lock and no
output: conservative abstention in this case, not successful localization. It
performs 4,886 search registrations and takes 558.14 s unpaced. This difficult
search remains far above the half-core goal. Parallel validation overlaps some
exploratory runs, whose timings are not controlled CPU comparisons.

The final production DDS measurements run serially on the authorized host after
all self-owned simulator, frontend and replay jobs stop. Each replays the full
35 s recording at 10 Hz and processes all 346 input pairs. The stable window
starts after 10 s; all 249 subsequent pairs are processed in every case.
CPU includes the in-process replay publisher, subscribers, serialization and
published-pose logging, and excludes FAST_LIO and the simulator.

| Query | Stable CPU cores | Whole-run CPU cores, including search | First path after map loading (s) | Current RSS (GiB) | Published maximum position (m) | Published maximum rotation (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Central reverse | 0.299 | 0.773 | 6.420 | 2.487 | 0.0194 | 0.1952 |
| South changed | 0.315 | 0.746 | 5.616 | 2.494 | 0.0636 | 0.1604 |
| North reference | 0.338 | 0.749 | 5.142 | 2.495 | 0.5125 | 0.4985 |

Map/node loading precedes those latency and CPU measurements and takes about
7.0 s per run. Current RSS is externally sampled from `/proc/<pid>/statm`, with
the median after 30 s from process launch shown here; high-water RSS is retained
separately in the JSON. Each case has 342 backend path poses. The live stream
begins only after authority becomes available and publishes 286, 294 and 298
poses respectively, all scored using the fixed mapping alignment. The probe's
`published_poses.csv` records actual subscriber output, including any projected
pose; the score is not inferred from backend success counters.

These three short normal-query tests meet half a core during stable tracking.
Initial search, the full short-run average and repetitive-geometry search do
not. This is not all-scenario, cross-process transport or RK3588 qualification.
The full-route map was acquired before the shortlist edit; shared loop retrieval
passes the existing suite, but a second complete remapping route has not been
run with this edit. See `corrected_localization_followup` in
[sprobo_results.json](sprobo_results.json) for source/artifact fingerprints,
failed trials, complete trajectories, memory samples and reproducible commands.

### Search accounting and exact geometry reuse (2026-10-18)

This stage starts from `7c7ea604c963eeda9a5741634c5c8c463d16be86` and keeps the
Product configuration, descriptor contents, registration algorithm and acceptance
thresholds unchanged. Measurements and artifact fingerprints are in
[search_cost_20261018.json](search_cost_20261018.json).

The localization core previously failed to forward measured search/tracking times
to the runtime logger, leaving those fields null even when the stages ran.
It now forwards both timings and the actual tracking-attempt flag. The shared
search serializer serves `/localization/status` and the optional runtime JSONL.
It records query aggregation, descriptor generation, first-use index rebuilding,
coarse/place/heading matching, target construction, registration, visibility,
fixed-pose reevaluation and free-space work. Counters distinguish executed work
from stages that did not run. The analysis tool exposes `relocalization_search`
and separates first-use rebuilding from warm descriptor samples.

Timing scopes matter: descriptor CPU is the calling thread's CPU; search CPU
is process CPU during the interval, including workers and background activity.
Alignment timings are sums of call wall times and can overlap across parallel
candidates. Descriptor total already contains its component phases. These values
must not be added together as a CPU breakdown. A clock failure is not reported
as a measured zero, and unexecuted descriptor phases have `calls=0`.
Optional motion-shadow retrieval is charged to `diagnostics_ms`, separately
from normal retrieval; enabling that diagnostic can warm the index first.

On the 1,276-keyframe corrected map, the four initial search prefixes give:

| Descriptor work | Warm mean thread CPU per query |
| --- | ---: |
| Generate place and heading | 0.87–1.10 ms |
| Coarse comparisons | 0.22–0.26 ms |
| Full place comparisons | 0.27–0.29 ms |
| Heading comparisons | 1.65–1.69 ms |
| Complete descriptor generation and retrieval | 3.05–3.34 ms |

The first query additionally spends 409–452 ms rebuilding the frame-level index;
this is separate from approximately 7 s of map/node loading. Each query performs
2,552 coarse comparisons across the two indexes, 200 full place comparisons and
50 heading comparisons, returning ten candidates before spatial grouping.

RHPD remains **2,973 place + 8,400 heading = 11,373 logical values**, plus its
53-value coarse key. The direction-preserving channels already incorporate the
SC role; no separate ScanContext descriptor is computed. Occupied cells are
stored sparsely: the observed warm queries retain about 2,896–3,105 numerical
values in total, excluding index/key/norm/container overhead. Logical dimension
is therefore not an allocation count. At ten descriptor queries per second,
the measured warm work would consume about 0.031–0.033 CPU cores on this host;
steady tracking does not necessarily retrieve on every input. Dimension
reduction is deferred because the measured opportunity is small relative to
geometric search and changing information content risks recall.

Two exact changes are retained:

- Cache the ordered indices of an Atlas visibility crop, sharing the immutable
  global points and prepared registration geometry. The sphere, finite-point
  filter, point order, seeds and registration target are unchanged. Reuse the
  existing search LRU, bounded by 128 MiB / eight entries, invalidated by map
  generation/structure/pose revision and cleared at lock. Empty-crop fallback
  remains candidate-only and cannot enter the hypothesis cache. Ordinary local
  targets retain their existing representation. The observed retained index
  working set is 20–32 MiB; it is not permanent tracking storage.
- Defer the terminal free-space veto until the temporal window and all other
  acceptance gates pass. Kill mode still executes before ranking because it
  changes the hypothesis set. Pending/rejected windows need no final veto;
  every otherwise acceptable solution still receives the same check.

Three serial alternating A/B rounds pin both variants to the same four physical
performance cores, with unchanged four-worker configuration and passive OpenMP
waiting. The baseline includes the same instrumentation. Medians are:

| Search workload | Process CPU before → after | Elapsed before → after | Crop builds before → after |
| --- | ---: | ---: | ---: |
| Central reverse, five observations to first lock | 15.680 → 15.206 s (−3.0%) | 6.528 → 6.092 s (−6.7%) | 33 → 12 |
| Repetitive central, first twenty observations | 141.759 → 137.685 s (−2.9%) | 45.706 → 42.020 s (−8.1%) | 119 → 35 |

Decisions match at every observation; the maximum pose-matrix difference is
1.18e-16. Registration counts remain 60 and 295. Crop wall sums fall 81–86%,
but indexed visibility wall sums increase about 13%, so the overall gain is
much smaller. This is a measured total tradeoff, not an 80% search speedup.
A preliminary copied-cloud cache retained roughly 112 MiB yet barely reused
central crops under the same budget; it was replaced by the index representation.
Earlier unpinned, non-interleaved timings are exploratory and are not the basis
for the reported CPU gain.

Early rejection was tested separately and **not adopted**. Stopping after a
coarse scale that failed convergence/fitness/inlier checks preserved several
ordinary replay prefixes, but lost 37 formerly successful results among 315
initial-pose perturbations across three scenes (13 central, five south, nineteen
north). Every lost result had an unconverged coarse stage; finer stages could
recover it. The reference is the baseline accepted pose transported to the
first query, not independent ground truth. A narrower rule, rejecting only an
already-converged coarse stage with excessive fitness error, lost no such cases
but produced no demonstrated end-to-end gain. Neither rule changes production.

The unmodified initial search rejects 7/60, 13/61 and 10/60 completed registrations
in the three positive queries, and 54/295 in the repetitive prefix. This is the
final registration-quality rejection count, not proof that these candidates
could have been identified safely before optimization. Likewise, a repeated
candidate can reuse invariant map geometry while a new scan or changed pose
still requires fresh geometric evidence. Difficult search remains above half
a core; this stage does not claim to solve that target or qualify RK3588.

Complete core replay covers six real recordings and three positive simulated
queries (6,482 frames). Every state, decision, seed/support/matched keyframe,
first-lock observation and output flag matches the instrumented baseline;
maximum pose-matrix difference is 3.98e-13. The full 346-frame repetitive query
also matches its previous production baseline exactly: zero locks/outputs and
4,886 registrations. The latter baseline is from the previous stage, so its
whole-run CPU cannot establish a controlled A/B gain. The controlled negative
CPU claim above is restricted to the twenty-frame prefixes.

Final-source verification passes all 61 CTest entries and serial local DDS
replays of the three positive queries at 10 Hz. Both baseline and selected node
process all 346 pairs per query, including all 249 pairs in the stable window
after ten seconds. The following are individual DDS runs, not repeated medians:

| Query | Stable CPU before → after (cores) | First path before → after (s) | Current RSS before → after (GiB) | Selected maximum position / rotation |
| --- | ---: | ---: | ---: | ---: |
| Central reverse | 0.310 → 0.312 | 6.621 → 6.000 | 2.495 → 2.421 | 0.0187 m / 0.1952° |
| South changed | 0.330 → 0.325 | 5.850 → 5.489 | 2.500 → 2.423 | 0.0636 m / 0.1604° |
| North reference | 0.354 → 0.353 | 5.110 → 5.012 | 2.491 → 2.445 | 0.5125 m / 0.4985° |

The selected whole-run averages, including search, are 0.785 / 0.751 / 0.760
cores. First-path times start after map/node loading, which takes another
7.21–7.25 s in these selected runs. CPU includes the node, in-process replay,
observers and serialization; it excludes simulator and FAST_LIO. RSS is the
external sample median after thirty seconds from process launch. Pose scoring
uses the unchanged mapping first-pose alignment, with no query fitting; the
1 m / 5° gross quality check is not a claim of centimetre-level accuracy.

Each selected runtime JSONL contains five finite, correctly forwarded search
timings and 341 actual tracking attempts with finite tracking timings.
Descriptor subtotals fit within their recorded totals, and all runtime records
pass schema validation. The previous instrumented snapshot still records zero
tracking attempts when only its logger is enabled, reproducing the flag defect;
the final core fixes it. Tests use the explicitly selected build libraries;
the existing installed workspace binaries are not replaced by these experiments.

For local reproduction, build the probes with the existing `build_probes.py`,
source the Humble/workspace setup, and put the corresponding build libraries
first in `LD_LIBRARY_PATH` (do not mix old installed reader/core libraries).
`replay_probe MAP ATLAS MANIFEST OUTPUT_CSV 0 MAX_FRAMES` writes both the normal
CSV and `OUTPUT_CSV.costs.jsonl`. The retained artifact root contains
`pinned_ab.py`, `full_replay.py`, `run_dds.py`, the rejected coarse-gate sources,
the perturbation probe, complete outputs and frozen baseline/selected libraries.
Their hashes, exact DDS arguments and timing audit are in the linked JSON.

### Registration reduction and persistent reacquisition (2026-10-18)

Against `472d5b1`, registration and fixed-pose linearization now accumulate each
worker's Hessian, gradient and error on its own stack, then publish once per
worker. This removes per-point writes to adjacent shared accumulators. Factors,
guided scheduling, optimizer iterations, distance gates and acceptance remain
unchanged. Upstream error evaluation is reused. No descriptor dimensions change.

[Measured results](../registration_reduction_20261018.json) contain three serial,
alternating A/B rounds on CPUs 0, 2, 4, 6 with `OMP_WAIT_POLICY=PASSIVE`:

| Scope | Baseline CPU | Candidate CPU | Change |
| --- | ---: | ---: | ---: |
| Four-thread prepared alignment, sum of four initial offsets | 52.673 ms | 50.336 ms | -4.44% |
| Complete 100-frame repetitive-corridor replay | 267.776 s | 264.481 s | -1.23% |
| 300-frame retained-map-scan replay | 6.706 s | 6.680 s | -0.38% |

The last row and single-thread synthetic episodes show little material change;
the latter vary by about +0.2–0.4%. The prepared-alignment gain is not a whole-node
gain. Across 3,336 replay frames including repetitions, decisions and work counts
match; maximum pose-matrix difference is 1.05e-13. Single-thread optimizer checks
remain exact; four-thread checks allow summation roundoff already present in the
baseline. Release/Humble CTest passes 61/61. These are regression and cost tests,
not new hardware, absolute-accuracy, RAM or difficult-search half-core acceptance.

The reboot removed the previous `/tmp` artifacts. Their committed measurements
remain historical evidence, but those raw files are no longer available. New
frozen libraries, input manifests, runners and outputs live in the workspace's
`build/n3mapping/registration_reduction_20261018`; raw simulation captures live in
`map/sprobo_20261018`. The first replacement adapter omitted the configured truth
topic. The old acquisition checker incorrectly accepted `central` without truth;
that recording is excluded. Startup now requires a truth publisher, and completed
recordings must contain one truth pose per native LiDAR scan. `coverage` verifies
that the old adapter is rejected before simulation begins. An adapter rebuilt in
the workspace, without changing simple_rbt sources, passes these checks in
`central_v2`: 350 scans, 7,000 IMUs, 350 truth poses and 346 LIO pairs; frontend
maximum errors are 0.0337 m / 0.1005 degrees.

Full `coverage_v2` reacquisition now passes: 133/133 waypoints, 1,273.09 m,
1,379.485 simulation seconds, 13,794 scans and truth poses, 275,897 IMUs and
13,791 exact-stamp LIO pairs. The original fixed-first-pose 1 m / 5 degree
mapping guard passes: FAST_LIO maximum 0.7564 m / 1.1391 degrees, n3m maximum
0.6970 m / 1.1023 degrees. This capture uses the `1ccbd63` core. Different live
command arrival times prevent treating old/new acquisition errors as a strict
algorithm A/B. Raw inputs, metadata and hashes are retained in the workspace;
the bag is closed and all acquisition processes have exited.

The map is `map/sprobo_20261018/coverage_v2/map/n3map.pbstream`: 1,271 keyframes.
Its verified Atlas contains 2,356,595 global points and 5,410,379 prepared points,
686,951,283 bytes on disk. `global_map.pcd` in the same directory exports those
global points without additional downsampling, ready for PCL viewing. The
independent `central_v2/manifest/frames.csv` contains 346 cloud/odometry pairs;
truth remains separate in `poses.csv`. This restores one independent query,
not all four historical query conditions or a new DDS/hardware qualification.

### Retained target-tree capacity (2026-10-18)

After its serial or OpenMP builder completes, small_gicp reduces the node count
but keeps a node allocation sized to the input point count. A private builder
wrapper now releases only that unused capacity before the target tree becomes
immutable. The same operation covers freshly prepared targets and decoded
Atlas targets. Point data, node order, search traversal, acceptance and cache
policy remain unchanged; temporary source trees do not incur the extra copy.

Against `1ccbd63`, three serial alternating fresh-process measurements give:

| Atlas-only process | Loaded RSS before → after | Load CPU before → after |
| --- | ---: | ---: |
| Retained real map | 160.74 → 154.88 MiB | 146.51 → 156.25 ms |
| New full simulation map | 1,277.36 → 1,169.48 MiB | 1,790.05 → 1,816.96 ms |

The large-map node allocation falls from 123.83 to 15.00 MiB; process RSS falls
107.87 MiB. Compaction costs a one-time copy: large-map load CPU increases
26.91 ms (1.50%). These are Atlas-only measurements, not full-node RSS or a
steady-state CPU improvement. All prepared values and 30,000 nearest-neighbor
queries per map have identical hashes before/after. Release/Humble CTest again
passes 61/61.

The complete independent central reverse query locks at frame index 4 in both
versions and emits 342 poses. All 346 frame decisions and work counts match;
maximum pose-matrix difference is 4.44e-16. Fixed-map truth scoring gives maximum
0.02054 m / 0.16619 degrees for both. This core/realtime-output replay does not
establish new DDS, repeated-geometry or half-core results. The frozen libraries,
probe, runner, individual measurements and hashes are under the existing
`build/n3mapping/registration_reduction_20261018` root and linked JSON.

An ancestor-bound search prototype was rejected. At the existing 2 m gate it
reduces point-distance evaluations from 3,665,892 to 3,039,952 with 20,000 exact
answers, but median query CPU rises from 12.508 to 13.515 ms (8.05%). Conservative
bound/rounding overhead exceeds the saved distance work; this code remains only
in the experiment directory. Reducing operation counts alone is not evidence
of an overall performance improvement. Difficult-search half-core operation
remains unresolved.

### Target storage locality and changed-scene regression (2026-10-19)

Against `f0a3590`, retained target points are now packed into their existing KD
leaf order during construction. An in-place cycle permutation reuses the tree's
index array; decoded normals and covariances move with their points. Fresh target
features are estimated after packing. This reduces scattered memory reads while
preserving tree traversal, physical tie choices, geometry and acceptance rules.
Source storage, descriptors and public interfaces are unchanged.

[Measurements and counterexample](../target_locality_20261019.json) retain three
serial alternating A/B rounds, CPUs 0, 2, 4, 6 and PASSIVE OpenMP. Each run consumes
the first 100 frames of the same frozen query. CPU includes all process threads.

| Query | Replay CPU median before → after | CPU change | CPU change including load |
| --- | ---: | ---: | ---: |
| Central | 17.068 → 15.664 s | -8.23% | -4.12% |
| Repetitive, changed scene | 24.245 → 22.461 s | -7.36% | -2.53% |

Packing adds roughly 0.6 s to full-core load CPU. Median total wall time including
load is 12.247 → 12.350 s / 14.353 → 14.421 s: cold-start latency has not improved.
Peak replay RSS stays approximately 1.865 GiB. One Atlas-only check also has the
same peak RSS and essentially the same retained RSS; this is not a memory-saving
claim. Complete 346-frame timing has only one pair per case and varies by case,
including a small CPU increase for the south query; no general steady-state or
DDS gain is established by these measurements.

All prepared records have identical unordered record checksums, and 30,000
nearest-neighbor queries return identical coordinates, associated features and
distances. Tests retain bit-for-bit record assertions across Atlas round trips,
comparing complete records independent of storage order. An additional 10,000
queries cover geometry, feature associations and duplicate-point tie choices.
Humble CTest passes 61/61. Three synthetic episode modes, 70 frames each, have
exactly equal non-timing fields and poses. The 1,200 frames in repeated timing
runs and 2,768 frames in complete query A/B runs have equal decisions and work
counts; maximum full-query pose-matrix difference is 1.43e-14.

Three additional 35 s acquisitions are now closed and retained beside
`central_v2`: `south_v2` (changed furniture), `north_v2` (reference geometry),
and `repetitive_v2` (repeated room templates, altered room positions/shapes and
removed landmarks). Each contains 350 scans, 7,000 IMUs, 350 truth poses and
346 exact-stamp LIO pairs. Frontend maximum position errors are respectively
0.0271 / 0.0365 / 0.0357 m. No simple_rbt source or installed binary was changed.
For the full central/south/north localization queries, fixed-map alignment gives
maximum position errors 0.02054 / 0.03019 / 0.51053 m in both versions.

**The changed repetitive scene fails localization safety in both versions.** It
locks at frame index 4 with about 58.8 m / 180 degrees error; the full query
reaches 74.893 m position error. Registration, five-frame confirmation and the
relative free-space veto all accept the wrong solution. This map/query mismatch
does not establish that a correct pose can always be recovered; it does establish
that the current logic fails to abstain. Its faster replay is a cost result for
unchanged incorrect behavior, not a successful difficult-localization result.

An offline first-frame audit finds that floor/ceiling support masks poor support
from walls and objects. Weighting all six pose directions by their weakest
support would also reject the correct central pose, so that proposal is excluded.
Planar position/yaw support separates the three normal accepted poses from this
wrong candidate in the audit, but arbitrary tilt, occlusion, dynamics and map
boundaries remained untested in that audit. No evidence gate was enabled at that
stage. The directional lock witness adopted in the following reliability stage
addresses the reproduced rejection failure; it does not establish universal
scene uniqueness or complete boundary/dynamic-object qualification.

Warm RHPD generation plus retrieval remains 3.10–3.20 ms calling-thread CPU in
these queries; initial frame-index construction is another 406–408 ms. The
2,973 place + 8,400 heading dimensions remain unchanged, with no independent SC
pass. Scheduling, nearest-neighbor certificate, iterative-traversal and prefetch
prototypes were not adopted: exploratory gains were absent or insufficiently
established. Their sources and outputs, the chosen frozen libraries, complete
inputs and runners remain under `build/n3mapping/search_scheduling_20261018`.
Difficult-search half-core operation, repeated-initialization elimination and
out-of-map/re-entry qualification remain open. This stage has no new DDS or
RK3588 qualification and does not replace the installed version.

### Directional lock witness (2026-10-19)

The `ea157e0` baseline could accept the changed repetitive scene because shared
floor and ceiling returns dominated its pooled visibility score. Stationary
five-frame confirmation repeated that evidence; a relative competitor veto
could not establish that the winner itself was plausible.

Before a prospective lock, the localizer now measures support separately in
all combinations of planar translation and yaw. Twenty-neighbor query normals
provide point-to-plane Jacobians at the existing nearest-return angular bins.
The odometry world's Z axis is transformed into the query frame, accounting for
sensor mounting tilt; no simulator pose, authored surface label or query truth
enters this calculation. This follows the existing odometry horizontal-plane
convention, not a new measurement of gravity.

Whitening against all query information makes the smallest eigenvalue of the
matching information equal to the least supported motion direction. It must
exceed one half: geometric support must outweigh non-support in every direction.
Rank-deficient observations cannot pass. This is an additional refusal gate;
ranking, old likelihood/margin/free-space gates and registration parameters
remain intact. Normal estimation occurs only for an otherwise eligible winner,
with the existing query bins and target crop; tracking does not prepare it.
The implementation lives in `visibility_planar_support.cpp`, sharing the
existing visibility projection rather than maintaining another map renderer.

[Reproduction and measurements](../directional_reliability_20261019.json) use the
same closed inputs, fixed CPUs 0/2/4/6 and PASSIVE OpenMP. Four complete queries,
346 frames each, run through the core and `RealtimeOutput` with truth used only
by the fixed-mapping-alignment scorer:

| Query | Accepted outputs before → after | First lock | Maximum position error after |
| --- | ---: | ---: | ---: |
| Central | 342 → 342 | frame index 4, unchanged | 0.02054 m |
| South, changed furniture | 342 → 342 | frame index 4, unchanged | 0.03019 m |
| North | 342 → 342 | frame index 4, unchanged | 0.51053 m |
| Repetitive, changed geometry | 342 → 0 | none after repair | unavailable: abstention |

The three normal traces retain every decision and work count checked, with pose
matrix differences at most 3.56e-15. Their lock witnesses are 0.8977 / 0.8988 /
0.7538. In the negative query, eleven otherwise eligible attempts are refused
with support 0.0339–0.1894; the baseline's maximum error was 74.893 m and almost
180 degrees. No output is a safety result here, not successful localization or
an accuracy score. The map/query geometry change prevents assuming that a true
pose must be recoverable.

The witness takes 16.94–22.36 ms per evaluated lock attempt in these complete
runs. Normal tracking does not execute it. The negative still consumes 1,759.57 s
process CPU and 558.35 s unpaced wall time for the 35 s recording: repeated
initialization and expensive failed searches remain unresolved. Faster
incorrect locking is not a useful CPU baseline for successful localization.
Normal status costs include `directional_checked`, `directional_check_ms` and
`directional_min_support` (null when unavailable). Rejection reasons distinguish
geometric disagreement, unobservable query geometry and unavailable visibility;
warning logs include the reason instead of displaying only the older gates.

Final Humble CTest passes 61/61. Synthetic episodes pass in default, Atlas and
experimental kill modes. The generated saved-map fixture retains all 300 frame
decisions, 296 outputs and its first lock at frame index 4; this is a behavior
regression, not an independent real-world accuracy measurement. Final-build
smoke replays preserve normal acceptance and negative refusal. One old
dense-trajectory test had required accepting a flat map with a different
non-planar query. It now checks refusal without adding keyframes/trajectory,
then uses matching geometry to verify the original mixed-source metadata.

Behavioral tests cover floor-dominated wrong walls, one unsupported horizontal
translation despite agreement in other directions, missing returns with sensor
tilt, and tilted floor-only observations. These checks do not qualify large
dynamic occlusions, arbitrary odometry-world tilt, all repeated environments,
out-of-map/re-entry behavior or RK3588 deployment. Installed programs and
simple_rbt are unchanged; no DDS timing or hardware-performance claim is made.

### Fresh confirmation with reusable seed plans (2026-10-19)

The subsequent **Incremental candidates and current-observation confirmation**
stage below replaces this stage's whole-plan equality requirement.

The next step reduces repeated initialization after a rejected confirmation
window. With the same freshly retrieved keyframe/yaw seeds and map revision,
the localizer keeps pose proposals and competing hypotheses, while clearing
the previous window's likelihoods, update counts, visibility evidence and winner
streak. Every confirmation window therefore needs fresh observations. Changed
seeds, a changed map or no surviving hypothesis still trigger initialization.
If none of the reused poses has acceptable geometry in the current observation,
the localizer retries the full seed plan on that same observation. This recovery
is necessary even when retrieval returns identical seeds: a previously rejected
pose mode may become the supported one as the robot moves.

The candidate initialization code now lives in
`src/world_localizing_initialization.cpp` (275 lines), reducing
`world_localizing_relocalization.cpp` to 771 lines. Preprocessing, registration
objective, numeric acceptance gates, RHPD and tracking are unchanged. Costs
expose `reused_seed_plan` and `retried_seed_plan`; these are diagnostics, not
inputs to the search policy. Reset, map replacement and successful lock release
the stored seed plan. The experimental persistent-evidence mode retains its
existing behavior.

[Measurements, input fingerprints and validation logs](../relocalization_reuse_20261019.json)
record serial, unpaced core + `RealtimeOutput` replays on CPUs 0/2/4/6 with
PASSIVE OpenMP. Process CPU includes all workers and replay overhead after map
load. The budget denominator is the fixed input timestamp span, not the slower
processing wall time; short CPU peaks are allowed. Normal simulation comparisons
use the preceding complete directional-witness replay, whose frozen binary hash
is recorded separately. The real-data comparison uses the frozen `5b64e6a`
baseline. Final timed runs did not overlap this session's builds.

| Input | Frames | Initializations before → after | Alignments before → after | Process CPU before → after |
| --- | ---: | ---: | ---: | ---: |
| Central simulation | 346 | 1 → 1 | 51 → 51 | 22.40 → 22.36 s |
| South simulation | 346 | 1 → 1 | 56 → 56 | 19.97 → 20.11 s |
| North simulation | 346 | 1 → 1 | 68 → 68 | 24.43 → 24.49 s |
| Repetitive changed scene | 346 | 70 → 65 | 4,445 → 4,297 | 1,759.57 → 1,670.06 s |
| Floor 7 outside room 710 | 100 | 20 → 18 | 1,380 → 1,288 | 84.54 → 78.48 s |

The complete negative remains at zero locks and zero authoritative outputs;
this is abstention, not a recovered pose or an accuracy score. Its CPU reduction
is about 5.1% in this single full comparison. The real difficult prefix saves
about 7.2%, also without locking; it has no independent position truth. The
three normal simulations retain their first lock at frame index 4, all 342
outputs, all decisions/work counts and pose matrix agreement within 3.56e-15.
Their whole-episode CPU/input ratios are 0.648 / 0.583 / 0.710; after-lock
tracking alone is 0.269 / 0.268 / 0.297 core. Tracking cost cannot be used to
claim that initial or difficult search meets the budget.

**The difficult-search half-core target is not achieved.** The complete negative
still needs 1,670.06 CPU seconds for 34.5 seconds of input, a required CPU/input
ratio of 48.41. This is a throughput deficit, not measured utilization of 48
physical cores. A small change in the descriptor seed set still rebuilds the
whole initialization plan. Incremental admission of changed seeds and the cost
of continuous hypothesis registration remain open. No input frames were dropped
and no acceptance thresholds were relaxed to obtain the reported reductions.

Original floor-7 bags were recovered from the persistent bag directory and
paired by exact cloud/odometry header timestamps. Their matching map is
`bagfile/n3mapping_test/n3map.pbstream` (SHA256
`a69db12c2476ec1eea2781068fe334415e102bf92d64871d3e707e99a9554d75`),
not the different map shipped in this repository. Exploratory runs using the
wrong pair and timings overlapping a build are excluded. Inside-room and
room-to-meeting prefixes each retain all 100 decisions, 96 outputs and first
lock at index 4. The generated saved-map fixture retains all 300 decisions and
296 outputs; it remains a behavior fixture, not independent accuracy evidence.

Final Humble CTest passes **61/61**. Nine synthetic episodes pass in default,
Atlas and experimental kill modes (94 observations per mode). Five ambiguous
confirmation windows now need one initialization instead of five, publish no
guess, and lock at the same third fresh observation after distinct geometry
appears. A separate episode holds retrieval seeds unchanged while a previously
unsupported place becomes valid; baseline and final pass, while unguarded
proposal reuse fails. The map-extension fixture now models a moving sensor
observing fixed world surfaces and checks known positions; baseline and final
both pass.

Three ablations were rejected: a stationary-endpoint cache saved only 35 of
1,380 registrations and increased CPU from 84.54 to 84.78 s; unguarded proposal
reuse could strand search in old pose modes; a scalar normal-projection objective
reduced some replay costs but failed the physically consistent map-extension
case. Their source/results remain in the build artifact directory, and none of
these rejected implementations or speedups is part of the delivered change.
Installed executables, simple_rbt and remote branches are unchanged. Full
out-of-map/re-entry behavior and RK3588 qualification remain unverified.

### Incremental candidates and current-observation confirmation (2026-10-19)

Baseline: `2c4f1ad`. The final frozen variant is `resumed_progress` in
`build/n3mapping/incremental_search_20261019` under the workspace. See
[incremental_confirmation_20261019.json](../incremental_confirmation_20261019.json)
for source/binary/input fingerprints, complete results, accuracy comparisons and
rejected ablations. The existing `run.py`, `validate_final.py` and `summarize.py`
in that artifact directory reproduce the measurements; these are build artifacts,
not a separate issue workspace or a patch delivery.

Candidate pose proposals now survive changes to individual keyframe/yaw seeds.
The cache retains the current and immediately previous seed plans, and clears on
map revision, reset or successful lock. Angular lookup uses the generator's
existing periodic `1e-6` rad equivalence: neighboring heading sectors previously
missed the cache because float/double expressions differed by roughly
`1e-9`–`1e-7` rad. The regression reproduces four cold registrations before the
fix, versus three freshly evaluated reused seeds and one new registration after.

Cached poses carry no temporal scores, visibility evidence or convergence credit.
Each use evaluates current full prepared geometry with the existing confidence,
fitness and inlier gates; failure retries the original descriptor seed immediately.
A finite, numerically acceptable endpoint that hit the iteration limit may be
retained only as an unfinished proposal: its next use must perform a full current
registration before admission. Failed continuation falls back to the original
seed in the same observation. Explicit geometric rejection invalidates the
affected seed basin; ambiguity alone does not discard other useful proposals.

Intermediate confirmation observations first check the odometry-propagated pose;
failure triggers full registration immediately. The final confirmation observation
still registers fully, and only a winner with current converged registration and
accepted current geometry may publish. Identical geometry checks within that
observation are reused only when pose, source, target and metric agree. New seeds
still receive full registration. Preprocessing, RHPD, registration objectives,
numeric acceptance gates, input selection, ROS interfaces and tracking are unchanged.

All measurements below process every selected input frame, serially, on CPUs
0/2/4/6 with PASSIVE OpenMP. CPU includes all workers and replay overhead after
map loading. These are single complete before/after pairs using the same frozen
inputs, not multi-run medians, DDS real-time measurements or RK3588 qualification.

| Input | Frames | CPU before → after (s) | Full registrations before → after | Final result |
|---|---:|---:|---:|---|
| Central | 346 | 22.36 → 18.19 | 51 → 36 | First lock index 4, 342 outputs |
| South | 346 | 20.11 → 17.24 | 56 → 41 | First lock index 4, 342 outputs |
| North | 346 | 24.49 → 20.98 | 68 → 53 | First lock index 4, 342 outputs |
| Changed repetitive scene | 346 | 1670.06 → 583.44 | 4297 → 1223 | No locks or outputs |
| Real outside-room prefix | 100 | 78.48 → 39.76 | 1288 → 464 | No locks or outputs |
| Real inside-room prefix | 100 | 4.04 → 3.25 | 81 → 54 | First lock index 4, 96 outputs |
| Real room-to-meeting prefix | 100 | 4.80 → 4.03 | 83 → 56 | First lock index 4, 96 outputs |

In the complete difficult simulation, original-seed registrations fall from 2517
to 674, with 84 additional unfinished-solve resumptions; confirmation registrations
fall from 1780 to 465. The total is 1223. Rebuilding lightweight confirmation
metadata is not a cold registration: the number of evaluated seed plans is 70
versus 65 before, while expensive work is substantially lower. The unfinished-solve
ablation reduces CPU a further 13.1% versus the already validated 671.59 s variant.
The real difficult prefix improves only about 2.5% in this final ablation; the
approximately 49% total reduction includes the earlier seed/confirmation changes.

All five positive replay cases retain their lock/publication decisions and first
lock frame. With a single fixed mapping alignment and no query fitting, simulated
maximum position errors are 0.02054 / 0.03019 / 0.51053 m. North's maximum changes
by about 1.9 micrometres and RMSE improves by about 4.9 micrometres; its largest
published-position change is 0.714 mm. This does not fix the pre-existing roughly
half-metre north-map error. The two real positive prefixes have no independent
truth; their largest published-position difference is 5.32 mm, not an accuracy
bound. Negative rejection reasons change as fresh confirmation proceeds; neither
negative produces a lock. Rejection is not recovery of the true pose.

Humble CTest passes **61/61**. All **10** synthetic E2E episodes pass in default,
Atlas and experimental kill modes (97 observations per mode), including unknown
yaw, same-seed new geometry, ambiguous-to-distinct transitions, lifecycle/map
extension, out-of-map projection/re-entry and odometry bias before publication.
An unguarded cache fails the new below-confidence recovery regression. Skipping
full registration of brand-new descriptor seeds fails the unknown-yaw episode
and is excluded. Fine-only continuation did not improve difficult CPU and is
also excluded. A stale-object ABI experiment is excluded entirely; all affected
targets were rebuilt before subsequent measurements.

Logs distinguish `seed_initializations`, `seed_continuations`, `seed_resumptions`,
`seed_retries`, `invalidated_seed_poses`, `confirmation_evaluations` and
`confirmation_registrations`. Resumptions are a subset of continuations, not free
cache hits. All 1684 replay observations satisfy
`alignments = seed_initializations + seed_resumptions + confirmation_registrations`.
Verbose `[RelocSeedReject]` diagnostics include termination, iterations, fitness
and inlier ratio. In the separate 20-frame diagnostic prefixes, all 46 / 29
recorded seed rejections hit the iteration limit; verbose-run timing is excluded.
Per-call stage durations are summed wall times that can overlap across workers,
so they must not be presented as process CPU percentages.

Confirmation is extracted into a 246-line implementation; the main relocalization
file decreases from 771 to 553 lines. Observation preparation and match-quality
evaluation have one shared implementation. Internal C++ layouts changed and
require a complete rebuild. Peak simulation replay RSS is about 1915 MiB versus
1912 MiB before; this stage does not demonstrate a memory reduction.

**The difficult average-half-core goal is still unmet.** The unchanged 34.5 s
input span permits 17.25 CPU s; this version needs 583.44 CPU s. The ratio 16.91
CPU s per input second describes required throughput, not actual utilization of
17 physical cores. New/rejected seeds, non-convergent solves, current geometric
checks and final registration remain substantial costs. Further work must reduce
those computations while preserving recoverable modes and fresh evidence. Full
real out-of-map/re-entry, broad adverse-scene reliability and RK3588 remain open.
No installation, simple_rbt changes or remote push were performed.

### i5-14600KF latency with all CPUs available (2026-10-19)

The host is an i5-14600KF, with 14 physical cores and 20 logical CPUs. The earlier
CPU-budget experiments deliberately used CPUs 0/2/4/6. A follow-up using committed
core `4982862` opens affinity to 0–19 and varies only the benchmark's per-registration
thread count. Product configuration is unchanged. Full settings, fingerprints and
samples are in [relocalization_latency_14600kf_20261019.json](../relocalization_latency_14600kf_20261019.json);
raw artifacts are in `build/n3mapping/incremental_search_20261019/allcore_latency`.

These wall times start with the first replayed LIO observation **after map load**
and end at confirmed lock, with the original 10 Hz pacing and five input frames.
They include replay/PCD overhead and RealtimeOutput, and exclude frontend startup
and DDS transport. Each row below uses all 20 logical CPUs as the allowed affinity,
14 threads per registration, PASSIVE OpenMP, and three fresh-process repetitions.

| Query | Median wall to lock (s) | Observed range (s) |
|---|---:|---:|
| Central simulation | 3.118 | 3.106–3.362 |
| South simulation | 3.055 | 2.934–3.075 |
| North simulation | 3.191 | 3.189–3.354 |
| Real inside room | 0.566 | 0.565–0.567 |
| Real room to meeting | 0.557 | 0.556–0.557 |

All positive samples lock at index 4, with maximum pose-matrix component difference
below `7e-16` from the committed four-CPU reference. This is a five-frame latency
check, not broad thread-count accuracy qualification. The changed repetitive scene
still produces no output in its tested five-frame window; its 1.85 s processing time
is not time to successful relocalization. Failed searches have no promised lock time.

The central three-run median is 3.126 s at 20 threads versus 3.118 s at 14 threads,
while process CPU increases from 12.02 to 13.45 s. A single default-four-thread run
with all CPUs available takes 3.702 s, versus 4.157 s when pinned to four physical
cores. More threads reduce elapsed latency but do not reduce the total CPU budget.
Map loading for this large map adds roughly 6 s in these fresh processes with an
uncontrolled warm filesystem cache; these figures are not complete ROS startup times.

## Four-core fixed-position suite and rigid tracking repair — 2026-10-19

Fourteen **independent, stationary 30 s recordings** now cover the existing 100 × 100 m reference scene: corridors, rooms, furniture types, doorways and different headings. Locations were frozen before evaluating localization. Each closed bag contains 300 raw MID360 scans, 6,000 IMU samples, 300 observer-only truth poses and 296 exact LIO cloud/odometry pairs. The 14 bags retain 4,200 raw scans, 84,000 IMU samples and 4,144 replay frames. The simulator, adapter, installed FAST_LIO and map were unchanged. This is input acquisition coverage, not 14 successful localization tests.

Dataset: `map/sprobo_20261018/fixed14_20261019_v2/`; each position contains `bag/`, `manifest/frames.csv`, point-cloud hashes, session configuration, sensor counts and frontend audit. The fixed plan is [`tools/fixtures/sprobo_fixed14.json`](../../../tools/fixtures/sprobo_fixed14.json). The dataset also contains `positions.png`. Persistent evidence is under the workspace; only transient simulator IPC uses a short temporary path to stay within the Unix socket name limit. The failed initial long-path acquisition remains preserved.

Per Xian, every measured replay uses **only four physical P cores of the i5-14600KF**, affinity `0,2,4,6`, four configured registration threads and `OMP_WAIT_POLICY=PASSIVE`. A/B variants run serially in alternating order. No all-core performance run started. Every pair is replayed at its original 10 Hz due time without dropping frames; overload accumulates backlog. CPU/input-second therefore measures required work per sensor second, not simultaneous occupied cores. Timings exclude map loading, ROS transport and the LIO/simulator process. Loading is separately recorded.

The frozen pre-repair A/B compares `2c4f1ad` with `4982862`: total CPU **2893.95 → 1255.40 s** (56.6% lower), complete alignments **8074 → 2812**. Median first-correct-lock wall time across the 11 intended-location cases is **6.306 → 5.097 s**. This is one paired run per position, not a multi-repeat timing median. Wrong/no-lock positions are excluded from that latency statistic and retained as failures below. These reductions mostly measure the preceding candidate/confirmation optimization, not the new normalization fix.

The new meeting-room input exposed an independent tracking defect, repaired in `9921dba`. Repeated conversion between rotation matrices and non-unit quaternion slerp outputs amplified roundoff until `RealtimeOutput` rejected a nonrigid transform. The existing 1e-6 rigid-pose guard remains unchanged. Tracking now normalizes both input quaternions and the interpolated output; the other correction-interpolation call also normalizes its output. No retrieval or acceptance threshold changed.

An output-aware repeat of `4982862` locked 4 times, produced 23 invalid-rigid frames and 60 odometry-coasted outputs. The normalized core locked **1 time**, with **0 invalid-rigid frames and 0 coasted outputs** through all 296 inputs. CPU **102.98 → 28.26 s**, alignments **306 → 88**; first lock remains frame index 14. Maximum emitted-pose position error **0.06798 → 0.06511 m**. Earlier runs had varying repeat-lock counts because this instability amplifies small numerical differences; their individual records are preserved.

A new end-to-end episode with fresh noisy stationary scans fails against the old frozen library at tracking index 51 (`invalid_nonrigid_pose`) and passes 200 consecutive updates with the normalized library, preserving the lock epoch and rigid output. Humble **61/61** checks passed; default, Atlas and experimental kill configurations each passed all **11** localization episodes. The final normalized library was also replayed across all 14 complete bags.

| Position | First lock after map load | Maximum emitted position error | Total CPU / 29.5 s input | Outcome |
| --- | ---: | ---: | ---: | --- |
| 01_west_corridor | 4.494 s | 0.0119 m | 14.75 s | intended location |
| 02_east_corridor | 10.669 s | 0.3987 m | 38.16 s | intended location |
| 03_south_corridor | 5.510 s | 0.1505 m | 19.32 s | intended location |
| 04_north_corridor | 5.326 s | 0.2941 m | 20.16 s | intended location |
| 05_southwest_office | 4.772 s | 0.0635 m | 16.86 s | intended location |
| 06_south_meeting | 8.340 s | 0.0651 m | 28.26 s | intended location |
| 07_south_pillars | 4.586 s | 0.2109 m | 17.50 s | intended location |
| 08_west_shelves | 3.752 s | 0.0799 m | 18.49 s | intended location |
| 09_east_shelves | — | — | 262.03 s | **NO LOCK** |
| 10_northwest_meeting | 5.470 s | 0.0894 m | 19.24 s | intended location |
| 11_northeast_pillars | 4.500 s | 22.6209 m | 20.83 s | **WRONG ROOM** |
| 12_north_office | 5.701 s | 0.3019 m | 20.92 s | intended location |
| 13_south_door | 5.234 s | 0.2427 m | 19.10 s | intended location |
| 14_north_door | — | — | 664.27 s | **NO LOCK** |

Final outcome: **11 intended-location locks, one wrong-room lock and two failures to lock**. The 1 m / 5° offline audit band separates gross wrong locks; it is not a new product threshold or an accuracy acceptance claim. Correct-position first locks range **3.752–10.669 s**, median **5.326 s** after map loading. Their maximum emitted position error is **0.3987 m**, including pre-existing mapping distortion. There is no demonstrated suite-wide accuracy improvement. All query scores use the same first mapping pose alignment; query estimates are never fitted to truth.

Position 11 is a pre-existing failure in both old versions and the normalized core: about 22.6 m away in a similar room. Detailed first-five-frame diagnostics show that nearby keyframe 759 enters retrieval on query 3, but confirmation only updates hypotheses seeded earlier; the new region is not geometrically introduced before the query-5 lock to 818/819. This identifies a candidate-admission gap, not proof that adding the candidate alone will resolve the ambiguity. The subsequent [incremental-admission stage](#incremental-admission-of-late-regions--2026-10-19) below repairs and verifies this gap; the table above retains the earlier baseline results.

A separate **offline oracle diagnostic**, excluded from success and speed metrics, starts the first five queries at their truth pose. All five registrations converge near keyframe 758 with production registration quality, visibility consistency 0.921–0.958 and final position errors 0.471–0.474 m. A geometrically better nearby solution therefore exists in this map; the remaining map/local-fit error is substantial. This does not establish recovery from descriptor seeds or passing the final temporal/directional lock checks. No truth pose was given to any measured localization run. Raw attempts are retained in `build/n3mapping/fixed14_20261019/11_oracle_diagnostic/`.

Positions 09 and 14 still consume far more than the half-core budget (14.75 CPU s for 29.5 s of input). Descriptor generation/index/retrieval totals only about 1.3–1.5 CPU s in each difficult run; reducing nominal RHPD dimensions would not address the dominant geometry cost. The descriptor remains 2,973 place + 8,400 heading logical values, sparse in storage; no separate Scan Context pass was added.

All 11 intended-location cases remain tracked throughout their final approximately 10 input seconds: whole-process CPU differences over that interval correspond to **0.219–0.377 cores**. This excludes initial search and frontend/DDS costs and does not establish half-core operation for the complete episode. Peak replay RSS is **1,915 MiB** (about 1.87 GiB); no memory reduction is claimed.

The replay probe now records the actual `RealtimeOutput` pose, including odometry-coasted outputs. Older traces with only a failed backend pose cannot score a coasted output and are explicitly marked unscorable, rather than silently substituting the wrong transform. The meeting repeats use the new probe and retain their own timing/accuracy records; they are not spliced into the original run. Detailed debug logging is explicitly disabled by the performance driver.

Reproduce acquisition on the authorized host with the ROS workspace sourced:

```bash
python3 src/n3mapping/tools/n3mapping_sprobo_fixed_suite.py \
  --workspace /home/user/ros_ws/to_migrate_ws \
  --sim-root /home/user/Projects/simple_rbt \
  --sim-executable /home/user/Projects/simple_rbt/build/sensor-consistency/simple-rbt-sim \
  --adapter /home/user/ros_ws/to_migrate_ws/build/n3m_sprobo_adapter/simple-rbt-ros-session \
  --output /home/user/ros_ws/to_migrate_ws/src/n3mapping/map/sprobo_20261018/fixed14_repeat
```

Replay with `tools/n3mapping_sprobo_fixed_eval.py --help`: supply the dataset, mapping run, workspace and one or more frozen library/probe directories; defaults are `--threads 4 --cpus 0,2,4,6 --rate 1`. It refuses to overwrite an output directory. Map load and probe hashes accompany every report. Use matching headers/library ABI when rebuilding a frozen probe.

Compact results: [`fixed14_20261019.json`](../fixed14_20261019.json). Full inputs, frozen binaries, logs, stage timings, source fingerprints and repeat results: `build/n3mapping/fixed14_20261019/`. The old-vs-new E2E failure is retained there as well. This stage does not qualify moving/out-of-map operation, dynamic scenes, RK3588, difficult-search half-core CPU, or all-scene reliable relocalization. Installed binaries and remote branches were not changed.

## Incremental admission of late regions — 2026-10-19

The candidate-admission gap identified by the fixed-position suite is now repaired against `9dcc4ce`. Every observation checks the existing spatial seed plan for newly supported regions. Regions already covered by a surviving hypothesis's seed anchor or propagated pose retain their hypotheses and evidence; only uncovered regions enter seed evaluation. The original basin radius, shortlist, yaw proposals and numerical acceptance gates remain unchanged. A failed region can still be retried through the existing seed cache; preservation does not mean retaining invalid geometry forever.

Each appended hypothesis has its own first observation and odometry origin. Admission already evaluates that scan, so confirmation cannot count the same scan twice. The final registration schedule, temporal window and motion-dependent visibility gate use the hypothesis's own history. Returning to the original window origin therefore cannot make a late, moving hypothesis appear stationary. Final publication still requires current converged registration, directional support and the existing free-space decision. Normal cost logs now distinguish admitted regions/modes and preserved regions; detailed logs expose each hypothesis's observation age.

The final core (`2e8cfbabaacabf7cec097e00fa06cb0eab6e06ca8d010f498241f289e217a7ef`) and baseline (`6d71afa54e133b4099a0c293d945b5a9a31d4e55bac0d3227d8ab72c2077fdf8`) were frozen with matching headers/probes. All 14 unchanged bags were replayed completely: **4,144 frames per variant, 8,288 total**, serial alternating A/B, four physical P cores `0,2,4,6`, four registration threads, passive OpenMP and original 10 Hz due times. Overload accumulates backlog without dropping inputs. Map loading, simulator, frontend and DDS are excluded. Map/atlas hashes were verified again after execution. No query-to-truth fitting or truth input to localization was used.

| Full-suite measure | Baseline | Incremental admission |
| --- | ---: | ---: |
| Intended-location locks | 11 | 12 |
| Wrong-room locks | 1 | 0 |
| Positions without a lock | 2 | 2 |
| Total process CPU | 1183.877 s | 1213.113 s |
| Full registrations | 2583 | 2802 |
| First-lock wall median, same 11 previously correct positions | 5.055 s | 5.640 s |

**This is a reliability repair, not a suite-wide speedup.** This single paired run uses 2.47% more CPU and has a slower paired latency median. Additional competitors must be evaluated; preserving old work does not remove that cost. These are not repeated-run timing statistics.

Position 11 now locks at query **7**, after the true region (seed 759) first appears at query **3**. Maximum emitted position error is **22.6209 → 0.4739 m**; the corrected location still has substantial residual error. At admission it performs **7 new seed solves and preserves two existing regions**. Queries 4–7 perform **zero seed initializations**. Separate diagnostic execution confirms that the new modes have exactly 1, 2, 3, 4 and 5 observations/updates at queries 3–7; those diagnostic timings are excluded. The previous wrong-room query-5 lock is not treated as successful latency.

There are two earlier correct locks: position 02 changes query **10 → 7**, wall **10.862 → 10.417 s**, total CPU **38.954 → 35.854 s**; position 06 changes query **15 → 6**, wall **8.902 → 5.902 s**, total CPU **28.435 → 21.286 s**. All 12 intended-location cases lock once and emit no coasted outputs. Their first correct locks range **4.389–10.417 s** after map loading, with median **5.722 s** over these 12 cases.

Precision is compared at shared timestamps as well as over all emitted outputs. Nine of the previously correct 11 cases retain numerically equivalent output positions (maximum delta below 8e-15 m). Position 02 has up to 15.705 mm output-position difference, but shared-time position RMSE increases only **0.00858 mm**, maximum position error **0.000191 mm**, and maximum rotation error **0.000217°**. Including its three newly emitted early poses raises whole-run maximum position error from **0.398728 to 0.399146 m**; that difference is not hidden. Position 06's shared-time RMSE improves **0.195 mm**, maximum position error improves **0.820 mm**, and maximum rotation error increases **0.000029°**. These measured differences do not establish exact output equivalence or universal/hardware accuracy. The 1 m / 5° reporting band only labels gross wrong-place locks; it is not an accuracy acceptance criterion or a changed product gate.

Positions 09 and 14 still fail to lock. Their CPU costs are **262.587 → 294.226 s** and **667.129 → 657.582 s** for approximately 29.5 s of input, against the half-core budget of **14.75 CPU s**. Descriptor generation/retrieval takes only **1.303 / 1.512 CPU s** in the final runs. Remaining costs are predominantly geometric; neither difficult-search half-core operation nor all-scene recovery is achieved. The 12 tracked cases use **0.217–0.372 cores** over their final approximately 10 input seconds, excluding initial search and frontend/DDS. Final peak RSS is about **1912 MiB**; no memory-reduction claim is made.

The existing changed/repetitive moving negative was also replayed through all **346 frames: zero locks and zero outputs**. This preserves rejection, not successful localization. It uses 693.760 CPU s / 240.282 wall s; this additional run is a functional negative control, not a fresh paired performance comparison. All 61 Humble checks passed, and default/Atlas/experimental-kill modes each passed 12 E2E episodes. The initial ROS live-output check needed the build-directory typesupport library on its test environment path; that failed setup was corrected and the check rerun. A new E2E exercises late-region admission without restarting surviving hypotheses; a motion regression checks the late candidate's own origin. Experimental kill mode may intentionally remove invalid hypotheses, so preservation assertions apply to survivors while confirmation timing and pose checks remain enforced in every mode.

Compact results, shared-time precision, stage counts, validation provenance and source/binary hashes: [`candidate_admission_20261019.json`](../candidate_admission_20261019.json). Full paired traces and separate diagnostics: `build/n3mapping/candidate_admission_20261019/`; final variant is `final`, paired report is `four_core_full_final/results.json`. Use the existing `n3mapping_sprobo_fixed_eval.py` driver with the two frozen variants to reproduce. Missing baseline admission counters are recorded as null, not zero. Full registrations reconcile as seed initializations + seed resumptions + confirmation registrations; `seed_continuations` also includes cheap evaluations and must not be counted as full solves.

Internal C++ state layout changed, so all affected targets were rebuilt together. Installed binaries, ROS interfaces, descriptor dimensions, simulation/FAST_LIO, map data and remote branches were not changed. The next work remains reducing geometric work on positions 09/14 and the moving negative while preserving rejection, and investigating the residual position errors. Moving out-of-map recovery, RK3588 and full live deployment throughput remain outside this validation.

## Exact fused retrieval and corrected heading seeds — 2026-10-19

Against `5ab6fc0`, both remaining fixed-position failures now lock without truth
input. The changes retain the incremental admission/cache/confirmation machinery:
new regions are evaluated without rebuilding surviving regions. Candidate count,
basin limits, registration objectives, temporal confirmation, directional support
and free-space gates retain their existing numerical settings.

### Separating local registration from retrieval

An independent oracle diagnostic starts the first five queries of positions 09 and
14 from the true pose and from two perturbations: `(dx, dy, yaw)` of
`(+0.5 m, +0.5 m, +5 degrees)` and `(-0.5 m, -0.5 m, -5 degrees)`.
All **30/30 actual refined registrations** converge and pass existing registration
quality; none of these results merely keeps the supplied initial pose. Residual
position error is **0.4556–0.4596 m** at 09 and **0.3543–0.3606 m** at 14.
This establishes local convergence from those starts, not temporal confirmation,
global retrieval or centimeter accuracy. Truth is used only by this diagnostic
and the observer/scorer; it never initializes the ordinary measured runs.

Local ICP failure would not by itself prove a descriptor cannot retrieve the
right region. Retrieval, local convergence, map/input consistency and final
publication are distinct checks. Here local geometry is usable, and the retrieval
audit exposes an earlier failure: the first queries' true keyframes 673 and 889
rank **137 and 213** under the coarse key, beyond the old shortlist of 100. Under
complete fused place/heading scoring both rank **1**. Position 09 also ranks 33
under exact place distance, beyond the old place-only top-20/30 cutoff.

Localization now scores the full place descriptors at both submap and single-frame
scales. The scales share one heading per keyframe, so they merge by the smaller
place distance before heading comparison. A nonnegative heading contribution
provides a lower bound: further heading evaluations stop only when they cannot
enter the existing final top ten. This replaces an unsafe heuristic cutoff with
exact fused selection, including deterministic ties. `rhpd_preselect_candidates`
continues to control approximate loop-closure retrieval; it no longer truncates
localization candidates. There is no new configuration flag or descriptor channel.
The place-index implementation shrinks from 332 to 254 lines.

The heading audit also finds a transform-direction error. Polar matching shifts
the reference image into the query image, whereas the pose seed must transform
query points into the reference frame. The seed now uses the inverse angle. The
existing uniform yaw grid is phased to that measured angle, preserving equally
precise opposite-heading competitors with the default four-bin grid. Correcting
the sign alone caused the symmetric-scene E2E to publish an ambiguous heading;
that intermediate version was rejected. The final version again refuses that
scene without changing acceptance gates. The neighboring-sector cache regression
still requires exactly one new full seed registration.

### Complete four-core paired replay

The same 14 frozen bags, map, atlas and input hashes are used, with **4,144 frames
per variant**. Baseline/final processes alternate serially on physical P cores
`0,2,4,6`, four registration threads, `OMP_WAIT_POLICY=PASSIVE`, at the original
10 Hz due times. There is no frame dropping or truth fitting. CPU includes all
process threads after map loading; simulator, frontend and DDS remain excluded.
This is one A/B per position, not repeated-run timing statistics.

| Measure | Baseline | Final |
| --- | ---: | ---: |
| Correct regions / wrong regions / no lock | 12 / 0 / 2 | **14 / 0 / 0** |
| Total CPU, all 14 complete inputs | 1214.097 s | **260.641 s** |
| Full registrations | 2802 | **764** |
| Cold seed solves / resumed solves / confirmation solves | 1487 / 241 / 1074 | **660 / 0 / 104** |
| First-lock wall median, same 12 previously correct cases | 5.699 s | **4.976 s** |
| CPU / total input duration | 2.940 | **0.631** |
| Maximum process RSS | 1912.32 MiB | 1912.36 MiB |

Total CPU falls **78.5%**, full registrations **72.7%**. Every final case locks on
query five, retains one lock epoch, produces 292 outputs and has zero coasted
outputs. First-lock wall time ranges **3.847–7.741 s after map loading**. Resolving
failed searches accounts for much of the total gain: individual CPU rises by
**10.8% at 03** and **4.9% at 07/08**, so this is not a universal per-case speedup.

| Previously failing position | Baseline CPU / lock | Final first-lock wall | Final CPU / 29.5 s input | Mean cores | Maximum emitted position error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09 east shelves | 294.757 s / none | 3.854 s | 13.157 s | **0.446** | 0.4619 m |
| 14 north door | 658.339 s / none | 7.741 s | 25.682 s | **0.871** | 0.3681 m |

Position 09 meets the **14.75 CPU s** whole-episode half-core budget; position 14
does not. Across all 14 cases the final approximately ten tracked input seconds
use **0.217–0.368 cores**, which excludes initial search. Extending a stationary
recording to dilute startup cost would not establish cheaper difficult search.
The complete-suite average is still above half a core, and no RAM reduction is
demonstrated.

### Precision, negative coverage and rejected ablations

The same fixed mapping-session alignment scores actual RealtimeOutput poses.
On shared timestamps, whole-run maximum position errors remain numerically
equivalent or improve for the previously correct cases. Position RMSE at 13
increases **0.359 mm**, with up to **12.586 mm** change in an emitted position;
its maximum position error is equivalent and maximum rotation error improves
0.02216 degrees. Some rotational errors increase: maximum rotation error at 04
increases **0.001279 degrees**, and rotation RMSE at 11 increases **0.000154
degrees**. These are recorded differences, not exact output equivalence or a
claim that every precision metric improves. Full errors and common-time results
are retained in the linked JSON. The suite's largest position error is **0.4739
m**, at 11. The 1 m / 5 degree offline band only distinguishes gross wrong-place
locks; it is not an accuracy acceptance criterion or a changed product gate.

The changed/repetitive moving negative completes **346 frames with zero locks and
zero outputs**. It still costs **727.271 CPU s**, 263.206 wall s and 1678 full
registrations for 34.5 s of input. This is safe refusal, not successful recovery,
and is far above the half-core budget. It is a functional control, without a
fresh paired timing baseline; the previous stage's 693.760 CPU s must not be
presented as a new paired performance comparison.

RHPD remains **2973 place + 8400 heading = 11373 logical values**, stored sparsely;
heading is already fused and no separate SC pass is added. Complete retrieval
increases per-query work, but the positive suite now stops searching much sooner:
descriptor calling-thread CPU totals **7.923 → 7.660 s** across all 14 runs. On the
full negative it costs **14.143 CPU s** (about 1.9% of total process CPU), so
geometric work still dominates. Descriptor size reduction alone cannot eliminate
the observed remaining cost.

Exact retrieval alone recovered all 14 cases in an exploratory full run, but
required 333.031 CPU s and 916 registrations. This was not the final paired A/B.
Two further cost experiments were excluded: using one registration thread saved
little CPU while increasing 02/09/14 first-lock wall from 4.554/4.350/8.031 s to
7.025/5.677/15.831 s; a mathematically certified nearest-neighbor cache passed
correctness checks but increased CPU on 02 and 14. Its implementation and added
tests were reverted. Neither experiment is counted as a delivered improvement.

Final Humble validation passes **61/61** checks. Default, Atlas and experimental
kill configurations each pass all **12 E2E episodes**, including opposite-heading
ambiguity and late-region admission. Both new targeted regressions fail against
the frozen baseline and pass against the final library. The final runtime is
byte-identical to the three-mode/negative-tested library:
`26762b2d8197d85bb3b457db0518e1106d68dc7c110301ecfc9853f9664a7458`.
Source, frozen library, map and atlas fingerprints were rechecked after the full
paired run.

Evidence: [`hard_regions_20261019.json`](../hard_regions_20261019.json).
Raw traces, independent diagnostics, rejected variants and validation logs remain
in `build/n3mapping/hard_regions_20261019/`; final paired report is
`four_core_final/results.json`. Reproduce with the existing
`tools/n3mapping_sprobo_fixed_eval.py`, the frozen `baseline` and `final` directories,
and the same `--threads 4 --cpus 0,2,4,6 --rate 1` settings. Further work remains on
negative/new-candidate geometric cost, residual map/local-fit error and moving
out-of-map/re-entry coverage. RK3588 and full live deployment throughput are not
qualified. Installed binaries, simulator/FAST_LIO and remote branches are unchanged.

## Exact correspondence bounds and map-error diagnosis — 2026-10-19

This stage starts from `c3dcc96`, which already recovers all 14 fixed locations.
It targets the remaining cost at 14 and in the changed/repetitive moving negative.
Only `src/registration_search.h` changes in production: nearest-neighbor traversal
carries the squared distances to the current cell along its three axes. Their
sum bounds every point in that cell, allowing branches outside the current
nearest-neighbor sphere to be skipped. The original near-first traversal, tie
order and distance gate remain; a conservative rounding guard retains boundary
branches. This changes neither the registration objective nor the candidate,
confirmation or acceptance policies. It adds no persistent cache or setting.

An additional per-factor leaf-cell cache passed correspondence and E2E checks but
increased CPU in both focused initial-search trials: 09 from 7.341 to 8.127 s,
14 from 17.622 to 20.811 s. That implementation and its added test were reverted.
Its unfinished negative-prefix baseline is excluded. The retained implementation
uses the existing exact-neighbor, tie, split-boundary and optimizer tests.

### Paired four-core cost and behavior

The complete 09/14 inputs and complete moving negative use the unchanged frozen
map, atlas and input hashes, serial A/B processes, physical P cores `0,2,4,6`, four
registration threads and `OMP_WAIT_POLICY=PASSIVE`. Every input is processed at
the original 10 Hz due time or later if the process falls behind. CPU covers all
process threads after map loading; frontend, simulator and DDS are excluded.
These are single complete pairs, not repeated-run timing statistics.

| Complete replay | Baseline CPU | Final CPU | CPU reduction | Final CPU / input duration |
| --- | ---: | ---: | ---: | ---: |
| 09, 296 frames / 29.5 s | 13.220 s | 13.007 s | 1.6% | 0.441 |
| 14, 296 frames / 29.5 s | 25.547 s | 24.796 s | 2.9% | 0.841 |
| Moving repetitive negative, 346 frames / 34.5 s | 726.944 s | 655.076 s | 9.9% | 18.988 |

The last ratio expresses how much CPU would be needed per input second to keep
up, **not actual simultaneous core usage**. That negative takes 263.546 → 234.649
wall seconds on the same four cores. Its half-core budget is 17.25 CPU s, so it
still falls far short of the target. At 14 the budget is 14.75 CPU s, also unmet.
09 remains below half a core; the small positive-case gains should not be read
as a universal or statistically established speedup.

First-lock wall time at 09 changes 3.979 → 3.734 s and at 14 changes
7.308 → 7.270 s. Both still lock on query five and emit all 292 subsequent poses.
Their full-registration counts stay 39 and 63. The negative retains zero locks,
zero outputs, 1678 full registrations, 1549 confirmation evaluations and the same
207 admitted / 734 preserved basin events. Its cold/resumed/confirmation solves
remain 1015/82/581. Thus the retained saving is inside the geometric computation,
not fewer safety checks or a change in candidate admission. No persistent-memory
reduction is claimed.

All 14 positions additionally receive a paired **first-30-frame** check, 420
frames per variant. Each retains query-five locking, 26 outputs and zero coasted
outputs. Prefix CPU totals 169.663 → 158.978 s (**6.3% less**); first-lock wall
median across positions changes 5.188 → 4.774 s. The per-case results are retained:
01 increases CPU 1.4% and first-lock wall 9.0%, whereas the other 13 use less CPU
in this single pair. Prefix results do not establish full-suite steady-state CPU
or a new complete-suite half-core measurement.

Across full 09/14, the negative and all prefixes, every compared state, seed,
support, decision, candidate and alignment-count field agrees. Maximum output
matrix component differences are `4.44e-16` for full 09/14 and `1.78e-15` for the
prefixes. Full 09/14 maximum position errors remain exactly 0.461898/0.368111 m
at reported precision. This supports no observed loss of accuracy or reliability
on these inputs; no-output negatives have no pose-accuracy score.

Humble validation passes **61/61** checks; default, Atlas and experimental kill
modes each pass **12 E2E episodes**. Source and map fingerprints match the frozen
measurements. Final core SHA-256 is
`3d933744b8a43bf3dfd180275edfac52fb800d0d34f7e8a11b0e2e5c3a9aadb3`;
baseline core is
`26762b2d8197d85bb3b457db0518e1106d68dc7c110301ecfc9853f9664a7458`.

### Separating map distortion from local fit

Xian's concern about the 0.3–0.4 m errors is valid: recovering the correct region
does not qualify the output for navigation accuracy. The existing acquisition
guard of 1 m / 5 degrees only rejects gross mapping drift. It never established a
10 cm map or localization guarantee.

An independent extraction of all **1271 saved optimized keyframe poses**, compared
with mapping-session truth using the unchanged first-pose alignment, finds
**0.2866 m position RMSE and 0.7205 m maximum error**. Even a diagnostic rigid fit
using only those mapping keyframes leaves 0.1938 m RMSE and 0.5243 m maximum
residual; a single global frame offset cannot explain the distortion. That fit
is not used for any published localization score.

For each previously measured output, a separate diagnostic transports its pose
through the mapping-truth pose of the **support keyframe actually reported by
localization**. It does not choose a keyframe using query truth. In notation, the
diagnostic pose is `T_truth_support * inverse(T_map_support) * T_map_query`.
This assumes the neighborhood can be approximated by a local rigid correction;
it is evidence about map bias, not an exact decomposition of every error source.

| Position | Actual output maximum error | Diagnostic residual after support-map truth transport |
| --- | ---: | ---: |
| 09 east shelves | 0.4619 m | 0.0148 m |
| 11 northeast pillars | 0.4739 m | 0.0109 m |
| 14 north door | 0.3681 m | 0.0169 m |

The error vectors also agree closely with those of the saved support poses.
Across all 14 positions this diagnostic maximum ranges from 0.0091 to 0.0516 m.
These counterfactual numbers are **not runtime-corrected output**, navigation
qualification or proof that every remaining error comes from the map. Together
they strongly support a dominant map contribution at 09/11/14. No truth enters
normal localization and no map, scoring alignment or threshold is changed.
At Xian's request, map correction is deferred while search-cost work continues.

### Reusing relocalization work for loop closure

The current loop pipeline already shares heading-seed generation and the
registration implementation. The exact complete fused retrieval introduced in
the previous stage currently applies to localization; approximate loop retrieval
still has its earlier shortlist. Extending that recall improvement is a concrete
next experiment, not a demonstrated map-consistency improvement in this stage.

The intended path is to retrieve old regions, estimate independent relative
measurements from source/target scan geometry, verify competing hypotheses and
neighborhood consistency, and submit accepted relative constraints to the
existing pose graph. A localization pose derived from the current map must not
be inserted as if it were a new independent observation: that would reuse the
map's own bias. Repeated constraints derived from the same scans also need to
retain their provenance rather than be treated as independent confirmations.
Map improvement requires its own before/after trajectory and overlap checks,
including false-loop controls in repeated regions. No new loop-recall or map
optimization behavior is delivered or claimed here.

Evidence: [`search_accuracy_20261019.json`](../search_accuracy_20261019.json).
Frozen variants, complete traces, map-pose extraction, independent diagnostic
scripts and rejected-cache evidence remain in
`build/n3mapping/search_accuracy_20261019/`. `validate_final.py` records the test
and replay commands; `summarize.py` checks identities and behavior comparisons.
No issue workspace, installation, push or simulator/FAST_LIO change is involved.
The next search-cost work still needs to reduce repeated/new-candidate geometry
while preserving fresh evidence and recovery. Map-consistency work is deferred
as requested; RK3588 and full live deployment throughput remain unqualified.
