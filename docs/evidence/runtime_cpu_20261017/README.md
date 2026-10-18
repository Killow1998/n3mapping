# Runtime CPU and relocalization measurements — 2026-10-17

Baseline: `45f4a171fcab1d9742ab7d575f41a632c7d75073`, branch
`dev/n3mapping-realtime-pipeline`. The Product V1 configuration remains unchanged
(SHA-256 `1b603a0a180330857aac150e0450cb5e94e284a84934e9b6ee387000f3f254d4`).
These are Release/Humble development builds with **UNVERIFIED** build identity;
they do not constitute a Product Bundle or hardware qualification.

## Measurement boundaries

CPU is process CPU time divided by elapsed time: 1.0 means one fully occupied
logical CPU. All threads are included. Map loading precedes the measurement;
initial relocalization is included. First-path latency starts with the first
input, after map loading. RSS is the process high-water mark, not live retained
memory. No CPU quota, input reduction, frequency reduction, or acceptance changes
were used. OpenMP has four configured threads and the existing launch default
`OMP_WAIT_POLICY=PASSIVE`; an isolated comparison also tests the unset policy.

The ROS probe puts the production node, replay publisher, and output subscribers
in one process. DDS subscriptions, exact message synchronization, the normal
backend loop, both live callback workers, cloud conversion, TF/output publication,
and complete path publication all execute. CPU includes the replay and subscriber
helpers as well. Fast DDS same-process communication works in this environment;
UDP/interface access is restricted. **Cross-process DDS, physical MID360 input,
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

This stage compares against `682f5af6b7848d7959d1e9e66f32e313a5ce8ce6`.
**The RHPD/SC fusion prototype was rejected; the default still uses SC.**
The existing 3037-value RHPD, map format, Product configuration and acceptance
thresholds remain in place.

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
The prototype and its map-format change are absent from the shipped source.

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
Relocalization latency and repeated initialization remain unresolved; SC removal
has not qualified for the default path.

## Reproduce

Build the normal Release/Humble workspace first with `BUILD_TESTING=ON`; research
tools need not be on.
Build the evidence probes against its generated include/link configuration:

```bash
python3 docs/evidence/runtime_cpu_20261017/build_probes.py \
  /path/to/workspace/build/n3mapping /tmp/n3m_probes
```

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
loaded SC descriptor. Compare `(query, keyframe, distance, shift)` rows exactly;
`MATCH_CPU_MS` excludes generation, loading and CSV output. The recorded corpus
uses observations 1, 50 and 100 from each of the five recordings.
