# Prepared place retrieval — 2026-10-26

Runtime revision: **`ab8e92a0b308558ab8d958acb5c369e0e2e71ee1`**.
Baseline runtime: `5169d62`; starting repository head: `8d7a60e`.
This follows the [backend repair](large_mapping_followup_20261026.md).
The change is local. Navigation installations, previous qualified packages and
rc1/rc2/rc3 tags remain unchanged. No remote publication or deployment occurred.

The place catalog is now compiled with the completed map's Atlas. Localization
loads that catalog once. This corrects the previous prototype's partial move of
preparation: it grouped viewpoints early but still reduced ranked keyframe
matches into places during each query. Query-time spatial grouping and that
raw-keyframe reduction are removed. Query-dependent descriptor and direction
comparisons remain necessary.

The original14 and two complete localization routes preserve lock events,
output counts and accuracy. Exact duplicate observations benefit. The noisy
repeat control does **not** pass the speed objective, and difficult-search
half-core operation is **not** accepted by this stage.

## Ownership and invalidation

- `compileRelocalizationPlaceCatalog` groups views within 0.25 m in 3D and one
  configured heading sector in orientation (6 degrees in these runs). Different
  floors and opposite headings retain separate retrieval identities. This is
  conservative viewpoint grouping, not an entire room represented by one vector.
- Equal raw-frame, causal-submap and heading features share their lowest-ID
  representative. Differing observations remain as separate views inside the
  place. Original keyframes and geometry are never deleted or averaged.
- `LocalizationAtlas::compileAndSave` builds the frame descriptor cache and
  place catalog from final map geometry. The existing Atlas compiler and save
  pipeline use this preparation; no additional save trigger is introduced.
- Atlas loading validates external cache data and its configuration signature.
  `WorldLocalizing` verifies referenced keyframe IDs and installs the index before
  the first query. The existing map SHA and geometry configuration checks remain.
  Old caches or changed descriptor settings rebuild from source clouds once at
  load. There is no second legacy descriptor algorithm.
- An unchanged map revision returns immediately from index preparation. A map
  generation, structure or pose revision invalidates the appropriate prepared
  state. Structure changes currently recompute frame descriptors when no matching
  cache is supplied; append-only incremental maintenance is not optimized here.
- Queries rank fixed places using their retained views. Exact RHPD and heading
  scores select the view. A nonnegative heading-distance bound avoids evaluations
  that cannot improve the selected places. More distinct places can enter the
  geometry stage; this explains why some cases do more registrations.

No registration solver, acceptance threshold or confirmation count is changed.
No new worker, mutex or speculative safety layer is added. The causal-submap
cloud routine is shared by map loading and Atlas compilation so both construct
the same descriptors. RHPD caches use immutable shared descriptors. The old
coarse-key scan and sort had no pruning benefit on this input and are removed.

## Qualification and measurement

Evidence root, relative to the ROS workspace:
`build/n3mapping/place_index_20261026/`.
`qualification.json` joins the individual results; `build_manifest.json` records
the source, package and executable identities. Raw failures and ablations remain.

Release uses ccache, research tools OFF, CPU affinity **0,2,4,6**, four algorithm
workers, `OMP_WAIT_POLICY=PASSIVE` and one OpenBLAS worker. CTest runs one test
process at a time. The final build passes **67/67 tests in 42.10 s**, including
the existing real-node E2Es. New behavioral coverage checks exact fused ranking,
distinct floors/headings, cache roundtrip/configuration invalidation, malformed
external catalogs and no index rebuild during unchanged-map queries.

Full accuracy comparisons replay frozen frontend clouds and odometry as fast as
possible (`rate=0`); they are not new live DDS or hardware acquisitions. Original14
alternates baseline/candidate order. Host frequency and competing load are not
held constant. Truth is an observer, using the mapping run's first-pose alignment;
query trajectories are not fitted to truth.

### Original14 and two routes

| Original14 aggregate | Baseline | Prepared places |
| --- | ---: | ---: |
| Locks | 14/14, frame 4 each | 14/14, frame 4 each |
| Published poses | 4,088 | 4,088 |
| Maximum position error | 2.481 cm | 2.481 cm |
| Maximum rotation error | 0.08384 degrees | 0.08384 degrees |
| Mean first-lock computation, excluding load | 2.793 s | 2.362 s |
| Mean map load | 4.874 s | 4.891 s |
| Mean load plus first lock | 7.668 s | 7.253 s |
| Total CPU time across recordings, excluding load | 166.885 s | 158.578 s |
| CPU / recorded input duration | 0.404 cores | 0.384 cores |

There are no query-time index rebuilds in any of the final14 cost traces. The
1,271-keyframe map contains 1,029 prepared viewpoint places with 1,271 retained
views; actual observations in this map do not have exactly equal features.

| Selected position | First lock, old→new | First-lock CPU, old→new | Geometric alignments, old→new |
| --- | ---: | ---: | ---: |
| 03 | 3.822→3.246 s | 8.983→8.027 s | 45→41 |
| 08 | 3.243→2.466 s | 7.201→5.407 s | 52→43 |
| 13 | 2.993→3.081 s | 6.812→7.971 s | 34→44 |
| 14 | 3.219→3.033 s | 7.156→7.459 s | 35→39 |

Position 13 is slower; positions 13/14 use more first-lock CPU. These regressions
remain visible rather than being averaged away. Their final accuracy is unchanged.

| Complete route | Outputs old/new | Lock frames old/new | Maximum position error old/new | CPU old→new |
| --- | ---: | --- | ---: | ---: |
| `reentry_20261019/query` | 2,232 / 2,232 | 4 / 4 | 7.933 / 7.933 cm | 53.485→52.803 s |
| `independent_v2` | 3,662 / 3,662 | 4,1938 / 4,1938 | 3.247 / 3.247 cm | 121.119→117.081 s |

Rotation maxima remain 0.14150 and 0.26417 degrees. Decision/source counts also
match. The second route retains 59 published poses without backend success;
this stage does not reinterpret those as new successful registrations.

**CPU denominator matters.** The reported 0.384 cores divides CPU time by the
whole recorded input duration, excluding load. It is not average CPU during
active difficult search. Position 03's first-lock computation still uses about
2.47 cores while active. Its whole-recording result is 0.496 cores in the main
matrix but about 0.526 in a later repeat. There is insufficient margin for a
general half-core acceptance claim.

### Repeated-observation ablation

The probe expands the recorded simulation map to 1×/2×/4× observations, adding 3 cm of X pose
jitter per copy while freezing the original causal submaps. It queries the first
cloud from each of the 14 positions, three times, averaging the last two searches.
This isolates retrieval scaling; it is not newly acquired repeated-route mapping.
Preparation is outside the query and separately reported in the JSON.

| Input | Raw keyframes | Retained views | Old query | New query |
| --- | ---: | ---: | ---: | ---: |
| Original | 1,271 | 1,271 | 27.586 ms | 28.100 ms |
| Exact observations, 2× | 2,542 | 1,282 | 33.766 ms | 28.364 ms |
| Exact observations, 4× | 5,084 | 1,314 | 45.096 ms | 29.343 ms |
| 4× with independent 1 cm XYZ noise | 5,084 | 5,084 | 53.214 ms | **84.289 ms** |

Noise uses fixed seed 20261026, Gaussian sigma 0.01 m on copied raw and submap
clouds. The new index retains these different observations. At 4× the old ten raw
candidates cover about three original IDs; the new ten places cover about ten.
Thus both methods do not offer equal competitor coverage. The noise result
exposes a real unresolved cost, not evidence to discard different views without
checking reliability. Exact-copy raw cloud pointers are shared; its RSS is not a
prediction for a physically recorded map with four times as many stored clouds.

An earlier spatial bounding-box index pruned no regions on this map and slowed
4× queries from 44.96 to 79.76 ms. That complexity was removed. An intermediate
load-time compaction prototype was replaced by the persisted catalog. These
ablations are retained under `candidate01`, `candidate02` and `bounded_prototype`;
they are not the qualified package.

### Cache cost, geometry equality and RAM

Atlas compilation adds **3.219 s** for frame/submap descriptors and the catalog
on this map. The sidecar grows from 614,043,150 to 625,350,425 bytes: **+10.78 MiB**.
Runtime decoding takes about 4 ms and catalog installation about 1 ms in position 03.
Global point order is identical. Prepared target ordering differs, but matching
by XYZ gives zero differences in points, normals and covariances for all three
levels. The benchmark does not silently change the reference geometry.

A position 03 control separates precomputation from merely shifting work to load:

| Core and cache | Load | First lock | Peak RSS |
| --- | ---: | ---: | ---: |
| Old core, old Atlas | 5.008 s | 4.326 s | 1,276.34 MiB |
| New core, old Atlas; prepare once at load | 5.447 s | 3.765 s | 1,300.57 MiB |
| New core, compiled catalog | 4.929 s | 3.743 s | 1,300.73 MiB |

This control adds about 24.4 MiB peak RSS over the original package. The main
matrix happens to show 1,300.27→1,277.32 MiB, so there is no demonstrated stable
RAM reduction. Report the control as a small memory increase and preserve the
run-to-run difference. No cause for that resource variation is proven.

## Build identity and reuse

The independent package is
`build/n3mapping/place_index_20261026/verified_ab8e92a_package`.
Its identity reports the full runtime commit, Release, research tools OFF and
`verified=true`. Product profile SHA256:
`96da2443dd1dbd19847d263b424ea77dc36808791c96a576cead67ab07ea2e01`.

| Artifact | SHA256 |
| --- | --- |
| Installed core | `6123e2e5aaac4b7bd451798ca985cb46c3081c3753f48595fc94b812e931e8af` |
| Humble node | `15f7c53da3ff3b530a5e635d327aa9274e5fda46edaf78c0771e7571dd96e9ab` |
| Matched repaired GTSAM | `84be7fb16cd93b8bf782674eae5e161489f08b3c5b964c14b29b632a9f65110b` |
| Compiled fixture Atlas | `edb204e8fb8d0485a06e3118e5af54d11e036e4dcd285a3dff2e20fb15efb138` |

Build-tree and installed core hashes differ after installation, but `.text` and
`.rodata` are byte-identical. A complete position 03 smoke replay against the
packaged core returns 292 poses, locks at frame 4, and retains the 9.946 mm maximum.
Dependency and wrapper hashes are recorded in `build_manifest.json`.
`protected_check.json` verifies unchanged frozen libraries and existing tags.

The existing compiler interface is unchanged:

```bash
n3mapping_localization_atlas_compile --map MAP.pbstream --output NEW_ATLAS.pb
```

Compile a separate sidecar when comparing against a frozen reference. Do not
overwrite the navigation thread's map or libraries. Replay commands and scoring
are retained in `run_validation.py`, `post_validation.py` and `build_probes.py`
under the evidence root; their existing output paths are the completed evidence,
so use fresh output directories for another run. Final results are in
`fixed14_final_ab8e92a/results.json`, `motion_*_final_ab8e92a_03/score.json`,
`repeated_*_03.json`, `repeated_noise_4_*.json` and `cache_controls/results.json`.

## Remaining work

The next optimization must address differing repeated observations inside a
precompiled place without removing a geometrically useful competing hypothesis.
Use the noisy-repeat failure alongside genuinely repeated-route inputs, and
measure descriptor, heading and geometry work separately. Positions 13/14 remain
regression cases. The stage does not qualify arbitrary repeated routes, larger
maps beyond the existing 1 GiB Atlas contract, Noetic Docker, RK3588 hardware,
general 10 cm accuracy or continuous difficult search at half a core.
