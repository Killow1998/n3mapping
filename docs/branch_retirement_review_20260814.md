# Branch Retirement Review — 2026-08-14

This review records the evidence mined from retired development lines before
their temporary preservation tags are deleted. It is intentionally a decision
ledger, not an instruction to merge old branches.

## Final ref policy

The remote is intended to retain exactly one long-lived branch:

- `main`: the canonical product and maintenance line.

The only permanent archive tags are:

- `archive/humble` -> `6327519ee3347a15016f4ebd6e9942c5f26b7d40`
- `archive/noetic` -> `32a3b131914b085624add24d79be84a54b9a0784`

All `archive/retired-20260814/*` tags were temporary review refs. Their peeled
targets and decisions are recorded below; after this ledger was pushed, the
tags were removed from both local and remote ref lists.

## FA-03 branch unification and remote recovery

The former long-lived branch tips were:

| Former line | Recorded tip |
| --- | --- |
| pre-unification main | `784d8c1ca25f92e69f680cc6a4ec411bf054e586` |
| realtime | `7b814f6e8f0cb850d8381ab4bcc622b965dfb67a` |
| recovery before final acceptance | `13c3aff4558cc10af64548b17acd1b0d587e79c3` |
| unified accepted source | `8f11ea22e4df506519dbeca6d833a29870f0c8b9` |

Every unique main/realtime change was semantically reviewed before history was
joined. Commit `98ace70ac69c96d19083257ce526ccf311b6c367` retained main ancestry and
commit `de8988dee92aec17363f9862833dd1e3219553b6` retained realtime ancestry,
both using the `ours` strategy. Their result tree was exactly
`8905d965ee1e3d6c1e5f7857f2c618513014045c`, the audited recovery tree.

After canonical main passed local and remote acceptance, the recovery and
realtime branch names became redundant. Deleting those names does not delete
their commits: all three recorded tips are ancestors of remote main and are
therefore recoverable from any fresh clone.

Exact remote recovery commands:

```bash
git fetch origin --tags
git switch -c recover/pre-unification-main 784d8c1ca25f92e69f680cc6a4ec411bf054e586
git switch -c recover/realtime 7b814f6e8f0cb850d8381ab4bcc622b965dfb67a
git switch -c recover/recovery-pre-acceptance 13c3aff4558cc10af64548b17acd1b0d587e79c3
git switch -c recover/unified-accepted 8f11ea22e4df506519dbeca6d833a29870f0c8b9
```

The local stash named
`deepseek WP-04C partial takeover 2026-08-08 Asia/Shanghai` is deliberately
excluded from remote recovery. It contains only an early incomplete change to
mapping resuming and its tests, and the takeover contract explicitly forbids
restoring or committing it.

## Reviewed temporary refs

| Temporary tag suffix | Peeled target | Review decision |
| --- | --- | --- |
| `archive/humble` | `6327519ee3347a15016f4ebd6e9942c5f26b7d40` | Preserved as permanent `archive/humble`. |
| `archive/noetic` | `32a3b131914b085624add24d79be84a54b9a0784` | Preserved as permanent `archive/noetic`. |
| `noetic` | `d6289c22885cbaadc25c709ce9d6dd1813f3027e` | Already an ancestor of permanent `archive/noetic`; no separate ref needed. |
| `deepseek/n3m-architecture-hardening-v1` | `86fc353878d213f59de1ab3bd5525bd837fc95e1` | Already an ancestor of canonical main. |
| `dev/better_loop` | `cab5805831aa84a696026c9963e4c86b73c314b9` | Already an ancestor of canonical main. |
| `overnight_indoor_static_retrieval_20260409` | `ac57cf859adbc61ef93452b7947b45440d767fe4` | Already an ancestor of canonical main. |
| `research/relocalization-benchmark-v1` | `72fa6f7e3fc22eae5bfc30700bc1e2cec7f7c4eb` | Already an ancestor of canonical main. |
| `research/relocalization-evidence-v2` | `c8952c3b32ef35f9295448483db1b87f89b65466` | Already an ancestor of canonical main. |
| `wip/freespace-select` | `32926e8eb8a28329023488f4be9d7dc9e04ebfe2` | Already an ancestor of canonical main. |
| `archive/better_loop` | `5443271c99e16bead0a52c466c1c7386aa30217e` | Keep the lesson that segment-consistency and candidate-source comparisons belong in diagnostics. Do not merge the reverted overlap experiments or restore removed benchmark tooling wholesale. |
| `archive/to-migrate-worktree-20260723` | `5bdf49564974022bc87bdfd69375c3b64c605c8b` | A 60-file snapshot of an uncommitted shared-workspace state. Forensic only; never merge wholesale. |
| `feature/builtinlio` | `d5afc2f5dd7972ccc43c60e3e9b40ce788778810` | Rejected as the product architecture. The maintained contract uses an external LIO frontend and a ROS-free backend; an internal FAST-LIO/DLIO fork would duplicate ownership and verification. |
| `wip/descriptor-select` | `b25f9bb34667b5f9ec8f6d8406332a767106b49e` | Reject descriptor authority changes. Retain diagnostic evidence only. |
| `wip/descsel-evbase` | `d3909494b3ab92d3313c52b3eb677ccb64eb1283` | Reject combined descriptor-selection and independent-view gate until qualified data falsifies the current shadow NO-GO. |
| `wip/diag-inliers` | `c95d18504e24cc8931af01a56b2ef96a0569c301` | The useful idea—per-hypothesis registration evidence—is already represented by current shadow diagnostics; no branch merge needed. |
| `wip/evidence-baseline` | `581398ed1a44bfcb3e891df4cacd790fea4c335a` | Reject promotion to authority. Current evidence does not justify changing ranking or lock defaults. |
| `wip/persist-evidence` | `98d432f9c6d16e24e59129ff8a24936625c82e1c` | Reject persistence as default behavior. C2 remains a shadow-only, data-qualified hypothesis. |
| `wip/product-v1-closeout-0725` | `0b829e2b75a89c3396978b76e660736fec56f16e` | Import only the source-tree build guard from `dd25f86`, with a current regression test. Do not import the V2/V3 migration, Atlas authority, or gravity-registration series: their data lineage was not release-qualified. |

## Lessons retained without old code

- Relocalization evidence, loop evidence, and graph-trial evidence remain
  diagnostic until a held-out dataset demonstrates a stable decision benefit.
- Oracle/GT-odometry dataset replay is a backend diagnostic, not an end-to-end
  localization or SLAM acceptance result.
- A self-consistent replay cannot establish frozen-map lineage or production
  eligibility by itself.
- The product boundary remains external LIO frontend plus ROS-free n3mapping
  backend with thin Humble and Noetic adapters.
- Dirty snapshots and large experimental branches are evidence sources, not
  integration units. Useful behavior must be reintroduced as a small current
  change with a focused test.
