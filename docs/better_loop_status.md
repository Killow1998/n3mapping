# Better Loop Status

Branch: `dev/better_loop`

## What Is Working

- Accepted far false loops are currently zero on the expanded smoke set after
  splitting heading labels:
  - KITTI360 drive0005 900 frames: `loop_accepted_far_false_loop=0`
  - M2DGR hall05 600 frames: `loop_accepted_far_false_loop=0`
- Near opposite/cross-heading revisits are now tracked separately from far false
  loops. Treating them as hard false positives was misleading.
- If a query has a selectable position-level loop candidate, selection is not
  the bottleneck:
  - drive0005: `query_position_selection_failure_count=0`
  - hall05: `query_position_selection_failure_count=0`

## Current Bottleneck

The bottleneck is loop verification, not candidate retrieval or best-loop
selection.

Expanded matrix evidence:

| run | position opportunity queries | selectable position queries | missed position queries |
| --- | ---: | ---: | ---: |
| drive0005_900 | 30 | 12 | 18 |
| hall05_600 | 32 | 29 | 3 |

Pair-level rejected position candidates:

| run | verification rejects | not selected |
| --- | ---: | ---: |
| drive0005_900 | 48 | 2 |
| hall05_600 | 33 | 8 |

## Rejected Experiment

Tested allowing `non-converged + high inlier` ICP results into LoopReferee.

Result:

- KITTI360 drive0005_700 improved locally.
- M2DGR hall05_600 introduced two real far false loops:
  - `60 -> 1`, GT distance 5.19 m
  - `65 -> 2`, GT distance 9.88 m

Verdict: do not use `non-converged + high inlier` as an accept path. Indoor
repeated structure makes it unsafe.

## Next Step

Do not tune LoopReferee weights yet. The next useful change is to improve
verification so more true position-level candidates become selectable without
adding far false loops.

Concrete next probe:

1. Add a second, stricter verification attempt only for candidates that already
   pass descriptor/spatial proposal and have a position-level opportunity in
   eval.
2. Compare against the same matrix fields:
   - `loop_accepted_far_false_loop`
   - `loop_query_with_selectable_position_candidate_count`
   - `loop_query_missed_position_candidate_count`
   - trajectory p95 metrics

If far false loops appear, reject the change.
