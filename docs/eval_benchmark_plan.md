# N3Mapping Evaluation Benchmark Plan

This document fixes the near-term benchmark target before loop-closure or
relocalization algorithm changes.

## Target

The working target is not "looks better in one RViz run". It is:

- Indoor loop/relocalization success: about 80% or better.
- Outdoor loop/relocalization success: about 80% or better.
- Accepted loop closures should be true loops, should improve the optimized
  trajectory, and should not create large Z/roll/pitch residuals.
- Relocalization should report lock only when the pose is actually close to
  ground truth.

## Current Baseline Dataset

- KITTI360 is the current outdoor benchmark path.
- Existing tools:
  - `n3mapping_kitti360_reader`
  - `n3mapping_kitti360_eval`
  - `n3mapping_loop_debug_analyze.py`
  - `n3mapping_eval_matrix.py`
- M2DGR is the first indoor adapter path:
  - `n3mapping_m2dgr_eval`
  - See `docs/m2dgr_eval.md`.

## Indoor Dataset Shortlist

Use these in this order unless access or format friction proves high.

1. M2DGR
   - Project: https://github.com/SJTU-ViSYS/M2DGR
   - Paper: https://arxiv.org/abs/2112.13659
   - Why: ground robot, indoor and outdoor sequences, lidar, synchronized
     sensors, and ground truth from mocap / laser tracker / RTK.
   - First use: indoor relocalization and loop smoke because the platform is
     closest to the target ground-robot use case.

2. Hilti SLAM Challenge / Hilti-Oxford
   - 2021 paper: https://arxiv.org/abs/2109.11316
   - 2023 dataset: https://www.hilti-challenge.com/dataset-2023.html
   - Why: office, lab, construction indoor scenes plus high-accuracy sparse
     ground truth. Good for difficult feature-sparse construction-like spaces.
   - First use: loop-closure robustness and Z/roll/pitch consistency.

3. NTU VIRAL
   - Project: https://ntu-aris.github.io/ntu_viral_dataset
   - Paper: https://arxiv.org/abs/2202.00379
   - Why: indoor/outdoor lidar with calibration and laser-tracker ground truth.
   - Caveat: UAV viewpoint differs from this project target, so keep it as a
     secondary robustness check rather than the first indoor benchmark.

## Matrix Metrics

The matrix summary is the gate for algorithm changes. Key fields:

- `loop_precision`: accepted true loops / accepted candidate count.
- `loop_gt_pair_coverage`: accepted true loops / GT loop pairs.
- `loop_accepted_false_loop`: must stay low, ideally zero in smoke tests.
- `optimization_high_residual_z_after_count`: should trend down.
- `trajectory_translation_p95_m`, `trajectory_xy_p95_m`,
  `trajectory_z_p95_m`: global consistency checks.
- `pose_success_rate`: relocalization success gate.
- `lock_success_rate`: lock behavior, interpreted together with pose success.

## Next Steps

1. Keep using KITTI360 drive_0005 smoke to compare outdoor loop changes.
2. Run the M2DGR eval adapter on a real indoor sequence and add the result to
   the matrix.
3. Run `n3mapping_eval_matrix.py` across KITTI360 and M2DGR runs.
4. Only then replace or tune loop/relocalization modules based on failure type:
   retrieval miss, false positive retrieval, ICP verification failure,
   optimization/Z residual failure, or relocalization lock failure.
