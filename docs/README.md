# N3Mapping 当前状态、问题与推进路线

更新时间：2026-06-11
当前基线提交：`15c6eb67640dc08e2354942fc57d336b35f4e78e`

本文是 `docs/` 下的主说明文档，替代旧的阶段性 TODO、数据集说明、GPT review prompt 和本地截图观察记录。目标是只保留当前开发真正需要的说明和问题上下文，避免旧文档让后续 agent 或 GPT Pro 误判当前状态。

## 目标

N3Mapping 当前的工程目标不是在单个 RViz 画面里看起来更顺，而是用可复现评估推动：

- 室外回环检测与处理成功率约 80% 或更高。
- 室内回环检测与处理成功率约 80% 或更高。
- 室外重定位成功率约 80% 或更高。
- 室内重定位成功率约 80% 或更高。
- accepted loop 必须是真回环，并且优化后应改善全局一致性。
- 重定位只应在位姿真正接近真值时报告 lock，false lock 比 no lock 更危险。

现在的推进原则：

- 先用真值和 debug artifact 定位 dominant failure。
- 只改一个模块。
- 重跑同一套 matrix。
- 如果改进只能靠调阈值或经验权重，先停下来交给 GPT Pro 纠偏方向。

## 明确不做的事

当前阶段不要做这些：

- 不盲调 RHPD、SC、ICP、重定位阈值。
- 不改变默认 `map_path` / `map_save_path`，仍先保留 `map/` 目录工作流。
- 不为了 KITTI360 继续扩工具而不解决诊断闭环。
- 不把 KITTI360 或 M2DGR 的 GT odometry 结果宣传成端到端 SLAM 成功率。
- 不把 `global_map.pcd` 作为定位、重定位或 TGW 读图依赖；它只是可视化/debug artifact。
- 不在没有证据时重写 retrieval 或迁移其他 loop closure 栈。

## 系统结构

核心后端保持 ROS-free，ROS Humble / Noetic wrapper 只负责输入输出、参数和 launch。

主要模块：

- `N3MappingCore`：mapping、localization、map_extension 三种模式的核心入口。
- `N3MappingSession`：组装 keyframe manager、matcher、loop detector、optimizer、serializer、world localizer。
- `LoopDetector`：RHPD primary retrieval + ScanContext auxiliary scoring / yaw。
- `LoopClosureManager`：对 verified loop 做过滤、选边和应用。
- `GraphOptimizer`：GTSAM 图优化，已改为 pending/committed 分离和事务式 rollback。
- `WorldLocalizing`：全局重定位和 tracking。
- `MapSerializer`：pbstream 保存/加载，默认 strict，显式 salvage。
- `N3NavResource reader`：轻量 downstream reader，strict 且不解析 TGW 不需要的 descriptor。

主要输出 artifact：

- `map/n3map.pbstream`：主地图格式。
- `map/global_map.pcd`：debug/可视化全局点云。
- `map/optimization.log`：优化日志。
- `loop_debug.jsonl`：mapping loop candidate / accepted / optimization 诊断。
- `relocalization_debug.jsonl`：重定位和 tracking 诊断。
- `accepted_loops.csv`：离线评估中 accepted loop 的摘要。
- `metrics.json` / `matrix_summary.json`：离线评估指标。

## 已完成的工程底座

### CI 与依赖

- GitHub CI 已覆盖 Humble + Noetic build/test。
- GTSAM 4.1.1 和 small_gicp 在 CI 中构建/缓存。
- small_gicp 使用固定 SHA，不使用 moving `master` / `main`。

### Pbstream 与 nav reader

已完成：

- pbstream 支持 dense optimized trajectory。
- old pbstream 可按需生成 keyframe fallback dense trajectory，并在 metadata 中标记 degraded。
- `MapSerializer` 和 `N3NavResource reader` 共享底层 ROS-free proto validation/parser。
- `MapSerializer` 默认 strict load；salvage 必须显式请求。
- reader 默认 strict，不静默 fallback 给 TGW。
- reader 不依赖 ROS、GTSAM、small_gicp、LoopDetector 或完整 backend。
- reader 解析 keyframe 时不解析未使用的 SC/RHPD descriptor，保持轻量。
- `MapSerializer::loadMap()` 使用 temp staging + `swapWith`，避免失败加载清空旧状态。

### Runtime robustness

已完成：

- PCL VoxelGrid 对极端坐标/过小 leaf size 的 integer overflow 做 guard。
- loop candidate 控制台噪声减少，详细信息写入 debug/optimization artifact。
- callback stall 优化，降低 bag replay 时主回调被全局地图构建拖慢的风险。
- `GraphOptimizer` pending/committed 状态分离，优化失败不会把失败 edge/node/factor 留进 committed graph。
- loop optimization failure accounting 已修复：优化失败不会计入 accepted loop、metrics 或 CSV。

### 诊断与评估工具

已完成：

- `loop_debug.jsonl`：记录 loop candidate、gate、ICP、residual、edge mode 和 optimization summary。
- `relocalization_debug.jsonl`：记录 relocalize / tracking 过程。
- `n3mapping_kitti360_reader`：KITTI360 frame-id 对齐 smoke reader。
- `n3mapping_kitti360_eval`：KITTI360 offline mapping_loop / relocalization eval。
- `n3mapping_m2dgr_eval`：M2DGR extracted cloud + TUM GT offline eval adapter。
- `n3mapping_loop_debug_analyze.py`：用 GT 标注 loop candidate / accepted loop failure class。
- `n3mapping_eval_matrix.py`：汇总多次 eval run 的 matrix。

## 当前关键证据

最近一次诊断基于 KITTI360 `2013_05_28_drive_0005_sync` smoke：

```bash
LD_LIBRARY_PATH=/home/user/ros_ws/to_migrate_ws/install/gtsam/lib:/home/user/ros_ws/to_migrate_ws/install/small_gicp/lib:$LD_LIBRARY_PATH \
  ./build/n3mapping/n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0005_sync \
  --mode mapping_loop \
  --max_frames 450 \
  --stride 5 \
  --output /tmp/n3mapping_kitti360_drive0005_vertical_diag_robust_stride5_450_20260611
```

再运行：

```bash
python3 src/n3mapping/tools/n3mapping_loop_debug_analyze.py \
  --loop_debug /tmp/n3mapping_kitti360_drive0005_vertical_diag_robust_stride5_450_20260611/loop_debug.jsonl \
  --keyframes_gt /tmp/n3mapping_kitti360_drive0005_vertical_diag_robust_stride5_450_20260611/keyframes_gt.csv \
  --accepted_loops /tmp/n3mapping_kitti360_drive0005_vertical_diag_robust_stride5_450_20260611/accepted_loops.csv \
  --output /tmp/n3mapping_vertical_diag_robust_stride5_analysis_v2_20260611

python3 src/n3mapping/tools/n3mapping_eval_matrix.py \
  --run drive0005_vertical_diag_robust_stride5=/tmp/n3mapping_kitti360_drive0005_vertical_diag_robust_stride5_450_20260611 \
  --output /tmp/n3mapping_eval_matrix_drive0005_vertical_diag_robust_stride5_v2_20260611
```

当前结果：

| 指标 | 值 | 解释 |
|---|---:|---|
| accepted loops | 8 | smoke 中真正进入图优化的回环数 |
| accepted false loops | 0 | 当前 smoke 未发现 accepted false loop |
| full6dof edges | 6 | 未降级的 6DoF 回环边 |
| planar_xy_yaw edges | 2 | 垂直分量降权的回环边 |
| bad Z measurement | 4 | ICP loop measurement 的 Z 分量相对 GT 明显偏差 |
| bad Z after optimization | 2 | 优化后仍留下明显 Z residual |
| corrected Z cases | 5 | 原本候选 residual 大，但优化后被修回 |
| max residual Z after | 约 1.35m | 当前最坏优化后 Z residual |
| trajectory XY p95 | 约 1.26m | outdoor smoke 全局 XY 误差 |
| trajectory Z p95 | 约 1.08m | outdoor smoke 全局 Z 误差 |

关键解释：

- 之前看起来像 `7` 个 accepted true loop 都是 bad-Z，其实这个统计过宽。
- 新分类显示：当前优化能修正其中 `5` 个，真正优化后仍 high-Z 的是 `2` 个。
- 这两个 remaining high-Z case 都是 full6dof edge，且 ICP measurement 本身在 Z 上错约 `±1.55m`。
- 新增的 runtime 信号没有把这两个坏例子从好例子里明显分开：
  - robust Z overlap 都是 `1`。
  - raw ICP vertical information ratio 与正常样本重叠。
- 因此继续用这些量硬切阈值，会进入调参区；应先让 GPT Pro 判断下一步是新增观测信号，还是允许基于 GT 做阈值/权重 tuning。

两个 remaining high-Z accepted loops：

| query -> match | edge mode | ICP Z error to GT | residual Z after | 备注 |
|---|---|---:|---:|---|
| 296 -> 16 | full6dof | -1.54m | 1.02m | measurement Z 错，优化后仍 high |
| 326 -> 56 | full6dof | +1.55m | 1.35m | measurement Z 错，优化后仍 high |

## 当前已知问题

### P0：回环 Z / roll / pitch 可观测性还不够可靠

现象：

- 真回环可以在 XY/yaw 上正确，但 ICP measurement 的 Z 分量可能错。
- 图优化有时能修正 Z，有时留下 1m 级 residual。

已排除或暂时弱化的假设：

- 不是当前 smoke 中 false loop 太多：accepted false loop 为 0。
- 不是所有 bad-Z 都会伤害优化：5 个被纠正。
- 当前 Z overlap / robust overlap / raw Hessian vertical ratio 不足以稳定区分剩余坏例子。

下一步可选方向：

1. 新增 runtime 观测信号，而不是调现有阈值：
   - visibility/raycast-aware submap validation
   - ground / vertical structure segmentation
   - multi-hypothesis ICP for vertical ambiguity
   - range image occlusion consistency
   - ICP covariance/eigen decomposition beyond simple diagonal ratio
2. 如果 GPT Pro 判断可以 tuning，再基于 GT residual 设计明确的阈值/权重实验。

### P1：回环 recall 还需要更可靠的 denominator

当前 smoke 能看 accepted precision，但 recall 还需要：

- GT loop opportunity count。
- retrieval true positives / misses。
- verification reject true loop。
- accepted true loop count / GT loop opportunity。

现在 analyzer 已经能标注部分字段，但后续还应把 recall denominator 固化到 matrix gate 中。

### P1：重定位还缺 false-lock 优先的评估闭环

重定位目标不是单纯提高 `lock_success_rate`，而是：

- `lock_precision` 高。
- `false_lock_rate` 接近 0。
- lock 时 pose error / yaw error 小。
- tracking after lock 不漂。

需要重点输出：

- correct_lock / false_lock。
- pose_error_at_lock p50/p95。
- yaw_error_at_lock p50/p95。
- lock_latency_frames。
- candidate retrieval miss / wrong basin / temporal guard too loose / too strict。

不要为了提高 lock 数放宽 guard；false lock 对机器人更危险。

### P1：室内数据还没形成稳定 benchmark

当前 outdoor 主路径是 KITTI360。

室内优先级：

1. M2DGR：ground robot、indoor/outdoor、lidar、GT 来源覆盖 mocap / laser tracker / RTK。
2. Hilti SLAM Challenge：office/lab/construction，适合 feature-sparse 和工程退化场景。
3. NTU VIRAL：indoor/outdoor + laser tracker GT，但 UAV viewpoint 与当前地面机器人目标不同，作为后续泛化检查。

M2DGR adapter 当前读取 extracted `.pcd` / KITTI-style `.bin` + TUM GT，不直接读 ROS bag。这样保持 Humble/Noetic 可用，也避免评估工具先被 ROS bag API 复杂度绑住。

### P2：GT odometry 只能隔离 backend，不能代表端到端

KITTI360 / M2DGR offline eval 如果用 GT pose 构造 `LioFrame.T_world_lidar`，评估的是：

```text
backend loop / relocalization under ideal odometry
```

不是：

```text
full SLAM with real LIO drift
```

后续 matrix 必须区分：

- `odom_source=gt`
- `odom_source=gt_noisy`
- `odom_source=dataset_odom`
- `odom_source=lio_frontend`

如果 ideal odom 下都失败，说明 backend 本身有问题。只有 frontend odom 通过，才接近实车端到端结论。

## 评估工具使用

### Humble build/test

```bash
cd /home/user/ros_ws/to_migrate_ws
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --packages-select n3mapping --symlink-install --parallel-workers 1 --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping --event-handlers console_direct+ --parallel-workers 1
colcon test-result --test-result-base build/n3mapping --verbose
```

Direct binary runs may need:

```bash
export LD_LIBRARY_PATH=/home/user/ros_ws/to_migrate_ws/install/gtsam/lib:/home/user/ros_ws/to_migrate_ws/install/small_gicp/lib:$LD_LIBRARY_PATH
```

### KITTI360 reader smoke

```bash
ros2 run n3mapping n3mapping_kitti360_reader \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --output /tmp/n3mapping_kitti360_reader_test \
  --max_frames 50 \
  --dump_sample_pcd
```

The reader aligns lidar `.bin` and pose rows by frame id, not line number.

### KITTI360 mapping loop eval

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0005_sync \
  --mode mapping_loop \
  --calib_mode auto \
  --max_frames 450 \
  --stride 5 \
  --output /tmp/n3mapping_kitti360_drive0005_mapping_loop
```

Expected artifacts:

- `metrics.json`
- `trajectory_est.txt`
- `trajectory_gt.txt`
- `keyframes_gt.csv`
- `accepted_loops.csv`
- `loop_debug.jsonl`

### M2DGR mapping loop eval

```bash
ros2 run n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode mapping_loop \
  --max_frames 300 \
  --stride 1 \
  --max_time_diff 0.05 \
  --output /tmp/n3mapping_m2dgr_hall03_mapping
```

M2DGR GT format:

```text
timestamp x y z qx qy qz qw
```

### Matrix summary

```bash
python3 src/n3mapping/tools/n3mapping_eval_matrix.py \
  --run kitti360_drive0005=/tmp/n3mapping_kitti360_drive0005_mapping_loop \
  --run m2dgr_hall03=/tmp/n3mapping_m2dgr_hall03_mapping \
  --output /tmp/n3mapping_eval_matrix
```

Important loop fields:

- `loop_precision`
- `loop_accepted_false_loop`
- `loop_accepted_true_loop_bad_z_measurement`
- `loop_accepted_true_loop_bad_z_after`
- `loop_accepted_true_loop_corrected_z`
- `optimization_high_residual_z_after_count`
- `trajectory_translation_p95_m`
- `trajectory_xy_p95_m`
- `trajectory_z_p95_m`

## Debug artifact contracts

### `loop_debug.jsonl`

Every line is single-line JSON. Candidate lines include:

- query / match ids
- candidate source
- RHPD / SC distances
- fused score
- yaw diff
- ICP convergence, fitness, inlier ratio
- ICP transform norms
- residual x/y/z and roll/pitch/yaw
- gate result and reject reason
- loop information diagonal
- edge mode: `full6dof`, `planar_xy_yaw`, rejection mode, etc.
- vertical downweight state
- Z span / robust span / overlap before and after
- source-target Z centroid delta before and after
- raw ICP vertical information ratio

Optimization summary lines include:

- accepted edge count
- loop residual before/after
- mean/max pose update translation
- mean/max pose update rotation

### `accepted_loops.csv`

Offline eval accepted loop rows include:

- query id
- match id
- fitness / inlier
- edge mode
- vertical observability score
- vertical downweighted
- Z distribution diagnostics
- raw ICP vertical information ratio

### `loop_candidates_labeled.csv`

Analyzer output adds GT labels:

- GT query-match translation
- GT query-match yaw
- GT loop classification
- ICP measurement error to GT x/y/z/roll/pitch/yaw
- residual Z after optimization
- failure class
- `z_measurement_bad`
- `z_after_bad`
- `z_corrected`

## Recommended next decision

Current evidence does not justify another hidden threshold tweak.

Ask GPT Pro to choose between:

1. Add new runtime evidence:
   - visibility/raycast consistency
   - ground/vertical segmentation
   - better ICP covariance/eigen analysis
   - multi-hypothesis vertical ICP
2. Allow a controlled tuning experiment:
   - explicitly define thresholds/weights
   - run against KITTI360 + M2DGR matrix
   - accept only if false loops stay zero and after-Z improves

If GPT Pro says to tune, record:

- exact parameter(s)
- baseline matrix
- changed matrix
- acceptance gate
- rollback condition

If GPT Pro says to add evidence, implement the smallest new diagnostic or structural signal first, then rerun the same matrix before changing behavior.

## Suggested PR order

1. `loop_failure_classifier_cleanup`
   - keep current classifier split and diagnostics.
   - verify KITTI360 + M2DGR artifact consistency.

2. `relocalization_false_lock_matrix`
   - add lock precision / false lock / pose error at lock metrics.
   - do not loosen lock thresholds.

3. `vertical_observability_next_signal`
   - only after GPT Pro chooses the next runtime evidence source.
   - avoid tuning existing thresholds without a written gate.

4. `indoor_benchmark_m2dgr`
   - run at least one M2DGR indoor smoke and one longer indoor sequence.
   - classify dominant indoor failures separately from KITTI360.

## Notes for future agents

- Start from this file, not old docs.
- Do not resurrect deleted phase notes unless a specific historical detail is needed.
- Before changing loop/relocalization behavior, run or inspect the latest matrix.
- If a proposed change cannot explain which failure class it targets, do not implement it yet.
- If the next step becomes parameter tuning, stop and ask the user/GPT Pro for direction.
