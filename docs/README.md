# N3Mapping 当前状态、问题与推进路线

更新时间：2026-06-12
当前基线提交：`cf0a2e05720884219ef7c2ec53eb8888f98c6516`

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

最近一次行为基线基于 KITTI360 `2013_05_28_drive_0005_sync` smoke：

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

### 2026-06-12 graph shadow trial diagnostic

已新增 `loop_graph_consistency_diagnostics`，只在 `loop_debug_enable=true` 时运行。它对即将进入 optimizer 的 best loop edge 构造临时 GraphOptimizer 进行 shadow/trial optimize，并把结果写入：

- `loop_debug.jsonl`
- `accepted_loops.csv`
- `loop_candidates_labeled.csv`
- `matrix_summary.json`

新增字段包括：

- `graph_trial_success`
- `graph_trial_residual_x/y/z_after`
- `graph_trial_residual_roll/pitch/yaw_after`
- `graph_trial_residual_translation_norm_after`
- `graph_trial_residual_rotation_norm_after`
- `graph_trial_mean/max_pose_update_translation`
- `graph_trial_mean/max_pose_update_rotation`
- `graph_trial_existing_loop_residual_delta`
- `graph_trial_odom_residual_delta`
- `graph_trial_consistency_score`
- `graph_trial_recommendation`

验证命令：

```bash
ulimit -v 25165824 && \
env LD_LIBRARY_PATH=/home/user/ros_ws/to_migrate_ws/install/gtsam/lib:/home/user/ros_ws/to_migrate_ws/install/small_gicp/lib:$LD_LIBRARY_PATH \
  ./build/n3mapping/n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0005_sync \
  --mode mapping_loop \
  --max_frames 450 \
  --stride 5 \
  --output /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_450_20260612_r1

python3 src/n3mapping/tools/n3mapping_loop_debug_analyze.py \
  --loop_debug /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_450_20260612_r1/loop_debug.jsonl \
  --keyframes_gt /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_450_20260612_r1/keyframes_gt.csv \
  --accepted_loops /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_450_20260612_r1/accepted_loops.csv \
  --output /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_analysis_20260612_r1

python3 src/n3mapping/tools/n3mapping_eval_matrix.py \
  --run drive0005_graph_trial_diag=/tmp/n3mapping_kitti360_drive0005_graph_trial_diag_450_20260612_r1 \
  --output /tmp/n3mapping_eval_matrix_drive0005_graph_trial_diag_20260612_r1
```

行为指标保持 baseline：

| 指标 | 值 |
|---|---:|
| accepted loops | 8 |
| accepted false loops | 0 |
| bad Z after optimization | 2 |
| max residual Z after | 1.3476m |
| trajectory XY p95 | 1.2607m |
| trajectory Z p95 | 1.0767m |

graph trial 分组结果：

| 分组 | consistency score |
|---|---:|
| remaining bad-Z-after mean | 0.3165 |
| remaining bad-Z-after min | 0.3091 |
| corrected-Z mean | 0.4774 |
| corrected-Z min | 0.3692 |

具体 accepted loop：

| query -> match | edge mode | z_after_bad | z_corrected | graph_trial_residual_z_after | graph_trial_score |
|---|---|---|---|---:|---:|
| 296 -> 16 | full6dof | true | false | 1.0200m | 0.3091 |
| 326 -> 56 | full6dof | true | false | 1.3478m | 0.3239 |
| 286 -> 4 | planar_xy_yaw | false | true | 0.2371m | 0.5175 |
| 291 -> 10 | full6dof | false | true | 0.1430m | 0.6121 |
| 301 -> 22 | full6dof | false | true | 0.3667m | 0.4147 |
| 311 -> 35 | full6dof | false | true | 0.2384m | 0.3692 |
| 321 -> 49 | planar_xy_yaw | false | true | 0.1334m | 0.4734 |

结论：

- graph shadow trial 是比单独 heightmap 更接近目标模型的 runtime evidence。
- 它对 remaining bad-Z-after 给出更低 score，但与部分 corrected-Z case 仍有接近区间。
- 目前不能直接把 `graph_trial_consistency_score` 变成 commit gate 或 edge-mode 阈值。
- 下一步若要改行为，必须先让 GPT Pro / 人工 review 判断是否接受“trial-gated commit / vertical-neutral constraint”的证明强度；否则应继续做 visibility/raycast consistency 或更强的局部可见性诊断。

### 2026-06-19 shadow LoopReferee evidence bundle

新增 shadow-only LoopReferee 诊断字段：

- `loop_referee_recommendation`
- `loop_referee_reason`
- `loop_referee_risk_flags`

这些字段只写入 `loop_debug.jsonl`、`accepted_loops.csv` 和
`loop_candidates_labeled.csv`。它们不改变 `verified`、`edge_mode`、
`applyEdges()` 或 optimizer commit。

本轮目标不是把某个 signal 直接变成 gate，而是验证“证据包 + referee”
这个方向能不能解释 M2DGR indoor failure。M2DGR run：

```text
/tmp/n3mapping_m2dgr_matrix_20260619_referee_diag_r2
```

Humble 本地验证：

```text
266 tests, 0 errors, 0 failures, 0 skipped
```

M2DGR 行为指标保持当前实现形状：

| run | yaw label | accepted loops | loop precision | relocalization |
|---|---|---:|---:|---|
| hall_05 mapping | yaw45 | 34 | 0.706 | - |
| hall_05 mapping | yaw180 diagnostic | 34 | 0.882 | - |
| gate_02 mapping | yaw45 | 1 | 0.0 | - |
| gate_02 mapping | yaw180 diagnostic | 1 | 1.0 | - |
| hall_05 relocalization | - | - | - | lock_precision=0, false_lock_rate=1.0 |
| gate_02 relocalization | - | - | - | no locks, pose_success=0 |

关键结论：

- 之前“单一 vertical signal 直接接行为”的路线已经被证伪。
- heightmap / vertical hypothesis / graph trial 在 M2DGR 上普遍触发，
  不能单独区分 true loop 与 false loop。
- 本轮保守 referee 把 accepted loops 标成 `needs_more_evidence`，并输出
  `risk_flags`，没有给出可执行 gate。
- 这不是算法改进结果，而是一个明确的反证：当前 runtime evidence 还不足
  以驱动 loop commit / neutralize / reject。
- 下一步如果要真正改善回环处理，应引入更强的 `LoopEvidenceBundle` /
  `LoopReferee` 结构，并增加 visibility / raycast / local geometry consistency
  等新证据；不能再用现有 `heightmap`、`vertical_hypothesis` 或
  `graph_trial_consistency_score` 单独调阈值。

### 2026-06-12 evidence correlation report

新增 `tools/n3mapping_loop_evidence_correlation.py`，用于比较当前 runtime evidence 对 `z_after_bad` 与 `z_corrected` 的区分力。

输入：

```bash
python3 src/n3mapping/tools/n3mapping_loop_evidence_correlation.py \
  --labeled_csv /tmp/n3mapping_kitti360_drive0005_graph_trial_diag_analysis_20260612_r1/loop_candidates_labeled.csv \
  --output /tmp/n3mapping_loop_evidence_correlation_drive0005_20260612_r1
```

输出：

- `loop_evidence_correlation.csv`
- `loop_evidence_correlation.json`

报告字段：

- `signal_name`
- `bad_z_after_mean`
- `corrected_z_mean`
- `auc_bad_greater`
- `auc_like_score`
- `direction`
- `overlap_count`
- `false_positive_if_thresholded`
- `false_negative_if_thresholded`

注意：

- 这是 offline analysis，不参与 runtime。
- `false_positive_if_thresholded` / `false_negative_if_thresholded` 只表示“若用均值中点做阈值”会发生什么，用来防止弱 signal 被误当强 gate。
- 这个工具不会给出可直接上线的阈值；是否把某个 evidence 接入行为，仍需要固定 matrix + GPT Pro / 人工 review。

KITTI360 drive_0005 450-frame smoke 上的初步结果：

| signal | direction | auc_like | overlap | mean bad-Z-after | mean corrected-Z |
|---|---|---:|---:|---:|---:|
| `graph_trial_consistency_score` | lower is bad | 1.0 | 0 | 0.3165 | 0.4774 |
| `graph_trial_residual_translation_norm_after` | higher is bad | 1.0 | 0 | 1.9891 | 0.6857 |
| `graph_trial_residual_z_after` | higher is bad | 1.0 | 0 | 1.1839 | 0.2237 |
| `best_z_offset_m` | lower is bad | 0.95 | 2 | -1.25 | 1.10 |

解释：

- 在这组 smoke 上，graph trial 系列信号是当前最强候选 evidence。
- 但样本只有 `2` 个 bad-Z-after 和 `5` 个 corrected-Z，不能直接推出 runtime gate。
- `graph_trial_consistency_score` 的均值中点阈值仍会产生一个 corrected-Z false positive；因此下一步必须先扩到 M2DGR indoor / 其它 KITTI360 sequence，再决定是否做 trial-gated behavior。

### 2026-06-12 multi-sequence KITTI360 expansion

为避免只在单个固定场景上判断 loop evidence，增加了一轮 KITTI360 多序列 smoke：

```text
artifact root: /tmp/n3mapping_kitti360_multi_sequence_20260612_r1
sequences:
  - 2013_05_28_drive_0003_sync
  - 2013_05_28_drive_0005_sync
  - 2013_05_28_drive_0007_sync
  - 2013_05_28_drive_0010_sync
  - 2013_05_28_drive_0000_sync
mode: mapping_loop
max_frames: 450
stride: 5
```

输出：

- `/tmp/n3mapping_kitti360_multi_sequence_20260612_r1/matrix/matrix_summary.csv`
- `/tmp/n3mapping_kitti360_multi_sequence_20260612_r1/matrix/matrix_summary.json`
- `/tmp/n3mapping_kitti360_multi_sequence_20260612_r1/combined_loop_candidates_labeled.csv`
- `/tmp/n3mapping_kitti360_multi_sequence_20260612_r1/combined_evidence_correlation/loop_evidence_correlation.json`

矩阵摘要：

| run | frames | candidates | accepted loops | true loops | false loops | high-Z after | max Z after m | XY p95 m | Z p95 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `drive_0003` | 202 | 297 | 0 | 0 | 0 | 0 | 0 | 0.000003 | 0.000000 |
| `drive_0005` | 450 | 787 | 8 | 8 | 0 | 2 | 1.3476 | 1.2607 | 1.0767 |
| `drive_0007` | 450 | 787 | 0 | 0 | 0 | 0 | 0 | 0.000023 | 0.000001 |
| `drive_0010` | 450 | 787 | 0 | 0 | 0 | 0 | 0 | 0.000011 | 0.000002 |
| `drive_0000` | 450 | 787 | 3 | 3 | 0 | 0 | 0.2403 | 3.2265 | 0.0408 |
| **total** | **2002** | **3445** | **11** | **11** | **0** | **2** | - | - | - |

跨序列 evidence correlation 结果：

| signal | direction | auc_like | overlap | false positives | false negatives | bad mean | corrected mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `graph_trial_residual_translation_norm_after` | higher is bad | 1.0 | 0 | 0 | 0 | 1.9891 | 0.6389 |
| `graph_trial_residual_z_after` | higher is bad | 1.0 | 0 | 0 | 0 | 1.1839 | 0.2120 |
| `graph_trial_consistency_score` | lower is bad | 0.875 | 2 | 2 | 0 | 0.3165 | 0.4718 |

解释：

- 多序列后，候选样本从单序列 `787` 增加到 `3445`，accepted true loop 从 `8` 增加到 `11`，false accepted loop 仍为 `0`。
- high-Z-after 仍来自 `drive_0005` 的 `2` 个 case；其它序列在这 450-frame smoke 窗口内 accepted loop 很少或没有 accepted loop。
- graph-trial residual 系列比 `graph_trial_consistency_score` 更干净，但样本中的 bad-Z-after 仍只有 `2` 个，不能直接变成 runtime gate。
- 当前结论应更新为：outdoor 多序列候选覆盖已扩大，但真正用于 Z 行为决策的 accepted-loop bad case 仍不足；下一步必须引入 M2DGR indoor / 更长 KITTI360 window，再考虑行为改动。

## 当前已知问题

### P0：回环 Z / roll / pitch 可观测性还不够可靠

现象：

- 真回环可以在 XY/yaw 上正确，但 ICP measurement 的 Z 分量可能错。
- 图优化有时能修正 Z，有时留下 1m 级 residual。

已排除或暂时弱化的假设：

- 不是当前 smoke 中 false loop 太多：accepted false loop 为 0。
- 不是所有 bad-Z 都会伤害优化：5 个被纠正。
- 当前 Z overlap / robust overlap / raw Hessian vertical ratio 不足以稳定区分剩余坏例子。
- 当前已加入 multi-hypothesis vertical ICP 诊断作为新的 runtime evidence；它只写诊断字段，不改变回环接受、edge mode 或优化行为。

下一步可选方向：

1. 新增 runtime 观测信号，而不是调现有阈值：
   - visibility/raycast-aware submap validation
   - ground / vertical structure segmentation
   - multi-hypothesis ICP for vertical ambiguity（已作为诊断落地）
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

### P1：重定位 false-lock 评估闭环

重定位目标不是单纯提高 `lock_success_rate`，而是：

- `lock_precision` 高。
- `false_lock_rate` 接近 0。
- lock 时 pose error / yaw error 小。
- tracking after lock 不漂。

需要重点输出：

- correct_lock / false_lock。
- pose_error_at_lock p50/p95。
- yaw_error_at_lock p50/p95。
- lock_latency_frames / first_lock_frame。
- candidate retrieval miss / wrong basin / temporal guard too loose / too strict。

当前 KITTI360 / M2DGR relocalization eval 已经输出：

- `correct_lock_count`
- `false_lock_count`
- `lock_precision`
- `false_lock_rate`
- `pose_error_at_lock_p50_m`
- `pose_error_at_lock_p95_m`
- `yaw_error_at_lock_p50_deg`
- `yaw_error_at_lock_p95_deg`
- `first_lock_frame`
- `lock_latency_p50_frames`
- `lock_latency_p95_frames`

`relocalization_queries.csv` 也会写出 `pose_success`、`lock_correct`、`false_lock`、`lock_latency_frames` 和 `failure_class`。后续算法修改必须优先保证 false lock 不增加。

不要为了提高 lock 数放宽 guard；false lock 对机器人更危险。

### P1：室内数据还没形成稳定 benchmark

当前 outdoor 主路径是 KITTI360。

室内优先级：

1. M2DGR：ground robot、indoor/outdoor、lidar、GT 来源覆盖 mocap / laser tracker / RTK。
2. Hilti SLAM Challenge：office/lab/construction，适合 feature-sparse 和工程退化场景。
3. NTU VIRAL：indoor/outdoor + laser tracker GT，但 UAV viewpoint 与当前地面机器人目标不同，作为后续泛化检查。

M2DGR adapter 当前读取 extracted `.pcd` / KITTI-style `.bin` + TUM GT，不直接读 ROS bag。这样保持 Humble/Noetic 可用，也避免评估工具先被 ROS bag API 复杂度绑住。

M2DGR 官方序列表说明它包含 36 条序列，覆盖 Street、Circle、Gate、Walk、Hall、Door、Lift、Room、Roomdark；GT 来源分别包括 RTK/INS、Leica laser tracker、Vicon mocap。当前最适合先做 n3mapping loop/relocalization smoke 的序列：

| priority | sequence | type | why |
|---|---|---|---|
| 1 | `hall_05` | indoor, Leica GT | 官方标注 `circle`，最适合先找室内闭环机会。 |
| 2 | `hall_01` / `hall_03` / `hall_04` | indoor, Leica GT | 官方标注 random walk，适合看 repeated-place / corridor ambiguity。 |
| 3 | `gate_02` | outdoor, RTK/INS | 官方标注 `loop back`，体量比 street 小。 |
| 4 | `street_04` / `street_08` | outdoor, RTK/INS | 官方标注 `loop back` 或 loop-like zigzag。 |
| 5 | `walk_01` | outdoor, RTK/INS | 官方标注 back-and-forth，可测试反向经过同一路段。 |
| 6 | `door_01` | indoor/outdoor, Leica GT | outdoor-to-indoor-to-outdoor long-term，适合后续跨场景鲁棒性。 |

本机当前没有发现已解压的 M2DGR rosbag / GT 数据；只检测到了 n3mapping 的 M2DGR eval 二进制。拿到某条序列后，先用 GT 轨迹筛 loop opportunity，再决定是否提取点云跑完整 eval。

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

当前 KITTI360 / M2DGR offline eval 会在 `metrics.json` 和 matrix 中记录：

- `odom_source`
- `alignment_input_lidar_count`
- `alignment_input_gt_count`
- `alignment_matched_count`
- `alignment_selected_count`
- `alignment_dropped_lidar_count`
- `alignment_dropped_gt_count`
- `alignment_time_diff_median_s`
- `alignment_time_diff_p95_s`
- `alignment_time_diff_max_s`

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

### M2DGR GT loop opportunity check

先不要直接跑完整 mapping。拿到某个 M2DGR GT 文件后，先判断这条轨迹是否真的有闭环机会：

```bash
python3 src/n3mapping/tools/n3mapping_tum_loop_opportunities.py \
  --trajectory /path/to/M2DGR/hall_05/groundtruth.txt \
  --sequence hall_05 \
  --output /tmp/n3mapping_m2dgr_hall05_loop_opportunities \
  --distance_threshold_m 3.0 \
  --yaw_threshold_deg 45 \
  --min_time_gap_s 20 \
  --min_index_gap 50 \
  --sample_stride 1
```

输出：

- `loop_opportunity_summary.json`
- `loop_opportunities.csv`
- `trajectory_xy.csv`

判断规则：

- `loop_opportunity_pair_count > 0`：有 GT 上可定义的回环对。
- `query_with_loop_opportunity_ratio` 越高，越适合评估 loop recall。
- `trajectory_xy.csv` 可直接用 Python/表格画 XY 轨迹；`has_loop_opportunity=true` 的 query 是优先检查位置。
- 如果 `hall_05` 机会太少，再试 `gate_02`、`street_04`、`street_08`、`door_01`。

### 2026-06-18 M2DGR hall_05 / gate_02 first matrix

已下载并提取：

```text
/home/user/dataset/M2DGR/hall_05/hall_05.bag
/home/user/dataset/M2DGR/hall_05/hall_05.txt
/home/user/dataset/M2DGR/hall_05/groundtruth_yaw_fallback.txt
/home/user/dataset/M2DGR/hall_05/velodyne_points/*.bin

/home/user/dataset/M2DGR/gate_02/gate_02.bag
/home/user/dataset/M2DGR/gate_02/gate_02.txt
/home/user/dataset/M2DGR/gate_02/velodyne_points/*.bin
```

点云提取结果：

```text
hall_05: 4017 lidar frames, 203348051 points, 3.1G extracted XYZI bins
gate_02: 3268 lidar frames, 157231672 points, 2.4G extracted XYZI bins
```

GT loop opportunity：

| sequence | pose count | path length XY m | loop opportunity pairs | query opportunity ratio |
|---|---:|---:|---:|---:|
| `hall_05` | 2698 | 285.50 | 428575 | 0.774 |
| `gate_02` | 33677 | 200.53 | 1681 | 0.002 |

Matrix artifact:

```text
/tmp/n3mapping_m2dgr_matrix_20260618_r1
```

Run settings:

```text
max_frames=800
stride=5
build_map_frames=250
max_time_diff=0.05
```

Mapping-loop summary with the default GT loop yaw threshold 45 deg:

| run | frames | keyframes | candidates | accepted loops | true loops | false loops | precision | high-Z after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hall_05_mapping_loop` | 531 | 225 | 337 | 33 | 20 | 13 | 0.606 | 0 |
| `gate_02_mapping_loop` | 654 | 202 | 297 | 1 | 0 | 1 | 0.0 | 0 |

Because `hall_05.txt` has zero quaternions, `groundtruth_yaw_fallback.txt` estimates yaw from local trajectory direction. That yaw is useful for running the tool, but it is not a trusted sensor attitude. Recomputing loop labels with yaw threshold 180 deg gives:

| run | accepted loops | true loops | false loops | precision |
|---|---:|---:|---:|---:|
| `hall_05_mapping_loop` | 33 | 28 | 5 | 0.848 |
| `gate_02_mapping_loop` | 1 | 1 | 0 | 1.0 |

Relocalization summary:

| run | queries | locks | pose successes | correct locks | false locks | lock precision | false lock rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hall_05_relocalization` | 281 | 2 | 112 | 0 | 2 | 0.0 | 1.0 |
| `gate_02_relocalization` | 404 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |

Interpretation:

- M2DGR exposes a different dominant failure from KITTI360. KITTI360 outdoor smoke had false accepted loop count 0; M2DGR hall_05 shows accepted false loops even with position-only yaw relaxation.
- `hall_05` is still the best indoor sequence because it has many GT loop opportunities, but the GT attitude contract is weak. Future loop metrics on hall_05 should prefer position-first labels or use yaw only as a secondary diagnostic.
- The next algorithm work should not return to Z-only evidence. On M2DGR, the first failure to inspect is retrieval/verification admitting false loop candidates in repeated indoor geometry.
- Relocalization should not be loosened: hall_05 already produced false locks, so lock precision must remain the primary metric.

Rejected behavior experiment:

```text
experiment: graph-prior conflict gate
artifact: /tmp/n3mapping_m2dgr_matrix_20260618_referee_r1
rule tried: reject loop if ||candidate_residual.translation|| > 3.5 * loop_max_icp_translation
```

Offline on the baseline `hall_05` yaw-relaxed labels this looked clean: it would reject the 5 false accepted loops without rejecting the 28 true accepted loops. A full rerun did not produce a safe system-level improvement:

| run / metric | baseline | experiment |
|---|---:|---:|
| `hall_05` yaw180 loop precision | 0.848 | 0.853 |
| `hall_05` yaw180 false accepted loops | 5 | 5 |
| `hall_05` high-Z-after count | 0 | 1 |
| `gate_02` yaw180 loop precision | 1.0 | 0.0 |
| `hall_05` relocalization pose_success_rate | 0.399 | 0.043 |
| `hall_05` false locks | 2 | 5 |

Conclusion: single-signal graph-prior gating is not stable enough. It slightly improved `hall_05` yaw-relaxed loop precision but damaged `gate_02`, introduced one high-Z-after case, and severely degraded `hall_05` relocalization. Do not reintroduce it as a hidden threshold. The next viable loop redesign should build a `LoopReferee` evidence bundle and prove its recommendation on KITTI360 + M2DGR before it controls commits.

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
- vertical hypothesis diagnostics:
  - `vertical_hypothesis_count`
  - `best_z_offset_m`
  - `best_z_offset_fitness`
  - `zero_z_fitness`
  - `fitness_gap_zero_vs_best`
  - `z_hypothesis_spread_m`
  - `vertical_ambiguity_score`
  - `vertical_hypothesis_edge_recommendation`
- heightmap diagnostics:
  - `heightmap_overlap_cell_count`
  - `heightmap_overlap_ratio`
  - `heightmap_ground_dz_median`
  - `heightmap_ground_dz_p90`
  - `heightmap_ground_dz_max`
  - `heightmap_ground_support_ratio`
  - `heightmap_vertical_consistency_score`

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
- heightmap consistency fields

## Falsified Edge-Model Attempt

On 2026-06-12 a behavior experiment used the existing vertical-hypothesis recommendation directly:

- if `vertical_hypothesis_edge_recommendation == planar_xy_yaw`, downweight Z/roll/pitch and keep the loop edge.
- KITTI360 drive_0005 smoke, 450 frames, stride 5.
- artifact: `/tmp/n3mapping_kitti360_drive0005_vertical_hyp_edge_model_450_20260612_r1`

This did not pass the gate:

- accepted loops: 9, all GT-true, false loops: 0.
- edge modes: 2 `full6dof`, 7 `planar_xy_yaw`.
- `optimization_high_residual_z_after_count`: 3, worse than the previous 2.
- `optimization_max_residual_z_after_m`: 2.525, worse than the previous 1.348.
- `trajectory_z_p95_m`: 1.253, worse than the previous 1.077.

The key regression was loop `331 -> 63`:

- GT true loop distance: 1.64 m.
- ICP Z measurement error: about -2.38 m.
- vertical hypothesis recommendation: `planar_xy_yaw`.
- residual Z after optimization: about 2.53 m.

Conclusion: the current vertical-hypothesis recommendation is useful diagnostic evidence, but it is not sufficient as a direct behavior rule. Continuing by lowering planar vertical weight or adding another residual threshold would be parameter tuning. The next algorithm step needs a new runtime signal or a controlled tuning plan with a written acceptance/rollback gate.

## Heightmap / Ground Consistency Diagnostic

The next runtime signal is a local 2.5D heightmap consistency check. It does not change loop behavior.

For each ICP-converged loop candidate with loop debug enabled:

- target submap and ICP-transformed source submap are binned in target-frame XY cells.
- each cell uses a low Z percentile as a lightweight ground / low-surface proxy.
- only overlapping cells with enough points on both sides are compared.
- `heightmap_ground_dz_*` reports absolute low-surface height disagreement.

The intended use is to test whether bad ICP Z measurements are visible in local low-surface disagreement. If the signal separates bad-Z after-optimization cases from corrected/healthy cases, a later PR can use it for a structural edge-model change. It should not be converted directly into another hidden threshold without a matrix gate.

### 2026-06-12 KITTI360 Result

KITTI360 drive_0005, 450 frames, stride 5:

- artifact: `/tmp/n3mapping_kitti360_drive0005_heightmap_diag_450_20260612_r1`
- accepted loops: 8.
- false loops: 0.
- `optimization_high_residual_z_after_count`: 2.
- `optimization_max_residual_z_after_m`: 1.348.
- `trajectory_xy_p95_m`: 1.261.
- `trajectory_z_p95_m`: 1.077.

The behavior metrics match the previous diagnostic baseline, so heightmap diagnostics do not change loop behavior.

However, the signal did not separate the remaining bad-Z cases well enough:

- `accepted_true_loop_bad_z_heightmap_high`: 7.
- `heightmap_separates_bad_z_count`: 2.
- the two remaining bad-Z-after loops were flagged.
- the five corrected-Z bad-measurement loops were also flagged.

Conclusion: heightmap consistency is useful as another artifact column, but this simple low-surface heightmap is not sufficient to drive `planar_xy_yaw_neutral_vertical`. Do not implement the neutral-vertical edge model from this signal alone.

## Recommended next decision

Current evidence does not justify another hidden threshold tweak.

Ask GPT Pro to choose between:

1. Add new runtime evidence:
   - visibility/raycast consistency
   - ground/vertical segmentation
   - better ICP covariance/eigen analysis
   - inspect current multi-hypothesis vertical ICP evidence, then decide whether another signal is needed
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
   - done for KITTI360 / M2DGR eval metrics and matrix summary.
   - keep future relocalization changes false-lock first.

3. `vertical_observability_next_signal`
   - multi-hypothesis vertical ICP diagnostics are implemented for KITTI360 / M2DGR eval.
   - graph shadow trial diagnostics are implemented and preserve current behavior.
   - evidence correlation report is implemented to compare vertical hypothesis / heightmap / graph trial signals.
   - next step is to inspect correlation on KITTI360 + M2DGR before any behavior change.
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
