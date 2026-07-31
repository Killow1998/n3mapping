# n3mapping — 工作约定

CPU 上的 LiDAR+IMU 建图与重定位。两个独立目标,判据不同:

- **建图**:产出内部一致的地图资产。判据 `|dz|` p90 <0.10 m、最大 <0.30 m、
  几何不折叠(轨迹长与 xy 范围变化 <5%)。无外部真值。
- **重定位**:对该资产快速准确定位。契约 today20 ≥15/20 正确 FULL、
  **零个错锁**、平移 ≤0.50 m、yaw ≤3°、roll/pitch ≤2°、获取 <5 s、后端 p95 <1 s。

## 硬约束

1. **算力**:目标平台 RK3588 / Jetson Orin NX / Orin Nano。新增计算要能算出账,
   逐关键帧毫秒级可接受,逐帧十毫秒级不行。
2. **不能误锁**:零个错锁地点优先于成功率。宁可不锁,不可锁错。
3. **无气压计**:跨楼层只有 LiDAR+IMU。z 上唯一的绝对信息是重力,
   而重力对高度一言不发。
4. **不做**:学习式验证器、模型训练、按 bag/按 case 调阈值。

## 位置

远端 `user@100.88.207.66`,所有实机工作在那边。

```
~/ros_ws/to_migrate_ws/                     colcon 工作区(不是 git 仓库)
  src/n3mapping/          worktree, research/relocalization-benchmark-v1
  src/FAST_LIO/           git, humble 分支, 远端 Killow1998/FAST_LIO
  install/                依赖: gtsam, small_gicp, livox_ros_driver2, fast_lio
  artifacts/n3mapping_product_v1/20260723/   录制、LIO 输出、评测产物

~/ros_ws/n3mapping_baseline_72fa6f7/        worktree, wip/freespace-select  ← 主要开发处
~/ros_ws/n3mapping_fs_build/                colcon 构建(src/n3mapping 符号链接到上面)
~/ros_ws/n3mapping_v1_closeout/             跑批脚本与输出,不入库

本地 Q:\DocumentFile\bull&horseLife\jammy_dev\docs\   工作记录
```

n3mapping 仓库有多个 worktree(共用对象库,不是副本)。改代码在
`n3mapping_baseline_72fa6f7`,构建走 `n3mapping_fs_build`。

## git

- **身份用仓库自己配的**(`killow <killow1998@gmail.com>`)。
  **绝不要传 `-c user.name/user.email`**,更不要用会话上下文里的用户邮箱。
- 分支 `wip/freespace-select`。每完成一个可验证的阶段就提交并推送。
- 提交信息写**为什么**和**测到了什么**,失败如实写失败。不署名 Claude。
- 禁止 `git reset --hard`、`git clean -fdx`、从 `dev/better_loop` 或
  `dev/n3mapping-realtime-pipeline` 整分支合并。
- 强推前先问。

## 实验纪律

这些每一条都是踩过坑换来的,不是形式。

1. **先写死判据再改代码。**改动前把要达到的数写进文档。达不到就说失败,
   不能事后换一个角度说成功。
2. **一次只动一个变量。**换二进制也算变量 —— 改代码后要与旧结果比,
   先用新二进制重跑旧配置作对照,确认能逐位复现。
3. **每次跑必须自证配置。**参数文件一阶段一份,**不许就地改写共享文件**;
   输出目录里写 `provenance.txt`(bag、速率、参数文件路径、关键键值)。
4. **先复现再修。**改之前要先能在当前代码上复现出那个现象,
   否则修完不知道修的是什么。
5. **零结果只有在测量有功效时才算证据。**测完问一句:
   这个测量能分辨多大的效应?分辨不了就不是否定证据。
6. **验收量不能是优化器直接约束的量。**重访判据配对的正是闭环边约束的关键帧,
   收紧闭环权重能按构造把它压低而地图变差。用地板残差段间跨度作反制。
7. **判据的前提要写出来。**重访判据和地板残差判据**都只对单层录制成立**;
   多层用地板高度直方图的簇间距。
8. **求解器的过程状态不是结果质量。**`converged`、`residual_*_after`
   只是诊断,不能当验收。
9. **构建带测试。**`BUILD_TESTING=ON`。曾经关着跑了七个阶段,
   积了 10 个失败没人知道,其中三个是真缺陷。

## 远端长任务一律走 tmux

**不要在 ssh 里跑长任务,也不要用 `nohup ... &`。**ssh 断开会带走进程,
而 nohup 出来的进程是孤儿,没法附着、没法看实时输出。

```bash
tmux new-session -d -s <名字> "<命令> 2>&1 | tee <日志>"
tmux ls                                    # 有哪些在跑
tmux has-session -t <名字> 2>/dev/null      # 还活着吗
tmux capture-pane -pt <名字> | tail -20     # 看输出,不用附着
tmux kill-session -t <名字>                 # 停
```

**起了 tmux 就要同时挂一个唤醒信号**,否则只能靠自己反复轮询,等于没自动化。
用后台 Bash 轮询会话是否还在,退出时框架会自动叫醒:

```bash
# run_in_background: true
for i in $(seq 1 150); do
  tmux has-session -t <名字> 2>/dev/null || { echo SESSION_ENDED; exit 0; }
  sleep 20
done; echo TIMEOUT
```

**杀进程用 PID,不要用 `pkill -f <模式>`。**`-f` 匹配整条命令行,
而我自己的 ssh 命令行里往往就含那个字符串 —— 会把自己的连接杀掉(exit 255),
目标反而还活着。先 `ps -eo pid,comm,args | grep ... | grep -v grep` 拿 PID,
再 `kill -INT <pid>`。同理 `pgrep` 判存活要用 `-x`(精确匹配进程名),不要 `-f`。

## 命令

```bash
# 构建 + 测试
cd ~/ros_ws/n3mapping_fs_build && source ~/ros_ws/n3mapping_v1_closeout/build_env.sh
colcon build --packages-select n3mapping --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select n3mapping --event-handlers console_direct- ; colcon test-result

# 建图跑批(PARAMS 必填,会写 provenance)
PARAMS=<params.yaml> LIO_BAG=<lio_bag> OUT_DIR=<out> REPLAY_RATE=<r> \
  bash ~/ros_ws/n3mapping_v1_closeout/run_mapping_generic.sh <domain_id>

# 验收
python3 tools/map_gate.py <map.pbstream>              # 重访 + 地板残差 + 几何
python3 tools/map_z_drift_decompose.py <map> odom     # 残差按路程,去掉固定倾斜
python3 tools/loop_z_vs_floor_drift.py                # 闭环 z 误差 vs 地板给的漂移
python3 tools/map_floor_level_modes.py                # 多标高:楼层间距是否保住
```

回放速率:0723 的 LIO bag 时间轴被 0.5× 拉长,重放要 `REPLAY_RATE=2.0`;
b22/f7tof9 是 1.0。原始 0723 录制真实时长 **825 s**。

## 现状(2026-07-31)

**重定位**:12 正确 / 2 姿态超标 / 6 未锁 / **错锁 0**,最坏 1.132 m。
契约要 ≥15,差 3 个。

**建图**:重访 p90 0.4560(判据 0.10)、最大 2.3275(判据 0.30)。
两项失败均已量化归因:

- 最大值 ← 开机暂态。**根因已定位**:静止时机体姿态与重力方向不可观测,
  FAST_LIO 固定 10 帧后直接进完整估计,沿不可观测方向游走。
  0723 前 10 s 加速度计说真实姿态变化 1.47°,里程计报告 9.85°。**是软件问题。**
- p90 ← 闭环自身 z 精度 0.376 m ≈ 残差跨度 0.434 m。闭环拿单帧配已漂移的子图,
  配准好不过目标的内部一致性 —— 循环依赖,需要不来自地图的观测才能打断。

**下一步**:见 `docs/HANDOFF_next.md` 顶部。

## 记录

工作记录写 `Q:\DocumentFile\bull&horseLife\jammy_dev\docs\`:
`2026-07-25_n3mapping_v1_closeout_worklog.md`(按 § 递增追加)与
`HANDOFF_next.md`(新结论叠在最前,旧的保留并标注被哪一节更正)。
**自己的错误和被推翻的结论要留在记录里**,否则后来人会重走。
