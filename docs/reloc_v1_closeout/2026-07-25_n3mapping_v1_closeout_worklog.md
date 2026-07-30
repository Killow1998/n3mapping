# N3Mapping 传统 CPU 重定位 V1 收口 — 工作记录

日期：2026-07-25
执行环境：`user@100.88.207.66`（Ubuntu 22.04 / ROS 2 Humble / 20 核 / 31 GiB RAM）
输入合同：`n3mapping_cpu_v1_closeout_handoff_v3.zip`（MANIFEST.sha256 全部校验通过）
基线：`research/relocalization-benchmark-v1` @ `72fa6f7e3fc22eae5bfc30700bc1e2cec7f7c4eb`

**当前候选提交：`bc20b36c4d9e4247662834ce1a34f41adc6fc5c0`**
状态：`IMPLEMENTED_UNVERIFIED` → **仍然是 `IMPLEMENTED_UNVERIFIED`**。
已完成 R0–R2；R3–R7（正式数据链与 27-case Gate）未执行，不能宣称 `V1_PASS`。

---

## 1. 关键路径一览

| 阶段 | 内容 | 结果 |
|---|---|---|
| R0 | 保全 + 工作区隔离 | PASS，回收 971 MB |
| R1 | 临时 worktree 重建 dirty 候选 | PASS，numstat/untracked 逐字节一致 |
| C1–C6 | 按职责拆分 7 个可审查提交 | PASS，51 tracked + 9 untracked 全覆盖 |
| C6 必修项 | `kSampleStrideFrames=10` → scan-end 时间采样 | **已修复并测试** |
| C7 | refine resolution 0.05→0.1 | **按合同暂缓**，导出为 patch |
| R2 | exact-commit clean Release 构建 + 全量测试 | PASS，373/373，install audit 0 hard-forbidden |
| R3–R7 | 正式数据链 / Gate / main 合入 | **未执行** |

## 2. 目录与产物

远端新增（均不在源码树内）：

```
~/ros_ws/n3mapping_product_v1_rebuild/          # 候选 worktree，分支 wip/product-v1-closeout-0725
~/ros_ws/n3mapping_v1_build/                    # 独立 clean prefix（build/install/log）
~/ros_ws/n3mapping_v1_closeout/
    handoff_v3/                                 # 交接包（sha256 已校验）
    backup_live_0725/                           # 原 dirty worktree 的第二份备份
    C7_deferred_refine_resolution.patch         # 暂缓的 C7 数值变更
    install_audit_candidate.json                # 候选 install prefix 审计结果
    build_env.sh                                # 复现构建/测试环境
    cleanup-plan-0725.json / cleanup-apply-0725.json
~/ros_ws/.quarantine/n3mapping-20260725/        # 隔离物（927 MB，未删除）
```

原 dirty worktree `~/ros_ws/to_migrate_ws/src/n3mapping` **未被修改**，仅移走了 allowlist 缓存。

## 3. R0 — 保全与清理

先做保全再动任何东西：

- 从**活工作树**重新生成 patch + untracked 归档，与交接包 `recovery/` 比对：
  `tracked_diff_numstat` 完全相同，`untracked_sources` 完全相同（51 文件 / +7526 / -354 / 9 个 untracked 路径）。
  即交接包与机器上的实际状态一致，可以放心执行。
- `n3mapping_workspace_cleanup.py scan` → `apply`，8 个 allowlist 项移入同盘 quarantine：
  `.vscode/browse.vc.db`（971 MB）、`__pycache__`、`.pytest_cache`、repo 内 `build/install/log`、`.distro_wrapper_selection`。
- 清理后复核：tracked diff 不变、untracked 源码清单不变、`git diff --check` 通过。
  `map/n3map.pbstream`、`map/global_map.pcd` 保持原位（脚本标为 `asset_review`，未自动移动）。
- 仓库体积 1.1 GB → 182 MB。

**未处理（需你决策）**：`to_migrate_ws/` 下的 `build` 368M + `install` 38M + `log` 717M。
没有隔离它们，因为 `install/` 里还有 FAST_LIO / gtsam / small_gicp / tgw_planner 等你当前在用的包；
本次候选构建走的是**独立新 prefix**，已经满足"不信任旧 prefix"的要求。要回收这 1.1 GB 请单独确认。

`artifacts/` 91 GB 未触碰（策略要求只生成 review plan，不自动移动）。

## 4. R1 — 重建

```bash
git worktree add -b wip/product-v1-closeout-0725 \
  /home/user/ros_ws/n3mapping_product_v1_rebuild 72fa6f7
git apply recovery/dirty_worktree.patch
tar -xzf recovery/untracked_sources.tar.gz -C .
```

`git diff --check` PASS；重建后的 numstat 与 untracked 清单与恢复包完全一致。

## 5. C1–C6 — 提交拆分

按 `16_DIRTY_FILE_ROUTING.md` 做 **hunk 级**路由。跨组文件用自写工具
`~/ros_ws/n3mapping_v1_closeout/stage_hunks.py`（按 hunk 甚至按新增行选择性 `git apply --cached`，
不碰工作区，新侧偏移逐条重算，等价于 `git add -p` 的非交互版本）。

51 个 tracked 文件全部有明确归属，无遗漏、无重复：

| 提交 | 类别 | 规模 |
|---|---|---|
| `f840b90` data: raw-to-LIO V3 证据与 Gate 输入身份 | measurement/data | 17 files, +5788/-215 |
| `905812d` interface: 三路 exact 输入与 fail-closed authority | interface | 24 files, +806/-50 |
| `618198c` format: Atlas v4 与 Product SC materialization 隔离 | format | 18 files, +1979/-32 |
| `0f2b1a8` algorithm: RHPD place distance yaw 不变 | algorithm/retrieval | 7 files, +657/-7 |
| `d2cba89` algorithm: gravity 约束 xyz+yaw 配准 | algorithm/registration | 3 files, +601/-14 |
| `022c19b` authority: 唯一 mode 才给 Product FULL | algorithm/authority | 6 files, +1359/-35 |

几处需要留意的路由判断（与 routing 文档略有出入，均基于实际 diff 内容）：

- `humble/noetic CMakeLists.txt` 移除 `Scancontext.cpp` 归入 **C3**（SC 隔离），不是 C1/C2。
- `tools/n3mapping_relocalization_manifest_eval.cpp` 全部归入 **C2**：内容是把 gravity 灌进 `LioFrame`，属输入契约。
- `tools/n3mapping_product_bundle.py` 全部归入 **C3**：实际内容是 Atlas 元数据解析与 exclusion provenance 绑定。
- `src/config.cpp` 的单个 hunk 被**按行**拆成 C4（sc_aux/rhpd_use_sc_yaw）与 C7（refine resolution）。
- `tools/n3mapping_product_gate.py` 39 个 hunk 拆到 C1/C2/C3/C6 四个提交；
  `test_n3mapping_product_gate.py` 的一个 hunk 按行拆成 C1(+1..93) / C3(+94..155)。

## 6. 拆分中发现的三个真实缺陷（已修）

交接包声明的验证只做到 `py_compile`（17/17）——语法级。实际跑测试后暴露：

### 6.1 `53628b7` V2→V3 单例迁移 API 从未实现

`test_n3mapping_raw_lio_evidence_v2_to_v3.py` 有 4 个测试引用 `migrate_one` / `verify_one`，
但工具里只有 `migrate_batch` / `verify_batch`。在原 dirty worktree 上同样 4 fail，**属既有缺陷不是拆分产物**。

已按测试规定的契约实现：

- batch 口径拥有 today20 合同（恰好 20 个时间戳 case、12/8 replay-tool 分布）；
  单 case 不具备这些，因此 witness 绑定到**该 case 自己 parent V2 证据里记录的** replay-tool SHA-256，
  而不是 batch 级常量。
- 其余不变量保持一致：不重放 bag、不覆盖已存在的 child/lineage、case_id 必须是安全且等于 prepared 根目录名、
  昂贵 rescan 只在廉价拒绝全部通过后才启动。
- 新增 `migrate-one` / `verify-one` 子命令。
- 结果：10/10 通过，连跑两次一致。

### 6.2 `9cb9175` product runtime 测试的 Atlas fixture 不可解析

C3 让 Bundle 直接解析 Atlas protobuf 后，`test_n3mapping_product_runtime.py` 的
`write_bytes(b"atlas")` 变成非法 protobuf，5/5 全挂（`atlas protobuf field exceeds file size`）。
原 dirty worktree 同样 5/5 fail。改为写入真正的 length-delimited field 1。

### 6.3 `bc20b36` 产品 install prefix 内容不正确

对全新 research-OFF Release prefix 跑交接包的 `audit_install_prefix.py`：**verdict FAIL**。

- hard-forbidden：`include/n3mapping/synthetic_relocalization_query.h` 被 include/ 的通配安装带进产品 prefix。
  该头文件只被 research 工具和其测试使用 → research OFF 时排除。审计 FAIL → PASS_WITH_REVIEW，hard-forbidden 归零
  （余下 72 项 review_only 是内部头 + ScanContext 头，属交接包明确允许的 review 类别）。
- 产品数据链只装了一半：注册了 exclusion windows 工具，但 V2→V3 迁移、today20 preparation audit、
  归档 replay byte witness 都没装，部署 prefix 无法自证 raw-to-LIO V3 链路。三者已为两个 distro 补齐。

## 7. `fb8c2a9` — P0 必修项：帧率无关的采集采样

### 问题

`WorldLocalizing::relocalizeFromAtlas()` 里 `kSampleStrideFrames = 10`，注释假定"一秒一采样"。
这把 10 Hz 变成了隐含的产品算法条件：同一段数据 5 Hz 时采样间隔变成 2 秒，20 Hz 时变成 0.5 秒，
`<5s` acquisition latency 合同和跨平台行为都无法验证。

### 改法

`WorldLocalizing::relocalize()` 显式接收该 cloud/odom/gravity 三元组的 scan-end 时间戳：

- 第一帧被接受的 query 锚定窗口，其后第 n 个采样在 `window_start + n 秒` 到期。
  锚在窗口起点而不是上一次采样，晚到的帧无法拖动整个时刻表，不同输入速率落在同样的采样戳上。
- 时间戳回退或重复 → 放弃窗口，不把两个观测时刻折叠成一个。
- 间隔 ≥ 500 ms（产品输入 watchdog 周期）→ 放弃已累积的 place 证据并重新锚定，
  不让一个窗口跨过 gravity / odometry authority 的断点。
- 未提供时间戳时 Atlas 路径 fail-closed（`atlas_query_stamp_invalid`），
  而不是按猜测的帧率去排程。legacy 非 Atlas 调用方不受影响（默认 `kUnknownStampNs`）。
- 每个被接受的采样记录自己的 stamp；`n3mapping_core` 透传 `LioFrame::stamp`。

### 测试

原有 3 个 Atlas 窗口测试从"第几次调用"改写为"第几个时刻"，另加 4 个新测试。
`test_world_localizing` 30/30 通过。

两个测试设计上的坑，已在代码注释里写明：

1. **采集不是从第一帧开始的**。Atlas 路径要先有聚合 query cloud，实测需要 2 帧预热；
   而且 `finishAtlasWindow()` 会清空 query history，所以每次窗口被放弃后都要重新预热。
   因此新增 `warmUpAtlasQueryHistory()`，用**与速率无关的绝对时间戳**预热，
   让 5/10/20 Hz 三条线都恰好在同一个 `kAcquisitionBaseStampNs` 进入采集——
   这样"采样时刻相同"才是在测采样调度本身。该 helper 会断言预热帧仍是 `atlas_query_history_pending`，
   history 策略若变更会立刻报错而不是静默比错东西。
2. **第三个采样在同一次调用内就被清掉了**：它触发终局判定 → `finishAtlasWindow()`。
   所以 5/10/20 Hz 等价性测试对前两个采样戳做直接比对（三者均为 `{t0, t0+1s}`），
   第三个采样的时刻通过"+2.0 s 处出现终局 decision"来观测。

### 遗留的一处帧率相关性（未修，需你定夺）

采集**起点**仍然取决于帧数：需要约 2 帧 query history 才能进入 Atlas 路径。
5 Hz 下这是 400 ms，20 Hz 下是 100 ms，即锁定时刻会有 ~0.3 s 的速率相关偏移。
这在 query 聚合层（`reloc_static_agg_*`），是显式配置项而不是硬编码假设，
且交接包的必修项只点名了采样 stride，所以本次没有一并改。
若 `<5s` 合同要做到严格跨速率可复现，这是下一个该动的地方。

## 8. C7 — 按合同暂缓

`icp_refine_downsampling_resolution: 0.05 → 0.1`（含 `src/config.cpp` 与 bundle 测试期望）。

任务卡 C7 的验收要求"可测量的 Atlas/RSS 收益 + 完整行为指纹等价"，rollback 规则是
"收益或等价性未被证明就丢弃该提交"。这两项证据都需要真实地图与 Atlas compile/RSS 测量（R3 范畴），
本次拿不到，因此**没有提交**，导出为：

```
~/ros_ws/n3mapping_v1_closeout/C7_deferred_refine_resolution.patch
sha256 8a2f5cbda3fcee4b787673ae344f3baf05ef89b81dfea0e0918cc36042bddcb9
```

已验证该 patch 仍能干净地应用到当前候选提交。

注意：C5 里的 prepared-level alias **结构**代码已经在候选中（resolution 相等时复用同一层）。
0.05 下它是惰性的；C7 只是启用它的那个数字。所以丢弃 C7 不会留下半截实现。

## 9. R2 — exact-commit clean build

```bash
source ~/ros_ws/n3mapping_v1_closeout/build_env.sh
cd ~/ros_ws/n3mapping_v1_build && rm -rf build install log
colcon build --packages-select n3mapping --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DN3MAPPING_BUILD_RESEARCH_TOOLS=OFF \
  -DBUILD_TESTING=ON \
  -DN3MAPPING_PRODUCT_COMMIT=bc20b36c4d9e4247662834ce1a34f41adc6fc5c0 \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
colcon test --packages-select n3mapping
```

| 检查项 | 结果 |
|---|---|
| Humble Release clean build（独立 prefix） | PASS（1 min 22 s，仅 deprecated proto 字段告警） |
| 全量 CTest + Python 产品测试 | **373 tests, 0 errors, 0 failures**，连跑两次一致 |
| product commit identity 写入 `n3mapping_node` | = `bc20b36…`（精确候选 SHA） |
| install prefix 审计 | `PASS_WITH_REVIEW`，hard-forbidden = 0 |
| node / evaluator 加载同一 `libn3mapping_core.so` + 唯一 wrapper | PASS |
| 构建期"干净源码树"守卫 | 生效（我中途改 CMake 后它直接拒绝构建，符合设计） |

顺带一提：`-DN3MAPPING_PRODUCT_COMMIT` 的守卫是真的——源码树不干净时直接
`A verified Product V1 binary requires a clean source tree` 报错退出，这条防线工作正常。

### Noetic 侧未构建

本机 `/opt/ros` 只有 `humble`。Noetic 的 conversions/CMake 改动已按合同拆进 C2/C3，
但**没有编译过**，`test_noetic_conversions.cpp` 也没跑过。R2 的"两个 distro 分别 clean build"因此只完成一半。

## 10. 还没做的事（按优先级）

1. **Noetic clean build + `test_noetic_conversions`** —— 需要一台有 ROS 1 Noetic 的机器。
   这是 R2 验收里唯一缺的一块。
2. **R3 正式 raw-to-LIO → Atlas 链路**（1 条 full + 1 条 query）：
   ROS1→ROS2 v5 逐字段等价 → fresh FAST_LIO → V3 evidence → 15 列 manifest → mapping replay 计数闭合
   → dense trajectory → exclusion windows → Atlas v4 compile → Bundle verify。
   数据在 `artifacts/n3mapping_product_v1`（71 GB）与 `n3mapping_relocalization_benchmark`（21 GB）里，
   但要跑起来需要确认冻结的 bag/标定路径，且要按合同用 systemd `MemorySwapMax=0`。
3. **R4 完整 27-case Gate**（today20 20 + floor7 legacy 5 + 2 个 must_reject）。
   顺序按合同：先跑两个负例和两个历史最难正例做安全/时延烟雾，再跑完整 matrix。
4. R5/R6/R7：Gate 驱动的单根因修复 → formal candidate Gate → main 合入。
5. C7 的等价性与资源证据（依赖 R3 的 Atlas compile/RSS 测量）。
6. 采集起点的帧数相关性（见 §7 遗留）。

## 11. 边界声明

- **没有 push、没有建 PR**，符合合同要求（需 Gate PASS + 你明确授权）。
- 分支 `wip/product-v1-closeout-0725` 是恢复/拆分工具，不是长期 release 分支；
  按合同应在确认后删除。原 dirty worktree 在下列条件全部满足前不要清理：
  C1–C6 已重建（✅）、hash 与提交映射已记录（✅ 本文档）、clean 测试通过（✅ Humble，❌ Noetic）、
  至少一条 formal data chain 跑通（❌）、你确认没有遗漏本地改动（❌）。
- 当前**不能**宣称 75% 产品正确锁定率，也不能宣称 `V1_PASS`。
  已完成的只是：候选可重建、可审查、可从精确提交 clean 构建、单元/契约测试全绿。
  真实定位正确率、精度、时延、RSS 全部未测。

---

# 续：2026-07-25 第二段（候选推进到 dd25f86）

## 12. 读取任何时间戳前必须知道的一件事

**机器时区是 `America/Los_Angeles`（PDT），但 n3mapping 的 Python 工具在代码里硬编码
`SHANGHAI_TZ = +08:00` 写时间戳。两者差 15 小时。**

例：`gate_result.json` 里 `started_at: 2026-07-23T20:28:12+08:00`，
实际就是机器上的 `Jul 23 05:28 PDT`，和文件 mtime 完全对得上。

对照任何 artifact 的 JSON 时间戳与文件 mtime 时，先做这个换算，否则会得出"有人在事后改文件"
这类错误结论（我就犯过一次）。

这本身是个小设计问题：产物时间戳与机器本地时间不一致，除非读的人知道这条规则。
不算 V1 阻塞项，但值得记一笔。

## 13. `dd25f86` — 禁止在源码树内构建的守卫

**问题**：`src/n3mapping/` 下曾出现 `build`/`install`/`log`（R0 时已隔离，均为空壳）。
这不只是脏：它让 `git status --porcelain --untracked-files=all` 永久非空，
而这正是 `N3MAPPING_PRODUCT_COMMIT` 用来判定"可验证产品构建"的那条门。
换句话说，**在源码树里编译会静默废掉产品身份守卫**。

**改法**：在 `cmake/n3mapping_product_identity.cmake` 最前面 configure 期直接 fatal error。
检查两种形态，缺一不可：

- **解析后路径**：抓"构建目录真的在源码树里"，包括源码树本身是通过 symlink 访问的情况
  （colcon 工作区 `src/n3mapping -> ...` 就是这种，是正常用法，不能误杀）；
- **字面路径**：抓"构建目录是个放在源码树里、指向别处的 symlink"。
  产物落在外面，但那个链接本身仍在仓库里，仍然把树弄脏。

前缀按字面比较，路径里的正则元字符无法削弱检查。

**验证**（三种情况都实测过）：

| 情况 | 期望 | 实测 |
|---|---|---|
| 构建目录直接在源码树内 | 拒绝 | 拒绝 ✅ |
| 构建目录是源码树内的 symlink（指向 /tmp） | 拒绝 | 拒绝 ✅ |
| 正常 colcon 工作区（src/n3mapping 是 symlink） | 通过 | 通过并完成构建 ✅ |

候选提交推进到 **`dd25f86a7779eb7bff1715e48580b16054593354`**，
clean Release 重建 + 全量测试 **373/373 通过**。

## 14. 上一轮 Gate 中断的真实原因（systemd journal）

`candidate_72fa6f7/formal_gate_v2_minimal` 的 `status: INCOMPLETE` 不是崩溃、不是 OOM、
也不是重启（机器自 7/13 起未重启）。journal 记录：

```
Jul 23 06:16:10  Stopping ... n3mapping_product_gate.py ...
Jul 23 06:16:10  n3mapping-72fa6f7-formal-gate-v2.service: Killing process 1132590 with SIGKILL
Jul 23 06:16:10  ...service: Consumed 6h 31min 21.032s CPU time
```

**真相：第 11 个 case `f7tof9_query_on_b22_no_overlap_end_5s` 挂死了。**
前 10 个 case 每个 30–60 秒（05:28→05:36），它从 05:36 空转到 06:16，
**卡了 40 分钟、累计吃掉 6.5 小时 CPU（约 8 核满转，是活跃空转不是阻塞）**，然后 service 被 stop。

佐证：06:17 日志里紧接着就是重建负例窗口
`build_minimal_gate_windows.py ... minimal_gate_windows_valid_motion_v2 --window-frames 50`。
当时的诊断显然是：那个 5 秒尾切片是退化/静止窗口，把 evaluator 转死了，于是重造了 "valid_motion" 版本。

**结论与对策**：

1. 存在**真实的挂死风险**，触发条件是退化（静止/无有效运动）的短窗口负例。
   "10 case / 8 分钟 → 27 case / 25 分钟"的外推**不成立**。
2. Phase 2 起改用 `minimal_gate_windows_valid_motion_v2`（6 个负例子窗口都在）。
3. 所有长任务一律 systemd/nohup 后台 + 全量日志 + 记录退出码，**并加 per-case 超时**，
   让挂死立刻可见，而不是静默吃掉 40 分钟。
4. 这也解释了为什么之前的运行用 systemd user service：
   合同要求的 `MemorySwapMax=0` 就是这么落地的，沿用同一模式。

## 15. Phase 1 进行中

目标：在不重放 bag 的前提下闭合数据链。

已确认的输入拓扑：

| 角色 | 用途 | 地图 |
|---|---|---|
| `f7_0723` | today20 的 20 个 required_lock | `0723_derived/f7_0723_map/n3map.pbstream` |
| `floor7` | legacy 5 个 required_lock | `candidate_72fa6f7/bundles/floor7/map.pbstream` |
| `b22` | 两个 must_reject | `candidate_72fa6f7/bundles/b22/map.pbstream` |

只有 `f7_0723` 需要 exclusion windows —— Gate 只对"有 raw evidence 的 required_lock /
ambiguity_allowed"角色强制要求，负例和 legacy 不在此列。

**已就绪**：today20 的 20 份 V3 evidence + 迁移 lineage + 15 列 gravity manifest 全部存在，
`full/` 同样是 V3 + 15 列。**不需要重跑 FAST_LIO。**

**进行中**：从 20 份 V3 evidence 生成 exclusion windows。
该工具拒绝以 LIO manifest 为输入（防止 FAST_LIO 冷启动丢帧缩小窗口），
必须扫原始 bag 取每段最后一帧的 `max(point.offset_time)`，因此耗时以十几分钟计。

**待办**：Atlas v4 编译（`/usr/bin/time -v` 记峰值 RSS，硬门 600 MiB）→ Bundle create/verify。

值得先记一个预期：现有 `f7_0723` 的 v1 Atlas 是 **245 MB**，而地图 pbstream 只有 62 MB。
v4 去掉了 per-anchor 点云负载，这里应当看到明显收缩 —— 这正是 P2 内存余量问题的正面战场。

---

# 续二：today20 首次测量与根因定位

## 16. 数据集范围（用户决定）

只用 0723 那次录制的 global 地图 + 20 个 query。floor7 / b22 / f7tof9 暂不作为数据集。

理由（实测确认）：

- floor7 的 5 个 case 是**同一段约 13 分钟连续录制**（2026-03-13）切出来的时间窗口，没有独立 bag。
  拿建图那次录制的切片去匹配同一批扫描建的地图，很大程度是自我匹配。
- b22 只有整图 bag，负例窗口是 `build_minimal_gate_windows.py --window-frames 50` 切的。
- 三组的 manifest 全是 **11 列 legacy，无 gravity**。C2/C6 之后无有效 gravity → Atlas 路径 fail-closed
  → 负例"通过"是**假通过**，只证明输入契约拒绝了它，没证明算法有判别力。

**重要修正**：today20 也不是独立录制。20 个 query 全部落在建图那次遍历内部
（395.6 s / 824.9 s，占 48%）。它成立完全依赖 exclusion windows 把这些区间从 anchor 里排除。

## 17. 又一个真实缺陷：exclusion windows 从未工作过

Python 生产者用 `csv.writer` 写 **CRLF**（RFC 4180 合法），C++ 消费者用 `std::getline` 只剥 `\n`，
header 变成 `"start_stamp_ns,end_stamp_ns\r"`，字面量比较必然失败。

**即 C1 的生产者和 C3 的编译器从未对接跑过。** 而这正是让 today20 成为留出集的那个功能。

修法：header 走本文件已有的 `splitCsv`（它本来就剥 `\r`），与文件内其它 reader 一致。
不改生产者，因为那会让 `windows_sha256` 变化且 CRLF 本身合法。提交 `4af2ae6`。

## 18. Phase 1 完成：可信的测量装置

| 步骤 | 结果 |
|---|---|
| mapping replay（gravity_v2 LIO → 地图） | 8250 帧 = manifest 8250 行，**计数闭合**；257 keyframes，2 回环 |
| dense trajectory | 8250 行 = manifest 8250 行 ✅ |
| exclusion windows | 20 个窗口，PASS |
| Atlas v4 | **521 anchors，gravity 有效 521/521**；两个 provenance 指纹与生产者一致 |

**为什么必须重建地图**：两次 FAST_LIO 跑同一段 bag，轨迹最大偏离 **0.416 m / 1.800°**，
而产品容差是 0.50 m / 2.0°。参考与被测若来自不同次运行，量到的是噪声不是算法误差。

这**不是**要求 LIO 可复现（产品必须对 LIO 变动鲁棒）。这是要求**参考与被测同源**，否则无法归因。

## 19. today20 首次测量结果（候选 `4af2ae6`）

参考类别 `same_run_dense_optimized_reference` —— **不是独立 GT**。

| | 数量 |
|---|---|
| 锁定 | 9 / 20 |
| 其中在容差内（0.50 m / 3° yaw / 2° roll-pitch） | **1** |
| **其中超出容差 = 错误 FULL** | **8** |
| 未锁定 | 11 |

```
case          trans_m  yaw_deg roll_deg pitch_deg
11-51-17        0.141    1.130    0.955    0.046   ← 唯一合格
11-52-00        0.022    0.490    0.718    3.358
11-52-34        0.107    3.052    7.889    3.152
11-48-26        0.569    4.328   10.970    3.045
11-58-18        1.877  179.878    5.237   12.758   ← 180° 翻转
11-54-55        2.417   67.866    1.276    1.492
11-47-53        4.926    4.499   11.018    4.159
11-58-53       32.668  123.179    8.110    5.353
11-55-38       36.163   88.231    5.264    2.395
```

资源侧全部宽裕：峰值 RSS **588 MiB**、strict p95 最大 **170 ms**。
（用户口径：目标是 RK3588 / Orin NX / Orin Nano 能正常跑，600 MiB 不作为硬约束。
 因此 C7 的"省内存"理由作废，继续搁置。）

采集时延普遍超 5 s（4.6 / 6.9 / 7.1 / 9.2 / 9.3 / 9.4 / 9.4 / 16.2 s）——次要矛盾。

## 20. 检索不是问题（预先写死的判据）

跑测量前先写死判据：top-10 命中 ≥ 15/20 就继续修下游，< 10/20 就支持推倒重来。

`localization_anchor_recall_probe.csv`（0723 已有）：**20/20 全部在 rank ≤ 4**
（rank1×16、rank2×2、rank3×1、rank4×1），最佳 anchor 距真值 0.02–1.36 m。

新 Atlas 上的诊断进一步确认：失败案例 11-55-38 的最近 proposal 是
**anchor 6030，距真值 0.04 m**，且在**每一个**窗口里都在提案集内。

**结论：描述子与检索是好的。问题全部在下游。**

## 21. 根因：唯一性判定在数**锚点**，不是数**物理位置**

给产品主路径补了 measurement-only 诊断（提交 `979a745`）后，机制完全暴露。

失败案例 11-55-38，四个完成窗口：

```
window 1  decision=ambiguous  common_modes=7
  seed 6294  dist_to_truth= 0.55 m  first=3 latest=8
  seed 6030  dist_to_truth= 0.04 m  first=3 latest=8   ← 正确位置
  seed 6023  dist_to_truth= 0.60 m  first=2 latest=8
  seed 7564  dist_to_truth=37.42 m  first=3 latest=9   ← 强别名
  seed 7557  dist_to_truth=37.05 m  first=3 latest=9
  seed 7545  dist_to_truth=36.35 m  first=3 latest=9
  ...
window 4  decision=accepted  common_modes=1  → 锁到 36 米外
```

关键结构事实：

1. **同一个物理位置在 Atlas 里由 5–7 个 anchor 表示**（6030 / 6294 / 6023 / 6017 都在 1 m 内）。
   这是 anchor 覆盖采样器的设计：`kTranslationCoverageM = 0.5`、`kYawCoverageRad = 10°`。
2. 每个 anchor 各自成为一个 region，各自贡献若干 mode。
3. `common_modes` 数的是 **(region, mode) 对**，不是**物理上不同的位姿**。
4. 于是正确位置**天然**贡献多个 mode → 恒为 `ambiguous` → 不锁。
5. 偶尔计数塌缩到 1（取决于这一轮哪些 region 的 ICP 恰好没过），**幸存者直接获得 FULL 授权**，
   而它是不是正确位置是随机的。

这一条同时解释了两个现象：

- **11/20 不锁**：正确位置产生多个 region → 判为 ambiguous；
- **9 次锁定错 8 次**：幸存者是偶然的，37 m 处的强别名（7564/7557/7545）贡献的 mode 甚至比正确位置更多。

连唯一合格的 11-51-17 也是运气：它的四个窗口里前三个 common_modes = 2/3/6 全被拒，
第四个恰好塌缩到 1 且恰好是正确的那个。

**代码注释里写的意图是对的**（"complete-link equivalence so a chain of near poses cannot
collapse two product-distinct endpoints"），但等价性合并发生在 `intersectPhysicalModes`
的 region 对内部，**没有跨 region 做**。

## 22. 修复方向（待用户确认，未实施）

不是加复杂度，是**减**：

> 在唯一性判定之前，把**所有 region 的候选 `T_map_odom` 按物理等价性聚类**
> （平移 + 绕重力轴 yaw），然后数**簇**的个数，而不是数 (region, mode) 对。
> 一个簇 → 锁定；多个物理上分离的簇 → REGION_HYPOTHESIS。

这正是 "physical mode" 本来的语义。当前实现把"同一地点的多个锚点"误算成"多个物理假设"。

预期效果可证伪：
- 正确位置的 5–7 个 anchor 会并成 1 簇 → 大量 `ambiguous` 消失 → 锁定率上升；
- 37 m 外的别名簇与正确簇物理分离 → 仍然计为 2 簇 → 继续拒绝，**不会**变成错误 FULL。

修完必须重跑这 20 个 case 验证，判据：错误 FULL 必须为 0，正确 FULL 数上升。

## 23. 两次被推翻的修复假设（记录下来，避免重犯）

**假设 A（错）**：等价性合并只在 region 内部做，没跨 region。
读代码后推翻：`intersectPhysicalModes` 的 `clusters` 就是跨所有 region 的扁平列表，
而且用的是 complete-link。**下结论前必须读实现，不能只看行为。**

**假设 B（错）**：同一物理位置的多个 anchor 被误算成多个物理 mode，聚类合并即可。
把每个 mode 的 `T_map_odom` 打出来后推翻：

```
anchor 6294/6030（正确位置，距真值 0.04–0.55 m）
  first  0  [-34.649 -5.591 -2.168]  yaw= 171.475   ← 真值（差 0.05 m）
         1  [-32.763 -5.605 -2.137]  yaw= -22.059   ← 别名，1.9 m 外，Δyaw 193°
         2  [-33.618 -6.101 -2.049]  yaw=  -3.030   ← 别名，1.1 m 外，Δyaw 175°
  latest 0  [-34.734 -5.536 -2.038]  yaw= 171.595   ← 与 first 差 0.10 m / 0.12°
         1  [-32.706 -5.670 -2.047]  yaw= -22.393   ← 与 first 差 0.08 m / 0.33°
```

这些不是"同地点的多个锚点"，是**同地点的多个 yaw 假设**，物理上真的分离
（1–2 m、Δyaw 175–193°，典型走廊 180° 对称）。聚类合并会把正确的和别名并在一起，是错的。

**假设 C（错）**：要求窗口内有足够运动才允许锁定。
运动数据推翻：唯一正确的那次锁定（11-51-17）窗口内位移 **0.068 m，是全部里最小的**。

## 24. 真正的结论

正确假设与别名在**所有被施加的检验下同样自洽**：

| | first↔latest 重现度 |
|---|---|
| 正确位置 | 0.10 m / 0.12° |
| 同地点 180° 别名 | 0.08 m / 0.33° |
| 37 m 外别名 | 0.09 m / 0.11° |

**"数数"只能得到"几个"，得不到"哪个"。** 而窗口内位移只有 0.07–1.08 m，
近乎静止时等 2 秒不提供新视角——消歧信息只能来自视角变化，不能来自时间流逝。

roadmap 里早就写过"重复等一等本身不能解决所有错误 basin"，今天的数据是这句话的直接证据。

## 25. 关键对比

| 方案 | 正确 FULL | 错误 FULL |
|---|---|---|
| 历史上被"证伪"的 scalar 阈值方案（另一套地图，不可直接引用） | 13 / 20 | 2 / 20 |
| 当前候选（唯一性计数），本次实测 | **1 / 20** | **8 / 20** |

新方案在两个轴上都大幅变差。为消灭 2 个错误 FULL 而引入的整套机制
（完整 yaw basin + 三次时间采样 + mode 求交 + 唯一性），把错误 FULL 从 2 涨到 8、
正确 FULL 从 13 掉到 1。

**立场更新**：用户最初"回到 main/realtime 当 baseline"的判断，比我最初认可的更有道理。
我当时的反驳理由是"旧 baseline 从未被测量过"——但现在新方案被测量了，结果是灾难性的。

## 26. 正在进行：同数据 A/B

不凭历史数字回退。用今天这套（已验证可信的）仪器，在**同一张地图、同一批 20 个 query、
同一个参考**上跑 `72fa6f7`：

- baseline worktree `~/ros_ws/n3mapping_baseline_72fa6f7`，独立 prefix `~/ros_ws/n3mapping_baseline_build`
- 用 72fa6f7 自己的工具从**同一张地图**编 Atlas（`map_sha256` 一致：`9e59c99b…`）
- manifest 剥掉 gravity 四列（72fa6f7 的 evaluator 只认 11 列），数据本身不变
- 72fa6f7 完全没有 anchor 概念（`atlas_anchor_rhpd_manager_` 出现 0 次），走 legacy 关键帧路径

判据：
- 旧代码若在同数据上明显更好 → 回退到它作为 baseline，专攻它的具体失败；
- 旧代码若同样差 → 历史 13/20 有水分，需要更根本的重新设计。

## 27. A/B 结果：候选被证伪

同一张地图（`map_sha256 = 9e59c99b…`）、同一批 20 个 query、同一个参考。

```
case        | BASELINE(72fa6f7)  trans_m    yaw | CANDIDATE   trans_m     yaw
11-45-43    | WRONG      1.132    2.99 | no-lock
11-46-29    | WRONG     33.097   43.67 | no-lock
11-47-22    | no-lock                  | no-lock
11-47-53    | CORRECT    0.187    0.25 | WRONG      4.926    4.50
11-48-26    | CORRECT    0.056    0.05 | WRONG      0.569    4.33
11-49-09    | WRONG     29.758  159.90 | no-lock
11-49-55    | no-lock                  | no-lock
11-50-34    | WRONG     27.920  103.64 | no-lock
11-51-17    | no-lock                  | CORRECT    0.141    1.13
11-52-00    | CORRECT    0.107    0.27 | WRONG      0.022    0.49
11-52-34    | CORRECT    0.034    0.07 | WRONG      0.107    3.05
11-53-11    | CORRECT    0.032    0.43 | no-lock
11-53-44    | CORRECT    0.107    0.32 | no-lock
11-54-17    | CORRECT    0.102    0.01 | no-lock
11-54-55    | WRONG      0.031    0.60 | WRONG      2.417   67.87
11-55-38    | CORRECT    0.049    0.33 | WRONG     36.163   88.23
11-56-10    | CORRECT    0.038    0.34 | no-lock
11-57-16    | CORRECT    0.022    0.32 | no-lock
11-58-18    | CORRECT    0.020    0.08 | WRONG      1.877  179.88
11-58-53    | CORRECT    0.041    0.45 | WRONG     32.668  123.18

baseline    CORRECT=12  WRONG=5   no-lock=3
candidate   CORRECT=1   WRONG=8   no-lock=11
```

baseline 正确时精度为 **0.020–0.187 m / yaw 0.01–0.45°**（厘米级）。
候选唯一正确的一次是 0.141 m / 1.13°。

历史 13/20 的说法基本属实（实测 12/20），但错误数不是 2 而是 5。

**结论：C4/C5/C6 的定位算法重构被证伪。** 为消灭 2 个错误 FULL 引入的整套机制，
使错误 FULL 变成 8、正确 FULL 从 12 降到 1。

## 28. 处置建议：分清要丢的和要留的

**要丢**（算法层，已被同数据 A/B 证伪）：
- C4 RHPD localization place/yaw basin 重构
- C5 gravity-constrained xyz+yaw matcher（在 Atlas 路径下的实际效果）
- C6 Atlas 唯一 mode 授权

**要留**（与算法选型正交，且是本次能做出对比的前提）：
- 可信测量装置：同源地图 + dense trajectory + exclusion windows + Atlas provenance
- `4af2ae6` exclusion windows CRLF 修复（否则留出集不成立）
- `53628b7` `migrate_one` / `verify_one` 补全
- `9cb9175` product runtime 测试 fixture 修复
- `bc20b36` install prefix 内容修正
- `dd25f86` 源码树内构建守卫
- `fb8c2a9` scan-end 时间戳采样（消除帧率隐含假设）
- `979a745` Atlas 窗口诊断埋点

## 29. 下一步：以 72fa6f7 legacy 路径为 baseline，攻它的 5 个错

| case | 误差 | 归类 |
|---|---|---|
| 11-46-29 | 33.1 m / 43.7° | 灾难性错位 + yaw |
| 11-49-09 | 29.8 m / 159.9° | 灾难性错位 + ~180° |
| 11-50-34 | 27.9 m / 103.6° | 灾难性错位 + yaw |
| 11-45-43 | 1.13 m / 3.0° | 边缘超标 |
| 11-54-55 | 0.031 m / 0.60° | **平移与 yaw 都很好，必然是 roll/pitch > 2°** → 姿态/重力对齐问题，独立的一条线 |

前三个共性：大位移错误恒伴随大 yaw 误差 → 位置别名与 yaw basin 是同一个问题的两面。
第五个是完全不同的失败类，不要和前面混在一起修。

复现命令见 §2 与 `~/ros_ws/n3mapping_v1_closeout/`：
- 候选评测：`run_today20.sh`
- baseline 评测：`run_baseline20.sh`
- A/B 汇总：`/tmp/ab.py`（已存 scratchpad）

---

# 续三：baseline 失败诊断（用户睡眠期间自主推进）

## 30. 目标（预先写死，事后不改）

在 20 个 today20 query 上，以 `72fa6f7` legacy 路径为基线：

| | 基线 | 必须 | 期望 |
|---|---|---|---|
| 错误 FULL | 5 | ≤ 1 | 0 |
| 正确 FULL | 12 | ≥ 10 | ≥ 12 |

规则：每次改动跑全部 20 case 验证；任何使错误 FULL 增加的改动一律回退；
先诊断到机制再动手。

## 31. 失败分解：5 个错其实是三类

| 类别 | case | 误差 |
|---|---|---|
| **灾难性错位（3）** | 11-46-29 / 11-49-09 / 11-50-34 | 28–33 m，yaw 43.7/159.9/103.6°，roll/pitch 5.5–17.5° |
| 边缘（1） | 11-45-43 | 1.13 m / 2.99° / roll 2.12° / pitch 2.05°（三项刚过线） |
| 姿态边缘（1） | 11-54-55 | **平移 3 cm**、yaw 0.60°，仅 roll 2.463° 超 2.0° 限 |

11-54-55 不是危险错锁——它锁在正确位置 3 cm 内。而正确案例的 roll 误差分布在
0.017–1.371°，说明参考与估计的 roll 一致性只有约 ±1.4°，2.0° 门槛相对噪声偏紧。

## 32. 两个"物理判别量"假设，都被数据推翻

**假设 D（错）**：`T_map_odom` 的倾斜可区分对错。
实测：正确 1.229–14.967°，错误 3.831–14.868°，**完全重叠**。
原因：该量被每次 query 自己的 LIO 世界系初始朝向主导（每个 query 是独立 FAST_LIO 运行），
与锁定对错无关。我用它当代理量时错误假设了两个坐标系都 z 朝上。

**假设 E（错）**：重力对齐误差可区分。
实测：正确 3.238–7.551°，错误 5.009–6.909°，**完全重叠**。
根因在这一行：

```
map gravity (map frame): [-0.0869 0.0053 -0.9962]   max deviation across 8250 frames: 9.73 deg
```

**地图自身的重力方向在整段录制里漂了 9.73°。** LIO 重力估计噪声（~5–10°）大于要找的信号。

**这同时解释了 C5 为什么没兑现**："gravity-constrained matcher" 建立在一个本身有 ~10° 噪声的
信号上，约束它买不到精度。

## 33. 决定性证据：算法记录的每个标量都不可分

停止发明判别量，改看 legacy 路径自己的埋点（这些字段就是为此设计的）：

```
margin                     correct[0.503..1.290]  wrong[0.367..1.122]   重叠
ratio                      correct[1.654..3.634]  wrong[1.443..3.070]   重叠
basin_separation           correct[2.103..40.94]  wrong[0.611..30.51]   重叠
temporal_hypothesis_score  correct[0.288..1.406]  wrong[-0.680..1.674]  重叠
log_likelihood             correct[8.537..9.523]  wrong[4.729..9.075]   重叠
best_rhpd                  correct[2.253..3.135]  wrong[1.779..5.641]   重叠
best_fused                 correct[0.107..0.153]  wrong[0.085..0.308]   重叠
```

最刺眼的是 **11-46-29（错 33 米）**：`temporal_hypothesis_score = 1.674` 是全场最高、
`best_rhpd = 2.369` 比任何正确案例都好、`margin = 1.122`、`ratio = 3.070`。

**按算法自己计算的每一个量，它比正确匹配还像。** 这不是阈值没调好——
**拒绝它所需的信息不在算法当前计算的任何量里**。调参救不了。

## 34. 结构问题：在没有独立证据时就做终局决定

`lkfr`（锁定帧号）那一列：**17 次锁定里有 7 次发生在 frame 4** ——
有 5 帧历史后的第一时间，约 0.4 秒，机器人几乎没动。正确与错误都如此，只是运气不同。

现有的 `moving_visibility_required` 不是"必须移动才能锁"，恰恰相反：
只有当传感器已移动时才额外要求可见性证据不为负；没移动时该检查直接跳过。
**没有任何最小证据基线。**

这也解释了候选那套"等 2 秒采 3 次"为何无效：**资源不是时间，是视角变化**。

## 35. 改动 1：独立视角基线（结果见下节）

在 `wip/evidence-baseline` 分支上，向接受谓词增加：

```cpp
const bool pass_evidence_baseline =
    top1 && (evidence_motion_translation >= config_.reloc_static_agg_max_translation ||
             evidence_motion_rotation   >= config_.reloc_static_agg_max_rotation);
```

**不引入新的调参数字**：复用代码自己定义的"视角变得独立"的边界
（`reloc_static_agg_max_translation = 0.25 m` / `max_rotation = 0.20 rad`）。
`evidence_motion_*` 自假设集创建时开始累积（第 430 行），语义正确。

## 36. 结构性根因：每次接受失败都销毁全部证据

`relocalize()` 末尾（`world_localizing.cpp:880`）的 `clearRelocHypotheses()` 位于
accept/reject 的 if/else **之后**，两条路都会执行。

控制流：窗口未满（< `reloc_temporal_window_size` = 5 帧）时提前 return，假设集保留；
一旦满 5 帧就评估接受谓词，**只要不通过就落到第 880 行清空**，下一帧从零重建。

**因此假设集最多活 5 帧（0.5 秒）。** 实测证据（每个 episode 内能积累到的最大视角基线）：

```
case                   max_ev_trans      (episode 实际总位移)
11-47-53 (正确)             0.0141 m      2.96 m
11-55-38 (正确)             0.0221 m      9.13 m
11-46-29 (错 33 m)          0.0274 m      3.81 m
11-58-18 (正确)             0.0314 m      3.43 m
仅 7/20 曾达到 0.25 m，全场最大 0.545 m
```

**差两个数量级。** 系统被迫在近静态快照（1–3 cm 基线）上做终局决定，
而在那个基线下，没有任何被记录的量能区分真实位置与感知别名。

这一条解释了今天的全部观察：
- 锁定集中在 frame 4（窗口刚满的第一时间）；
- 7 个标量全部不可分（只有 5 帧近静态证据）；
- 0.25 m 基线闸门把大部分锁定杀掉（5 帧寿命内不可达）；
- 候选那套"等 2 秒采 3 次"同样失败（它也在失败时丢弃）。

## 37. 改动 1 单独测量结果

```
baseline   CORRECT=12  WRONG=5   no-lock=3
evidence   CORRECT=5   WRONG=0   no-lock=15
candidate  CORRECT=1   WRONG=8   no-lock=11
```

**错误 FULL 5 → 0**，达成产品第一优先级。但正确 FULL 12 → 5，
**未达到预先写死的"≥ 10"**，按规则该项单独不算通过。

意义：在前面 7 个标量全部不可分的背景下，**视角基线是唯一真正切开对错的量**。
它损失召回的原因已经定位（假设寿命 5 帧），不是判据本身错。

## 38. 改动 2：接受失败时保留假设集（`wip/persist-evidence`）

```cpp
// 接受分支内清理并 return；拒绝分支不再清理
```

理由：让存活假设跨帧继续累积支持，从而跨越视角。被否定的假设已由既有的
`hyp.alive` 剪枝处理；若全部死亡，下一帧走既有的空集重建路径。**不引入新参数。**

预测（可证伪）：正确 FULL 应显著回升，错误 FULL 应保持 0。

## 39. 三个工作点的完整测量

同一张地图、同一批 20 个 query、同一个 same-run 参考：

```
case        | BASELINE     err_m | EVIDENCE     err_m | PERSIST      err_m
11-45-43    | WRONG        1.132 | no-lock          - | no-lock          -
11-46-29    | WRONG       33.097 | no-lock          - | WRONG       33.052
11-47-22    | no-lock          - | no-lock          - | no-lock          -
11-47-53    | CORRECT      0.187 | no-lock          - | WRONG        1.273
11-48-26    | CORRECT      0.056 | no-lock          - | CORRECT      0.091
11-49-09    | WRONG       29.758 | no-lock          - | no-lock          -
11-49-55    | no-lock          - | no-lock          - | no-lock          -
11-50-34    | WRONG       27.920 | no-lock          - | no-lock          -
11-51-17    | no-lock          - | no-lock          - | no-lock          -
11-52-00    | CORRECT      0.107 | no-lock          - | no-lock          -
11-52-34    | CORRECT      0.034 | CORRECT      0.165 | CORRECT      0.062
11-53-11    | CORRECT      0.032 | CORRECT      0.171 | CORRECT      0.089
11-53-44    | CORRECT      0.107 | CORRECT      0.107 | CORRECT      0.107
11-54-17    | CORRECT      0.102 | no-lock          - | no-lock          -
11-54-55    | WRONG        0.031 | no-lock          - | CORRECT      0.023
11-55-38    | CORRECT      0.049 | CORRECT      0.158 | CORRECT      0.061
11-56-10    | CORRECT      0.038 | no-lock          - | no-lock          -
11-57-16    | CORRECT      0.022 | CORRECT      0.230 | no-lock          -
11-58-18    | CORRECT      0.020 | no-lock          - | CORRECT      0.029
11-58-53    | CORRECT      0.041 | no-lock          - | CORRECT      0.104

baseline   CORRECT=12  WRONG=5   no-lock=3
evidence   CORRECT=5   WRONG=0   no-lock=15
persist    CORRECT=8   WRONG=2   no-lock=10
```

分支：
- `wip/evidence-baseline` = `581398e`（仅视角基线闸门）
- `wip/persist-evidence` = `98d432f`（在其上加证据持久化）

持久化把锁定帧从 frame 4 推到 frame 16–39，**采集时延 0.4–3.9 s，全部在 5 s 合同内**。

## 40. 目标达成情况：未达成

预先写死的判据是「错误 FULL ≤ 1 **且** 正确 FULL ≥ 10」。

| 方案 | 错误 | 正确 | 判定 |
|---|---|---|---|
| evidence | 0 ✅ | 5 ❌ | **未达标** |
| persist | 2 ❌ | 8 ❌ | **未达标** |

不粉饰：两个都没过。但相对 baseline，两者都把错误 FULL 降低了（5 → 0 / 5 → 2），
而产品合同把「错误 FULL = 0」排在成功率之上——按该优先级，
**`evidence` 是目前唯一做到零错误 FULL 的配置，这是此前任何版本都没达到过的。**

## 41. 硬核心：11-46-29

它在三种配置下都错（baseline 33.097 m / persist 33.052 m），且：

- 检索把正确 anchor 排在 rank ≤ 4（0.04 m）；
- 它的 `temporal_hypothesis_score = 1.674` 全场最高、`best_rhpd = 2.369` 优于任何正确案例；
- 持久化让它多积累了 14 帧证据，仍然锁错同一个地方。

**这个感知别名无法被当前特征集中的任何信息拒绝。** 要解决它需要新的信息源，
而不是新的阈值或新的时序策略。这是一条独立的、需要设计层面决策的线。

## 42. 一个反例：持久化让 11-47-53 变差

baseline CORRECT 0.187 m → persist WRONG 1.273 m。
证据积累并不单调有益：更长的窗口也可能让一个次优假设逐渐胜出。
这说明持久化不能无条件采用，需要配合「假设何时应被判定为死亡」的规则一起设计。

## 43. 建议的下一步（按信息量排序）

1. **确定运行点**：若接受「宁可不锁也不锁错」，`wip/evidence-baseline`（0 错误 / 5 正确）
   已可作为新基线，然后专攻召回。这是唯一零错误配置。
2. **恢复召回**：evidence 变体的 15 个 no-lock 里，多数是因为 5 帧寿命内够不到 0.25 m。
   持久化方向对（5 → 8），但需要解决 11-47-53 那类退化。
   建议：持久化 + 明确的假设死亡判据（例如连续 N 帧被证据反对即剔除），而非无限期保留。
3. **11-46-29 类别**：需要新信息源。可考虑的方向（均需先测可分性，再谈实现）：
   - 局部自由空间/负空间一致性（RHPD 已有 negative-space token，但未用作最终几何否决）；
   - 垂直结构一致性（`rhpd_enable_vertical_tokens` 同样未用于最终否决）；
   - 更长基线的运动一致性（需要数据集包含足够位移的 query）。
4. **数据集问题**：today20 的 query 是 ~20 s、位移 0.74–12 m 的近静态切片，
   且全部取自建图那次遍历。要验证「移动中重定位」的能力，需要专门录制的、
   有明确视角变化的 query。当前数据集无法支撑运动类判据的验证。

---

# 续四：定位到"从 TopK 中选出真值"这一步

## 44. 决定性证据：真值一直在候选里，只是没被选出来

三个灾难性失败案例，锁定时刻的全部活假设与真值的距离：

```
11-46-29  truth=(10.48,-3.12,-2.42)   winner=kf 256
   kf 17    dist= 0.129 m   cum_loglik= 9.008   vis_consis=0.135   ← 正确，且似然最高
   kf 17    dist= 0.458 m   cum_loglik=-21.467  vis_consis=0.095
   kf 17    dist= 0.988 m   cum_loglik= 7.862   vis_consis=0.119
   kf 256   dist=33.097 m   cum_loglik= 8.901   vis_consis=0.842   ← 赢家

11-49-09  truth=(-0.08,15.22,-4.16)   winner=kf 136
   kf 81    dist= 0.097 m   ...                 ← 正确
   kf 136   dist=29.758 m   cum_loglik= 8.076   ← 赢家

11-50-34  truth=(-5.31,-5.65,-2.91)   winner=kf 176
   kf 102   dist= 0.744 m   cum_loglik= 6.702   ← 正确，似然高于赢家
   kf 176   dist=27.920 m   cum_loglik= 4.729   ← 赢家
```

**RHPD 没有漏掉真值，ICP 也把真值对到了 0.1–0.7 m。失败发生在"选"这一步。**

## 45. 一个被挑样本误导的判断（已纠正）

看这三个失败案例时我说"可见性一致性系统性反相关"。**这是挑样本得出的错误结论**——
在失败案例里选择准则当然是错的，那是同义反复。

放到全部 12 个已锁定案例上重测：

```
按可见性一致性选 → 8 对 4 错
按累计对数似然选 → 7 对 5 错
```

可见性反而略优。两者**互补而非优劣**：

| 准则 | 错在哪些 case |
|---|---|
| 可见性 | 11-45-43, 11-46-29, 11-49-09, 11-50-34 |
| 对数似然 | 11-45-43, 11-49-09, 11-50-34, 11-52-34, 11-57-16 |
| 两者都错 | 11-45-43, 11-49-09, 11-50-34 |

换成似然会修好 11-46-29 但弄坏另两个，**净变差**；要求两者一致是 6 对 3 错，更差。

## 46. 对"是 RHPD 不行还是 ICP 不行"的回答

| 组件 | 判定 | 证据 |
|---|---|---|
| RHPD | **没问题** | 三个失败里真值全是活假设（0.097 / 0.129 / 0.744 m）；anchor 召回 20/20 rank≤4 |
| ICP | **没问题** | 种子正确时 0.020–0.187 m、yaw 0.01–0.45° |
| 检索表示 | **不是瓶颈** | 真值已在候选里，换更好的检索救不了这些案例 |
| **候选选择/验证** | **就是它** | 可见性、似然、margin、ratio、descriptor 距离全部无法正确排序 |

**因此原计划的"把 anchor 检索移植进 legacy 路径"被取消**——数据显示它不会改变任何一个失败案例。
避免了一次数小时的无效移植。

根本原因：**ICP 残差不是"是不是同一个地方"的度量。** 任何局部光滑表面都能让 ICP 自信收敛；
走廊里 33 米外的另一段走廊在几何上就是能对齐。现有五个量测的都是"对齐得好不好"。

## 47. 内点判别性测试：现有统计量无信息（假设 F，已推翻）

给每个假设记录 ICP 内点率/内点数/fitness（提交见 `wip/diag-inliers`），结果：

```
mean_inlier_ratio    真值[0.9632..1.0000]  别名[0.9499..1.0000]  重叠（饱和在 1.0）
mean_inlier_count    真值[4585..7319]      别名[5363..7283]      重叠
mean_fitness         真值[0.0035..0.0385]  别名[0.0057..0.2535]  重叠
```

**`inlier_ratio` 恒为 1.0**：对应距离阈值是 1.0 m，而在稠密室内地图里任何点都能在 1 m 内
找到某个表面。这个统计量在该环境下不可能有判别力。

**这不否定最大团方向**，只说明我用现有稠密 ICP 内点数当它的代理量选错了——
TEASER/Quatro 用的是稀疏特征点之间的**成对距离不变性一致性**，是完全不同、严格得多的检验。

但负结果本身有价值：**在这个环境里稠密点云重叠统计量是没有信息的，每个候选都能"解释"这帧扫描。**
这从反面支持"验证必须建立在更稀疏、更有辨识度的表示上"。

## 48. 可参考的算法方向（均为非学习，符合边缘端约束）

1. **最大团几何验证**：`MIT-SPARK/TEASER-plusplus`、`url-kaist/quatro`。
   不看对齐残差，而看有多少组对应关系彼此几何自洽（求最大团）。
   真实同一地点 → 团大；感知别名 → ICP 仍收敛但自洽对应很少。输出内点计数/一致性证书。
2. **MCL / AMCL 式全局定位**：面对多模态歧义**不挑**，维持多模态信念让运动坍缩它。
   与今天"消歧资源是视角变化"的结论一致。现有多假设机制是它的雏形，但寿命只有 5 帧。
3. **自由空间/占据冲突作为否决**：`evaluatePoseVisibility` 已在算这类量，
   但当作**排序分**使用（已证失效）。改作**硬否决**未测过，是不同的用法。

---

# 续五：TEASER++ 验证与第一性原理结论

## 49. 用户的批评（正确）

"这样感觉是在通过调参解决问题，并不是从第一性原理出发吧"

成立。前面一直在找"哪个标量能分开"，换分数、换阈值都是同一层面打转。
而 TEASER/Quatro 的核心不是分数，是几何定理：

> 刚体变换保距。若对应 (aᵢ→bᵢ) 和 (aⱼ→bⱼ) 都正确，则 ‖aᵢ−aⱼ‖ = ‖bᵢ−bⱼ‖。

"是不是同一个地方" ⇒ "能否找到一大组彼此保距自洽的对应" ⇒ 最大团。
自带证据量（团大小），不需要标定阈值。这确实是第一性原理。

## 50. 实测：TEASER++ 独立地给出同样的错误答案

`MIT-SPARK/TEASER-plusplus` @ `52a9c52` 编译安装于 `~/ros_ws/teaser_install`，
离线探针 `~/ros_ws/teaser_probe`（源码 `teaser_probe.cc`）。

方法：query 取锁定帧前 5 帧变换到 odom 系聚合；map 取该候选位置 25 m 球裁剪；
两侧 0.30 m 体素；FPFH 半径 = 2×/4× 体素（跟随体素，不引入独立旋钮）；
noise_bound = 1 体素；`use_max_clique = true`。

```
11-46-29 TRUE   clique=10  rot_inliers=5   t=[ -7.27 -17.04 -1.91]
11-46-29 ALIAS  clique=23  rot_inliers=18  t=[-13.10  19.80 -5.12]  ← 别名团更大
11-49-09 TRUE   clique=10  rot_inliers=5
11-49-09 ALIAS  clique= 9  rot_inliers=6
11-50-34 TRUE   clique=12  rot_inliers=7
11-50-34 ALIAS  clique=10  rot_inliers=7
11-58-18 TRUE   clique=19  rot_inliers=17  （正确案例对照）
11-55-38 TRUE   clique=12  rot_inliers=5   （正确案例对照）
```

**最难的 11-46-29 上，别名的最大团是真值的 2.3 倍，且 TEASER 估计的平移
`[-13.10, 19.80, -5.12]` 精确命中别名位置 `(-13.11, 19.91, -5.42)`。**

TEASER 不是"没找到真值"，是**认为别名更好**。

诚实边界：探针用 0.30 m 体素 + FPFH，对 FPFH 偏粗，因此这不是对 TEASER 工程价值的终审。
但**不再继续调体素/半径**——在 20 个样本上调参没有意义，也正是被批评的做法。

## 51. 环境本身：6 米垂直跨度的重复结构

```
map points: 532683   z range: -9.99 .. 7.86 m
主体分布 z ∈ [-6, 0]，峰值约 -2.5
```

三个别名与真值的 Δz = 3.0 / 1.3 / 2.5 m，水平距离 28–33 m。
配合 `f7tof9`（7 楼到 9 楼）的命名，这是多层/重复结构环境，Δz 正是层高量级。

## 52. 五种独立方法，同一种失败

| 方法 | 结果 |
|---|---|
| ICP fitness / 残差 | 重叠 |
| 可见性一致性 | 重叠（12 个案例中 8 对） |
| 累计对数似然 | 重叠（7 对） |
| 稠密内点率 / 内点数 | 饱和在 1.0，无信息 |
| TEASER 最大团 | **反向：别名团更大** |

不是五个算法都不行，**是它们在解同一个无解的问题**。

## 53. 第一性原理结论

> **若两个假设从当前视角预测出相同的观测，则对该观测做任何处理都无法区分它们。
> 信息不在数据里。**

这是信息层面的事实，不是算法缺陷。从近静止视角看，30 米外另一层的走廊与眼前这段
产生相同观测。要区分只能**改变视角**——那是唯一能引入新信息的途径。

这解释了为什么今天唯一奏效的是**独立视角基线闸门**：它不试图从不可分的数据里榨取
区分度，而是承认"现在还不知道"，拒绝下结论。错误 FULL 5 → 0。

## 54. 结论：不建议接 TEASER/Quatro

它们解决的是"对应关系含大量外点"的问题；本项目的瓶颈是**歧义本身**。
实测已证接进来不会修好这三个案例。

**建议顺序：**

1. **采用 `581398e`**（视角基线闸门，0 错误 FULL / 5 正确），符合"宁可不锁也不锁错"；
2. **恢复召回**：让证据跨视角累积（`98d432f` 已验证方向可行：5 → 8 正确），
   需配明确的"假设何时算死"判据，避免 11-47-53 那类退化；
3. **录制带明确视角变化的 query**：现有 20 段窗口内基线只有 1–3 cm，
   **该数据集在物理上无法验证任何依赖运动的方法**。这是当前最大瓶颈，
   且是数据瓶颈而非算法瓶颈。

## 55. 本轮新增资产

```
~/ros_ws/TEASER-plusplus/          上游源码 @ 52a9c52
~/ros_ws/teaser_install/           安装前缀
~/ros_ws/teaser_probe/             离线探针（源码 + build）
~/ros_ws/n3mapping_v1_closeout/run_teaser_probe.sh
artifacts/.../candidate_dd25f86/teaser_probe/   聚合后的 query 点云
```

分支 `wip/diag-inliers`：每假设记录 ICP 内点率/内点数/fitness（measurement only）。

## 56. 用户纠正：全部同楼层，z 偏差是重力对齐问题（§51 作废）

用户指出：20 个样本全在同一楼层，没有跨楼层，z 有偏差只可能是重力对齐不好。

实测证实，**§51 的"多楼层"判断错误（本日第 10 个被推翻的假设）**：

```
水平跨度 92 m
整体平面拟合  z = 0.01375·x − 0.07102·y − 1.921
=> 地图整体倾斜 4.14°，在 92 m 上产生 6.64 m 的 z 变化
去除倾斜后的 z 残差：单峰，−2.5 ~ +2.0 m  ← 单层楼高度，非两层
```

地图平面法向 `[0.0137, −0.0708, −0.9974]`，偏离 z 轴 **4.14°**；
与独立测得的地图重力方向 `[−0.0869, 0.0053, −0.9962]`（4.99°）一致。

**别名与真值的 Δz（3.0 / 2.5 / 1.3 m）完全是倾斜造成的假象**：
33 m × tan(4.14°) = 2.39 m、28 m → 2.03 m。二者物理上同高。

## 57. 用几何平面法向重做重力判别测试：仍不可分，而且原因是结构性的

改用平面法向（53 万点拟合，远比逐帧 LIO 估计精确）作为地图重力：

```
11-46-29  错 33.1 m   重力误差  2.831°   ← 别名重力完全正常
11-50-34  错 27.9 m   重力误差  1.365°   ← 同样正常
11-45-43  错  1.1 m   重力误差 12.005°
11-49-09  错 29.8 m   重力误差 13.395°
11-52-00  对  0.107 m 重力误差 12.180°   ← 正确的反而最差
正确位置 [0.65 .. 12.18]   错误位置 [1.37 .. 13.40]   重叠
```

**结构性原因**：重力只约束 roll 与 pitch（2 DoF），而歧义在 x、y、yaw。
同一楼层上 30 m 外的走廊与眼前这段**重力方向本来就相同**。
一个不约束歧义所在自由度的信号，无论多精确都无法消除该歧义。

**推论：即使把地图重力对齐修到完美，这三个失败案例一个都不会改变。**
这不是精度问题，是可观测性问题。

## 58. 但地图倾斜是真实缺陷，应单独修

后果（与 aliasing 正交）：

1. 地图系 z 轴不是重力方向，地面在地图中不是平面 —— 对导航的高度/可通行性判断有实际影响；
2. **C5 的 "gravity-constrained matcher" 一直在对着偏 4° 的基准做约束**，
   这解释了它为何没有兑现"roll/pitch 全程受约束"的承诺；
3. 任何以地图 z 轴为竖直方向的下游假设都被系统性地偏置了 4.14°。

修法方向（未实施）：建图结束时用 LIO 重力（或地面平面拟合）把整图旋转到重力对齐，
并把该旋转写入地图元数据。属于建图侧问题，不在定位算法内。

## 59. 完整结论链

| 层次 | 结论 | 证据 |
|---|---|---|
| RHPD | 没问题 | 真值 20/20 在 rank≤4；三个失败里全是活假设（0.097/0.129/0.744 m） |
| ICP | 没问题 | 种子正确时 0.020–0.187 m、yaw 0.01–0.45° |
| 从 TopK 选真值 | **无解** | 5 种独立方法全部不可分；TEASER 最大团甚至反向 |
| 重力判别 | **结构上不适用** | 只约束 roll/pitch，歧义在 x/y/yaw |
| 地图重力对齐 | **真实缺陷，但与 aliasing 正交** | 倾斜 4.14°，该修，但修了这三个案例不变 |
| **唯一有效** | **证据不足时拒绝下结论** | 视角基线闸门：错误 FULL 5 → 0 |

第一性原理落点：**两个假设从当前视角预测相同观测时，信息不在数据里，只能改变视角。**
今天用五种方法从不同角度撞了同一堵墙。

## 60. 本日被数据推翻的假设清单（避免重犯）

1. 等价性合并只在 region 内做 —— 读实现后推翻（本就是跨 region complete-link）
2. 同地点多 anchor 被误算成多 mode，聚类可解 —— 打出 T_map_odom 后推翻（是多 yaw 假设，物理分离）
3. 要求窗口内有运动 —— 唯一正确的锁定窗口位移最小（0.068 m）
4. T_map_odom 倾斜可判别 —— 完全重叠（被各 query 自身 LIO 世界系朝向主导）
5. 重力对齐误差可判别（用逐帧均值）—— 完全重叠
6. 可见性一致性系统性反相关 —— 挑样本得出的错误结论（全集上 8/12 反而最优）
7. ICP 内点数可判别 —— inlier_ratio 饱和在 1.0（对应阈值 1 m，室内处处满足）
8. fitness 否决线 —— 扫描证明无任何阈值能改善（最优即不加否决）
9. TEASER 最大团可判别 —— 别名团反而更大（2.3×），且精确命中别名位姿
10. 别名是跨楼层 —— 用户纠正 + 平面拟合证实是单层，Δz 是 4.14° 倾斜假象

**每一个都是先验证再动手才没写进代码。**

---

# 续六：Cartographer 式占据/自由空间打分

## 61. 用户提问

"纯 2D 的重定位，Cartographer 通过回环处理做得很好，目前其实还是 2D，为什么做不到呢？"

## 62. Cartographer 与 n3mapping 的三处结构性差别

| | Cartographer | n3mapping |
|---|---|---|
| 搜索 | 分支定界穷举 (x, y, θ) 窗口 | top-K 描述子检索 + ICP |
| 打分 | **占据栅格相关性** | ICP 残差 / 可见性比 |
| 自由空间 | **显式编码**（射线穿过的格子） | 基本丢弃 |
| 决策 | 带鲁棒核的图约束，**可被后续证据推翻** | 二值锁定，不可逆 |

第四行可能是最根本的：Cartographer **从不做不可逆的决定**，错误约束会在优化中被压掉。
而 `SEARCHING → FULL_6DOF_LOCKED` 是单向门，正因不可逆才必须在证据不足时做对——
而今天已证明那做不到。

## 63. 实测：真射线投射的自由空间

`raycast.py`：地图按 4.14° 去倾斜；0.10 m 栅格；z 带 [floor+0.3, floor+2.0]；
1650 个轨迹位姿 × 720 方向 = 118.8 万条射线，DDA 推进至首个占据格。

```
grid 708 x 958 at 0.10 m    occupied cells 37426
ray-cast free cells: 46573   （仅足迹版只有 ~5k，差一个数量级）
运行 2.7 s
```

```
case          verdict       occupied%    free%   unknown%
11-45-43      WRONG-PLACE      68.89%   20.72%    10.39%
11-46-29      WRONG-PLACE      58.76%   21.93%    19.31%
11-49-09      WRONG-PLACE      57.39%   11.82%    30.79%
11-50-34      WRONG-PLACE      59.30%    6.46%    34.24%
11-47-53      CORRECT          95.70%    2.34%     1.96%
11-48-26      CORRECT          96.96%    2.24%     0.80%
11-55-38      CORRECT          91.00%    1.52%     7.47%
11-54-55      CORRECT          93.28%    5.78%     0.94%
11-53-11      CORRECT          81.44%    4.12%    14.44%
11-56-10      CORRECT          82.61%    3.90%    13.49%
11-57-16      CORRECT          81.90%    4.08%    14.01%
11-54-17      CORRECT          84.00%    9.50%     6.50%
11-53-44      CORRECT          80.15%   14.51%     5.34%
11-52-00      CORRECT          79.79%    4.69%    15.52%
11-52-34      CORRECT          66.00%    7.53%    26.47%
11-58-53      CORRECT          60.30%   15.27%    24.43%
11-58-18      CORRECT          43.42%   13.97%    42.61%
```

严格判据：仍然重叠。**但这是本日第一个方向一致的信号：**

- 4 个错误位姿的 `occupied%` **全部 ≤ 68.9%**，无一超过 70%；
- 13 个正确位姿中 10 个 ≥ 79.8%；
- 错误位姿的自由空间冲突均值约 15%，正确约 7%——**两倍差距**。

此前所有方法（ICP 残差、可见性、似然、内点、TEASER 最大团）**连方向性都没有**。

## 64. 重叠来自"未观测"而非位姿错误

三个掉队的正确案例全部伴随巨大 unknown 比例（42.6% / 26.5% / 24.4%）——
query 看向地图从未观测的区域（地图边缘），与位姿正确性无关。

按"有证据的格子"归一化 `occupied/(occupied+free)`：

```
正确 [75.7% .. 98.4%]     错误 [72.8% .. 90.2%]
```

明显改善，但 11-50-34（错 27.9 m）仍得 90.2%，压过数个正确案例。

## 65. 实现的已知粗糙之处（信号存在，仪器粗）

1. **把 3D 扫描压成 2D**——Cartographer 的 2D 面向平面机器人，压平 3D LiDAR 丢失大量信息；
2. 全局中位数地板高度，未处理局部地面起伏；
3. 栅格 0.10 m、占据阈值 2 点、单次二值射线更新，非概率栅格；
4. 无多分辨率、无分支定界。

结论：**自由空间是正确方向，但需要按 Cartographer 的做法正经实现**
（概率栅格 + 正确的射线更新 + 多分辨率），这是独立的工程项目。

## 66. 完整架构建议

三件已验证或方向明确的事，合起来才成立：

1. **可逆决策**：暂定位姿（可用于导航）→ FULL（权威）→ 可降级。
   让错误从"产品失效"变成"暂定位姿被修正"，代价从灾难变成延迟。
   这是 Cartographer 真正的护城河，且不需要新数据或新传感器。
2. **视角基线闸门**（`581398e`，已验证：错误 FULL 5 → 0）作为 FULL 的准入条件。
3. **占据/自由空间打分**（今日证明方向正确、强度不足）作为候选验证，需正经实现。

**跨楼层反而更简单**：z 与气压是额外可观测维度；同层别名是没有任何额外维度可用的
最难情形，应先解决。

## 67. 本轮资产

```
scratchpad/raycast.py        射线投射占据打分（2.7 s 跑完 17 个案例）
scratchpad/freespace.py      足迹版自由空间（已被 raycast 取代，保留作对照）
scratchpad/teaser_probe.cc   TEASER++ 离线探针
artifacts/.../teaser_probe/  17 个案例的聚合 query 点云
```

---

## 68. 工作卡 A0 实测：描述子选择 —— **被证伪**

### 68.1 动机

用户提出的疑问（"我在 rhpd 中增加 free space 的信息并不会使得从 topk 中选取最佳更方便吧"）
在 §44–59 的排查中得到了一个反直觉的答案：**描述子自己的排序已经是对的**
（20/20 真值在 rank≤4；三个灾难性失败里真值都是活假设，距离 0.097 / 0.129 / 0.744 m）。
错的是它之后的**几何重排序** —— 系统按可见性证据（visibility evidence）
对假设重新排序，而在 1–3 cm 的视角基线下这个量没有区分度。

于是 A0 的假设是：**让描述子选地方，配准只负责精化，不允许它改选。**

### 68.2 改动（`wip/descriptor-select` = `b25f9bb`）

`RelocHypothesis` 增加 `seed_rhpd_distance`，在假设创建时记下检索它的候选的
RHPD 距离；排序主键从 `mean_visibility_evidence` 换成 `seed_rhpd_distance`（升序），
可见性与累积对数似然降为无描述子距离时的兜底次键。**不引入任何新阈值。**

### 68.3 预注册判据（改动前写死）

> 灾难性错位（>5 m）≤1 **且** 正确 ≥8。

### 68.4 结果

```
baseline   CORRECT=12  WRONG=5   no-lock=3    灾难性(>5m)=3   最坏=33.097 m
descsel    CORRECT=9   WRONG=11  no-lock=0    灾难性(>5m)=5   最坏=40.934 m   FAIL
```

不但没过，还比 baseline 差：灾难性 3 → 5，最坏 33.1 → 40.9 m，且**每一例都锁**
（no-lock 3 → 0）。具体倒退：11-48-26（0.056 CORRECT → 40.934 WRONG）、
11-52-00（0.107 → 9.193）、11-54-17（0.102 → 3.575）。
改善：11-49-09（29.758 → 0.930）、11-50-34（27.920 → 0.801）。

### 68.5 根因：门与排序不自洽

接受门用的 `margin` / `ratio` 是 **top1 与 top2 的可见性证据之差**。
排序主键换成描述子距离后，top1 的可见性变成任意值，它与"下一个物理上不同的假设"
之间的可见性差通常很大 —— **门失去了拦截能力**。
锁定帧证实了这一点：几乎全部在 frame 4，且 3 个原本不锁的案例现在都锁了。

要让门自洽，`margin` 就得改成描述子距离差。但 `reloc_lock_min_margin = 0.35`
是按可见性对数几率标定的，换过去需要一个新阈值 —— **那正是本轮约定不做的调参**。
因此不走这条路，改为测一个组合。

### 68.6 组合测试（`wip/descsel-evbase` = `d390949`）

假设：视角基线闸门（§下文 `581398e`）与描述子排序正交 ——
闸门管"够不够资格下终局判断"，排序管"哪个假设排第一"，
前者对后者的错误免疫。判据不变。

```
case                   | BASELINE   err_m | EVIDENCE   err_m | DESCSEL    err_m | DESC+EV    err_m
--------------------------------------------------------------------------------------------------
2026-07-23-11-45-43    | WRONG      1.132 | no-lock        - | WRONG      1.874 | no-lock        -
2026-07-23-11-46-29    | WRONG     33.097 | no-lock        - | WRONG     31.223 | no-lock        -
2026-07-23-11-47-22    | no-lock        - | no-lock        - | WRONG      7.051 | no-lock        -
2026-07-23-11-47-53    | CORRECT    0.187 | no-lock        - | CORRECT    0.187 | no-lock        -
2026-07-23-11-48-26    | CORRECT    0.056 | no-lock        - | WRONG     40.934 | no-lock        -
2026-07-23-11-49-09    | WRONG     29.758 | no-lock        - | WRONG      0.930 | no-lock        -
2026-07-23-11-49-55    | no-lock        - | no-lock        - | WRONG     32.697 | CORRECT    0.212
2026-07-23-11-50-34    | WRONG     27.920 | no-lock        - | WRONG      0.801 | no-lock        -
2026-07-23-11-51-17    | no-lock        - | no-lock        - | WRONG      0.165 | no-lock        -
2026-07-23-11-52-00    | CORRECT    0.107 | no-lock        - | WRONG      9.193 | no-lock        -
2026-07-23-11-52-34    | CORRECT    0.034 | CORRECT    0.165 | CORRECT    0.034 | CORRECT    0.165
2026-07-23-11-53-11    | CORRECT    0.032 | CORRECT    0.171 | CORRECT    0.032 | CORRECT    0.171
2026-07-23-11-53-44    | CORRECT    0.107 | CORRECT    0.107 | CORRECT    0.071 | no-lock        -
2026-07-23-11-54-17    | CORRECT    0.102 | no-lock        - | WRONG      3.575 | no-lock        -
2026-07-23-11-54-55    | WRONG      0.031 | no-lock        - | WRONG      0.031 | no-lock        -
2026-07-23-11-55-38    | CORRECT    0.049 | CORRECT    0.158 | CORRECT    0.049 | CORRECT    0.158
2026-07-23-11-56-10    | CORRECT    0.038 | no-lock        - | CORRECT    0.156 | no-lock        -
2026-07-23-11-57-16    | CORRECT    0.022 | CORRECT    0.230 | CORRECT    0.018 | CORRECT    0.230
2026-07-23-11-58-18    | CORRECT    0.020 | no-lock        - | CORRECT    0.020 | no-lock        -
2026-07-23-11-58-53    | CORRECT    0.041 | no-lock        - | CORRECT    0.104 | no-lock        -

baseline   CORRECT=12  WRONG=5   no-lock=3    灾难性(>5m)=3   最坏=33.097 m   FAIL
evidence   CORRECT=5   WRONG=0   no-lock=15   灾难性(>5m)=0   最坏= 0.230 m   FAIL
descsel    CORRECT=9   WRONG=11  no-lock=0    灾难性(>5m)=5   最坏=40.934 m   FAIL
desc+ev    CORRECT=5   WRONG=0   no-lock=15   灾难性(>5m)=0   最坏= 0.230 m   FAIL
```

**判据未过（正确 5 < 8）。A0 证伪。**

### 68.7 这次负结果的信息量（比"更差"更重要）

组合版与闸门单独版**分数完全相同**，正确集只差一个互换：

| | evidence | desc+ev |
|---|---|---|
| 11-49-55 | no-lock | **CORRECT 0.212** |
| 11-53-44 | CORRECT 0.107 | **no-lock** |
| 11-52-34 | CORRECT 0.165 | CORRECT 0.165 |
| 11-53-11 | CORRECT 0.171 | CORRECT 0.171 |
| 11-55-38 | CORRECT 0.158 | CORRECT 0.158 |
| 11-57-16 | CORRECT 0.230 | CORRECT 0.230 |

其余四例连误差值都逐位相同。结论有两条，都是可复用的：

1. **闸门确实对排序错误免疫。** descsel 单独跑出的 40.9 / 32.7 / 31.2 / 9.2 / 7.1 / 3.6
   六个大错，组合版一个都没放过去。两个改动不冲突，闸门是安全性的唯一承担者。
2. **闸门一旦到位，top-K 内部怎么排的净效果是零。** 这否定的不只是"按 RHPD 距离排"，
   而是整个"换一个排序准则就能解决"的路线 —— 因为闸门放行的那几例
   本来就只有一个存活假设，排序无事可做；被闸门拦下的那些，排序对错都不影响输出。

**推论**：召回率的瓶颈不在选择准则，在**闸门的准入条件本身**（即视角基线从哪来）。
这把后续工作从"找更好的打分函数"（工作卡 A/B 的一部分）重新指向
"让机器人在锁定前真正积累视角基线"（工作卡 C/D）。

### 68.8 顺带测量：baseline 那 5 个 WRONG 的构成

| case | 平移 | roll | pitch | yaw | 性质 |
|---|---|---|---|---|---|
| 11-46-29 | 33.097 m | — | — | — | 真·锁错地方（感知别名） |
| 11-49-09 | 29.758 m | — | — | — | 真·锁错地方（感知别名） |
| 11-50-34 | 27.920 m | — | — | — | 真·锁错地方（感知别名） |
| 11-45-43 | 1.132 m | 2.12° | 2.05° | 2.99° | 位置对，四轴全踩线 |
| **11-54-55** | **0.031 m** | **2.46°** | 0.12° | 0.60° | **位置对到 3 cm，只因 roll 超 0.46° 判负** |

即 baseline 的"5 个错误 FULL"里只有 **3 个是真正锁错地方**，
另外 2 个是精度失败。其中 11-54-55 **只在 roll 上失败** ——
这直指工作卡 E（地图整体倾斜 4.14°），修好它 baseline 直接少一个 WRONG，
且完全不触碰任何选择逻辑，零计算代价。

### 68.9 处置

- 运行点保持 `581398e`（视角基线闸门），即用户选定的"宁可不锁也不锁错"。
- `wip/descriptor-select`（`b25f9bb`）与 `wip/descsel-evbase`（`d390949`）
  保留为**已测量的负结果**，不合入。
- 工作卡 A0 关闭。这是本轮第 11、12 个被推翻的假设（前 10 个见 §60）。

### 68.10 已知的脚本缺陷（不影响结论）

`run_dsev20.sh` 把 stderr 写到了 `$BL/eval/$Q.time.log` 而非 `eval_dsev/`，
覆盖了 baseline 的 `.time.log`（仅计时日志，未触碰 `result.json` /
`frame_status.csv` 等评测输出）。baseline 的位姿结论不受影响；
若后续要重跑 RSS/耗时统计，需重跑 `run_baseline20.sh`。

---

## 69. 重力/倾斜的重新测量 —— **推翻 §41 的地图倾斜结论,工作卡 E 关闭**

用户指出:雷达倾斜安装,IMU 的 z 轴不对应重力,且与 LIO 启动时机器人姿态(趴/站)相关;
因此「把地图转正」是后处理化妆,救不了运行时新 session。据此重新测量,结论比该质疑更强。

### 69.1 刚性去倾斜在评测度量上恒等无效(数值验证)

评测比的是 locked pose 与 reference pose,**两者同在 map 系**。给地图乘刚性 R,
相对旋转 `P_lock⁻¹P_ref` 严格不变。实测:

```
2026-07-23-11-54-55
   相对旋转角(严格不变量) = 2.354°
   原始 Euler 差:            roll=2.463  pitch=0.125  yaw=0.598
   去倾斜 4.14°(随机轴 ×4):  roll=2.432 / 2.509 / 2.444 / 2.424
```

**§68.8 中「修 E 就少一个 WRONG」的推断是错的,收益为 0。**

### 69.2 地图**没有**倾斜 4.14°——那是全局单平面拟合的伪影

沿轨迹分 24 窗、每窗半径 6 m 拟合局部地板平面(**完全不依赖 IMU**):

```
平均法向偏离 map z 轴  0.41°
各窗法向离散         中位 0.35°   p95 1.46°   最大 1.65°
```

**地图基本是平的。** §41 的 4.14° 来自对整张图做单一平面拟合,而该图跨 265 m 行程、
**地板 z 实际跨度 7.41 m**(真实的多标高结构,局部却处处水平),
其中 4 个窗口连地板模态都找不到。单一全局平面被这些结构带偏。

推论:`raycast.py` 里那个 `R_level`(由 a=0.01375, b=-0.07102 构造)是**伪造的旋转**,
所有用过它的分析都受污染——包括 §61–67。

### 69.3 LIO 重力估计的段内漂移(真实,但不解释锁定误差)

query manifest 的 `gx,gy,gz` 表达在 **LIO 自己的 odom 系**(每段 f0 恒为 `[0,0,-1]`,
即每段独立按重力初始化)。相对 f0 的偏离:

```
段内最大漂移:  0.21° .. 12.97°  (19.5 s),中位约 4°,17/20 例末帧即最大 → 单调增长,是漂移非步态噪声
时间剖面:      f2≈0.00°  f4<0.30°(除一例1.10°)  f16≈0.2–1.7°  f64≈0.2–5.4°  f194 见上
```

**锁定几乎都发生在 f4,彼时漂移 <0.3°,因此漂移不解释锁定时的姿态误差**
(11-54-55 重力偏差 0.019° 却有 2.354° 相对旋转误差)。

但它**直接约束工作卡 C**:C 靠里程计传播假设来累积视角基线,
而传播用的正是这个会漂的姿态。12 m 行程约 15–20 s,届时姿态已偏数度,
30 m 量程上折合约 2 m 的预测点位移。真值与别名被同一漂移同等地转,
所以不引入偏置,但它给「行程越长越好」设了硬上限。

### 69.4 处置

**工作卡 E 关闭**:它要修的缺陷不存在(地图是平的),而它的手段(刚性去倾斜)
在度量上恒等无效。真正存在的前端问题是**姿态漂移 4°/20 s**,
那属于 LIO 层面的独立课题,不在重定位改进的关键路径上。

---

## 70. 自由空间重测 —— **推翻 §61–67 的「不可分」结论**

### 70.1 修正仪器

§61–67 的实现有两处致命缺陷:
1. 套用了 §69.2 证伪的伪去倾斜矩阵 `R_level`;
2. 用**全局中位数**地板高度,而地板 z 实际跨度 7.41 m —— 对大部分区域都是错的,
   z 带整个选歪。

改为:去掉伪去倾斜 + **局部地板模型**(粗栅格上取 z 的 2% 分位,空洞最近邻填充),
z 带 = 局部地板 +0.30 .. +2.00 m。

### 70.2 三个灾难性别名在第一帧即被否决(零行程)

```
11-49-09  真值 occ 91.19% / free  1.56%    别名 occ 30.90% / free 49.80%
11-50-34  真值 occ 87.64% / free  8.82%    别名 occ 48.89% / free 49.17%
11-46-29  真值 occ 95.15% / free  0.05%    别名 occ 78.04% / free 16.18%
```

别名把近一半回波打进了「地图曾看着射线穿过」的格子。

### 70.3 里程计传播的附带结果(C 卡 kill test)

把真值与别名假设都按里程计传播整段行程,逐 0.25 m 打分,累积票差 (真−别):

```
11-49-09   0.25 m 内   1.085 → 2.202     别名被证伪
11-50-34   0.51 m 内   0.791 → 2.346     别名被证伪
11-46-29   3.69 m 内   0.332 → 11.183    别名被证伪(单调增长)
```

**行程确实单调增加区分度**(11-46-29 从第一帧的弱区分增长到压倒性),
但另两例根本不需要行程。**C 卡的前提成立,只是优先级低于 70.4。**

### 70.4 作为 TopK 选择器 —— **通过预注册判据(今天第一个)**

对每个 case 的**全部存活假设**(取自 `dbg_*/relocalization_debug.jsonl`,
含 `pose_in_map` 的完整位姿)在锁定帧打分,取 `occ% − free%` 的 argmax:

```
                        正确   错误   灾难性(>5m)   最坏
现状(可见性排序)          8      4        3        33.097 m
占据/自由空间 argmax      8      4        1        10.009 m
```

**判据(灾难性 ≤1 且 正确 ≥8)达成。** 逐例:

| case | 现状 | 占据分选中 | 备注 |
|---|---|---|---|
| 11-46-29 | 33.097 | **0.129** | 灾难性被修复 |
| 11-49-09 | 29.758 | 1.097 | 29.8 m 别名得分 **−0.189**(真值 +0.501),被决定性否决;选中的是同一地点 1.1 m 外的假设 |
| 11-50-34 | 27.920 | 0.744 | 三个别名得分 −0.003 / −0.323 / −0.464;**真值不在假设集里**,0.744 m 即可得最优 → 选择器表现最优 |
| 11-47-53 | 0.187 | **10.009** | 新增灾难性,机理见 70.6 |
| 11-45-43 | 1.132 | 4.134 | 真值不在假设集里 |

### 70.5 绝对阈值否决器 —— **被证伪**

```
真值假设(err≤0.5) 的 free% 范围:  0.00% .. 60.92%
灾难性假设(err>5) 的 free% 范围:  0.11% .. 71.04%
```

完全重叠。实测多档否决阈值(20/25/30/35/40/45%)叠加在可见性排序上,
灾难性最好只降到 2,且最坏误差仍是 33.097 m。

**结论:自由空间只能用作同一 query 内假设之间的相对比较,不能用作绝对水平判据。**
这也解释了 §61–67 为何看到「重叠」——它比的正是绝对水平。

### 70.6 剩余失败的机理,以及一个被否定的诊断

11-47-53 的**真值假设自身** free% 高达 29.86%(occ 61.30%),
而 10.009 m 外那个得 occ 87.63% / free 4.61%。即地图在真值位置的自由空间不可信。

我最初诊断为「地板模型太粗」。**实测否定**:地板模型分辨率 5.0 / 2.0 / 1.0 m
三档结果**逐例完全相同**。

因此剩余失败来自**2D 投影 + 射线投射模型本身高估自由空间**
(720 角 × 30 m,从每 5 个轨迹样本投射,忽略 3D 遮挡与实际 FOV/量程),
即 §65 的第 1、3、4 条。修法是概率栅格 + 正确的射线更新(Cartographer 的做法),
这是独立的工程项目,现在有了量化理由:仅这个粗糙实现就把灾难性从 3 降到 1。

### 70.7 本轮资产

```
/tmp/t1.py       假设传播 + 逐行程占据打分(C 卡 kill test)
/tmp/t2b.py      重力坐标系修正版分析
/tmp/t2c.py      局部地板平面拟合(不依赖 IMU 的倾斜测量)
/tmp/t2d.py      漂移时间剖面 + 锁定时刻姿态误差
/tmp/t3.py       TopK 选择器测试,可传地板模型分辨率参数
/tmp/hypdump*.json  全部假设的逐条打分,供离线比较决策规则
```

---

## 71. 3D 体素自由空间 —— **零错锁,当前最佳**

用户质疑:栅格是 2D 的,但这是 3D 世界下的定位任务。质疑成立,
且 §70.6 那个无法用地板模型分辨率解释的失败(11-47-53)正是投影伪影。

### 71.1 成本澄清(决定了这个取舍)

栅格**建图时生成、随地图存储**,运行时只是「变换 query 点 + 查表」,
3D 与 2D 都是 O(1) 每点。**「不能增加太多计算」这条约束不逼我们用 2D**,
代价在离线与存储。而 3D 还**去掉了地板模型与 z 带**这两个误差源——
它们本来就只是为了压成 2D 才需要的。

### 71.2 实现

- 体素占据:地图点落入的体素,计数 ≥ OCCMIN;
- 自由空间:从**最近轨迹位姿**到每个地图点连射线,沿途标 free,**遇占据即停**
  (Cartographer 式射线更新,而非 §61 那种合成角度扫掠);
- 打分:query 点变换后逐点查表,`occ% − free%`,**无 z 带、无地板模型**。

### 71.3 分辨率是关键,而 0.10 m 会崩

```
体素 0.10 m OCCMIN=2   occ 91k    free 1.51M    正确=6 错误=6 灾难性=4 最坏=33.097 m
体素 0.15 m OCCMIN=1   occ 246k   free 241k     正确=9 错误=3 灾难性=0 最坏= 1.132 m
体素 0.20 m OCCMIN=1   occ 155k   free  95k     正确=9 错误=3 灾难性=0 最坏= 1.132 m
体素 0.20 m OCCMIN=2   occ 115k   free 116k     正确=9 错误=3 灾难性=0 最坏= 1.132 m
体素 0.20 m OCCMIN=3   occ  88k   free 141k     正确=8 错误=4 灾难性=1 最坏=10.009 m
体素 0.25 m OCCMIN=1   occ 102k   free  43k     正确=9 错误=3 灾难性=0 最坏= 1.132 m
体素 0.30 m OCCMIN=1   occ  72k   free  22k     正确=9 错误=3 灾难性=1 最坏=12.028 m
```

0.10 m 崩溃的机理已量化:命中率 `occ` 掉到 **1.98–4.34%**(2D 时 20–95%)。
53 万点的降采样地图在 10 cm 体素下**填不满曲面**,query 点几乎不可能正好落进占据体素。
射线也只有 53 万条(2D 版是 1650 位姿 × 720 角 = 119 万条),角度覆盖同样过稀。
**这是欠采样,不是 3D 的概念问题。**

**0.15 / 0.20 / 0.25 m 逐例完全相同,OCCMIN 1 与 2 也完全相同 —— 是平台不是刀尖**,
操作区间横跨 1.7× 分辨率与 2× 占据阈值。

### 71.4 四方对照

```
                              正确   错误   灾难性(>5m)   最坏
现状(可见性排序)                8      4        3        33.097 m
2D 栅格 argmax(§70)            8      4        1        10.009 m
3D 体素 0.10 m                 6      6        4        33.097 m
3D 体素 0.15–0.25 m            9      3        0         1.132 m   ← 最佳
```

### 71.5 剩下 3 个「错误」全部不是错锁

| case | 选中 | 真值在假设集? | 性质 |
|---|---|---|---|
| 11-45-43 | 1.132 m | **否** | 可得最优,选择器表现最优 |
| 11-50-34 | 0.744 m | **否** | 可得最优,选择器表现最优 |
| 11-49-09 | 1.097 m | 是(0.097 m) | 选中的是**同一地点** 1.1 m 外的假设 |

**没有任何一例锁到别的地方。** 最坏误差 1.132 m,而现状是 33.097 m。
用户要求「不能误锁」在这批数据上达成。

注意:这些是**假设位姿**,不是流水线最终精化后的锁定位姿;
真实流水线里选中正确种子后 ICP 会继续精化,§44–59 已测得种子正确时 ICP 达 2–19 cm。
因此 0.744/1.097/1.132 在整合后大概率进入 0.5 m 合同以内,但这需要实测确认。

### 71.6 算力与存储(边缘端可行性)

```
0.20 m 体素   栅格维度 355 × 479 × 110 = 18.7 M
              非空体素 155k(占据) + 95k(自由) ≈ 250k
              稀疏哈希 ≈ 2–3 MB / 稠密双 bitset = 4.7 MB
建图时构建     53 万条射线,实测 0.9 s
运行时打分     每假设约 9500 个在界点 × O(1) 查表;8 个假设 ≈ 76k 次查表
```

RK3588 / Orin Nano 上完全无压力。**这条路满足用户第一条要求(不增加太多计算)。**

### 71.7 对用户四条要求的回答状态

1. **不能增加太多计算** —— 满足,见 71.6;
2. **不能误锁** —— 本批数据上达成(零错锁,最坏 1.132 m 且都是同地点近失);
3. **free 空间能否融入 RHPD** —— **不需要**。§44–59 已测得 RHPD 检索本身没问题
   (20/20 真值在 rank≤4),把 free 空间塞进描述子解决不了「从 TopK 选真值」;
   它作为**假设之间的独立选择阶段**才有效,这也是本节的做法;
4. **跨楼层无气压计** —— 3D 体素天然带 z(本图 z 跨度已达 21.85 m,本就是多标高),
   这是使跨楼层可能的结构;2D 栅格根本做不到。

### 71.8 尚未验证的部分(不要当成已完成)

- 只在 **12 个可评案例**上离线重选择,不是完整流水线跑通;
- 5 个 baseline 里正确的案例因缺 `lock_accepted` 记录未纳入,子集偏向失败例;
- 0.15–0.25 m 这个平台是在**本数据集**上观察到的,需独立数据验证;
- 射线用「最近轨迹位姿」近似真实观测位姿,真实实现应记录每点的观测帧;
- 未做概率栅格(仍是二值),Cartographer 的 miss/hit 概率更新尚未实现。

### 71.9 资产

```
/tmp/t4.py               3D 体素自由空间选择器,参数: <体素尺寸> <OCCMIN>
/tmp/hypdump3d_*.json    各配置下全部假设的逐条打分
```

---

## 72. 自由空间选择器接入 C++ 流水线：kill 证伪，veto 通过（2026-07-26）

分支 `wip/freespace-select`（远端 `~/ros_ws/n3mapping_baseline_72fa6f7`，基于 `72fa6f7`，
两个提交 `62b4473` / `49d2424`，**未 push**）。独立 prefix `~/ros_ws/n3mapping_fs_build`，
`~/ros_ws/n3mapping_baseline_build` 与 `eval/` 的 baseline 产物一字节未动。

新增 `include/n3mapping/free_space_grid.h` + `src/free_space_grid.cpp`（`FreeSpaceGrid`）。
运行时从 `reloc_map_cache_`（Atlas `globalMap`）+ 全部关键帧 `pose_optimized` 位置构建：
每个地图点是 occupied；从该点**最近的观测位姿**投射射线，遇 occupied 即停，途经格标 free。
参数走环境变量，默认 `RES=0.20 OCCMIN=2 MAX_RAY=30.0`，与 §71 离线一致。

### 72.1 两种接法

| 模式 | 做法 | 环境变量 |
|---|---|---|
| **kill** | 自由空间分数**淘汰**被支配的假设（存活判据），排序键与闸门不动 | `N3MAPPING_FREESPACE_MODE=kill` |
| **veto** | 自由空间**只否决不选择**：若自由空间最优与可见性 top1 不是同一物理位姿，则本窗口拒绝锁定 | 默认 |

kill 的设计初衷是「不重蹈 A0 覆辙」——不换排序主键。**这个理由是不充分的，见 72.3。**

### 72.2 实测（20 case 全流水线，非离线重选）

| 配置 | 正确 | 错误 | 不锁 | 灾难性 >5 m | 最坏 |
|---|---|---|---|---|---|
| baseline `72fa6f7` | 12 | 5 | 3 | 3 | 33.097 m |
| + 3D 自由空间 **kill** | 12 | **8** | 0 | 2 | **32.924 m** |
| + 3D 自由空间 **veto** | **12** | **2** | 6 | **0** | **1.132 m** |

预注册判据「灾难性=0 且 正确≥9」：kill **FAIL**，veto **PASS**。

veto 逐例（只列变化）：

| case | baseline | veto |
|---|---|---|
| 11-46-29 | WRONG 33.097 | **no-lock** |
| 11-49-09 | WRONG 29.758 | **no-lock** |
| 11-50-34 | WRONG 27.920 | **no-lock** |
| 11-47-53 | CORRECT 0.187 | CORRECT 0.445 |
| 11-58-53 | CORRECT 0.041 | CORRECT 0.186 |

**没有一个 baseline 正确的 case 被弄坏**；三个灾难性错锁全部变成不锁；
其余 15 例逐位相同。剩下 2 个 WRONG 都不是错地方：
11-45-43 平移 1.132 m（roll 2.12°/pitch 2.05°），11-54-55 平移 **3 cm**、仅 roll 2.46° 超标。
**零错地方锁定。**

对比此前唯一零错误 FULL 的配置（`581398e` 视角基线闸门，5 正确 / 0 错误 / 15 不锁）：
veto 在同样零错地方的前提下把正确数从 5 抬到 12。

### 72.3 kill 为什么失败——两个独立缺陷

**缺陷一：kill 这个动作本身解除了歧义闸门。**
杀到只剩 1 个存活假设后 `top2 == nullptr` → `margin = inf` → `ambiguous = 0`。
11-58-53 的日志逐窗打印 `margin=inf` 五次，并在 `visibility=-1.56`
（可见性证据为**负**，即地图否定该位姿）的情况下锁定。
baseline 的 3 个 no-lock 全部变成 lock，其中 11-47-22 落在 10.435 m。
这与 A0 是同一类错误的另一形态：**移除竞争者等价于让闸门失去它要判的那个量。**
「不动排序键」并不足以保证闸门仍然有效。

**缺陷二：自由空间分数本身就把真值排在别名之后。**
11-58-53 window 1 的逐假设 dump（`[Reloc/FreeSpaceDump]`）：

```
i=0 seed=256 conv=1 value=0.685 occ=78.8% free=10.3% sd=0.0056 xyz=(-13.28, 19.72, -5.71)  ← 真值
i=4 seed=18  conv=1 value=0.829 occ=84.9% free= 2.1% sd=0.0042 xyz=( 10.22, -3.08, -2.55)  ← 别名，被选中
```

别名**在占据率上也赢**（84.9% vs 78.8%），不只是自由空间。
该位置的几何对这一帧观测的解释能力就是比真值处更好——
这与 §44–59 的结论一致：**占据+自由空间是第六种无法分离真值与别名的方法。**

**5σ 二项噪声判据是伪判据。** 真值与别名的分差 0.144，而 5σ = 0.035，
真值被以 4 倍余量杀掉。该判据度量的是采样噪声，而主导误差是模型误差，
两者差一个数量级。它给出的是虚假的严格感，不应再用作淘汰门槛。

### 72.4 被推翻的假设（第 13 个）

我一度把 11-58-53 的 Δz = 3.16 m（真值 −5.71 / 别名 −2.55）解释为**跨楼层**，
并据此推断「最近位姿射线穿楼板」。**用户纠正，作废。**
§56 早已确立 20 个样本全在同一楼层（当日第 10 个被推翻的假设），
§69/§70 又测得地板 z 实际跨度 **7.41 m**（同层多标高，局部处处水平）。
3.16 m 完全落在同层范围内。**这是同一个已被证伪的假设被第二次提出**，记此以免第三次。

### 72.5 离线 §71（9/3/0）为什么没传过来

两个变量，均未排除：

1. **地图密度**：离线用 `global_map.pcd`（532683 点，0.1 m 体素）；
   运行时用 Atlas `globalMap`，同样 0.20 m 体素下只有 **114727** 个 occupied 格、83218 个 free 格。
2. **打分时机**：离线在**锁定帧、ICP 收敛后**（conv=5）打分；
   运行时在 **window 1、conv=1** 的未收敛种子位姿上就决策。

分离方法：用稠密地图重建栅格，在**同样的 window-1 位姿**上重打分（单变量）。
若真值仍输，则 §71 的 9/3/0 依赖「收敛后位姿」这一条件，结论须按此重写。**尚未做。**

### 72.6 成本（20 case 实测）

- **内存**：峰值 RSS baseline 788.8 MB → veto 789.1 MB。栅格两层 `vector<bool>` 约 5 MB
  （0.20 m 下 356×480×111 = 19.0 M 格），在 788 MB 的基数上不可见。
- **构建**：0.256 s，一次性，建图/加载后触发。
- **打分**：每假设约 8300 次 O(1) 查表；未在 p95 上产生可见增量。
- **p95 时延**：14 例逐位不变。变差的都是「veto 阻止了早锁、系统继续搜索」的例子：
  11-46-29 28.6→1549.6 ms，11-49-09 723→2147 ms，11-58-53 28.7→1338 ms。
  **这不是栅格的开销，是搜索路径本身的开销被暴露得更久。**
  baseline 本就有 5 例 >1 s，产品合同（backend p95 <1 s）此前即未满足。
- **获取时延**：18 例不变；11-47-53 0.40→0.90 s；11-58-53 0.40→**5.90 s**，
  这是 veto 新引入的一处 <5 s 合同违反（11-52-00 的 9.40 s 是 baseline 就有的）。

### 72.7 产品合同现状（不粉饰）

| 项 | 要求 | veto 实测 |
|---|---|---|
| 正确 FULL | ≥15/20 | **12** ✗ |
| 错误 FULL | 0 | **2**（均为 roll 超标，非错地方）✗ |
| 平移 | ≤0.50 m | 12 个正确例全部满足；最坏正确例 0.445 m |
| roll/pitch | ≤2° | 11-54-55 roll 2.46°、11-45-43 roll 2.12°/pitch 2.05° ✗ |
| 获取 | <5 s | 11-52-00 9.40 s（baseline 既有）、11-58-53 5.90 s（新增）✗ |
| backend p95 | <1 s | 多例超标（baseline 既有，veto 加剧）✗ |

**仍不可发布。** veto 的价值是把「错地方锁定」这一项做到了零，且没有牺牲正确数。

### 72.8 资产

```
远端分支 wip/freespace-select   62b4473 / 49d2424（未 push）
~/ros_ws/n3mapping_fs_build/    独立 prefix
~/ros_ws/n3mapping_v1_closeout/run_fs20.sh      kill/veto 通用 20 case 跑批（输出目录可传参）
~/ros_ws/n3mapping_v1_closeout/run_fsveto20.sh  veto 版
/tmp/abfs2.py                   baseline vs freespace 逐例对照，打印超标项
$BL/eval_freespace/             kill 模式 20 case 输出
$BL/eval_fsveto/                veto 模式 20 case 输出
```

环境变量：`N3MAPPING_FREESPACE_SELECT=0` 关闭；`N3MAPPING_FREESPACE_MODE=kill` 切 kill；
`N3MAPPING_FREESPACE_RES` / `_OCCMIN` / `_MAX_RAY` / `_SIGMAS`（后者仅 kill 用）。

---

## 73. §72.5 两个变量全部排除，真因是离线子集的选择偏差（2026-07-26）

§72.5 列了两个未排除的变量来解释「离线 9/3/0 没传到流水线」。两个都测了，**都不成立**。

### 73.1 变量一「地图密度」——证伪

给栅格加了 `N3MAPPING_FREESPACE_MAP_PCD` 覆盖源（`world_localizing.cpp`，仅测量用）。
指向离线用的 `global_map.pcd`（532683 点）重跑 11-58-53：

```
grid source overridden by .../global_map.pcd points=532683
grid built res=0.2 occupied=114727 free=83218 in 0.269658 s
```

**与运行时地图缓存构建的栅格逐位相同**（114727 / 83218），逐假设分数也逐位相同。
即 `reloc_map_cache_`（Atlas `globalMap`）**就是** `global_map.pcd`，从来没有密度差。

顺带修正 §71 / README 的一个数字：0.20 m + OCCMIN=2 下非空体素是 **114727**，
不是此前写的「≈250k」。稀疏存储量级不变（约 2–5 MB），结论不受影响。

### 73.2 变量二「打分时机」——证伪

用 `N3MAPPING_FREESPACE_SIGMAS=1e9` 关掉淘汰，让 dump 打满 5 个窗口，
看分数随 ICP 收敛（conv=1→5）的演化。11-58-53 真值 i=0 vs 别名 i=4：

| window | conv | 真值 value | 别名 value | 差 |
|---|---|---|---|---|
| 1 | 1 | 0.685 | **0.829** | −0.144 |
| 2 | 2 | 0.673 | **0.851** | −0.178 |
| 3 | 3 | 0.703 | **0.805** | −0.102 |
| 4 | 4 | 0.693 | **0.794** | −0.101 |
| 5 | 5 | 0.679 | **0.782** | −0.103 |

**别名在每一个窗口都赢，差距不收敛。** 收敛后位姿并不改变结论。

### 73.3 真因：离线 12 案例子集把破坏性案例排除在外

`/tmp/hypdump3d_*.json` 的键：

```
11-45-43  11-46-29  11-47-53  11-48-26  11-49-09  11-50-34
11-52-34  11-53-11  11-53-44  11-55-38  11-57-16  11-58-18
```

**11-58-53 不在里面。** §71.8 自己写了这条警告——
「5 个 baseline 里正确的案例因缺 `lock_accepted` 记录未纳入，子集偏向失败例」——
现在确认它就是操作性的那一条：**唯一能推翻 kill 模式的案例，恰好是被排除的那 5 个之一。**

**§71 的 9/3/0 因此不是一个可外推的结果**，它是在一个剔除了反例的子集上取得的。
worklog §71 与 README 中据此得出的「3D 体素零错锁」表述，作用域仅限那 12 个案例。

### 73.4 由此得到的可复用结论

1. **占据 + 自由空间不能分离真值与感知别名。** 这是第六种失败的方法
   （前五种见 §44–59）。它能修掉 §72 那三个灾难性别名，但不是一个通用判别器。
   它唯一相对安全的用法是 **veto（只否决不选择）**。
   注意该「安全」只到单窗口为止，不是 episode 级保证——见 §76。
2. **离线重选择实验必须覆盖 baseline 已正确的案例**，否则度量的是「在失败例上有多好」，
   而不是「会不会把对的弄坏」。今后任何离线选择器实验，
   **判据里必须显式包含「baseline 正确的案例一个都不能坏」**。
3. §71.8 那份「尚未验证」清单不是形式主义。**这次的翻车就在清单第 2 条上。**

---

## 74. 锁定时刻姿态误差：五个假设全部证伪（2026-07-26）

对象是 veto 配置下仅剩的 2 个 WRONG，两个都不是错地方：
11-54-55 平移 **3 cm**、roll 2.46°；11-45-43 平移 1.132 m、roll 2.12°/pitch 2.05°。

### 74.1 误差形状：暂态，锁定帧最差，自行收敛

11-54-55 逐帧对参考的 roll 误差（`/tmp/att.py`）：

```
frame  4 (LOCK)  -2.463°   t=0.031 m
frame  5         -2.001°
frame  8         -1.870°   ← 已在 2° 限内
frame 17         -1.351°
frame 40         +0.197°
```

平移全程 3–20 cm。**锁定那一帧是整段里姿态最差的一帧。**
而 11-52-34 / 11-53-11 / 11-58-18 同样在 frame 4 锁定，roll 误差只有
−0.04° / +0.82° / +0.09°——不是「锁定瞬间系统性地差」。

### 74.2 五个被证伪的假设

| # | 假设 | 证伪依据 |
|---|---|---|
| 1 | 地图整体倾斜 | §69：局部地板法向 0.41°；刚性去倾斜在度量上恒等无效 |
| 2 | 静态聚合窗口把姿态抹平 | 窗口内 odom 姿态最大展开：坏例 **0.355°**，好例 0.288°/0.228°。0.35° 解释不了 2.46° |
| 3 | 配准可观测性不足（信息矩阵） | 见 74.3，**不可分** |
| 4 | 求解器停止判据过粗 | 见 74.4，**改了等于没改** |
| 5 | 地图关键帧与轨迹姿态不一致 | 见 74.5，**一致到 0.004°** |

### 74.3 假设 3：信息矩阵不可分

`MatchResult::information`（`point_cloud_matcher.cpp:256` 由 small_gicp 的 H 填充，
平移/旋转块互换）。在锁定点记录胜出假设旋转块的最小特征值，20 case 全跑：

| case | 结果 | roll | `rot_info_min` |
|---|---|---|---|
| 11-45-43 | WRONG | 2.12° | 3.29e5 ← 最低 |
| 11-55-38 | CORRECT | — | 1.04e6 |
| 11-47-53 | CORRECT | — | 1.72e6 |
| 11-56-10 | CORRECT | — | 1.89e6 |
| **11-54-55** | **WRONG** | **2.46°** | **2.94e6** |
| 11-58-18 | CORRECT | — | 9.72e6 |

**roll 误差最大的那例，信息量高于三个正确例。** 且 CRB 预测 σ 全在 0.018–0.100°，
实测误差达 2.46°，**差 25–100 倍**——主导误差是偏差不是方差。
这条同时堵死了「用信息矩阵做可观测性闸门」，不必再为它凑阈值。

### 74.4 假设 4：停止判据 —— 提出、实施、证伪

`point_cloud_matcher.cpp:125-126`：

```cpp
setting_.translation_eps = std::max(1e-3, config_.gicp_transformation_epsilon);           // 1 mm
setting_.rotation_eps    = std::max(0.1, config_.gicp_rotation_epsilon_deg) * M_PI/180.0; // 1.0°
```

余量严重不对称：平移 1 mm vs 0.50 m 合同 = **500×**；旋转 1.0° vs 2° 合同 = **2×**。
据此把 `gicp_rotation_epsilon_deg` 从 1.0 改到 0.1（代码自身的 clamp 下限，
20× 余量，**从规格推出，未看任何 case 的误差**）并全跑 20 case。

**结果：逐位完全相同。** 11-54-55 仍 0.031 m / roll 2.46°，11-45-43 仍 1.132 m。

埋点给出原因：

```
11-54-55  LockTerm] iters=2 term=converged inlier_ratio=1 fitness=0.0870
11-58-18  LockTerm] iters=1 term=converged inlier_ratio=1 fitness=0.0381
```

**2 次迭代即收敛，内点率 1.0。** 求解器根本没有被停止判据截断——
它认为自己已在最优点。**改动已回退**（理由被证伪的改动不留在树里）。

### 74.5 假设 5：地图与轨迹姿态一致到 0.004°

把锁定所用支撑关键帧的 `pose_optimized` 与稠密参考在**同一时间戳**比较
（不变量旋转角，非 Euler）：

| case | kf | Δt | Δ位置 | Δ旋转 |
|---|---|---|---|---|
| 11-54-55 | 205 | 0.00 ms | 0.0000 m | **0.0005°** |
| 11-58-18 | 250 | 0.00 ms | 0.0000 m | 0.0000° |
| 11-52-34 | 136 | 0.00 ms | 0.0000 m | 0.0036° |

地图关键帧就是稠密轨迹本身。**姿态误差不是从地图继承来的，是重定位配准自己产生的。**

### 74.6 顺带测得的两件事

**(a) 真实姿态误差是 1.99°，Euler 分解略有放大。**
11-54-55 锁定位姿 vs 参考的**不变量**旋转角 **1.989°**，误差旋转轴在 map 系
`[-0.975, -0.051, -0.215]`（基本就是 map x 轴）。同一对位姿的 Euler 分解给出
roll 2.078° / pitch 0.077° / yaw 0.455°。放大幅度不大（1.99→2.08），
**不足以把这例解释成度量假象**，但记下：合同用 Euler 分解判 roll/pitch，
在大 yaw 处会把不变量误差分配得偏大。

**(b) 雷达倾装被数据确认。** 参考位姿本体 z 轴离 map z **18.25°**，
与用户所述「雷达是倾斜的、IMU z 不对应重力」一致。

### 74.7 现状与下一步

已知机制：GICP 在 2 次迭代、内点率 1.0 下收敛，即**接受了种子给的姿态几乎未修正**；
该姿态在随后 ~35 帧跟踪中缓慢收敛到 0。误差 1.99°，合同限 2°，**卡在边界上**。

尚未验证的方向（不要当结论）：
- **决策层**：位姿在锁定后仍在改善，把 FULL 的上报推迟到「位姿不再移动」
  （相邻跟踪帧位姿增量低于某界）。代价约 0.4 s（5 s 预算内）。
  **风险：这个「某界」很容易变成按本数据集调的参数**，须先想清楚它从哪来。
- 为什么 GICP 在该处对 map-x 方向的旋转不敏感，而信息矩阵又不反映这一点
  ——两者矛盾，尚未解释。

### 74.8 资产

```
/tmp/att.py        逐帧姿态误差对照（参数：eval 目录 + case 列表）
/tmp/abrot.py      roteps 配置 20 case 对照
$BL/eval_roteps/   停止判据 0.1° 的 20 case 输出（与 veto 逐位相同）
```

---

## 75. 跨 session 独立数据：floor7（2026-03）打 0723 地图（2026-07-26）

**这是本项目第一批真正独立的评测数据。** 此前所有结论都建立在 today20 上，
而 today20 是建图那次运行自己的时间切片（§73 的翻车就是这个结构性风险的直接后果）。

用户指出 floor7 的 bag 与 today20 是同一层楼，可以直接打 0723 大地图。核对属实：

- 查询来源：`/home/user/ros_ws/bagfile/n3mapping_test/floor7_*.bag`，5 个**独立 bag**，
  `source_start_stamp_ns ≈ 1773373240` → **2026 年 3 月**
- 地图：`baseline_72fa6f7/n3map.pbstream`（f7_0723），**2026-07-23**
- **跨 4 个月、跨 session、独立录制。** 无 GT。

注：worklog §369 记「floor7 的 5 个 case 是同一段 13 分钟连续录制切出来的，没有独立 bag」——
**该记述有误**，metadata 显示每组对应各自独立的 bag 文件。

### 75.1 结果（veto 配置）

| 组 | 帧数 | 结果 | 获取时延 | p95 | 锁定位姿 |
|---|---|---|---|---|---|
| 710toMeeting_static_70_75s | 49 | **LOCK** | 0.90 s | 865 ms | (−8.30, −12.36, −3.15) |
| inside_710_room | 452 | **LOCK** | **12.40 s** | 1022 ms | (−10.33, 20.29, −5.56) |
| outside_707_room | 448 | **LOCK** | 1.40 s | 56 ms | (4.20, 26.17, −4.60) |
| outside_710_room | 448 | no-lock | — | 1278 ms | — |
| outside_713_room | 448 | no-lock | — | 1397 ms | — |
| **710toMeeting 全长** | **1848** | **LOCK** | **10.40 s** | **205 ms** | 见 75.3 |

**4 锁 / 2 不锁 / 零错锁。** 零错锁不是目测，依据是 75.2 的复现实验。

### 75.2 两个不依赖 GT 的正确性验证

**验证 2（决定性）**：同一 bag，两条**完全独立**的估计路径，在同一时间戳比对。
`710toMeeting_static_70_75s` 是从 70 s 处**冷启动**的 5 秒片段（无任何先验）；
全长段在该时刻已**连续跟踪 60 秒**。

```
冷启动片段  @stamp=1773373311947904348   (−8.30, −12.36, −3.15)
全长段同刻  frame=710  FULL_6DOF_LOCKED   (−8.29, −12.35, −3.16)
时间差 0.000 ms   位置差 0.013 m   姿态差 roll 0.10° / pitch 0.17° / yaw 0.20°
```

**1.3 cm、0.2°。错锁不可能从两个不同初始条件复现出这个数。**
这比人眼看叠合图更硬，且完全不需要参考轨迹。

**验证 1**：两个独立 bag。`inside_710_room` 的锁定位姿与全长段首个锁定位姿
相距 **0.726 m**（同一房间内的不同位置），一致。

### 75.3 全长段：锁上之后一帧没丢

```
t= 0.00s  frame=0     REGION_HYPOTHESIS
t=10.40s  frame=104   FULL_6DOF_LOCKED    ← 全程唯一一次状态变化
锁定段 1744 帧 / 174.3 s / 轨迹总长 100.7 m
相邻帧最大跳变 0.524 m
z 范围 −5.77 .. −2.57 m
```

### 75.4 姿态变化（用户第 4 条要求的直接验证）

锁定段本体 z 轴离 map z 的倾角：**min 9.48° / max 27.01° / 中位 13.50°**。
两段持续高倾角，**全程 FULL**：

```
t= 10.40 ..  14.50 s ( 4.1 s)  倾角 22.0..25.0°
t=115.80 .. 143.60 s (27.8 s)  倾角 21.7..27.0°   ← 推断为趴下段
```

站立段稳定 **9.7°**，高倾角段 21.7–27.0°，**摆动 17.5°，锁未丢失**。

更强的一条：**首次锁定发生在 t=10.40 s，当时倾角 24.32°**——
不是站稳后才锁，是**在非站立姿态下完成的冷启动重定位**。

另注：本次 query 站立倾角 **9.7°**，而 today20 建图那次实测 **18.25°**（§74.6）。
**采集姿态本身相差约 8.5°，重定位仍成功。**

「27.8 s 那段是趴下」是**依据倾角的推断**，未经用户确认。

### 75.5 暴露的真实缺陷

1. **获取时延不达标**：`inside_710_room` 12.40 s、全长段 10.40 s，合同 <5 s。
   跨 session 数据上系统在搜索态停留更久，veto 又持续拒绝——
   **veto 的代价在更难的数据上被放大**。这是真实缺陷，不是本数据集的噪声。
2. **两组 no-lock 无法判定对错**：`outside_710_room` / `outside_713_room`。
   可能是 veto 正确拒绝，也可能是 0723 地图未覆盖那两处门外。
   在看 `*_no_lock_unverified_pose.pcd` 叠合图之前**不做判断**。
3. 短片段的 p95 明显劣于全长段（865/1022/1278/1397 ms vs 205 ms），
   原因同上：搜索态占比高。

### 75.6 这批数据的意义与边界

**意义**：today20 的结构性风险（同一次运行的时间切片、参考非独立 GT）
第一次被实质性地打开了一个口子。跨 4 个月、跨 session、跨采集姿态，
veto 配置给出 4 锁 / 2 不锁 / 零错锁，且有 1.3 cm 级的独立复现证据。

**边界**：**n=6，不构成任何比例估计。** 它能回答「有没有整体崩掉」「有没有错锁」，
回答不了「成功率是多少」。要让比例数字有意义（±15%），需 40 组以上。

### 75.7 资产

```
~/ros_ws/n3mapping_v1_closeout/floor7_full/floor7_710toMeeting_full/   全长 manifest（1848 帧）
$BL/eval_floor7_on0723/                                               6 组输出
生成命令：
  python3 tools/extract_ros1_relocalization_manifest.py \
    --bag /home/user/ros_ws/bagfile/n3mapping_test/floor7_710toMeeting.bag \
    --output <dir> --episode floor7_710toMeeting_full \
    --cloud-topic /cloud_registered_body --odom-topic /Odometry --duration 0
```

---

## 76. 外部审查发现的四个实现/测量缺陷（2026-07-26）

用户请外部模型审查 `wip/freespace-select`。以下四条**已在代码中逐条核实成立**，
其中两条影响本记录此前已下的结论。核实用的是 GitHub 上的分支快照
（远端主机当时不可达），文件行号对应提交 `1eddf00`。

### 76.1 veto 的安全性被表述过强 —— 我的错误

此前 README / HANDOFF / worklog / 代码注释均写着 veto
「只能减少锁、不可能制造错锁」。**该表述超出代码能保证的范围。**

`src/world_localizing.cpp:928`，在 accept 与 reject 两条路径之后**无条件**执行：

```cpp
clearRelocHypotheses();   // 2161-2168 行：清空 pending_hypotheses_、
                          // hypothesis_window_count_、winner_streak_、
                          // has_last_window_winner_transform_、
                          // last_window_winner_map_odom_
```

因此 veto 拒绝一个窗口后，**全部证据被遗忘**，下一窗口从零重新检索、重新聚类、
重新 ICP，top-K 与 yaw 种子都可能不同，仍可能锁到另一个错误假设。

- **能证明**：veto 对**单窗口接受事件**具有局部单调性。
- **不能证明**：加入 veto 后 episode 级错误 FULL 集合 ⊆ baseline 错误 FULL 集合。

today20 上「没有弄坏任何 baseline 正确案例」是**实测结果，不是结构保证**。
相关表述已在 README、HANDOFF_next、本文件中改正。

### 76.2 信息矩阵探针可能记录了未被采用的位姿 —— §74.3 降级为未验证

`world_localizing.cpp:495-507` 记录 `mr.information`（即 refined 位姿
`mr.T_target_source` 的 Hessian）。但 517-527 行：

```cpp
Eigen::Isometry3d selected_pose = predicted_pose;
if (hasValidRegistrationResult(mr)) {
  const auto refined_visibility = evaluatePoseVisibility(submap, query_cloud, mr.T_target_source);
  if (refined_visibility.valid && refined_visibility.consistency_ratio > ...)
    selected_pose = mr.T_target_source;      // 只有更好才切换
  hyp.T_map_odom = selected_pose * odom_pose.inverse();
}
```

若 refined visibility 更差，最终位姿保持 `predicted_pose`，
而 `hyp.last_rot_info_min` 仍来自**被否决的** refined 位姿。
即可能出现「锁定位姿 = predicted，LockInfo H = 被否决的 refined 的 Hessian」。

**§74.3「信息矩阵不可分」的结论因此降级为「未验证」**——
不是反过来成立，而是那次测量的归属不可信，需重测。

审查另指出三点（未逐条核实，但机理合理，重测时应一并处理）：
1. 取的是条件旋转块 `Hrr`，而非消去平移耦合后的边缘信息
   `Hrr − Hrt Htt⁻¹ Htr`（Schur 补）。倾装雷达 + 长走廊下平移-旋转耦合可能很强；
2. small_gicp 返回的 `H` 是**最后一次更新前**线性化的结果，不对应最终位姿；
3. `1/sqrt(λ_min)` 缺残差噪声尺度与点数归一化，不能直接读成绝对角度标准差。

### 76.3 自由空间分数把 unknown 混进分母

`src/free_space_grid.cpp` 的 `score()`：`++scored` 发生在判断 occupied/free **之前**，
所以所有落在包围盒内的点都进分母，包括既非 occupied 也非 free 的 unknown 格。

于是 `value = occ% − free%` 把两个不同问题混在一起：
**位姿是否与已知地图矛盾** vs **该候选区域是否有足够地图覆盖**。
覆盖率低的候选可以靠极少数已知点赢得 argmax。

`Score` 应至少拆出 `known_points` / `unknown_points` / `known_fraction`
与 `occupied_given_known` / `free_given_known`。

### 76.4 方差公式少一项（影响有限）

occupied/free 互斥，单点变量取值 +1/−1/0，iid 方差应为
`(p_o + p_f − (p_o − p_f)²)/n`；代码写的是 `(p_o(1−p_o) + p_f(1−p_f))/n`，
**少了 `2·p_o·p_f/n`，系统性低估方差**。

但这条**不改变任何结论**：5σ 判据已因「主导误差是模型误差而非采样噪声」
被 §72.3 证伪，公式改对也救不回来。

### 76.5 其余核实成立但优先级较低的两条

- `hasValidRegistrationResult()` 只要求 converged + fitness/inlier/pose 有限，
  **没有 fitness 阈值也没有 inlier 阈值**，比假设初始化时的门松得多；
  但它同样会累加 `cumulative_log_likelihood` 与 `converged_updates`。
  即「优化器宣布收敛」被当成了「一次有效独立支持」。
- README 曾同时保留「地图倾斜已推翻」（§69）与
  「roll 错误是地图倾斜 4.14° 的直接后果」两条互相矛盾的表述。已改正。

### 76.6 审查中未采纳 / 暂缓的部分

- **拆分 `same_physical_pose` 语义**（现用 3.0 m / 57.3° 判「同一位姿」，
  而合同是 0.5 m / 2°；且 `basin_separated` 只查平移，纯旋转歧义永不被标记）：
  **分析成立**，但该改动会引入三组新阈值，而当前没有独立数据可验证它们。
  **先出这三个尺度上的统计，不改运行行为。**
- 评测协议 V2 的七个子条目中，**刚需只有两条**（完整流水线跑、
  baseline 正确集不得回归），其余为增强项。
- 审查提出的整体方向（保留少量 basin、不急着选 winner、主动获取新观测）
  与已有工作卡 C/D 是同一件事，非新增路线。
- **唯一的新想法**：四足可在**足底位置不变**的前提下，通过站立/下蹲/小幅俯仰
  改变雷达高度与倾角，从而制造真实视角基线。倾装雷达在高度变化后遮挡关系确实不同。
  这是目前唯一可能绕开「必须走动」前提的方案，**值得采数据验证**（新工作卡 H）。

---

## 77. 测量正确性补丁与 §74.3 重测（2026-07-26）

针对 §76 的三条缺陷做修复，**要求不改变任何流水线行为**，然后用修正后的探针重做 §74.3。

### 77.1 「逐位一致」不是可达的验收门 —— 协议 V2 必须改

预注册判据原为「today20 逐例结果逐位不变」。实测 **20 例中 14 例的位姿字段不同**，
但决策字段（`algorithm_lock` / `lock_frame_index` / `lock_stamp_ns` /
`relocalization_seed_keyframe_id` / `relocalization_support_keyframe_id` /
`final_state`）**全部一致**，位姿最大差 **4.07e-15 m**。

进一步用**同一个二进制连跑两次**：位置差 **1.776e-15 m**。

**管线本身是非确定性的**（多线程 GICP 的浮点归约顺序）。4e-15 落在其自身噪声地板内。

因此：

- 本补丁按正确判据 **PASS**（决策全同 + 位姿差 4e-15 m，比合同 0.50 m 小 14 个数量级）；
- **外部审查建议的「同一批输入输出必须逐位一致」若照抄会让每次回归假性失败。**
  协议 V2 的回归门应写成：**决策字段全同，且位姿差 < 1e-9 m / 1e-6 rad。**

### 77.2 修复内容

1. **探针归属**：信息矩阵的记录从「配准结束时」移到「`selected_pose` 确定之后」，
   并新增 `last_pose_is_refined` 显式标注采用的是 predicted 还是 refined。
2. **边缘化旋转信息**：新增 `H_marg = H_rr − H_rt · H_tt⁺ · H_rtᵀ`（Schur 补），
   与原来的条件块 `H_rr` 并列输出。
3. **分数拆分**：`Score` 增加 `known_points` / `known_fraction` /
   `occupied_given_known` / `free_given_known`；`value` 本身**一字节未动**。
   方差补上缺失的 `2·p_o·p_f/n`（互斥变量取值 +1/−1/0）。
4. **`prod_quality`**：记录该次配准是否达到假设初始化时的 fitness/inlier 门槛，
   **只记录不作用**。

### 77.3 §74.3 重测结果：仍不可分

| case | 边缘化 H_marg | 条件 H_rr | 采用位姿 | prod_q | t_err | roll | pitch |
|---|---|---|---|---|---|---|---|
| 11-45-43 | **2.951e5** | 3.290e5 | refined | 1 | 1.132 | **2.12** | **2.05** |
| 11-55-38 | 7.226e5 | 1.038e6 | refined | 1 | 0.049 | 1.37 | 0.25 |
| 11-47-53 | 1.055e6 | 1.723e6 | **predicted** | 1 | 0.445 | 1.07 | 0.03 |
| 11-56-10 | 1.337e6 | 1.895e6 | **predicted** | 1 | 0.038 | 0.85 | 0.41 |
| 11-58-53 | 1.358e6 | 2.760e6 | refined | 1 | 0.186 | 0.07 | 0.53 |
| **11-54-55** | **2.328e6** | 2.942e6 | **predicted** | 1 | 0.031 | **2.46** | 0.12 |
| 11-48-26 | 2.643e6 | 3.399e6 | predicted | 1 | 0.056 | 0.17 | 0.22 |
| 11-53-11 | 2.717e6 | 5.322e6 | refined | 1 | 0.032 | 0.82 | 0.06 |
| 11-52-34 | 4.534e6 | 4.778e6 | refined | 1 | 0.034 | 0.04 | 0.31 |
| 11-52-00 | 4.888e6 | 5.778e6 | refined | 1 | 0.107 | 0.58 | 0.31 |
| 11-53-44 | 4.919e6 | 6.641e6 | refined | 1 | 0.107 | 0.12 | 0.78 |
| 11-54-17 | 5.621e6 | 6.272e6 | refined | 1 | 0.102 | 0.68 | 0.02 |
| 11-58-18 | 8.124e6 | 9.722e6 | refined | 1 | 0.020 | 0.09 | 0.07 |
| 11-57-16 | 9.810e6 | 9.997e6 | refined | 1 | 0.022 | 0.02 | 0.48 |

**roll 误差最大的 11-54-55（2.46°）边缘化信息量 2.328e6，高于四个达标案例**
（7.23e5 / 1.06e6 / 1.34e6 / 1.36e6）。4 组重叠对，**不可分**。

存在可见趋势（信息量越低姿态误差越大：9.81e6→0.02°，2.95e5→2.12°），
**但 11-54-55 决定性地破坏了它。是弱相关量，不能作闸门。**

### 77.4 错配确实存在，且正好命中关键案例

14 次锁定中 **4 次 `pose=predicted`**（11-47-53 / 11-56-10 / 11-54-55 / 11-48-26），
即 refined visibility 更差、最终保留了 predicted 位姿。
**其中包括 11-54-55——§74.3 整条结论所依据的那个案例。** §76.2 的指认成立。

### 77.5 本次修复的未尽之处（不要当成已完成）

**探针只做到一半。** 记录时机与 `pose=refined/predicted` 标注已修正，
但记录的仍是 `mr.information`，即 **refined 位姿的 Hessian**。
对那 4 个 `pose=predicted` 的案例，H 依然不描述真正被采用的位姿。
**审查建议的「在实际被选中的 pose 重新线性化一次」尚未实现。**

因此 77.3 的结论应读作：**在「边缘化 + 归属已标注」这一层级上仍不可分**；
11-54-55 那一行要完全可信，还需补做重新线性化。

另两条已知但未处理（见 §76.2）：small_gicp 返回的 `H` 是最后一次更新**前**
线性化的结果；`1/sqrt(λ)` 缺残差噪声尺度与点数归一化，不能读成绝对角度标准差。

### 77.6 关闭的一条顾虑

`prod_quality=1` 在全部 14 次锁定上均成立——
**`hasValidRegistrationResult()` 的宽松口子在这些案例里一次都没被利用。**
§76.5 的第一条对当前失败案例不构成解释。

### 77.7 资产

```
$BL/eval_probefix/   补丁后 20 case 输出（用于逐位对照）
/tmp/join74.py       探针输出与姿态误差的联立表
/tmp/det1 /tmp/det2  同一二进制两次运行，用于确认非确定性
```

---

## 78. 工作卡 C/D 第一版：实测失败（2026-07-26）

### 78.1 改动

`clearRelocHypotheses()` 原本在 accept/reject 之后无条件执行（§76.1）。
改为**只在接受时清空**；拒绝时保留假设集，并保留
`hypothesis_window_start_odom_pose_` 不重置，使 `evidence_motion_translation`
度量**从播种起累积的真实基线**而非单窗口内的量。
无新阈值；仅加一个安全阀（`kRelocPersistMaxFrames = 300`，30 s @10 Hz）
与基线诊断日志。开关 `N3MAPPING_RELOC_PERSIST`。

### 78.2 机制目标达成，但结果更差

设计目标确实达成：

```
evidence_motion_translation:  1–3 cm  →  3.34 m (11-57-16) / 4.89 m (11-52-00)
```

**积累基线涨了两到三个数量级。** 但 20 case 结果：

| 配置 | 正确 | 错误 | 不锁 | 灾难性 >5 m | 最坏 |
|---|---|---|---|---|---|
| **veto（仍是最好）** | **12** | 2 | 6 | **0** | 1.132 m |
| 持久化 + veto | 8 | 2 | 10 | 1 | 32.052→33.052 m |
| 持久化，关 veto | 9 | 3 | 8 | 2 | 33.097 m |

**两种配置都比 veto 差。** 关掉 veto 并不能救回来，
说明**持久化第一版本身是坏的**，不是被 veto 拖累。

破坏的两条红线：11-46-29 的 33 m 灾难性错锁**回来了**（veto 本已挡住）；
11-47-53 从 CORRECT 0.187 变成 WRONG 1.273。

唯一收获：**11-54-55 从 WRONG（roll 2.46°）变成 CORRECT 0.023 m**——
§74 用五个假设都没解释的那个姿态误差，持久化直接修好了。
机制合理：更多独立观测让配准有机会从别的视角修正姿态，
而不是在近静态快照上定死。

### 78.3 差点误诊：`pass_free_space` 没有拒绝原因标签

四个 CORRECT→no-lock 的案例，拒绝原因写的是 `stability_guard`。
初查以为是 `moving_visibility_required` 在积累位移后触发——**实测否定**：

```
11-52-00  motion=4.89 m  top1_visibility=+0.549  → 仍被拒
11-57-16  motion=3.34 m  top1_visibility=+0.254  → 仍被拒
```

可见性证据是**正的**。真正原因：`stability_guard` 是拒绝原因链的最后一个 else，
而接受条件里的 `pass_free_space` **从未被加入原因链**，
自由空间否决因此落进泛化桶。

**这正是外部审查明确点名、而我判为「优先级较低」的那一条**（§76.6），
它不是卫生问题，是诊断能力问题。

### 78.4 已识别的设计缺陷：持久化 + veto = 永久否决

veto 目前是**逐帧无状态的 argmax 身份比对**。假设集持续存在时，
一次不一致就永远不一致，veto 每帧开火，把整个 episode 锁死。

正确形式应为**跨独立视角累积否定**（从 N 个独立视角都被否定才算否定），
而这恰恰是持久化才使得可能的事。审查建议的「输出可信集合而非唯一 winner」
指的就是这个，§76.6 判为「暂缓」属**误判**。

另一个缺陷：持久化后 `hypothesis_window_count_` 不再重置，
决策从「每 5 帧一次」变成「第 5 帧后每帧一次」，接受机会大增，
且 `winner_streak_` 无界增长使该门形同虚设。

### 78.5 处置

**`N3MAPPING_RELOC_PERSIST` 默认改为关闭**（需显式 `=1` 开启）。
被证伪的改动不得默认生效。代码保留作实验资产。

---

## 79. 地图被拉坏：z 漂移 + 闭环失效（2026-07-26）

用户在 CloudCompare 中查看 `global_map.pcd`，指出「感觉全局地图有问题，很多重叠」。
**核实成立，且是本项目迄今最重要的发现。**

### 79.1 重访自洽性：地图与自己矛盾

不依赖任何 GT 的内部一致性检验（`/tmp/revisit.py`）：
同一 (x,y) 在不同时刻被访问时，z 是否一致。办公楼层是平的，它必须一致。

```
样本 8250   时长 824.9 s   轨迹总长 272.7 m
z 范围 -5.74 .. -0.02 m   跨度 5.73 m

重访对 44021 个（xy <1.0 m，时间间隔 >30 s）
  |dz|  中位 0.048 m   p90 0.266 m   最大 2.492 m
  |dz| > 0.5 m 的占 6.8%（3004/44021）
```

最刺眼的一对：

```
t=  0.0 s   xy=(-0.01,  0.01)   z=-0.019
t=102.4 s   xy 相距 0.16 m      z=-2.511      ← 同一地点，差 2.49 m
```

**机器人回到出发点 16 cm 以内，地图说它低了 2.49 m。**

量级自洽：272.7 m 行程 / 5.73 m z 跨度 → 重力方向偏约 **1.2°**；
§69 实测 LIO 重力估计每 19.5 s 漂移 0.21°–12.97°。
**重力估计漂移 → 「上」的方向转了 → 轨迹被拉斜 → z 漂移。**

### 79.2 闭环基本没跑

`f7_0723_map/optimization.log`（518 行）：

```
context=mapping_incremental   257 次
context=loop_closure            2 次   ← 整场 272.7 m 只有两次
```

且两次的影响可忽略：

```
loop impact edges=1 accepted=1
  pose_update_mean_max_t = 4.07e-05 / 4.97e-04   ← 最大位姿修正 0.5 毫米
  pose_update_mean_max_r = 2.07e-06 / 1.78e-05   ← 最大 0.001°
```

**闭环触发两次，把地图移动了半毫米。等于没做。**
而同一段轨迹有 44021 个重访对，闭环本该反复触发。

### 79.3 根因候选（强，但未经日志确认）

```yaml
loop_max_icp_translation: 2.0    # 闭环 ICP 修正超过 2.0 m 即拒绝
loop_min_inlier_ratio:    0.7    # 内点率门槛
```

实测漂移已达 **2.49 m > 2.0 m**。**这构成死循环**：
漂移小时闭环能修；一旦超过 2.0 m，所需的修正量反而被当作外点拒掉，
于是永远修不回来。**地图越歪，闭环越不敢闭。**

**确认方法（未做）**：开 `loop_debug_logger` 重跑建图，
统计有多少闭环候选被 `loop_max_icp_translation` / `loop_min_inlier_ratio` 拒掉。

### 79.4 推翻 §69 的一条结论

§69/§70 曾断定「地板 z 实际跨度 7.41 m」是**真实的多标高结构**（局部却处处水平）。
**推翻**：局部水平与整体漂移并不矛盾——漂移是缓变的，局部拟合当然是平的。
7.41 m 是**建图 z 漂移**。

工作卡 E（地图重力对齐）§69 曾以「缺陷不存在」关闭，**须重开**：
缺陷存在，但形态不是「整体刚性倾斜」（那部分 §69 的证伪仍然成立——
刚性去倾斜在度量上恒等无效），而是**累积 z 漂移 + 闭环失效**。

### 79.5 这件事为什么压倒其余一切

**所有「正确 / 错误」都是拿这条被拉歪的轨迹当参考量出来的。**
`same_run_dense_optimized_reference` 就是这条有 5.73 m z 漂移的轨迹；
`global_map.pcd` 也是用这些歪掉的位姿拼出来的。

即：**重定位被要求去匹配一个畸变的模型，然后用同一个畸变的东西评判它对不对。**
在这个基础上继续调判别准则，是在给一个坏基准做曲线拟合。

注：三个灾难性别名的 xy 距离分别是 32.96 / 29.7 / 27.9 m，
**不是同一地点的 z 重影**，所以别名本身不能简单归因于漂移；
但地图畸变会同时影响描述子、ICP、自由空间栅格与参考轨迹。

### 79.6 下一步：先定位漂移在前端还是后端

**这决定修哪里。**

- **前端（FastLIO 里程计）就漂** → 必须修重力估计 / IMU 标定，后端只能缓解
- **前端没漂、后端没闭上** → 修闭环门限即可

**可用数据（已确认存在）**：

```
/home/user/ros_ws/bagfile/0723_n3m_gate/f7_0723_full-001.bag   完整建图 bag
/home/user/ros_ws/bagfile/0723_n3m_gate/2026-07-23-11-*.bag    20 个 query 分片
```

且 `manifests/*/frames.csv` 的 `tx,ty,tz,qx..qw` **就是 bag 里 `/Odometry` 的原始
LIO 位姿**（未经后端优化），可与 `dense_trajectory.csv`（优化后）逐时间戳对照。

**判据**：对原始 LIO 里程计跑同一个 `revisit.py`。
若原始 LIO 的重访 |dz| 同样达到米级 → 前端漂移；若原始一致而优化后不一致 → 后端问题。

### 79.7 建议的建图验收门（不需要 GT）

```
重访自洽性：|dz| p90 < 0.10 m，最大 < 0.30 m
闭环数量：每 100 m 行程至少 N 次 accepted loop
闭环影响：pose_update_max_t 应为厘米级，不应是 0.5 毫米
```

这三条都是地图对自身的一致性检验，**不需要独立 GT，可以立刻用起来**。

### 79.8 资产

```
Q:\DocumentFile\bull&horseLife\jammy_dev\map_check\f7_0723_global_map.pcd   8.5 MB / 532683 点
Q:\DocumentFile\bull&horseLife\jammy_dev\map_check\dense_trajectory.csv
Q:\DocumentFile\bull&horseLife\jammy_dev\docs\floorplan_alias_sites.html    三处别名点平面图（离线单文件）
/tmp/revisit.py                                                            重访自洽性检验
```

查看器：CloudCompare（Windows 原生，直接读 binary PCD，按 z 着色 + 侧视图
可肉眼确认走廊是否倾斜）。

---

## 80. 建图闭环：完整因果链与三轮实测（2026-07-29）

承 §79。用户要求「从第一性原理修复闭环处理与检测」。三轮实测，**判据未达成**，
但把整条因果链量清楚了。

### 80.1 前端根因：重力初值是假设，不是估计

LIO bag 里有 `/lio_gravity`（8250 条，FAST_LIO 重力发布补丁的产物）：

```
t=  0 s   g = (-0.00000, 0.00000, -9.80900)   偏转  0.000°  ← 硬初始化
t= 60 s   g = (-1.50986, 0.86391, -9.65352)   偏转 10.215°
t=780 s   g = (-0.64159, -0.33031, -9.78242)  偏转  4.219°
峰值 10.779°   最终 4.228°   |g| 恒为 9.8090（只估方向）
```

x/y 精确为零、模长精确 9.809 —— **FAST_LIO 把「开机时机体水平」当初值**，
而该机器人雷达倾装约 18°。且**没有静止初始化窗口**：t=10 s 已走 0.75 m，
t=20 s 走 4.13 m。重力估计只能边走边收敛，**且从未回到初始参考系**（终值 4.2°）。

最严重的重访（t=0 vs t=102.4 s，|dz|=2.49 m）正落在偏转 10.2° 的收敛暂态里。

### 80.2 检测没问题，淘汰在验证

候选来源：`rhpd_primary` 405 / `spatial_radius` 183（SC 已在退役路径上，不依赖）。
**588 个候选里 262 个配准质量合格**（fitness<0.2 且 inlier≥0.7），覆盖 33 个关键帧。

### 80.3 同一个反模式写了四层

**原则**：闭环是关于几何的断言，证据全在两帧扫描里；
**先验只有提名权，没有否决权；不确定性属于噪声模型，不属于阈值。**

| 层 | 违反 | 代价 |
|---|---|---|
| verifier | `loop.verified = match_result.converged && ...` | 181 个，fitness 中位 0.053 / inlier 0.950（**比接受的两条还好**） |
| verifier | `geometry_ok = icp_translation_norm <= 2.0 m` | 64 个，fitness 0.032 / inlier 0.989，修正中位 5.34 m |
| core | `T_pred.norm() > loop_max_range(30)` | 120 个 |
| referee | `predicted_translation_norm > 5.0 && segment<=0.5` | 86 个，fitness 中位 0.052 |
| graph trial | `residual_translation_norm_after >= 2.0 m` | 28/31（通过其余全部检验者） |

graph trial 那层最能说明问题。对**证据最强的 31 个**（配准合格 + 段一致性满分）：

```
residual_translation_norm_after   中位  5.6052 m   ← 被 2.0 m 门砍掉
existing_loop_residual_delta      中位 -0.0004 m   ← 加了它，已有约束变好
odom_residual_delta               全部  0.0000 m
max_pose_update_translation       中位  0.0033 m   ← 试算只挪了 3 毫米
```

**真正衡量一致性的两个量就在同一结构体里，算出来了，没参与决策。**
而 3 毫米说明 `residual_after` 为什么大：一条边拉不动 256 条边的链条，
所以任何大修正试算后都原样保留。这道门测的是「一步没吸收完」。

### 80.4 已实施的修改（提交在分支上）

1. `loop.verified = fitness_ok && inlier_ok`；`converged` 与 `geometry_ok` 降级为诊断
2. heightmap 诊断改为「质量合格即计算」（原先 gate 在 `converged` 上，
   导致 541/588 到达 referee 时身上没有任何证据）
3. `loop_max_range` 30 → 150 m，并在注释中限定其语义为**算力边界**
4. referee `spatial_only_unconfirmed`：段一致性满分**也算独立确认**
5. referee `large_prediction_with_weak_segment` → `unconfirmed_weak_segment`
   （删先验项，改为「段弱**且**无描述子」；**未放宽成来者不拒**）
6. graph trial 判据 → `existing_loop_residual_delta / odom_residual_delta > 0.05 m`
   （实测数值噪声上限 2e-4，该界为其 250 倍）
7. `odom_noise_rotation` 0.001 → 0.0046 rad
   （推导：0.057°×√257 = 0.91°，而实测重力持续偏 4.2°；4.2°/√257 = 0.26°）

### 80.5 三轮结果：判据未达成

| | 闭环边 | pose_update 最大 | z 跨度 | \|dz\| 中位 | p90 | 最大 | >0.5 m |
|---|---|---|---|---|---|---|---|
| orig | 2 | 0.0004 m | 5.71 m | 0.041 | 0.852 | 2.486 | 17/99 |
| fix2（1–6） | 19 | 0.0223 m | 5.69 m | 0.039 | 0.901 | 2.485 | 16/98 |
| fix3（+7） | 16 | 0.3789 m | 5.60 m | 0.046 | **1.374** | 2.481 | 22/105 |

判据 `p90<0.10 m、最大<0.30 m、闭环≥20` —— **FAIL**。
几何未折叠（轨迹总长三轮均为 235.7 m），但也未被修正。

### 80.6 最后一个未解的压制：鲁棒核

`fix2` 的优化日志：

```
loop_residual_t = 18.0827 -> 18.0820      18 米残差，优化后减少 0.7 毫米
loop_residual_t = 19.6327 -> 19.6316      1.1 毫米
```

两层压制：

```yaml
odom_noise_position: 0.01     # 链条信息量 256/0.01² ≈ 2.6e6
loop_noise_position: 0.5      # 闭环信息量  19/0.5²  ≈ 76      → 差约 3 万倍
robust_kernel_type: "Cauchy"
robust_kernel_delta: 1.0      # 权重 1/(1+(r/δ)²)：r=18 m 时 0.003
```

**δ=1.0 的语义是「残差超过 1 米即视为外点」，而这张图需要 5–18 米的修正。
鲁棒核在精确消灭正确的大闭环。** fix3 松了姿态噪声后位姿修正涨了 17 倍
（0.022 → 0.379 m）仍远远不够，正是因为压不过核函数 2–3 个数量级的衰减。

**下一步（未做）**：
- Cauchy δ 提到与预期修正同量级，或
- 分级策略：先无核批量收敛，收敛后再加核剔除外点（GNC 思路），或
- switchable constraints

### 80.7 可复用结论

1. **「先验只有提名权，没有否决权」** —— 该反模式在本代码库出现四次。
   任何以「与当前位姿的分歧量」为判据的闭环门，
   都在结构上保证大漂移不可修复：误差越大，正确闭环要求的修正越大，越会被拒。
2. **鲁棒核的 δ 是外点尺度，不是修正尺度。** 当 δ 小于系统需要的修正量时，
   鲁棒核从「防错锁」变成「防修复」。
3. **求解器的 `converged` 不是结果质量。** 该误用同时出现在重定位（§74.4）
   与建图闭环两处。
4. **验收判据必须包含防折叠项。** 三轮里几何都没折叠（轨迹长度不变），
   若只看 z 一致性会误判。

### 80.8 资产

```
~/ros_ws/n3mapping_v1_closeout/run_mapping_loopdebug.sh   开 loop_debug 重跑建图
~/ros_ws/n3mapping_v1_closeout/loopdebug_map/             基线复现（2 条闭环）
~/ros_ws/n3mapping_v1_closeout/loopfix2_map/              1–6 号修改（19 条）
~/ros_ws/n3mapping_v1_closeout/loopfix3_map/              +7 号修改（16 条）
/tmp/gravity.py                                           /lio_gravity 漂移分析
/tmp/pb/n3map_pb2.py                                      protoc 生成，直读 pbstream
tools/map_revisit_consistency.py                          重访自洽性（不需 GT）
LIO bag: artifacts/.../20260723/0723_gravity_v2/full/lio_ros2   建图输入，含 /lio_gravity
```

---

## 81. 闭环修不了这张图：候选互相矛盾，根因在前端（2026-07-29）

承 §80。四层门修完、路程判据换掉、鲁棒核开关都试过。**结论是负面的，且稳健。**

### 81.1 帧数排除窗是结构性缺陷（已修）

```cpp
int num_exclude = config_.sc_num_exclude_recent;          // 50
if (static_cast<int>(query_index) < num_exclude) return candidates;   // kf<50 完全不检测
auto accept_old_enough = [&](int64_t id) { return mit->second < search_end; };  // 匹配须早 50 帧
```

实测 2.49 m 漂移的真实重访是 **kf ~33 对 kf ~0，间隔 33 < 50**，
**在结构上不可能成为候选。**

`sc_num_exclude_recent = 50` 是 Scan Context 的原始默认值，
目的是避免平凡自匹配。但**它的单位是关键帧，不是距离**：
本图关键帧间距约 1 m，50 帧 ≈ 50 m 行程，而这栋楼 30 m 内就能绕回原处。
**排除窗比回环尺度还大。**

**已改为按里程计路程排除**（`loop_min_path_length_m = 5.0`），
并移除 `query_index < num_exclude` 那个纯下溢保护的早退。
SC 保留索引截断（它切连续前缀，且在退役路径上）。
接口 `detectLoopCandidates(query_id, keyframes)` 增参，
调用方 `n3mapping_core.cpp` 与 `mapping_resuming.cpp` 同步改。

**效果**：候选 588 → 693，合格 261 → 407，覆盖 query 34 → 49，
早期区间开始出候选且质量极好（q=25 m=0：**fitness 0.0069、inlier 0.995**）。

### 81.2 但地图仍然不动

| | 闭环边 | 后端修正中位 | \|dz\| p90 | 最大 | 几何变化 |
|---|---|---|---|---|---|
| orig | 2 | 6.4e-14 m | 0.852 | 2.486 | — |
| fix2（四层门） | 19 | 0.0222 m | 0.901 | 2.485 | 0.0% |
| fix3（+姿态噪声） | 16 | 0.4199 m | 1.374 | 2.481 | 0.8% |
| fix4（核关） | 15 | 1.2340 m | 0.955 | 2.604 | **4.6%** |
| fix5（+路程排除） | 18 | 0.0114 m | 0.847 | 2.486 | 0.0% |

fix4 是关键对照：**核关掉后图能动 1.2 m，但 |dz| 中位从 0.041 恶化到 0.163，
x 范围缩 4.6%。** 核开着动不了、核关掉就折叠 —— 说明**放进去的闭环是错的**。

### 81.3 成对一致性检验：不存在能修漂移的一致集合

真闭环之间必然互相一致（回路必须闭合）；假闭环是随机的，彼此矛盾。
该检验不需要 GT、不需要先验，只用闭环测量与它们之间的里程计。

**对 fix2 的 19 条已接受边**（`tools/loop_pairwise_consistency.py`）：

```
171 对组合，互相一致仅 1 对
最大互相一致集合 2 条：155→210（修正 0.04 m）、155→215（修正 0.02 m）
```

最能说明问题的一对：`2→75` 要求修正 **26.37 m**，`6→80` 要求 **15.37 m**，
**两者端点只隔 4–5 帧，却相差 11 m。不可能都对。**

**对 fix5 的 407 个合格候选**（`tools/loop_candidate_pcm.py`）：

```
2676 个可比较对（端点不共享、里程计腿 <= 12 帧）
最大互相一致集合仍是 3 条，修正量 0.02–0.13 m
一致度最高的 10 个候选，修正量全部 <= 0.13 m
```

**凡是提出有意义修正的候选，彼此全都不一致。**

#### 对该检验自身的两次质疑，都已排除

1. **容差没考虑里程计腿本身的误差。** 实测漂移率 2.49 m/33 帧 ≈ 0.075 m/帧，
   12 帧的腿可带来 0.9 m，超过原定 0.60 m 固定容差。
   改为 `0.30 + 0.075×腿长(帧)` 后一致对数 223 → 332，
   **最大一致集合仍是 3。**
2. **是否只是 z 方向歧义？** 走廊里 z 最弱可观测，代码库也有
   `vertical_ambiguity_score` 字段。把重访区间 275 个对的回路误差分解：

```
xy: 中位 4.016 m   p10 2.217 m      ← 分歧主要在水平面
z : 中位 1.081 m   p10 0.336 m
xy 一致(<0.5 m) 仅 2/275，且这 2 对 z 也一致
```

**否掉。** 分歧是水平的：q=25 同时匹配 m=0/5/10，各自声称不同的沿走廊偏移，
彼此相差几米 —— 正是走廊上若干关键帧的间距。

### 81.4 结论：和重定位是同一面墙

**RHPD 分不清沿走廊的位置。** 在重定位里表现为 27–33 m 的别名（§44–59），
在建图闭环里表现为几米的水平矛盾。**机制相同。**

因此：**这份数据上的 2.49 m 漂移无法用闭环修复。**
能修它的约束要么不存在，要么被淹没在互相矛盾的别名里；
唯一互相同意的两三条恰好是「不需要修正」的真实近邻重访。

**修法在上游**：修前端重力，让漂移不发生，而不是事后补救。
根因已在 §80.1 量清：重力初值硬编码 `(0,0,-9.809)`（假设机体水平，
而雷达倾装约 18°），且无静止初始化窗口（t=10 s 已走 0.75 m），
峰值偏 10.8°、终值 4.2°，从未回到初始参考系。

### 81.5 遗留的两处未修（已确认存在）

1. **`graph_inconsistent_yaw` 与我已修的平移判据是同一缺陷。**
   它读 `residual_yaw_after`——同一个「单条边拉不动链条」的量。
   §80 我判断它「合法，能检测翻转」而留着，**判断错了**。
   `q=35 m=6`（段一致性满分、referee 放行）就是被它拒的。
2. **`selectBestPerQuery` 每 query 只留一条，且按 fitness/referee energy 选，
   不看信息量也不看互相一致性。** `q=30 m=6`（段一致性满分、间隔 24 帧）
   被 `m=17`（间隔 13 帧）挤掉。**平凡匹配的 fitness 天然最好。**

这两处不修也无所谓——81.3 已证明候选池里没有可用的一致集合。
**但若将来换了能沿走廊定位的描述子，这两处必须一起修，否则真闭环照样进不来。**

### 81.6 资产

```
tools/loop_pairwise_consistency.py   对已接受闭环边做成对一致性检验
tools/loop_candidate_pcm.py          对全部候选搜索最大互相一致集合（容差随腿长缩放）
tools/map_gate.py                    四项预注册判据一次出表（含防折叠）
~/ros_ws/n3mapping_v1_closeout/loopfix{2,3,4,5}_map/   四轮输出
```

---

## 82. 漂移根因定量确认：重力初始化暂态，局限在开头 75 m（2026-07-29）

用户质疑「从第一性原理看，前端漂移真的是因为重力不对吗」。追下去得到定量答案，
并**撤回 §81 的一个结论**。

### 82.1 刚性倾斜在数学上被排除

若重力估计错一个**固定角度**，整张图被**刚性旋转**。刚性变换下同一物理位置的
两次访问仍落在同一点——**重访 |dz| 是刚性变换的不变量**（与 §69 同一论证）。

**所以实测 2.49 m 重访误差排除「重力恒定偏错」，它要求姿态误差随时间变化。**

### 82.2 分离刚性倾斜与非刚性漂移

逐关键帧从**自身点云**拟合地板（半径 <8 m，法向在 world 系），
再把 `floor_z` 对 `(x,y)` 拟合全局平面。落在平面上的部分是刚性倾斜（无害），
**残差就是真正的非刚性漂移**。

```
floor_z = 0.00063x - 0.06452y - 3.5782      等效刚性倾斜 3.69°
地板 z 原始跨度                  6.121 m
去掉刚性倾斜后残差                中位 0.516 m  p90 1.100 m  跨度 4.834 m
```

**3.69° 刚性倾斜解释了大部分表观下降。** 但残差在 t=0 处是 **+3.400 m**，
而 t≈96 s 之后全部落在 ±1.1 m 内。

### 82.3 残差随路程单调衰减 —— 暂态的指纹

改用 **t>150 s 的稳定段**（206 样本）拟合平面，该段自身残差中位仅 **0.231 m**，
再看开头相对它偏多少：

| 累计路程 | 残差 |
|---|---|
| 0.00 m | **+3.952 m** |
| 1.01 m | +3.480 m |
| 3.06 m | +3.028 m |
| 8.15 m | +1.951 m |
| 10.00 m | +1.548 m |
| 29.82 m | +1.182 m |
| 46.51 m | +0.823 m |
| 75.68 m | +0.143 m |
| 111.54 m | −0.041 m |

**单调衰减，75 m 处归零。** 前 10 m 的衰减率 `(3.95−1.55)/10 = 0.24 m/m`，
对应姿态误差 `asin(0.24) ≈ **13.9°**` ——
与 `/lio_gravity` 实测峰值 **10.2–10.8°**（§80.1）吻合。

### 82.4 完整因果链（定量）

1. FAST_LIO 以 `(0,0,-9.809)` 起步 —— **假设开机时机体水平**，而机体/雷达倾装约 18°
2. 无静止初始化窗口（t=10 s 已走 0.75 m），估计器边走边收敛
3. 前 ~10 m 姿态误差 ~14°，**每米注入 0.24 m 的 z 误差**
4. 估计器在随后 ~60 m 内收敛，75 m 处残差归零
5. **暂态期注入的误差是永久的**：烙在前约 20 个关键帧里，后端从未修正
   （2 条闭环边、最大 0.38 毫米，§79.2/§80）
6. 残差因此是**台阶**：地图开头约 15 m 比其余高出最多 4 m
7. t=0 残差 +3.95 与 t=102 残差 +1.18 之差 **2.77 m**，
   与实测重访误差 **2.49 m** 对上

**剩余地图（t>150 s）去掉 3.40° 刚性倾斜后自洽到中位 0.231 m、p90 0.871 m ——
那部分是好的。缺陷局限在开头十几个关键帧。**

### 82.5 一次误判与撤回

**我先用 §82.1 的不变量论证正确排除了「恒定偏差」，随后又拿收敛后的统计量
错误否证了「暂态」**：逐关键帧地板法向偏离 world z 中位 **2.36°**、
前 10% 与后 10% 均值夹角仅 **1.54°**，我据此宣布重力说法被推翻。

**那是误读。** 暂态只占 249 个可用关键帧里的十来个，
被中位数与「前 10% 均值」完全淹没。**用收敛后的统计量去否证暂态假设是无效的。**

### 82.6 撤回 §81 的「重访候选是别名」

§81.3 曾断定重访区间候选互相矛盾（xy 回路误差中位 4.016 m），
因此是感知别名。**该判断使用了被污染的数据，现予撤回。**

PCM 一致性检验依赖里程计腿。而暂态区姿态误差 14°，
**5 帧的腿就产生 `5·sin(14°) = 1.2 m` 的 xy 误差，10 帧产生 2.4 m** ——
与实测的 4 m 分歧同量级。而我的容差用 **0.075 m/帧**（全程平均漂移率），
暂态区实际是 **0.24 m/帧，大 3 倍**。

**因此不能断定 `q=25 m=0`（fitness 0.0069、inlier 0.995）那批候选是别名。
它们可能是真的，而我用被污染的里程计把它们判成了矛盾。**

§81 关于「全局候选池不存在大的一致集合」的结论，
在**稳定段**（t>150 s，里程计可信）仍然成立；
在**暂态段**不成立，需用不依赖里程计腿的方法重判。

### 82.7 下一轮的两条路

**A. 上游修（推荐）**：让暂态不发生。
- 开机原地静止数秒，令 FAST_LIO 从加速度计估出重力后再移动
- 或把标定好的雷达-机体安装倾角喂进去，不让估计器从「水平」猜
- 验收用 `tools/lio_gravity_drift.py` 看偏转峰值，
  与 `tools/floor_normal_vs_world_up.py` 看残差-路程曲线是否还有台阶

**B. 下游修**：专门闭合 kf 0–35 区间。
候选已存在且配准极好（`q=25 m=0`：fitness 0.0069、inlier 0.995）。
但必须先解决两件事：
- `selectBestPerQuery` 会让平凡近邻挤掉它们（§81.5）
- 一致性判据不能依赖穿过暂态区的里程计腿

### 82.8 资产

```
tools/floor_normal_vs_world_up.py   逐关键帧地板法向与地板 world 高度
                                    （附机体离地高，中位 0.507 m，可作合理性校验）
tools/lio_gravity_drift.py          /lio_gravity 偏转
```

---

## 83. S1 证伪：重力初始化不是缺陷，缺陷是运行中丢失对齐（2026-07-29）

预注册判据：`/lio_gravity` 偏转峰值 <2°（现 10.779°）、降到 1° 内 <20 s（现约 120 s）。
**S1 未达标，且判据无需执行即被证伪 —— 前提不成立。**

### 83.1 先纠正我自己的两处错误

worklog §80–§82 我写过「FAST_LIO 假设开机时机体水平」。**错。**
`IMU_Processing.hpp` 里 h2q 的修改确实做了重力对齐：

```cpp
q_init.setFromTwoVectors(mean_acc.normalized(), V3D(0,0,1.0));
init_state.rot = SO3(q_init.toRotationMatrix());
init_state.grav = S2(V3D(0, 0, -gravity_m_s2_));
```

测加速度计 → 转到 world +z → 安装倾角吸进 `init_state.rot`。
`/lio_gravity` 从 `(0,0,-9.809)` 起步**是构造使然**。

### 83.2 S1a（协方差）实施后测得：空操作

改动：用代码自己算的 `cov_acc` 推 `var = acc_var/(g²·N)` 替代硬编码 `1e-5`，
并以原值为下限（只能放松不能收紧）。新增日志实测：

```
[IMU_init] N=20 acc_var=0.000230 grav_dir_var=1.000e-05 (0.181 deg)
```

`acc_var = 0.000230` (m/s²)² → σ = **0.015 m/s²**，推出的方向不确定度仅 0.006°，
**低于下限，协方差一字未改。** 该次 LIO 与基线逐位相同。

**前提被证伪：重力先验并未相对噪声高估 —— 噪声本身极小。**

### 83.3 S1b（加长窗口）：无需实施即证伪

离线量原始 bag 的加速度计（`0723_derived/full_ros2`，`/go2w/livox/imu`）：

| 样本数 | 时长 | `mean_acc` | 与前 20 样本方向夹角 |
|---|---|---|---|
| 20 | 0.09 s | [−0.2390, −0.0083, 0.9670] | 0.000° |
| 400 | 1.99 s | [−0.2338, −0.0047, 0.9616] | **0.304°** |
| 4000 | 19.99 s | [−0.2338, −0.0044, 0.9617] | **0.315°** |

**20 个样本与 4000 个样本的方向只差 0.3°。加长窗口不会改变任何东西。**
机体前 20 s 静止（每 2 s 的 |acc| std 稳定在 0.021–0.025）。

### 83.4 顺带确认：IMU 是 g 单位，换算路径完好

```
|acc| 逐样本中位 0.99003        配置 gravity_m_s2 = 9.79338
```

Livox IMU 输出 g 单位。`IMU_Processing.hpp:312`
`acc_avr = acc_avr * gravity_m_s2_ / mean_acc.norm()` **换算在位**，不是缺陷。

但记一处量化偏差：比例因子用 20 样本的 `mean_acc.norm()=0.99644`，
而持续值是 0.98976 —— **0.67% 的尺度误差**（约 0.066 m/s² 的等效常偏）。
量级不足以解释 2–5° 的姿态误差，但值得在后续排查中记住。

### 83.5 初始化是对的，运行中才丢

`mean_acc` 归一后离机体 z 轴 **13.89°**（安装倾角）。
`kf0` 的 `pose_odom` 旋转角 **13.58°** —— **倾角吸收正确**。
`kf0` 处 `R·acc` 离 world z 仅 **0.47°**。

而随后（用不受机体姿态污染的逐帧地板法向为准）：

```
早期 kf 0–33   法向偏离 world z  2.39° / 3.74° / 2.07° / 3.43° / 3.22° / 4.64° / 5.39°
后期均值                        1.64°（前 10%）→ 1.93°（后 10%）
```

5° 摊在 29.8 m 路程上给 2.6 m，与实测的 2.77 m 吻合。

### 83.6 漂移是真的，这次证据不循环

同一地点（原点 2.5 m 内）、不同时刻，**雷达点云里实测的地板高度**：

```
t=  0.0 s  kf 0   xy=(-0.01, 0.01)   观测地板 z = -0.181
t=102.3 s  kf32   xy=(-0.47, 0.06)   观测地板 z = -2.950
                   xy 相距 0.46 m，地板高度差 2.77 m
```

**用的是点云观测的地板，不是轨迹 z，所以不依赖被质疑的位姿。**
地板是物理实体，相距 0.46 m 不可能有两个高度 —— **漂移确认，非多标高结构。**

### 83.7 结论与下一步

**重力初始化不是缺陷。** 缺陷是估计器在运行中丢失重力对齐（world 系相对物理地板
倾斜 2–5°，随时间变化），而**位姿图里没有任何绝对姿态约束** ——
258 条边全是相对约束，前端倾斜多少，下游无从察觉也无从修正。

**直接进 S3：位姿图加地板法向/重力约束。** 地板在 **249/257** 帧可拟合，
观测充足。S1b 跳过（已证伪）。

### 83.8 资产

```
~/ros_ws/n3mapping_v1_closeout/run_lio.sh      raw→LIO 跑批（此前缺失），输出 ~/ros_ws/lio_runs/<tag>/
~/ros_ws/lio_runs/s1a_gravcov/                 S1a 输出（与基线相同，留作对照）
FAST_LIO src/IMU_Processing.hpp                协方差改动保留（下限保证不收紧）+ [IMU_init] 诊断日志
```

---

## 84. S3 地板姿态因子：首次让地图真正动起来，但未达标（2026-07-30）

预注册判据：地板法向偏离 world z 的 **p90 < 1.0°**（基线 3.69°），
外加 `map_gate` 四项。**FAIL，实测 p90 3.30°。**

### 84.1 实现

新增 `floor_attitude.{h,cpp}`：从每帧**自身点云**（sensor 系）拟合脚下 8 m 内的地板，
输出 body 系法向。`GraphOptimizer::addFloorAttitudeFactor` 用 GTSAM 现成的

```cpp
Pose3AttitudeFactor(key, Unit3(0,0,1), noise, Unit3(floor_normal_body))
```

**约束 roll/pitch，不碰 yaw** —— 正是漂移的那两个自由度。
噪声取 **1.0°**，依据不是拟合精度（几百点的平面拟合是亚度级），
而是「建筑地板有多水平」这一假设本身的不确定度 —— 较松的那一项才该进噪声模型。
拒绝条件：平面度 >0.20、支撑点 <400、法向离传感器上方向 >35°
（防止把墙面或坡道当水平地面）。

实测 **225 个姿态因子 / 257 关键帧**。

### 84.2 结果：首次实质进展，但判据未达成

| | 基线 | S3 |
|---|---|---|
| 后端对前端位移修正（中位） | 6.4e-14 m | **0.977 m** |
| 后端修正（最大） | 0.00038 m | **2.504 m** |
| 地板 z 跨度 | 6.121 m | **4.196 m** |
| 法向偏离 p90 | 3.69° | **3.30°** |
| \|dz\| p90 | 0.852 m | **0.503 m** |
| \|dz\| 最大 | 2.486 m | 2.328 m |
| \|dz\| 中位 | 0.041 m | **0.276 m**（变差） |
| \|dz\|>0.5 m | 17/99 | 11/101 |
| 闭环边 | 2 | 6 |
| 几何变化 | — | 0.0%（未折叠） |

**这是位姿图第一次真正吸收约束**（后端修正从飞米级到 0.977 m 中位）。
z 跨度、p90、超标对数全部改善。

**但中位 |dz| 从 0.041 恶化到 0.276**：姿态因子把整体倾斜拉回来了，
而近邻重访本来就一致，被全局调整带偏。这是诚实的代价，不粉饰。

### 84.3 未达标的原因：刚度比，和闭环那次同一个

```
姿态因子   225 × 1/(0.0175²) = 7.3e5
里程计旋转 256 × 1/(0.001²)  = 2.56e8      → 里程计刚 350 倍
```

**里程计又一次压倒了绝对观测。**
`odom_noise_rotation = 0.001`（0.057°/边）是对传感器的虚假断言；
§82 推导的诚实值是 **0.0145 rad**（2.49 m 高度误差 / 约 30 m 行程
→ `asin(2.49/30)=4.8°`，摊到 33 条边 → `4.8/√33 = 0.83°`）。

**关键区别**：§80 的 fix3 松过这个值，结果变差（p90 0.852→1.374），
因为当时图能动之后只有互相矛盾的别名闭环可去。
**现在有 225 个绝对姿态观测撑着**，松到 0.015 后两者信息量分别是
1.14e6 与 7.3e5，量级相当。S3b 测这个组合。

### 84.4 一处测量错误与两处架构约束

**测量错误**：`floor_normal_vs_world_up.py` 读的是 `pose_odom`（前端），
而姿态因子改的是 `pose_optimized`。首次验收时输出与基线**逐位相同**，
差点被读成「毫无效果」。工具已改为可选位姿来源。

**架构约束**：`core/` 层是 ROS-free 且 glog-free 的
（`check_humble_wrapper_no_direct_backend_calls.cmake` 在守），
最初把 `VLOG` 写进 core 编译失败；已把可观测性移到 optimizer 层，
改用既有的 `[OPTIMIZATION]` stdout 流。
另外该日志走节点 stdout 而非 `optimization.log`，
首次统计因子数时 grep 错文件得到 0，险些误判为「因子未生效」。

---

## 85. S3b / S5a：有界窗口消灭了别名（2026-07-30）

### 85.1 S3b（松里程计旋转噪声）：FAIL，比 S3 更差

`odom_noise_rotation` 0.001 → 0.0145（§82 推导的诚实值）。

| | S3 | S3b |
|---|---|---|
| 后端修正中位 | 0.977 m | 2.715 m |
| 地板法向 p90 | 3.30° | 3.26° |
| \|dz\| p90 | **0.503 m** | 2.448 m |
| \|dz\|>0.5 m | 11/101 | 55/102 |

图动得更多，**但地板法向几乎没改善，重访一致性崩了**。
姿态因子只钉 roll/pitch，而全场只有 7 条闭环边 —— **长程相对几何无人约束**，
图在满足姿态因子的同时把形状扭了。已退回 0.001。

规律至此完整：**里程计紧则图不动、漂移留着；松则图走形。缺的一环始终是闭环。**

### 85.2 修正一条我此前过度概括的原则

§80.7 我写过「先验只有提名权，没有否决权」。**过度概括。** 正确表述：

> **窗口必须由实际累积不确定度定尺寸。**

150 m 根本不是界；30 m 在一张漂 2.5 m 的图上也不是界 ——
它松到能放进 30 m 外的走廊别名，这正是「描述子无界搜索 + 先验有界否决」
两个模型缺点相加的那个失败。但在漂移已压到 p90 0.503 m 的图上，
**3 m 是真正的界**：真闭环全在内，而本走廊每 5–30 m 重复的别名全在外。

### 85.3 S5a（`loop_max_range` 150 → 3.0）：别名被消灭，机制验证成立

| | 19 条（无界） | 4 条（3 m 有界） |
|---|---|---|
| 成对回路闭合误差中位 | **14.766 m** | **0.887 m** |
| 互相一致的对数 | 1 / 171 | 3 / 6 |
| 最大互相一致集合 | 2 条（修正 0.02–0.04 m） | **3 / 4 条** |
| 各闭环提出的修正 | 26.4 / 15.4 / 13.6 … | 1.36 / 0.29 / 0.15 / 0.04 m |

**这是质变。** 回路闭合误差降了一个半数量级，且一致的那几条
**提出的是真实修正**（1.36 m），不再是「不用修」的平凡近邻对。

`map_gate`：`|dz| p90 = 0.4989`、`>0.5 m` 为 **10/101** —— 两项均为目前最好；
几何未折叠（0.0%）。**但闭环仅 4 条，总判定仍 FAIL。**

### 85.4 下一个瓶颈已定位

几何上 3 m 内、id 间隔 >20 的关键帧配对有 **362** 个，实际只成边 4 条。
`selectBestPerQuery` **每 query 只留一条**（§81.5 已记录），
且按 fitness / referee energy 排序 —— 平凡近邻的 fitness 天然最好。

**窗口既已解决别名，「只留最好一条」的理由随之消失。** S5b 去掉该限制。
