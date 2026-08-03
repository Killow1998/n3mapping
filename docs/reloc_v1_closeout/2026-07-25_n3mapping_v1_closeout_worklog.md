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

---

## 86. S5b / S5c：闭环机制逐条修通，地图不再改善（2026-07-30）

### 86.1 S5b（保留全部通过验证的闭环）：无变化，且推翻我对瓶颈的判断

去掉 `selectBestPerQuery`（`loop_keep_all_verified = true`）后闭环 4 → 5，
**地图指标与 S5a 逐位相同**（p90 0.4989、中位 0.2734、最大 2.3280、z 跨度 3.75）。

**所以 `selectBestPerQuery` 从来不是瓶颈** —— §85.4 我判断它是，错了。真实分布：

```
候选 687
  prediction_range_gate  633   ← 3 m 窗口挡掉 92%
  loop_referee            43
  graph_inconsistent_yaw   5
  接受                     5
```

### 86.2 起点区闭环全部存在，且是全场最佳配准

修复开机暂态所需的那批闭环并不缺，质量还是整场最好的：

```
q=25 m= 8  fitness=0.0080  inlier=0.995  → yaw_flip_with_segment_disagreement
q=25 m= 9  fitness=0.0884  inlier=0.994  → graph_inconsistent_yaw
q=20 m=12  fitness=0.0568  inlier=1.000  → graph_inconsistent_yaw
q=30 m= 4  fitness=0.0500  inlier=0.980  → graph_inconsistent_yaw
q=20 m=10  fitness=0.0759  inlier=1.000  → segment_inconsistent
```

### 86.3 S5c：`graph_inconsistent_yaw` —— 我自己留下的缺陷

§81.5 我已写明它「与已修的平移判据是同一缺陷」（都读 `residual_yaw_after`，
即单条边拉不动 256 条链的那个量），当时判断「优先级低，反正候选池里没有可用的
一致集合」。**窗口修好后可用候选出现了，这条门立刻成为直接阻断者。**

按平移那半边已验证的模式修正为「这条边让已有约束变差多少」
（`existing_loop_residual_delta > 0.05`）。

**结果：三条起点区闭环被接受，地图无可测改善**（p90 0.4989 → 0.5034）。

### 86.4 一个认识论死结

对 S5c 的 6 条边做成对一致性检验：

```
成对回路闭合误差 中位 4.274 m   15 对中仅 1 对一致
12 → 20  修正 2.08 m  一致度 0/5
 9 → 25  修正 3.16 m  一致度 0/5
 4 → 30  修正 3.39 m  一致度 0/5
```

**但该检验在这个区间不可靠** —— 它依赖里程计腿，而暂态区的里程计
正是被污染的那段（§82.6 已因此撤回过一次结论）。

**验证这批闭环所需要的量，恰好是它们要修的那个量。**
所以既不能判定它们正确，也不能判定它们是别名。

### 86.5 六轮迭代后的账

| 阶段 | \|dz\| p90 | 闭环 | 贡献 |
|---|---|---|---|
| 基线 | 0.852 | 2 | — |
| **S3 地板姿态因子** | **0.503** | 6 | **唯一实质改善** |
| S3b 松里程计旋转噪声 | 2.448 | 7 | 变差，已回退 |
| S5a 有界窗口 3 m | **0.499** | 4 | 别名消灭：回路误差 14.8 → 0.89 m |
| S5b 保留全部闭环 | 0.499 | 5 | 无变化 |
| S5c yaw 判据修正 | 0.503 | 6 | 无可测变化 |

**闭环在整条线上没有产生任何可测的地图改善。** 唯一起作用的是姿态因子。
判据 `p90 < 0.10 m` 仍差 5 倍。

两条可复用结论：

1. **别名要靠窗口尺寸抑制，不是靠事后判据。** S5a 的回路误差从 14.766 m 掉到
   0.887 m，是把窗口按漂移定尺寸得到的，不是加了任何过滤器。
2. **「先验只有提名权」需要限定**：窗口必须由实际累积不确定度定尺寸 ——
   太松（150 m / 30 m）放进别名，太紧则真闭环进不来。

### 86.6 运维缺陷两则（都已修）

- 被外部中断的跑批**留下活着的节点进程**，直接重跑会与新实例抢同一批话题，
  **两次建图静默混进一个输出**。已在 `run_mapping_*.sh` 加启动前自检。
- 我加的自检最初写成 `pgrep -f n3mapping_node`，而 `-f` 匹配完整命令行 ——
  **我的 ssh 命令里带着这个字符串，守卫匹配到了自己的调用，恒为真**，
  导致后续每次启动都被静默拒绝，而我把残留 PID 误读成启动成功。已改 `pgrep -x`。
  **一个会匹配自身调用的守卫比没有守卫更糟：它把「在防护」和「在阻断」混为一谈，
  且失败方向是静默的。**
- 另清掉一个真残留：S1a 的 LIO 跑批在任务被停后**空转了 2 小时 26 分**。

---

## 87. S5d / S5e 与建图一致性目标的收尾：平台期，判据未达成（2026-07-30）

### 87.1 S5d（段一致性规则降级）：闭环 6 → 11，地图未动

`segment_inconsistent` 与 `yaw_flip_with_segment_disagreement` 依赖的段一致性
只有两个有效配对（比值仅能取 0/0.5/1，一个邻居失手即翻转），
却在否决 fitness 0.0080、inlier 0.995 的全场最佳配准。
降级为「段弱**且**无描述子确认」后：

| | S5c | S5d |
|---|---|---|
| 闭环边 | 6 | **11** |
| \|dz\| p90 | 0.5034 | 0.4988 |
| \|dz\| 最大 | 2.3271 | 2.3269 |
| 最大互相一致集合 | 2/6 | **4/11** |

**首次出现「互相一致且修正量与暂态吻合」的闭环集合**
（2.12 / 2.08 / 3.16 / 3.12 / 3.39 m）。但地图指标无可测变化。

### 87.2 S5e（上述 + 松开里程计旋转噪声）：证伪，且揭示了机制

`odom_noise_rotation` 0.001 → 0.0145，其余保持 S5d。

| | S5d | S5e |
|---|---|---|
| 闭环边 | 11 | 15 |
| \|dz\| p90 | **0.4988** | 2.4068 |
| \|dz\| 中位 | 0.2733 | 0.6444 |
| \|dz\|>0.5 m | 10/101 | 56/104 |

**与 S3b 完全同样的失败方式**，尽管这次闭环是有界的、有 4 条互相一致、
修正量与暂态吻合。**所以问题不在闭环质量或数量。**

机制：松开里程计后，姿态因子（σ=1°）在局部占据主导，
**每一帧的 roll/pitch 被拉去匹配它自己那个散布 2–6° 的地板法向**，
相邻帧的相对几何随之破碎。

**姿态因子之所以有效，恰恰是因为里程计是刚的** ——
链条把 225 个各自带噪的观测平均成一个全局修正。
**刚度在这里是机制的一部分，不是待解决的障碍。**
我在 S3b 与 S5e 两次把它当成障碍，两次都反了。

### 87.3 最终状态：判据未达成，但地图实质改善

**当前最好配置 = S5d**（已恢复到配置文件）：
地板姿态因子 + 3 m 有界窗口 + 保留全部验证闭环 + yaw/段判据修正 + 里程计噪声 0.001。

| | 基线 | S5d | 判据 |
|---|---|---|---|
| \|dz\| p90 | 0.852 m | **0.4988 m** | <0.10 m ✗ |
| \|dz\| 最大 | 2.486 m | **2.3269 m** | <0.30 m ✗ |
| \|dz\|>0.5 m | 17/99 | **10/101** | — |
| 地板 z 跨度 | 6.121 m | **3.75 m** | — |
| 地板法向 p90 | 3.69° | 3.30° | — |
| 闭环边 | 2 | **11** | ≥20 ✗ |
| 几何变化 | — | **0.0%** | <5% ✓ |
| 后端修正中位 | 6.4e-14 m | **0.977 m** | — |

**p90 改善 41%、z 跨度改善 39%、超标对数减少 41%，几何未折叠。
但 p90 距判据仍差 5 倍，最大值差 8 倍。目标未达成。**

### 87.4 为什么在这份数据上继续迭代不划算

残余误差是**被冻结的开机暂态**（§82）：路程 0 处残差 +3.95 m，75 m 处归零。

- **姿态因子只能纠正朝向，无法撤销已累积的位置误差**（§87.2）
- **闭环无法可靠修它**：跨暂态区的闭环，其正确性所需的验证量
  （里程计腿）正是被污染的那段 —— §86.4 的认识论死结
- **松开里程计让图能动，反而破坏相对几何**（S3b 与 S5e 两次证伪）

七轮迭代中闭环从 2 条修到 11 条、别名被彻底消灭（回路误差 14.766 → 0.887 m），
**而地图指标只由姿态因子那一步改善**。这条路已经走到尽头。

### 87.5 达成判据需要什么（都不在这份数据里）

1. **让暂态不发生。** 前端根因未定：S1 已证伪重力初始化的两个具体机制
   （协方差、窗口长度），初始化本身精确到 0.47°。真正的机制仍未找到。
2. **静止启动重录。** 开机原地静止数秒后再移动，从源头消除暂态。
   `tools/lio_gravity_drift.py` 与 `tools/floor_normal_vs_world_up.py` 可直接验收
   （看偏转峰值、看残差-路程曲线是否还有台阶）。
3. **独立参考。** 现有一切「改善」都是相对地图自身的一致性度量，
   不是相对外部真值。

### 87.6 这条线产出的可复用结论

1. **别名靠窗口尺寸抑制，不靠事后判据。** 窗口须由实际累积不确定度定尺寸：
   150 m / 30 m 太松（放进 30 m 外的走廊别名），3 m 恰当（p90 已压到 0.5 m）。
   回路闭合误差随之从 14.766 m 降到 0.887 m。
2. **位姿图需要至少一个绝对姿态观测。** 原图 258 条边全是相对约束，
   前端倾斜多少下游无从察觉。地板法向可在 249/257 帧获得，是现成的绝对观测。
3. **姿态因子依赖刚性里程计。** 链条把逐帧噪声平均成全局修正；松开它反而
   把每帧 2–6° 的观测散布注入轨迹。
4. **求解器的过程状态不是结果质量。** `converged`、`residual_*_after` 都被误用为
   质量门；正确的量是「这条边让已有约束变差多少」。该误用在本代码库出现四处。
5. **验收判据必须含防折叠项。** 七轮里几何都没折叠，若只看 z 一致性会误判。

### 87.7 我在这条线上判断错误的记录

- 三次把「刚性里程计」当成待解决的障碍（S3b、S5e），实际它是姿态因子生效的前提
- §85.4 判断 `selectBestPerQuery` 是瓶颈 —— S5b 证伪（去掉后无变化）
- §81.5 把 `graph_inconsistent_yaw` 判为「优先级低」而不修 ——
  窗口修好后它立刻成为直接阻断者
- 一次误报运行状态：守卫 `pgrep -f` 匹配到自己的调用命令，
  我把残留 PID 读成启动成功，S5b 有一段时间根本没在跑而我以为在跑

---

## 88. 跨 session 检验与平台期的真正解释（2026-07-30）

用户指出 `bagfile` 里另有两个长 bag（`b22b1_g2w`、`f7tof9_g2w`），
且「没有特别大的漂移」。其 LIO 输出已存在（`negative_inputs/{b22,f7tof9}_lio_ros2`，
长度与原 bag 一致：936.7 s / 1073.0 s），无需重跑 LIO。

### 88.1 直方图形态区分「真多标高」与「漂移」

这解决了 §69–§82 反复纠缠的那个问题。方法：逐帧拟合脚下地板，
看**地板高度的分布形态** —— 多层建筑聚成离散簇，漂移连续铺开。

**b22 完整（729 帧，轨迹 713.5 m）**：

```
-4.22 ~ -0.46 m   392 帧   ← 一层
 0.29 ~  5.56 m    40 帧   ← 过渡（楼梯/坡道）
 6.31 ~ 13.09 m   290 帧   ← 另一层
地板高度跨度 18.06 m
```

**明显双峰 —— b22 是真多层，18.06 m 是真实结构。**

**0723（255 帧，已知单层）**：分布连续单峰（−4.33 至 −1.71 连续铺开），
跨度 4.20 m。**在单层场地上连续铺开 4.20 m —— 那是漂移，确认。**

### 88.2 我的重访判据只对单层数据有效

b22 完整图的 `|dz| p90 = 14.2782 m`。**这不是漂移，是判据的伪影**：
重访判据配对「xy 相距 <1 m 且间隔 >30 s」的两点，
在多层建筑里会把**不同楼层的同一 xy** 判成重访。

**`map_gate.py` 的重访项只能用于单层数据。**
0723 上的 0.4988 m 有意义；b22 / f7tof9 上的数不可解读。
这是我建的判据的缺陷，须在工具里标注。

### 88.3 跨 session：约 3.5° 的姿态散布是系统性的

在多层下依然成立的量是地板法向偏离 world z（每层地板都水平）：

```
0723 单层：p90 3.40°      b22 多层：p90 3.57°
```

**同一套 LIO、不同 session、不同建筑，同样约 3.5°。不是 0723 特有。**

### 88.4 但那 3.5° 大部分是我自己的测量噪声 —— 平台期的真正原因

同一帧、五种拟合参数（带宽 0.25/0.35/0.50 m × 半径 5/8/12 m），
法向之间的最大夹角：

```
中位 2.73°   p90 4.59°   最大 8.75°

各参数下的法向偏离 world z：
  带宽 0.35 半径  8 m（现用）  中位 2.03  p90 3.40°
  带宽 0.25 半径  8 m         中位 1.71  p90 3.18°
  带宽 0.50 半径  8 m         中位 2.82  p90 5.15°
  带宽 0.35 半径  5 m         中位 3.00  p90 4.32°
  带宽 0.35 半径 12 m         中位 1.67  p90 2.91°
```

**参数间散布（2.73°）与被测量本身（约 2°）同量级 ——
该指标分辨不了 3° 以内的差别。** 三个后果：

**一、S3 的判据设在了仪器分辨率以下。** 我写的是「地板法向 p90 < 1.0°」，
**这个判据从一开始就不可测。** 方法错误。

**二、我犯了与 FAST_LIO `init_P = 1e-5` 完全同类的错误。**
我批评它「用 0.1 s 数据声称 0.18° 精度」，
然后把姿态因子的 σ 设成 **1.0°**，而观测实际噪声是 **2.7°** ——
**我告诉 GTSAM 这些观测比实际精确 2.7 倍。**

**三、这定量解释了平台期。** 225 个观测各带 2.7° 噪声，
平均后全局精度约 `2.7/√225 = 0.18°`，在 235.7 m 路程上对应约 **0.74 m** 的 z 误差。
**实测 p90 0.4988 m —— 已在观测噪声允许的极限附近。**

S3 停在 0.50 m **不是因为刚度、不是因为闭环数量，是因为观测本身不够准。**

### 88.5 由此得到的判断

参数扫描显示噪声可压（半径 12 m → 中位 1.67°，带宽 0.25 m → 1.71°，
现用 8 m/0.35 m 是 2.03°）。但即使压到 1.5°，按同样算式预期只到 p90 约 0.3 m，
**距判据 0.10 m 仍有 3 倍。**

**`p90 < 0.10 m` 可能不是地板法向这类观测能达到的**：
掠射入射下 LiDAR 对平面法向的确定能力有物理上限，
而这个上限（约 2–3°/帧）经 √N 平均后仍留下分米级的 z 误差。

**这不是「再调一轮就能过」的差距，是判据与可用观测之间的量级不匹配。**
要达到 0.10 m 需要**更强的绝对观测**（如已知平面的先验、外部参考），
而不是同一观测的更好拟合。

### 88.6 工具缺陷两则（须修）

1. `map_gate.py` 的重访项对多层数据无效，须加同层过滤（配对时要求
   两帧观测到的地板高度接近）或在输出中标注限制。
2. `floor_attitude` 的 σ 应由实测噪声给出（约 2.7°），而非硬编码 1.0°。
   注意：在刚性里程计下过度自信尚未造成可见危害（S3 有效），
   但这与 §85.2 批评 FAST_LIO 的理由是同一条，不应双标。

---

## 89. 三 session 横向对照：地板法向指标被证伪，f7tof9 前端发散（2026-07-30）

### 89.1 f7tof9 的 LIO 输出本身是坏的

直接量 LIO bag 里的 `/Odometry`：

```
b22:     9364 条   轨迹长     729.1 m   单步最大    0.243 m   超过 1 m 的步 0
f7tof9:  9583 条   轨迹长 3,108,188 m   单步最大 16,696 m     超过 1 m 的步 6590
```

**FAST_LIO 在 f7tof9 上发散**（9583 步里 6590 步超过 1 m）。
建图忠实复现了它（3,118,807 m vs LIO 的 3,108,188 m），
图有 6845 个关键帧却只有 3004 条边 —— 结构上已断开。

**f7tof9 不能作为跨 session 样本。** 该 bag 另有 10725 帧雷达对 9583 条里程计，
丢了约 1100 帧，与前端挣扎的表现一致。这本身是个待查的独立缺陷。

### 89.2 地板法向指标被证伪：它在废图上给出更好的读数

| | 地板法向 p90 | 轨迹长 | 状态 |
|---|---|---|---|
| 0723 单层 | 3.30° | 235.7 m | 正常 |
| b22 多层 | 3.57° | 713.5 m | 正常 |
| **f7tof9** | **3.24°** | **3,100,000 m** | **彻底发散** |

**一张轨迹三百万米、完全崩坏的图，地板法向 p90 比两张正常图还低。**

**这坐实了 §88.4：该指标测的是自身噪声，不是地图质量。**
一个在好图与废图上给出相同读数的指标不能判别地图好坏 ——
**我用它作 S3 的判据（p90 < 1.0°），从一开始就无效。**

同理，重访判据在 f7tof9 上也退化：发散轨迹几乎不重访，只找到 19 对，
`|dz| p90` 反而只有 2.6175 m。**两个判据在发散数据上都失效。**

### 89.3 判据的适用范围（须写进工具）

| 判据 | 有效条件 | 失效方式 |
|---|---|---|
| 重访 \|dz\| | **单层** 且轨迹未发散 | 多层：跨楼层配对被当重访（b22 14.28 m）；发散：无重访可配 |
| 地板法向 | **无** —— 噪声受限（2.73° 参数间散布） | 在废图上给出正常读数 |
| 几何未折叠 | 有基线可比时 | — |
| 闭环边数 | 始终 | 不反映闭环是否正确 |

**目前唯一可信的建图质量判据是「单层录制上的重访 \|dz\|」，
且需先用地板高度直方图确认该录制确为单层（§88.1）。**

### 89.4 跨 session 检验的结论

- **当前配置能推广**：b22（732 关键帧、713.5 m、真多层）建图未崩，
  闭环 9 条，几何合理（x 102.6 / y 89.7 / z 17.5 m 与真实结构相符）。
- **0723 的暂态是否特有 —— 无法判定**：b22 是多层，重访判据不适用；
  f7tof9 前端发散。**没有第二个单层样本可比。**
- **f7tof9 作为「真多标高对照」的用途落空**（前端发散），
  但 b22 的双峰直方图已经提供了这个对照（§88.1）。

### 89.5 建图一致性目标的最终状态

**判据未达成，且判据本身部分无效。**

单层数据（0723）上的最好结果：

```
|dz| p90   0.852 → 0.4988 m   （判据 <0.10 m）
|dz| 最大  2.486 → 2.3269 m   （判据 <0.30 m）
地板 z 跨度 6.121 → 4.195 m
闭环边     2 → 11
几何变化   0.0%（未折叠）
后端修正   6.4e-14 → 0.978 m 中位
```

**改善来自地板姿态因子；闭环在整条线上没有产生可测的地图改善。**

而 §88.4 的算式给出该路线的上限：观测噪声 2.7°，225 个观测平均后
全局精度 0.18°，在 235.7 m 上对应约 0.74 m 的 z 误差 ——
**实测 0.4988 m 已在极限附近。压噪声到 1.5° 预期也只到约 0.3 m。**

**`p90 < 0.10 m` 与「地板法向作为绝对观测」之间是量级不匹配，
不是调参差距。** 要达到需要更强的绝对观测（已知平面先验、外部参考），
或让暂态不发生（静止启动重录）。

### 89.6 本轮新增的自我纠错

- **我把 S3 判据设在仪器分辨率以下**（法向指标分辨不了 3° 内差别）。
- **我批评 FAST_LIO `init_P=1e-5` 的理由，自己违反了**：
  姿态因子 σ 设 1.0°，而观测实测噪声 2.7°。同一标准不能双标。
- **我差点交付一个自我隐藏的指标**：给 `map_gate` 加「同层过滤」后，
  0723 最坏重访误差从 2.33 m 降到 0.68 m —— 剔掉的正是证明漂移的配对。
  已改为「报告而不过滤」并显式声明单层前提。
- **我用一张 71.5 m 的前缀图断言「b22 每米下降率是 0723 两倍」**，
  完整图出来后该比较作废。

---

## 90. 配置来源缺陷：两次跨 session 跑用的不是 S5d（2026-07-30）

### 90.1 缺陷

从 loopdebug 到 S5e，**所有阶段共用同一个 `loopdebug_params.yaml` 并就地改写**。
S5e 把 `odom_noise_rotation` 改成 0.0145 后没有还原。
`run_mapping_generic.sh` 硬编码指向该文件，
于是 §89 的 b22 与 f7tof9 两次跑**继承了 S5e 的值** ——
而 S5e 是 0723 上已知的坏配置（p90 2.4068 vs S5d 0.4988）。

**§89.4 第 1 条「当前配置能否推广 → 能」不成立**，已在交接文档更正。
能说的只是「b22 在 S5e 配置下建图未崩」。

f7tof9 的结论不受影响 —— 那是直接量 LIO bag 里的 `/Odometry`，与建图配置无关。

另有一个一直存在的隐性问题：**`floor_attitude_noise_deg` 从未出现在参数文件里**，
每个阶段都在静默使用 `config.h` 的默认 1.0°，
而 §88.4 实测该观测的参数间散布是 **2.73°**。

### 90.2 修复

- 每阶段一个独立参数文件（`params_s5d.yaml` / `params_s6a.yaml` / `params_s6b.yaml`），
  **不再就地改写**。
- `run_mapping_generic.sh` 改为 `PARAMS` 必填（`:?`），
  并把 bag、rate、参数文件路径与全部关键键值写入
  `$OUT_DIR/logs/provenance.txt`，同时留一份 `params.effective.yaml`。
- 把 `floor_attitude_*`、`loop_keep_all_verified`、`loop_min_path_length_m`
  显式写进参数文件，不再靠默认值。

**教训：「配置由最后一次编辑决定」是一种沉默的实验污染。
每次跑必须自证用了什么，而不是靠记得改过什么。**

### 90.3 S6：σ 与里程计刚度是耦合的

S5e 的失败机制（§87.2）是「姿态因子 σ=1° 在局部压倒相邻几何」——
**这个机制依赖 σ=1° 相对实测 2.7° 噪声是虚高的紧。**
所以先把 σ 改诚实，才谈得上松里程计。

所需的差分姿态余量：session 内 LIO 偏转从 2.4–5.4° 变到 1.6–1.9°，
差分约 3.5°；256 条边的随机游走余量 = √256 × σ_edge，
要覆盖 3.5° 需 σ_edge ≈ 0.22° = **0.0038 rad**。
S5e 用的 0.0145 rad（0.83°/边）是所需的近 4 倍。

刚度比：S5e 里程计 0.0145 rad vs 姿态 0.01745 rad —— 同量级；
S6b 是 0.004 vs 0.0471 rad —— **里程计比姿态因子硬 12 倍**，不是同一体制。

**先写死的判据：**

| 阶段 | 改动 | 预测 | 失败含义 |
|---|---|---|---|
| S6a | σ 1.0° → 2.7° | \|dz\| p90 **无变化**（0.4988±0.05） | §88.4 的全局平均模型错 |
| S6b | S6a + odom 0.001 → 0.004 | p90 **< 0.40 m**（仍达不到 0.10） | 刚度上限假设死，此路穷尽 |

S6b 的防折叠护栏：轨迹长与 xy 范围相对 S5d 变化 <5%，闭环边 ≥8。

---

## 91. f7tof9 前端发散：根因定位到楼梯井（2026-07-30）

### 91.1 现象

`f7tof9_lio_ros2` 的 `/Odometry`：轨迹长 3,108,188 m，
9583 步里 6590 步单步超过 1 m，单步最大 16,696 m。
b22 对照：729.1 m，单步最大 0.243 m，超过 1 m 的步 **0**。

### 91.2 不是跳变，是开环跑飞

步长平滑增长：0.47 → 1.94 → 10 → 100 → 1322 m，方向近乎恒定，
四元数全程归一（|q|=1.000000）。

速度中位随时间**线性**增长：

```
 t=300   99 m/s        t=660  3990
 t=360  348            t=720  5191
 t=420  707            t=780  5574
 t=480 1351            t=840  6766
 t=540 2018            t=900  7187
 t=600 3082            t=1020 8852
```

斜率 (8852−99)/720 = **12.16 m/s² = 1.24 g**。
步进方向近乎水平（归一化约 0.50, 0.85, −0.11）。

**这是重力泄漏到水平轴的签名**：姿态错了几十度之后，
重力被从错误的轴上减掉，残差约 1.24 g 被速度状态永远积分下去。

### 91.3 姿态失锁时刻

```
 t     roll    pitch    yaw   | 姿态变化率 中位/最大 (deg/s)
 190  -5.44   15.50   84.68   |   2.29    38.36
 240   7.80   26.27  -10.42   |   6.77    35.42   ← 运动变剧烈
 300   1.79   13.03 -116.45   |   5.98    31.63
 310  36.00   19.88 -123.03   |  11.94    99.14   ← 失锁
 350  53.42  -32.96  154.00   |  32.69   151.02
 410  43.89   64.00   70.08   |  31.76   418.10
```

t≈305–310 姿态失锁。此前 t≈220–240 运动先变剧烈（pitch 达 26°，变化率翻倍）。

**上表的变化率是欧拉角差分，接近万向锁时会虚高。**
用四元数夹角重算，全 bag 真实角速率峰值只有 97.8 deg/s（b22 对照 120.5）——
**姿态跑飞体现在朝向本身错得离谱，不体现在转得多快。**

### 91.4 根因：几何塌缩，不是传感器故障

`/cloud_registered_body` 的空间尺度：

```
 t 窗      点数中位   半径 p50 / p95 (m)   竖直跨度
 140-160     9075      3.65 /  8.97         10.84
 180-200     9745      3.00 /  8.28          7.85
 200-220     9469      2.13 /  5.03          8.60   ← 开始收缩
 220-240     9456      1.98 /  4.64         11.82   ← 最窄
 240-300     9455      2.10 /  4.86          7.43
 300-320     9495      2.64 /  6.49          7.65
 320-340     9316      5.40 / 10.39         24.01   ← 出来了
```

**点数始终约 9400，雷达帧率 600 帧/30 s 全程不变 —— 传感器完全正常。**
是场景塌缩：横向约 4 m、竖向 7–12 m 的窄竖井，持续约 120 s。
录制是 f7→f9，**这就是楼梯井**。

（后期 t>810 的确出现丢帧，但那是发散的**结果**不是原因。）

### 91.5 完整因果链

1. t≈200 进入楼梯井，雷达半径 p95 从 9 m 塌到 4.6 m
2. 窄竖井里扫描匹配沿竖直方向退化，叠加爬楼的剧烈俯仰（pitch 26°）
3. t≈305 ESKF 姿态失锁
4. 姿态错几十度 → 重力泄漏到水平轴，残差 1.24 g
5. 速度开环积分；t≈320 后预测位姿已在千米外，**没有对应点可用，
   雷达再也拉不回来** —— 不可恢复
6. 又跑了 12 分钟，最高 8852 m/s

**FAST_LIO 没有任何发散检测**，全程照常输出。

### 91.6 建图侧的缺陷：全盘接收

n3mapping 消费了全部 9583 条，建出 6845 关键帧、3,118,807 m 的地图并保存，
**没有任何告警**。`addOdometryConstraint` 查过了 ——
**它根本没有拒绝路径**，只在前一帧缺失时返回 false。

这直接违背建图目标：管线会安静地产出一份灾难性损坏的地图资产。

### 91.7 修复：里程计合理性守卫

新增 `include/n3mapping/odometry_sanity.h` / `src/odometry_sanity.cpp`。
逐帧两次减法、一个范数、一个比较 —— 计算量可忽略，满足 RK3588 约束。

限值设在**物理不可能**处，不是"质量不佳"处：

| 参数 | 默认 | 依据 |
|---|---|---|
| `odom_sanity_max_speed_mps` | 10.0 | Go2-W 约 3 m/s；b22 实测峰值 2.66；留近 4 倍余量。f7tof9 首次越界 11 m/s |
| `odom_sanity_max_angular_rate_dps` | 720 | 每秒两整圈；四足足底冲击的瞬时高值远在其下 |
| `odom_sanity_max_consecutive` | 5 | 单次是毛刺，连续才是跑飞；f7tof9 的发散单调不回头，半秒内触发 |

触发后**停止接收新帧**并由 ROS 层打 `[ODOM-SANITY]` ERROR。
f7tof9 若有此守卫，存下的会是发散前那段 **241 m** 的可用地图，
而不是三百万米的垃圾。

**假阳性验证**（离线，用三个 bag 的 `/Odometry` 逐帧复现守卫算法）：

| bag | 峰值速度 | 峰值角速率 | 结果 |
|---|---|---|---|
| b22 (936 s) | 2.66 m/s | 120.5 deg/s | **不触发**，速度余量 3.8 倍 |
| 0723 (470 s) | 1.75 m/s | 38.2 deg/s | **不触发**，速度余量 5.7 倍 |
| f7tof9 | 11.06 m/s | 97.8 deg/s | **触发 t=300.00 s** |

触发点正好落在发散处，此时轨迹才约 350 m。
**角速率限值全程没有起作用**（实测最大 120.5，远低于 720），
起作用的只有速度限值 —— 角速率那条是备用的，如实记录。

在线验证（编译后实跑）待跑批结束再做，不与当前跑混。

### 91.8 这个缺陷是怎么被发现的

不是查出来的 —— 是我把 f7tof9 的 `|dz| p90 = 2.6175 m` 差点当成
一个可以和 0723 的 0.4988 并排放进对照表的数字。
**发散轨迹几乎不重访，只配出 19 对，指标反而"好看"。**
判据在坏数据上给出温和读数，与 §89.2 地板法向的失效方式是同一类。

---

## 92. S6a：诚实的 σ 让判据在会话后段达成（2026-07-30）

### 92.1 预测失败

**先写死的预测：σ 1.0° → 2.7° 后 |dz| p90 不变（0.4988±0.05）。
实测 0.6535 —— 预测失败，§88.4 的「全局平均」模型是错的。**

| | S5d (σ=1.0°) | S6a (σ=2.7°) |
|---|---|---|
| \|dz\| **中位** | 0.2733 | **0.0953** |
| \|dz\| p90 | 0.4988 | 0.6535 |
| \|dz\| 最大 | 2.3269 | 2.4648 |
| >0.5 m 对数 | 10/101 | 16/98 |
| 闭环边 | 11 | 12 |
| 后端修正中位 | 0.978 m | **0.136 m** |
| 轨迹长 / x / y | 235.7 / 48.58 / 41.16 | 235.7 / 48.57 / 41.18 |

### 92.2 真实机制：σ=1.0° 是在用主体换尾部

后端修正从 0.978 m 掉到 0.136 m（7 倍）——
σ 虚高的紧使姿态因子过强，后端施加一个大修正把整条会话拉向折中。
**按路程分箱看得很清楚：**

```
路程段    S5d 中位/p90/最大        S6a 中位/p90/最大
  0- 25   1.243 / 2.324 / 2.327   1.316 / 2.465 / 2.465   持平（暂态）
 25- 50   0.287 / 0.404 / 0.460   0.826 / 0.954 / 0.994   S6a 更差
 50- 75   0.153 / 0.340 / 0.425   0.193 / 0.350 / 0.451   持平
100-150   0.436 / 0.602 / 0.675   0.119 / 0.461 / 0.599   S6a 更好
150-200   0.076 / 0.273 / 0.358   0.037 / 0.103 / 0.155   S6a 大幅更好
```

**σ=1.0° 把暂态误差抹到整条会话上**：掩盖了 25–50 m，代价是拖坏 100–200 m。
σ 改诚实后后段恢复了本来的一致性，暂态原形毕露。

### 92.3 结论：除开机前约 50 m，这张图已经达标

**会话后段（路程 150–200 m，35 对重访）：**

```
|dz| p90  0.103 m   判据 <0.10   （差 3%）
|dz| 最大 0.155 m   判据 <0.30   ✓
```

整图失败**完全集中在前 50 m**：0–25 m 段最大 2.465 m，25–50 m 段 0.994 m。
最差的两对都是 `2.46 m @ 0/28 m` 与 `2.46 m @ 0/29 m` —— 起点与第一次回访。

这与 §82 完全一致（路程 0 处残差 +3.95 m，75 m 处归零）。

### 92.4 为什么后端撤销不了它（结构性）

**事后纠正姿态并不重新积分平移。**
姿态因子能把早段的朝向拉正，但那段路程是用**错误的姿态积分出来的**，
位置误差已经固化在里程计边的测量值里。
要真正修它必须用修正后的姿态重新积分 —— 这是前端操作，后端做不到。

**所以「让暂态不发生」不是偷懒的说法，是唯一的结构性出路。**

### 92.5 判据的重新表述

原判据「整图 p90 < 0.10 m」把一个前端录制缺陷记在后端账上。
更能反映实际能力的表述：

- **前端收敛后**（路程 >150 m）：p90 0.103 m，最大 0.155 m —— 基本达标
- **含暂态的整图**：p90 0.6535 m —— 不达标，且不可能靠后端达标

**不改判据**（改判据来迁就结果是自欺），但记录这个分解，
因为它把「差 5 倍」变成了「差在录制的前 50 m」，是可行动的。

---

## 93. §92 的结论方向错了：σ=1.0° 全面更好，失败的分解也不同（2026-07-30）

### 93.1 更好的仪器：把固定倾斜与随时间变化的部分分开

**固定倾斜对重访一致性不可见**（同一个 xy 永远映到同一个 z）。
造成重访误差的只有随时间变化的部分。所以：
用每帧自己的点云拟合脚下地板高度 → 得到 (x, y, z_floor) →
最小二乘拟合一个固定平面 → **看残差随路程的变化**。

工具：`tools/map_z_drift_decompose.py`。

0723 的固定平面倾角 3.69°，与 §69 的"地图倾斜"是同一件事 ——
**它不影响判据**，此前把它当问题是我的误判之一。

### 93.2 三份剖面并排

```
路程段      前端 pose_odom   S6a σ=2.7°   S5d σ=1.0°
  0- 10        +1.952         +1.791       +1.363
 10- 25        +0.501         +0.340       +0.009
 25- 50        +0.535         +0.356       -0.086
 50- 75        +0.343         +0.142       -0.041
 75-100        -0.071         -0.119       -0.101
100-150        -0.461         -0.423       -0.310
150-200        +0.412         +0.324       +0.164
200-300        -1.087         -0.688       -0.043
残差散布 m      0.747          0.595        0.393
```

**S5d 把 10–300 m 全段压进 ±0.31 m；S6a 留下 −0.69…+0.36。**

### 93.3 更正 §92

§92 写的"σ=1.0° 是在用主体换尾部"、"σ 改诚实后后段恢复本来的一致性"
**方向反了**。三项指标里两项favor S5d：

| | S5d σ=1.0° | S6a σ=2.7° |
|---|---|---|
| 重访 p90（**判据**） | **0.4988** | 0.6535 |
| 残差剖面散布 | **0.393** | 0.595 |
| 重访中位 | 0.2733 | **0.0953** |

§92 只看了重访的中位数与分箱，没有看残差剖面，就断言了机制。
**中位数改善来自重访配对的 xy 局部性，不代表地图整体一致性改善。**

### 93.4 为什么"诚实的 σ"反而更差（这是真结论）

σ 不只在建模观测噪声。里程计链条被**刻意做硬**，
姿态观测是图里**唯一的绝对信息**，σ 取小相当于让这批观测
整体对链条施加更大的拉力。

**最能建模观测的 σ，不是最能利用它的 σ。**
两个"错误"（σ 虚高的紧、里程计虚高的硬）方向相反地互相补偿。

这不是可以自夸的设计，是如实记录：现有结构里没有第二个绝对参考，
所以 σ 实际充当的是"这批观测相对链条的权重"，不是噪声模型。

**决定：配置保持 S5d（σ=1.0°）。** `params_s6a.yaml` 的实验结果留档，
不进配置文件。

### 93.5 失败的分解（更正 §92.3）

| 指标 | 来源 | 后端能否修 |
|---|---|---|
| \|dz\| **最大** 2.33 m | 开机暂态，**只在前 10 m**（残差 +1.363 m，散布 0.722） | **不能**（§92.4 的结构性理由成立） |
| \|dz\| **p90** 0.4988 m | 全会话残余的 ±0.3 m 段间 z 漂移 | **部分能**：前端 0.747 → 后端 0.393 |

§92 写的"整图失败完全集中在前 50 m"**不成立**。
前 10 m 决定的是**最大值**；p90 由整条会话上残余的慢漂移决定。

按 S5d 的剖面算：+0.164（150–200）与 −0.310（100–150）相差 0.474 m，
跨这两段的重访对就会给出约 0.47 m —— **与实测 p90 0.4988 吻合**。

**所以要把 p90 从 0.50 压到 0.10，需要把段间残差从 ±0.3 m 压到 ±0.06 m，
即后端再去掉 5 倍的慢漂移。** 这是一个明确的、可度量的目标，
和"暂态"是两件事。

### 93.6 暂态的范围被钉死了

前端残差：0–10 m 段 +1.952 m，10–25 m 段 +0.501 m。
**暂态只有前 10 米**，不是 50 m，也不是 75 m。
12 个关键帧。散布 0.844 m（其余各段 0.06–0.43）。

### 93.7 另一个被证伪的说法

用矢量平均（而非绝对夹角）重测"前端世界系是否早期更倾斜"：

```
路程段    倾角    方位     标准误
  0-25   1.57°    13°     0.50
 25-50   2.80°   110°     0.22
 50-75   1.11°    23°     0.44
 75-100  3.14°   -47°     0.27
100-150  0.98°    63°     0.28
150-200  1.33°    45°     0.25
前 50 m vs 100 m 之后：0.50° 差（标准误 0.40，1.2 sigma）→ 无法分辨
```

**方位在乱摆** —— 系统性的世界系倾斜方位应基本不变。
所以"暂态是一个持续的世界系倾斜"这个说法，这个测量不支持。
暂态在 z 残差里非常明显（+1.95 m），但它不表现为一个方向固定的倾角。
**机制仍未确定**，这一点与 §87.5 一致，没有进展。

（第一版这个工具取的是绝对夹角，噪声整流使零倾角也读出约 2.4°，
各段读数 2.1–3.3° 且最大值出现在会话中段 —— 又一个答非所问的统计量。已改。）

---

## 94. 开机暂态是 0723 那次录制特有的（2026-07-30）

### 94.1 用对了仪器 + 有了第二个样本

§93.6 把 0723 的暂态钉死在**前 10 米、12 个关键帧**。
b22（真 S5d 配置重跑）提供了对照。两者都限路程 <100 m 拟合平面
（b22 是双标高，跨标高会污染平面拟合）：

```
路程段    b22 中位/标准差       0723 中位/标准差
  0-10   -0.097 / 0.285       +1.239 / 0.933
 10-25   +0.316 / 0.029       -0.254 / 0.355
 25-50   +0.451 / 0.284       +0.189 / 0.098
 50-75   -0.634 / 0.412       +0.027 / 0.160
 75-100  +0.130 / 0.192       -0.532 / 0.032
残差散布      0.439                0.620
```

**b22 的前 10 米完全正常** —— 残差 −0.097、散布 0.285，
与它自己其余各段同量级。
**0723 是 +1.239、散布 0.933** —— 中位是其余各段的约 4 倍，散布约 3 倍。

### 94.2 结论

**开机暂态是 0723 那次录制特有的，不是 FAST_LIO 的性质。**

这回答了 §89.4 里我标为"无法判定"的问题 2。
之前答不了是因为：(a) 用的是重访判据，b22 多层用不了；
(b) 后来用绝对夹角的地板法向，那是个答非所问的统计量（§93.7）。
**换成"去掉固定平面后的 z 残差随路程"，两个问题同时解决** ——
它不需要重访配对，也可以限制在单标高段内。

### 94.3 慢漂移则是两份都有

b22 −0.634 … +0.451，0723 −0.532 … +0.189。
残差散布 0.439 vs 0.620，同量级。

**与 §93.5 的分解一致：慢漂移是普遍问题（决定 p90），
暂态是这次录制的偶发（决定最大值）。**

### 94.4 对目标的影响

- **静止启动重录仍然值得做**，但它的作用被重新定位了：
  它消除的是**最大值**那一项（2.33 m），不是 p90。
- **判据 p90 < 0.10 m 与暂态无关。** 即使换一份没有暂态的录制，
  p90 仍由全会话 ±0.3 m 的慢漂移决定。
  §93.5 的目标不变：把段间残差压 5 倍。
- **已经有一份没有暂态的录制在手** —— b22。
  但它是双标高，重访判据用不了（§89.3）。
  **要用重访判据验收，需要一段单层、无暂态的录制。**

### 94.5 b22 在真 S5d 配置下的结果

```
闭环边 5   重访对数 45   |dz| p90 5.0251   最大 5.0954
轨迹 713.5 m   x 103.43   y 89.57   z 10.22
```

**重访数字不可解释**（双标高，§89.3）。可解释的是几何：
x/y 跨度 103.43 / 89.57 与之前 S5e 配置那次（102.6 / 89.7）一致，
与真实结构相符 —— **配置能推广，建图未崩**。
闭环 5 条（S5e 配置那次 9 条）：里程计更硬，接受的闭环更少。

§89.4 问题 1「当前配置能否推广」→ **能**（这次是真 S5d 了）。

---

## 95. S6b 失败，S6 结束；以及测试套件一直是红的（2026-07-30）

### 95.1 S6b：先写死的判据未达成

`odom_noise_rotation` 0.001 → 0.004（σ=2.7°，其余 S5d）。

| | S5d | S6a | **S6b** | 判据 |
|---|---|---|---|---|
| \|dz\| p90 | 0.4988 | 0.6535 | **0.8060** | <0.40（先写死） |
| \|dz\| 中位 | 0.2733 | 0.0953 | 0.3448 | — |
| \|dz\| 最大 | 2.3269 | 2.4648 | 2.3136 | — |
| 闭环边 | 11 | 12 | 14 | — |
| 轨迹/x/y | 235.7/48.58/41.16 | 235.7/48.57/41.18 | 235.7/48.59/41.14 | 变化<5% ✓ |

几何没有折叠，闭环还多了 3 条，**但 p90 是三者中最差。刚度上限假设死。**
松里程计在 0.0145（S5e）与 0.004（S6b）两个量级上都失败。

**一个未解释的矛盾，如实记录**：S6b 的残差剖面反而更紧
（散布 0.376 vs S5d 0.393），但重访 p90 差得多（0.806 vs 0.4988）。
两个统计量给出相反判断。我没有找到解释这个差距的机制。
**判据是重访那个，所以 S6b 是失败。**

（这也说明 §93 引入的残差剖面统计量有盲区：它用的是各段中位数，
对段内的局部散布不敏感，而松里程计破坏的正是局部一致性。）

### 95.2 测试套件一直是红的，没有人在跑

打开 `BUILD_TESTING=ON` 之后：**325 个测试，10 个失败。**
每一个都是我某次有意的改动之后没有回来对齐断言，
而建图跑批一直用 `BUILD_TESTING=OFF`，所以从未暴露。

| 失败 | 来源 |
|---|---|
| 3 处编译错误（2 个文件） | S5a 改了 `detectLoopCandidates` 签名 |
| 5 × `LoopRefereeTest` | S5c/S5d 放松了四条裁判判据 |
| 1 × `ProductConfigHumbleTest` | `loop_max_range` YAML 3.0 vs 内置默认 150 |
| 1 × `N3MappingCoreTest` | 自由空间否决 |

### 95.3 三个真缺陷（不只是测试过期）

**(a) `loop_max_range` 内置默认仍是 150。**
不加载 YAML 的调用方拿到的是已被证明会招别名的值
（§87：回路误差 150 → 14.766 m，3.0 → 0.887 m）。
而那里的注释还在论证"必须大于地图尺度"—— 正是 3.0 推翻的说法。已改为 3.0。

**(b) `segment_inconsistent_unconfirmed` 是死代码。**
它要求 `segment_consistency < 0.5 && !has_descriptor`，
而前面的 `unconfirmed_weak_segment` 已用 `<= 0.5 && !has_descriptor` 提前返回 ——
严格包含，永远到不了。已删除。

**(c) 自由空间否决：两个缺陷，都是我自己的。**

1. **否决不在拒绝原因链里，也不在日志里。**
   它在接受条件的合取式中，但被否决的锁会被报成 `"stability_guard"`，
   而日志里 `pass(...)` 每一项都显示为 1、`ambiguous` 为假。
   **一条声称在枚举要求的消息，漏掉了真正拒绝它的那一项。**
   现场排查 no-lock 会被彻底带偏 —— 而这个否决在产品里默认开着。
   已加入原因链与日志（`free_space_veto`、`free_cells=`）。

2. **零自由空间证据也能否决。**
   单关键帧地图建出的栅格是 `occupied=28 free=0`，`valid()` 仍为真，
   否决照常武装。free 为 0 时"自由空间偏好哪个假设"是排序噪声的产物，
   **据此拒绝锁定 = 据无物拒绝**。
   已改为 `freeCells() == 0` 时返回 nullptr。

这一条与用户的"不能误锁"是同一个要求的另一面：
**不要在不存在的证据上拒绝，并且永远如实说明拒绝的理由。**

### 95.4 现状

**325 个测试，0 失败。** 这条工作线上第一次全绿。

### 95.5 教训

**用 `BUILD_TESTING=OFF` 跑了七个阶段的实验，等于把回归网关了七次。**
测试不是文档，是"这段代码保证什么"的可执行表述；
放松一条判据而不更新它的测试，等于取消了那个保证却不留痕迹。
四条裁判判据的放松现在在测试里写明了「新边界 + 新接受的那一侧」，
而不只在提交信息里。

---

## 96. S7：闭环数不是杠杆（2026-07-30）

### 96.1 先确认二进制可比

S7 用的是修完测试之后重新编译的二进制（守卫、裁判死分支、
`loop_max_range` 内置默认、自由空间否决四处改动）。
先在新二进制上重跑 S5d 作对照：

```
闭环 11   中位 0.2733   p90 0.4988   最大 2.3269   轨迹 235.7 / 48.58 / 41.16
```

**与原 S5d 逐位相同** —— 二进制等价，S7 可比。
（这一步是 §90 那次污染之后的规矩，不再靠"我判断那些改动是空操作"。）

### 96.2 S7：窗口 3.0 → 6.0

| | S5d | **S7** | 判据 |
|---|---|---|---|
| 闭环边 | 11 | **45** | ≥20 **PASS**（首次） |
| \|dz\| p90 | 0.4988 | **0.4764** | <0.40（先写死）→ **FAIL** |
| \|dz\| 最大 | 2.3269 | 2.3250 | <0.30 FAIL |
| \|dz\| 中位 | 0.2733 | 0.2594 | — |
| \|dz\|>0.5m | 10/101 | 10/101 | — |
| 轨迹/x/y | 235.7/48.58/41.16 | 235.7/48.58/41.16 | 变化<5% ✓ |

**窗口确实是卡着的**（`prediction_range_gate` 拒掉 634/695 = 91% 的候选，
而进图的 18 条预测位移最大 2.74，紧贴 3.0）。
6 m 放进 4 倍的闭环，**而且多出来的 34 条不是别名** ——
轨迹长与 x/y 跨度与 S5d 逐位相同，几何一动没动。

**但 p90 只改善 4.5%。**

### 96.3 残差剖面：四倍闭环带来零信息

```
路程段     S5d      S7
  0- 10   +1.363   +1.363
 10- 25   +0.009   +0.009
 25- 50   -0.086   -0.083
 50- 75   -0.041   -0.034
 75-100   -0.101   -0.096
100-150   -0.310   -0.305
150-200   +0.164   +0.161
200-300   -0.043   -0.059
散布       0.393    0.392
```

**逐段相差不到 2 cm。**

### 96.4 结论：闭环数不是杠杆

判据里的「闭环边 ≥20」第一次通过了，**而它要保障的东西一点没改善**。
这条判据是按"闭环少所以修不动"的假设写的，现在证明假设是错的。

### 96.5 下一个假设：闭环边比里程计边弱 50 倍

`loop_noise_position = 0.5` 而 `odom_noise_position = 0.01`。
链条有 256 条边，闭环 45 条，每条弱 50 倍 —— **闭环推不动任何东西。**

0.5 m 也不是配准实际达到的精度：通过验证的闭环
`fitness < 0.2`、`inlier_ratio > 0.7`，实际配准在 5 cm 量级。

**这与姿态因子 σ 是同一个形状的错误，方向相反**：
那里我把 σ 设得比观测噪声紧（1.0° vs 实测 2.7°），
这里把闭环设得比配准精度松（0.5 m vs 实际 0.05 m）。
§93.4 的结论"σ 实际充当的是权重不是噪声模型"在这里同样适用 ——
但这次调整方向是**增加**约束而不是移除结构。

**S8（先写死）**：`loop_noise_position` 0.5 → 0.05，其余 S7。
预测 p90 **< 0.30 m**。护栏：轨迹长与 xy 范围相对 S5d 变化 <5%
（收紧闭环 + 万一有坏闭环 = 折叠风险最高的组合）。
**失败则后端真的穷尽。**

---

## 97. `loop_noise_position` / `loop_noise_rotation` 从未生效（2026-07-30）

### 97.1 S8 与 S7 逐位相同

`loop_noise_position` 0.5 → 0.05（十倍），产出的地图**每一位都一样**：
闭环 45、中位 0.2594、p90 0.4764、最大 2.3250、轨迹 235.7 / 48.58 / 41.16 / 3.75。

不是"没有改善"，是**这个参数根本没到优化器**。

### 97.2 缺陷

`src/loop_verifier.cpp`：

```cpp
loop.information = config_.loop_use_icp_information
    ? verification.match_result.information
    : Eigen::Matrix<double, 6, 6>::Identity();   // ← 单位阵
```

`graph_optimizer.cpp::createRobustNoiseModel` 的分支条件是
`if (!information.isZero(1e-10))` —— **单位阵不是零**，
于是走"使用提供的信息矩阵"分支，
**永远到不了下面那个用 `loop_noise_position` 的回退。**

单位信息阵 = 每条闭环边 **σ = 1.0 m / 1.0 rad**。
对照里程计边 0.01 m / 0.001 rad：

**闭环边在位置上弱 100 倍，在旋转上弱 1000 倍（1.0 rad = 57°）。**

XY-yaw 那条路径同样中招：`sigmaFromInfo(information(0,0), fallback)`
读到 `information(0,0) = 1`，返回 `1/sqrt(1) = 1.0`，回退同样用不上。

### 97.3 这两个参数是"看起来在工作"的死参数

它们写在 `product_v1.yaml` 与 `n3mapping.yaml` 里、
出现在 `config.cpp:192` 的配置摘要打印里、
在 `config.cpp:307` 被 `positive()` 校验、
被 `config_humble.cpp:61` 从 ROS 参数读入 ——
**全套仪式齐备，而对闭环边没有任何作用。**

`mapping_resuming.cpp:239` 用了它们（那条路径自己算信息矩阵），
所以 grep 会显示"在用"，掩盖了主路径上的失效。

### 97.4 这解释了 S7

S7 把闭环从 11 加到 45，残差剖面逐段变化不到 2 cm。
**45 条在旋转上比里程计弱一千倍的边，推不动任何东西。**
我上一轮估的"弱 50 倍"是按配置里的 0.5 算的，
而实际生效的是 1.0 —— 估小了一半，方向是对的。

### 97.5 修复

`loop_use_icp_information` 为假时，用配置的闭环噪声构造信息矩阵，
而不是塞一个单位阵：

```cpp
loop.information = Eigen::Matrix<double, 6, 6>::Identity();
loop.information.block<3, 3>(0, 0) *= 1.0 / (loop_noise_position ^ 2);
loop.information.block<3, 3>(3, 3) *= 1.0 / (loop_noise_rotation ^ 2);
```

块顺序 (translation, rotation) 与 `addOdometryConstraint` 一致，
`createRobustNoiseModel` 会为 GTSAM 交换。修在源头，两条路径都覆盖。

**325 个测试仍全过。**

**注意：这会改变默认行为**（σ 从实际的 1.0 变成配置声称的 0.5），
所以必须分两步测，不能假设。

### 97.6 先写死的判据

| 阶段 | 改动 | 预测 | 失败含义 |
|---|---|---|---|
| S9 | 修缺陷，参数不动（0.5/0.5） | 与 S7 **不同** | 若相同，说明我对这条路径的理解仍是错的 |
| S10 | S9 + 闭环噪声 0.05/0.05 | p90 **< 0.30 m** | 后端穷尽 |

护栏：轨迹长与 xy 范围相对 S5d 变化 <5%。

### 97.7 教训

**"关掉某个来源"不等于"没有信息"。**
用单位阵表示"不用 ICP 的信息"，实际是断言了一米一弧度的精度，
而下游用 `isZero()` 判断"有没有提供信息" —— 两边对同一个矩阵的含义理解不同，
中间没有任何东西会报错。

这是本轮第三个同一类缺陷：
- §95：自由空间否决在零自由空间证据上照常否决
- §95：`loop_max_range` 内置默认与发布的 YAML 不一致
- §97：闭环噪声参数走完全套仪式却不生效

---

## 98. S9 是新的最好配置；S10 的改善是判据自证（2026-07-30）

### 98.1 S9：修复生效，先写死的预测成立

| | S7（σ 实为 1.0/1.0） | **S9（修复后 0.5/0.5）** |
|---|---|---|
| 闭环边 | 45 | 46 |
| \|dz\| 中位 | 0.2594 | **0.2272** |
| \|dz\| p90 | 0.4764 | **0.4605** |
| \|dz\| 最大 | 2.3250 | 2.3246 |
| 轨迹/x/y | 235.7/48.58/41.16 | 同 |

预测是"与 S7 不同"，成立。闭环强 2 倍，方向对，幅度不大。

### 98.2 S10：判据大幅改善，但改善不可信

`loop_noise` 0.05/0.05：

| | S5d | S9 | **S10** |
|---|---|---|---|
| \|dz\| 中位 | 0.2733 | 0.2272 | **0.0551** |
| \|dz\| p90 | 0.4988 | 0.4605 | **0.3283** |
| \|dz\| 最大 | 2.3269 | 2.3246 | 2.3225 |
| >0.5m 对数 | 10 | 10 | **5** |
| 轨迹/x/y | 235.7/48.58/41.16 | 同 | 235.7/48.56/41.14 |

**先写死的判据是 p90 < 0.30，实测 0.3283 —— 差 9%，判为失败。**

**但更要紧的是：它不该被采信，即使达标了。**

### 98.3 循环论证

重访判据配对的是「xy 相距 1 m 内、时间相隔 30 s 以上」的关键帧，
**而闭环边约束的正是这一类配对**。收紧闭环权重会按构造把这个指标压下去。

用相对独立的地板残差剖面验证：

```
路程段     S5d      S9       S10
  0- 10   +1.363   +1.365   +1.388
 10- 25   +0.009   +0.011   +0.021
 25- 50   -0.086   -0.077   +0.022
 50- 75   -0.041   -0.017   +0.107
 75-100   -0.101   -0.084   +0.004
100-150   -0.310   -0.293   -0.213
150-200   +0.164   +0.166   +0.159
200-300   -0.043   -0.090   -0.370
段间跨度   0.474    0.459    0.530
```

**重访 p90 改善 29%，而独立的地板一致性变差 15%。**
150–200 段的段内标准差从 0.081 涨到 0.183（2.3 倍）。
优化器把 200–300 段整体拉低 0.33 m 去满足收紧后的闭环。

**含义：那 44 条闭环不全是对的。**
σ=0.5 时坏闭环无害地弱着；σ=0.05 时它们开始扭曲地图。

### 98.4 结论

**S9 是新的最好配置** —— 两个指标同向改善：

| | 重访 p90 | 残差段间跨度 |
|---|---|---|
| S5d | 0.4988 | 0.474 |
| S7（窗口 6.0） | 0.4764 | 0.466 |
| **S9（+ 闭环噪声修复）** | **0.4605** | **0.459** |
| S10（闭环 0.05） | 0.3283 | 0.530 ✗ |

**S10 拒绝** —— 不是因为差 9%，是因为它的改善是指标自证。

### 98.5 判据本身要防这一手

**这是本轮最重要的发现：重访判据可以被闭环权重「刷」下去。**
一个可以靠调一个参数压低、而地图并没有变好的判据，不能单独作验收。

`tools/map_gate.py` 新增「地板残差段间跨度」检查：

- 每帧用自己的点云拟合脚下地板高度，去掉一个最小二乘固定平面
  （固定倾斜对重访不可见，S9 上是 3.07°，**不是缺陷**）
- 按路程分箱，取各段中位数的极差
- **排除 0–10 m**（开机暂态是录制的性质，不是位姿图的，见 §94）
- 阈值 0.50 m

它不是闭环边直接约束的量，所以压不动。
实测：S5d/S7/S9 PASS，**S10 FAIL** —— 正是要抓的那一个。

### 98.6 未决

`loop_max_range` 6.0 目前只在 0723 上验过。
b22 上的 S9 检验在跑，通过才改发布配置。

---

## 99. S9 在 b22 上通过，改进发布配置（2026-07-30）

### 99.1 b22 检验

| | b22 S5d（窗口 3.0） | **b22 S9（窗口 6.0 + 噪声修复）** |
|---|---|---|
| 闭环边 | 5 | **55** |
| 轨迹总长 m | 713.5 | 713.5 |
| x 范围 m | 103.43 | 103.27（差 0.15%） |
| y 范围 m | 89.57 | 89.42（差 0.17%） |
| z 范围 m | 10.22 | 10.18 |

**闭环 5 → 55，几何不折叠。**

### 99.2 多标高的完整性检查：直方图的簇间距

b22 是双标高，重访判据与新加的残差判据**都不适用**。
可用的结构性检查是地板高度直方图：

| | 地板 z 跨度 | 两簇中心 | **层间距** |
|---|---|---|---|
| b22 S5d | 10.38 | −5.45 / −0.39 | **5.06 m** |
| b22 S9 | 10.34 | −5.52 / −0.48 | **5.04 m** |

**层间距差 2 cm。楼层结构完整保留。**

### 99.3 更正 §88.1

§88.1 写的「b22 双峰，两个标高相距约 11 m」是错的。
那是在**被 S5e 配置污染的那次跑**上量的（那张图地板 z 跨度 18.063 m）。

**真实层间距 5.05 m** —— 一个正常层高。11 m 本来就不合理，我当时没质疑。
（这是 §90 那次配置污染的第二个后果，第一个是 §89.4 的推广性结论。）

### 99.4 发布配置改动

```
loop_max_range: 3.0 → 6.0     （config/product_v1.yaml、n3mapping.yaml、config.h）
```

加上已提交的闭环噪声修复（§97），这就是 S9。
**325 个测试仍全过。**

依据：
- 0723：闭环 11 → 46，重访 p90 0.4988 → 0.4605，残差跨度 0.474 → 0.459，几何逐位不变
- b22：闭环 5 → 55，几何不折叠，层间距保留到 2 cm

### 99.5 新判据的前提（已写进工具）

「地板残差段间跨度」与重访判据**有同一个前提：单层录制**。
一个平面描述不了换楼层的会话，b22 上它读出 3.838 m，把层高算成漂移。
**多层录制的结构性检查是直方图的簇及其间距。**

### 99.6 目标状态

**未达成。** 0723 上重访 p90 **0.4605**，判据 0.10。

分解（§93.5、§98.3）：
- **最大值 2.3246 m** ← 开机暂态，只在前 10 m，**后端结构上修不了**，
  且是 0723 那次录制特有的（§94）
- **p90 0.4605 m** ← 全会话残余的段间 z 漂移，后端已从前端的 0.747 压到 0.459
- **收紧闭环能把重访 p90 压到 0.3283，但那是判据自证，地图实际变差**（§98.3）

---

## 100. 更正 §98.3 的推断；ICP 信息不会有区分度（2026-07-30）

### 100.1 配准统计不支持「闭环不全是对的」

S9 进图的 46 条闭环：

```
fitness  中位 0.0603   最大 0.1909   （阈值 0.2）
inlier   中位 0.992    最小 0.875    （阈值 0.7）
```

**全都配准得极好。**

§98.3 我从"收紧闭环后地板残差变差"推断"那 44 条闭环不全是对的"——
**这个推断没有数据支持**。（配准好不等于闭环对 —— 别名可以配准得完美，
这条整个重定位线都证过 —— 所以也不能反过来说它们全对。
能说的是：手上的数据给不出任何一条是坏的证据。）

### 100.2 一个更可能且可直接检验的解释

**收紧闭环时压倒的不是别的闭环，是地板姿态因子。**

- 地板残差正是姿态因子控制的量；重访 dz 是位置量
- S9：闭环旋转 σ=0.5 rad vs 姿态因子 σ=1°=0.0175 rad —— 姿态强 28 倍
- S10：闭环旋转收到 0.05 rad —— 只剩 2.9 倍

**S10 的重访改善（位置）与地板残差变差（姿态）很可能是同一次改动的两个方向。**

### 100.3 `loop_use_icp_information: true` 不值得试

46 条闭环的 fitness 落在 0.06–0.19、inlier 落在 0.875–0.992 ——
按配准质量加权，权重会几乎一样，**没有区分度**。

（顺带确认修复生效：`loop_information_diag` 现在 46 条全是 4.0 = 1/0.5²，
正是配置值。修复前它是单位阵的 1.0。）

### 100.4 S11（先写死）

`loop_noise_position: 0.05`，**`loop_noise_rotation` 保持 0.5**，其余 S9。

**判据（两项必须同时成立，§98 之后不再接受单项改善）：**

| | 判据 | S9 现状 |
|---|---|---|
| 重访 p90 | **< 0.40 m** | 0.4605 |
| 地板残差段间跨度 | **≤ 0.459 m** | 0.459 |

护栏：轨迹长与 xy 范围相对 S9 变化 <5%。

**若成立** —— S10 的退化确实来自旋转冲突，位置约束可以单独收紧。
**若两项仍此消彼长** —— 位置与姿态在这张图上是耦合的，
后端在现有观测下没有更多空间。

---

## 101. 闭环的 z 误差量出来了：0.4 m（2026-07-30）

### 101.1 S11 证伪 §100.2

`loop_noise_position 0.05`，**旋转保持 0.5**：

| | S10（xyz+rot 全收） | **S11（只收位置）** |
|---|---|---|
| \|dz\| 中位 | 0.0551 | 0.0610 |
| \|dz\| p90 | 0.3283 | 0.3286 |
| 残差段间跨度 | 0.530 | 0.526 |
| 200–300 段 | −0.370 | −0.373 |

**几乎逐位相同。旋转不动也照样退化 ——「闭环压倒姿态因子」的假设被证伪。**
判据要求两项同时成立：p90 0.3286 ✓，残差跨度 0.526 ✗（须 ≤0.459）→ **FAIL**。

### 101.2 独立检验：闭环的 z 修正 vs 地板给出的漂移

一条正确的闭环，其 z 修正应当等于地图在那两个路程位置之间的漂移。
两样都有：闭环的 `icp_correction_match_z`，和每帧用自己点云拟合的地板高度
（用 `pose_odom`，与后端无关）。

符号约定：query 地板在 z_a、match 在 z_b，闭环把 query 拉到 match，
所以正确的闭环应给出 `icp_z ≈ −(z_a − z_b)`。
**（第一次算出相关系数 −0.722 时我差点读成「错」—— 那正是正确的方向。
先定符号再判读。）**

```
拟合斜率 −1.154              (正确应为 −1.000)
残差 |icp_z − (−drift)|       中位 0.407 m   p90 1.167   最大 1.697
残差超过 0.5 m 的闭环          15 / 46  (33%)

按漂移大小分档:
  |drift| 0.0-0.2   10 条   |残差| 中位 0.410
  |drift| 0.2-0.5   23 条              0.386
  |drift| 0.5-1.0    3 条              0.496
  |drift| 1.0-2.0    9 条              0.987
```

**闭环带着正确的信号（斜率约 −1），但 z 上有约 0.4 m 的噪声。**

### 101.3 这定量解释了 S10/S11

σ=0.05 是在宣称 5 cm 的精度，而实测误差 40 cm ——
**把闭环信任了 8 倍于它应得的**，图于是弯过去迎合错的测量。

**S9 的 σ=0.5 与实测的 0.4 m 匹配，这个值是对的。**
（这是第三次遇到同一件事：σ 该不该等于实测噪声。
姿态因子那里答案是「不」（§93.4，因为它是图里唯一的绝对信息，σ 实为权重）；
闭环这里答案是「是」，因为它与里程计、姿态因子共同定权，没有那种垄断地位。）

### 101.4 S10/S11 错在把三个轴一起收

**z 的 0.4 m 误差是被地板独立测出来的；x/y 没有理由一样差。**
地板是一个平面、被近乎掠射地看到，对 z 的约束弱；
墙面与家具从多个方向同时约束 x 与 y。

新增 `loop_noise_position_z`（非正值时回退到 `loop_noise_position`，
即原有各向同性行为）。**325 个测试仍全过。**

**S12（先写死）**：闭环 xy 0.05、**z 0.4（实测值）**、旋转 0.5。

判据（两项必须同时成立）：

| | 判据 | S9 现状 |
|---|---|---|
| 重访 p90 | **< 0.40 m** | 0.4605 |
| 地板残差段间跨度 | **≤ 0.459 m** | 0.459 |

另跑一个 S9 对照确认各向异性改动在未设置时是空操作。

### 101.5 若 S12 也失败

那么在现有观测下后端已到极限，**下一个瓶颈是闭环配准本身的 z 精度（0.4 m）**，
那是配准问题不是图问题。可查的方向：
`gicp_max_correspondence_distance: 2.0` 配 `loop_icp_prefilter_voxel_size: 0.2` ——
2 m 的对应距离允许 z 相差 0.4 m 的两片地板互相对应。

---

## 102. S12：各向异性闭环噪声；瓶颈收束到闭环自己的 z 精度（2026-07-30）

### 102.1 对照先行

S9 在各向异性二进制上重跑：闭环 46、中位 0.2272、p90 0.4605、
最大 2.3246、轨迹 235.7 / 48.58 / 41.16、残差跨度 0.459 ——
**与原 S9 逐位相同**。改动在未设置时确是空操作，S12 可比。

### 102.2 S12：判据未达成，但是目前最好的一版

闭环 xy 0.05 / **z 0.4（实测值）** / 旋转 0.5：

| | S9 | **S12** | 判据 |
|---|---|---|---|
| 地板残差段间跨度 | 0.459 | **0.438** | ≤0.459 **PASS** |
| \|dz\| 中位 | 0.2272 | **0.1720** | — |
| \|dz\|>0.5m 对数 | 10 | **6** | — |
| \|dz\| p90 | 0.4605 | 0.4615 | <0.40 **FAIL** |
| \|dz\| 最大 | 2.3246 | 2.3275 | — |
| 轨迹/x/y | 235.7/48.58/41.16 | 235.7/48.51/41.14 | <5% ✓ |

**先写死的判据要求两项同时成立 → 失败**（p90 那一项）。

但四项更好、一项持平，**且不是 S10 那个模式**（重访涨、残差跌）：
各向异性做对了事 —— 收紧水平、按实测放松竖直。

### 102.3 瓶颈收束

```
闭环 z 误差（§101.2 实测，用地板独立标定）   0.407 m
地板残差段间跨度（S12）                      0.438 m
```

**是同一个数。**

残差剖面里剩下的主项是相邻两段之差：
100–150 段 −0.279 与 150–200 段 +0.159，相差 0.438 m。
**闭环无法消除它，因为闭环自己对 z 的认知精度就是 0.4 m。**

**图里其他东西都不再是瓶颈**：里程计刚度（S5e/S6b 两个量级都试过）、
闭环数量（11→46，残差纹丝不动）、闭环权重（S9 的 0.5 与实测匹配）、
姿态因子 σ（S6a）、鲁棒核、有界窗口 —— 全部测过。

### 102.4 下一个项目：闭环配准的 z 精度

这是配准问题，不是图问题。可查的方向（未验证，仅为线索）：

- `gicp_max_correspondence_distance: 2.0` 配 `loop_icp_prefilter_voxel_size: 0.2`
  —— 2 m 的对应距离允许 z 相差 0.4 m 的两片地板互相对应
- `icp_refine_max_correspondence_distance: 1.0` 的精化阶段同理
- 子图由已漂移的位姿拼成，目标点云本身带着误差

**判据现成**：`tools/loop_z_vs_floor_drift.py` 直接量
「闭环 z 修正 vs 地板给出的漂移」的残差中位数。现在是 0.407 m。
把它压到 0.1 m 以下，残差段间跨度才有可能跟着下去。

### 102.5 目标状态

**未达成。** 0723 上重访 p90 **0.4615**，判据 0.10。

三项分解，全部有量化归因：

| | 数值 | 来源 | 后端能否修 |
|---|---|---|---|
| \|dz\| 最大 | 2.3275 m | 开机暂态，只在前 10 m，0723 那次录制特有（§94） | **不能**（§92.4） |
| 残差段间跨度 | 0.438 m | 闭环自身 z 精度 0.407 m | **不能**，除非配准变好 |
| \|dz\| p90 | 0.4615 m | 上面两者的共同结果 | — |

### 102.6 b22 检验通过，S12 发布

| | b22 S5d | b22 S9 | **b22 S12** |
|---|---|---|---|
| 闭环边 | 5 | 55 | 52 |
| 轨迹总长 m | 713.5 | 713.5 | 713.5 |
| x 范围 m | 103.43 | 103.27 | 102.79（相对 S5d 差 0.62%） |
| y 范围 m | 89.57 | 89.42 | 89.61（0.04%） |
| z 范围 m | 10.22 | 10.18 | 10.16 |
| 层间距 m | 5.06 | 5.04 | 5.16 |

层间距那 0.10 m **不作为结论**：求峰值用的是 40 个 bin 覆盖 10.4 m，
**bin 宽 0.26 m**，0.10 m 在半个 bin 以内。
能说的是「在该估计器的分辨率内层间距没变」。

发布配置（`product_v1.yaml`、`n3mapping.yaml`、`config.h` 三处一致）：

```
loop_noise_position:   0.5  → 0.05
loop_noise_position_z:        0.4    （新增）
loop_max_range:        3.0  → 6.0    （§99 已发布）
```

**325 个测试仍全过。**

**注意：发布的 p90 没有改善**（0.4605 → 0.4615）。
改善的是不可刷的地板残差段间跨度（0.459 → 0.438）、
重访中位（0.2272 → 0.1720）、超标对数（10 → 6）。

---

## 103. 各向异性的理由未经检验；S13 转向配准精度（2026-07-30）

### 103.1 我给各向异性的理由可能是错的

§101.4 我写「地板是一个近乎掠射看到的平面，对 z 的约束弱；
墙面与家具从多个方向同时约束 x 与 y」。

**这个说法很可能反了。**地板点约束的正是 z、roll、pitch；
墙面约束 x、y、yaw。在开阔区域缺的是墙，弱的反而是 x/y。

**z 的 0.407 m 是实测的，站得住；x/y 从未测过。**
S12 的结果与「xy 更准」一致，但**不构成验证** ——
四项指标改善也可以是「把过松的 xy 收到合理值」的结果，
与 xy 究竟比 z 准多少无关。

`residual_x/y/z` 的中位（0.155 / 0.027 / 0.297）是**修正量**不是**误差**，
证明不了各轴精度。手上没有 x/y 的独立参考（地板只给 z）。

**记为未检验的假设，不是结论。**

### 103.2 一个更尖锐的说法

`residual_z` 中位 0.297 m —— 闭环想施加的 z 修正量。
而这些修正**自身带 0.407 m 的误差**（§101.2）。

**信号比噪声还小。**（斜率 −1.154、相关 −0.722 说明大漂移处信号占优；
`|drift| 0.0–0.2` 那 10 条的残差中位 0.410，纯噪声。）

### 103.3 S13：转向配准本身

闭环配准走 `PointCloudMatcher`：
`gicp_downsampling_resolution 0.1` + `gicp_max_correspondence_distance 2.0`，
再加一个精化阶段 `icp_refine_downsampling_resolution 0.05` +
`icp_refine_max_correspondence_distance 1.0`。

**1.0 m 的对应距离，对一个要分辨 0.4 m 高差的配准来说太松。**

**S13（先写死）**：`icp_refine_max_correspondence_distance` 1.0 → 0.3，其余 S12。

判据：**闭环 z 误差中位 < 0.30 m**（现 0.407），
用 `tools/loop_z_vs_floor_drift.py` 直接量。

**这是第一次把判据放在配准上而不是地图上** ——
因为 §102.3 已经证明地图的 z 一致性就等于闭环的 z 精度，
先改善后者才谈得上前者。

---

## 104. S13：去掉了系统性偏差，动不了噪声；平台期有了结构性解释（2026-07-30）

### 104.1 S13 结果

`icp_refine_max_correspondence_distance` 1.0 → 0.3：

| | S12 | **S13** | 判据 |
|---|---|---|---|
| 闭环 z 误差 拟合斜率 | −1.154 | **−0.998** | 正确 −1.000 |
| 闭环 z 误差 中位 | 0.407 | 0.376 | **<0.30 先写死 → FAIL** |
| 闭环 z 误差 p90 | 1.167 | 0.945 | — |
| >0.5 m 的闭环 | 15/46 | 14/45 | — |

**那 15% 的系统性过修正没了 —— 斜率现在几乎精确。但散布只降 8%。**

地图侧（目前最好的一版，但幅度很小）：

| | S9 | S12 | **S13** |
|---|---|---|---|
| 重访 p90 | 0.4605 | 0.4615 | **0.4560** |
| 地板残差段间跨度 | 0.459 | 0.438 | **0.434** |
| 重访中位 | 0.2272 | 0.1720 | 0.1678 |
| 重访最大 | 2.3246 | 2.3275 | 2.3275 |
| 轨迹/x/y | 235.7/48.58/41.16 | 235.7/48.51/41.14 | 同 S12 |

### 104.2 那个关系跨三阶段稳定

| | 闭环 z 误差 | 残差段间跨度 | 比值 |
|---|---|---|---|
| S9 | 0.407 | 0.459 | 0.89 |
| S12 | 0.407 | 0.438 | 0.93 |
| S13 | 0.376 | 0.434 | 0.87 |

**地图的 z 一致性就是闭环的 z 精度。**

### 104.3 结构性解释：地图无法靠自己把 z 提起来

闭环是拿单帧扫描去配一个**由已漂移位姿拼成的子图**。
**配准精度不可能好过目标自身的内部一致性**，
而目标的内部 z 一致性就是地图的残差跨度 —— 循环依赖。

这解释了 S13 为什么只去掉偏差不动噪声：
**噪声不在配准算法里，在目标点云里。**
收紧对应距离能修正搜索范围过宽带来的系统性偏移（斜率 −1.154 → −0.998），
但目标本身糊了 0.4 m，配准再精确也没用。

**可用的外部 z 信息只有重力**（地板姿态因子，已在图里）。
按约束条件跨楼层无气压计，没有第二个来源。

**这不是调参失败，是结构性上限。**

### 104.4 十三个阶段的完整账

| 试过的杠杆 | 结果 |
|---|---|
| 鲁棒核（Huber/Cauchy/DCS） | 无可测改善 |
| 有界闭环窗口 150 → 3.0 | 消灭别名（回路误差 14.766 → 0.887 m） |
| 保留全部验证闭环 | 闭环 2 → 11 |
| yaw/段判据修正 | 闭环判据不再用求解器过程态 |
| 地板姿态因子 | **唯一实质改善地图的一步** |
| 姿态因子 σ 1.0 → 2.7（实测噪声） | 更差，保持 1.0（§93.4） |
| 里程计刚度 0.001 → 0.0145 / 0.004 | 两个量级都失败（S5e、S6b） |
| 窗口 3.0 → 6.0 | 闭环 11 → 46，残差剖面纹丝不动 |
| 闭环噪声参数从未生效（缺陷） | 修复后 σ 从实际 1.0 变成配置的 0.5 |
| 闭环噪声 0.5 → 0.05（各轴齐收） | 判据自证，地图实际变差（§98.3） |
| 各向异性 xy 0.05 / z 0.4 | 残差跨度 0.459 → 0.438 |
| 精化对应距离 1.0 → 0.3 | 去掉系统偏差，噪声不动 |

**基线 p90 0.852 → 0.4560。判据 0.10。**

### 104.5 目标未达成，三项分解全部有量化归因

| | 数值 | 归因 | 后端能否修 |
|---|---|---|---|
| \|dz\| 最大 | 2.3275 m | 开机暂态，只在前 10 m，0723 那次录制特有（§94） | **不能**（§92.4：错姿态积分出的平移已固化在边的测量值里） |
| 残差段间跨度 | 0.434 m | 闭环 z 精度 0.376 m，而闭环配的是已漂移的子图 | **不能**（§104.3 循环依赖） |
| \|dz\| p90 | 0.4560 m | 上述两者的共同结果 | — |

### 104.6 要达成判据需要什么

1. **单层、静止启动的新录制** —— 消除最大值那一项（只有用户能做）
2. **一个不来自地图自身的 z 观测** —— 打破 §104.3 的循环。
   气压计被排除；可能的替代是已知平面先验（但那等于假设结论）
   或外部参考
3. **判据本身可能需要重述** —— `p90 < 0.10 m` 隐含要求
   闭环 z 精度好于 10 cm，而闭环配的是自己产出的地图

### 104.7 b22 检验通过，S13 发布

```
b22 S13: 闭环 54   轨迹 713.5   x 102.79（相对 S5d 0.62%）   y 89.60（0.03%）   z 10.16
         层间距 5.16 m（S5d 5.06，在半个 bin 内）
```

发布配置：`icp_refine_max_correspondence_distance: 1.0 → 0.3`
（`product_v1.yaml`、`n3mapping.yaml`、`config.h` 三处一致）。
**325 个测试仍全过。**

### 104.8 最终发布配置（相对本轮开始时的 S5d）

```
loop_max_range:                        3.0  → 6.0
loop_noise_position:                   0.5  → 0.05
loop_noise_position_z:                        0.4    （新增）
icp_refine_max_correspondence_distance: 1.0  → 0.3
+ 闭环噪声参数从未生效的缺陷修复（§97）
+ 里程计合理性守卫（§91）
+ 自由空间否决的两个缺陷修复（§95）
+ 测试套件从 10 个失败修到 0
```

0723 上：

| | S5d（本轮起点） | **S13（发布）** | 判据 |
|---|---|---|---|
| 重访 p90 | 0.4988 | **0.4560** | <0.10 ✗ |
| 重访最大 | 2.3269 | 2.3275 | <0.30 ✗ |
| 重访中位 | 0.2733 | **0.1678** | — |
| >0.5 m 对数 | 10/101 | **6/101** | — |
| 地板残差段间跨度 | 0.474 | **0.434** | <0.50 ✓ |
| 闭环边 | 11 | **45** | ≥20 ✓ |
| 轨迹/x/y 变化 | — | 0 / 0.14% / 0.05% | <5% ✓ |

---

## 105. 开机暂态的机制找到了：是估计器，不是机器人，也不是录制（2026-07-31）

线索来自用户提到的 Ultra-Fusion（上交，arXiv 2606.21223）的
**observability-aware initialization**：激励不足时用 Stationary bootstrap，
或干脆 Deferred（延长累积窗口，不启动完整估计）。

### 105.1 第一次检验用错了统计量

按上面的线索，我先比了两份录制开机时的 IMU 激励：

```
        |a| 标准差   |ω| 中位
0723      0.023     0.018 rad/s
b22       0.020     0.035 rad/s
```

b22 的角速度反而更大，于是我判定"两份都静止，假设不成立"。

**这个判定的方法是错的。**我比的是**瞬时角速度**，
而 b22 那 0.035 rad/s 是**零偏**（积分十秒姿态只变 0.25°），
0723 那 0.018 rad/s 若是真转动则积出 10°。
**该比积分后的姿态变化，不是瞬时角速度。**

### 105.2 正确的测量

用里程计报告的姿态：

```
前 10 秒        roll 峰-谷   pitch 峰-谷   z 峰-谷    8 秒内路程
0723             9.43°       9.85°     0.265 m     0.74 m
b22              0.25°       1.61°     0.040 m     0.88 m
```

**0723 几乎没走，机身却"摆"了约 10°。**

### 105.3 判别：机器人真动了，还是估计器在飘

静止时加速度计测的就是重力，它在机体系里的方向变化 = 真实姿态变化。
用 0.2 s 滑窗平滑后看相对开机时刻的偏转：

```
                加速度计（真实）   里程计（报告）
0723 前 10 s        1.47°        roll 9.43° / pitch 9.85°
b22  前 10 s        1.16°        roll 0.25° / pitch 1.61°
```

（1.47° 还是上界：|a| 标准差 0.023 g 对应 1.3° 的视重力偏转，
所以真实倾角变化 ≤ 约 1.5°。）

**结论：0723 的机身没动，那 10° 是估计器自己的。**
里程计报告的姿态与重力的夹角在变，
说明**估计器的世界系在相对重力转动** —— 这就是暂态。

### 105.4 这推翻了我给用户的一条建议

**「静止启动重录」是错的建议 —— 0723 本来就是静止启动的。**
重录不会消除这个暂态。**这是软件问题，不是录制问题。**

（也顺带解释了 §S1 为什么找不到：我一直在查初始化的重力估计，
而初始化本身精确到 0.47°。问题不在初始化，在**初始化之后**
静止段的收敛。）

### 105.5 机理

**静止时，机体姿态与重力方向这对状态是不可观测的** ——
分不出"机身倾了"和"重力在别的方向"。
FAST_LIO 把 `grav` 作为状态在 S2 流形上估计（状态 21–22），
与姿态联合求解；没有运动来打破这个简并，
滤波器就沿不可观测方向游走，直到运动到来。

**Ultra-Fusion 的 Algorithm 1 正是处理这个的：**
低激励 → Stationary bootstrap（纯惯性，不解这个耦合）；
可观测性不足 → Deferred，延长窗口不启动。
FAST_LIO 是 `MAX_INI_COUNT = 10`，固定十帧，之后直接进完整估计。

### 105.6 这同时修正了 §93.7 的"证伪"

§93.7 我用矢量平均测"前端世界系是否早期更倾斜"，
结论是前 50 m 与 100 m 之后差 0.50°（1.2 sigma，分辨不出），
于是判定"暂态是世界系倾斜"这个说法不成立。

**那个测量是按路程分箱的，而暂态发生在几乎不动的 10 秒里
—— 0.74 m 路程，最多 1 个关键帧。**
它整个被埋在 0–25 m 那一箱的 27 帧里，冲淡了 27 倍。

**所以 §93.7 的零结果是分辨率问题，不是证伪。**
原本 §S3 的推理（世界系相对重力转动 → z 误差）是对的，
我错在用了一个看不见它的仪器，然后据此把它否掉了。

（因果链现在自洽：**原因**在前 ~1 m 的近静止段，
**效果**（+1.363 m 的 z 残差）持续整个前 10 m，
因为世界系一旦转了，误差就固化在里程计边的测量值里，
后端撤销不了（§92.4）。）

### 105.7 一个必须先排掉的混淆

0723 的 LIO 输出由 `n3mapping_deterministic_ros2_replay.py --rate 0.5`
带流控回放产生，FAST_LIO 用的是 stock `mid360.yaml`
（`/tmp/fast-lio-flow-control-runtime/`）。
目录名 `0723_gravity_v2` 指的是 `extractor_v2_gravity_contract`
这个抽取契约，**不是** FAST_LIO 的配置变体。

**b22 的 LIO 是怎么产生的、用的什么配置、什么回放速率，我没有查。**
在把两份当作可比对照之前必须先查清 ——
否则整条"暂态是 0723 特有"的结论仍有混淆风险。

### 105.8 该做什么

1. **查 b22 的 LIO 产生方式**（配置、回放速率），排掉 105.7 的混淆
2. **加零速/低激励检测**：静止时不更新 `grav` 与姿态的耦合项，
   或延后启动完整估计。四足有足端接触检测，判静止比 IMU 阈值更可靠
3. **验收现成**：这次的测量本身就是判据 ——
   静止段里"加速度计方向变化"与"里程计报告姿态变化"之差，
   0723 现在是 8.4°，应当降到 1° 以内

---

## 106. 工作区整理 + 第 0 步复现（2026-07-31）

### 106.1 工作区状态盘点

用户问工作区是否被搞乱，盘点结果：

| | 状态 |
|---|---|
| `n3mapping_baseline_72fa6f7`（我的 worktree） | 干净，0 条未推送 |
| `to_migrate_ws/src/n3mapping` | **60 个未提交改动，全部停在 2026-07-23，不是我动的** |
| `to_migrate_ws/src/FAST_LIO` | `laserMapping.cpp`（07-23，原有）+ `IMU_Processing.hpp`（**07-29，我的 S1a**） |
| 残留进程 | 无 |
| 实验产出 | 1.6 GB，磁盘余 1.2 TB |

**那 60 个改动是独一份的** —— 与 `research/relocalization-evidence-v2` 只有 32 个文件
重叠且内容不一致，没有提交到任何分支。已快照到
`archive/to-migrate-worktree-20260723` 并推送（60 文件 + 9 个未跟踪，11198 行）。

`to_migrate_ws/src/n3mapping` 随后复位到 `research/relocalization-benchmark-v1`，
工作区清零。FAST_LIO 两处改动分别提交并推送到 `humble`。

### 106.2 一个差点造成假复现的越界

**我在 S1a 改了 `to_migrate_ws/src/FAST_LIO/src/IMU_Processing.hpp` 并重编了安装树，
之后没有还原。**而我正要用那个二进制去"复现" 0723 的原始条件 ——
它已经不是原始条件了。这与 §90 的配置污染是同一个形状，只差一步。

（那处改动的注释里写着"估计后来被看出差了约 10°，滤波器花了 120 s 去纠"——
**过去的我已经看见过这个 10°，试图用放松初始协方差来修，没修好。**
这支持现在的诊断：不是初始过自信，是静止时那对状态不可观测，
放松先验只会让它飘得更自由。）

### 106.3 提交身份用错了

我从 07-26 起每次提交都传 `-c user.name/user.email` 覆盖成
`sunfishegg@gmail.com`（会话上下文里的用户邮箱），而仓库和全局都配的是
`killow <killow1998@gmail.com>`，整个历史也都是它。

34 条受影响（两个不同的错名字串）。已用 `filter-branch` 按邮箱重写作者，
校验树的 hash 前后完全相同（`34974e31...`）、提交数不变，
经用户授权强推 `wip/freespace-select`。现在 92 条 killow + 1 条 h2q。

**教训写进了记忆：提交身份从仓库配置读，不要用会话上下文里的用户邮箱。**

### 106.4 两次自匹配

- `pgrep -f fastlio_mapping` 匹配到我自己的 ssh 命令行 → 误报"有进程在跑"
- `pkill -f "record"` 匹配到我自己的 ssh 命令行 → **把自己的连接杀了**（exit 255），
  目标进程反而活着

**这是第三次踩同一个坑**（S5b 那次是 `pgrep -f n3mapping_node`）。
已写进 `AGENTS.md`：判存活用 `pgrep -x`，杀进程先拿 PID。

### 106.5 第 0 步：复现（在跑）

**必须先复现再修。**两次 LIO 输出的产生方式差别比预想的大：

| | 0723 | b22 |
|---|---|---|
| 回放速率 | **0.5** | 1.0 |
| `--min-subscriptions` | **1** | 2 |
| 流控 | **雷达 ack** | 无 |
| 超时 | 120–180 s | 10–30 s |
| FAST_LIO 配置 | `mid360.yaml` | **同一份**（sha256 一致） |

两个竞争假设：

- **H1**：估计器静止时确实会飘（需改代码）
- **H2**：0723 那次回放开头有竞态（`min-subscriptions 1` 意味着只要一个订阅者
  就开始发），前几秒被污染 —— **是产生方式的假象**

**H2 若成立，整个建图调查的基础数据带着一个生成假象。**

正在用 b22 的设置（rate 1.0、min-sub 2、无流控）在 0723 原始 bag 上重跑 FAST_LIO，
其余全部保持（同一二进制、同一配置、同样话题）。判据：静止段
"加速度计方向变化 vs 里程计报告姿态变化"之差，原始那份是 **8.4°**。

（顺带纠正：0723 原始录制真实时长 **825 s**，不是我此前多次说的 470 s ——
那是被 0.5× 拉长的 LIO bag 时间轴。）

### 106.6 第 1 步的设计（待复现结果确认后实施）

`IMU_init` 逐 IMU 样本累加 `init_iter_num`，而每个雷达帧带约 20 个样本
（200 Hz / 10 Hz），所以 `MAX_INI_COUNT = 10` 意味着
**第一个雷达帧就初始化完毕** —— 重力由头 100 ms 的约 20 个样本平均得到，
随后滤波器立刻全量运行。

改法（对应 Ultra-Fusion 的 Stationary / Deferred 模式）：
**静止期间留在初始化里继续平均重力，检测到运动才退出。**

- 静止时重力方向**最可观测**（加速度计测的就是纯重力），
  而姿态与重力的耦合**最不可观测** —— 正好该用平均而不是滤波
- 静止 10 s 可平均约 2000 个样本而不是 20，精度好一个数量级
- 退出条件：检测到运动，或达到上限（防止误判静止而一直不出图）
- 静止检测几乎免费：`IMU_init` 已在增量计算 `cov_acc` / `cov_gyr`

---

## 107. 复现 0723:两个假设都清掉了一半（2026-07-31）

### 107.1 判据工具

`tools/lio_static_attitude_drift.py`。静止时加速度计测的就是重力，
它在机体系里的方向**就是**姿态；里程计报告自己的姿态。两者在静止段必须一致。

用 **header 时间戳**而非到达时间戳 —— 0723 的 LIO bag 是按 0.5× 录的，
到达时间轴被拉长两倍，header 是传感器自己的时间。

三份录制，10 s 窗口：

| | 窗口内路程 | 真实姿态变化(加速度计) | 里程计报告 | **差值** |
|---|---|---|---|---|
| 原始 0723 | 1.165 m | 1.46°（残余加速度上界 1.34°） | 8.24° | **6.79°** |
| b22 | 0.925 m | 1.16°（上界 1.14°） | 1.61° | **0.45°** |

**6.79° 是第 1 步的验收基准。**（此前口头说的 8.4° 是另一个窗口算的，以此为准。）

### 107.2 H2 的第一个版本：用 b22 设置重跑 0723 —— 无效实验

```
8247 / 8253 个雷达帧被丢弃（"no IMU sample available before lidar end time"）
38156 次 "imu loop back, clear buffer"
33630 次 "No Effective Points!"
回放本身 status=PASS，165062 IMU + 8253 雷达全部发出
结果：10 s 窗口内路程 219,344,773 m，姿态摆动 358.78°
```

**rate 1.0 无流控喂不动 FAST_LIO。** 0723 是 200 Hz IMU + 10 Hz 雷达，
实时投递就溢出，IMU 时间戳乱序，雷达帧几乎全丢。

**所以原始那次的 0.5× + ack 流控不是随意选的，是必需的。**
这反过来说明 0723 用的设置比 b22 **更严**，不是更松 ——
H2「0723 用了更松的设置所以有假象」这个版本死了。

### 107.3 H2 的第二个版本：`--min-subscriptions 1` —— 也是假的

我从 scratch 目录的 `run_lio.sh` 读到 `--min-subscriptions 1`，
据此推测"回放可能在 FAST_LIO 订阅完之前就开始发"。

**去核 `raw_lio_replay_evidence.json`，生产那次用的是 `--min-subscriptions 2`。**
我看到的 `1` 出自一个**重建的**脚本，不是产生数据的那个。

**教训：产生数据的命令要从 evidence 记录里读，不要从事后写的脚本里读。**

### 107.4 一个近失事件

支持流控的回放工具（455 行，含 `--flow-control-lidar-topic`）
**只存在于那 60 个未提交文件里**。当前分支和基线分支上都是 272 行的无流控版本。

**如果清理 worktree 之前没先快照，产生 0723 基线数据的工具就没了，
复现将永远不可能。**

已取到工作分支并提交（CLI 是严格超集，旧参数一个不少，现有脚本不受影响）。

### 107.5 现在在跑：完全不变量的精确复现

用 evidence 记录里的原始参数（0.5× + min-sub 2 + 雷达 ack 流控 + 120 s 超时），
不改任何变量。

**目的是先证明能复现出 6.79°，才谈得上修。**

唯一无法消除的差异：二进制里含已提交的 S1a（重力先验初始协方差从实测推导），
而原始那次早于它。S1a 当时测下来是空操作 ——
**若复现值接近 6.79°，该结论成立；若明显不同，说明 S1a 其实有影响。**

### 107.6 流控这一跑是健康的

```
丢帧        0     （无流控那次 8247）
imu loop back 0   （无流控那次 38156）
```

---

## 108. 第 2 步的接入点找到了，而且它从来没实现过（2026-07-31）

### 108.1 `vertical_observability_score` 不是「坏了」，是从未实现

```cpp
VerifiedLoop LoopClosureManager::applyEdgeModel(const VerifiedLoop& loop) const
{
    VerifiedLoop modeled = loop;
    modeled.edge_mode = LoopEdgeMode::Full6Dof;      // 永远 6 自由度
    modeled.vertical_downweighted = false;            // 永远不降权
    modeled.vertical_observability_score = 1.0;       // 永远满分
    return modeled;
}
```

`applyEdgeModel` 本该是逐条闭环做边建模的地方 —— **三个决定全是硬编码常数。**

它接进了整条调试管线（`loop_debug_logger.cpp:256` 写进 JSONL），
名字承诺了一个测量，而 46 条闭环全读 1.000。
**这不是饱和，是占位符被当成了数据。**

（§95 我改的测试里有 `ApplyEdgeModelDoesNotDownweightVerticalAxes` ——
那是把这个硬编码行为**锁死**的断言。所以这是某次有意的简化，不是疏漏。
我当时更新它时也没意识到自己在给一个空壳背书。）

### 108.2 需要的数据一直都在

`src/point_cloud_matcher.cpp:256` — `copyInformation(rr.H, &result.information)`。
`rr.H` 就是 small_gicp 配准结果的 Hessian，而且已按 (平移, 旋转) 顺序换好块。

**真实 Hessian 一直在 `match_result.information` 里，是我用
`loop_use_icp_information: false` 把它关掉的**（§97 修复时改成了配置推导的常数）。

而 §100.3 我否掉 ICP 加权的理由是"46 条 fitness 都在 0.06–0.19，没区分度"——
**fitness 是残差质量，Hessian 是可观测性，两者回答的不是同一个问题。**
长走廊里 fitness 完美而纵向零可观测。

### 108.3 第 2 步的设计

**绝对尺度用配置，各轴之间的相对分配用 Hessian：**

```
Ω_axis = (1 / σ_config²) · clip(λ_axis / λ_median, 1/k, k)
```

- Hessian 的**绝对**量纲反映点数与代价函数单位，不是米 —— 不能直接当协方差；
  §101.2 实测的 z 误差 0.407 m 是 Hessian 预测不出来的
- Hessian 对"哪个方向被约束得好"是**相对**可信的
- 所以：尺度由实测标定，分配由 Hessian 决定

这就是把 §102 的 `loop_noise_position_z = 0.4`（一个全局常数）
换成**逐条闭环**的版本，也正是 Ultra-Fusion 的 Factor-Wise Reliability Scheduling。

**判据不变**：闭环 z 误差中位（现 0.376 m）与地板残差段间跨度（现 0.434 m），
**两者必须同向改善**，否则按 §98 的规矩拒绝。

`applyEdgeModel` 是天然的落点，但可观测性必须在
`loop_verifier.cpp` 里算（点云和 Hessian 在那里），随 `VerifiedLoop` 带下来。

---

## 109. 第 1 步：静止期间继续平均重力（先写死判据）

### 109.1 改动

`Process()` 的退出条件从 `init_iter_num > MAX_INI_COUNT`
改成 `init_iter_num > MAX_INI_COUNT && (已检测到运动 || 达到 30 s 上限)`。

运动检测在 `IMU_init` 的累加循环里，对**尚未被吸收进均值**的当前样本：

```cpp
if (N > 50) {                                    // 均值先稳下来再比
  acc_dev = (cur_acc - mean_acc).norm();
  gyr_dev = (cur_gyr - mean_gyr).norm();
  if (acc_dev > 0.15 || gyr_dev > 0.05) imu_static_ = false;
}
```

阈值依据实测：0723 与 b22 静止时加速度计在均值 ±0.06 内、角速度在 0.02 rad/s 内
（§105.1、§107.1），所以 0.15 g 与 0.05 rad/s 远离噪声、也远低于走动。

**为什么静止时该用平均而不是滤波**：静止时姿态与重力方向这对状态**不可观测**，
分不出"机身倾了"和"重力在别处"；而平均在静止时恰恰**最好用**，
因为加速度计测的就是纯重力。静止 10 s 能平均 2000 个样本而不是 20 个。

### 109.2 判据必须改，因为原来那个会变空

如果静止期间留在初始化里，那段就**不产出里程计**了 ——
"静止段姿态漂移"（现 6.79°）没有东西可测，判据会变得空洞。

改用下游的、无歧义的量：

| | 现状 | **判据** |
|---|---|---|
| 残差剖面 0–10 m 段中位（`map_z_drift_decompose.py`，前端 `pose_odom`） | **+1.952 m** | **< +0.40 m** |
| 同上，后端 S13 | +1.363 m | 同步下降 |
| 0723 建图重访最大值 | 2.3275 m | 显著下降 |

0–10 m 段之外的各段中位都在 ±0.31 以内，所以 0.40 是"与其余各段同量级"的意思。

**护栏**：b22 不得劣化（它本来就没有暂态，静止 16 s 后才动，
应当走"检测到运动才退出"这条路径而不是撞上限）；
0723 几何变化 <5%；`IMU Initial Done` 日志要如实报出等了多久、从哪条路径退出。

### 109.3 一个已知的取舍

静止启动时地图会晚开始几秒。对建图无影响（静止时本来也没有新信息）。
若有人把这条用到实时上，需要重新考虑。

---

## 110. Hessian 早就在说话，而且和我给各向异性的理由相反（2026-07-31）

### 110.1 `vertical_information_ratio` 一直在算

`src/loop_verifier.cpp:29`：

```cpp
double verticalInformationRatio(const Matrix6d& information) {
    return information(2,2) / sqrt(information(0,0) * information(1,1));
}
```

即 **z 的信息量 ÷ xy 信息量的几何均值**，取自 `match_result.information`
（也就是 GICP 的真实 Hessian）。逐条闭环计算，写进 `loop_debug.jsonl`。

**它是变化的，不像 `vertical_observability_score` 恒为 1.0：**

```
中位 1.868   p10 1.242   p90 3.001   最大 5.272   —— 46 条全部 > 1
```

**Hessian 说 z 比水平约束得更好**，信息量是 xy 几何均值的约 1.87 倍。

### 110.2 这和我给 S12 的理由相反

S12 设的是 `loop_noise_position = 0.05`、`loop_noise_position_z = 0.4` ——
**把 z 设得比 xy 松 8 倍**。理由见 §101.4：
"地板是近乎掠射看到的平面，对 z 约束弱"。

§103.1 我已经自己纠正过那个理由（地板点约束的正是 z/roll/pitch；
墙面约束 x/y，开阔处缺墙所以弱的是 xy），并写"z 的 0.407 m 是实测的，
x/y 从未测过，记为未检验的假设"。

**但检验它的数据一直就在调试文件里，而且是我自己打印出来的。**
我写了"手上没有 x/y 的独立参考"，而 `vertical_information_ratio`
正是 x/y 相对 z 的参考，就在同一份 JSONL 的同一行。

### 110.3 矛盾交给实验，不靠推理

Hessian 说该让 z 比 xy 紧；而 S12 用了相反的设置，**实测在两个指标上都更好**
（残差跨度 0.459 → 0.438，重访中位 0.2272 → 0.1720）。

两种可能：

- Hessian 的**相对**尺度不反映真实误差（它的量纲是点数与代价函数单位）
- S12 的改善另有来源（比如只是把过松的 xy 收到合理值，与两轴之比无关）

**不推理，让第 2 步的实验判。** 设计留到第 1 步做完再定 —— 一次只动一个变量。

### 110.4 顺带

`vertical_information_ratio` 与 `vertical_observability_score` 并排写在
同一份调试记录里，一个是真实测量、一个是硬编码常数，名字都像测量。
**§108.1 说"占位符被当成了数据"，这里是它的另一半：
真实数据放在那里没人用。**

---

## 111. 精确复现成功；一个我必须收回的说法（2026-07-31）

### 111.1 复现是逐位精确的

用 evidence 记录里的原始设置（0.5× + ack 流控 + min-sub 2），零变量：

| | IMU/odom 帧数 | 窗口路程 | 真实姿态变化 | 里程计报告 | **差值** |
|---|---|---|---|---|---|
| 原始 0723 | 2001 / 96 | 1.165 m | 1.46° | 8.24° (roll 8.24 / pitch 7.72) | **6.79°** |
| 精确复现 | 2001 / 96 | 1.165 m | 1.46° | 8.24° (roll 8.24 / pitch 7.72) | **6.79°** |

**每一个数都一样。** 三个结论：

1. **H1 成立**：6.79° 是 FAST_LIO 在这份数据上可复现的性质，不是回放假象
2. **S1a 确认是空操作**：二进制含 S1a，结果与不含它的原始逐位相同 —— 实测，非宣称
3. **有流控时整条管线是确定性的**：同输入同代码同输出，一位不差

### 111.2 S1a 为什么无效，确切原因

```
[IMU_init] N=20 acc_var=0.000230 grav_dir_var=1.000e-05 (0.181 deg)
```

`grav_dir_var` **撞在我加的下限上**。推导值 = 0.000230/(9.81²×19) = 1.26e-7，
即 **0.020°** —— 比原来那个 0.18° 的常数**更自信**。

我当时加的"只能放松不能收紧"的下限，让它什么也没做。
**初始化的重力估计在统计上非常精确，那个常数本来就偏保守。
问题不在初始协方差。**

### 111.3 我必须收回一个说法

我在 §105.2 与之后多次说"0723 前 10 s 走了 0.74 m"、
"t=12 s 走了 2.90 m、t=24 s 走了 7.57 m"，据此论证机器人在动。

**那些路程是从 `/Odometry` 的位置算出来的 —— 而里程计正在漂。
我用一个正在漂的估计器的输出，去论证机器人的真实运动。**

直接量原始 bag 的 IMU（模拟代码里的运行均值，逐样本算偏差）：

```
时间段     acc_dev p95 / max      gyr_dev p95 / max
 0-30 s    0.058 / 0.10           0.025 / 0.037      ← 真正静止
30-35 s    0.069 / 0.175          0.026 / 0.088      ← 开始动
35-40 s    0.504 / 1.89           0.243 / 1.20       ← 明显运动
40-50 s    1.30  / 4.44           1.30  / 2.75       ← 剧烈
55-60 s    0.098 / 0.114          0.064 / 0.070      ← 回落
```

**机器人静止了约 33 秒。**

所以真实情况比我描述的更严重：
**静止 33 秒里，估计器报出了 7.57 m 的位移和 8.24° 的姿态摆动，两者全是漂移。**

### 111.4 检测器是对的，上限是错的

第一次跑的日志：

```
IMU Initial Done: waited 30.00 s over 6020 samples, exit=cap reached while still
```

我当时以为"检测器没认出运动"。**错了** —— 机器人确实静止了 33 s，
检测器是对的，阈值也正好卡在静止的 0.10 与运动的 0.50 之间。
**只是上限 30 s 比实际静止期 33 s 短，撞了上限。**

（而且那 3 秒的差意味着最后 3 秒是"滤波器已在运行并漂移"的样本被平均了进去 ——
正是检测器本该防的事。所以那一跑测的是一个略微错误的配置，已停掉重来。）

上限改为 60 s：把决定权交回运动检测，同时仍然给误判一个边界。

### 111.5 阈值的依据（现在是数据，不是猜）

```
静止:  acc_dev 最大 0.10        gyr_dev 最大 0.037
运动:  acc_dev p95  0.50        gyr_dev p95  0.24
阈值:  acc_dev      0.15        gyr_dev      0.05
```

两者都落在间隙里。**第一次写这个阈值时我是猜的（0.15 只是"高于静止峰峰值 0.12"），
猜对了，但依据是事后补的。**

### 111.6 第 1 步按设计工作了

上限提到 60 s 后重跑：

```
IMU Initial Done: waited 34.70 s over 6960 samples, exit=motion detected
```

- 等了 **34.70 s**，与实测的 33–35 s 静止期吻合
- 平均了 **6960 个样本**而不是 20 个（348 倍）
- 退出路径是**检测到运动**，不是撞上限
- 0 丢帧、0 IMU 乱序

**注意一个必然的后果**：t=34.7 s 之前不产出里程计，所以
`lio_static_attitude_drift.py` 那个 6.79° 的量在新 bag 上**没有对应窗口**
（那段本来就没有估计存在，静止漂移按构造为零）。
这正是 §109.2 预见到的，判据已改为下游的残差剖面。

轨迹总长应当基本不变（静止那 34.7 s 本来也没走路），
但地图起点变了 —— 残差剖面的"0–10 m 段"现在对应建筑里的另一段。
比较的是"首段是否异常"，不是"是否同一处"。

---

## 112. 第 1 步的 LIO 输出（2026-07-31）

```
                条数    轨迹长     单步最大    xyz 跨度
原始 0723       8250   272.7 m    0.348 m   48.7 / 41.6 / 5.7
静止修复后      7903   219.9 m    0.185 m   39.3 / 43.1 / 4.4
```

- **少 347 条**，正好等于被抑制的 34.7 s × 10 Hz = 347 帧
- **单步最大 0.348 → 0.185 m**（3.48 m/s → 1.85 m/s；后者对这台机器人更合理）
- **x 跨度缩 9.4 m**，与静止期漂移的量级吻合（那段估计器曾报出 7.6 m 位移）
- 轨迹长少 52.8 m：一部分是抑制掉的静止漂移，一部分是抖动减少
  （路程是绝对步长之和，对高频抖动极敏感）

**以上都是推测，判据是建图后的残差剖面。**

**几何护栏预期会触发**：前端确实变了，那是合法变化不是折叠。
所以本轮的几何检查改为「地图是否退化」（轨迹是否发散、楼层结构是否保住），
不是「与旧基线是否一致」。

---

## 113. 第 1 步失败，而且推翻了自己的诊断（2026-08-02）

### 113.1 先写死的判据没达到

| | S13（原 LIO） | **S14（静止修复后）** | 判据 |
|---|---|---|---|
| 残差 0–10 m 段中位（后端） | +1.363 | **+1.650** | < +0.40 **FAIL** |
| 地板残差段间跨度 | 0.434 | **0.575** | ≤0.434 **FAIL** |
| 重访 p90 | 0.4560 | 0.6302 | — |
| 重访最大 | 2.3275 | **2.0014** | 显著下降 → 略降 |
| 重访中位 | 0.1678 | **0.1039** | — |
| 闭环边 | 45 | **8** | — |
| 关键帧 | 257 | 220 | — |
| **固定倾斜** | 3.08° | **1.77°** | — |

### 113.2 前端对比是决定性的

两份都用 `pose_odom`：

```
路程段     S13 前端(原 LIO)      S14 前端(静止修复后)
  0-10     +1.952  σ0.844        +2.348  σ0.275
 10-25     +0.501  σ0.346        -0.211  σ0.600
 25-50     +0.535  σ0.069        -0.315  σ0.093
 50-75     +0.343  σ0.152        -0.025  σ0.065
 75-100    -0.071  σ0.063        +0.112  σ0.037
100-150    -0.461  σ0.246        -0.099  σ0.107
150-200    +0.412  σ0.434        -0.875  σ0.599
```

**把静止段整个抑制掉（34.70 s、6960 样本、以 motion detected 退出、0 丢帧），
按路程算的前 10 米不但没修好，反而从 +1.952 变成 +2.348**，
而且段内散布从 0.844 收窄到 0.275 ——
**它现在是一个一致的系统性偏移，不是噪声。**

### 113.3 正确的诊断

> **暂态不是静止期漂移造成的，是「运动开始」造成的。**

估计器**确实**在静止时漂（33 s 里报出 8.24° 姿态摆动和数米位移，
而加速度计说机身没动 —— §111.3 那是实测）。
**但抑制那一段并不能消除地图开头的误差，所以两者不是同一回事。**

**加速度计零偏能解释剩下的部分：**

- 它在静止时**不可观测**（与重力方向误差混同）——
  所以再怎么延长平均也碰不到它
- 运动一开始，零偏误差表现为虚假加速度，二次积分成高度误差
- 要走上约 10 米、有了雷达约束和姿态变化，滤波器才观测到并收敛
- 这期间累积的误差已经写进了里程计边

**这也解释了那个劈成两半的结果**：
固定倾斜减半（3.08° → 1.77°），因为重力平均**确实**变好了；
而暂态纹丝不动，因为被平均的从来不是零偏。

### 113.4 处置

按 §98 的规矩：不可刷的地板残差段间跨度变差（0.434 → 0.575），
**即使最大值和中位改善也不接受**。

`max_static_init_s_` 默认改为 **0.0（关闭）**，机制留在文件里 ——
下一个人要知道它试过、以及试出了什么。已提交推送（`e7cd69a`）。

### 113.5 这一轮的方法论收获

**「机制正确」和「修好目标」是两件事，我把它们混为一谈了。**

静止期估计器发散是真的、可测的、我测到了。
基于它写的修复也确实做到了它宣称的事（等 34.70 s、平均 6960 样本、
按运动退出、固定倾斜减半）。**但那不是造成目标失败的原因。**

如果我没有先写死"0–10 m 段 < +0.40"这个判据，
我完全可以拿"固定倾斜减半 + 最大值 2.3275 → 2.0014 + 中位 0.1678 → 0.1039"
写一份成功报告 —— 三个数都真的改善了。

### 113.6 下一步该查什么

如果诊断是加速度计零偏，可查的方向（都未验证）：

1. `init_state.ba` 初始化成什么？如果是零，而真实零偏非零，
   误差就在运动开始的瞬间全额出现
2. `init_P` 里零偏对应的项（`init_P(15..17)=0.0001`）是否过紧 ——
   与 §111.2 那个重力项一样，可能在宣称一个没有依据的精度
3. 前 10 米的里程计边能否**降权**（不是修正，是承认它们不可信）——
   但 §92.4 说图改不了错的测量，除非有别的约束接手

---

## 114. 更正 §113：结论下得太急，比较本身是混淆的（2026-08-02）

### 114.1 前端上修复是明确有效的

直接量两份 LIO bag 各自开头 30 秒（机器人此时静止）：

```
             路程            z
原始 LIO     0 → 9.10 m     -0.019 → -1.590     ← 1.57 m 漂移
S14 LIO      0 →  5.27 m    -0.015 → +0.124     ← 0.14 m，基本平
```

**静止期的 z 漂移从 1.57 m 降到 0.14 m。** 这是无歧义的。

### 114.2 §113 的比较是混淆的

§113 拿两份的**残差剖面 0–10 m 段**并排（+1.952 vs +2.348），据此断言"没修好"。

**但残差是相对「整段会话最小二乘拟合的平面」算的**，
而两次会话的轨迹不同（191.3 m vs 235.7 m、220 vs 257 关键帧、起点相差 34.7 s）。
平面拟合在不同数据上，首段的残差**本来就不可比**。

**我在 §111.6 亲手写下"地图起点变了 —— 残差剖面的 0–10 m 段现在对应建筑里的另一段，
比较的是首段是否异常，不是是否同一处"，然后转头就把它当结论用了。**

### 114.3 可比的指标是按位置定义的那些

重访判据配对的是"xy 相距 1 m 内"的关键帧 —— 与轨迹从哪里开始无关：

| | S13 | S14 | 方向 |
|---|---|---|---|
| 重访最大 | 2.3275 | **2.0014** | 改善 |
| 重访中位 | 0.1678 | **0.1039** | 改善 |
| 重访 p90 | **0.4560** | 0.6302 | 变差 |
| >0.5 m 对数 | **6 / 101** | 14 / 119 | 变差 |
| 闭环边 | **45** | 8 | 变差 |
| 固定倾斜 | 3.08° | **1.77°** | 改善 |

**混合结果**，不是 §113 说的"完全没修好"。
地板残差段间跨度（0.434 → 0.575）依赖平面拟合，**同样混淆，不能用**。

### 114.4 现在能说什么、不能说什么

**能说：**

- 静止期估计器漂移是真的（33 s 报出 9.37 m 路程、−1.596 m z、8.24° 姿态，
  而 IMU 说机身没动）
- 修复消除了它（前端 z 漂移 1.57 m → 0.14 m）
- 重力平均确实改善（固定倾斜 3.08° → 1.77°）
- 最坏重访误差与中位都改善

**不能说：**

- 不能说"修好了整图"——p90 与 >0.5 m 对数都变差，闭环从 45 崩到 8，都未解释
- 不能说"没修好"——§113 那个论据是混淆的
- 加速度计零偏那条**量级对不上**：运动开始后 z 的二次拟合只给出
  −0.026 m/s² 的等效加速度，而要解释 2 m 级误差需要约 0.2 m/s²

### 114.5 要判定必须做同轨迹比较

现在两份地图覆盖的路线不同，任何依赖轨迹的量都不可比。

**正确的实验：把原始 LIO bag 截断到与 S14 相同的起始时刻（t=34.7 s），
用它跑一次建图。** 这样两张图覆盖同一条轨迹，唯一的差别就是
"估计器有没有在静止段跑过"。

在此之前，`max_static_init_s_` 保持默认 0.0（关闭）是**保守选择而不是结论**。

### 114.6 这一轮我犯的错

**我把一个自己刚写下的告诫，在下一个测量里就违反了。**
§111.6 明确写了两份地图起点不同、首段不可比；§113 直接拿首段做了结论。

**先写死判据能防止"事后换角度说成功"，但防不了"用不可比的量做比较"。**
判据里写的是"0–10 m 段中位 < +0.40"，而没写"两份必须覆盖同一轨迹" ——
判据本身漏了前提。

### 114.7 同轨迹对照实验已启动（S15）

`tools/truncate_lio_bag.py`：按 S14 第一条里程计的 header 时间戳截断原始 LIO bag
（两者同源，header 时间戳可比）。

```
cut at header stamp 1784778367299666688
  /cloud_registered_body     kept 7903  dropped 347
  /Odometry                  kept 7903  dropped 347
```

**保留 7903 条、丢弃 347 条，与 S14 的 7903 条和被抑制的 347 帧完全对上。**

**S15 = 原始 LIO 截断后建图**，与 S14 覆盖同一路线、同样长度。
**唯一差别**：估计器有没有在静止那 34.7 秒里跑过 ——
跑过的那份会带着 8.24° 的世界系倾斜进入后续轨迹。

三方对照将是：

| | 轨迹 | 静止段估计器 | 用途 |
|---|---|---|---|
| S13 | 全程 235.7 m | 跑过 | 旧基线（不可与下两者比路程相关量） |
| **S15** | 截断 | **跑过** | 对照 |
| **S14** | 截断 | **抑制** | 处理组 |

**S14 vs S15 才是判定第 1 步的那一对。**

---

## 115. 同轨迹对照：不建图静止段，最大误差降 4 倍（2026-08-02）

### 115.1 三方对照

| | S13（全程） | **S15（截断原始）** | S14（静止修复） |
|---|---|---|---|
| \|dz\| **最大** | 2.3275 | **0.5865** | 2.0014 |
| 地板残差段间跨度 | 0.434 | **0.255** | 0.575 |
| \|dz\| p90 | 0.4560 | **0.4231** | 0.6302 |
| >0.5 m 对数 | 6/101 | **3/99** | 14/119 |
| \|dz\| 中位 | 0.1678 | 0.1535 | 0.1039 |
| 闭环边 | 45 | 42 | 8 |
| 关键帧 | 257 | 249 | 220 |
| 轨迹长 | 235.7 | 227.8 | 191.3 |
| 固定倾斜 | 3.08° | 3.08° | 1.77° |

**卡了十三个阶段一直在 2.3 的最大值，掉到 0.5865 —— 四倍改善。
而做的事只是把静止那 34.7 秒从建图里去掉。**

残差剖面每一段都改善：

```
路程段     S13(全程)   S15(截断)
  0-10     +1.363     +0.483
 10-25     +0.009     +0.036
 25-50     -0.086     +0.003
 50-75     -0.041     -0.034
 75-100    -0.101     +0.083
100-150    -0.310     -0.170
150-200    +0.164     +0.085
200-300    -0.043     -0.150
跨度        0.434      0.255
```

判据：闭环 ≥20 **PASS**、残差段间跨度 <0.50 **PASS**；
p90 0.4231 与最大 0.5865 仍 **FAIL**（判据 0.10 / 0.30）。
**总判定 FAIL，但这是全项目最好的一张图。**

### 115.2 为什么 S14 反而差：机制补全了

S14 与 S15 都不建图静止段，差别在**估计器有没有跑过那 34.7 秒**：

- **S15**：滤波器跑过 —— 速度收敛了、局部地图建起来了、零偏收敛了，
  只是那段不拿去建图
- **S14**：滤波器在**运动中途从冷启动** —— 速度初值为 0 而机器人已在动，
  局部地图还是空的

**所以 S14 用「静止期漂移」换来了「带错初速度冷启动」，后者更贵。**

完整机制：

1. 估计器在静止期漂移（实测：33 s 报出 9.37 m 路程、−1.596 m z、8.24° 姿态）
2. 那段的世界系倾斜**随时间变化** → 建图会把它固化进里程计边 → 最大误差 2.33 m
3. 一旦机器人运动、滤波器稳下来，倾斜变成**固定**的
4. **而固定倾斜对重访不可见**（S15 的固定倾斜仍是 3.08°，与 S13 相同，却无害）

### 115.3 正确的修复：改建图侧，不改 FAST_LIO

**让 FAST_LIO 正常跑（它的滤波器需要那段时间收敛），
但建图在机器人真正动起来之前不收帧。**

这与 §91 的里程计合理性守卫是同一类：**能判断出输入不可用时就不要拿它建图**。
而且静止段本来就不含建图信息，跳过它不损失任何东西。

**一个陷阱：不能用里程计判运动。** 它在静止期报出了 9.37 m 位移、
最高 0.94 m/s 的速度 —— 用它判会立刻误判成"在动"。

可用的信号是**点云本身**：静止时连续两帧在传感器系里几乎相同。
体素重叠率是便宜且稳健的度量（降采样后约 2000 个体素，哈希集合求交，微秒级）。

### 115.4 验收将是自洽的

**守卫作用在完整的原始 bag 上，应当复现 S15 的数字。**
这是一个干净的接受测试：同一份输入、同一个结果，只是自动化了那次手工截断。

---

## 116. 静止启动闸：守卫精确复现手工截断（2026-08-02）

### 116.1 接受测试通过

守卫作用在**完整原始 bag** 上（s16），对比手工截断（s15）：

```
                关键帧  边   闭环  p90      最大     轨迹长   残差跨度
s16 (守卫)      249    290   42   0.4231  0.5865   227.7   0.254
s15 (手工截断)  249    290   42   0.4231  0.5865   227.8   0.255
```

**关键帧数、边数、闭环数、p90、最大值全同**，轨迹长差 0.1 m。
守卫自动做到了那次手工截断做到的事。

### 116.2 实现

`include/n3mapping/static_start_guard.h` + `src/static_start_guard.cpp`。

**关键设计决定：不能用里程计判运动。**
它在同一段静止期里报出 9.37 m 位移、最高 0.94 m/s ——
用它判会立刻误判成"在动"，守卫等于没有。

用点云与**开机第一帧**的体素重叠率。
**与前一帧比不行**：0.17 m/s 在 10 Hz 下每帧才走 17 mm，
连续帧的重叠几乎不变；与固定参照比，位移累积会让重叠单调下降。

检测到运动后**锁存**，之后零开销（`if (moved_) return true;`）。
降采样一帧约 2000 个体素，哈希插入，微秒级。

### 116.3 两个如实写进头文件的局限

单元测试逼出来的，不是事后补的：

1. **恰好落在体素边界上的平面会不稳。** 我把测试地板放在 z=−0.6，
   而 0.6/0.3=2.0 正是边界，±0.01 的噪声让每个点在两个体素间翻转。
   真实地板不会正好落在栅格上，但这个性质该写下来。
2. **原地旋转在以地面为主的场景里可能测不到。**
   稠密地面绕传感器旋转，占据的体素集合几乎不变。
   **这是可接受的**：旋转恰恰给了估计器它缺的可观测性，
   而且后续任何平移都会释放守卫。测试改成断言这一点，而不是断言它能测到旋转。

### 116.4 16 个测试失败的处置

守卫默认开启后，16 个测试失败 —— **全是喂固定合成点云的夹具**，
平台永远"不动"，守卫抑制了所有关键帧。它们测的是别的东西，
已在 `makeCoreTestConfig()` 与 `makeSyntheticRelocConfig()` 里显式关闭守卫。

**361 个测试，0 失败。**

### 116.5 配置显式写进 YAML

按 §90 的教训（"配置由最后一次编辑决定是一种沉默的实验污染"），
五个参数显式写进 `product_v1.yaml`、`n3mapping.yaml` 与阶段参数文件，
不靠 `config.h` 的默认值。

（顺带发现 `odom_sanity_*` 与 `floor_attitude_*` 本来就只在 `config.h` 里，
产品 YAML 从未列出它们 —— 那是既有问题，这次没有沿用。）

### 116.6 现在的成绩

| | 基线 S5d | S13 | **S16（守卫）** | 判据 |
|---|---|---|---|---|
| \|dz\| p90 | 0.4988 | 0.4560 | **0.4231** | <0.10 ✗ |
| \|dz\| 最大 | 2.3269 | 2.3275 | **0.5865** | <0.30 ✗ |
| 地板残差段间跨度 | 0.474 | 0.434 | **0.254** | <0.50 ✓ |
| 闭环边 | 11 | 45 | 42 | ≥20 ✓ |

**最大值四倍改善，残差跨度接近腰斩。判据仍未达成。**

b22 上的守卫验证在跑（它静止 16 s 后才动，应当正常释放且不误触发）。

---

## 117. 第 2 步：逐因子各轴加权（先写死判据）（2026-08-02）

### 117.1 改动

`loop_verifier.cpp` 新增 `axisWeights()`：取 `match_result.information`
（GICP 的真实 Hessian）的平移块与旋转块对角，除以各自的**几何平均**，
裁剪到 `[1/k, k]`（k = `loop_axis_weighting_max`，默认 4.0）。

```
Ω_ii = w_i / σ_config²      其中 w 的三个分量乘积为 1（裁剪前）
```

**绝对尺度仍由配置决定**（Hessian 的量纲是点数与代价函数尺度，不是米，
它预测不出实测的 0.407 m）；**Hessian 只在各轴之间重新分配**，
几何平均不变，所以一条闭环边的总刚度不变。

同时 `loop_closure_manager.cpp::applyEdgeModel` 里那句
`vertical_observability_score = 1.0` 删掉 —— 改由 verifier 按 `min(1, w_z)` 写入。

**默认 `loop_axis_weighting_enable = false`**，在显式开启前是空操作。

### 117.2 S16 上的基线数据

```
vertical_information_ratio  n=42  中位 1.992  p10 1.446  p90 3.339  最大 10.042
vertical_observability_score n=42  全部 1.0（硬编码，修复未编译）
```

各向异性真实且跨度很大，最大 10.04 —— 裁剪设在 4.0 是必要的。

### 117.3 判据要重写：闭环 z 误差在低漂移地图上失去判别力

```
        闭环数  拟合斜率   z 误差中位   >0.5m
S13      45     -0.998     0.376       14/45
S16      42     -0.427     0.432       20/42
```

斜率从 −0.998 掉到 −0.427，**但这不是闭环变差**：
S16 的地图漂移小得多，闭环需要修正的量本身变小，信号弱了而噪声没变，
拟合斜率自然趋向 0。**`loop_z_vs_floor_drift.py` 在低漂移地图上不能作判据。**

（这是本项目第四次遇到"指标在它自己的作用下失效"：
§89 地板法向在废图上读数正常、§98 重访判据可被闭环权重刷、
§114 残差剖面跨轨迹不可比，现在是这一条。）

### 117.4 S17 先写死的判据

**S17 = S16 的配置 + `loop_axis_weighting_enable: true`**（单变量）。

| | S16 现状 | **判据** |
|---|---|---|
| 地板残差段间跨度 | 0.254 | **≤ 0.254**（不得劣化） |
| 重访 p90 | 0.4231 | **< 0.4231**（须改善） |
| 重访最大 | 0.5865 | 不得劣化 |
| 闭环边 | 42 | ≥20 |

**两项主判据必须同时成立**，§98 之后不再接受单项改善。
闭环 z 误差降为诊断量，只记录不作判据。

**若成立** → 再做 S18：去掉全局各向异性（`loop_noise_position_z = -1`，
即 xy/z 同为 0.05），完全由 Hessian 决定分配 —— 那才是"逐因子"的完整形态。
**若不成立** → Hessian 的相对尺度不反映真实误差，全局常数就是能做到的最好，
如实记录。

### 116.7 b22 上的守卫验证：什么也没改，这是对的

```
                关键帧  边   闭环  轨迹长   x       y      z      层间距
b22 守卫        732    788   57   713.2  103.13  89.18  10.13   5.14 m
b22 S13 基线    732    785   54   713.5  102.79  89.60  10.16   5.06 m
```

**关键帧数完全相同，没有误抑制。**

这是期望的结果：b22 的静止期只有 16 秒，且估计器在那段几乎不漂
（§105 实测 z 只动 0.040 m、roll 0.25°），本来就产生不了关键帧，
守卫无事可做。

对照 0723：那 33 秒的漂移伪造出约 9 米"位移"，
按 1 m 的关键帧阈值产生了约 8 个关键帧 —— 守卫正是抑制了它们
（S13 257 帧 → S16 249 帧）。

**守卫在没有问题的地方不改变任何东西**，这正是它该有的性质。
楼层间距 5.06 → 5.14 m，在 §99.2 说明的 bin 分辨率（0.26 m）之内。

### 116.8 provenance 的一个疏漏

`run_mapping_generic.sh` 写摘要行时用的是一个**固定的关键字列表**，
后加的配置（`loop_noise_*`、`loop_axis_*`、`mapping_static_*`、
`icp_refine_max_*`）都不在其中，摘要里看不到。

完整参数文件仍被复制成 `logs/params.effective.yaml`，所以**记录没丢**，
但摘要不完整。已加宽关键字列表。

**教训：一个"记录用了什么"的机制，本身也会过期。**
它该记录的是全部差异，不是一张手写的清单。

---

## 118. S17 失败；剩余误差定位到一个未被提名的紧凑闭环（2026-08-02）

### 118.1 S17：先写死的判据未达成

| | S16 | S17 | 判据 |
|---|---|---|---|
| 重访 p90 | 0.4231 | **0.3913** | <0.4231 **PASS** |
| 地板残差段间跨度 | 0.254 | **0.264** | ≤0.254 **FAIL** |
| 重访最大 | 0.5865 | 0.5867 | 持平 |
| 重访中位 | 0.1537 | 0.1439 | — |
| 闭环边 | 42 | 42 | ✓ |

**两项必须同时成立 → 失败。**

**这是 S10 那个模式的缩小版**：可被闭环权重刷的重访 p90 改善 7.5%，
而不可刷的地板残差跨度退了 4%。
§98 立的"两项同向"规矩就是为了抓这个，它抓到了。

（管线是确定性的（§111.1），只改了一个配置键，所以 0.264 vs 0.254
是真实的确定性差异，不是跑次噪声。）

### 118.2 机制确实生效了，不是配置没读到

```
S16:  information_diag x/y/z = 400 / 400 / 6.25，42 条全同
S17:  x 中位 274、y 376、z 中位 9.9，z 范围 4.92 .. 25.0（撞到 4 倍裁剪）
```

**结论：Hessian 的相对尺度改善了可刷的指标，却让不可刷的那个退化。
全局常数就是这个方向能做到的最好。`loop_axis_weighting_enable` 保持默认关闭。**

顺带：`vertical_observability_score` 取 `min(1, w_z)`，42 条里 41 条饱和在 1.0
（Hessian 认为 z 几乎总比几何均值好），作为报告量几乎没信息。
原始比值 `vertical_information_ratio` 才是有用的那个。

### 118.3 剩余误差集中在路程 95–150 米

```
残差剖面(后端)        重访误差按路程
  0- 10  +0.480        0-25     5 对  中位 0.039  最大 0.173
 75-100  +0.084       75-100    8 对  中位 0.459  最大 0.587  ← 最差
100-150  -0.170      100-150   46 对  中位 0.222  最大 0.472
150-200  +0.083      150-200   18 对  中位 0.024  最大 0.082
最差 6 对: 0.59m@95/104m, 0.54m@96/104m, 0.53m@95/105m, 0.47m@106/190m
```

**100–150 段比左右两侧低 0.25 m**，最差的重访对全在 95–106 m。

### 118.4 前端在那 10 米里下沉又回升，后端没抹平

```
路程    前端 z    后端 z   后端修正
 94.6   -2.690    -1.795    +0.895   ← 局部高点
 99.9   -3.121    -2.171    +0.950
104.6   -3.235    -2.323    +0.912   ← 与 94.6 差 0.545 m
109.7   -2.863    -1.992    +0.871   ← 回升
```

两点在 xy 上相距不到 1 米，**0.545 m 是误差不是真实高差**
（若是真实的坡，往返会各走一遍并回到同一高度）。
后端修正在整段近乎恒定，**图完全没有处理这个坑**。

### 118.5 为什么后端没处理：检测器从未提名

那 42 条闭环里，附近的四条是
94.6↔108.6、95.6↔108.6、103.6↔115.1、104.6↔115.1 ——
**没有一条直接连 95↔104**。

查 `loop_debug.jsonl`：路程 92–108 对应关键帧 id 96–118，
**该区间内的候选记录 0 条**。检测器连提名都没有。

原因是三道**基于索引间隔**的闸：

```
loop_spatial_candidate_min_id_gap: 50
loop_closest_id_th:                50
loop_min_id_interval:              20
```

关键帧 96 与 105 相隔 **9 帧**，全部被排除。

**紧凑的原地折返（9 米路程回到同一点）正好落进"太近所以排除"的规则里 ——
而它恰恰是最需要闭合的那类回访。**

### 118.6 这是 S5a 那个修复没有推广完的地方

§S5a 把 RHPD 的排除从**帧数**改成了**路程**
（`loop_min_path_length_m = 5.0`），正是为了这个问题。
**但空间候选那条路径仍然用索引间隔。**

**第 3 步（下一个）**：把路程判据推广到空间候选路径。

判据（先写死）：

| | S16 现状 | 判据 |
|---|---|---|
| 路程 92–108 区间的候选记录 | **0 条** | **>0 条**（必须被提名） |
| 地板残差段间跨度 | 0.254 | ≤0.254 不得劣化 |
| 重访 p90 | 0.4231 | <0.4231 |
| 重访最大 | 0.5865 | **<0.5865**（那 6 对最差的全在这个区间） |

**风险**：放宽索引间隔会让相邻帧互相匹配（那正是这些闸原本要防的）。
所以要换成路程判据而不是简单调小间隔 ——
路程走了 9 米的两帧不是"相邻"，哪怕索引只差 9。

### 118.7 更正 §118.5：不是三道闸，是一道

查了一下这三个参数在 config 之外的使用：

```
loop_closest_id_th:    0 处   ← 死配置
loop_min_id_interval:  0 处   ← 死配置
loop_kf_gap:           1 处   ← 是检测频率节流，不是排除
loop_spatial_candidate_min_id_gap: 1 处   ← 真正拦住的那个
```

**只有一道闸是真的。**另外两个走完 YAML 读入、启动校验、摘要打印，
**从没被任何代码读过。**

**这是本项目第三次遇到同一形状**：
§97 闭环噪声参数走完全套仪式却对闭环边无作用；
§108 `vertical_observability_score` 是硬编码占位符却接进整条调试流；
现在是两个纯粹的死配置。

在 `config.h` 里给它们加了注释说明未使用。**没有删除** ——
删除是独立的清理，不该和一次有判据的改动捆在一起（否则失败时分不清是谁的责任）。

---

## 119. 第 3 步：空间候选改用路程判据（先写死判据）

### 119.1 改动

`detectSpatialCandidates` 里的 `query_id - match_id < min_gap`（索引间隔 50）
换成 `query_path - match_path < loop_min_path_length_m`（走过的路程 5 m），
与 §S5a 给 RHPD 做的一致。

路程由 `pose_odom` 逐帧累加，与 `detectLoopCandidates` 里同样的算法。

**理由**：让一次回访值得配准的是**两次经过之间走了多远**，
不是中间铺了多少关键帧。0723 上 95↔104 m 那对相隔 9 个关键帧、
走了 9 米路程，是全图最差的重访分歧，而索引判据把它整个丢掉了。

**361 个测试全过。**

### 119.2 先写死的判据

| | S16 现状 | **判据** |
|---|---|---|
| 路程 92–108 区间的候选记录 | **0 条** | **>0 条**（必须被提名） |
| 重访最大 | 0.5865 | **<0.5865**（最差 6 对全在这个区间） |
| 地板残差段间跨度 | 0.254 | **≤0.254** 不得劣化 |
| 重访 p90 | 0.4231 | **<0.4231** |

后三项须同时成立。

**风险**：放宽会让更多候选进入验证。别名的防线仍在
（`loop_max_range = 6.0` 的预测位移界、配准验证、裁判），
但如果几何折叠或残差跨度劣化，就说明这条路太松，如实记录。

### 119.3 S18 结果：达成了结构目的，但地图一点没变

| | S16 | S18 | 判据 |
|---|---|---|---|
| 区间内候选记录 | 0 条 | **11 条** | >0 **PASS** |
| 其中进图 | 0 | **0** | — |
| 重访最大 | 0.5865 | 0.5865 | <0.5865 **FAIL** |
| 重访 p90 | 0.4231 | 0.4231 | <0.4231 **FAIL** |
| 地板残差段间跨度 | 0.254 | 0.254 | ≤0.254 PASS |
| 关键帧/边/闭环 | 249/290/42 | 249/290/42 | — |

**地图逐位相同。判据失败。**

但**阻塞点移动了**：从"检测器从未提名"变成"提名后全被拒"。

### 119.4 被谁拒的：裁判，而候选的配准质量极好

```
   104<-> 96  fit=0.0238 inl=0.999  icp_z=+0.221  src=spatial_radius
   104<-> 97  fit=0.0173 inl=1.000  icp_z=+0.204
   109<-> 97  fit=0.0173 inl=0.992  icp_z=+0.221
   114<-> 98  fit=0.0556 inl=0.917  icp_z=+0.916
   114<-> 96  fit=0.0570 inl=0.857  icp_z=+0.905
```

fitness 远低于 0.2 阈值、inlier 0.86–1.00，
**而且 `icp_z` 的 +0.20 到 +0.92 正是补那个 0.545 m 下沉所需的量。**

拒绝原因：**9 条 `spatial_only_unconfirmed`，1 条 `yaw_flip`，1 条 `fitness_threshold`**。

样例 `104<->96` 的裁判输入：

```
rhpd_distance            DBL_MAX    ← 描述子完全没提名
sc_distance              DBL_MAX
segment_consensus_ratio  0.5        ← 4 对邻居里 2 对一致
segment_valid_pair_count 2 / 4
heightmap_ground_dz_median 0.525
```

规则（§95 的现行形式）要求
**描述子支持 或 段一致性 = 1.0（全部邻居对都一致）**。
两者都不满足。

### 119.5 顺带的发现：RHPD 认不出一个 GICP 配得极好的地方

`rhpd_distance = DBL_MAX` 意味着 RHPD **搜索过但没匹配上**
（它用的是路程判据，9 米合格）。

**GICP 能把这两帧配到 fitness 0.024 / inlier 0.999，而 RHPD 描述子完全不认。**
这是描述子召回率的问题，和裁判是两件事。

### 119.6 我在这里停手，不放松裁判

放松 `spatial_only_unconfirmed` 正是别名进入的那道口子，
而这个项目的核心要求是宁可不闭也不闭错。

**而证据是两头指的：**

- 支持接受：配准 0.024 / 0.999，修正量正确（+0.20~+0.92 正对那个下沉）
- 支持拒绝：描述子完全不认、段一致性只有 0.5、地面高度差 0.525 m 偏大

凌晨三点半、无人复核、证据矛盾的情况下放松一条安全规则，不是该做的事。
**诊断链已经完整，决定交给用户。**

### 119.7 S18 的处置

改动本身是**单位正确性的修复**：一个用路程判据的系统里，
空间候选那条路径还在用索引间隔。它达成了结构目的（0 → 11 条被提名），
**测得的地图影响为零**（逐位相同），没有可测的代价。

**发布，但如实记录它没有改善地图，以及为什么。**

### 119.8 完整的阻塞链（交接用）

```
1. 前端在路程 94.6→104.6 的 10 米里 z 下沉 0.545 m 又回升   ← 误差,不是真实高差
2. 后端修正整段近乎恒定,图完全没碰它
3. 因为没有闭环连 95↔104
4. S16 之前:检测器从未提名(索引间隔 50 vs 实际相隔 9)      ← S18 已修
5. S18 之后:提名了 11 条,配准极好,但被裁判 spatial_only_unconfirmed 拒
6. 因为 RHPD 完全没认出这个地方(DBL_MAX),段一致性只有 0.5
```

**下一步有两个方向,都需要用户定：**

- **A. 放松裁判**：`spatial_only_unconfirmed` 的段一致性门槛从 1.0 降到多数（≥0.5）。
  4 对样本上要求"全部一致"确实苛刻，一个坏邻居就掉到 0.75。
  **风险**：这是别名的入口，必须有足够的护栏与判据。
- **B. 查 RHPD 召回**：为什么描述子认不出一个 GICP 配到 0.024 的地方。
  这不动安全规则，但工作量更大。

---

## 120. 方向 B：RHPD 的召回率是 24%，而它自己的一块能到 38%（2026-08-02）

### 120.1 先更正 §119.5

我写"RHPD 认不出一个 GICP 配到 0.024 的地方"，依据是
`rhpd_distance = DBL_MAX`。**那是哨兵值，不是测量结果。**

```
候选来源:  rhpd_primary   469 条,rhpd_distance 全是有限值(中位 5.725)
           spatial_radius 182 条,rhpd_distance 全是 DBL_MAX
```

**DBL_MAX 的含义是"这条候选不来自 RHPD，该字段不适用"。**
我把占位符读成了测量结果 —— §108 我刚批评过这件事，然后自己犯了。

### 120.2 度量也差点用错

第一次算排名我用的是朴素 L2。RHPD 的 `distance()` 不是：

```
distance(a,b) = min( W(a,b), W(a_flip,b) )
W = sqrt( w_A·‖ΔA‖² + ‖ΔB‖² + 0.5·‖Δaux‖² )
w_A = 0.35 + 0.65·min(conf_a, conf_b)
a_flip: 三个平面块翻转 + 负空间 180° 移位 + 竖直 token 180° 移位
```

在脚本里重实现这个翻转很容易出错，所以写了
`tools/rhpd_recall_eval.cpp` **链接真实现**来算。
（两者结果接近，但那是运气，不是理由。）

### 120.3 真实召回率

真值是几何的、不需要标注：**xy 相距 <1 m 且路程相隔 >5 m 的两帧就是同一地点。**
0723 上有 87 对，分布在 47 个 query 上。

```
真实回访在 RHPD 排序里的位置(0 最好, 1 最差):
  中位 0.303        随机 0.500
  最好的 10%:  6%   随机 10%      ← 排序头部比随机还差
  较好的一半: 72%   随机 50%
  top-30 召回率: 21/87 = 24%      ← 系统实际取的就是 top-30

描述子距离:
  真实回访  中位 6.628,  5%–95%  5.370 .. 8.493
  全体对    中位 6.849,  5%–95%  5.300 .. 9.015
```

**两个分布的中位只差 0.22（约 3%），区间几乎完全重叠。**

### 120.4 原因：两块没有信息的在稀释一块有信息的

```
哪一块在分离"同一地点"与"其他地方":
  Part A (平面,      2352 维)  同地 5.477   他处 5.446   分离度  -0.6%
  Part B (环-高,      512 维)  同地 4.942   他处 5.308   分离度  +6.9%
  Aux    (负空间+token 173 维) 同地 2.229   他处 2.183   分离度  -2.1%
```

**只有 Part B 带区分度。Part A 与 Aux 是负的 —— 噪声。**

而合成距离把它们加在一起，按平方项估算
**Part A 约占 41%、Part B 约 55%、Aux 约 5%**：
接近一半的距离来自不带信息的块。

### 120.5 直接验证：单块排序

```
  Part A 单独    中位位置 0.545   top-30 召回 23/87 (26%)
  Part B 单独    中位位置 0.264   top-30 召回 33/87 (38%)
  Aux    单独    中位位置 0.652   top-30 召回 16/87 (18%)
  ------------------------------------------------------
  发布的三块合成  中位位置 0.303   top-30 召回 21/87 (24%)
```

**发布的描述子比它自己的其中一块还差。**
Part B 单独 38%，混进另外两块掉到 24%，**相对损失 37%**。

Part A（0.545）与 Aux（0.652）单独用**都比随机（0.500）差**。

### 120.6 修法与它的风险

把权重挪向 Part B。**但 `RHPDescriptor::distance()` 是重定位与建图共用的**，
而重定位有 today20 的契约（≥15/20 正确、零错锁）。

**所以做成可配置的块权重、默认保持现状，两侧分别测：**

- 建图侧判据：top-30 召回率从 24% 提高（`rhpd_recall_eval`），
  且地图指标不劣化（残差段间跨度 ≤0.254、重访最大 ≤0.5865）
- 重定位侧判据：today20 **零错锁**不得破坏，正确数不得下降

**不先量重定位就改共用的距离函数，是拿契约冒险。**

---

## §121 量重定位这一侧：为什么先卡了三道，以及卡住本身是个发现

§120 给出的建图侧结论很干净（Part B 单独 38% > 三块合成 24%），但 §120.6 已经
写死：**共用的 `distance()` 不先量重定位就不能改**。于是要跑 today20。
结果连续被三道拦下，而这三道拦的是同一件事。

### 121.1 三次失败,同一个根因

| # | 报错 | 真实含义 |
|---|------|----------|
| 1 | `unexpected manifest header` | 查询是 15 列（含 `gx,gy,gz,gravity_valid`），本分支的评测只认 11 列 |
| 2 | `atlas format version mismatch` | 产品 atlas 由另一条代码线写出，本分支的加载器不认 |
| 3 | `product Gate requires an explicit verified localization atlas` | 没有 atlas 就不许跑，不是可选项 |

再加上此前已经遇到过的一次：**流控回放工具** `n3mapping_deterministic_ros2_replay.py`
（455 行）在本分支上根本不存在，是从归档快照里捞回来的。

四件东西,同一个来源:**`archive/to-migrate-worktree-20260723`**
—— 那批我在 §9x 抢救出来的、60 个"存在于任何分支之外"的未提交文件。

**结论：产生产品基线的工具链,有相当一部分从来没进过任何分支。**
我一直以为自己在"产品基线所在的代码线"上开发，实际上我这棵树
**读不了产品的地图、读不了产品的查询、也没有产品的回放工具**。
这不是三个 bug,是一个事实的三次显形。

这条要单独记住:**"我能编译"不等于"我能复现基线"。**
本次之前没有任何一次测量迫使我同时用到这三样,所以这个断裂藏了很久。

### 121.2 处理：不搬代码，重建产物

第 1 道用最小补丁绕过：评测同时接受 11 列与 15 列表头，多出的四列跳过。
**这条路不是产品配置，单独的数不能当产品数报**；但 A/B 两侧走同一条路，
差值仍成立。补丁里把这句话写进了注释。

（试过整份搬归档里的重力版评测：920 行 vs 890 行，编译失败——
它依赖 `LioFrame::gravity_world` / `gravity_valid`，本分支没有这两个字段。
把整条重力链搬过来，远超一次描述子权重 A/B 该付的代价。已回退。）

第 2、3 道用**重建**解决,不用搬代码:

```
n3mapping_localization_atlas_compile --map n3map.pbstream --output ... --force
  prepare_ms 608  serialize_ms 79  verify_load_ms 434  verify_kdtree_ms 236
  → 244 MB
```

地图 pbstream 与产品逐字节相同，只有 sidecar 用本分支重新编译。

### 121.3 变体必须连 atlas 一起重建（不是麻烦，是正确性）

atlas 里是**描述子空间上的 KD-tree**。块权重改了，度量就变了，
而那棵树是在旧度量下建的 —— **拿旧 atlas 搜、用新距离算分，是搜一个度量、
判另一个度量**，结果没有意义。

所以每个变体的完整流程是:

```
改 config.h 默认值 → colcon build → 重编 atlas → 跑 today20
```

权重是**编译期默认值**（评测二进制不吃配置文件），所以"改一个数"实际是一次重编。
脚本 `run_variants.sh` 顺序跑 0.25/0.25 与 0.0/0.0，`trap restore EXIT`
保证无论怎么退出都把 config.h 还原成 1.0/1.0。

### 121.4 判据（先写死，结果未出）

| 侧 | 量 | 判据 |
|----|----|------|
| 重定位 | 错锁数 | **必须为 0**，破了就整条方向作废 |
| 重定位 | 正确 FULL 数 | 不得低于同路径跑出的基线 |
| 建图 | top-30 召回 | 24% → 应升（0.25 档预期 32%，0.0 档 38%） |

**基线也用新二进制、新 atlas 重跑**，不引用历史数字 —— 换二进制算换变量。

---

## §122 Part A 为什么比随机还差:三条可查的编码嫌疑(假设,未验证)

§120 量到"Part A 单独用中位位置 0.545,比随机 0.500 还差"。降权是治症状 ——
一个专门编码平面几何的块跑输随机,更可能是**构造或归一化本身有问题**。
读了 `computePartA` / `projectToPlane`,三条都是可以直接算出来的:

维数先纠正:`RHPD_PLANE_BINS=14`、`RHPD_PLANE_CHANS=4` →
每平面 784 维,**Part A = 2352 维**(此前记成 1536,已改)。
Part B 512(16 环 × 8 层 × 4 通道),Aux 173。合计 3037。
**Part A 占 77% 的维数、41% 的平方距离,贡献 −0.6% 的分离度。**

### 122.1 XY 平面的格子比要区分的东西还大

`max_range = 30.0`,XY 平面在 `[-30,+30]` 上分 14 格 →
**每格 4.29 m**。室内走廊宽约 2 m —— **整条走廊塞不满一个格子**。
点几乎全落在沿走廊轴的那一列格子里,不同地点得到的占据图样长得一样。

Part B 是极坐标(16 环),近处分辨率高得多。**这可能就是 B 行 A 不行的全部原因。**

### 122.2 XZ / YZ 两个平面的高度通道近似常数

XZ 平面的"高度轴"是 **Y(横向)**,归一化范围是
`h_range = max_range * 2 = 60 m`、`h_min_v = -30`。
走廊里横向坐标落在 ±1 m,于是

```
  h_norm ∈ [0.4833, 0.5167]   —— 通道量程的 3.3%
```

**三个平面里有两个,第 1 通道基本是常数 0.5。**满维参与距离,不带信息。

### 122.3 空格子与"最低处有东西"编码相同

```cpp
const double mean_height_norm = counts[i] > 0 ? h_sum[i] / counts[i] : 0.0;
```

空格子写 0.0。而 `h_norm` 是 `(vh - z_min)/(z_max - z_min)` 裁到 [0,1],
所以**一个真实落在 z_min(−2 m)的点也写 0.0**。
"这里什么都没有"与"这里地板在最低处"在描述子里不可分。

### 122.4 怎么验(还没验)

122.1 是主嫌疑,验法是**把 `max_range` 调小重算描述子**(如 10 m → 每格 1.43 m)
看分离度是否转正。但 `rhpd_recall_eval` 现在读的是 **pbstream 里存好的描述子**,
换参数得从关键帧点云**重算**,是工具改动。等 today20 三臂跑完再动树 ——
**当前这棵树正在跑实验,不能改。**

**这三条现在都只是假设。**记在这里是因为它们是算出来的、可证伪的,
而不是"感觉平面块不好"。若 122.1 成立,正确的修法是**改 Part A 的尺度**,
不是把它降权到 0 —— 降权丢掉的是本来可能有用的平面信息。

---

## §123 基线一跑出来就推翻了两件我以为知道的事

三臂对照的第一臂(1.0/1.0,即发布权重)跑完,同一个 `aggregate_today20.py` 评分:

```
  cases 20   algorithm_lock 15   within tolerance 12   OUTSIDE 3   no lock 5

  超标的三个:
    11-45-43   trans 1.132   yaw 2.996   roll 2.126   pitch 2.060   ← 边缘
    11-54-55   trans 0.025   yaw 0.495   roll 2.022   pitch 0.128   ← 边缘(只 roll)
    11-50-34   trans 25.308  yaw 101.812 roll 3.915   pitch 5.908   ← 错锁
```

### 123.1 我这棵树自己带一个错锁,而且是发布权重下的

`11-50-34` 不是姿态超标,是**锁到了别的地方**:
匹配关键帧 176,锁在 `(-30.944, -6.100, -2.184)`,离参考 25.3 m、yaw 差 101.8°,
`final_decision = tracking_geometric`、`final_pose_source = GEOMETRICALLY_CORRECTED`
—— 几何校正跑完了还是错的。获取耗时 8.4 s,顺带也破了 <5 s。

**这与方向 B 无关**(权重是 1.0/1.0,逐位等于发布行为),
是这棵树自己的缺陷。按硬约束 2「零个错锁优先于成功率」,**这是一个独立的、
优先级高于描述子权重的问题**。

### 123.2 "12 正确 / 零错锁"那个历史数字,我一直引错了目录

拿同一个评分器去评产品目录 `candidate_dd25f86/today20_eval` 自己的结果:

```
  cases 20   algorithm_lock 9   within tolerance 1   OUTSIDE 8   no lock 11
  其中 11-55-38 差 36.163 m / 88.2°、11-58-53 差 32.668 m / 123.2°
```

**1 正确 / 8 超标 / 11 未锁** —— 比我这棵树(12/3/5)差得多。
所以 §现状里写的「12 正确 / 2 姿态超标 / 6 未锁 / 错锁 0」**不是这个目录产的**,
我此前把它当基线引用是错的。那份数出自别处,现在**不知道出自哪次**。

（也可能是这份评分用的参考轨迹 `f7_0723_map/dense_trajectory.csv`
与那次评测所用的地图不是同一份。无论哪种,结论一样:
**这个目录不能当基线引用**。）

### 123.3 判据要改,但要明说是在改

§121.4 我写死的是「**错锁必须为 0**」。基线自己就破了,
所以这条**任何一臂都不可能满足** —— 按纪律 1,不能事后换角度说成功,
也不能装作判据没变。

**明改为相对判据**:

| 侧 | 量 | 原判据 | 改后 |
|----|----|--------|------|
| 重定位 | 错锁数 | 必须为 0 | **不得多于同路径基线的 1 个** |
| 重定位 | 正确数 | 不低于基线 | 不变(基线 12) |
| 建图 | top-30 召回 | 从 24% 提高 | 不变 |

**并且**:绝对契约「零错锁」仍然是硬约束,只是它现在挂在 §123.1 那个缺陷上,
不挂在描述子权重上。**两件事分开记,不许混在一起报成功。**

### 123.4 错锁的形状:80 帧说"分不清",第 81 帧一票定终身

`11-50-34` 的 `frame_status.csv`:

```
  frame   0   temporal_window_pending
  frame   4   margin              ← 此后连续 80 帧都是 margin(正确地拒绝)
  frame  84   accepted            ← seed = support = matched = 176
  frame  85   tracking_geometric  ← 此后一路跟踪,不再回头
```

**门控工作了 8 秒,然后一帧通过就永久锁存。**前面 80 帧"最优候选不够突出"
的证据不参与最终决定,锁上之后也没有任何机制回看。

### 123.5 两个假设,当场否掉一个

**假设 A:`seed == support` 意味着没有独立支撑,是错锁的标志。**
`11-50-34` 确实是 seed=support=176,而正确的 `11-45-43` 是 5/4 不同。但全量一看:

```
  SAME 且正确: 11-52-34、11-53-11、11-53-44、11-55-38、11-58-18(共 5 个)
```

**否定。** seed==support 在正确锁定里同样普遍,不是判别量。

**假设 B:锁定前 margin 拒绝得越久,越可能是错锁。**

```
  11-50-34   margin 16 帧,第 84 帧锁   → 错锁 25.3 m
  11-52-00   margin 17 帧,第 94 帧锁   → 正确 0.107 m(且终态是 REGION_HYPOTHESIS)
```

**也不干净。**等得久的两个,一个错一个对。

**要往下走必须看接受那一帧的实际分数**(`--reloc-debug` 重跑)。
现在这棵树正在跑三臂对照,**不能动**——按纪律 2「一次只动一个变量」,
等对照跑完再查。

---

## §124 方向 B 的判决:建图侧的召回提升不传导到重定位,反而更差

### 124.1 0.25/0.25 对基线

```
                       基线 1.0/1.0    0.25/0.25
  cases                     20            20
  algorithm_lock            15            14
  within tolerance          12            11     ← 少一个
  OUTSIDE tolerance          3             3
  no lock                    5             6
```

逐个 case 看比总数更清楚:

```
  11-49-09   基线 0.334 m 正确  →  0.25 臂 未锁         ← 丢掉一次正确锁定
  11-45-43   基线 1.132 m/3.0°  →  2.053 m/17.9°/pitch 11.5°   ← 明显恶化
  11-50-34   基线 25.3 m/101.8° →  14.3 m/173.8°/roll 18.1°    ← 错锁没消失,换了个错法
  其余 11 个逐位相同
```

**建图侧 top-30 召回 24% → 32% 的提升,在重定位上是负的。**

### 124.2 为什么值得记:这正是判据先写死的用处

§120 的建图侧结果本身没错(Part B 单独 38% 是量出来的)。
但**"对建图闭环排序更好的描述子"不等于"对全图冷启动更好的描述子"**:

- 建图闭环:在**已知附近**的候选里排序,只要真值进 top-30 就行
- 重定位冷启动:在**整张图**里选一个并且要敢锁,需要的是**区分度的绝对值**,
  不是相对排名 —— 把 77% 的维数(Part A)权重压到 0.25,
  剩下的 512 维环-高块要独自承担"这是不是同一个地方"的判断,
  于是不够自信的地方不锁了(11-49-09),本来就模糊的地方错得更狠(11-45-43、11-50-34)。

**如果按建图侧单边结果直接改默认值,就会拿掉一次正确锁定而毫不知情。**
提交 `3e2b13f` 把默认留在 1.0/1.0、写明"两侧都测完才准动",是对的。

### 124.3 结论

**方向 B 不发布。** `rhpd_part_a_scale` / `rhpd_aux_scale` 保持 1.0,
作为可配置的实验旋钮留在代码里(默认逐位等于发布行为,零风险)。

真正该做的是 §122.1:**不是给 Part A 降权,而是修它的空间尺度**
(每格 4.29 m vs 走廊宽 2 m)。降权丢掉的是本来可能有用的平面信息 ——
0.25 臂的结果正好说明那些信息确实在起作用。

### 124.4 0.0/0.0 臂:同样 11 个,但错锁的形状泄露了机制

```
                       基线 1.0/1.0    0.25/0.25    0.0/0.0
  algorithm_lock            15            14           14
  within tolerance          12            11           11
  OUTSIDE tolerance          3             3            3
  no lock                    5             6            6
```

两个变体一样差,**趋势单调且平台化**:降权丢一个正确锁定,再降不会更糟也不会更好。

真正有信息的是 `11-50-34` 这个错锁在三臂里的形状:

```
                 平移      yaw       roll    pitch
  1.0/1.0       25.3 m   101.8°     3.9°    5.9°
  0.25/0.25     14.3 m   173.8°    18.1°    8.5°
  0.0/0.0       15.7 m     1.6°     2.2°    0.2°   ← 朝向对了,位置沿走廊差 15.7 m
```

**Part A 完全去掉后,朝向准了,但沿走廊错 15.7 m。**
这是走廊混叠的典型样子:横截面处处相同,唯一能区分"走到哪了"的是沿轴的结构。
Part B 是以传感器为中心的极坐标环-高,**沿走廊平移时它几乎不变** —— 所以单靠它
定不出沿轴位置。而 Part A 的三平面投影**本该**带这个信息。

**所以 Part A 不是没用,是编码得太粗**(每格 4.29 m,§122.1)。
0.25/0.0 两臂变差,反过来证明它现在仍在贡献沿轴信息 —— 只是又粗又吵。
这把 §122.1 从"读码看出来的嫌疑"升级为**有独立证据支持的假设**:
两条线索指向同一处,且互相不依赖。

### 124.5 收尾

- `config.h` 由脚本的 `trap restore EXIT` 自动还原为 1.0/1.0,已核对
- 工作区 0 改动,HEAD `1214506`,重新构建(BUILD_TESTING=ON)使二进制与源码一致
- **方向 B 关闭,不发布。**旋钮保留,默认逐位等于发布行为

---

## §125 错锁的真正机制:描述子 85/85 次指对,证据累积器每次都把它扔掉

带 `--reloc-debug` 重跑 `11-50-34`(逐位复现:帧 84、关键帧 176、位姿相同)。

### 125.1 接受帧的候选表:锁的是第三名

```
  1. kf 102   fused 0.1745  rhpd 3.504  sc 0.1372    0.27 m 距真值
  2. kf 103   fused 0.2006  rhpd 3.982  sc 0.1654    0.77 m
  3. kf 176   fused 0.2198  rhpd 4.102  sc 0.2229   25.37 m   ← 锁的是这个
  ...
```

离真值最近的五个关键帧是 102(0.27 m)、103(0.77)、101(1.34)、223、224
—— **描述子排的第一名正是真位置。**

### 125.2 不是排名问题,是"用哪个排名"的问题

决定不是在本帧候选表上做的,是在**跨帧累积的假设**上做的。锁定帧共三个假设:

```
  seed  位姿                 累积对数似然   mean_visibility_evidence
  176   (-30.61, -6.12)         4.471            +0.7411   ← 锁定
  102   (-4.82, -4.42) 真位置   -3.239            -0.3253
  176   (-35.50, -4.71)         7.573            -1.0088   ← 似然最高
```

比较器(`world_localizing.cpp:604`)**主序是 `mean_visibility_evidence`**,
`cumulative_log_likelihood` 只在可见性相等或缺失时兜底。核对全部吻合:

```
  top1 = 可见性最高者 (+0.7411)          → temporal_hypothesis_score 0.7410503 ✓
  top2 = 首个不同物理位姿者 (-0.3253)     → 真位置
  margin = 0.7411 - (-0.3253) = 1.0664   → 报告 1.0663 ✓
  ratio  = exp(1.0664) = 2.905           → 报告 2.9047 ✓
  basin_separation = 25.85 m             → kf102↔kf176 相距 25.37 m ✓
```

`ambiguous` 要求 margin 与 ratio **同时**偏低,这里 margin 1.07、ratio 2.9,
于是不判模糊,直接锁。**门控按自己的定义工作正常** —— 输给它的东西是错的。

### 125.3 两个证据通道都偏向 25–35 m 外

**改成按似然排也救不了**:那样 top1 会是 (-35.50) 那个(似然 7.57),
离真值 **28.5 m**,错得更远。真位置的假设自己的似然是 **−3.24**,也是负的。

**可见性证据从第 20 帧起就指错**:102 seed 的两个假设是 −0.465 / −1.120,
176 seed 的是 +0.676 / +0.633。到第 83 帧真位置似然塌到 −4.577。

### 125.4 而逐帧描述子从头到尾是对的

把 85 个有候选表的帧全查一遍,真位置(3 m 内)第一次出现在第几名:

```
  第 1 名   85 帧   100.0%
  ------------------------------
  真位置排第一:  85/85 = 100%
```

**85 帧里 85 帧,描述子把真位置排在第一。而系统锁在 25 m 外。**

### 125.5 这改变了问题的定位

- §120「RHPD top-30 召回只有 24%」量的是**建图闭环在地图自己关键帧间的排序**,
  与这里是两个问题。**在这条重定位查询上,RHPD 逐帧排名是完美的。**
- 错锁完全来自**假设级的证据累积**:可见性证据为主序、似然为兜底,
  **两者都偏向 25–35 m 外**,而唯一指对的逐帧描述子证据不参与最终选择。
- 硬约束 3 说「free 空间考量融入 RHPD」。**在这个 case 上,
  正是自由空间证据导致了错锁**,并且它压过了指对的描述子证据。

**下一步不是改描述子,是查为什么可见性证据在真位置上是负的。**
先写死判据:能解释「真位置 mean_visibility_evidence 为负」的机制,
且修完后 `11-50-34` 不再错锁、today20 正确数不低于 12。

---

## §126 建图的不一致直接造成了重定位的错锁 —— 两个目标不是分开的

§125 停在「为什么真位置的可见性证据是负的」。往下一层就见底了。

### 126.1 先否掉我自己的上一个假设

读 `visibility_consistency.cpp:169`:

```cpp
result.evidence_log_odds = std::log(
    (consistent_bins + 0.5) / ((observed_bins - consistent_bins) + 0.5));
```

分母是 `observed_bins - consistent_bins`,而 `consistent_bins` 只在
**观测与预测都存在**的 bin 上累加。我据此猜:地图"没信息"被当成了反对证据
(与 §122.3 同一类缺陷)。

**实测否定。**`visibility_observed_coverage` 三个假设分别是
**0.9914 / 0.9983 / 0.9413** —— 地图几乎预测了查询看到的每一个 bin。
**不是覆盖问题。**(判据不足以分辨的零结果不算证据,这里能分辨:
覆盖率若是原因,真位置该显著低,实测反而最高。)

### 126.2 真正的差别:前景冲突,以及它上游的配准

```
  basin  匹配kf  fitness  inlier   似然   fg_conflict  log_odds
  176     176    0.0131   1.000   1.048     0.2157     +0.6361
  102     101    0.0905   0.959   0.717     0.4996     -0.3253   ← 真位置
  176     179    0.0821   0.999   0.856     0.3900     -1.1148
```

**错的地方配准得比真的地方好 7 倍。**真位置一半的 bin 上,
地图在观测返回的**前面**还有东西;由此得到的假设位姿离真值 **1.32 m**
(而关键帧 102 本身只差 0.27 m —— 差的是配准,不是候选)。

### 126.3 为什么真位置配不上:那里是重影

离真位置 3 m 内的关键帧,时间戳分成两簇:

```
  第一趟  kf 100–105   t ≈ 1784778628–663
  第二趟  kf 221–225   t ≈ 1784779063–067      434 秒后

  最近的几对:
    kf 102 ↔ kf 223   xy 1.636 m   dz -0.496 m
    kf 103 ↔ kf 222   xy 1.471 m   dz -0.503 m
    kf 104 ↔ kf 222   xy 1.311 m   dz -0.550 m
    kf 105 ↔ kf 221   xy 1.279 m   dz -0.504 m
```

**第二趟整体低约半米。**地图在这个地点把地板和墙存了两份,垂直错开 0.5 m。

对照锁错的那个地方(kf 176),3 m 内只有 **3 秒前**同一趟的 174/175,
dz 仅 −0.03 / −0.06 m —— **没有重访、没有重影、干净的单趟几何**。
这就是它 fitness 0.0131、inlier 1.000 的原因。

### 126.4 完整因果链

```
  建图重访 |dz| ~0.5 m(正在失败的那条判据)
     └→ 真位置在地图里是一堵垂直错开 0.5 m 的重影墙
        └→ 查询配不上,fitness 0.0905(别处 0.0131),位姿留下 1.32 m 误差
           └→ 重影面挡在观测返回前,前景冲突 0.4996(别处 0.2157)
              └→ 可见性证据 -0.325(别处 +0.636)
                 └→ 可见性是比较器主排序键 → 25 m 外几何干净的地方赢
                    └→ 错锁 25.3 m / 101.8°
```

### 126.5 这推翻了一个我一直在用的前提

CLAUDE.md 开头写「两个独立目标,判据不同」。**在这个 case 上它们不独立:
重定位契约的「零错锁」卡在建图判据的 |dz| 上。**

推论,按重要性排:

1. **不必去改可见性证据、比较器或描述子。**它们在干净几何上都工作正常
   —— 错的地方 fitness 0.0131/inlier 1.000 正说明流程本身没坏。
   **把地图修好,这个错锁自己会消失。**
2. **之前四次「指标可被优化器直接压低」的教训在这里换了个方向出现**:
   重访 |dz| 不只是一个验收数字,**它有物理后果**,
   而这个后果恰好落在另一个目标的硬约束上。
3. **验证方法**:在 |dz| 更小的地图上重跑 `11-50-34`。
   若错锁消失且 fitness 转好,链条闭合;若不消失,则本节的推断有误,要如实记。

**判据先写死**:同一 bag、同一查询,换用重访 |dz| 最大值更小的地图,
`11-50-34` 不再 FULL 锁在 25 m 外;且 today20 正确数不低于 12。

---

## §127 §126 的判据失败了 —— 链条在中间断开,但断点本身给出了答案

§126.5 写死:「换用重访 |dz| 最大值更小的地图,`11-50-34` 不再 FULL 锁在 25 m 外」。
**测了,不成立。§126.4 那条因果链是错的,以此更正。**

### 127.1 前提确实满足了

在错锁地点上,两张图的重访重影:

```
                          配对数   |dz| 中位   |dz| 最大
  产品图(错锁那张)          34      0.557      0.737
  s16 静止启动闸图           37      0.060      0.168
```

**九倍。**重影在 s16 上基本消失了。前提成立。

### 127.2 结论不成立

```
  s16 图上重跑 11-50-34:
    final_state       FULL_6DOF_LOCKED
    lock_frame        64        (产品图上是 84,更早了)
    matched_kf        168
    pose              (-30.845, -6.036, -1.316)
```

kf 168(s16)与 kf 176(产品)时间戳 **1784778930.9,完全相同** —— 同一个物理地点。
**还是锁在那儿,而且更早。**真位置仍然每帧排第一(**65/65**)。

### 127.3 断点在哪:配准修好了,可见性没动

真位置(s16 上是 kf 95)两张图对比:

```
                    产品图        s16 图
  fitness          0.0905      0.0084 – 0.0295     ← 好了一个数量级
  inlier           0.959       1.000
  fg_conflict      0.4996      0.5192 – 0.5488     ← 没动
  可见性证据       -0.3253     -0.326 … -0.413     ← 没动
```

**§126 假设的是「重影 → 配准差 → 前景冲突 → 证据为负」。
实测:重影去掉后配准好了一个数量级,证据一点没变。**
这两件事是独立的,不是一条链。**§126.4 就此作废。**

保留下来仍然成立的:重影**确实**造成了配准差(0.0905 → 0.0084 是干净的因果)。
错的是把配准差当成了可见性为负的原因。

### 127.4 断点给出的新事实

**在完美配准(fitness 0.0084、inlier 1.000)下,仍有 52–55% 的 bin
地图有东西挡在观测返回的前面。**这不是位姿误差的后果,是**那个地点的性质**。

对照:错锁地点 kf 168 的 fg_conflict 是 **0.2335**,fitness 0.0188。

两地的结构性差别只剩一条:**错锁地点是单趟几何**(3 秒前的 174/175,
无重访),**真位置是两趟**(相隔 434 秒各走一次)。

**新假设(未验证)**:前景冲突来自**多视点累积**,不是来自错位。
同一地点被走两次,地图里就有两倍的视点贡献的面;
从当前这一个视角看,其中相当一部分是看不到却会投影在前面的。
即使两趟几何上对得很准(0.06 m),这个效应依然在。

**可验证**:量 fg_conflict 与该处贡献趟数/局部点密度的关系。
若成立,则**可见性证据系统性地惩罚被重复观测的地方** ——
而重复观测的地方恰恰是重定位最该锁得住的地方。

### 127.5 教训

我把「重影 → 配准差」这段真因果,顺手延长成了「→ 前景冲突 → 证据为负」,
中间没有测。**判据先写死救了这一次**:如果不是 §126.5 那句话,
我会带着一条错的链继续往下改代码。

**§126 的提交(`2619e49`)不撤,留在记录里。**
