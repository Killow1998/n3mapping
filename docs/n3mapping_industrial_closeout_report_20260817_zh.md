# n3mapping 工业化验收与收尾报告 — 2026-08-17

记录时区：America/Los_Angeles（PDT，UTC-07:00）。

本报告针对已验证实现 `aa8352fda003eb72efb75399a043cbd62ca898ca`。后续仅提交本报告和证据文件，不改变该实现的代码树。

## 结论

- **Architecture-hardening：完成。** Roadmap 中 WP-00～WP-C4、SG-01～SG-09、PERF-ME-01 均已有 PASS、NO-GO 或 default-off 终态，没有遗留代码项或人工 bag gate。
- **当前 floor7 冻结流程：可以受控使用并进入维护冻结。** mapping、localization、map extension/resuming 在同一输入、同一节点、同一配置和同一主机上的 FA-01 为 PASS。
- **回环后端：通过 KITTI-360/M2DGR 自动化辅助验收。** 真值只用于离线 oracle，产品运行时不读取真值；不需要 RViz 人工打标签。
- **通用工业发布：仍是 NO-GO。** 缺的是部署硬件预算、真实 LIO 端到端公开数据集门、故障矩阵、大图扩展和远端 release 治理，不是继续堆算法功能。
- **建议：停止开放式功能开发。** 保持一个 `main`，只接受可复现 bug、兼容性、安全、发布和运维修复。

换句话说：项目已达到“冻结配置下的工程可交付候选”，尚未达到“任意地图、任意硬件、任意传感器前端均可承诺”的通用工业产品级别。

## 1. 最终可复现基线

| 项目 | 结果 |
| --- | --- |
| 验证实现 | `aa8352fda003eb72efb75399a043cbd62ca898ca` |
| 分支 | `main` |
| 工作树 | 验证前干净 |
| 构建 | 全 workspace、Release、research tools OFF，PASS |
| Product identity | commit 精确匹配，verified=true |
| CTest | 55/55 PASS |
| FA-01 | PASS，0 evidence / 0 performance / 0 quality issue |
| refs | 仅一个长期 branch：`main`；永久 tags：`archive/humble`、`archive/noetic` |

最终机器结果见[收尾结果](evidence/final_acceptance_20260817/result_aa8352f.json)与[完整 FA-01 verdict](evidence/final_acceptance_20260817/fa01_verdict_aa8352f.json)。

验证实现相对当时的 `origin/main`（`43aa502b`）领先 21 个本地提交。本报告提交后会再多一个 docs-only 提交；当前新代码尚未 push，因此不能把旧远端 CI 当作这 21 个提交的 CI 结果。

## 2. 三种模式的真实性能

以下 CPU/RSS 只统计 n3mapping 进程，不包含外部 FAST_LIO。资源仍是 report-only，因为尚未定义支持硬件、温度和功耗合同。

| 模式 | 旧 FA-01 callback p95 | 当前 p95 | 当前 >100 ms | 当前 CPU p95 | 当前 RSS p95 | 处理率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mapping | 0.337 ms | 0.330 ms | 0/621 | 0.020 cores | 82.5 MiB | 100% |
| Localization | 59.908 ms | 45.617 ms | 3/616，最长 2 帧 | 1.089 cores | 430.2 MiB | 100% |
| Map extension / resuming | 44.669 ms | 31.001 ms | 10/616，最长 1 帧 | 3.565 cores | 1415.1 MiB | 100% |

与旧正式基线相比：

- localization callback p95 降低约 23.9%，CPU p95 降低约 29.0%；代价是缓存使 RSS p95 增加约 130 MiB；
- map extension callback p95 降低约 30.6%，CPU p95 降低约 56.8%；RSS p95 增加约 110 MiB；
- mapping 本来就很轻，回环后台单次 work p95 为 37.06 ms、最大 66.48 ms，没有阻塞 100 ms 传感器预算。

### 2.1 Localization 为什么原来贵

根因不是图优化，而是 616 帧中反复为少量相同局部 anchor 构造多层 small_gicp `PreparedTarget`。当前使用按 map revision、anchor 和 submap range 键控的同步 LRU：

- 598 hits / 18 misses，hit rate 97.1%；
- 上限 128 MiB / 8 entries；实测峰值 123,249,280 bytes / 8 entries；
- target preparation p95 从 25.144 ms 降至 0.00323 ms；
- 616/616 tracking 成功，12 次 retry 与旧基线一致，逐帧分类和位姿未改变。

因此 localization 的主要剩余尾延迟来自真实 registration/retry，而不是重复 target build。

### 2.2 Resuming 为什么原来更贵

已确认有三个成本源：

1. loaded-map 邻域预取曾把已缓存或 pending 的近邻排除在 frontier 计数之外，导致越预取越远、缓存抖动；
2. 新 keyframe 提交后曾反复重建 loaded-map reference，并触发全局图相关工作；
3. 每帧严格 loaded-map tracking 仍需要 visibility、source/submap preparation 和 registration。

当前处理为：

- 三近邻 bounded prefetch，已缓存和 pending 项占用 frontier；后台线程总预算受控；
- loaded reference 保持不可变，写图只更新新 session；配对实验中 graph update mean 从 298.0 ms 降至 1.56 ms、p95 从 340.2 ms 降至 3.23 ms；
- 最终 exact-HEAD run 的 graph update mean/p95 为 3.44/4.48 ms；
- 606 hits / 10 misses，616/616 strict tracking 成功，0 retry/failure；
- 当前最大剩余 stage 是 visibility，p95 18.16 ms。

endpoint-first visibility fast path 的确更快，但会改变大量 dense pose 和一个 anchor 选择，独立 session 与错误锁定安全面没有被覆盖，因此保持 **default OFF**。这比为了跑分直接切默认更符合产品边界。

## 3. 是否应改成 Cartographer 式两层图

当前架构已经具备部分两层结构：

```text
FAST_LIO odometry/cloud
        ↓
frame → local keyframe-neighborhood/submap registration
        ↓
keyframe constraints → global keyframe graph
        ↓
SubmapBuilder + shadow global-submap graph（无 writeback）
```

它不是完整的 Cartographer active-submap 权威架构。SG-01～SG-09 已把 submap ownership、pose projection、factor、隔离优化和对照工具建好，但仍是 shadow-only：真实有环图会让 keyframe/地图产生约 5 cm 变化，内部 objective 下降不能证明物理地图更正确，外部真值也没有证明 writeback 有收益。因此当前结论是：

- **保留现行局部 registration + keyframe graph；**
- **保留 submap graph 作为诊断，不接管输出；**
- **不为“架构看起来更先进”再重写一次后端。**

只有当更大地图或目标硬件明确使当前方案失败时，才应开一个独立项目：先冻结 submap 边界和外部 GT，再做 offline authoritative graph，对轨迹和几何设预注册阈值，最后才讨论 runtime writeback。

## 4. 回环是否处理好

### 4.1 当前真实同输入

最终 mapping 处理 621 帧、23 个 keyframe、31 个检测候选，接受 3 条 loop：`7→1`、`12→7`、`22→17`。保存图为 23 KF / 22 odom / 3 loop，0 duplicate、0 dangling edge。

旧 FA-01 参考要求 4 条 loop。候选级回放证明旧第 4 条是 `7→0`：它与同 query 的 `7→1` 推导出的全局 query pose 相差 2.043 m / 0.424 rad，超过预注册的 1 m / 10° 一致性界限。新规则保留历史排序更好的 `7→1`，拒绝无一致支持的 `7→0`。因此将参考从 4 条改为 3 条是清除过时 oracle，不是降低质量门槛。证据见[FA-01 参考修订](evidence/fa01_quality_reference_revision_20260817/result_bf60e2f.json)。

### 4.2 随机抖动真值 Odom 是否可行

可行，但必须把它定位为“后端鲁棒性试验”，不能冒充真实前端验收。本轮已经做了比逐帧白噪声更接近真实漂移的、可复现的空间相关扰动，固定 seeds 17、73、211：

- 3/3 正样本通过；
- 80/80 已接受 loop 正确；
- 0 catastrophic false loop；
- 一个 control case 保持 0 loop。

这证明同一 query consensus 修复能够承受冻结的相关漂移模型。它不能证明 FAST_LIO 的真实错误分布，因为人工扰动没有覆盖 IMU bias、时间同步、deskew、外参误差、点云遮挡、运动畸变、dropout 和前端失锁后的相关耦合。正确的敏捷做法不是拒绝 synthetic test，而是：

1. 用 synthetic correlated drift 快速、自动、可复现地测后端；
2. 用少量真实 LIO 端到端数据测系统边界；
3. 两者不互相冒充。

详细结果见[FA-02B](evidence/fa02b_correlated_drift_20260815/result_96123e2.json)。

## 5. KITTI-360 与 M2DGR 到底证明了什么

当前正式自动 gate 使用 `GT pose + LiDAR` 驱动 n3mapping 后端，GT 只在离线 evaluator 中评分。应用同一安全修复后的 clean matrix 结果为：

- KITTI-360 drive 0005：900 帧，23/23 loop 正确，ATE translation RMSE/p95 0.278/0.645 m，rotation RMSE 0.062°；
- KITTI-360 drive 0003 control：900 帧，0 loop；
- M2DGR gate_02 low-overlap control：600 帧，0 loop；
- M2DGR hall_05：500 帧，73 条 position-only loop 中 71 条 place-consistent，ATE translation RMSE/p95 0.106/0.187 m；
- 总计 0 catastrophic false loop，3/3 revisit segments 命中。

这足以关闭“后端回环能否自动验收”的问题，但不能关闭“传感器到最终地图是否端到端工业合格”的问题。

早期公开数据集重定位辅助实验不是正式 gate：KITTI cross-drive 只有 3/5 正确锁定，M2DGR gate_02 为 1/5，hall_05 为 0/3；M2DGR LiDAR→IMU、真实 frontend 配置和 odometry 也未冻结。它们应保留为 fail-closed 诊断，不能被包装成 localization PASS。原始边界记录见[辅助数据集结果](evidence/industrial_closeout_20260814/auxiliary_dataset_results.json)。

## 6. 地图与跨 session 结论

最终 map extension：

- 616/616 strict loaded-map tracking 成功；
- 新增 24 个 keyframe；
- 保存图 241 KF / 263 edges / 239 odom / 23 loop / 1 session anchor / dense 8867；
- 0 duplicate keyframe、0 dangling edge；
- 217 个 loaded keyframe 最大位姿变化 0.0299 m / 0.00091 rad，低于冻结上限 0.065 m / 0.002 rad。

这证明冻结同路线下的 map extension 正确性和旧地图稳定性。它不等于“任意跨 session 自动 merge/federation 已认证”。自动发现任意 session、冲突事务、批量地图合并和全局 rollback 仍属于另一个产品阶段，本轮明确不实现。

## 7. 相比原远端 main 到底改了多少

验证实现相对 `origin/main`：

- 21 commits；
- 50 files；
- 总计 +4747 / -182 行；
- 产品 source +781 / -89；
- tests +854 / -8；
- docs/evidence +1563 / -7；
- tools/config 等 +1549 / -78。

有效改动主要分为：

1. 同 query 回环一致性和 FA-02B 相关漂移 gate；
2. loaded-map prefetch 抖动修复与 initial warmup；
3. ordinary localization bounded PreparedTarget LRU；
4. loaded-map visibility shadow/default-off trial；
5. immutable loaded-map tracking reference 和完整 FA-01 instrumentation/evidence；
6. 质量参考修订。

中间尝试过多种 iSAM2 增量更新方案，但最终同输入证据没有证明它解决主要瓶颈，因此产品实现已恢复原 transactional graph update；`src/graph_optimizer.cpp` 相对 `origin/main` 的净 diff 为 0，只保留“失败事务后下一次合法事务仍可用”的 8 行回归测试。也就是说，试验历史存在，但未把未证明的代码留在最终产品树。

旧 deepseek/recovery/realtime 分支语义已审计并统一到 `main`。当前 branch 列表只剩 `main`，历史通过 main ancestry 可恢复；永久 tag 只保留 `archive/humble` 和 `archive/noetic`。详见[分支退役审计](branch_retirement_review_20260814.md)。

## 8. 工业验收矩阵

| 验收面 | 当前结论 | 说明 |
| --- | --- | --- |
| Architecture-hardening | PASS | 所有 WP 有终态，NO-GO 未越权启用 |
| Humble/Jammy 本机 Product build | PASS | exact commit，55/55 CTest |
| 三模式同输入性能与质量 | PASS | 0 issue，100% 输入处理 |
| floor7 冻结流程 | GO | 受控配置和证据边界内 |
| 后端回环 | PASS | KITTI-360/M2DGR + correlated odom 自动 gate |
| 当前代码的远端 CI / Noetic-Focal | 未完成 | 21 个提交尚未 push，旧 CI 不能代替 |
| 公开数据集真实 LIO 端到端 | 未建立 | 当前正式 gate 以 GT pose 驱动后端 |
| 任意跨 session federation | 未认证 | 不在冻结产品范围 |
| 部署硬件 CPU/RSS/温度合同 | 未建立 | 当前资源只 report-only |
| >1 GiB 大图 | 未解决 | 已观察到 atlas 超出 loader contract |
| restart/power-loss/disk-full/corrupt-map | 未认证 | 缺长期故障矩阵 |
| versioned artifact / rollback bundle | 未完成 | 远端 release 治理尚未执行 |

## 9. 推荐收尾方式

现在应进入 maintenance freeze，不再继续 architecture-hardening 或 Cartographer 化重写。若只需要当前 floor7 产品，可在冻结配置、地图和硬件上使用；本轮不再需要人工 RViz/bag 检验。

如果以后要求“通用工业发布”，另开一个有明确终点的 release 项目，只做以下五件事：

1. 声明支持硬件，冻结各模式 CPU/RSS/p95/max-consecutive/温度预算；
2. 冻结一个真实 LiDAR/IMU frontend，把 KITTI-360/M2DGR GT 仅作为 evaluator oracle，补端到端 gate；
3. 跑长时、重启、断电、磁盘满、坏 map、OOM/资源耗尽矩阵；
4. 解决或明确拒绝 >1 GiB 地图的分片/加载产品合同；
5. push 当前 main，跑 Humble/Noetic required CI，保护 release ref，发布可复现 binary/config/map/rollback bundle。

在这五项没有被正式提出前，继续调阈值、启用 C2/C3、写回 submap graph 或实现任意跨 session federation 都会增加维护风险，而不会让当前 floor7 产品更接近真实验收。
