# 实机拉取、测试与问题反馈

## SpRobo live extension — 2026-10-27

按 Xian 要求，使用实际 SpRobo 原生仿真与 ROS 输入，先采集局部场景，再重新启动完整 SLAM 会话续建。
使用既有 `building-planar-floors-ceilings` 双层大场景（701 个对象），本轮只走下层走廊拐角；
固定几何不变，仅选取已验证路线上的出生点。初始扫描实际看到的点全部保留，没有裁掉远处可见几何。
真值只用于采集路线控制和评分，订阅清单确认未进入 FAST-LIO/N3M。

运行版本：N3M **`ef05c1e`** 的独立冻结 node/core/wrapper，配套修复后的 GTSAM；
FAST-LIO 使用此前已验证的 `1b3c0b4` 独立构建；不修改任何算法或仿真程序。
全流程 affinity `0,2,4,6`，独立 ROS 域 224，Fast DDS 共享内存段 64 MiB。
两次运行均完整记录原始 LiDAR/IMU/真值、前端点云/里程计和 N3M 输出；所有自有进程正常退出。

| 项目 | 初始局部建图 | 续建并返回 |
|---|---:|---:|
| 路线 | 计划 7.52 m 往返 | 计划 42.73 m，实际 XY 42.41 m |
| 仿真时间 | 38.27 s | 171.08 s |
| 原始 LiDAR / IMU | 382 / 7,653 | 1,710 / 34,215 |
| 保存关键帧 | 33 | 225 |
| 0.1 m 体素导出点数 | 25,101 | 67,666 |
| 在线位置误差 RMSE | 0.60 cm | 1.53 cm |
| 在线位置误差 P95 / 最大 | 0.90 / 0.97 cm | 3.29 / 10.22 cm |
| N3M 运行期平均 CPU | 0.048 核 | 0.278 核 |
| N3M 1 秒采样 CPU 峰值 | 0.569 核 | 0.600 核 |
| N3M 采样 RSS 峰值 | 155.0 MiB | 221.8 MiB |

CPU/RAM 是 N3M 进程的运行期采样，不含仿真和 FAST-LIO，也不包含最后保存/退出阶段；
1 秒采样不能约束更短的瞬时峰值。这一局部案例不代表完整大图、所有困难搜索或 RK3588 的性能验收。

续建只有一次初始重定位和一条 SESSION_ANCHOR。新增 192 个关键帧中，184 个取得原图跟踪约束，
7 个只通过会话里程计继续建图，另外 1 个是会话锚定帧；没有为失锁预测伪造原图回环。
91.62–92.03 s 出现短暂 LOST，随后恢复，之后未回到 initializing。末帧位置误差 4.64 mm。
原始 33 帧点云和里程计保持一致（四元数数值往返差不超过 2.9e-17 rad），
这些帧的优化位置最大变化 1.56 mm。新增地图中 16,934 个点距原图最近点超过 0.2 m。

原生 N3M 加载器重载两图后，关键帧平移和点云差为 0，旋转差为浮点舍入量；两份 Atlas 的地图 SHA 和关键帧数匹配。
续建评分始终使用第一轮建图首个 N3M 输出确定的固定对齐，不对续建轨迹重新拟合。
在线 1,681 帧只有 1 帧超过 10 cm，峰值在 90.60 s；保存轨迹 RMSE 1.42 cm、P95 2.88 cm、最大 10.68 cm，也只有 1 帧超过 10 cm。
**Xian 已确认这次约 0.06% 的小幅孤立超限可以接受：本轮按整体稳定性通过，不为该样本继续优化。**
原始统计保留；后续关注持续偏差、明显误定位和地图稳定性。

证据目录：工作空间 `build/n3mapping/sprobo_extension_20261027/`。
`qualification.json` 汇总版本、完整性、精度、资源和 SHA；`map_analysis.json` 记录图与重载检查；
`map_comparison.png` 为同一坐标系、同一视窗的观测点云对比。
`partial01/`、`extension01/` 各自保留关闭后的 `bag/`、原始仿真记录、`map/n3map.pbstream`、Atlas 与 `map/global_map.pcd`。
本轮完成局部 SpRobo 实采续建回归，尚未替代原 Go2/MID360 现场故障包、跨楼层或全场景验收。

## Map extension doorway — 2026-10-27

[Issue #7](https://github.com/Killow1998/n3mapping/issues/7) 的本地修复恢复了已有续建路径。
此前核心入口在原图跟踪失败时提前返回，底层已支持的会话里程计链没有机会追加关键帧；
公共状态输出又把续建失锁归入 `initializing`。两处问题均在共用核心中处理，ROS 适配层不增加策略。

| 阶段 | 对外质量 | 图与会话行为 |
|---|---|---|
| 尚未初次重定位 | `initializing` | 不追加新会话关键帧 |
| 原图几何跟踪通过 | `tracking / GEOMETRICALLY_CORRECTED` | 使用现有 SESSION_ANCHOR / 原图跟踪约束 |
| 已锚定，原图跟踪失败 | `degraded` 或 `lost`，`ODOM_PREDICTED` | 保留会话，沿连续 LIO 添加局部关键帧；不制造原图回环 |
| 返回原图 | 几何门槛通过后恢复 `tracking` | 恢复原图约束，不再重复初始化 |
| 输入异常或已声明的来源坐标系改变 | `error` | 拒绝写图，保留已提交内容 |

预测阶段的 `success` 表示局部建图处理成功，不代表获得全局定位权威。
定位状态仍暴露失锁，实时输出不产生新的全局锁定事件。原图配准门槛、检索算法和异步几何优化算法均未放宽。
真正更换 LIO 坐标系时，应保存当前会话并以健康前端重新启动／加载地图，不能把旧里程计边直接接到新原点。
同名坐标系内未报告的复位、缓慢漂移不能仅靠速度体检识别；本轮不声称解决了所有前端失效。

收尾保持现有模块职责：共用建图输入检查；合并续建输出字段处理，续建文件由 198 行减至 188 行。
不新增生产策略开关、工作线程、候选管线或保存格式。
既有速度体检现在暴露当前违规计数，已判异常的样本在发散锁存阈值之前也不能入图。
三个定位夹具曾用一秒内移动 20–120 m 来构造不同地点，已改为合理时间间隔；定位断言和门槛保持不变。
保留有行为约束的旧测试，增加独立窄门路线 E2E，并在实时输出序列中验证失锁和恢复，避免堆叠只核对字段赋值的测试。

四核 Release 验证：CPU affinity `0,2,4,6`，ccache，独立 build/install，完整 **68/68**（41.91 s），含 Humble 节点 E2E。
新增核心 E2E 使用两个房间、1.2 m 门洞、带遮挡的射线观测、默认定位门槛，覆盖旧区→新房间→旧区。
旧地图保留初始位置实际观测到的全部返回，不生成未观测表面。
验证关键帧和非关键帧、持续失锁、只有一次初始化、预测帧不写原图回环、返回后的原图约束、异步收敛、保存重载和异常输入。
这是一条无噪声合成路线，误差约 1 mm；不代表真实场景 10 cm 已验收。

消融采用同一输入，独立测试进程加载替代实现，正式核心库不被改写：

| 对照 | 结果 |
|---|---|
| 冻结基线 `2d1917d` | 新区预测期间新增关键帧 0，路线断言失败 |
| 仅恢复旧的失锁提前返回 | 同样无法在新区续建 |
| 仅移除发散锁存前的异常输入拦截 | 异常样本被提交，输入可靠性断言失败 |
| 完整修复 | 新区预测期间新增关键帧 17，返回原图成功，只有一次初始化；全部断言通过 |

证据目录：工作空间 `build/n3mapping/map_extension_20261027/`，含 `ctest.log`、各对照 XML、资源采样、消融编译命令及 `qualification.json`。
其中 `final/` 保存核心库与路线执行文件的指纹；当前构建身份为 **UNVERIFIED 开发候选**，不是新的正式 RC。
使用配套 `reliability_precision_20261026/dependency/gtsam_install` 依赖；原导航冻结库、已安装 rc3、FAST-LIO、small_gicp 与旧标签不变。
没有重跑原有 14 点、两条定位路线或 Noetic 原生运行；本轮没有修改重定位算法，也不恢复已暂停的性能优化。

限制必须保留：原 Go2/MID360 事件没有原始 bag、独立真值与逐帧可见性，尚不能宣称原实机事件已验收。
调试时另试了人为裁掉旧扫描中门后已观测返回的参考图，基线与候选均出现约 3 m 的错误配准。
该对照记录保留在 `baseline_route.log`、`route_debug.log` 和 `route_tracking.jsonl`；
它提示不完整参考图仍有既有误配准风险，本轮没有把这一问题标为已解决，也没有降低门槛使其通过。
下一步应使用现场原始输入复测上述进出路线，并单独处理缺失几何的参考图风险。

## 大场景建图崩溃修订（2026-10-26）

Humble 后续修订 **`111b1ac`** 使用独立 Release 安装及带官方竞争修复的 GTSAM。
导航接入请按[后续验证与接入记录](evidence/large_mapping_followup_20261026.md#integration-boundary)
选择配套前缀，并核对节点、核心库和依赖身份。原 rc3、导航冻结候选和旧标签均保留。
已复现并修复一处 GTSAM 消元竞争；尚未证明它是原 SIGSEGV 的唯一原因。
新修订核对实际加载库的 build ID。身份不匹配时保留图内串行保护，不静默恢复并行。
记录列明并行化的 CPU/RAM 代价、保存等待和仍未解决的精度问题；不能据此认定 10 cm 达标。
同一原始采图的后续对照已证实启动条件对漂移有明显影响。经 Xian 授权，FAST-LIO fork
已加入最近稳定窗口检查、配置重力一致性和去畸变起始状态修复。它要求静止启动，
不能把固定等待一秒或延后 N3M 输出视为通用修复。健康启动后，
在线后端仍有超过 10 cm 的持续误差；最终保存轨迹不能替代在线验收。
详见[启动与前缀消融证据](evidence/large_mapping_followup_20261026.md#raw-startup-isolation-and-online-accuracy-follow-up)。
前端修订 `1b3c0b4` 使用独立构建。原 FAST-LIO 安装和导航冻结库未替换；N3M 算法仍为
`111b1ac`。详见[前端修复验证](evidence/large_mapping_followup_20261026.md#frontend-initialization-repair-in-the-authorized-fork)。
上一版 **`5e9bf8b`** 的[串行隔离验证](evidence/large_mapping_crash_20261025.md)继续保留。
以下 rc3 章节保留原版本的构建与验收范围；本轮未生成新的 Noetic 构建。

## 使用范围与收尾原则

当前主线是 CPU 优先的外部 LIO 后端：接收去畸变的 body-frame 点云和
odometry，提供 mapping、localization、map_extension。现有受控验收不代表
任意传感器、地图和机器都已通过工业验收，详见
[收尾报告](n3mapping_industrial_closeout_report_20260817_zh.md)。

后续优先修复可复现的正确性、资源、兼容性和地图安全问题，不因新论文扩展算法。
本流程不改变默认匹配、锁定或回环规则，也不自动启动前端、播放 bag 或设置 ROS_DOMAIN_ID。

以下实机运行命令针对 **Ubuntu 22.04 / ROS 2 Humble**。日志打包命令只依赖
Python 标准库，不要求 ROS 正在运行；也可以打包按同样目录布局保存的 Noetic 文本日志。
`prepare` 生成的是 ROS 2 参数文件，不能直接交给 ROS 1 使用。

本次本地实机候选为 **v1.1.0-rc3**，运行代码变更止于 `ea0ced6`；候选还包括
此后的历史合并与验收文档。两套 ROS 的 package 版本仍为 **1.1.0**。
Humble 和 Noetic/Focal Docker 使用独立 Release 构建；身份须匹配所用候选提交，
且 `verified=true`。保留 **v1.1.0-rc1 / v1.1.0-rc2**，未推送；
由 Xian 检阅后决定是否推送。构建身份确认不等于真机验收。

## 收尾要求核对（2026-10-23）

| Xian 的要求 | 结果与验收范围 |
| --- | --- |
| 冻结功能，保持算法与可靠性门槛 | 完成。此次移动的 43 个成员函数体一致；地图格式、算法配置和门槛不变。 |
| 单一职责、缩短文件 | 完成。核心入口 762→303 行；建图、续建、加载、保存分别维护。序列化 833 行拆为加载 401、保存 416、PCD 导出 33 行。没有新增状态持有者。 |
| 两套 ROS 薄封装、职责对齐 | 完成。节点按相同职责分文件，共用核心、RealtimeOutput、MapArtifacts 和日志模块。传输 API 与 Humble 专用诊断探针仍有差别。 |
| 减少模块编译依赖 | 完成。序列化头文件不再导入图优化、回环检测和生成 protobuf 头；PCD 导出编译依赖中也已排除这些头。 |
| 消融冗余抽象、设计和测试 | 完成此次审计。移除旧库别名、9 处无用 ROS 测试初始化和两项赋值读回测试。核心测试统一注册；两套 ROS 均有 42 项，Noetic 补齐 16 项。保留真实状态、事务与故障断言。 |
| 开发与发布隔离、ccache、四核、无硬编码 O3 | 完成。rc3 使用独立 colcon/catkin 构建目录；Release 的 O3 来自 CMake 配置。既有开发目录和缓存保留。容器构建使用源码及缓存的所属用户。 |
| 共用异步日志，跨重启总额 100 MB | 已实现并复验。有限队列、批量写入、关键事件、写失败和溢出统计；按同一实例目录管理当前与轮转日志。前端及外部启动器的日志不属于 n3m 实例。 |
| E2E 确认完整性 | Release：Humble 66/66，Noetic 64/64。包含地图/Atlas 事务、恢复锁定、异步回环与两套真实节点 E2E；真实节点覆盖持续启动输入、无效输入恢复、保存、写入故障和重启。 |
| 分支整合 | 完成。Noetic 模块化及旧 issue #6 分支均进入 main；issue #6 补丁已等价应用，仅补齐历史，源码树未变。旧分支名称保留，没有未整合的独有代码。 |
| 定版本、保留回滚点、检阅后再推送 | 本地 rc3 候选完成；rc1/rc2 不移动。未推送。 |

证据见 [收尾与 rc3 验证](evidence/runtime_cpu_20261017/README.md#core-closeout-and-rc3--2026-10-23)。
完整套件验收对应运行代码 `ea0ced6`。候选 `ee2b44b` 仅增加文档与合并记录；
两套 Release 已更新为候选身份，并各通过 4 项来源、依赖及真实节点 E2E 检查。
本轮没有重放原有 14 点、两条独立路线和六条真实数据；此前结果作为历史基线保留。
**10 cm 全场景精度、所有困难搜索平均半核、RK3588/MID360 真机仍未验收。**
M2DGR 时间/参考点与相关观测不确定度仍有开放项。算法探索保持暂停。
模块内修改可使用定向测试；接口、异步时序或保存格式变化仍需集成验证。
当前日志策略见 [共用日志](../README.md#25-map-files-and-logs)。

## 1. 拉取与构建

先按 [README](../README.md) 安装 ROS、GTSAM、small_gicp 等依赖，准备自己的 LIO
前端，并一次性安装编译缓存：`sudo apt-get install ccache`。
下面以 `~/ros_ws` 为工作空间；如果实机路径不同，只改这个路径。
新机器将仓库 clone 到该工作空间的 `src/n3mapping`；已有机器按以下方式更新。

```bash
cd ~/ros_ws/src/n3mapping
git status --short --branch
```

使用已经传到实机仓库的候选标签，且确认没有未提交的本地修改；有修改时先保留
自己的工作，不要使用 reset/clean 强制覆盖。实机收到仓库/标签后固定到该版本：

若标签尚未推送，可先从交付的 Git bundle 导入。它保留提交历史和标签：

```bash
git fetch /absolute/path/to/n3mapping-v1.1.0-rc3.bundle \
  refs/tags/v1.1.0-rc3:refs/tags/v1.1.0-rc3
```

```bash
git switch --detach v1.1.0-rc3
N3M_COMMIT=$(git rev-parse HEAD)

source /opt/ros/humble/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh humble
export CCACHE_DIR="$PWD/build/ccache" CCACHE_TEMPDIR="$PWD/build/ccache/tmp"
mkdir -p "$CCACHE_TEMPDIR"
ccache --max-size 8G
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAKEFLAGS=-j4
colcon --log-base log/n3mapping-v1.1.0-rc3 build \
  --symlink-install --executor sequential \
  --packages-select n3mapping \
  --build-base build/n3mapping-v1.1.0-rc3 \
  --install-base install/n3mapping-v1.1.0-rc3 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DN3MAPPING_BUILD_RESEARCH_TOOLS=OFF \
  -DN3MAPPING_PRODUCT_COMMIT="$N3M_COMMIT" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

**构建成功后**再测试，避免 `install/` 旧库污染结果：

```bash
source ~/ros_ws/install/n3mapping-v1.1.0-rc3/setup.bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
ROS_LOG_DIR="$HOME/ros_ws/log/n3mapping-v1.1.0-rc3/ros_test" \
  ctest --test-dir ~/ros_ws/build/n3mapping-v1.1.0-rc3/n3mapping --output-on-failure -j1
ros2 pkg prefix n3mapping
ros2 run n3mapping n3mapping_node --build-identity-json
```

开发构建使用 README 中的独立 `RelWithDebInfo` / `Debug` 目录。不要在候选目录中
切换优化级别；保留既有 build/install/log 和 ccache。Release 的 `-O3 -DNDEBUG`
由 CMake 的 Release 配置提供，不再被包级无条件 `-O3` 覆盖。可在该构建目录的
`compile_commands.json` 核对命令，用 `ccache --show-stats` 核对缓存命中。

identity 应报告本次提交且 `verified=true`。这是构建来源确认，不等于实机功能验收。
如果产品构建拒绝 dirty checkout，不要绕过检查；把自己的运行配置放在仓库外。
依赖若安装在其他 workspace，应在构建前 source 对应 underlay。新终端先 source
ROS 和依赖，再最后 source 此版本安装目录；核对 `ros2 pkg prefix n3mapping`
指向它。不要用旧 `install/n3mapping` 的库去测试新构建的可执行文件。
本机 14600KF 对照使用 `taskset -c 0,2,4,6` 限定四个物理核心；其他 CPU 请按
实际拓扑选择四核，并在配置中保留 `num_threads: 4`。

## 2. 每次测试建立独立目录

准备一份自己的完整 YAML 配置，topic、frame、前端外参保持现有已知配置。
localization / map_extension 必须设置已有地图的**绝对路径** `map_path`；
不要用保存输出目录冒充输入地图，也不要原地覆盖原始地图。

```bash
source /opt/ros/humble/setup.bash
source ~/ros_ws/install/n3mapping-v1.1.0-rc3/setup.bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
RUN="$HOME/n3mapping_runs/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
NODE="$(ros2 pkg prefix n3mapping)/lib/n3mapping/n3mapping_node"

ros2 run n3mapping n3mapping_support_bundle.py prepare \
  --run-dir "$RUN" \
  --config /absolute/path/to/your_n3mapping.yaml \
  --mode localization \
  --node-executable "$NODE" \
  --diagnostics
echo "RUN=$RUN"
```

将 `--mode` 换成 `mapping` 或 `map_extension` 即可测试对应模式。
这个命令只读取节点的 build identity 并准备目录，**不会启动节点**。
它保存配置副本、构建信息和反馈说明，并把本次地图输出导向新目录，不改输入地图路径。
使用普通节点配置；不要与锁定参数的 `n3mapping_product_runtime.py` 混用。

`--diagnostics` 显式开启已有的重定位、回环和逐帧性能 JSONL，适合短时故障复现。
这会增加日志 IO 和计算开销，不应把带诊断的资源数字当作默认配置性能。
正常长期运行可省略该选项；工具不覆盖原配置中的 enable 值。
本候选由共用模块轮转，当前文件和历史文件合计不超过 100 MB。
长期运行应给每个实例配置固定的 `N3MAPPING_LOG_DIR`，跨重启沿用。
下面的 `RUN` 是独立的短时测试实例；验证重启时也沿用这个 `RUN`。
反馈包上限与运行日志上限分别生效；前端、RViz 等进程有各自的日志策略。

同一台机器用 Fast DDS 传大点云时，启动所有参与进程之前，按
[DDS 共享内存说明](../README.md#humble--fast-dds-large-local-point-clouds)选择
足够的 SHM 段。仓库的 `config/fastdds_local.xml` 为 64 MiB、仅本机的受控配置；
雷达或前端跨机器时先调整网络接口，不能直接照用 loopback 限制。

## 3. 像平常一样分别启动节点

在准备目录的同一终端启动 n3mapping：

```bash
set -o pipefail
N3MAPPING_LOG_DIR="$RUN/managed_logs" ROS_LOG_DIR="$RUN/ros_log" stdbuf -oL -eL "$NODE" \
  --ros-args --params-file "$RUN/config.yaml" \
  --params-file "$RUN/overrides.yaml" \
  2>&1 | tee "$RUN/n3mapping.log"
```

前端、RViz、bag（如需要）仍由你在其他终端启动。不额外设置 domain，不使用
systemd runner。新终端需要重新 source，并将 `RUN` 设为上面打印的同一个目录。
前端命令末尾可加 `2>&1 | tee "$RUN/frontend.log"`，不要记录无关终端输出。
需要 RViz 时可单独执行 `ros2 run rviz2 rviz2 -d "$(ros2 pkg prefix --share n3mapping)/launch/n3.rviz"`。

节点运行时，可在另一终端保存实际参数（先设置同一个 `RUN`）：

```bash
ros2 param dump /n3mapping_node > "$RUN/parameters.yaml"
```

如果服务调用失败，请保留错误，不把空文件当成成功参数快照。其他命令行覆盖也要写入
`issue.md`；准备阶段的配置副本不能代替运行时参数。

mapping / map_extension 测试结束前，在另一终端保存地图：

```bash
ros2 service call /n3mapping/save_map std_srvs/srv/Trigger '{}' \
  2>&1 | tee "$RUN/save_map.log"
```

检查返回 `success=True`。地图保留在本次 `RUN`，**不会进入日志反馈包**。
然后正常 Ctrl+C 停止各节点，让异步日志排空。本候选可以用相同命令在同一
`RUN` 中重启，完整日志会保留并受同一总额约束；不要重复执行 `prepare` 覆盖目录。
旧 rc2 仍会截断部分日志，使用旧版时为每次测试创建新 `RUN`。
新版本没有普通节点终端日志，查看 `$RUN/managed_logs/runtime.log`，或执行
`ros2 topic echo /n3mapping/logging_status` 检查写入失败及溢出计数。

## 4. 导出反馈包

先填写 `$RUN/issue.md`，特别是问题发生时间、预期/实际行为、复现步骤。
截图或视频单独保存，不能仅凭“程序没报错”判断位置/yaw 正确。

```bash
ros2 run n3mapping n3mapping_support_bundle.py pack \
  --run-dir "$RUN" --output "${RUN}.zip"
unzip -l "${RUN}.zip"
unzip -p "${RUN}.zip" bundle_report.json
```

如果 ROS 环境不可用，可用仓库里的脚本离线打包：

```bash
python3 -B ~/ros_ws/src/n3mapping/tools/n3mapping_support_bundle.py pack \
  --run-dir "$RUN" --output "${RUN}-offline.zip"
```

- 只选择本次目录中约定名称的配置/文本日志、`managed_logs/` 的事件和健康记录、数字后缀轮转文件，以及 `ros_log/` 下的 `.log`。
- 不扫描整个 `~/.ros`，不收集环境变量、密钥、地图、bag 或指向外部文件的符号链接。
- 默认单文件最多 8 MiB、总文本最多 64 MiB。大日志保留头尾并以 `.excerpt` 标识，
  `bundle_report.json` 记录保留字节范围；它不是完整 JSONL，不能直接代替原日志跑分析。
- 缺文件、截断或日志在打包时变化都会报告 `incomplete=true`；这不是运行成功/失败判定。
  原始文件不修改、不删除。需要更多原始日志时，在确认磁盘和传输预算后调整
  `--max-file-mib` / `--max-total-mib`，另取一个输出文件名。
- ZIP 不自动脱敏，配置/路径/轨迹仍可能含敏感信息。分享前先解压到独立目录审阅，
  对要分享的副本脱敏，不覆盖原始诊断记录。

将审阅后的 ZIP 和截图发回当前开发任务，或作为 GitHub Issue 附件（上传能力和大小限制
以界面为准）。**不要 git add 日志、地图或 bag，也不要把私密场景材料提交到公开 Issue。**
文本日志通常足以定位阶段和异常；需要重放才能解释的问题，再单独提供最小 bag/地图，
不是每次反馈都上传全量数据。

## 5. 第一轮实机验收

依次测试建图、保存重载、定位、续建图及再次保存重载。记录正确性、失败时是否明确拒绝、
CPU/内存是否满足实机需求。新硬件的预算由实际任务决定，不直接套用 Jammy 上的历史数字。
遇到问题先保留本次运行目录，不靠反复调阈值掩盖。修复只针对可复现问题，并保留对应回归。
