# 实机拉取、测试与问题反馈

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
