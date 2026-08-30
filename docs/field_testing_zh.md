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

## 1. 拉取与构建

先按 [README](../README.md) 安装 ROS、GTSAM、small_gicp 等依赖，准备自己的 LIO
前端。下面以 `~/ros_ws` 为工作空间；如果实机路径不同，只改这个路径。
新机器将仓库 clone 到该工作空间的 `src/n3mapping`；已有机器按以下方式更新。

```bash
cd ~/ros_ws/src/n3mapping
git status --short --branch
```

确认当前为 `main`，且没有未提交的本地修改后再执行；有修改时先保留自己的工作，
不要使用 reset/clean 强制覆盖。

```bash
git pull --ff-only origin main
N3M_COMMIT=$(git rev-parse HEAD)

source /opt/ros/humble/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh humble
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DN3MAPPING_BUILD_RESEARCH_TOOLS=OFF \
  -DN3MAPPING_PRODUCT_COMMIT="$N3M_COMMIT"
```

**构建成功后**再测试，避免 `install/` 旧库污染结果：

```bash
source ~/ros_ws/install/setup.bash
ROS_LOG_DIR="$HOME/ros_ws/log/n3mapping_ctest" \
  ctest --test-dir ~/ros_ws/build/n3mapping --output-on-failure
ros2 run n3mapping n3mapping_node --build-identity-json
```

identity 应报告本次提交且 `verified=true`。这是构建来源确认，不等于实机功能验收。
如果产品构建拒绝 dirty checkout，不要绕过检查；把自己的运行配置放在仓库外。

## 2. 每次测试建立独立目录

准备一份自己的完整 YAML 配置，topic、frame、前端外参保持现有已知配置。
localization / map_extension 必须设置已有地图的**绝对路径** `map_path`；
不要用保存输出目录冒充输入地图，也不要原地覆盖原始地图。

```bash
source /opt/ros/humble/setup.bash
source ~/ros_ws/install/setup.bash
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
运行中的日志不自动轮转，短时复现结束后及时停止；打包上限不能限制运行日志占用。

## 3. 像平常一样分别启动节点

在准备目录的同一终端启动 n3mapping：

```bash
set -o pipefail
ROS_LOG_DIR="$RUN/ros_log" stdbuf -oL -eL "$NODE" \
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
然后正常 Ctrl+C 停止各节点，保留终端末尾信息。不要在旧 `RUN` 中重启节点，
因为 `optimization.log` 等文件会被截断；下一次重新执行 `prepare`。

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

- 只选择本次目录中约定名称的配置/文本日志，以及 `ros_log/` 下的 `.log`。
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
