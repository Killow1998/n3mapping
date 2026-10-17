# 重定位开源实现调查与组件验证 — 2026-10-17

目标：在保持 N3M 现有接口、接受门槛、输入频率和功能范围的前提下，为
RK3588 / MID360 10 Hz / 约 100 × 100 m 地图寻找 CPU 与内存优化依据。
基线为 `3874efc668b4fabddfbbfe988e3c290c5a247bae`。

已核查 12 个独立 GitHub 项目的指定版本，其中 5 个运行了原作者的独立组件。
这不是 12 套完整 ROS 系统的精度排名。其余项目完成源码核查，未运行其完整系统。
以下把完整定位框架、全局配准和回环模块分别注明，避免把组件等同于完整定位能力。
逐项来源、文件指纹、原始样本和验证范围见 [results.json](results.json)。

## 逐项核查

| 项目与固定版本 | 可借鉴的实现 | 本轮验证与 N3M 决策 |
| --- | --- | --- |
| [lidar_localization_ros2](https://github.com/rsasaki0109/lidar_localization_ros2/tree/3e232cad48789b31617a9c6163cbbb1aada006e5)；完整 ROS 2 定位 | 按移动距离复用局部目标，独立的地图校正门控，BBS 候选搜索 | 原作者 C++ BBS 的 4 个 golden fixture 与 yaw-window 对照通过。N3M 已有目标缓存和实时校正输出；soft gate 会改变接受策略，本轮不移植。没有运行其整套定位节点。 |
| [hdl_localization](https://github.com/koide3/hdl_localization/tree/de2dc7769aa2412876d05ae6161408d3295b6a3c)；完整定位 | UKF/NDT 跟踪；全局查询结束后补偿查询期间的运动 | 源码确认 `DeltaEstimater` 补偿，以及服务请求只取一个候选。N3M 的 `RealtimeOutput` 已将可信校正投影到实时 odom，已有运动 E2E 覆盖；不改成单候选直接接受。未运行上游 ROS 1 节点。 |
| [hdl_global_localization](https://github.com/koide3/hdl_global_localization/tree/a6a6b69929ac371d9531d0dcdef3c1593fd49245)；全局定位服务 | BBS 二维占据金字塔，以及 FPFH/RANSAC、TEASER 引擎；地图特征预计算 | 核查地图准备、BBS 与 FLANN 评分实现。N3M 已持久化地图派生数据、使用有界精确近邻；单纯切换引擎没有等价性证据。未运行服务，不把其 README 提到的闭源 GLIL 算作开源项目。 |
| [3d_bbs](https://github.com/KOKIAOKI/3d_bbs/tree/41529a34a2fb9618b5ff560fb3c2363f1615666d)；全局配准模块 | 稀疏体素金字塔与分支定界；重力对齐后搜索 xyz/yaw | 运行原始 CPU 实现：3 个自地图裁剪查询成功得到粗位姿，随机负例拒绝。单线程查询需 1.19–2.19 CPU 秒；不适合在本次输入上每帧全图搜索。可作为未来初始化候选研究，当前不引入新搜索通路。 |
| [FAST-LIO-Localization-QN](https://github.com/engcang/FAST-LIO-Localization-QN/tree/123473d59bac4f8cc3b038c3d863a0684bd39ee8)；定位集成 | Quatro 粗配准接 Nano-GICP；近邻关键帧目标；里程计与地图匹配分开调度 | 核查 callback/timer 与 map matcher。作者明确要求两侧点云规模接近；不能直接将单帧对全图。N3M 已有粗到细验证与异步输出；未构建 ROS 1/Quatro/TEASER 全链路。README 标注 CC-BY-NC-SA 4.0，不按宽松许可源码直接搬入。 |
| [FAST_LIO_LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION/tree/2bc274ed0b36d14a5c34ada5e1473d52aa1db0d2)；定位集成 | FOV 裁图、两尺度 ICP、map-to-odom 与实时 odom 融合 | 源码默认 `FREQ_LOCALIZATION = 0.5`，需要粗初值。降低地图匹配频率不能作为本轮“不降性能”的优化证据；输出解耦 N3M 已具备。未运行 Open3D/ROS 1 节点。 |
| [mcl_3dl](https://github.com/at-wat/mcl_3dl/tree/0399995b018095f2ac50e9bb9c789b8362e079b3)；粒子定位框架 | 分块 KD-tree，距离似然与射线模型 | 原始分块索引对 20000 个查询与 PCL 全树结果一致；查询耗时少 36.8%，进程峰值 RSS 多约 63%。这不是与 N3M 的局部有界树相比，不能据此替换索引；粒子滤波及其轮式运动假设不纳入本轮。 |
| [MOLA / mola_relocalization](https://github.com/MOLAorg/mola/tree/b2010eb9e69bb8b14dc1536abcac4df1165ffd08/mola_relocalization)；模块化重定位 | SE(2) 初值格点、可配置 ICP、多解聚合与歧义处理 | 核查 ICP/likelihood 实现。N3M 已有 seed 合并、多假设和歧义拒绝，重复场景 E2E 覆盖现有行为；未运行 MRPT/mp2p 管线，也不将 SE(2) 入口当作完整盲搜 SE(3)。该模块 README 标注 GPL v3。 |
| [LTAOM](https://github.com/hku-mars/LTAOM/tree/aeb0bedcdfe4ec51e0d12f7b3804552e6066c0ae)；长期 SLAM/回环系统 | STD 描述子、先验描述子加载、跨会话地图 | 核查 prior descriptor 加载和候选流程。可学习不可变先验与离线预处理，N3M 已有相应地图索引/Atlas；引入整个 SLAM、前端及依赖分支超出本轮范围。未运行上游系统。 |
| [FAST-LOCALIZATION](https://github.com/YWL0720/FAST-LOCALIZATION/tree/93da4f462b8da16679c18a8aea9545ed4cb3c71b)；定位与 LIO 集成 | Scan Context 检索、两轮 ICP、连续初始化结果校验 | 源码有 MID360 使用入口，初始化使用 5 m/1 m 对应距离及两次位置一致性检查；还存在同一查询两次调用 `detectLoopClosureID()`。N3M 已有更完整的时序接受/歧义管理，保留现有合同；未运行其耦合 LIO 前端。 |
| [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher/tree/8b58b118e00c7a5a22448741f8d956d638a44ae2)；全局配准模块 | FasterPFH、图剪枝与鲁棒求解 | 原始 FasterPFH 在 1694 点查询上生成 1690 个有限描述子。单线程中位 26.60 CPU ms；四线程 wall time 降低但 CPU 增至 29.91 ms。仅运行特征阶段，未验证 ROBIN/位姿求解；没有足够证据替换 N3M 检索。 |
| [MapClosures](https://github.com/PRBonn/MapClosures/tree/95920fa549326a9a04a5b1efc4441e5071a5b835)；回环模块 | 地面对齐密度图、ORB/HBST 检索、二维几何验证 | 运行原始密度图生成器与相同 ORB 参数。全图 19 个特征，小查询 0 个；固定地面变换时垂直平移 5 m 的密度图相同。只能作为候选描述子研究，仍需三维验证；没有运行完整地面对齐、HBST 或回环流程。 |

BSD/MIT/Apache/GPL 等标记来自固定版本仓库元数据或模块 README；完整源码接入时还需逐依赖核对。
本轮未向生产代码引入上述第三方实现。

## 测量边界与运行结果

运行环境为 x86_64 / i5-14600KF / Ubuntu 22.04；N3M 为 Humble Release 开发构建，
identity 为 UNVERIFIED。没有运行 GPU，也没有 RK3588 实测。
原始地图包含 152116 个有限点，范围约 50 × 59 × 14 m，并非目标 100 × 100 m 地图。
三个查询来自同一张地图的 8 m 半径裁剪，0.5 m 体素保留首点，再施加已知位姿；
因此只能检验组件、计算量和粗位姿，不能证明独立传感器扫描的召回率或误锁率。
输入文件 SHA-256 和生成参数已保留，未发布地图或原始传感器数据。

| 实验 | 结果 | 适用范围 |
| --- | --- | --- |
| 3d_bbs，单线程，5 s timeout | 3 个正例分别 1325 / 2194 / 1185 CPU ms；平移误差约 0.65 / 0.57 / 0.40 m；随机负例 2070 CPU ms 后拒绝 | 分辨率 0.5 m、6 层、全图平移范围、固定 roll/pitch、完整 yaw；每输入一次测量，不是精度基准 |
| 3d_bbs，4 线程，1 s timeout | 正例 wall time 374 / 635 / 354 ms，总 CPU 1365 / 2278 / 1258 ms | 多线程缩短等待，不等于减少半核预算；单线程 1 s 试验均超时 |
| mcl_3dl，3 次交替进程 | PCL 全树/分块：建树中位 14.42 / 33.84 ms；20000 次查询 20.70 / 13.09 ms；RSS 35208 / 57364 KiB | chunk 20 m、查询半径 1.5 m；距离差均不超过 1e-6，1771 次命中；RSS 在额外创建验证用全树之前记录 |
| FasterPFH，3 次交替进程 | 1/4 线程：CPU 中位 26.60 / 29.91 ms，wall 中位 26.65 / 8.20 ms | 查询点数 1694，法线/特征半径 1.5/2.5 m，linearity 0.99；不含地图特征和位姿求解 |
| MapClosures，3 次交替进程 | 全图密度图 119×101，中位生成 1.23 ms；查询 32×30，0.025 ms，ORB 为 0 | 默认 ORB 边缘宽度 31 像素；小图不满足现有提取条件。固定 identity 地面对齐的 z 平移实验只说明此二维描述的高度信息损失，不是完整算法缺陷结论 |
| lidar_localization_ros2 原作者 BBS fixture | 4 个 golden case 及 yaw-window 过滤一致性断言通过 | 直接编译原测试，未定义 NDEBUG；没有把它记为实图定位通过 |

## N3M 内存消融：拒绝合入

尝试只保留配准实际读取的数组：plane target 保留点与法线；plane source 保留点；
refine GICP 保留点与协方差。正常数值、采样点、门槛及线程配置保持一致。

1. `trim_storage` 在原有计算后释放不用的数组；`needed_features` 进一步只估计 plane target 法线。
2. 初版在 legacy E2E 通过，但 Atlas 写入崩溃：Atlas v1 文件合同要求完整协方差。
   这说明单模块“未使用”不等于整个系统不需要，初版没有通过集成验证。
3. 修订版在保存边界补齐协方差，读取时先完整验证再释放 plane 协方差；通过全部 61 项 CTest。
   基线与修订版写出的 Atlas 文件逐字节一致，四种新旧读写组合均通过。
4. 随后使用同一个冻结 E2E 二进制，只更换 core 动态库，交替运行三次。
   18 个进程、108 次场景、936 帧的全部非计时字段一致，但 CPU 未改善，故撤回全部生产与测试改动。

整张地图同时作为 source/target 的控制实验中，预处理数组容量由 127.77 降为 69.08 MiB，
进程 RSS 中位由 160.25 降为 134.99 MiB，CPU 中位约 657 / 656 ms，基本持平。
这是内存压力实验，不是正常单帧规模，也不是整个 N3M 部署 RAM 估计。

最终修订版的 E2E 进程结果（中位数，含测试建图、保存、加载及全部线程）：

| 配置 | 基线 CPU → 修订版 CPU | 增幅 | 基线 RSS → 修订版 RSS |
| --- | --- | --- | --- |
| legacy | 4.43 → 4.49 s | +1.35% | 73740 → 69380 KiB |
| atlas | 4.19 → 4.31 s | +2.86% | 79856 → 76200 KiB |
| kill | 3.64 → 3.68 s | +1.10% | 75512 → 65996 KiB |

三次样本不足以建立严格的性能统计结论，但没有证据满足“不降性能”的要求；
尤其 Atlas 补齐数组增加了保存工作，不能只拿数组容量下降作为接受理由。
最终生产代码恢复到基线，重新完整构建及 61/61 CTest 通过；只保存调查与消融证据，不新增后台任务、索引或算法入口。
此前精确近邻剪枝带来的收益仍保留，见 [已有 CPU 证据](../registration_cpu_optimization_20261017.json)。

Atlas 是 `.localization_atlas.pb` 地图旁缓存，保存多分辨率配准点、法线、协方差和指纹；
它减少地图预处理，不负责地图外的位置推算。出地图后的连续输出来自可信地图校正与有效 odom 的组合。

## 复现组件测量

原始查询生成器及六个小型探针保存在 [证据目录](.)。
它们只用于此调查，不添加生产 CMake 目标。外部项目检出表中的固定提交；
`results.json` 记录实际读取/编译的上游文件指纹。表中性能来自格式化前的等价探针，两个源码指纹均保留。

以下从 n3mapping 仓库执行，`SURVEY_SOURCES` 指向包含 `owner/repository` 布局的外部源码目录，
`SURVEY_RUN` 指向可写的临时输出目录，`SURVEY_MAP` 为上述 SHA-256 对应的 PCD。
需要 Eigen、PCL、OpenCV、TBB 和 C++17 编译器；命令不下载地图、不运行 GPU。

```bash
SURVEY_PROBES="$PWD/docs/evidence/framework_survey_20261017"
mkdir -p "$SURVEY_RUN"
python3 "$SURVEY_PROBES/make_inputs.py" "$SURVEY_MAP" "$SURVEY_RUN"

SURVEY_BBS="$SURVEY_SOURCES/KOKIAOKI/3d_bbs/bbs3d"
g++ -O3 -std=c++17 -fopenmp -I/usr/include/eigen3 -I"$SURVEY_BBS/include" \
  "$SURVEY_PROBES/bbs_probe.cpp" "$SURVEY_BBS/src/cpu_bbs3d/bbs3d.cpp" \
  "$SURVEY_BBS/src/cpu_bbs3d/voxelmaps.cpp" -o "$SURVEY_RUN/bbs_probe"
OMP_WAIT_POLICY=PASSIVE "$SURVEY_RUN/bbs_probe" \
  "$SURVEY_RUN/map.bin" "$SURVEY_RUN/query1.bin" 1 5000

SURVEY_KISS="$SURVEY_SOURCES/MIT-SPARK/KISS-Matcher/cpp/kiss_matcher/core"
g++ -O3 -std=c++17 -I/usr/include/eigen3 -I"$SURVEY_KISS" \
  "$SURVEY_PROBES/fpfh_probe.cpp" "$SURVEY_KISS/kiss_matcher/FasterPFH.cpp" \
  -ltbb -o "$SURVEY_RUN/fpfh_probe"
"$SURVEY_RUN/fpfh_probe" "$SURVEY_RUN/query1.bin" 1

SURVEY_DENSITY="$SURVEY_SOURCES/PRBonn/MapClosures/cpp/map_closures"
g++ -O3 -std=c++17 -I/usr/include/eigen3 -I"$SURVEY_DENSITY" \
  "$SURVEY_PROBES/density_probe.cpp" "$SURVEY_DENSITY/DensityMap.cpp" \
  $(pkg-config --cflags --libs opencv4) -o "$SURVEY_RUN/density_probe"
"$SURVEY_RUN/density_probe" "$SURVEY_RUN/query1.bin"

g++ -O3 -std=c++17 -I/usr/include/eigen3 -I/usr/include/pcl-1.12 \
  -I"$SURVEY_SOURCES/at-wat/mcl_3dl/include" "$SURVEY_PROBES/chunk_probe.cpp" \
  -lpcl_io -lpcl_kdtree -lpcl_common -o "$SURVEY_RUN/chunk_probe"
"$SURVEY_RUN/chunk_probe" "$SURVEY_MAP" full
"$SURVEY_RUN/chunk_probe" "$SURVEY_MAP" chunk

SURVEY_ROS2="$SURVEY_SOURCES/rsasaki0109/lidar_localization_ros2"
g++ -O3 -std=c++17 -I"$SURVEY_ROS2/include" \
  "$SURVEY_ROS2/test/test_bbs_branch_and_bound.cpp" -o "$SURVEY_RUN/bbs2d_test"
"$SURVEY_RUN/bbs2d_test"
```

N3M 探针使用现有正常构建的 include/link 配置：`n3m_preparation_probe PCD THREADS`、
`atlas_compat_probe PCD ATLAS_PATH write|read`。前者统计数组 capacity 与实际被读取的特征指纹，
后者验证 Atlas 内容；不向生产增加额外接口。被拒绝的候选没有发布成可部署版本，
其变更步骤与源码/库指纹记录在证据中，不能用基线冒充候选重跑。
E2E 对照沿用 `N3MAPPING_EPISODE_TRACE`、`N3MAPPING_EPISODE_ATLAS`、
`N3MAPPING_EPISODE_FREE_SPACE_MODE`；字段名以 `_ms` 结尾的计时列排除后逐项比较。

## 后续：复用对应点，缩小精确搜索范围

内存裁剪撤回后，继续检查配准中的重复搜索。每个 small_gicp factor 已保存上一次迭代的对应点编号。
同一目标不变时，这个点到当前查询的距离可作为最近邻距离的上界。修订仅借它缩小搜索范围，
仍从原树根按原顺序搜索，保留距离门槛和等距点选择；没有直接把旧对应关系当作当前结果。
上界向正无穷保留一个可表示步长，避免提前排除恰好等距的点。

实现集中在 `registration_search.h` 和 `point_cloud_registration.cpp`，继续调用上游的三个 factor 与优化器。
不增加每点状态或地图索引。第一次迭代、无有效旧对应点及非有限输入继续走原有搜索；
普通查询不携带提示状态，临时提示仅在单次 factor 线性化调用内有效。
现有精确搜索测试扩展为有效、过时、越界和等距提示对照，随机查询/半径组合共 60000 次。

初版将提示编号存进共用搜索对象，在合成 E2E 上减少了 CPU 用时，但未绑核的短配准样本出现四线程回退；
再次轮换时方向又变化，因此没有用这些毫秒级单次样本确认收益或归因。
最终改为临时提示视图，并在同类物理核心上批量复测。所有初步样本也保留在
[对应点复用证据](../registration_hint_optimization_20261017.json)，没有只留下有利样本。

实图对照复用前述 152116 点地图和 1694 点自地图查询，固定 4 个初值偏移，
预处理统一用单线程生成相同输入；配准分别用 1/4 线程。每个初值预热一次、测量 50 次，
基线/修订版轮换 3 轮，共 2400 次计时配准。只测配准，不含预处理或真实传感器输入。
本机单线程固定一个性能核，四线程固定四个不同的性能核；不改变生产线程配置或亲和性。

| 配准线程数 | 四个初值合计 CPU 中位，基线 → 修订版 | wall time，基线 → 修订版 |
| --- | --- | --- |
| 1 | 13.993 → 13.485 ms（减少 3.63%） | 13.994 → 13.487 ms |
| 4 | 14.986 → 14.769 ms（减少 1.45%） | 3.905 → 3.856 ms |

这里先将每个初值的 50 次测量折算成一次，再合计四种初值；不能当成部署单帧耗时。
单线程位姿与数值逐项一致；四线程离散判定、内点与迭代数一致，所比较最终位姿矩阵元素最大差约 3.33e-16，
每批内部也没有成功/收敛/内点/最终迭代数变化。四线程个别初值的 wall time 仍有约 0.01 ms 波动，
不声称所有输入的延迟都严格降低。该自地图试验不证明实际重定位精度或误锁率。

复现配准对照使用 [n3m_alignment_probe.cpp](n3m_alignment_probe.cpp)：
`n3m_alignment_probe MAP_PCD query1.bin THREADS`。输入由同目录 `make_inputs.py` 生成，
探针链接正常构建的 core；更换 `LD_LIBRARY_PATH` 中的 core 目录，保持探针二进制与输入不变。
工具只记录与验证此实验，不增加生产 CMake 目标。

最终 Release 完整构建与 **61/61 CTest** 通过。同一冻结 E2E 二进制仅更换 core 动态库，
legacy/atlas/kill 三配置各轮换三次，共 18 个进程、108 次场景、936 帧，非计时字段全部一致。
进程 CPU 中位分别为 **4.41 → 4.25 s、4.20 → 4.04 s、3.64 → 3.50 s**，减少 **3.63%–3.85%**。
严格跟踪 p95 分别为 **9.442 → 9.082 ms、10.111 → 9.250 ms、9.450 → 9.193 ms**。
进程峰值 RSS 中位为 70956 / 81568 / 72412 KiB，对照组为 71608 / 79832 / 70568 KiB，
没有证明 RAM 改善。这些是本机受测场景的结果，不能换算成 RK3588 半核结论。

## 尚未证明的事项与下一步

- 没有证明 RK3588 上达到半核，也没有新增“不漂移的地图外定位”能力。10 Hz 半核按单核 100% 口径是平均每帧 50 CPU ms，必须计入后台线程。
- 7 个项目没有运行上游完整系统；已运行的 5 个也只是所列组件。源码已有的等价机制不重复接入；改变接受规则、降频、二维替换三维或更换前端的方案不属于本轮等价优化。
- 后续优先用目标机独立 MID360 回放测量现有局部索引、缓存准备和失败重试的 CPU 分布，再决定是否值得验证分块索引；当前 PCL 微基准不足以说明它能击败 N3M 的现有索引。
- 若以后允许改变候选生成策略，3d_bbs/FasterPFH/密度图可分别开展初始化召回、误锁和重复楼层消融；必须保留现有三维验证与接受门槛，并包含无匹配负例。当前没有将这些未验证方案接入生产。
- 下一项可验证的现有链路优化是延后 source 的 GICP 特征准备：目前 `prepareSourceCloud` 在粗配准前准备 refine 协方差，粗配准全部被拒绝时这些工作没有使用。若实现按查询共享且线程安全的按需准备，应保持原采样结果和粗配准门槛，并验证并发候选、无候选和回图路径；目前尚未实施。
- 线程预算也需按预处理、候选并行与单次配准分别测量。当前同一个 `num_threads` 影响多个阶段；不能通过同时改变上游并行采样结果来宣称单纯调度收益。下一步在目标机固定输入、保留 10 Hz 与恢复延迟要求后，再判断是否减少短任务的并行开销。
