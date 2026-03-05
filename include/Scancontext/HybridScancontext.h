#pragma once

/**
 * @brief Hybrid ScanContext - 多通道场景描述子 (v2)
 *
 * 7 通道描述子，纵向拼接为 (7*NUM_RING, NUM_SECTOR) 矩阵:
 *
 * 基础通道 (极坐标 BEV):
 *   Ch0: Max Height          - 与原始 SC 兼容
 *   Ch1: Height Variance     - 区分有结构的区域
 *   Ch2: Point Density       - 区分墙壁/空旷/门窗
 *
 * 高度分层占据率通道 (4层，室内关键区分力):
 *   Ch3: 占据率 层0 (脚下)   - 台阶、地面杂物
 *   Ch4: 占据率 层1          - 桌椅、栏杆、门下半
 *   Ch5: 占据率 层2          - 门上半、窗户、管道
 *   Ch6: 占据率 层3 (头上)   - 天花板、空调管道、灯具
 *
 * 高度分层以雷达为中心，固定范围 [OCCUPY_Z_MIN, OCCUPY_Z_MAX] (相对雷达),
 * 等分 4 层。每个 (ring, sector) bin 中按层统计点数，除以该 bin 总点数归一化
 * (高度直方图)，具有位置不变性。
 *
 * Ring Key 维度: 7*NUM_RING, 用于 KD-tree 检索
 * Sector Key: 每列均值, 用于粗对齐
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

using SCPointType = pcl::PointXYZI;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

class HybridSCManager
{
  public:
    // ======== 通道数 ========
    static constexpr int NUM_BASE_CHANNELS = 3;   // height, height_var, density
    static constexpr int NUM_OCCUPY_LAYERS = 4;    // 4层高度占据率
    static constexpr int NUM_CHANNELS = NUM_BASE_CHANNELS + NUM_OCCUPY_LAYERS; // 7

    // ======== 极坐标参数 (可配置) ========
    int    PC_NUM_RING    = 20;
    int    PC_NUM_SECTOR  = 60;
    double PC_MAX_RADIUS  = 80.0;
    double PC_MIN_RADIUS  = 0.5;
    double LIDAR_HEIGHT   = 2.0;
    bool   USE_LOG_POLAR  = true;

    // ======== 高度分层占据率参数 (相对于雷达z=0) ========
    double OCCUPY_Z_MIN   = -2.0;  // 雷达下方 2m (覆盖地面)
    double OCCUPY_Z_MAX   =  6.0;  // 雷达上方 6m (覆盖天花板/树冠)

    // ======== 检索参数 ========
    double SEARCH_RATIO   = 0.1;

    // ======== 通道权重 (距离计算时的加权) ========
    double W_HEIGHT       = 1.0;
    double W_HEIGHT_VAR   = 0.8;
    double W_DENSITY      = 0.6;
    double W_OCCUPY_L0    = 0.7;
    double W_OCCUPY_L1    = 0.7;
    double W_OCCUPY_L2    = 0.7;
    double W_OCCUPY_L3    = 0.7;

    // ======== 描述子总行数 ========
    int totalRows() const { return NUM_CHANNELS * PC_NUM_RING; }

    HybridSCManager() = default;

    /**
     * @brief 从点云生成 7 通道描述子
     *
     * 返回矩阵大小: (7*PC_NUM_RING, PC_NUM_SECTOR)
     * 通道布局 (行方向):
     *   [0R,  1R)   - max height
     *   [1R,  2R)   - height variance (sqrt)
     *   [2R,  3R)   - point density (normalized)
     *   [3R,  4R)   - 占据率 层0
     *   [4R,  5R)   - 占据率 层1
     *   [5R,  6R)   - 占据率 层2
     *   [6R,  7R)   - 占据率 层3
     */
    Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType>& scan);

    Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd& desc);
    Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd& desc);
    double distDirectSC(Eigen::MatrixXd& sc1, Eigen::MatrixXd& sc2);
    int fastAlignUsingVkey(Eigen::MatrixXd& vkey1, Eigen::MatrixXd& vkey2);
    std::pair<double, int> distanceBtnScanContext(Eigen::MatrixXd& sc1, Eigen::MatrixXd& sc2);

    // ======== 兼容 SCManager 的内部数据 ========
    double PC_UNIT_SECTORANGLE = 360.0 / 60.0;

    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;
    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

    void makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType>& scan);

  private:
    int rangeToRingIndex(double range) const;
    static float xy2theta(float x, float y);
    static Eigen::MatrixXd circshift(Eigen::MatrixXd& mat, int num_shift);
};
