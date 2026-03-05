#include "HybridScancontext.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

// ============================================================
//  Utility
// ============================================================

float HybridSCManager::xy2theta(float x, float y)
{
    // atan2 返回 [-pi, pi]，我们要 [0, 360)
    float rad = std::atan2(y, x);             // [-pi, pi]
    float deg = rad * 180.0f / static_cast<float>(M_PI); // [-180, 180]
    if (deg < 0.0f)
        deg += 360.0f;
    return deg; // [0, 360)
}

Eigen::MatrixXd HybridSCManager::circshift(Eigen::MatrixXd& mat, int num_shift)
{
    if (num_shift == 0) {
        return mat;
    }
    const int cols = mat.cols();
    num_shift = ((num_shift % cols) + cols) % cols; // 保证正数
    Eigen::MatrixXd shifted(mat.rows(), cols);
    for (int c = 0; c < cols; ++c) {
        shifted.col((c + num_shift) % cols) = mat.col(c);
    }
    return shifted;
}

int HybridSCManager::rangeToRingIndex(double range) const
{
    if (range < PC_MIN_RADIUS || range > PC_MAX_RADIUS)
        return -1;

    int idx;
    if (USE_LOG_POLAR) {
        // 对数极坐标：近场更多bin
        // ring_idx = floor(NUM_RING * log(r/r_min) / log(r_max/r_min))
        double log_ratio = std::log(range / PC_MIN_RADIUS) / std::log(PC_MAX_RADIUS / PC_MIN_RADIUS);
        idx = static_cast<int>(std::floor(log_ratio * PC_NUM_RING));
    } else {
        // 线性极坐标（原始 SC）
        idx = static_cast<int>(std::ceil((range / PC_MAX_RADIUS) * PC_NUM_RING)) - 1;
    }

    return std::max(0, std::min(PC_NUM_RING - 1, idx));
}

// ============================================================
//  Core: 多通道描述子生成
// ============================================================

Eigen::MatrixXd HybridSCManager::makeScancontext(pcl::PointCloud<SCPointType>& scan)
{
    const int R = PC_NUM_RING;
    const int S = PC_NUM_SECTOR;
    const int total_rows = NUM_CHANNELS * R;  // 7 * R

    // ---------- 高度分层参数 ----------
    const double dz = (OCCUPY_Z_MAX - OCCUPY_Z_MIN) / static_cast<double>(NUM_OCCUPY_LAYERS);

    // ---------- 累加器 ----------
    Eigen::MatrixXd sum_z  = Eigen::MatrixXd::Zero(R, S);
    Eigen::MatrixXd sum_z2 = Eigen::MatrixXd::Zero(R, S);
    Eigen::MatrixXd count  = Eigen::MatrixXd::Zero(R, S);
    Eigen::MatrixXd max_z  = Eigen::MatrixXd::Constant(R, S, -1e6);

    // 每个 bin 每个高度层的点数 [layer][ring][sector]
    std::vector<Eigen::MatrixXd> layer_count(NUM_OCCUPY_LAYERS, Eigen::MatrixXd::Zero(R, S));

    // ---------- 遍历点云 ----------
    for (size_t i = 0; i < scan.points.size(); ++i) {
        float x = scan.points[i].x;
        float y = scan.points[i].y;
        float z_raw = scan.points[i].z;  // 雷达坐标系下的 z
        float z = z_raw + static_cast<float>(LIDAR_HEIGHT);  // 地面坐标系

        double range = std::sqrt(x * x + y * y);
        if (range < PC_MIN_RADIUS || range > PC_MAX_RADIUS)
            continue;

        float angle = xy2theta(x, y);

        int ring_idx = rangeToRingIndex(range);
        int sector_idx = std::max(0, std::min(S - 1,
                            static_cast<int>(std::floor(angle / 360.0f * S))));

        if (ring_idx < 0)
            continue;

        // 基础通道累加
        sum_z(ring_idx, sector_idx) += z;
        sum_z2(ring_idx, sector_idx) += z * z;
        count(ring_idx, sector_idx) += 1.0;
        if (z > max_z(ring_idx, sector_idx))
            max_z(ring_idx, sector_idx) = z;

        // 高度分层: 使用雷达坐标系 z_raw (相对雷达)
        if (z_raw >= OCCUPY_Z_MIN && z_raw < OCCUPY_Z_MAX) {
            int layer = static_cast<int>((z_raw - OCCUPY_Z_MIN) / dz);
            layer = std::min(layer, NUM_OCCUPY_LAYERS - 1);  // 边界保护
            layer_count[layer](ring_idx, sector_idx) += 1.0;
        }
    }

    // ---------- 构建 7 通道描述子 ----------
    Eigen::MatrixXd desc = Eigen::MatrixXd::Zero(total_rows, S);

    // density 归一化因子
    double max_count = count.maxCoeff();
    if (max_count < 1.0) max_count = 1.0;

    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            double n = count(r, s);
            if (n < 1.0)
                continue;  // 空 bin: 全零

            double mean_z = sum_z(r, s) / n;
            double var_z  = (sum_z2(r, s) / n) - (mean_z * mean_z);
            if (var_z < 0.0) var_z = 0.0;

            // Ch0: Max Height
            double mz = max_z(r, s);
            if (mz < -999.0) mz = 0.0;
            desc(r, s) = mz;

            // Ch1: Height Variance (sqrt for better dynamic range)
            desc(R + r, s) = std::sqrt(var_z);

            // Ch2: Point Density (normalized by max bin count)
            desc(2 * R + r, s) = n / max_count;

            // Ch3-Ch6: 高度分层占据率 (layer_count / bin_total_count)
            for (int l = 0; l < NUM_OCCUPY_LAYERS; ++l) {
                desc((NUM_BASE_CHANNELS + l) * R + r, s) =
                    layer_count[l](r, s) / n;
            }
        }
    }

    return desc;
}

// ============================================================
//  Ring Key (旋转不变) - 用于 KD-tree 初筛
// ============================================================

Eigen::MatrixXd HybridSCManager::makeRingkeyFromScancontext(Eigen::MatrixXd& desc)
{
    // 每行取均值
    Eigen::MatrixXd key(desc.rows(), 1);
    for (int r = 0; r < desc.rows(); ++r) {
        key(r, 0) = desc.row(r).mean();
    }
    return key;
}

// ============================================================
//  Sector Key (旋转变化) - 用于快速粗对齐
// ============================================================

Eigen::MatrixXd HybridSCManager::makeSectorkeyFromScancontext(Eigen::MatrixXd& desc)
{
    // 每列取均值（跨所有通道）
    Eigen::MatrixXd key(1, desc.cols());
    for (int c = 0; c < desc.cols(); ++c) {
        key(0, c) = desc.col(c).mean();
    }
    return key;
}

// ============================================================
//  多通道加权距离
// ============================================================

double HybridSCManager::distDirectSC(Eigen::MatrixXd& sc1, Eigen::MatrixXd& sc2)
{
    const int R = PC_NUM_RING;
    const int S = sc1.cols();
    const double weights[NUM_CHANNELS] = {
        W_HEIGHT, W_HEIGHT_VAR, W_DENSITY,
        W_OCCUPY_L0, W_OCCUPY_L1, W_OCCUPY_L2, W_OCCUPY_L3
    };
    double weight_sum = 0.0;

    double total_dist = 0.0;

    // 对每个通道分别计算余弦距离
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        int row_start = ch * R;

        int num_eff_cols = 0;
        double sum_sim = 0.0;

        for (int c = 0; c < S; ++c) {
            Eigen::VectorXd col1 = sc1.block(row_start, c, R, 1);
            Eigen::VectorXd col2 = sc2.block(row_start, c, R, 1);

            double n1 = col1.norm();
            double n2 = col2.norm();
            if (n1 < 1e-6 || n2 < 1e-6)
                continue;

            double sim = col1.dot(col2) / (n1 * n2);
            sum_sim += sim;
            num_eff_cols++;
        }

        if (num_eff_cols > 0) {
            double channel_dist = 1.0 - (sum_sim / num_eff_cols);
            total_dist += weights[ch] * channel_dist;
            weight_sum += weights[ch];
        }
    }

    if (weight_sum < 1e-6)
        return 1.0;

    return total_dist / weight_sum;
}

// ============================================================
//  快速粗对齐
// ============================================================

int HybridSCManager::fastAlignUsingVkey(Eigen::MatrixXd& vkey1, Eigen::MatrixXd& vkey2)
{
    int argmin_shift = 0;
    double min_diff = std::numeric_limits<double>::max();

    for (int shift = 0; shift < vkey1.cols(); ++shift) {
        Eigen::MatrixXd shifted = circshift(vkey2, shift);
        double diff = (vkey1 - shifted).norm();
        if (diff < min_diff) {
            min_diff = diff;
            argmin_shift = shift;
        }
    }

    return argmin_shift;
}

// ============================================================
//  带旋转搜索的多通道距离
// ============================================================

std::pair<double, int> HybridSCManager::distanceBtnScanContext(Eigen::MatrixXd& sc1, Eigen::MatrixXd& sc2)
{
    // 1. 用 sector key 粗对齐
    Eigen::MatrixXd vkey1 = makeSectorkeyFromScancontext(sc1);
    Eigen::MatrixXd vkey2 = makeSectorkeyFromScancontext(sc2);
    int coarse_shift = fastAlignUsingVkey(vkey1, vkey2);

    // 2. 在粗对齐附近精搜索
    const int SEARCH_RADIUS = std::max(1, static_cast<int>(std::round(0.5 * SEARCH_RATIO * sc1.cols())));
    std::vector<int> shifts;
    shifts.push_back(coarse_shift);
    for (int i = 1; i <= SEARCH_RADIUS; ++i) {
        shifts.push_back((coarse_shift + i + sc1.cols()) % sc1.cols());
        shifts.push_back((coarse_shift - i + sc1.cols()) % sc1.cols());
    }

    double min_dist = std::numeric_limits<double>::max();
    int best_shift = 0;

    for (int shift : shifts) {
        Eigen::MatrixXd sc2_shifted = circshift(sc2, shift);
        double dist = distDirectSC(sc1, sc2_shifted);
        if (dist < min_dist) {
            min_dist = dist;
            best_shift = shift;
        }
    }

    return { min_dist, best_shift };
}

// ============================================================
//  保存接口 (兼容 SCManager)
// ============================================================

void HybridSCManager::makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType>& scan)
{
    Eigen::MatrixXd sc = makeScancontext(scan);
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);

    std::vector<float> ringkey_vec(ringkey.data(), ringkey.data() + ringkey.size());

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(ringkey_vec);
}
