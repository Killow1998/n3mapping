#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "n3mapping/loop_heightmap_diagnostics.h"

namespace n3mapping {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeGridCloud(double z, double x_offset = 0.0)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (int ix = 0; ix < 4; ++ix) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int sample = 0; sample < 3; ++sample) {
                pcl::PointXYZI point;
                point.x = static_cast<float>(x_offset + ix + 0.05 * sample);
                point.y = static_cast<float>(iy + 0.05 * sample);
                point.z = static_cast<float>(z + 0.01 * sample);
                point.intensity = 1.0F;
                cloud->push_back(point);
            }
        }
    }
    return cloud;
}

TEST(LoopHeightmapDiagnosticsTest, DetectsKnownVerticalShift)
{
    const auto target = makeGridCloud(0.0);
    const auto source = makeGridCloud(1.5);
    const auto diagnostics = computeHeightmapConsistency(
        target, source, Eigen::Isometry3d::Identity(), 1.0, 3, 0.10);

    EXPECT_EQ(diagnostics.overlap_cell_count, 16);
    EXPECT_DOUBLE_EQ(diagnostics.overlap_ratio, 1.0);
    EXPECT_DOUBLE_EQ(diagnostics.ground_support_ratio, 1.0);
    EXPECT_NEAR(diagnostics.ground_dz_median, 1.5, 1e-6);
    EXPECT_NEAR(diagnostics.ground_dz_p90, 1.5, 1e-6);
    EXPECT_NEAR(diagnostics.ground_dz_max, 1.5, 1e-6);
    EXPECT_GT(diagnostics.vertical_consistency_score, 0.0);
    EXPECT_LT(diagnostics.vertical_consistency_score, 0.5);
}

TEST(LoopHeightmapDiagnosticsTest, ReportsNoSupportWhenCellsDoNotOverlap)
{
    const auto target = makeGridCloud(0.0);
    const auto source = makeGridCloud(0.0, 100.0);
    const auto diagnostics = computeHeightmapConsistency(
        target, source, Eigen::Isometry3d::Identity(), 1.0, 3, 0.10);

    EXPECT_EQ(diagnostics.overlap_cell_count, 0);
    EXPECT_DOUBLE_EQ(diagnostics.overlap_ratio, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.ground_support_ratio, 0.0);
    EXPECT_FALSE(std::isfinite(diagnostics.ground_dz_median));
    EXPECT_FALSE(std::isfinite(diagnostics.ground_dz_p90));
    EXPECT_FALSE(std::isfinite(diagnostics.ground_dz_max));
    EXPECT_DOUBLE_EQ(diagnostics.vertical_consistency_score, 0.0);
}

}  // namespace
}  // namespace n3mapping
