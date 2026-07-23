#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "n3mapping/dense_trajectory_export.h"

namespace {

n3mapping::N3NavResource nativeResource(
    const std::vector<double>& timestamps) {
    n3mapping::N3NavResource resource;
    resource.dense_trajectory_source = "native";
    resource.dense_trajectory_degraded = false;
    resource.has_native_dense_trajectory = true;
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
        n3mapping::core::DenseTrajectoryPose pose;
        pose.seq = index + 10U;
        pose.timestamp = timestamps[index];
        pose.pose_world_lidar.translation().x() = static_cast<double>(index);
        resource.dense_optimized_trajectory.push_back(pose);
    }
    return resource;
}

TEST(DenseTrajectoryExport, ConvertsSecondsWithLlroundAndKeepsPose) {
    const auto resource = nativeResource({1.0000000004, 1.0000000016});
    std::vector<n3mapping::tools::DenseTrajectoryCsvRow> rows;
    std::string error;
    ASSERT_TRUE(n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(
        resource, &rows, &error)) << error;
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].stamp_ns, 1000000000LL);
    EXPECT_EQ(rows[1].stamp_ns, 1000000002LL);
    EXPECT_EQ(rows[0].seq, 10U);
    EXPECT_DOUBLE_EQ(rows[1].translation.x(), 1.0);
    EXPECT_DOUBLE_EQ(rows[0].orientation.w(), 1.0);
}

TEST(DenseTrajectoryExport, RejectsDegradedOrNonNativeTrajectory) {
    auto resource = nativeResource({1.0});
    resource.dense_trajectory_degraded = true;
    resource.has_native_dense_trajectory = false;
    std::vector<n3mapping::tools::DenseTrajectoryCsvRow> rows;
    std::string error;
    EXPECT_FALSE(n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(
        resource, &rows, &error));
    EXPECT_NE(error.find("native non-degraded"), std::string::npos);
}

TEST(DenseTrajectoryExport, RejectsDuplicateNanosecondsAfterRounding) {
    const auto resource = nativeResource({1.0000000001, 1.0000000004});
    std::vector<n3mapping::tools::DenseTrajectoryCsvRow> rows;
    std::string error;
    EXPECT_FALSE(n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(
        resource, &rows, &error));
    EXPECT_NE(error.find("collide after llround"), std::string::npos);
}

TEST(DenseTrajectoryExport, RejectsNonMonotonicAndNonFiniteData) {
    std::vector<n3mapping::tools::DenseTrajectoryCsvRow> rows;
    std::string error;
    EXPECT_FALSE(n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(
        nativeResource({2.0, 1.0}), &rows, &error));
    EXPECT_NE(error.find("strictly increasing"), std::string::npos);

    auto resource = nativeResource({1.0});
    resource.dense_optimized_trajectory.front().pose_world_lidar.translation().x() =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(
        resource, &rows, &error));
    EXPECT_NE(error.find("non-finite pose"), std::string::npos);
}

}  // namespace
