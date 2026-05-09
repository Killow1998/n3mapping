#include <gtest/gtest.h>

#include "n3mapping/core/pipeline_coordinator.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (float x = -1.0f; x <= 1.0f; x += 0.1f) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = 0.5f;
        point.z = 0.25f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(PipelineCoordinatorTest, ExternalMappingFrameAddsKeyframe) {
    Config config;
    config.mode = "mapping";
    config.frontend_mode = "external";
    config.keyframe_distance_threshold = 0.5;
    config.rhpd_submap_voxel_size = 0.0;

    core::PipelineCoordinator pipeline(config);
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    auto output = pipeline.addExternalFrame(core::TimeStamp{100000000}, pose, makeCloud());
    ASSERT_TRUE(output.success) << output.error;
    EXPECT_TRUE(output.has_lio_frame);
    EXPECT_TRUE(output.accepted_keyframe);
    EXPECT_EQ(output.keyframe_id, 0);
    EXPECT_EQ(pipeline.session().keyframes().size(), 1u);
}

TEST(PipelineCoordinatorTest, BuiltinFrontendReportsFactoryErrorForNow) {
    Config config;
    config.frontend_mode = "fast_lio";
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    EXPECT_TRUE(pipeline.ready()) << pipeline.error();
    core::RawLidarFrame raw;
    raw.stamp_begin.nsec = 1;
    raw.stamp_end.nsec = 2;
    raw.points = makeCloud();
    auto output = pipeline.addRawLidar(raw);
    EXPECT_FALSE(output.has_lio_frame);
    EXPECT_FALSE(output.success);
#else
    EXPECT_FALSE(pipeline.ready());
    EXPECT_NE(pipeline.error().find("fast_lio_core"), std::string::npos);
#endif
}

TEST(PipelineCoordinatorTest, FastLioBuiltinPredictionOnlyCanFeedMapping) {
    Config config;
    config.mode = "mapping";
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    config.keyframe_distance_threshold = 0.1;
    config.rhpd_submap_voxel_size = 0.0;
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    core::ImuSample imu0;
    imu0.stamp.nsec = 1;
    core::ImuSample imu1;
    imu1.stamp.nsec = 1000001;
    pipeline.addImu(imu0);
    pipeline.addImu(imu1);
    core::RawLidarFrame raw;
    raw.stamp_begin.nsec = 1;
    raw.stamp_end.nsec = 1000001;
    raw.points = makeCloud();
    auto output = pipeline.addRawLidar(raw);
    EXPECT_TRUE(output.has_lio_frame);
    EXPECT_TRUE(output.success) << output.error;
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

TEST(PipelineCoordinatorTest, DlioBuiltinFrontendReportsFactoryErrorForNow) {
    Config config;
    config.frontend_mode = "dlio";
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_DLIO_CORE
    EXPECT_TRUE(pipeline.ready()) << pipeline.error();
    core::RawLidarFrame raw;
    raw.stamp_begin.nsec = 1;
    raw.stamp_end.nsec = 2;
    raw.points = makeCloud();
    auto output = pipeline.addRawLidar(raw);
    EXPECT_FALSE(output.has_lio_frame);
    EXPECT_FALSE(output.success);
#else
    EXPECT_FALSE(pipeline.ready());
    EXPECT_NE(pipeline.error().find("dlio_core"), std::string::npos);
#endif
}

TEST(PipelineCoordinatorTest, DlioBuiltinPredictionOnlyCanFeedMapping) {
    Config config;
    config.mode = "mapping";
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    config.keyframe_distance_threshold = 0.1;
    config.rhpd_submap_voxel_size = 0.0;
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_DLIO_CORE
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    core::ImuSample imu0;
    imu0.stamp.nsec = 1;
    core::ImuSample imu1;
    imu1.stamp.nsec = 1000001;
    pipeline.addImu(imu0);
    pipeline.addImu(imu1);
    core::RawLidarFrame raw;
    raw.stamp_begin.nsec = 1;
    raw.stamp_end.nsec = 1000001;
    raw.points = makeCloud();
    auto output = pipeline.addRawLidar(raw);
    EXPECT_TRUE(output.has_lio_frame);
    EXPECT_TRUE(output.success) << output.error;
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

TEST(PipelineCoordinatorTest, BuiltinFrontendRejectsExternalFrameInput) {
    Config config;
    config.frontend_mode = "fast_lio";
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    auto output = pipeline.addExternalFrame(
        core::TimeStamp{100000000}, Eigen::Isometry3d::Identity(), makeCloud());
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("requires frontend_mode=external"), std::string::npos);
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

TEST(PipelineCoordinatorTest, LocalizationWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "localization";
    config.frontend_mode = "external";
    core::PipelineCoordinator pipeline(config);
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();

    auto output = pipeline.addExternalFrame(
        core::TimeStamp{100000000}, Eigen::Isometry3d::Identity(), makeCloud());
    EXPECT_TRUE(output.has_lio_frame);
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
}

TEST(PipelineCoordinatorTest, MapExtensionWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "map_extension";
    config.frontend_mode = "external";
    core::PipelineCoordinator pipeline(config);
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    EXPECT_EQ(pipeline.mode(), core::PipelineCoordinator::RunMode::MapExtension);

    auto output = pipeline.addExternalFrame(
        core::TimeStamp{100000000}, Eigen::Isometry3d::Identity(), makeCloud());
    EXPECT_TRUE(output.has_lio_frame);
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
}

}  // namespace test
}  // namespace n3mapping
