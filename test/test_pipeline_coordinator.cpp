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

TEST(PipelineCoordinatorTest, ForwardsFastLioDebugCallbacks) {
    Config config;
    config.mode = "mapping";
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
    config.frontend_publish_debug = true;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_timing = true;
    config.rhpd_submap_voxel_size = 0.0;
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    int odom_count = 0;
    int cloud_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    pipeline.setLioDebugCallbacks(callbacks);

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
    EXPECT_EQ(odom_count, 1);
    EXPECT_EQ(cloud_count, 1);
    EXPECT_EQ(timing_count, 1);
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

TEST(PipelineCoordinatorTest, ForwardsDlioDebugCallbacks) {
    Config config;
    config.mode = "mapping";
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
    config.frontend_publish_debug = true;
    config.frontend_debug_publish_odom = true;
    config.frontend_debug_publish_deskewed_cloud = true;
    config.frontend_debug_publish_timing = true;
    config.rhpd_submap_voxel_size = 0.0;
    core::PipelineCoordinator pipeline(config);
#ifdef N3MAPPING_BUILD_DLIO_CORE
    ASSERT_TRUE(pipeline.ready()) << pipeline.error();
    int odom_count = 0;
    int cloud_count = 0;
    int timing_count = 0;
    lio::LioDebugCallbacks callbacks;
    callbacks.odom = [&](const core::LioFrame&) { ++odom_count; };
    callbacks.deskewed_cloud =
        [&](const core::LioFrame::PointCloud::ConstPtr&) { ++cloud_count; };
    callbacks.timing = [&](const lio::LioTimingStats&) { ++timing_count; };
    pipeline.setLioDebugCallbacks(callbacks);

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
    EXPECT_EQ(odom_count, 1);
    EXPECT_EQ(cloud_count, 1);
    EXPECT_EQ(timing_count, 1);
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

TEST(PipelineCoordinatorTest, FastLioBuiltinLocalizationWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "localization";
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
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
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

TEST(PipelineCoordinatorTest, DlioBuiltinLocalizationWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "localization";
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
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
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
#else
    EXPECT_FALSE(pipeline.ready());
#endif
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

TEST(PipelineCoordinatorTest, FastLioBuiltinMapExtensionWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "map_extension";
    config.frontend_mode = "fast_lio";
    config.frontend_prediction_only_output = true;
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
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

TEST(PipelineCoordinatorTest, DlioBuiltinMapExtensionWithoutMapReportsNotLoaded) {
    Config config;
    config.mode = "map_extension";
    config.frontend_mode = "dlio";
    config.frontend_prediction_only_output = true;
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
    EXPECT_FALSE(output.success);
    EXPECT_NE(output.error.find("map is not loaded"), std::string::npos);
#else
    EXPECT_FALSE(pipeline.ready());
#endif
}

}  // namespace test
}  // namespace n3mapping
