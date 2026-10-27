#include <gtest/gtest.h>
#include <array>
#include <future>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include "n3mapping/point_cloud_matcher.h"
#include "../src/point_cloud_target_view.h"
#include "../src/point_cloud_sampling.h"
#include "support/registration_fixture.h"

namespace n3mapping {
namespace test {

using PointCloudMatcherTest = RegistrationFixture;

TEST_F(PointCloudMatcherTest, ProposalSamplingKeepsObservedSurfacesAndIgnoresInputOrder) {
    small_gicp::PointCloud input;
    // Two observed surfaces surround an empty cell centre, on either side
    // of zero. A centroid would fabricate a point in that empty space.
    for (int i = -6; i < 6; ++i) {
        input.points.emplace_back(i + 0.125, 0.875, 0.5, 1.0);
        input.points.emplace_back(i + 0.875, 0.125, 0.5, 1.0);
    }
    const auto first = registration_detail::sampleRepresentativeCoordinates(input, 1.0);
    std::reverse(input.points.begin(), input.points.end());
    const auto reversed = registration_detail::sampleRepresentativeCoordinates(input, 1.0);
    ASSERT_EQ(first->size(), 12U);
    ASSERT_EQ(reversed->size(), first->size());
    for (std::size_t i = 0; i < first->size(); ++i) {
        EXPECT_EQ(first->point(i), reversed->point(i));
        EXPECT_TRUE(std::any_of(input.points.begin(), input.points.end(),
            [&](const auto& point) { return point == first->point(i); }));
        EXPECT_EQ(first->point(i).x(), static_cast<double>(i) - 6.0 + 0.125);
    }
}

TEST_F(PointCloudMatcherTest, BoundedTrackingCannotAcquireADistantParallelSurface) {
    config_.icp_refine_use_gicp = false;
    PointCloudMatcher matcher(config_);
    const auto cloud = createPlaneCloud(3000, 8.0, 0.0);
    const auto target = matcher.prepareTargetCloud(cloud);
    const auto source = matcher.prepareSourceCloud(
        transformCloud(cloud, createPose(0.0, 0.0, 1.0)));
    auto bounded = matcher.getSettings();
    bounded.max_correspondence_distance = 0.3;
    const auto result = matcher.alignPrepared(
        target, source, Eigen::Isometry3d::Identity(), bounded);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.num_inliers, 0u);
    EXPECT_LT(result.T_target_source.translation().norm(), 1e-9);
}

// 测试预处理功能
TEST_F(PointCloudMatcherTest, PreprocessPointCloud) {
    auto cloud = createPlaneCloud(1000);
    
    auto [processed, kdtree] = matcher_->preprocessPointCloud(cloud);
    
    ASSERT_NE(processed, nullptr);
    ASSERT_NE(kdtree, nullptr);
    EXPECT_GT(processed->size(), 0u);
    // 下采样后点数应该减少
    EXPECT_LT(processed->size(), cloud->size());
}

TEST_F(PointCloudMatcherTest, CompactTargetsPreserveSerialNeighborsAndFeatures) {
    config_.gicp_downsampling_resolution = 0.1;
    PointCloudMatcher matcher(config_);
    const auto prepared = matcher.prepareTargetCloud(createPlaneCloud(12000));
    auto levels = prepared.plane_levels;
    if (prepared.has_refine_level) levels.push_back(prepared.refine_level);
    for (const auto& level : levels) {
        ASSERT_NE(level.cloud, nullptr);
        ASSERT_NE(level.kdtree, nullptr);
        EXPECT_EQ(level.cloud->points.capacity(), level.cloud->size());
        EXPECT_EQ(level.cloud->normals.capacity(), level.cloud->size());
        EXPECT_TRUE(level.cloud->covs.empty() ||
                    level.cloud->covs.capacity() == level.cloud->size());
        PointCloudMatcher::SmallGicpKdTree serial(level.cloud);
        for (std::size_t i = 0; i < level.cloud->size(); i += 13) {
            std::array<std::size_t, 20> actual{}, expected{};
            std::array<double, 20> actual_dist{}, expected_dist{};
            const auto& point = level.cloud->point(i);
            ASSERT_EQ(level.kdtree->knn_search(point, 20, actual.data(), actual_dist.data()),
                      serial.knn_search(point, 20, expected.data(), expected_dist.data()));
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(actual_dist, expected_dist);
        }
        auto reference = *level.cloud;
        small_gicp::estimate_normals_covariances_omp(
            reference, serial.kdtree, config_.gicp_num_neighbors, 1);
        for (std::size_t i = 0; i < reference.size(); ++i) {
            EXPECT_TRUE(reference.normal(i).isApprox(level.cloud->normal(i), 1e-12));
            if (!level.cloud->covs.empty())
                EXPECT_TRUE(reference.cov(i).isApprox(level.cloud->cov(i), 1e-12));
            else {
                const auto normal = level.cloud->normal(i).head<3>();
                const Eigen::Matrix3d reconstructed = Eigen::Matrix3d::Identity() -
                    .999 * normal * normal.transpose();
                EXPECT_TRUE((reference.cov(i).topLeftCorner<3, 3>().isApprox(reconstructed, 1e-12)));
            }
        }
    }
}

TEST_F(PointCloudMatcherTest, PreparedTargetBudgetCountsRetainedCapacity) {
    PointCloudMatcher::PreparedTarget target;
    auto cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>();
    cloud->resize(10);
    target.plane_levels.push_back({0.1, cloud, nullptr});
    const auto small = estimatePreparedTargetMemoryBytes(target);
    const auto capacity = cloud->covs.capacity();
    cloud->covs.reserve(1000);
    EXPECT_GE(estimatePreparedTargetMemoryBytes(target) - small,
              (cloud->covs.capacity() - capacity) * sizeof(Eigen::Matrix4d));
}

TEST_F(PointCloudMatcherTest, OnDemandTargetsPreserveConcurrentAndRepeatedRegistration) {
    config_.icp_refine_use_gicp = true;
    config_.icp_refine_downsampling_resolution = 0.2;
    PointCloudMatcher matcher(config_);
    const auto near = createSphereCloud(3000, 3.0);
    auto target_cloud = pcl::make_shared<PointCloudMatcher::PointCloudT>(*near);
    *target_cloud += *transformCloud(near, createPose(40, 0, 0));
    const auto source = matcher.prepareSourceCloud(near);
    const auto eager = matcher.prepareTargetCloud(target_cloud);
    const auto lazy = matcher.prepareTargetCloud(
        target_cloud, PointCloudMatcher::TargetFeatures::OnDemand);
    const auto guess = createPose(.05, -.03, .02, 0, 0, .02);
    const auto reference = matcher.alignPrepared(eager, source, guess);
    auto future = std::async(std::launch::async, [&] {
        return matcher.alignPrepared(lazy, source, guess);
    });
    const auto concurrent = matcher.alignPrepared(lazy, source, guess);
    const auto other = future.get();
    for (const auto& value : {concurrent, other, matcher.alignPrepared(lazy, source, guess)}) {
        EXPECT_EQ(value.termination, reference.termination);
        EXPECT_EQ(value.iterations, reference.iterations);
        EXPECT_EQ(value.num_inliers, reference.num_inliers);
        EXPECT_NEAR(value.fitness_score, reference.fitness_score, 1e-12);
        EXPECT_TRUE(value.T_target_source.matrix().isApprox(reference.T_target_source.matrix(), 1e-12));
        EXPECT_TRUE(value.information.isApprox(reference.information, 1e-10));
    }
    const auto check = matcher.evaluatePreparedPose(lazy, source, reference.T_target_source, reference.metric);
    const auto expected = matcher.evaluatePreparedPose(eager, source, reference.T_target_source, reference.metric);
    EXPECT_EQ(check.num_inliers, expected.num_inliers);
    EXPECT_TRUE(check.information.isApprox(expected.information, 1e-10));
    // Unvisited distant geometry keeps its full search support but needs no
    // feature evaluation. This is the observable work saved by the policy.
    for (const auto& level : lazy.plane_levels) {
        ASSERT_TRUE(level.lazy_normals);
        EXPECT_GT(level.lazy_normals->computedCount(), 0u);
        EXPECT_LT(level.lazy_normals->computedCount(), level.cloud->size());
    }
}

// 测试空点云预处理
TEST_F(PointCloudMatcherTest, PreprocessEmptyCloud) {
    Keyframe::PointCloudT::Ptr empty_cloud(new Keyframe::PointCloudT());
    
    auto [processed, kdtree] = matcher_->preprocessPointCloud(empty_cloud);
    
    EXPECT_TRUE(processed->empty());
    EXPECT_EQ(kdtree, nullptr);
}

// 测试空指针预处理
TEST_F(PointCloudMatcherTest, PreprocessNullCloud) {
    auto [processed, kdtree] = matcher_->preprocessPointCloud(nullptr);
    
    EXPECT_TRUE(processed->empty());
    EXPECT_EQ(kdtree, nullptr);
}

TEST_F(PointCloudMatcherTest, PreprocessFiltersInvalidSmallGicpVoxelCoordinates) {
    auto cloud = createPlaneCloud(500);
    pcl::PointXYZI nan_point;
    nan_point.x = std::numeric_limits<float>::quiet_NaN();
    nan_point.y = 0.0f;
    nan_point.z = 0.0f;
    nan_point.intensity = 1.0f;
    cloud->push_back(nan_point);

    pcl::PointXYZI huge_point;
    huge_point.x = 1.0e9f;
    huge_point.y = 0.0f;
    huge_point.z = 0.0f;
    huge_point.intensity = 1.0f;
    cloud->push_back(huge_point);
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;

    testing::internal::CaptureStderr();
    auto [processed, kdtree] = matcher_->preprocessPointCloud(cloud);
    const std::string stderr_output = testing::internal::GetCapturedStderr();

    ASSERT_NE(processed, nullptr);
    ASSERT_NE(kdtree, nullptr);
    EXPECT_GT(processed->size(), 0u);
    EXPECT_EQ(stderr_output.find("voxel coord is out of range"), std::string::npos);
}

// 测试相同点云配准 (应该返回单位变换)
// Requirements: 3.1, 3.2
TEST_F(PointCloudMatcherTest, AlignIdenticalClouds) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    EXPECT_LT(result.fitness_score, config_.gicp_fitness_threshold);
    
    // 变换应该接近单位矩阵
    EXPECT_TRUE(posesAreClose(result.T_target_source, Eigen::Isometry3d::Identity(), 0.1, 0.05));
}

// 测试已知变换的配准
// Requirements: 3.1, 3.2
TEST_F(PointCloudMatcherTest, AlignWithKnownTransform) {
    auto target_cloud = createPlaneCloud(2000);
    
    // 创建较小的已知变换（GICP 对大变换敏感）
    Eigen::Isometry3d known_transform = createPose(0.2, 0.1, 0.05, 0.0, 0.0, 0.05);
    
    // 变换源点云
    auto source_cloud = transformCloud(target_cloud, known_transform.inverse());
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), target_cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), source_cloud);
    
    // 使用已知变换作为初始猜测
    auto result = matcher_->align(target_kf, source_kf, known_transform);
    
    EXPECT_TRUE(result.success);
    EXPECT_LT(result.fitness_score, config_.gicp_fitness_threshold);
    
    // 估计的变换应该接近已知变换
    EXPECT_TRUE(posesAreClose(result.T_target_source, known_transform, 0.3, 0.2))
        << "Estimated transform should be close to known transform";
}

// 测试配准得分计算
// Requirements: 3.3
TEST_F(PointCloudMatcherTest, FitnessScoreCalculation) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    // 相同点云配准得分应该很低
    EXPECT_LT(result.fitness_score, 0.1);
    EXPECT_GT(result.num_inliers, 0u);
}

// 测试无效输入处理
TEST_F(PointCloudMatcherTest, AlignWithInvalidInput) {
    auto cloud = createPlaneCloud(1000);
    Keyframe::PointCloudT::Ptr empty_cloud(new Keyframe::PointCloudT());
    
    auto valid_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto empty_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), empty_cloud);
    
    // 空源点云
    auto result1 = matcher_->align(valid_kf, empty_kf);
    EXPECT_FALSE(result1.success);
    
    // 空目标点云
    auto result2 = matcher_->align(empty_kf, valid_kf);
    EXPECT_FALSE(result2.success);
    
    // nullptr 关键帧
    auto result3 = matcher_->align(nullptr, valid_kf);
    EXPECT_FALSE(result3.success);
    
    auto result4 = matcher_->align(valid_kf, nullptr);
    EXPECT_FALSE(result4.success);
}

// 测试批量配准
// Requirements: 3.7, 4.8, 13.1
TEST_F(PointCloudMatcherTest, AlignBatch) {
    auto cloud1 = createPlaneCloud(1500);
    auto cloud2 = createPlaneCloud(1500);
    
    // 创建多个配准对
    std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>> pairs;
    std::vector<Eigen::Isometry3d> init_guesses;
    
    for (int i = 0; i < 3; ++i) {
        auto target_kf = Keyframe::create(i * 2, i * 0.1, Eigen::Isometry3d::Identity(), cloud1);
        auto source_kf = Keyframe::create(i * 2 + 1, i * 0.1 + 0.05, Eigen::Isometry3d::Identity(), cloud2);
        
        pairs.emplace_back(target_kf, source_kf);
        init_guesses.push_back(Eigen::Isometry3d::Identity());
    }
    
    auto results = matcher_->alignBatch(pairs, init_guesses);
    
    EXPECT_EQ(results.size(), pairs.size());
    
    // 所有配准应该成功
    for (const auto& result : results) {
        EXPECT_TRUE(result.success);
    }
}

// 测试空批量配准
TEST_F(PointCloudMatcherTest, AlignBatchEmpty) {
    std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>> pairs;
    std::vector<Eigen::Isometry3d> init_guesses;
    
    auto results = matcher_->alignBatch(pairs, init_guesses);
    
    EXPECT_TRUE(results.empty());
}

// 测试初始猜测对配准的影响
TEST_F(PointCloudMatcherTest, AlignWithInitialGuess) {
    auto target_cloud = createSphereCloud(2000);
    
    // 创建较大的变换
    Eigen::Isometry3d known_transform = createPose(1.0, 0.5, 0.2, 0.0, 0.0, 0.2);
    auto source_cloud = transformCloud(target_cloud, known_transform.inverse());
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), target_cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), source_cloud);
    
    // 使用接近真实值的初始猜测
    Eigen::Isometry3d init_guess = createPose(0.9, 0.4, 0.15, 0.0, 0.0, 0.15);
    
    auto result = matcher_->align(target_kf, source_kf, init_guess);
    
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(posesAreClose(result.T_target_source, known_transform, 0.2, 0.15));
}

TEST_F(PointCloudMatcherTest, PreparedCloudsMatchRawPathAcrossInitialGuesses) {
    auto target_cloud = createSphereCloud(2500);
    const Eigen::Isometry3d known_transform =
        createPose(0.25, -0.15, 0.08, 0.01, -0.02, 0.08);
    auto source_cloud = transformCloud(target_cloud, known_transform.inverse());

    const auto prepared_target = matcher_->prepareTargetCloud(target_cloud);
    const auto prepared_source = matcher_->prepareSourceCloud(source_cloud);
    ASSERT_EQ(prepared_target.plane_levels.size(), 2u);
    ASSERT_EQ(prepared_source.plane_levels.size(), 2u);

    auto expect_equivalent = [](const MatchResult& raw,
                                const MatchResult& prepared) {
        EXPECT_EQ(prepared.success, raw.success);
        EXPECT_EQ(prepared.converged, raw.converged);
        EXPECT_EQ(prepared.termination, raw.termination);
        EXPECT_EQ(prepared.iterations, raw.iterations);
        EXPECT_EQ(prepared.num_inliers, raw.num_inliers);
        EXPECT_NEAR(prepared.fitness_score, raw.fitness_score, 1e-9);
        EXPECT_NEAR(prepared.inlier_ratio, raw.inlier_ratio, 1e-12);
        EXPECT_TRUE(prepared.T_target_source.matrix().isApprox(
            raw.T_target_source.matrix(), 1e-9));
        EXPECT_TRUE(prepared.information.isApprox(raw.information, 1e-8));
        EXPECT_EQ(prepared.metric.type, raw.metric.type);
        EXPECT_DOUBLE_EQ(prepared.metric.resolution, raw.metric.resolution);
        EXPECT_DOUBLE_EQ(prepared.metric.max_correspondence_distance, raw.metric.max_correspondence_distance);
        ASSERT_EQ(prepared.stages.size(), raw.stages.size());
        for (size_t i = 0; i < raw.stages.size(); ++i) {
            EXPECT_EQ(prepared.stages[i].stage, raw.stages[i].stage);
            EXPECT_DOUBLE_EQ(prepared.stages[i].resolution,
                             raw.stages[i].resolution);
            EXPECT_EQ(prepared.stages[i].converged,
                      raw.stages[i].converged);
            EXPECT_EQ(prepared.stages[i].iterations,
                      raw.stages[i].iterations);
            EXPECT_EQ(prepared.stages[i].num_inliers,
                      raw.stages[i].num_inliers);
            EXPECT_EQ(prepared.stages[i].termination,
                      raw.stages[i].termination);
            EXPECT_NEAR(prepared.stages[i].fitness_score,
                        raw.stages[i].fitness_score, 1e-9);
            EXPECT_NEAR(prepared.stages[i].inlier_ratio,
                        raw.stages[i].inlier_ratio, 1e-12);
        }
    };

    const std::array<Eigen::Isometry3d, 2> guesses = {
        Eigen::Isometry3d::Identity(),
        createPose(0.2, -0.1, 0.05, 0.0, 0.0, 0.05)
    };
    for (const auto& guess : guesses) {
        const auto raw = matcher_->alignCloud(target_cloud, source_cloud, guess);
        const auto prepared =
            matcher_->alignPrepared(prepared_target, prepared_source, guess);
        expect_equivalent(raw, prepared);
    }

    auto wide = matcher_->getSettings();
    wide.max_correspondence_distance *= 1.5;
    wide.max_iterations += 5;
    const auto raw_wide =
        matcher_->alignCloud(target_cloud, source_cloud, guesses[1], wide);
    const auto prepared_wide = matcher_->alignPrepared(
        prepared_target, prepared_source, guesses[1], wide);
    expect_equivalent(raw_wide, prepared_wide);
}

TEST_F(PointCloudMatcherTest, FixedPoseEvaluationMeasuresSelectedPoseWithoutOptimizing) {
    config_.icp_refine_use_gicp = false;
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
    const auto cloud = createPlaneCloud(2000, 10.0, 0.0);
    const auto target = matcher_->prepareTargetCloud(cloud);
    const auto source = matcher_->prepareSourceCloud(cloud);
    const auto endpoint = matcher_->alignPrepared(target, source, Eigen::Isometry3d::Identity());
    ASSERT_TRUE(endpoint.converged);
    ASSERT_LT(endpoint.fitness_score, 1e-9);

    Eigen::Isometry3d selected_pose = Eigen::Isometry3d::Identity();
    selected_pose.translation().z() = 0.5;
    const auto evaluated = matcher_->evaluatePreparedPose(
        target, source, selected_pose, endpoint.metric);
    EXPECT_EQ(evaluated.termination, MatchTermination::FixedPoseEvaluation);
    EXPECT_FALSE(evaluated.converged);
    EXPECT_EQ(evaluated.iterations, 0u);
    EXPECT_TRUE(evaluated.stages.empty());
    EXPECT_TRUE(evaluated.T_target_source.isApprox(selected_pose, 1e-12));
    EXPECT_NEAR(evaluated.fitness_score, 0.125, 1e-8);
    EXPECT_NEAR(evaluated.inlier_ratio, 1.0, 1e-12);
    EXPECT_NEAR(evaluated.optimizer_error, evaluated.num_inliers * 0.125, 1e-6);
    EXPECT_TRUE(evaluated.information.allFinite());
    EXPECT_GT(evaluated.information(2, 2), 0.0);

}

TEST_F(PointCloudMatcherTest, SourceRefinementDoesNotNeedTargetNormals) {
    config_.icp_refine_use_gicp = true;
    config_.num_threads = 1;
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
    const auto cloud = createPlaneCloud(2000, 10.0, 0.0);
    const auto target = matcher_->prepareTargetCloud(cloud);
    const auto source = matcher_->prepareSourceCloud(cloud);
    ASSERT_TRUE(source.has_refine_cloud);
    ASSERT_TRUE(target.has_refine_level);
    ASSERT_EQ(source.refine_cloud->size(), target.refine_level.cloud->size());
    for (std::size_t i = 0; i < source.refine_cloud->size(); ++i) {
        // Storage order is independent; compare features at the same point.
        std::size_t target_index;
        double distance;
        ASSERT_EQ(target.refine_level.kdtree->nearest_neighbor_search(
            source.refine_cloud->point(i), &target_index, &distance), 1u);
        EXPECT_TRUE(source.refine_cloud->point(i).isApprox(
            target.refine_level.cloud->point(target_index), 1e-12));
        EXPECT_TRUE(source.refine_cloud->cov(i).isApprox(
            registration_detail::TargetView{*target.refine_level.cloud}.covariance(target_index), 1e-12));
    }
    auto redundant_source = source;
    redundant_source.refine_cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>(
        *target.refine_level.cloud);
    auto& redundant = *redundant_source.refine_cloud;
    redundant.covs.resize(redundant.size());
    for (size_t i = 0; i < redundant.size(); ++i)
        redundant.cov(i) = registration_detail::covarianceFromNormal(redundant.normal(i));
    const auto guess = createPose(0.0, 0.0, 0.05, 0.0, 0.0, 0.0);
    const auto lean = matcher_->alignPrepared(target, source, guess);
    const auto with_normals = matcher_->alignPrepared(target, redundant_source, guess);
    ASSERT_TRUE(lean.converged);
    ASSERT_FALSE(lean.stages.empty());
    EXPECT_EQ(lean.stages.back().stage, "refine_gicp");
    EXPECT_EQ(lean.converged, with_normals.converged);
    EXPECT_TRUE(lean.T_target_source.isApprox(with_normals.T_target_source, 1e-12));
    EXPECT_NEAR(lean.fitness_score, with_normals.fitness_score, 1e-12);
    EXPECT_EQ(lean.num_inliers, with_normals.num_inliers);
}

TEST_F(PointCloudMatcherTest, ResumedConfirmationSolvesFreshGeometryAndRejectsMissingSupport) {
    config_.num_threads = 1;
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
    const auto cloud = createPlaneCloud(2000, 10.0, 0.0);
    const auto target = matcher_->prepareTargetCloud(cloud);
    const auto displaced = matcher_->prepareSourceCloud(
        transformCloud(cloud, createPose(0, 0, -0.12, 0, 0, 0)));
    for (const auto type : {small_gicp::RegistrationSetting::PLANE_ICP,
                            small_gicp::RegistrationSetting::GICP}) {
        MatchMetric metric{type,
            type == small_gicp::RegistrationSetting::GICP
                ? config_.icp_refine_downsampling_resolution
                : config_.gicp_downsampling_resolution,
            config_.icp_refine_max_correspondence_distance};
        const auto current = matcher_->resumePrepared(
            target, displaced, Eigen::Isometry3d::Identity(), metric);
        ASSERT_TRUE(current.success);
        EXPECT_TRUE(current.converged);
        EXPECT_NEAR(current.T_target_source.translation().z(), 0.12, 1e-5);
        const auto unsupported = matcher_->resumePrepared(
            target, displaced, createPose(0, 0, 5, 0, 0, 0), metric);
        EXPECT_FALSE(unsupported.success);
        metric.resolution = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(matcher_->resumePrepared(target, displaced,
            Eigen::Isometry3d::Identity(), metric).termination, MatchTermination::Invalid);
    }
}

TEST_F(PointCloudMatcherTest, FixedPoseEvaluationUsesObjectiveAndCorrespondenceGate) {
    const auto cloud = createPlaneCloud(2000, 10.0, 0.0);
    const auto target = matcher_->prepareTargetCloud(cloud);
    const auto source = matcher_->prepareSourceCloud(cloud);
    MatchMetric metric{small_gicp::RegistrationSetting::GICP,
                       config_.icp_refine_downsampling_resolution,
                       config_.gicp_max_correspondence_distance};
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().z() = 0.5;
    const auto gicp = matcher_->evaluatePreparedPose(target, source, pose, metric);
    ASSERT_EQ(gicp.termination, MatchTermination::FixedPoseEvaluation);
    EXPECT_GT(gicp.fitness_score, 0.125);
    EXPECT_TRUE(gicp.T_target_source.isApprox(pose));

    metric.max_correspondence_distance = 0.25;
    const auto rejected = matcher_->evaluatePreparedPose(target, source, pose, metric);
    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.num_inliers, 0u);
    EXPECT_EQ(rejected.termination, MatchTermination::Invalid);
    EXPECT_TRUE(rejected.information.isZero());
    pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(matcher_->evaluatePreparedPose(target, source, pose, metric).termination,
              MatchTermination::Invalid);
    EXPECT_EQ(matcher_->evaluatePreparedPose({}, source, Eigen::Isometry3d::Identity(), metric).termination,
              MatchTermination::Invalid);
}

// 测试信息矩阵输出
TEST_F(PointCloudMatcherTest, InformationMatrixOutput) {
    auto cloud = createPlaneCloud(2000);
    
    auto target_kf = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(), cloud);
    auto source_kf = Keyframe::create(1, 0.1, Eigen::Isometry3d::Identity(), cloud);
    
    auto result = matcher_->align(target_kf, source_kf);
    
    EXPECT_TRUE(result.success);
    
    // 信息矩阵应该是正定的 (对角线元素为正)
    for (int i = 0; i < 6; ++i) {
        EXPECT_GE(result.information(i, i), 0.0);
    }
}

}  // namespace test
}  // namespace n3mapping
