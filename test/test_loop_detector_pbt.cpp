#include "n3mapping/pcl_compat.h"

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <random>
#include "n3mapping/loop_detector.h"

namespace n3mapping {
namespace test {

inline std::map<int64_t, Keyframe::Ptr> lineOfKeyframes(int count,
                                                        double spacing_m = 1.0) {
    std::map<int64_t, Keyframe::Ptr> keyframes;
    for (int i = 0; i < count; ++i) {
        auto kf = std::make_shared<Keyframe>();
        kf->pose_odom = Eigen::Isometry3d::Identity();
        kf->pose_odom.translation().x() = i * spacing_m;
        keyframes[i] = kf;
    }
    return keyframes;
}

class LoopDetectorPBTTest : public ::testing::Test {
protected:
    void SetUp() override {

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        rng_.seed(static_cast<unsigned int>(seed));

        std::cout << "Random seed: " << seed << std::endl;
    }

    void TearDown() override {
    }

    Keyframe::PointCloudT::Ptr generateRandomCloud(size_t min_points = 500,
                                                    size_t max_points = 2000) {
        std::uniform_int_distribution<size_t> num_dist(min_points, max_points);
        std::uniform_real_distribution<double> pos_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> height_dist(-5.0, 5.0);

        size_t num_points = num_dist(rng_);
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        cloud->points.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            pcl::PointXYZI pt;
            pt.x = static_cast<float>(pos_dist(rng_));
            pt.y = static_cast<float>(pos_dist(rng_));
            pt.z = static_cast<float>(height_dist(rng_));
            pt.intensity = 1.0f;
            cloud->points.push_back(pt);
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Keyframe::PointCloudT::Ptr generateCircularCloud(double radius = 20.0) {
        std::uniform_int_distribution<size_t> num_dist(800, 1500);
        std::uniform_real_distribution<double> height_dist(-3.0, 3.0);
        std::uniform_real_distribution<double> radius_noise(-2.0, 2.0);

        size_t num_points = num_dist(rng_);
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        cloud->points.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            double angle = 2.0 * M_PI * static_cast<double>(i) / num_points;
            double r = radius + radius_noise(rng_);

            pcl::PointXYZI pt;
            pt.x = static_cast<float>(r * std::cos(angle));
            pt.y = static_cast<float>(r * std::sin(angle));
            pt.z = static_cast<float>(height_dist(rng_));
            pt.intensity = 1.0f;
            cloud->points.push_back(pt);
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Config generateRandomConfig() {
        Config config;

        std::uniform_int_distribution<int> path_length_dist(5, 20);
        std::uniform_int_distribution<int> candidates_dist(3, 20);
        std::uniform_real_distribution<double> threshold_dist(0.1, 0.5);

        config.loop_min_path_length_m = path_length_dist(rng_);
        config.sc_num_candidates = candidates_dist(rng_);
        config.sc_dist_threshold = threshold_dist(rng_);
        config.num_threads = 2;

        return config;
    }

    std::mt19937 rng_;
};

TEST_F(LoopDetectorPBTTest, Property4_ExcludeShortPathsAcrossSamplingDensities) {
    constexpr int NUM_ITERATIONS = 100;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {

        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);
        const double spacing_m = std::ldexp(1.0, std::uniform_int_distribution<int>(-2, 1)(rng_));
        const int minimum_segments = static_cast<int>(std::ceil(config.loop_min_path_length_m / spacing_m));

        std::uniform_int_distribution<int> num_frames_dist(
            minimum_segments + 5, minimum_segments + 50);
        int num_frames = num_frames_dist(rng_);

        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }

        std::uniform_int_distribution<int> query_dist(
            minimum_segments, num_frames - 1);
        int query_id = query_dist(rng_);

        auto candidates = detector->detectLoopCandidates(
            query_id, lineOfKeyframes(num_frames, spacing_m));
        ASSERT_FALSE(candidates.empty()) << "Iteration: " << iter;

        for (const auto& candidate : candidates) {
            EXPECT_GE(candidate.path_length_m, config.loop_min_path_length_m);
            EXPECT_DOUBLE_EQ(candidate.path_length_m, (query_id - candidate.match_id) * spacing_m);
            EXPECT_GE(candidate.match_id, 0);
            EXPECT_EQ(candidate.query_id, query_id);
        }
    }
}

TEST_F(LoopDetectorPBTTest, Property7_DescriptorDimensionInvariance) {
    constexpr int NUM_ITERATIONS = 100;

    Config config;
    auto detector = std::make_unique<LoopDetector>(config);

    auto [expected_rows, expected_cols] = detector->getDescriptorDimensions();

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {

        Keyframe::PointCloudT::Ptr cloud;

        std::uniform_int_distribution<int> cloud_type_dist(0, 2);
        int cloud_type = cloud_type_dist(rng_);

        switch (cloud_type) {
            case 0:

                cloud = generateRandomCloud(100, 5000);
                break;
            case 1:

                cloud = generateCircularCloud();
                break;
            case 2: {

                cloud = generateRandomCloud(50, 200);
                break;
            }
        }

        auto descriptor = detector->computeHeading(cloud);

        ASSERT_TRUE(descriptor);
        EXPECT_EQ(descriptor->rings * RHPDHeadingParams::channels, expected_rows)
            << "Iteration: " << iter
            << ", Cloud type: " << cloud_type
            << ", Cloud size: " << cloud->size();

        EXPECT_EQ(descriptor->sectors, expected_cols)
            << "Iteration: " << iter
            << ", Cloud type: " << cloud_type
            << ", Cloud size: " << cloud->size();
    }
}

TEST_F(LoopDetectorPBTTest, InsufficientTravelReturnsEmptyEvenWithManyFrames) {
    constexpr int NUM_ITERATIONS = 50;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);

        const int num_frames = std::uniform_int_distribution<int>(51, 100)(rng_);
        const double spacing_m = config.loop_min_path_length_m / num_frames;

        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }

        auto candidates = detector->detectLoopCandidates(num_frames - 1,
                                                        lineOfKeyframes(num_frames, spacing_m));

        EXPECT_TRUE(candidates.empty())
            << "Iteration: " << iter
            << ", Num frames: " << num_frames
            << ", Minimum travel in metres: " << config.loop_min_path_length_m;
    }
}

TEST_F(LoopDetectorPBTTest, CandidateCountLimit) {
    constexpr int NUM_ITERATIONS = 50;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        Config config = generateRandomConfig();
        auto detector = std::make_unique<LoopDetector>(config);

        const int num_frames = static_cast<int>(config.loop_min_path_length_m) +
            config.sc_num_candidates + 20;

        for (int i = 0; i < num_frames; ++i) {
            auto cloud = generateCircularCloud();
            detector->addDescriptor(i, cloud);
        }

        auto candidates = detector->detectLoopCandidates(num_frames - 1,
                                                        lineOfKeyframes(num_frames));

        EXPECT_LE(static_cast<int>(candidates.size()), config.sc_num_candidates)
            << "Iteration: " << iter
            << ", Candidates: " << candidates.size()
            << ", Max: " << config.sc_num_candidates;
    }
}

}
}
