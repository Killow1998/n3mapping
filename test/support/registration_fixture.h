#pragma once
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping::test {
class RegistrationFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.gicp_downsampling_resolution = 0.5;
        config_.gicp_max_correspondence_distance = 2.0;
        config_.gicp_max_iterations = 30;
        config_.gicp_fitness_threshold = 0.3;
        config_.gicp_num_neighbors = 20;
        config_.num_threads = 4;

        matcher_ = std::make_unique<PointCloudMatcher>(config_);
    }

    void TearDown() override {
        matcher_.reset();
    }

    Keyframe::PointCloudT::Ptr createPlaneCloud(size_t num_points = 1000,
                                                  double size = 10.0,
                                                  double noise = 0.01) {
        Keyframe::PointCloudT::Ptr cloud(new Keyframe::PointCloudT());
        cloud->points.resize(num_points);

        std::mt19937 rng(42);  // 固定种子以保证可重复性
        std::uniform_real_distribution<float> pos_dist(-size/2, size/2);
        std::normal_distribution<float> noise_dist(0.0f, static_cast<float>(noise));

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = pos_dist(rng);
            cloud->points[i].y = pos_dist(rng);
            cloud->points[i].z = noise_dist(rng);  // 平面 z=0 加噪声
            cloud->points[i].intensity = 1.0f;
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Keyframe::PointCloudT::Ptr createSphereCloud(size_t num_points = 1000,
                                                   double radius = 5.0,
                                                   double noise = 0.01) {
        Keyframe::PointCloudT::Ptr cloud(new Keyframe::PointCloudT());
        cloud->points.resize(num_points);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> phi_dist(0.0f, M_PI);
        std::normal_distribution<float> noise_dist(0.0f, static_cast<float>(noise));

        for (size_t i = 0; i < num_points; ++i) {
            float theta = theta_dist(rng);
            float phi = phi_dist(rng);
            float r = static_cast<float>(radius) + noise_dist(rng);

            cloud->points[i].x = r * std::sin(phi) * std::cos(theta);
            cloud->points[i].y = r * std::sin(phi) * std::sin(theta);
            cloud->points[i].z = r * std::cos(phi);
            cloud->points[i].intensity = 1.0f;
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Keyframe::PointCloudT::Ptr transformCloud(const Keyframe::PointCloudT::Ptr& cloud,
                                                const Eigen::Isometry3d& transform) {
        Keyframe::PointCloudT::Ptr transformed(new Keyframe::PointCloudT());
        transformed->points.resize(cloud->size());

        for (size_t i = 0; i < cloud->size(); ++i) {
            Eigen::Vector3d pt(cloud->points[i].x,
                               cloud->points[i].y,
                               cloud->points[i].z);
            Eigen::Vector3d pt_transformed = transform * pt;

            transformed->points[i].x = static_cast<float>(pt_transformed.x());
            transformed->points[i].y = static_cast<float>(pt_transformed.y());
            transformed->points[i].z = static_cast<float>(pt_transformed.z());
            transformed->points[i].intensity = cloud->points[i].intensity;
        }

        transformed->width = cloud->size();
        transformed->height = 1;
        transformed->is_dense = true;
        return transformed;
    }

    Eigen::Isometry3d createPose(double x, double y, double z,
                                  double roll = 0, double pitch = 0, double yaw = 0) {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(x, y, z);

        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

        pose.rotate(yawAngle * pitchAngle * rollAngle);
        return pose;
    }

    bool posesAreClose(const Eigen::Isometry3d& pose1,
                       const Eigen::Isometry3d& pose2,
                       double trans_tol = 0.1,
                       double rot_tol = 0.05) {
        // 平移误差
        double trans_error = (pose1.translation() - pose2.translation()).norm();

        // 旋转误差 (角度)
        Eigen::Matrix3d R_diff = pose1.rotation().transpose() * pose2.rotation();
        double rot_error = std::acos(std::min(1.0, std::max(-1.0,
            (R_diff.trace() - 1.0) / 2.0)));

        return trans_error < trans_tol && rot_error < rot_tol;
    }

    Config config_;
    std::unique_ptr<PointCloudMatcher> matcher_;
};
}  // namespace n3mapping::test
