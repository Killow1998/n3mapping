// Offline factorial diagnosis only. Truth seeds never enter the runtime core.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <glog/logging.h>
#include <json/json.h>
#include <pcl/io/pcd_io.h>

#include "n3mapping/config.h"
#include "n3mapping/point_cloud_matcher.h"

namespace {
using Cloud = n3mapping::PointCloudMatcher::PointCloudT;

Cloud::Ptr readCloud(const std::filesystem::path& path) {
    auto cloud = pcl::make_shared<Cloud>();
    if (pcl::io::loadPCDFile(path.string(), *cloud) != 0 || cloud->empty())
        throw std::runtime_error("Missing or empty point cloud: " + path.string());
    return cloud;
}

Eigen::Isometry3d readTruthSeed(const Json::Value& job) {
    if (job["truth"].size() != 16)
        throw std::runtime_error("Expected a row-major 4x4 diagnostic truth seed");
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    for (int i = 0; i < 16; ++i)
        pose.matrix()(i / 4, i % 4) = job["truth"][i].asDouble();
    if (!pose.matrix().allFinite() ||
        !(pose.rotation().transpose() * pose.rotation()).isIdentity(1e-6) ||
        std::abs(pose.rotation().determinant() - 1.0) > 1e-6 ||
        !pose.matrix().row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-9))
        throw std::runtime_error("Invalid diagnostic truth pose");
    return pose;
}
}  // namespace

int main(int argc, char** argv) {
    const bool help = argc == 2 && std::string(argv[1]) == "--help";
    if (argc != 5 || help) {
        std::cerr << "ORIGINAL_MAP.pcd REFERENCE.pcd JOBS.json OUTPUT.csv\n"
                  << "Offline truth-seeded ICP only; not accepted localization.\n";
        return help ? 0 : 2;
    }
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    try {
        if (std::filesystem::exists(argv[4]))
            throw std::runtime_error("Refusing to overwrite existing results");
        Json::Value jobs;
        std::ifstream input(argv[3]);
        if (!(input >> jobs) || !jobs.isArray() || jobs.empty())
            throw std::runtime_error("Missing diagnostic jobs");
        auto config = n3mapping::makeProductLocalizationConfig("diagnostic", "diagnostic");
        config.num_threads = 4;
        n3mapping::PointCloudMatcher matcher(config);
        std::ofstream output(argv[4]);
        if (!output) throw std::runtime_error("Cannot create results");
        output << std::setprecision(17)
               << "id,target,source,success,converged,iterations,inlier_ratio,"
                  "fitness,translation_m,rotation_deg";
        for (int i = 0; i < 16; ++i) output << ",pose" << i;
        output << '\n';
        for (int target = 0; target < 2; ++target) {
            const auto prepared = matcher.prepareTargetCloud(readCloud(argv[1 + target]));
            for (const auto& job : jobs) {
                const auto truth = readTruthSeed(job);
                for (const auto* source : {"fastlio", "native_deskew"}) {
                    const auto path = std::filesystem::path(job["path"].asString()) /
                                      (std::string(source) + ".pcd");
                    const auto query = matcher.prepareSourceCloud(readCloud(path));
                    const auto result = matcher.alignPrepared(prepared, query, truth);
                    const Eigen::Isometry3d delta = truth.inverse() * result.T_target_source;
                    output << job["id"].asString() << ','
                           << (target ? "reference" : "original") << ',' << source << ','
                           << result.success << ',' << result.converged << ','
                           << result.iterations << ',' << result.inlier_ratio << ','
                           << result.fitness_score << ',' << delta.translation().norm() << ','
                           << Eigen::AngleAxisd(delta.rotation()).angle() * 180 / M_PI;
                    for (int i = 0; i < 16; ++i)
                        output << ',' << result.T_target_source.matrix()(i / 4, i % 4);
                    output << std::endl;
                }
            }
        }
        if (!output) throw std::runtime_error("Failed writing results");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
