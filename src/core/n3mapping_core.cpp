#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>

#include <pcl/filters/voxel_grid.h>
#include <pcl/memory.h>

#include "n3mapping/core/n3mapping_session.h"

namespace n3mapping {
namespace core {
namespace {

double toSeconds(const TimeStamp& stamp) {
    return static_cast<double>(stamp.nsec) * 1e-9;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr maybeVoxelize(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    double voxel_size) {
    if (!cloud || cloud->empty() || voxel_size <= 1e-4) return cloud;
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.setInputCloud(cloud);
    auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    voxel.filter(*filtered);
    return filtered->empty() ? cloud : filtered;
}

}  // namespace

class N3MappingCore::Impl {
public:
    explicit Impl(const Config& config)
        : session_(config) {}

    MappingOutput processLioFrame(const LioFrame& frame) {
        MappingOutput output;
        output.T_world_lidar = frame.T_world_lidar;
        if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
            return output;
        }
        auto& keyframes = session_.keyframes();
        auto& loop_detector = session_.loopDetector();
        auto& optimizer = session_.graphOptimizer();
        const auto& config = session_.config();

        if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
            return output;
        }

        const int64_t kf_id = keyframes.addKeyframe(
            toSeconds(frame.stamp), frame.T_world_lidar, frame.undistorted_cloud);
        output.accepted_keyframe = true;
        output.keyframe_id = kf_id;

        auto sc = loop_detector.addDescriptor(kf_id, frame.undistorted_cloud);
        keyframes.updateDescriptor(kf_id, sc);

        auto kf = keyframes.getKeyframe(kf_id);
        if (kf) {
            const int submap_radius = std::max(0, config.rhpd_submap_kf_radius);
            auto rhpd_cloud = keyframes.buildCausalSubmapInRootFrame(kf_id, submap_radius, kf_id);
            rhpd_cloud = maybeVoxelize(rhpd_cloud, config.rhpd_submap_voxel_size);
            kf->rhpd_descriptor = loop_detector.addRHPD(kf_id, rhpd_cloud);
        }

        if (kf_id == 0) {
            optimizer.addPriorFactor(kf_id, frame.T_world_lidar);
        } else {
            auto prev_kf = keyframes.getKeyframe(kf_id - 1);
            if (prev_kf) {
                EdgeInfo edge;
                edge.from_id = kf_id - 1;
                edge.to_id = kf_id;
                edge.measurement = prev_kf->pose_odom.inverse() * frame.T_world_lidar;
                edge.information = frame.covariance.inverse();
                if (!edge.information.allFinite()) {
                    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
                    edge.information.block<3, 3>(0, 0) *=
                        1.0 / (config.odom_noise_position * config.odom_noise_position);
                    edge.information.block<3, 3>(3, 3) *=
                        1.0 / (config.odom_noise_rotation * config.odom_noise_rotation);
                }
                edge.type = EdgeType::ODOMETRY;
                optimizer.addOdometryEdge(edge);
            }
        }

        optimizer.incrementalOptimize();
        keyframes.updateOptimizedPoses(optimizer.getOptimizedPoses());
        if (optimizer.hasNode(kf_id)) {
            try {
                output.T_world_lidar = optimizer.getOptimizedPose(kf_id);
            } catch (...) {
                output.T_world_lidar = frame.T_world_lidar;
            }
        }
        return output;
    }

    RelocResult relocalize(const LioFrame& frame) {
        if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
            return RelocResult{};
        }
        return session_.worldLocalizing().relocalize(frame.undistorted_cloud, frame.T_world_lidar);
    }

    bool saveMap(const std::string& path) {
        return session_.saveCurrentMap(path);
    }

    bool loadMap(const std::string& path) {
        return session_.loadMapForLocalization(path);
    }

private:
    N3MappingSession session_;
};

N3MappingCore::N3MappingCore(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

N3MappingCore::~N3MappingCore() = default;

MappingOutput N3MappingCore::processLioFrame(const LioFrame& frame) {
    return impl_->processLioFrame(frame);
}

RelocResult N3MappingCore::relocalize(const LioFrame& frame) {
    return impl_->relocalize(frame);
}

bool N3MappingCore::saveMap(const std::string& path) {
    return impl_->saveMap(path);
}

bool N3MappingCore::loadMap(const std::string& path) {
    return impl_->loadMap(path);
}

}  // namespace core
}  // namespace n3mapping
