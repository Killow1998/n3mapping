// MapSerializer: Protobuf-based map serialization/deserialization and global PCD export.
#include "n3mapping/map_serializer.h"

#include <filesystem>
#include <fstream>
#include <boost/make_shared.hpp>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace n3mapping {

namespace { constexpr const char* MAP_VERSION = "1.0.0"; }

MapSerializer::MapSerializer(const Config& config) : config_(config) {}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        n3mapping::N3Map map_proto;
        auto* meta = map_proto.mutable_metadata();
        meta->set_version(MAP_VERSION);
        meta->set_creation_timestamp(std::time(nullptr));
        auto keyframes = keyframe_manager.getAllKeyframes();
        meta->set_num_keyframes(keyframes.size());
        auto desc_vec = loop_detector.getDescriptors();
        std::map<int64_t, Eigen::MatrixXd> desc_map;
        for (const auto& item : desc_vec) desc_map[item.first] = item.second;
        // Collect RHPD descriptors from the manager
        auto rhpd_vec = loop_detector.getRHPDManager().getAll();
        std::map<int64_t, Eigen::VectorXd> rhpd_map;
        for (const auto& item : rhpd_vec) rhpd_map[item.first] = item.second;
        for (const auto& kf : keyframes) {
            auto* kp = map_proto.add_keyframes();
            keyframeToProto(kf, kp);
            if (kp->sc_descriptor().values_size() == 0 && desc_map.count(kf->id))
                descriptorToProto(desc_map[kf->id], kp->mutable_sc_descriptor());
            if (kp->rhpd_descriptor().values_size() == 0 && rhpd_map.count(kf->id))
                rhpdToProto(rhpd_map[kf->id], kp->mutable_rhpd_descriptor());
        }
        auto edges = optimizer.getEdges();
        int n_odom = 0, n_loop = 0;
        for (const auto& e : edges) {
            edgeToProto(e, map_proto.add_edges());
            (e.type == EdgeType::ODOMETRY) ? ++n_odom : ++n_loop;
        }
        meta->set_num_odometry_edges(n_odom);
        meta->set_num_loop_edges(n_loop);
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open() || !map_proto.SerializeToOstream(&ofs)) return false;
        return true;
    } catch (...) { return false; }
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer) {
    try {
        if (!std::filesystem::exists(filepath)) return false;
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) return false;
        n3mapping::N3Map map_proto;
        if (!map_proto.ParseFromIstream(&ifs)) return false;
        keyframe_manager.clear(); loop_detector.clear(); optimizer.clear();
        std::vector<Keyframe::Ptr> keyframes;
        for (int i = 0; i < map_proto.keyframes_size(); ++i) {
            auto kf = protoToKeyframe(map_proto.keyframes(i));
            kf->is_from_loaded_map = true;
            keyframes.push_back(kf);
        }
        keyframe_manager.loadKeyframes(keyframes);
        std::vector<std::pair<int64_t, Eigen::MatrixXd>> descriptors;
        for (const auto& kf : keyframes)
            if (kf->sc_descriptor.size() > 0) descriptors.emplace_back(kf->id, kf->sc_descriptor);
        loop_detector.loadDescriptors(descriptors);
        // Restore RHPD descriptors into RHPDManager
        std::vector<std::pair<int64_t, Eigen::VectorXd>> rhpd_data;
        for (const auto& kf : keyframes)
            if (kf->rhpd_descriptor.size() > 0) rhpd_data.emplace_back(kf->id, kf->rhpd_descriptor);
        if (!rhpd_data.empty()) {
            loop_detector.getRHPDManager().loadAll(rhpd_data);
            LOG(INFO) << "[MapSerializer] Loaded " << rhpd_data.size() << " RHPD descriptors from map.";
        } else {
            // Old map without RHPD — recompute from keyframe clouds
            LOG(INFO) << "[MapSerializer] No RHPD in map, recomputing from " << keyframes.size() << " keyframe clouds...";
            int count = 0;
            for (const auto& kf : keyframes) {
                if (kf && kf->cloud && !kf->cloud->empty()) {
                    kf->rhpd_descriptor = loop_detector.addRHPD(kf->id, kf->cloud);
                    ++count;
                }
            }
            LOG(INFO) << "[MapSerializer] Recomputed RHPD for " << count << " keyframes.";
        }
        std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
        for (const auto& kf : keyframes) nodes.emplace_back(kf->id, kf->pose_optimized);
        std::vector<EdgeInfo> edges;
        for (int i = 0; i < map_proto.edges_size(); ++i) edges.push_back(protoToEdge(map_proto.edges(i)));
        optimizer.loadGraph(nodes, edges);
        return true;
    } catch (...) { return false; }
}

bool MapSerializer::saveGlobalMap(const std::string& filepath, const KeyframeManager& keyframe_manager, double voxel_size) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        auto keyframes = keyframe_manager.getAllKeyframes();
        if (keyframes.empty()) return false;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> transformed(keyframes.size());
#pragma omp parallel for num_threads(config_.num_threads) schedule(dynamic)
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const auto& kf = keyframes[i];
            if (!kf || !kf->cloud || kf->cloud->empty()) { transformed[i] = nullptr; continue; }
            auto tc = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
            pcl::transformPointCloud(*kf->cloud, *tc, kf->pose_optimized.matrix().cast<float>());
            transformed[i] = tc;
        }
        auto global = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        for (const auto& c : transformed) if (c && !c->empty()) *global += *c;
        if (voxel_size > 0.0) {
            auto ds = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
            pcl::VoxelGrid<pcl::PointXYZI> vf;
            vf.setInputCloud(global); vf.setLeafSize(voxel_size, voxel_size, voxel_size); vf.filter(*ds);
            global = ds;
        }
        return pcl::io::savePCDFileBinary(filepath, *global) != -1;
    } catch (...) { return false; }
}

void MapSerializer::keyframeToProto(const Keyframe::Ptr& kf, n3mapping::KeyframeProto* proto) {
    proto->set_id(kf->id); proto->set_timestamp(kf->timestamp);
    poseToProto(kf->pose_odom, proto->mutable_pose_odom());
    poseToProto(kf->pose_optimized, proto->mutable_pose_optimized());
    if (kf->cloud && !kf->cloud->empty()) pointCloudToProto(kf->cloud, proto->mutable_cloud());
    if (kf->sc_descriptor.size() > 0) descriptorToProto(kf->sc_descriptor, proto->mutable_sc_descriptor());
    if (kf->rhpd_descriptor.size() > 0) rhpdToProto(kf->rhpd_descriptor, proto->mutable_rhpd_descriptor());
}

Keyframe::Ptr MapSerializer::protoToKeyframe(const n3mapping::KeyframeProto& proto) {
    auto kf = std::make_shared<Keyframe>();
    kf->id = proto.id(); kf->timestamp = proto.timestamp();
    kf->pose_odom = protoToPose(proto.pose_odom());
    kf->pose_optimized = protoToPose(proto.pose_optimized());
    if (proto.has_cloud()) kf->cloud = protoToPointCloud(proto.cloud());
    if (proto.has_sc_descriptor()) kf->sc_descriptor = protoToDescriptor(proto.sc_descriptor());
    if (proto.has_rhpd_descriptor() && proto.rhpd_descriptor().values_size() > 0)
        kf->rhpd_descriptor = protoToRhpd(proto.rhpd_descriptor());
    return kf;
}

void MapSerializer::edgeToProto(const EdgeInfo& edge, n3mapping::EdgeProto* proto) {
    proto->set_from_id(edge.from_id); proto->set_to_id(edge.to_id);
    poseToProto(edge.measurement, proto->mutable_measurement());
    informationToProto(edge.information, proto->mutable_information());
    proto->set_type(edge.type == EdgeType::ODOMETRY ? n3mapping::EdgeProto::ODOMETRY : n3mapping::EdgeProto::LOOP);
}

EdgeInfo MapSerializer::protoToEdge(const n3mapping::EdgeProto& proto) {
    EdgeInfo edge;
    edge.from_id = proto.from_id(); edge.to_id = proto.to_id();
    edge.measurement = protoToPose(proto.measurement());
    edge.information = protoToInformation(proto.information());
    edge.type = (proto.type() == n3mapping::EdgeProto::ODOMETRY) ? EdgeType::ODOMETRY : EdgeType::LOOP;
    return edge;
}

void MapSerializer::poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto) {
    proto->set_tx(pose.translation().x()); proto->set_ty(pose.translation().y()); proto->set_tz(pose.translation().z());
    Eigen::Quaterniond q(pose.rotation());
    proto->set_qx(q.x()); proto->set_qy(q.y()); proto->set_qz(q.z()); proto->set_qw(q.w());
}

Eigen::Isometry3d MapSerializer::protoToPose(const n3mapping::Pose3D& proto) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(proto.tx(), proto.ty(), proto.tz());
    pose.linear() = Eigen::Quaterniond(proto.qw(), proto.qx(), proto.qy(), proto.qz()).toRotationMatrix();
    return pose;
}

void MapSerializer::pointCloudToProto(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, n3mapping::PointCloudData* proto) {
    proto->set_num_points(cloud->size());
    proto->mutable_points()->Reserve(cloud->size() * 4);
    for (const auto& pt : cloud->points) { proto->add_points(pt.x); proto->add_points(pt.y); proto->add_points(pt.z); proto->add_points(pt.intensity); }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MapSerializer::protoToPointCloud(const n3mapping::PointCloudData& proto) {
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->resize(proto.num_points());
    for (uint32_t i = 0; i < proto.num_points(); ++i) {
        cloud->points[i].x = proto.points(i*4); cloud->points[i].y = proto.points(i*4+1);
        cloud->points[i].z = proto.points(i*4+2); cloud->points[i].intensity = proto.points(i*4+3);
    }
    cloud->width = cloud->size(); cloud->height = 1; cloud->is_dense = false;
    return cloud;
}

void MapSerializer::descriptorToProto(const Eigen::MatrixXd& desc, n3mapping::ScanContextDescriptor* proto) {
    proto->set_rows(desc.rows()); proto->set_cols(desc.cols());
    proto->mutable_values()->Reserve(desc.size());
    for (int i = 0; i < desc.rows(); ++i) for (int j = 0; j < desc.cols(); ++j) proto->add_values(desc(i, j));
}

Eigen::MatrixXd MapSerializer::protoToDescriptor(const n3mapping::ScanContextDescriptor& proto) {
    Eigen::MatrixXd desc(proto.rows(), proto.cols());
    int idx = 0;
    for (unsigned int i = 0; i < proto.rows(); ++i) for (unsigned int j = 0; j < proto.cols(); ++j) desc(i, j) = proto.values(idx++);
    return desc;
}

void MapSerializer::informationToProto(const Eigen::Matrix<double, 6, 6>& info, n3mapping::InformationMatrix* proto) {
    proto->mutable_values()->Reserve(21);
    for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) proto->add_values(info(i, j));
}

Eigen::Matrix<double, 6, 6> MapSerializer::protoToInformation(const n3mapping::InformationMatrix& proto) {
    Eigen::Matrix<double, 6, 6> info;
    int idx = 0;
    for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) { info(i, j) = proto.values(idx); if (i != j) info(j, i) = proto.values(idx); ++idx; }
    return info;
}

void MapSerializer::rhpdToProto(const Eigen::VectorXd& rhpd, n3mapping::RHPDDescriptor* proto) {
    proto->set_dim(static_cast<uint32_t>(rhpd.size()));
    proto->mutable_values()->Reserve(rhpd.size());
    for (int i = 0; i < rhpd.size(); ++i) proto->add_values(rhpd(i));
}

Eigen::VectorXd MapSerializer::protoToRhpd(const n3mapping::RHPDDescriptor& proto) {
    Eigen::VectorXd rhpd(proto.dim());
    for (uint32_t i = 0; i < proto.dim() && i < static_cast<uint32_t>(proto.values_size()); ++i)
        rhpd(i) = proto.values(i);
    return rhpd;
}

} // namespace n3mapping
