#include "n3mapping/map_serializer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace n3mapping {

namespace {
constexpr const char* MAP_VERSION = "1.0.0";
}

MapSerializer::MapSerializer(const Config& config)
  : config_(config)
{
}

// ==================== 地图保存 ====================

bool
MapSerializer::saveMap(const std::string& filepath,
                       const KeyframeManager& keyframe_manager,
                       const LoopDetector& loop_detector,
                       const GraphOptimizer& optimizer)
{
    try {
        // 确保目录存在
        std::filesystem::path file_path(filepath);
        if (file_path.has_parent_path()) {
            std::filesystem::create_directories(file_path.parent_path());
        }

        n3mapping::N3Map map_proto;
        auto* metadata = map_proto.mutable_metadata();
        metadata->set_version(MAP_VERSION);
        metadata->set_creation_timestamp(std::time(nullptr));

        auto keyframes = keyframe_manager.getAllKeyframes();
        metadata->set_num_keyframes(keyframes.size());

        auto descriptors_vec = loop_detector.getDescriptors();
        std::map<int64_t, Eigen::MatrixXd> descriptor_map;
        for (const auto& item : descriptors_vec) {
            descriptor_map[item.first] = item.second;
        }

        for (const auto& kf : keyframes) {
            auto* kf_proto = map_proto.add_keyframes();
            keyframeToProto(kf, kf_proto);

            if (kf_proto->sc_descriptor().values_size() == 0 && descriptor_map.count(kf->id)) {
                descriptorToProto(descriptor_map[kf->id], kf_proto->mutable_sc_descriptor());
            }
        }

        auto edges = optimizer.getEdges();
        int num_odom = 0, num_loop = 0;
        for (const auto& edge : edges) {
            auto* edge_proto = map_proto.add_edges();
            edgeToProto(edge, edge_proto);
            if (edge.type == EdgeType::ODOMETRY)
                num_odom++;
            else
                num_loop++;
        }
        metadata->set_num_odometry_edges(num_odom);
        metadata->set_num_loop_edges(num_loop);

        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) return false;
        if (!map_proto.SerializeToOstream(&ofs)) return false;
        ofs.close();

        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool
MapSerializer::loadMap(const std::string& filepath, KeyframeManager& keyframe_manager, LoopDetector& loop_detector, GraphOptimizer& optimizer)
{
    try {
        // 检查文件存在性
        if (!std::filesystem::exists(filepath)) {
            LOG(ERROR) << "Map file does not exist: " << filepath;
            return false;
        }

        // 读取文件
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            LOG(ERROR) << "Failed to open map file: " << filepath;
            return false;
        }

        // 解析 Protobuf
        n3mapping::N3Map map_proto;
        if (!map_proto.ParseFromIstream(&ifs)) {
            LOG(ERROR) << "Failed to parse map file (corrupted?): " << filepath;
            return false;
        }

        ifs.close();

        // 验证版本兼容性
        if (map_proto.metadata().version() != MAP_VERSION) {
            LOG(INFO) << "Warning: Map version mismatch: file=" << map_proto.metadata().version() << ", current=" << MAP_VERSION;
        }

        // 清空现有数据
        keyframe_manager.clear();
        loop_detector.clear();
        optimizer.clear();

        // 加载关键帧
        std::vector<Keyframe::Ptr> keyframes;
        for (int i = 0; i < map_proto.keyframes_size(); ++i) {
            auto kf = protoToKeyframe(map_proto.keyframes(i));
            kf->is_from_loaded_map = true;
            keyframes.push_back(kf);
        }
        keyframe_manager.loadKeyframes(keyframes);

        // 加载 ScanContext 描述子
        std::vector<std::pair<int64_t, Eigen::MatrixXd>> descriptors;
        for (const auto& kf : keyframes) {
            if (kf->sc_descriptor.size() > 0) {
                descriptors.emplace_back(kf->id, kf->sc_descriptor);
            }
        }
        loop_detector.loadDescriptors(descriptors);

        // 加载约束边和节点到图优化器
        std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
        for (const auto& kf : keyframes) {
            nodes.emplace_back(kf->id, kf->pose_optimized);
        }

        std::vector<EdgeInfo> edges;
        for (int i = 0; i < map_proto.edges_size(); ++i) {
            edges.push_back(protoToEdge(map_proto.edges(i)));
        }

        optimizer.loadGraph(nodes, edges);

        LOG(INFO) << "Map loaded successfully from: " << filepath;
        LOG(INFO) << "  Keyframes: " << keyframes.size();
        LOG(INFO) << "  Edges: " << edges.size();

        return true;

    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception in loadMap: " << e.what();
        return false;
    }
}

bool
MapSerializer::saveGlobalMap(const std::string& filepath, const KeyframeManager& keyframe_manager, double voxel_size)
{
    try {
        // 确保目录存在
        std::filesystem::path file_path(filepath);
        if (file_path.has_parent_path()) {
            std::filesystem::create_directories(file_path.parent_path());
        }

        // 获取所有关键帧
        auto keyframes = keyframe_manager.getAllKeyframes();
        if (keyframes.empty()) {
            LOG(ERROR) << "No keyframes to save";
            return false;
        }

        LOG(INFO) << "Building global map from " << keyframes.size() << " keyframes...";

        // 并行变换所有关键帧点云
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> transformed_clouds(keyframes.size());

#pragma omp parallel for num_threads(config_.num_threads) schedule(dynamic)
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const auto& kf = keyframes[i];
            if (!kf || !kf->cloud || kf->cloud->empty()) {
                transformed_clouds[i] = nullptr;
                continue;
            }

            // 使用优化后位姿变换点云
            auto transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
            Eigen::Matrix4f transform = kf->pose_optimized.matrix().cast<float>();
            pcl::transformPointCloud(*kf->cloud, *transformed_cloud, transform);
            transformed_clouds[i] = transformed_cloud;
        }

        // 合并所有变换后的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& cloud : transformed_clouds) {
            if (cloud && !cloud->empty()) {
                *global_cloud += *cloud;
            }
        }

        LOG(INFO) << "Global cloud size before downsampling: " << global_cloud->size() << " points";

        // 下采样
        if (voxel_size > 0.0) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
            voxel_filter.setInputCloud(global_cloud);
            voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
            voxel_filter.filter(*downsampled_cloud);
            global_cloud = downsampled_cloud;

            LOG(INFO) << "Global cloud size after downsampling: " << global_cloud->size() << " points";
        }

        // 保存为 PCD 文件
        if (pcl::io::savePCDFileBinary(filepath, *global_cloud) == -1) {
            LOG(ERROR) << "Failed to save global map to: " << filepath;
            return false;
        }

        LOG(INFO) << "Global map saved successfully to: " << filepath;
        return true;

    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception in saveGlobalMap: " << e.what();
        return false;
    }
}

// ==================== Proto 转换辅助函数 ====================

void
MapSerializer::keyframeToProto(const Keyframe::Ptr& kf, n3mapping::KeyframeProto* proto)
{
    proto->set_id(kf->id);
    proto->set_timestamp(kf->timestamp);

    poseToProto(kf->pose_odom, proto->mutable_pose_odom());
    poseToProto(kf->pose_optimized, proto->mutable_pose_optimized());

    if (kf->cloud && !kf->cloud->empty()) {
        pointCloudToProto(kf->cloud, proto->mutable_cloud());
    }

    if (kf->sc_descriptor.size() > 0) {
        descriptorToProto(kf->sc_descriptor, proto->mutable_sc_descriptor());
    }
}

Keyframe::Ptr
MapSerializer::protoToKeyframe(const n3mapping::KeyframeProto& proto)
{
    auto kf = std::make_shared<Keyframe>();

    kf->id = proto.id();
    kf->timestamp = proto.timestamp();
    kf->pose_odom = protoToPose(proto.pose_odom());
    kf->pose_optimized = protoToPose(proto.pose_optimized());

    if (proto.has_cloud()) {
        kf->cloud = protoToPointCloud(proto.cloud());
    }

    if (proto.has_sc_descriptor()) {
        kf->sc_descriptor = protoToDescriptor(proto.sc_descriptor());
    }

    return kf;
}

void
MapSerializer::edgeToProto(const EdgeInfo& edge, n3mapping::EdgeProto* proto)
{
    proto->set_from_id(edge.from_id);
    proto->set_to_id(edge.to_id);

    poseToProto(edge.measurement, proto->mutable_measurement());
    informationToProto(edge.information, proto->mutable_information());

    if (edge.type == EdgeType::ODOMETRY) {
        proto->set_type(n3mapping::EdgeProto::ODOMETRY);
    } else {
        proto->set_type(n3mapping::EdgeProto::LOOP);
    }
}

EdgeInfo
MapSerializer::protoToEdge(const n3mapping::EdgeProto& proto)
{
    EdgeInfo edge;

    edge.from_id = proto.from_id();
    edge.to_id = proto.to_id();
    edge.measurement = protoToPose(proto.measurement());
    edge.information = protoToInformation(proto.information());

    if (proto.type() == n3mapping::EdgeProto::ODOMETRY) {
        edge.type = EdgeType::ODOMETRY;
    } else {
        edge.type = EdgeType::LOOP;
    }

    return edge;
}

void
MapSerializer::poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto)
{
    // 平移
    proto->set_tx(pose.translation().x());
    proto->set_ty(pose.translation().y());
    proto->set_tz(pose.translation().z());

    // 旋转 (四元数)
    // 注意：Eigen 构造函数是 Quaterniond(w, x, y, z)，但内部存储是 (x, y, z, w)
    Eigen::Quaterniond q(pose.rotation());
    proto->set_qx(q.x());
    proto->set_qy(q.y());
    proto->set_qz(q.z());
    proto->set_qw(q.w());
}

Eigen::Isometry3d
MapSerializer::protoToPose(const n3mapping::Pose3D& proto)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    // 平移
    pose.translation() = Eigen::Vector3d(proto.tx(), proto.ty(), proto.tz());

    // 旋转 (四元数)
    // Eigen 构造函数顺序是 (w, x, y, z)
    Eigen::Quaterniond q(proto.qw(), proto.qx(), proto.qy(), proto.qz());
    pose.linear() = q.toRotationMatrix();

    return pose;
}

void
MapSerializer::pointCloudToProto(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, n3mapping::PointCloudData* proto)
{
    proto->set_num_points(cloud->size());

    // 交错存储 x, y, z, intensity
    proto->mutable_points()->Reserve(cloud->size() * 4);
    for (const auto& pt : cloud->points) {
        proto->add_points(pt.x);
        proto->add_points(pt.y);
        proto->add_points(pt.z);
        proto->add_points(pt.intensity);
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr
MapSerializer::protoToPointCloud(const n3mapping::PointCloudData& proto)
{

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->resize(proto.num_points());

    // 从交错存储恢复
    for (uint32_t i = 0; i < proto.num_points(); ++i) {
        cloud->points[i].x = proto.points(i * 4 + 0);
        cloud->points[i].y = proto.points(i * 4 + 1);
        cloud->points[i].z = proto.points(i * 4 + 2);
        cloud->points[i].intensity = proto.points(i * 4 + 3);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;

    return cloud;
}

void
MapSerializer::descriptorToProto(const Eigen::MatrixXd& descriptor, n3mapping::ScanContextDescriptor* proto)
{
    proto->set_rows(descriptor.rows());
    proto->set_cols(descriptor.cols());

    // 按行优先存储
    proto->mutable_values()->Reserve(descriptor.size());
    for (int i = 0; i < descriptor.rows(); ++i) {
        for (int j = 0; j < descriptor.cols(); ++j) {
            proto->add_values(descriptor(i, j));
        }
    }
}

Eigen::MatrixXd
MapSerializer::protoToDescriptor(const n3mapping::ScanContextDescriptor& proto)
{
    Eigen::MatrixXd descriptor(proto.rows(), proto.cols());

    // 按行优先恢复
    int idx = 0;
    for (unsigned int i = 0; i < proto.rows(); ++i) {
        for (unsigned int j = 0; j < proto.cols(); ++j) {
            descriptor(i, j) = proto.values(idx++);
        }
    }

    return descriptor;
}

void
MapSerializer::informationToProto(const Eigen::Matrix<double, 6, 6>& info, n3mapping::InformationMatrix* proto)
{
    // 上三角存储 (21 个元素)
    proto->mutable_values()->Reserve(21);
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            proto->add_values(info(i, j));
        }
    }
}

Eigen::Matrix<double, 6, 6>
MapSerializer::protoToInformation(const n3mapping::InformationMatrix& proto)
{

    Eigen::Matrix<double, 6, 6> info;

    // 从上三角恢复
    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            info(i, j) = proto.values(idx);
            if (i != j) {
                info(j, i) = proto.values(idx); // 对称矩阵
            }
            idx++;
        }
    }

    return info;
}

} // namespace n3mapping
