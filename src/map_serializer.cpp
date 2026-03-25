// MapSerializer: Protobuf-based map serialization/deserialization and global PCD export.
#include "n3mapping/map_serializer.h"

#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/memory.h>

namespace n3mapping {

namespace {
constexpr const char* MAP_VERSION = "2.0.0";

struct SemVer {
    int major = 0;
    int minor = 0;
    int patch = 0;
};

bool parseSemVer(const std::string& version, SemVer* out) {
    if (!out) return false;
    std::stringstream ss(version);
    std::string major_str;
    std::string minor_str;
    std::string patch_str;
    if (!std::getline(ss, major_str, '.')) return false;
    if (!std::getline(ss, minor_str, '.')) return false;
    if (!std::getline(ss, patch_str, '.')) return false;
    try {
        out->major = std::stoi(major_str);
        out->minor = std::stoi(minor_str);
        out->patch = std::stoi(patch_str);
    } catch (...) {
        return false;
    }
    return true;
}

int compareSemVer(const SemVer& a, const SemVer& b) {
    if (a.major != b.major) return (a.major < b.major) ? -1 : 1;
    if (a.minor != b.minor) return (a.minor < b.minor) ? -1 : 1;
    if (a.patch != b.patch) return (a.patch < b.patch) ? -1 : 1;
    return 0;
}
}  // namespace

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

        bool force_rebuild_rhpd = false;
        const std::string file_version = map_proto.metadata().version();
        SemVer file_semver;
        SemVer code_semver;
        const bool file_version_ok = parseSemVer(file_version, &file_semver);
        const bool code_version_ok = parseSemVer(MAP_VERSION, &code_semver);
        if (file_version_ok && code_version_ok) {
            const int version_cmp = compareSemVer(file_semver, code_semver);
            if (version_cmp > 0) {
                LOG(ERROR) << "[MapSerializer] Reject map: file version " << file_version
                           << " is newer than supported " << MAP_VERSION;
                return false;
            }
            if (version_cmp < 0) {
                force_rebuild_rhpd = true;
                LOG(INFO) << "[MapSerializer] Map version " << file_version
                          << " is older than " << MAP_VERSION << ", RHPD will be rebuilt.";
            }
        } else {
            force_rebuild_rhpd = true;
            LOG(WARNING) << "[MapSerializer] Invalid map version format (file=" << file_version
                         << ", code=" << MAP_VERSION << "), force rebuilding RHPD.";
        }

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
        // Restore/rebuild RHPD descriptors.
        std::vector<std::pair<int64_t, Eigen::VectorXd>> rhpd_data;
        bool rhpd_missing_or_invalid = false;
        for (const auto& kf : keyframes) {
            if (!kf) continue;
            if (kf->rhpd_descriptor.size() == RHPD_DIM) {
                rhpd_data.emplace_back(kf->id, kf->rhpd_descriptor);
            } else {
                rhpd_missing_or_invalid = true;
            }
        }

        if (!force_rebuild_rhpd && !rhpd_missing_or_invalid) {
            loop_detector.getRHPDManager().loadAll(rhpd_data);
            LOG(INFO) << "[MapSerializer] Loaded " << rhpd_data.size() << " RHPD descriptors from map.";
        } else {
            LOG(INFO) << "[MapSerializer] Rebuilding RHPD descriptors for " << keyframes.size() << " keyframes.";
            loop_detector.getRHPDManager().clear();
            int rebuilt_count = 0;
            for (const auto& kf : keyframes) {
                if (kf && kf->cloud && !kf->cloud->empty()) {
                    kf->rhpd_descriptor = loop_detector.addRHPD(kf->id, kf->cloud);
                    ++rebuilt_count;
                }
            }
            LOG(INFO) << "[MapSerializer] Rebuilt RHPD for " << rebuilt_count << " keyframes.";
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

        auto global = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        for (const auto& kf : keyframes) {
            if (!kf || !kf->cloud || kf->cloud->empty()) continue;
            pcl::PointCloud<pcl::PointXYZI> transformed;
            pcl::transformPointCloud(*kf->cloud, transformed, kf->pose_optimized.matrix().cast<float>());
            global->points.insert(global->points.end(), transformed.points.begin(), transformed.points.end());
        }

        global->width = static_cast<uint32_t>(global->points.size());
        global->height = 1;
        global->is_dense = false;

        if (voxel_size > 0.0) {
            struct VoxelKey {
                int x;
                int y;
                int z;
                bool operator==(const VoxelKey& other) const {
                    return x == other.x && y == other.y && z == other.z;
                }
            };
            struct VoxelKeyHash {
                size_t operator()(const VoxelKey& key) const {
                    size_t h = std::hash<int>()(key.x);
                    h ^= std::hash<int>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    return h;
                }
            };
            struct VoxelAccum {
                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
                double si = 0.0;
                uint32_t count = 0;
            };

            std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxel_map;
            voxel_map.reserve(global->points.size());

            const double inv_leaf = 1.0 / voxel_size;
            for (const auto& pt : global->points) {
                VoxelKey key{
                    static_cast<int>(std::floor(static_cast<double>(pt.x) * inv_leaf)),
                    static_cast<int>(std::floor(static_cast<double>(pt.y) * inv_leaf)),
                    static_cast<int>(std::floor(static_cast<double>(pt.z) * inv_leaf))
                };
                auto& acc = voxel_map[key];
                acc.sx += pt.x;
                acc.sy += pt.y;
                acc.sz += pt.z;
                acc.si += pt.intensity;
                ++acc.count;
            }

            auto ds = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
            ds->points.reserve(voxel_map.size());
            for (const auto& entry : voxel_map) {
                const auto& acc = entry.second;
                if (acc.count == 0) continue;
                const float inv_count = 1.0f / static_cast<float>(acc.count);
                pcl::PointXYZI p;
                p.x = static_cast<float>(acc.sx * inv_count);
                p.y = static_cast<float>(acc.sy * inv_count);
                p.z = static_cast<float>(acc.sz * inv_count);
                p.intensity = static_cast<float>(acc.si * inv_count);
                ds->points.push_back(p);
            }

            ds->width = static_cast<uint32_t>(ds->points.size());
            ds->height = 1;
            ds->is_dense = false;
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
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
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
