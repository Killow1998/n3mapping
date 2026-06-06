// MapSerializer: Protobuf-based map serialization/deserialization and global PCD export.
#include "n3mapping/map_serializer.h"

#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include "n3mapping/pcl_compat.h"

namespace n3mapping {

namespace {
constexpr const char* MAP_VERSION = "2.3.0";
constexpr uint32_t MAX_POINTS_PER_KEYFRAME = 5000000u;
constexpr uint64_t MAX_DESCRIPTOR_VALUES = 1000000u;

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

std::string buildRhpdSchema(const Config& config) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6)
       << "dim=" << RHPD_DIM
       << ";submap_mode=causal"
       << ";submap_kf_radius=" << config.rhpd_submap_kf_radius
       << ";submap_voxel_size=" << config.rhpd_submap_voxel_size
       << ";azimuth_bins=" << RHPDescriptor::Params{}.azimuth_bins
       << ";v2=" << config.rhpd_v2_enable
       << ";v3=" << config.rhpd_v3_enable
       << ";v3_visibility_bins=" << RHPDescriptor::Params{}.v3_visibility_bins
       << ";v3_free_space_margin_m=" << RHPDescriptor::Params{}.v3_free_space_margin_m
       << ";negative_space=" << config.rhpd_enable_negative_space
       << ";vertical_tokens=" << config.rhpd_enable_vertical_tokens
       << ";pca_confidence=" << config.rhpd_enable_pca_confidence
       << ";max_range=" << config.rhpd_max_range
       << ";z_min=" << config.rhpd_z_min
       << ";z_max=" << config.rhpd_z_max;
    return ss.str();
}

pcl::PointCloud<pcl::PointXYZI>::Ptr maybeVoxelizeRhpdCloud(
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

bool isFinitePoseProto(const n3mapping::Pose3D& proto) {
    const double values[] = {
        proto.tx(), proto.ty(), proto.tz(),
        proto.qx(), proto.qy(), proto.qz(), proto.qw()
    };
    for (double value : values) {
        if (!std::isfinite(value)) return false;
    }
    const double q_norm2 = proto.qx() * proto.qx() + proto.qy() * proto.qy() +
                           proto.qz() * proto.qz() + proto.qw() * proto.qw();
    return std::isfinite(q_norm2) && q_norm2 > 1e-12;
}

bool isFinitePose(const Eigen::Isometry3d& pose) {
    return pose.matrix().array().isFinite().all();
}

bool isFinitePoint(const pcl::PointXYZI& pt) {
    return std::isfinite(pt.x) && std::isfinite(pt.y) &&
           std::isfinite(pt.z) && std::isfinite(pt.intensity);
}

uint64_t countFinitePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!cloud) return 0;
    uint64_t count = 0;
    for (const auto& pt : cloud->points) {
        if (isFinitePoint(pt)) ++count;
    }
    return count;
}

bool atomicWriteProto(const std::string& filepath, const n3mapping::N3Map& map_proto) {
    const std::filesystem::path final_path(filepath);
    const std::filesystem::path tmp_path = final_path.string() + ".tmp";
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs.is_open()) return false;
        if (!map_proto.SerializeToOstream(&ofs)) {
            ofs.close();
            std::filesystem::remove(tmp_path);
            return false;
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::filesystem::remove(tmp_path);
            return false;
        }
    }
    std::filesystem::rename(tmp_path, final_path);
    return true;
}
}  // namespace

MapSerializer::MapSerializer(const Config& config) : config_(config) {}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer) {
    static const std::vector<core::DenseTrajectoryPose> kEmptyDenseTrajectory;
    const core::DenseTrajectoryMetadata metadata;
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer, kEmptyDenseTrajectory, metadata);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer,
                            const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory) {
    core::DenseTrajectoryMetadata metadata;
    metadata.source = dense_optimized_trajectory.empty() ? "none" : "native";
    metadata.degraded = dense_optimized_trajectory.empty();
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer, dense_optimized_trajectory, metadata);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer,
                            const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory,
                            const core::DenseTrajectoryMetadata& dense_trajectory_metadata) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        n3mapping::N3Map map_proto;
        auto* meta = map_proto.mutable_metadata();
        meta->set_version(MAP_VERSION);
        meta->set_creation_timestamp(std::time(nullptr));
        meta->set_map_frame(config_.world_frame);
        meta->set_body_frame(config_.body_frame);
        auto keyframes = keyframe_manager.getAllKeyframes();
        if (keyframes.empty()) {
            LOG(ERROR) << "[MapSerializer] Refuse to save map with no keyframes.";
            return false;
        }
        std::unordered_set<int64_t> saved_keyframe_ids;
        saved_keyframe_ids.reserve(keyframes.size());
        for (const auto& kf : keyframes) {
            if (!kf || !kf->cloud || kf->cloud->empty()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe with missing or empty cloud.";
                return false;
            }
            if (kf->id < 0) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe with negative id=" << kf->id;
                return false;
            }
            if (!std::isfinite(kf->timestamp) ||
                !isFinitePose(kf->pose_odom) ||
                !isFinitePose(kf->pose_optimized)) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with non-finite timestamp or pose.";
                return false;
            }
            if (kf->cloud->size() > MAX_POINTS_PER_KEYFRAME) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with too many points=" << kf->cloud->size();
                return false;
            }
            const uint64_t finite_points = countFinitePoints(kf->cloud);
            if (finite_points == 0 || finite_points != kf->cloud->size()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with invalid finite point count=" << finite_points;
                return false;
            }
            saved_keyframe_ids.insert(kf->id);
        }
        for (const auto& pose : dense_optimized_trajectory) {
            if (!std::isfinite(pose.timestamp) || !isFinitePose(pose.pose_world_lidar)) {
                LOG(ERROR) << "[MapSerializer] Refuse to save non-finite dense trajectory pose seq=" << pose.seq;
                return false;
            }
        }
        meta->set_num_keyframes(keyframes.size());
        meta->set_rhpd_schema(buildRhpdSchema(config_));
        std::string dense_source = dense_trajectory_metadata.source.empty()
            ? (dense_optimized_trajectory.empty() ? "none" : "native")
            : dense_trajectory_metadata.source;
        bool dense_degraded = dense_trajectory_metadata.degraded;
        if (dense_optimized_trajectory.empty()) {
            dense_source = "none";
            dense_degraded = true;
        } else if (dense_source == "none") {
            LOG(ERROR) << "[MapSerializer] Refuse to save non-empty dense trajectory with source=none.";
            return false;
        }
        meta->set_dense_trajectory_source(dense_source);
        meta->set_dense_trajectory_degraded(dense_degraded);
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
            Eigen::VectorXd save_rhpd;
            const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
            auto rhpd_cloud = keyframe_manager.buildCausalSubmapInRootFrame(kf->id, submap_radius, kf->id);
            rhpd_cloud = maybeVoxelizeRhpdCloud(rhpd_cloud, config_.rhpd_submap_voxel_size);
            if (rhpd_cloud && !rhpd_cloud->empty()) {
                save_rhpd = loop_detector.computeRHPD(rhpd_cloud);
            }
            if (save_rhpd.size() == RHPD_DIM && !save_rhpd.isZero()) {
                rhpdToProto(save_rhpd, kp->mutable_rhpd_descriptor());
            } else if (kf->rhpd_descriptor.size() == RHPD_DIM && !kf->rhpd_descriptor.isZero()) {
                rhpdToProto(kf->rhpd_descriptor, kp->mutable_rhpd_descriptor());
            } else if (rhpd_map.count(kf->id) && rhpd_map[kf->id].size() == RHPD_DIM && !rhpd_map[kf->id].isZero()) {
                rhpdToProto(rhpd_map[kf->id], kp->mutable_rhpd_descriptor());
            }
        }
        auto edges = optimizer.getEdges();
        int n_odom = 0, n_loop = 0;
        for (const auto& e : edges) {
            if (!isFinitePose(e.measurement) || !e.information.array().isFinite().all()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save malformed edge from="
                           << e.from_id << " to=" << e.to_id;
                return false;
            }
            if (saved_keyframe_ids.find(e.from_id) == saved_keyframe_ids.end() ||
                saved_keyframe_ids.find(e.to_id) == saved_keyframe_ids.end()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save edge with missing endpoint from="
                           << e.from_id << " to=" << e.to_id;
                return false;
            }
            edgeToProto(e, map_proto.add_edges());
            (e.type == EdgeType::ODOMETRY) ? ++n_odom : ++n_loop;
        }
        for (const auto& pose : dense_optimized_trajectory) {
            denseTrajectoryToProto(pose, map_proto.add_dense_optimized_trajectory());
        }
        meta->set_num_odometry_edges(n_odom);
        meta->set_num_loop_edges(n_loop);
        return atomicWriteProto(filepath, map_proto);
    } catch (...) { return false; }
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer, nullptr);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer, dense_optimized_trajectory, nullptr);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            core::DenseTrajectoryMetadata* dense_trajectory_metadata) {
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
        const std::string expected_rhpd_schema = buildRhpdSchema(config_);
        const std::string file_rhpd_schema = map_proto.metadata().rhpd_schema();
        if (file_rhpd_schema.empty()) {
            force_rebuild_rhpd = true;
            LOG(INFO) << "[MapSerializer] Missing RHPD schema, RHPD will be rebuilt.";
        } else if (file_rhpd_schema != expected_rhpd_schema) {
            force_rebuild_rhpd = true;
            LOG(INFO) << "[MapSerializer] RHPD schema mismatch, RHPD will be rebuilt."
                      << " file=" << file_rhpd_schema
                      << " expected=" << expected_rhpd_schema;
        }

        keyframe_manager.clear(); loop_detector.clear(); optimizer.clear();
        std::vector<Keyframe::Ptr> keyframes;
        keyframes.reserve(map_proto.keyframes_size());
        std::unordered_set<int64_t> loaded_keyframe_ids;
        loaded_keyframe_ids.reserve(map_proto.keyframes_size());
        for (int i = 0; i < map_proto.keyframes_size(); ++i) {
            auto kf = protoToKeyframe(map_proto.keyframes(i));
            if (!kf) {
                LOG(WARNING) << "[MapSerializer] Skip malformed keyframe index=" << i;
                continue;
            }
            if (!loaded_keyframe_ids.insert(kf->id).second) {
                LOG(ERROR) << "[MapSerializer] Reject map: duplicate keyframe id=" << kf->id;
                return false;
            }
            kf->is_from_loaded_map = true;
            keyframes.push_back(kf);
        }
        if (keyframes.empty()) {
            LOG(ERROR) << "[MapSerializer] Reject map: missing valid keyframes.";
            return false;
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
            loop_detector.loadRHPDDescriptors(rhpd_data);
            LOG(INFO) << "[MapSerializer] Loaded " << rhpd_data.size() << " RHPD descriptors from map.";
        } else {
            LOG(INFO) << "[MapSerializer] Rebuilding RHPD descriptors for " << keyframes.size() << " keyframes.";
            loop_detector.clearRHPD();
            int rebuilt_count = 0;
            for (const auto& kf : keyframes) {
                if (kf && kf->cloud && !kf->cloud->empty()) {
                    const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
                    pcl::PointCloud<pcl::PointXYZI>::Ptr rhpd_cloud = kf->cloud;
                    if (submap_radius > 0) {
                        rhpd_cloud = keyframe_manager.buildCausalSubmapInRootFrame(kf->id, submap_radius, kf->id);
                        if (!rhpd_cloud || rhpd_cloud->empty()) rhpd_cloud = kf->cloud;
                    }
                    rhpd_cloud = maybeVoxelizeRhpdCloud(rhpd_cloud, config_.rhpd_submap_voxel_size);
                    kf->rhpd_descriptor = loop_detector.addRHPD(kf->id, rhpd_cloud);
                    ++rebuilt_count;
                }
            }
            LOG(INFO) << "[MapSerializer] Rebuilt RHPD for " << rebuilt_count << " keyframes.";
        }

        std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
        std::unordered_set<int64_t> valid_keyframe_ids;
        valid_keyframe_ids.reserve(keyframes.size());
        for (const auto& kf : keyframes) {
            nodes.emplace_back(kf->id, kf->pose_optimized);
            valid_keyframe_ids.insert(kf->id);
        }
        std::vector<EdgeInfo> edges;
        edges.reserve(map_proto.edges_size());
        for (int i = 0; i < map_proto.edges_size(); ++i) {
            EdgeInfo edge = protoToEdge(map_proto.edges(i));
            if (!isFinitePose(edge.measurement) || !edge.information.array().isFinite().all()) {
                LOG(WARNING) << "[MapSerializer] Skip malformed edge index=" << i;
                continue;
            }
            if (valid_keyframe_ids.find(edge.from_id) == valid_keyframe_ids.end() ||
                valid_keyframe_ids.find(edge.to_id) == valid_keyframe_ids.end()) {
                LOG(WARNING) << "[MapSerializer] Skip edge with missing endpoint index=" << i
                             << " from=" << edge.from_id << " to=" << edge.to_id;
                continue;
            }
            edges.push_back(edge);
        }
        optimizer.loadGraph(nodes, edges);
        if (dense_trajectory_metadata) {
            dense_trajectory_metadata->source = "none";
            dense_trajectory_metadata->degraded = true;
        }
        if (dense_optimized_trajectory) {
            dense_optimized_trajectory->clear();
            dense_optimized_trajectory->reserve(
                map_proto.dense_optimized_trajectory_size() > 0
                    ? map_proto.dense_optimized_trajectory_size()
                    : keyframes.size());
            if (map_proto.dense_optimized_trajectory_size() > 0) {
                if (dense_trajectory_metadata) {
                    dense_trajectory_metadata->source = map_proto.metadata().dense_trajectory_source().empty()
                        ? "native"
                        : map_proto.metadata().dense_trajectory_source();
                    dense_trajectory_metadata->degraded = map_proto.metadata().dense_trajectory_degraded();
                }
                for (int i = 0; i < map_proto.dense_optimized_trajectory_size(); ++i) {
                    const auto& proto_pose = map_proto.dense_optimized_trajectory(i);
                    if (!isFinitePoseProto(proto_pose.pose_world_lidar()) ||
                        !std::isfinite(proto_pose.timestamp())) {
                        LOG(WARNING) << "[MapSerializer] Skip malformed dense trajectory pose index=" << i;
                        continue;
                    }
                    dense_optimized_trajectory->push_back(protoToDenseTrajectoryPose(proto_pose));
                }
            } else {
                if (dense_trajectory_metadata) {
                    dense_trajectory_metadata->source = "keyframe_fallback";
                    dense_trajectory_metadata->degraded = true;
                }
                uint64_t seq = 0;
                for (const auto& kf : keyframes) {
                    if (!kf || !isFinitePose(kf->pose_optimized)) continue;
                    core::DenseTrajectoryPose pose;
                    pose.seq = seq++;
                    pose.timestamp = kf->timestamp;
                    pose.pose_world_lidar = kf->pose_optimized;
                    dense_optimized_trajectory->push_back(pose);
                }
            }
        }
        return true;
    } catch (...) { return false; }
}

bool MapSerializer::saveGlobalMap(const std::string& filepath, const KeyframeManager& keyframe_manager, double voxel_size) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        auto global = buildGlobalMap(keyframe_manager, voxel_size);
        if (!global || global->empty()) return false;
        return pcl::io::savePCDFileBinary(filepath, *global) != -1;
    } catch (...) { return false; }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MapSerializer::buildGlobalMap(const KeyframeManager& keyframe_manager,
                                                                   double voxel_size) const {
    auto keyframes = keyframe_manager.getAllKeyframes();
    if (keyframes.empty()) return pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

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

    return global;
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
    if (proto.id() < 0) {
        LOG(WARNING) << "[MapSerializer] Reject keyframe with negative id: id=" << proto.id();
        return nullptr;
    }
    if (!std::isfinite(proto.timestamp())) {
        LOG(WARNING) << "[MapSerializer] Reject keyframe with non-finite timestamp: id=" << proto.id();
        return nullptr;
    }
    if (!isFinitePoseProto(proto.pose_odom()) || !isFinitePoseProto(proto.pose_optimized())) {
        LOG(WARNING) << "[MapSerializer] Reject keyframe with non-finite pose: id=" << proto.id();
        return nullptr;
    }
    auto kf = std::make_shared<Keyframe>();
    kf->id = proto.id(); kf->timestamp = proto.timestamp();
    kf->pose_odom = protoToPose(proto.pose_odom());
    kf->pose_optimized = protoToPose(proto.pose_optimized());
    if (!proto.has_cloud()) {
        LOG(WARNING) << "[MapSerializer] Reject keyframe with missing cloud: id=" << proto.id();
        return nullptr;
    }
    kf->cloud = protoToPointCloud(proto.cloud());
    if (!kf->cloud || kf->cloud->empty()) {
        LOG(WARNING) << "[MapSerializer] Reject keyframe with malformed or empty cloud: id=" << proto.id();
        return nullptr;
    }
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

void MapSerializer::denseTrajectoryToProto(const core::DenseTrajectoryPose& pose,
                                           n3mapping::DenseTrajectoryPose* proto) {
    proto->set_seq(pose.seq);
    proto->set_timestamp(pose.timestamp);
    poseToProto(pose.pose_world_lidar, proto->mutable_pose_world_lidar());
}

core::DenseTrajectoryPose MapSerializer::protoToDenseTrajectoryPose(const n3mapping::DenseTrajectoryPose& proto) {
    core::DenseTrajectoryPose pose;
    pose.seq = proto.seq();
    pose.timestamp = proto.timestamp();
    pose.pose_world_lidar = protoToPose(proto.pose_world_lidar());
    return pose;
}

void MapSerializer::poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto) {
    proto->set_tx(pose.translation().x()); proto->set_ty(pose.translation().y()); proto->set_tz(pose.translation().z());
    Eigen::Quaterniond q(pose.rotation());
    proto->set_qx(q.x()); proto->set_qy(q.y()); proto->set_qz(q.z()); proto->set_qw(q.w());
}

Eigen::Isometry3d MapSerializer::protoToPose(const n3mapping::Pose3D& proto) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    if (!isFinitePoseProto(proto)) {
        LOG(WARNING) << "[MapSerializer] Invalid pose proto, using identity.";
        return pose;
    }
    pose.translation() = Eigen::Vector3d(proto.tx(), proto.ty(), proto.tz());
    Eigen::Quaterniond q(proto.qw(), proto.qx(), proto.qy(), proto.qz());
    q.normalize();
    pose.linear() = q.toRotationMatrix();
    return pose;
}

void MapSerializer::pointCloudToProto(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, n3mapping::PointCloudData* proto) {
    uint32_t count = 0;
    for (const auto& pt : cloud->points) {
        if (!isFinitePoint(pt)) continue;
        ++count;
    }
    proto->set_num_points(count);
    proto->mutable_points()->Reserve(static_cast<int>(count * 4u));
    for (const auto& pt : cloud->points) {
        if (!isFinitePoint(pt)) continue;
        proto->add_points(pt.x); proto->add_points(pt.y); proto->add_points(pt.z); proto->add_points(pt.intensity);
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MapSerializer::protoToPointCloud(const n3mapping::PointCloudData& proto) {
    const uint64_t required_values = static_cast<uint64_t>(proto.num_points()) * 4u;
    if (proto.num_points() > MAX_POINTS_PER_KEYFRAME ||
        required_values > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        proto.points_size() != static_cast<int>(required_values)) {
        LOG(WARNING) << "[MapSerializer] Reject malformed point cloud: num_points="
                     << proto.num_points() << " values=" << proto.points_size();
        return nullptr;
    }
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->resize(proto.num_points());
    for (uint32_t i = 0; i < proto.num_points(); ++i) {
        cloud->points[i].x = proto.points(i*4); cloud->points[i].y = proto.points(i*4+1);
        cloud->points[i].z = proto.points(i*4+2); cloud->points[i].intensity = proto.points(i*4+3);
        if (!isFinitePoint(cloud->points[i])) {
            LOG(WARNING) << "[MapSerializer] Reject point cloud with non-finite point at index=" << i;
            return nullptr;
        }
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
    const uint64_t required_values = static_cast<uint64_t>(proto.rows()) * static_cast<uint64_t>(proto.cols());
    if (proto.rows() == 0 || proto.cols() == 0 ||
        required_values > MAX_DESCRIPTOR_VALUES ||
        required_values > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        proto.values_size() != static_cast<int>(required_values)) {
        LOG(WARNING) << "[MapSerializer] Reject malformed ScanContext descriptor: rows="
                     << proto.rows() << " cols=" << proto.cols()
                     << " values=" << proto.values_size();
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd desc(proto.rows(), proto.cols());
    int idx = 0;
    for (unsigned int i = 0; i < proto.rows(); ++i) {
        for (unsigned int j = 0; j < proto.cols(); ++j) {
            const double value = proto.values(idx++);
            if (!std::isfinite(value)) {
                LOG(WARNING) << "[MapSerializer] Reject ScanContext descriptor with non-finite value.";
                return Eigen::MatrixXd();
            }
            desc(i, j) = value;
        }
    }
    return desc;
}

void MapSerializer::informationToProto(const Eigen::Matrix<double, 6, 6>& info, n3mapping::InformationMatrix* proto) {
    proto->mutable_values()->Reserve(21);
    for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) proto->add_values(info(i, j));
}

Eigen::Matrix<double, 6, 6> MapSerializer::protoToInformation(const n3mapping::InformationMatrix& proto) {
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
    if (proto.values_size() != 21) {
        LOG(WARNING) << "[MapSerializer] Malformed information matrix: values="
                     << proto.values_size() << ", using identity.";
        return info;
    }
    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            const double value = proto.values(idx);
            if (!std::isfinite(value)) {
                LOG(WARNING) << "[MapSerializer] Malformed information matrix: non-finite value.";
                return Eigen::Matrix<double, 6, 6>::Identity();
            }
            info(i, j) = value;
            if (i != j) info(j, i) = value;
            ++idx;
        }
    }
    return info;
}

void MapSerializer::rhpdToProto(const Eigen::VectorXd& rhpd, n3mapping::RHPDDescriptor* proto) {
    proto->Clear();
    proto->set_dim(static_cast<uint32_t>(rhpd.size()));
    proto->mutable_values()->Reserve(rhpd.size());
    for (int i = 0; i < rhpd.size(); ++i) proto->add_values(rhpd(i));
}

Eigen::VectorXd MapSerializer::protoToRhpd(const n3mapping::RHPDDescriptor& proto) {
    if (proto.dim() != static_cast<uint32_t>(RHPD_DIM) || proto.values_size() != static_cast<int>(proto.dim())) {
        LOG(WARNING) << "[MapSerializer] Reject malformed RHPD descriptor: dim="
                     << proto.dim() << " expected=" << RHPD_DIM
                     << " values=" << proto.values_size();
        return Eigen::VectorXd();
    }
    Eigen::VectorXd rhpd = Eigen::VectorXd::Zero(proto.dim());
    for (uint32_t i = 0; i < proto.dim(); ++i) {
        if (!std::isfinite(proto.values(i))) {
            LOG(WARNING) << "[MapSerializer] Reject malformed RHPD descriptor with non-finite value.";
            return Eigen::VectorXd();
        }
        rhpd(i) = proto.values(i);
    }
    return rhpd;
}

} // namespace n3mapping
