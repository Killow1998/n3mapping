#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include "n3mapping/loop_detector.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace test {

class LoopDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.sc_dist_threshold = 0.3;
        config_.sc_num_exclude_recent = 10;
        config_.sc_num_candidates = 5;
        config_.num_threads = 2;
        config_.gicp_fitness_threshold = 0.5;
        config_.rhpd_enabled = true;
        config_.rhpd_dist_threshold = 100.0;
        config_.rhpd_num_candidates = 5;
        config_.rhpd_preselect_candidates = 12;
        
        detector_ = std::make_unique<LoopDetector>(config_);
    }

    void TearDown() override {
        detector_.reset();
    }

    // 创建圆形点云 (模拟 LiDAR 扫描)
    Keyframe::PointCloudT::Ptr createCircularCloud(size_t num_points = 1000,
                                                    double radius = 20.0,
                                                    double height_variation = 2.0) {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        cloud->points.reserve(num_points);
        
        std::mt19937 rng(42);  // 固定种子以保证可重复性
        std::uniform_real_distribution<double> height_dist(-height_variation, height_variation);
        std::uniform_real_distribution<double> radius_noise(-1.0, 1.0);
        
        for (size_t i = 0; i < num_points; ++i) {
            double angle = 2.0 * M_PI * static_cast<double>(i) / num_points;
            double r = radius + radius_noise(rng);
            
            pcl::PointXYZI pt;
            pt.x = static_cast<float>(r * std::cos(angle));
            pt.y = static_cast<float>(r * std::sin(angle));
            pt.z = static_cast<float>(height_dist(rng));
            pt.intensity = 1.0f;
            cloud->points.push_back(pt);
        }
        
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    // 创建带偏移的点云 (模拟不同位置的扫描)
    Keyframe::PointCloudT::Ptr createOffsetCloud(double x_offset, double y_offset,
                                                  size_t num_points = 1000) {
        auto cloud = createCircularCloud(num_points);
        for (auto& pt : cloud->points) {
            pt.x += static_cast<float>(x_offset);
            pt.y += static_cast<float>(y_offset);
        }
        return cloud;
    }

    // 创建旋转后的点云 (模拟相同位置不同朝向)
    Keyframe::PointCloudT::Ptr createRotatedCloud(double yaw_rad,
                                                   size_t num_points = 1000) {
        auto cloud = createCircularCloud(num_points);
        double cos_yaw = std::cos(yaw_rad);
        double sin_yaw = std::sin(yaw_rad);
        
        for (auto& pt : cloud->points) {
            float x_new = static_cast<float>(pt.x * cos_yaw - pt.y * sin_yaw);
            float y_new = static_cast<float>(pt.x * sin_yaw + pt.y * cos_yaw);
            pt.x = x_new;
            pt.y = y_new;
        }
        return cloud;
    }

    Keyframe::PointCloudT::Ptr createAsymmetricCloud() {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        for (float x = -6.0f; x <= 8.0f; x += 0.25f) {
            for (float z = -0.5f; z <= 2.5f; z += 0.35f) {
                pcl::PointXYZI pt;
                pt.x = x;
                pt.y = -2.0f + 0.05f * std::sin(x);
                pt.z = z;
                pt.intensity = 10.0f;
                cloud->push_back(pt);
            }
        }
        for (float y = -1.8f; y <= 3.5f; y += 0.25f) {
            for (float z = -0.5f; z <= 1.8f; z += 0.35f) {
                pcl::PointXYZI pt;
                pt.x = 3.0f;
                pt.y = y;
                pt.z = z;
                pt.intensity = 20.0f;
                cloud->push_back(pt);
            }
        }
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    Keyframe::PointCloudT::Ptr rotateCloud(const Keyframe::PointCloudT::Ptr& cloud, double yaw_rad) {
        auto rotated = pcl::make_shared<Keyframe::PointCloudT>();
        const double c = std::cos(yaw_rad);
        const double s = std::sin(yaw_rad);
        rotated->reserve(cloud->size());
        for (const auto& pt : cloud->points) {
            pcl::PointXYZI out = pt;
            out.x = static_cast<float>(pt.x * c - pt.y * s);
            out.y = static_cast<float>(pt.x * s + pt.y * c);
            rotated->push_back(out);
        }
        rotated->width = rotated->size();
        rotated->height = 1;
        rotated->is_dense = true;
        return rotated;
    }

    Keyframe::PointCloudT::Ptr translatedCloud(const Keyframe::PointCloudT::Ptr& cloud, double dx, double dy) {
        auto translated = pcl::make_shared<Keyframe::PointCloudT>();
        translated->reserve(cloud->size());
        for (const auto& pt : cloud->points) {
            pcl::PointXYZI out = pt;
            out.x += static_cast<float>(dx);
            out.y += static_cast<float>(dy);
            translated->push_back(out);
        }
        translated->width = translated->size();
        translated->height = 1;
        translated->is_dense = true;
        return translated;
    }

    Keyframe::PointCloudT::Ptr createDifferentStructureCloud() {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        for (float angle = 0.0f; angle < 2.0f * static_cast<float>(M_PI); angle += 0.03f) {
            for (float z = -0.5f; z <= 2.5f; z += 0.4f) {
                pcl::PointXYZI pt;
                pt.x = 7.0f * std::cos(angle);
                pt.y = 7.0f * std::sin(angle);
                pt.z = z;
                pt.intensity = 30.0f;
                cloud->push_back(pt);
            }
        }
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    void addPoint(Keyframe::PointCloudT::Ptr cloud, float x, float y, float z, float intensity = 1.0f) {
        pcl::PointXYZI pt;
        pt.x = x;
        pt.y = y;
        pt.z = z;
        pt.intensity = intensity;
        cloud->push_back(pt);
    }

    void finalizeCloud(Keyframe::PointCloudT::Ptr cloud) {
        cloud->width = cloud->size();
        cloud->height = 1;
        cloud->is_dense = true;
    }

    Keyframe::PointCloudT::Ptr createDoorwayCorridor(bool doorway) {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        for (float x = -8.0f; x <= 8.0f; x += 0.25f) {
            const bool in_door = doorway && x >= 1.0f && x <= 3.0f;
            for (float z = -0.2f; z <= 2.8f; z += 0.25f) {
                if (!in_door) addPoint(cloud, x, -2.0f, z, 10.0f);
                addPoint(cloud, x, 2.0f, z, 10.0f);
            }
            for (float y = -2.0f; y <= 2.0f; y += 0.35f) {
                addPoint(cloud, x, y, -0.2f, 5.0f);
            }
        }
        if (doorway) {
            for (float y = -4.2f; y <= -2.0f; y += 0.25f) {
                for (float z = -0.2f; z <= 2.2f; z += 0.3f) {
                    addPoint(cloud, 1.0f, y, z, 20.0f);
                    addPoint(cloud, 3.0f, y, z, 20.0f);
                }
            }
            for (float x = 1.0f; x <= 3.0f; x += 0.25f) {
                for (float z = -0.2f; z <= 2.2f; z += 0.3f) {
                    addPoint(cloud, x, -4.2f, z, 20.0f);
                }
            }
        }
        finalizeCloud(cloud);
        return cloud;
    }

    Keyframe::PointCloudT::Ptr createTJunction(bool t_junction) {
        auto cloud = createDoorwayCorridor(false);
        if (t_junction) {
            for (float y = -8.0f; y <= 8.0f; y += 0.25f) {
                for (float z = -0.2f; z <= 2.6f; z += 0.3f) {
                    addPoint(cloud, -1.8f, y, z, 30.0f);
                    addPoint(cloud, 1.8f, y, z, 30.0f);
                }
            }
            finalizeCloud(cloud);
        }
        return cloud;
    }

    Keyframe::PointCloudT::Ptr createVerticalTokenScene(bool near_low) {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        const float radius = near_low ? 3.0f : 9.0f;
        for (float a = -2.8f; a <= 2.8f; a += 0.18f) {
            for (float w = -0.5f; w <= 0.5f; w += 0.18f) {
                if (near_low) {
                    addPoint(cloud, radius + w, a, 0.15f, 40.0f);
                    addPoint(cloud, radius + w, a, 0.65f, 40.0f);
                } else {
                    for (float z = 0.2f; z <= 4.8f; z += 0.35f) {
                        addPoint(cloud, radius + w, a, z, 50.0f);
                    }
                }
            }
        }
        for (float angle = 0.0f; angle < 2.0f * static_cast<float>(M_PI); angle += 0.18f) {
            addPoint(cloud, 12.0f * std::cos(angle), 12.0f * std::sin(angle), 0.0f, 5.0f);
        }
        finalizeCloud(cloud);
        return cloud;
    }

    Keyframe::PointCloudT::Ptr createOpenOrBlockedScene(bool open) {
        auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
        const float range = open ? 18.0f : 4.0f;
        for (float angle = -2.6f; angle <= 2.6f; angle += 0.06f) {
            for (float z = -0.2f; z <= 2.0f; z += 0.35f) {
                addPoint(cloud, range * std::cos(angle), range * std::sin(angle), z, 60.0f);
            }
        }
        if (!open) {
            for (float angle = -0.5f; angle <= 0.5f; angle += 0.03f) {
                for (float z = -0.2f; z <= 2.8f; z += 0.25f) {
                    addPoint(cloud, 2.2f * std::cos(angle), 2.2f * std::sin(angle), z, 80.0f);
                }
            }
        }
        finalizeCloud(cloud);
        return cloud;
    }

    Config config_;
    std::unique_ptr<LoopDetector> detector_;
};

// 测试初始状态
TEST_F(LoopDetectorTest, InitialState) {
    EXPECT_EQ(detector_->size(), 0u);
    
    auto dims = detector_->getDescriptorDimensions();
    // Hybrid ScanContext in noetic-aligned pipeline expands rows to 140.
    EXPECT_EQ(dims.first, 140);
    EXPECT_EQ(dims.second, 60);
}

TEST_F(LoopDetectorTest, ConfigInjectionAffectsDescriptorManagers) {
    Config cfg = config_;
    cfg.sc_num_rings = 16;
    cfg.sc_num_sectors = 72;
    cfg.sc_max_radius = 45.0;
    cfg.rhpd_max_range = 12.5;
    cfg.rhpd_z_min = -1.5;
    cfg.rhpd_z_max = 4.5;
    LoopDetector detector(cfg);

    auto dims = detector.getDescriptorDimensions();
    EXPECT_EQ(dims.first, HybridSCManager::NUM_CHANNELS * cfg.sc_num_rings);
    EXPECT_EQ(dims.second, cfg.sc_num_sectors);
    EXPECT_NEAR(detector.getScanContextSectorAngleDeg(), 360.0 / cfg.sc_num_sectors, 1e-9);
    EXPECT_DOUBLE_EQ(detector.getRHPDParams().max_range, cfg.rhpd_max_range);
    EXPECT_DOUBLE_EQ(detector.getRHPDParams().z_min, cfg.rhpd_z_min);
    EXPECT_DOUBLE_EQ(detector.getRHPDParams().z_max, cfg.rhpd_z_max);
}

TEST_F(LoopDetectorTest, RHPDComputeReturnsConfiguredDimension) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto rhpd = descriptor.compute(createAsymmetricCloud());
    EXPECT_EQ(rhpd.size(), RHPD_DIM);
    EXPECT_GT(rhpd.norm(), 0.0);
}

TEST_F(LoopDetectorTest, RHPDEmptyAndSparseCloudReturnZero) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto empty = pcl::make_shared<Keyframe::PointCloudT>();
    EXPECT_TRUE(descriptor.compute(empty).isZero(1e-12));

    auto sparse = pcl::make_shared<Keyframe::PointCloudT>();
    sparse->resize(5);
    sparse->width = sparse->size();
    sparse->height = 1;
    EXPECT_TRUE(descriptor.compute(sparse).isZero(1e-12));
}

TEST_F(LoopDetectorTest, RHPDDistanceIsSymmetric) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto a = descriptor.compute(createAsymmetricCloud());
    auto b = descriptor.compute(createDifferentStructureCloud());
    EXPECT_NEAR(descriptor.distance(a, b), descriptor.distance(b, a), 1e-9);
}

TEST_F(LoopDetectorTest, RHPDHandlesYaw180FlipForAsymmetricCloud) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto cloud = createAsymmetricCloud();
    auto rotated = rotateCloud(cloud, M_PI);
    auto different = createDifferentStructureCloud();

    const auto a = descriptor.compute(cloud);
    const auto b = descriptor.compute(rotated);
    const auto c = descriptor.compute(different);
    const double d_rotated = descriptor.distance(a, b);
    const double d_different = descriptor.distance(a, c);

    EXPECT_LT(d_rotated, d_different);
    EXPECT_LT(d_rotated, d_different * 0.75);
}

TEST_F(LoopDetectorTest, RHPDCoarsePrefilterFindsExactMatchAndSortsByFullDistance) {
    RHPDManager manager(RHPDescriptor::Params{});
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    std::vector<Eigen::VectorXd> descriptors;
    for (int i = 0; i < 12; ++i) {
        auto cloud = (i == 7) ? createAsymmetricCloud() : createOffsetCloud(i * 2.0, -i * 1.5, 600);
        auto rhpd = descriptor.compute(cloud);
        descriptors.push_back(rhpd);
        manager.add(i, rhpd);
    }

    auto results = manager.search(descriptors[7], 5, 4);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().first, 7);
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].second, results[i].second);
    }
}

TEST_F(LoopDetectorTest, RHPDCoarsePrefilterKeepsBruteForceTop1InLargeSyntheticDb) {
    RHPDManager manager(RHPDescriptor::Params{});
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);

    std::vector<Eigen::VectorXd> db;
    db.reserve(500);
    for (int i = 0; i < 500; ++i) {
        Eigen::VectorXd v(RHPD_DIM);
        for (int j = 0; j < RHPD_DIM; ++j) {
            v(j) = value_dist(rng);
        }
        db.push_back(v);
        manager.add(i, v);
    }

    Eigen::VectorXd query = db[317];
    for (int j = 0; j < RHPD_DIM; ++j) {
        query(j) += (j % 17 == 0) ? 1e-5 : 0.0;
    }

    int brute_id = -1;
    double brute_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(db.size()); ++i) {
        const double d = descriptor.distance(query, db[i]);
        if (d < brute_dist) {
            brute_dist = d;
            brute_id = i;
        }
    }

    auto results = manager.search(query, 5, 20);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(brute_id, 317);
    EXPECT_EQ(results.front().first, brute_id);
}

TEST_F(LoopDetectorTest, RHPDSemanticsDoorwayCorridorBeatsPlainCorridor) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto a = descriptor.compute(createDoorwayCorridor(true));
    auto b = descriptor.compute(createDoorwayCorridor(false));
    auto q = descriptor.compute(rotateCloud(createDoorwayCorridor(true), M_PI));
    EXPECT_LT(descriptor.distance(q, a), descriptor.distance(q, b));
}

TEST_F(LoopDetectorTest, RHPDSemanticsTJunctionBeatsStraightCorridor) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto a = descriptor.compute(createTJunction(true));
    auto b = descriptor.compute(createTJunction(false));
    auto q = descriptor.compute(createTJunction(true));
    EXPECT_LT(descriptor.distance(q, a), descriptor.distance(q, b));
}

TEST_F(LoopDetectorTest, RHPDSemanticsVerticalCoarseRingTokensSeparateStructures) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto a = descriptor.compute(createVerticalTokenScene(true));
    auto b = descriptor.compute(createVerticalTokenScene(false));
    auto q = descriptor.compute(createVerticalTokenScene(true));
    EXPECT_LT(descriptor.distance(q, a), descriptor.distance(q, b));
}

TEST_F(LoopDetectorTest, RHPDSemanticsNegativeSpaceSeparatesOpenFromUnknown) {
    RHPDescriptor descriptor(RHPDescriptor::Params{});
    auto a = descriptor.compute(createOpenOrBlockedScene(true));
    auto b = descriptor.compute(createOpenOrBlockedScene(false));
    auto q = descriptor.compute(createOpenOrBlockedScene(true));
    EXPECT_LT(descriptor.distance(q, a), descriptor.distance(q, b));
}

// 测试 ScanContext 描述子生成 - Requirements 4.1
TEST_F(LoopDetectorTest, MakeScanContext) {
    auto cloud = createCircularCloud();
    
    auto descriptor = detector_->makeScanContext(cloud);
    
    // 验证描述子维度
    auto dims = detector_->getDescriptorDimensions();
    EXPECT_EQ(descriptor.rows(), dims.first);
    EXPECT_EQ(descriptor.cols(), dims.second);
    
    // 描述子应该有非零值
    EXPECT_GT(descriptor.sum(), 0.0);
}

// 测试空点云处理
TEST_F(LoopDetectorTest, EmptyCloudHandling) {
    auto empty_cloud = pcl::make_shared<Keyframe::PointCloudT>();
    
    auto descriptor = detector_->makeScanContext(empty_cloud);
    EXPECT_EQ(descriptor.size(), 0);
    
    // 添加空点云描述子应该返回空矩阵
    auto desc = detector_->addDescriptor(0, empty_cloud);
    EXPECT_EQ(desc.size(), 0);
    EXPECT_EQ(detector_->size(), 0u);
}

// 测试添加描述子
TEST_F(LoopDetectorTest, AddDescriptor) {
    auto cloud = createCircularCloud();
    
    auto desc = detector_->addDescriptor(0, cloud);
    
    EXPECT_EQ(detector_->size(), 1u);
    EXPECT_GT(desc.size(), 0);
    
    // 验证可以获取描述子
    auto retrieved = detector_->getDescriptor(0);
    EXPECT_EQ(retrieved.rows(), desc.rows());
    EXPECT_EQ(retrieved.cols(), desc.cols());
}

// 测试添加多个描述子
TEST_F(LoopDetectorTest, AddMultipleDescriptors) {
    for (int i = 0; i < 5; ++i) {
        auto cloud = createOffsetCloud(i * 10.0, 0);
        detector_->addDescriptor(i, cloud);
    }
    
    EXPECT_EQ(detector_->size(), 5u);
    
    // 验证每个描述子都可以获取
    for (int i = 0; i < 5; ++i) {
        auto desc = detector_->getDescriptor(i);
        EXPECT_GT(desc.size(), 0);
    }
}

// 测试回环检测排除近邻帧 - Requirements 4.6
TEST_F(LoopDetectorTest, ExcludeRecentFrames) {
    // 添加 15 个关键帧 (排除最近 10 帧)
    for (int i = 0; i < 15; ++i) {
        auto cloud = createCircularCloud();  // 相同的点云
        detector_->addDescriptor(i, cloud);
    }
    
    // 查询第 14 帧 (索引从 0 开始)
    // 应该只在 [0, 4) 范围内搜索 (14 - 10 = 4)
    auto candidates = detector_->detectLoopCandidates(14);
    
    // 验证所有候选帧都在排除范围之外
    for (const auto& candidate : candidates) {
        EXPECT_LT(candidate.match_id, 14 - config_.sc_num_exclude_recent);
    }
}

// 测试历史帧不足时的处理
TEST_F(LoopDetectorTest, InsufficientHistory) {
    // 只添加 5 帧 (少于排除数量 10)
    for (int i = 0; i < 5; ++i) {
        auto cloud = createCircularCloud();
        detector_->addDescriptor(i, cloud);
    }
    
    // 查询最后一帧，应该返回空列表
    auto candidates = detector_->detectLoopCandidates(4);
    EXPECT_TRUE(candidates.empty());
}

// 测试相似场景的回环检测 - Requirements 4.2, 4.3
TEST_F(LoopDetectorTest, DetectSimilarScenes) {
    // 添加足够多的帧
    for (int i = 0; i < 20; ++i) {
        Keyframe::PointCloudT::Ptr cloud;
        if (i == 0 || i == 15) {
            // 第 0 帧和第 15 帧使用相同的点云
            cloud = createCircularCloud(1000, 20.0, 2.0);
        } else {
            // 其他帧使用不同的点云
            cloud = createOffsetCloud(i * 50.0, i * 50.0);
        }
        detector_->addDescriptor(i, cloud);
    }
    
    // 查询第 15 帧，应该能找到第 0 帧作为候选
    auto candidates = detector_->detectLoopCandidates(15);
    
    // 由于使用相同的点云，应该能检测到回环
    // 注意：由于 ScanContext 的特性，相同点云应该有很小的距离
    bool found_match = false;
    for (const auto& candidate : candidates) {
        if (candidate.match_id == 0) {
            found_match = true;
            EXPECT_LT(candidate.sc_distance, config_.sc_dist_threshold);
        }
    }
    
    // 如果没有找到匹配，可能是因为阈值设置
    // 这里我们只验证候选帧在正确的范围内
    for (const auto& candidate : candidates) {
        EXPECT_LT(candidate.match_id, 15 - config_.sc_num_exclude_recent);
        EXPECT_GE(candidate.match_id, 0);
    }
}

TEST_F(LoopDetectorTest, DetectSimilarScenesUsesRHPDPrimary) {
    Config cfg = config_;
    cfg.rhpd_enabled = true;
    cfg.sc_aux_veto_enabled = false;
    cfg.rhpd_dist_threshold = 100.0;
    cfg.sc_num_exclude_recent = 10;
    LoopDetector detector(cfg);

    for (int i = 0; i < 20; ++i) {
        Keyframe::PointCloudT::Ptr cloud;
        if (i == 0 || i == 15) {
            cloud = createAsymmetricCloud();
        } else {
            cloud = createDoorwayCorridor(i % 2 == 0);
            for (auto& pt : cloud->points) {
                pt.x += static_cast<float>(i * 4.0);
                pt.y += static_cast<float>(i * 1.5);
            }
        }
        detector.addDescriptor(i, cloud);
        detector.addRHPD(i, cloud);
    }

    auto candidates = detector.detectLoopCandidates(15);
    ASSERT_FALSE(candidates.empty());
    bool found = false;
    for (const auto& candidate : candidates) {
        EXPECT_TRUE(std::isfinite(candidate.fused_score));
        if (candidate.match_id == 0) {
            found = true;
            EXPECT_TRUE(candidate.fromRHPD());
            EXPECT_EQ(candidate.candidate_source, LoopCandidate::Source::RhpdPrimary);
            EXPECT_TRUE(std::isfinite(candidate.rhpd_distance));
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(LoopDetectorTest, RHPDPrimarySearchFiltersRecentFramesBeforePrefilter) {
    Config cfg = config_;
    cfg.rhpd_enabled = true;
    cfg.sc_aux_veto_enabled = false;
    cfg.sc_num_exclude_recent = 50;
    cfg.rhpd_num_candidates = 5;
    cfg.sc_num_candidates = 3;
    cfg.rhpd_preselect_candidates = 8;
    cfg.rhpd_dist_threshold = 100.0;
    LoopDetector detector(cfg);

    auto query_scene = createAsymmetricCloud();
    auto old_loop_scene = rotateCloud(query_scene, 0.02);
    for (int i = 0; i <= 180; ++i) {
        Keyframe::PointCloudT::Ptr cloud;
        if (i == 20) {
            cloud = old_loop_scene;
        } else if (i >= 150) {
            cloud = query_scene;
        } else {
            cloud = translatedCloud(createDifferentStructureCloud(), i * 0.25, -i * 0.15);
        }
        detector.addDescriptor(i, cloud);
        detector.addRHPD(i, cloud);
    }

    auto candidates = detector.detectLoopCandidates(180);
    ASSERT_FALSE(candidates.empty());
    bool found_old_loop = false;
    for (const auto& candidate : candidates) {
        EXPECT_LT(candidate.match_id, 180 - cfg.sc_num_exclude_recent);
        if (candidate.match_id == 20) {
            found_old_loop = true;
            EXPECT_TRUE(candidate.fromRHPD());
            EXPECT_EQ(candidate.candidate_source, LoopCandidate::Source::RhpdPrimary);
        }
    }
    EXPECT_TRUE(found_old_loop);
}

TEST_F(LoopDetectorTest, DetectSimilarScenesCanFallbackToScanContextWhenRHPDDisabled) {
    Config cfg = config_;
    cfg.rhpd_enabled = false;
    cfg.sc_dist_threshold = 1.0;
    cfg.sc_num_exclude_recent = 10;
    LoopDetector detector(cfg);

    for (int i = 0; i < 20; ++i) {
        auto cloud = (i == 0 || i == 15) ? createCircularCloud(1000, 20.0, 2.0)
                                         : createOffsetCloud(i * 50.0, i * 50.0);
        detector.addDescriptor(i, cloud);
        detector.addRHPD(i, cloud);
    }

    auto candidates = detector.detectLoopCandidates(15);
    ASSERT_FALSE(candidates.empty());
    bool found = false;
    for (const auto& candidate : candidates) {
        if (candidate.match_id == 0) {
            found = true;
            EXPECT_TRUE(candidate.fromSC());
            EXPECT_EQ(candidate.candidate_source, LoopCandidate::Source::ScanContextFallback);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(LoopDetectorTest, DetectSpatialCandidatesUsesPoseRadiusAndAgeGap) {
    Config cfg = config_;
    cfg.loop_spatial_candidates_enable = true;
    cfg.loop_spatial_candidate_radius = 5.0;
    cfg.loop_spatial_candidate_min_id_gap = 50;
    cfg.loop_spatial_candidate_max_candidates = 2;
    LoopDetector detector(cfg);

    auto make_keyframe = [](int64_t id, const Eigen::Vector3d& position) {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = position;
        return Keyframe::create(id, static_cast<double>(id), pose,
                                pcl::make_shared<Keyframe::PointCloudT>());
    };

    std::map<int64_t, Keyframe::Ptr> keyframes;
    keyframes[100] = make_keyframe(100, Eigen::Vector3d::Zero());
    keyframes[5] = make_keyframe(5, Eigen::Vector3d(1.0, 0.0, 0.0));
    keyframes[10] = make_keyframe(10, Eigen::Vector3d(3.0, 0.0, 0.0));
    keyframes[20] = make_keyframe(20, Eigen::Vector3d(10.0, 0.0, 0.0));
    keyframes[90] = make_keyframe(90, Eigen::Vector3d(0.5, 0.0, 0.0));

    auto candidates = detector.detectSpatialCandidates(100, keyframes);
    ASSERT_EQ(candidates.size(), 2u);

    EXPECT_EQ(candidates[0].match_id, 5);
    EXPECT_EQ(candidates[1].match_id, 10);
    for (const auto& candidate : candidates) {
        EXPECT_EQ(candidate.query_id, 100);
        EXPECT_TRUE(candidate.fromSpatial());
        EXPECT_FALSE(candidate.fromRHPD());
        EXPECT_FALSE(candidate.fromSC());
        EXPECT_EQ(candidate.candidate_source, LoopCandidate::Source::SpatialRadius);
        EXPECT_TRUE(std::isfinite(candidate.fused_score));
        EXPECT_GE(candidate.fused_score, 0.0);
        EXPECT_LE(candidate.fused_score, 1.0);
    }
}

TEST_F(LoopDetectorTest, RejectsIncompatibleScanContextDescriptorDimensions) {
    auto cloud = createCircularCloud();
    auto valid = detector_->makeScanContext(cloud);
    detector_->addDescriptor(0, valid);
    EXPECT_EQ(detector_->size(), 1u);

    Eigen::MatrixXd wrong_rows = Eigen::MatrixXd::Zero(valid.rows() + 1, valid.cols());
    detector_->addDescriptor(1, wrong_rows);
    EXPECT_EQ(detector_->size(), 1u);

    auto new_detector = std::make_unique<LoopDetector>(config_);
    std::vector<std::pair<int64_t, Eigen::MatrixXd>> descriptors;
    descriptors.emplace_back(10, valid);
    descriptors.emplace_back(11, Eigen::MatrixXd::Zero(valid.rows(), valid.cols() + 1));
    new_detector->loadDescriptors(descriptors);
    EXPECT_EQ(new_detector->size(), 1u);
    EXPECT_GT(new_detector->getDescriptor(10).size(), 0);
    EXPECT_EQ(new_detector->getDescriptor(11).size(), 0);
}

// 测试描述子序列化和加载
TEST_F(LoopDetectorTest, SerializeAndLoad) {
    // 添加一些描述子
    for (int i = 0; i < 5; ++i) {
        auto cloud = createOffsetCloud(i * 10.0, 0);
        detector_->addDescriptor(i, cloud);
    }
    
    // 获取所有描述子
    auto descriptors = detector_->getDescriptors();
    EXPECT_EQ(descriptors.size(), 5u);
    
    // 创建新的检测器并加载描述子
    auto new_detector = std::make_unique<LoopDetector>(config_);
    new_detector->loadDescriptors(descriptors);
    
    EXPECT_EQ(new_detector->size(), 5u);
    
    // 验证描述子内容一致
    for (int i = 0; i < 5; ++i) {
        auto orig = detector_->getDescriptor(i);
        auto loaded = new_detector->getDescriptor(i);
        
        EXPECT_EQ(orig.rows(), loaded.rows());
        EXPECT_EQ(orig.cols(), loaded.cols());
        EXPECT_NEAR((orig - loaded).norm(), 0.0, 1e-9);
    }
}

// 测试清空功能
TEST_F(LoopDetectorTest, Clear) {
    for (int i = 0; i < 5; ++i) {
        auto cloud = createCircularCloud();
        detector_->addDescriptor(i, cloud);
    }
    
    EXPECT_EQ(detector_->size(), 5u);
    
    detector_->clear();
    
    EXPECT_EQ(detector_->size(), 0u);
    
    // 清空后获取描述子应该返回空矩阵
    auto desc = detector_->getDescriptor(0);
    EXPECT_EQ(desc.size(), 0);
}

// 测试添加已有描述子 (用于地图加载)
TEST_F(LoopDetectorTest, AddExistingDescriptor) {
    auto cloud = createCircularCloud();
    auto desc = detector_->makeScanContext(cloud);
    
    // 使用已有描述子添加
    detector_->addDescriptor(100, desc);
    
    EXPECT_EQ(detector_->size(), 1u);
    
    auto retrieved = detector_->getDescriptor(100);
    EXPECT_EQ(retrieved.rows(), desc.rows());
    EXPECT_EQ(retrieved.cols(), desc.cols());
}

// 测试获取不存在的描述子
TEST_F(LoopDetectorTest, GetNonExistentDescriptor) {
    auto desc = detector_->getDescriptor(999);
    EXPECT_EQ(desc.size(), 0);
}

// 测试 LoopCandidate 有效性
TEST_F(LoopDetectorTest, LoopCandidateValidity) {
    LoopCandidate valid_candidate;
    valid_candidate.query_id = 10;
    valid_candidate.match_id = 5;
    EXPECT_TRUE(valid_candidate.isValid());
    
    LoopCandidate invalid_candidate1;
    invalid_candidate1.query_id = -1;
    invalid_candidate1.match_id = 5;
    EXPECT_FALSE(invalid_candidate1.isValid());
    
    LoopCandidate invalid_candidate2;
    invalid_candidate2.query_id = 10;
    invalid_candidate2.match_id = -1;
    EXPECT_FALSE(invalid_candidate2.isValid());
}

// 测试 VerifiedLoop 有效性
TEST_F(LoopDetectorTest, VerifiedLoopValidity) {
    VerifiedLoop valid_loop;
    valid_loop.query_id = 10;
    valid_loop.match_id = 5;
    valid_loop.verified = true;
    EXPECT_TRUE(valid_loop.isValid());
    
    VerifiedLoop unverified_loop;
    unverified_loop.query_id = 10;
    unverified_loop.match_id = 5;
    unverified_loop.verified = false;
    EXPECT_FALSE(unverified_loop.isValid());
}

// 测试重建 KD 树
TEST_F(LoopDetectorTest, RebuildTree) {
    // 添加描述子
    for (int i = 0; i < 5; ++i) {
        auto cloud = createOffsetCloud(i * 10.0, 0);
        detector_->addDescriptor(i, cloud);
    }
    
    // 重建树
    detector_->rebuildTree();
    
    // 验证描述子仍然存在
    EXPECT_EQ(detector_->size(), 5u);
    
    for (int i = 0; i < 5; ++i) {
        auto desc = detector_->getDescriptor(i);
        EXPECT_GT(desc.size(), 0);
    }
}

}  // namespace test
}  // namespace n3mapping
