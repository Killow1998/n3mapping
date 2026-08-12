#include "n3mapping/submap_graph_trial_runtime.h"

#include "n3mapping/product_build_identity.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace n3mapping {
namespace {

class ScopedTestDirectory {
  public:
    explicit ScopedTestDirectory(const std::string& name)
      : path_(std::filesystem::temp_directory_path() /
              ("n3mapping_submap_graph_trial_runtime_" + name)) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~ScopedTestDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

SubmapGraphSnapshot makeReadySingleNodeSnapshot() {
    SubmapGraphSnapshot snapshot;
    snapshot.valid = true;
    SubmapGraphNodeProjection node;
    node.submap_id = 4;
    node.session_id = 2;
    node.anchor_keyframe_id = 40;
    node.keyframe_count = 1;
    node.closed = true;
    node.content_revision = 3;
    node.T_map_submap = Eigen::Isometry3d::Identity();
    node.T_map_submap.translation() = Eigen::Vector3d(1.0, -2.0, 0.5);
    snapshot.nodes.push_back(node);
    snapshot.keyframe_ownership.emplace(40, 4);
    return snapshot;
}

SubmapGraphSnapshot makeDisconnectedSnapshot() {
    SubmapGraphSnapshot snapshot = makeReadySingleNodeSnapshot();
    SubmapGraphNodeProjection second = snapshot.nodes.front();
    second.submap_id = 9;
    second.anchor_keyframe_id = 90;
    second.T_map_submap.translation().x() = 8.0;
    snapshot.nodes.push_back(second);
    snapshot.keyframe_ownership.emplace(90, 9);
    return snapshot;
}

std::vector<std::string> readLines(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);
    return lines;
}

TEST(SubmapGraphTrialRuntimeTest, CheckpointPolicyIsLowFrequencyAndExact) {
    EXPECT_TRUE(isSubmapGraphTrialCheckpointContext("save_map"));
    EXPECT_TRUE(isSubmapGraphTrialCheckpointContext("save_extended_map"));
    EXPECT_TRUE(isSubmapGraphTrialCheckpointContext("loop_commit"));
    EXPECT_TRUE(isSubmapGraphTrialCheckpointContext("cross_session_loop"));
    EXPECT_FALSE(isSubmapGraphTrialCheckpointContext("graph_update"));
    EXPECT_FALSE(isSubmapGraphTrialCheckpointContext("submap_append"));
    EXPECT_FALSE(isSubmapGraphTrialCheckpointContext("keyframe_commit"));
    EXPECT_FALSE(isSubmapGraphTrialCheckpointContext("save_map_extra"));
    EXPECT_FALSE(isSubmapGraphTrialCheckpointContext(""));
}

TEST(SubmapGraphTrialRuntimeTest,
     DisabledOrNonCheckpointContextDoesNotSolveOrWrite) {
    ScopedTestDirectory directory("disabled_or_non_checkpoint");
    Config config;
    config.map_save_path = directory.path().string();
    const auto snapshot = makeReadySingleNodeSnapshot();

    const auto disabled = runSubmapGraphTrialCheckpoint(
        snapshot, config, "core", "save_map");
    EXPECT_FALSE(disabled.checkpoint);
    EXPECT_FALSE(disabled.persisted);
    EXPECT_FALSE(disabled.diagnostics.attempted);

    config.submap_shadow_enable = true;
    const auto high_frequency = runSubmapGraphTrialCheckpoint(
        snapshot, config, "core", "graph_update");
    EXPECT_FALSE(high_frequency.checkpoint);
    EXPECT_FALSE(high_frequency.persisted);
    EXPECT_FALSE(high_frequency.diagnostics.attempted);
    EXPECT_FALSE(std::filesystem::exists(
        directory.path() / "submap_graph_trial.jsonl"));
}

TEST(SubmapGraphTrialRuntimeTest,
     ReadyCheckpointPersistsDeterministicNoWritebackEvidence) {
    ScopedTestDirectory directory("ready_checkpoint");
    Config config;
    config.submap_shadow_enable = true;
    config.map_save_path = directory.path().string();
    const auto snapshot = makeReadySingleNodeSnapshot();
    const Eigen::Matrix4d before =
        snapshot.nodes.front().T_map_submap.matrix();

    const auto result = runSubmapGraphTrialCheckpoint(
        snapshot, config, "core\"runtime", "save_map");

    EXPECT_TRUE(result.checkpoint);
    EXPECT_TRUE(result.persisted);
    EXPECT_EQ(result.output_path,
              (directory.path() / "submap_graph_trial.jsonl").string());
    ASSERT_TRUE(result.diagnostics.valid)
        << result.diagnostics.failure_reason;
    EXPECT_TRUE(result.diagnostics.solved);
    ASSERT_EQ(result.diagnostics.nodes.size(), 1u);
    EXPECT_TRUE(snapshot.nodes.front().T_map_submap.matrix().isApprox(
        before, 0.0));

    const auto lines = readLines(result.output_path);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_NE(lines.front().find(
                  "\"schema\":\"n3mapping_submap_graph_trial_v1\""),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"runtime_source\":\"core\\\"runtime\""),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"context\":\"save_map\""),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"no_writeback\":true"),
              std::string::npos);
    const auto identity = productBuildIdentity();
    EXPECT_NE(lines.front().find(
                  "\"product_commit\":\"" + identity.commit + "\""),
              std::string::npos);
    EXPECT_NE(lines.front().find(
                  std::string("\"product_verified\":") +
                  (identity.verified ? "true" : "false")),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"valid\":true"),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"solved\":true"),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"submap_id\":4"),
              std::string::npos);
    EXPECT_NE(lines.front().find("\"gauge_anchor\":true"),
              std::string::npos);
}

TEST(SubmapGraphTrialRuntimeTest,
     UnreadyCheckpointPersistsFailClosedReasonAndAppends) {
    ScopedTestDirectory directory("unready_checkpoint");
    Config config;
    config.submap_shadow_enable = true;
    config.map_save_path = directory.path().string();
    const auto snapshot = makeDisconnectedSnapshot();

    const auto first = runSubmapGraphTrialCheckpoint(
        snapshot, config, "mapping_resuming", "save_extended_map");
    const auto second = runSubmapGraphTrialCheckpoint(
        snapshot, config, "mapping_resuming", "cross_session_loop");

    EXPECT_TRUE(first.checkpoint);
    EXPECT_TRUE(first.persisted);
    EXPECT_FALSE(first.diagnostics.valid);
    EXPECT_FALSE(first.diagnostics.attempted);
    EXPECT_EQ(first.diagnostics.failure_reason, "topology_not_ready");
    EXPECT_TRUE(second.persisted);
    EXPECT_EQ(second.diagnostics.failure_reason,
              first.diagnostics.failure_reason);

    const auto lines = readLines(first.output_path);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_NE(lines[0].find("\"valid\":false"), std::string::npos);
    EXPECT_NE(lines[0].find("\"attempted\":false"), std::string::npos);
    EXPECT_NE(lines[0].find(
                  "\"failure_reason\":\"topology_not_ready\""),
              std::string::npos);
    EXPECT_NE(lines[1].find("\"context\":\"cross_session_loop\""),
              std::string::npos);
}

TEST(SubmapGraphTrialRuntimeTest,
     PersistenceFailureDoesNotInvalidateOrMutateTrialResult) {
    ScopedTestDirectory directory("persistence_failure");
    const auto blocking_file = directory.path() / "not_a_directory";
    {
        std::ofstream output(blocking_file);
        ASSERT_TRUE(output.is_open());
        output << "block child creation";
    }
    Config config;
    config.submap_shadow_enable = true;
    config.map_save_path = (blocking_file / "child").string();
    const auto snapshot = makeReadySingleNodeSnapshot();

    const auto result = runSubmapGraphTrialCheckpoint(
        snapshot, config, "core", "save_map");

    EXPECT_TRUE(result.checkpoint);
    EXPECT_FALSE(result.persisted);
    EXPECT_TRUE(result.diagnostics.valid)
        << result.diagnostics.failure_reason;
    EXPECT_TRUE(result.diagnostics.solved);
    EXPECT_TRUE(snapshot.nodes.front().T_map_submap.matrix().isApprox(
        makeReadySingleNodeSnapshot().nodes.front().T_map_submap.matrix(),
        0.0));
}

TEST(SubmapGraphTrialRuntimeTest, PathResolutionUsesExistingDebugConvention) {
    Config config;
    config.map_save_path.clear();
    EXPECT_EQ(resolveSubmapGraphTrialPath(config),
              "submap_graph_trial.jsonl");
    config.map_save_path = "/tmp/n3mapping map";
    EXPECT_EQ(resolveSubmapGraphTrialPath(config),
              "/tmp/n3mapping map/submap_graph_trial.jsonl");
}

}  // namespace
}  // namespace n3mapping
