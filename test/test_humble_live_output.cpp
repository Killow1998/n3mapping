// Exercise the production node's subscriptions, executors and publishers.
// The test subclass pauses conversion; it does not replace output decisions.
#include "../humble/src/n3mapping_node.h"

#include <condition_variable>
#include <filesystem>
#include <future>
#include <sstream>
#include <pcl_conversions/pcl_conversions.h>
#include <gtest/gtest.h>
#include <json/json.h>
#include <tf2_msgs/msg/tf_message.hpp>

namespace n3mapping {
class HumbleOutputTestAccess {
 public:
  static void configure(N3MappingNode& node, CoreRunMode mode, bool exact = true) {
    node.authority_probe_timer_->cancel();
    node.run_mode_ = mode;
    node.config_.sync_queue_size = 8;
    node.last_tf_stamp_.reset();
    node.product_profile_v1_ = exact;
    node.realtime_output_ = std::make_unique<core::RealtimeOutput>(mode, "map", 8);
    node.live_odom_sub_.reset();
    node.live_cloud_sub_.unsubscribe();
    node.live_paired_odom_sub_.unsubscribe();
    node.live_approx_sync_.reset();
    node.live_exact_sync_.reset();
    node.live_cloud_worker_.reset();
    node.live_odom_worker_.reset();
    node.initializeLiveSubscriptions();
  }
  static void clock(N3MappingNode& node, int64_t now) { node.authority_probe_now_nsec_ = now; }
  static void backend(N3MappingNode& node, const nav_msgs::msg::Odometry& source,
                      core::BackendOutput output, int64_t observation) {
    auto header = source.header;
    header.stamp = rclcpp::Time(observation);
    auto publication = node.publishRelocalizationOutput(header, source, output);
    if (publication.publish_global_pose) node.publishPath(header, &output.T_world_lidar);
  }
  static void odom(N3MappingNode& node, nav_msgs::msg::Odometry source) {
    node.liveOdomCallback(std::make_shared<nav_msgs::msg::Odometry>(std::move(source)));
  }
  static void cloud(N3MappingNode& node, sensor_msgs::msg::PointCloud2 cloud,
                    nav_msgs::msg::Odometry source) {
    node.liveCloudCallback(std::make_shared<sensor_msgs::msg::PointCloud2>(std::move(cloud)),
                          std::make_shared<nav_msgs::msg::Odometry>(std::move(source)));
  }
  static std::unique_lock<std::mutex> backendLock(N3MappingNode& node) {
    return std::unique_lock<std::mutex>(node.data_mutex_);
  }
  static void coupleOdometry(N3MappingNode& node, rclcpp::Executor& backend) {
    auto& worker = *node.live_odom_worker_;
    worker.executor_->remove_callback_group(worker.group_);
    backend.add_callback_group(worker.group_, node.get_node_base_interface());
    node.live_cloud_worker_->start();
  }
  static void enableSave(N3MappingNode& node, const std::string& directory, bool real_core) {
    node.authority_probe_scenario_.reset();
    node.run_mode_ = CoreRunMode::MAPPING;
    node.config_.map_save_path = directory;
    node.config_.mapping_static_start_guard_enable = false;
    node.config_.rhpd_submap_kf_radius = 1;
    node.config_.rhpd_submap_voxel_size = 0.0;
    if (real_core) node.n3mapping_core_ = std::make_unique<N3MappingCore>(
        node.config_, N3MappingCore::WorldCloudOutput::Disabled);
  }
  static void process(N3MappingNode& node, const sensor_msgs::msg::PointCloud2& cloud,
                      const nav_msgs::msg::Odometry& odom) {
    node.processSynchronizedFrame(std::make_shared<sensor_msgs::msg::PointCloud2>(cloud),
                                 std::make_shared<nav_msgs::msg::Odometry>(odom));
  }
};

namespace {
using namespace std::chrono_literals;
constexpr int64_t epoch = 1700000000000000001LL;

class Gate {
 public:
  ~Gate() { open(); }
  void pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }
  bool entered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 2s, [&] { return entered_; });
  }
  void open() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_ = false, released_ = false;
};

class ControlledNode : public N3MappingNode {
 public:
  ControlledNode() : N3MappingNode("localization_authority_sequence") {}
  std::shared_ptr<Gate> conversion_gate;
  std::atomic<int> conversions{0};
  std::atomic<bool> saved{false};
 protected:
  sensor_msgs::msg::PointCloud2 transformLiveCloud(
      const sensor_msgs::msg::PointCloud2& cloud, const Eigen::Isometry3d& pose) override {
    ++conversions;
    if (conversion_gate) conversion_gate->pause();
    return N3MappingNode::transformLiveCloud(cloud, pose);
  }
  void saveMapOnShutdown() override { saved = true; }
};

nav_msgs::msg::Odometry raw(int64_t stamp = epoch, double x = 2.0) {
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = rclcpp::Time(stamp);
  odom.header.frame_id = "continuous";
  odom.child_frame_id = "body";
  odom.pose.pose.position.x = x;
  odom.pose.pose.orientation.w = 1.0;
  odom.twist.twist.linear.x = 1.5;
  odom.twist.covariance[0] = 0.25;
  return odom;
}

core::BackendOutput trusted(double x = 12.0) {
  core::BackendOutput output;
  output.success = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  output.T_world_lidar.translation().x() = x;
  return output;
}

sensor_msgs::msg::PointCloud2 scan(int64_t stamp = epoch) {
  pcl::PointCloud<pcl::PointXYZI> points;
  points.push_back(pcl::PointXYZI{1.0F, 2.0F, 3.0F, 4.0F});
  sensor_msgs::msg::PointCloud2 cloud;
  pcl::toROSMsg(points, cloud);
  cloud.header.stamp = rclcpp::Time(stamp);
  cloud.header.frame_id = "body";
  return cloud;
}

Json::Value json(const std::string& text) {
  Json::Value result;
  std::istringstream stream(text);
  stream >> result;
  return result;
}

class HumbleLiveOutput : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
  }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
  void SetUp() override {
    node = std::make_shared<ControlledNode>();
    HumbleOutputTestAccess::configure(*node, CoreRunMode::LOCALIZATION);
    HumbleOutputTestAccess::clock(*node, epoch + 1000);
    observer = std::make_shared<rclcpp::Node>("humble_live_output_observer");
    collector.add_node(observer);
    input_odom = observer->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 100);
    input_cloud = observer->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 10);
    odom_sub = observer->create_subscription<nav_msgs::msg::Odometry>("/n3mapping/odometry", 100,
        [&](nav_msgs::msg::Odometry::ConstSharedPtr msg) { odometry.push_back(*msg); });
    status_sub = observer->create_subscription<std_msgs::msg::String>("/localization/status", 100,
        [&](std_msgs::msg::String::ConstSharedPtr msg) { statuses.push_back(json(msg->data)); });
    body_sub = observer->create_subscription<sensor_msgs::msg::PointCloud2>("/n3mapping/cloud_body", 100,
        [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { body.push_back(*msg); });
    path_sub = observer->create_subscription<nav_msgs::msg::Path>("/n3mapping/path", 100,
        [&](nav_msgs::msg::Path::ConstSharedPtr msg) { paths.push_back(*msg); });
    tf_sub = observer->create_subscription<tf2_msgs::msg::TFMessage>("/tf", 100,
        [&](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
          for (const auto& transform : msg->transforms) transforms.push_back(transform);
        });
  }
  void TearDown() override {
    if (backend_gate) backend_gate->open();
    if (node && node->conversion_gate) node->conversion_gate->open();
    if (node) node->shutdown();
    backend_stopping = true;
    backend_executor.cancel();
    if (backend_thread.joinable()) backend_thread.join();
    if (observer) collector.remove_node(observer);
    node.reset();
    observer.reset();
  }
  template <typename Predicate> bool until(Predicate predicate, std::chrono::milliseconds timeout = 1000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
      collector.spin_some();
      if (predicate()) return true;
      std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
  }
  void drain() { until([] { return false; }, 50ms); }
  void seed(int64_t stamp = epoch - 10, double x = 12.0) {
    HumbleOutputTestAccess::backend(*node, raw(stamp), trusted(x), stamp + 1);
    drain();
  }
  void worldSubscriber() {
    world_sub = observer->create_subscription<sensor_msgs::msg::PointCloud2>("/n3mapping/cloud_world", 100,
        [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { world.push_back(*msg); });
    ASSERT_TRUE(until([&] { return node->conversions == 0 && observer->count_publishers("/n3mapping/cloud_world") == 1; }));
  }
  void blockBackend(bool coupled = false) {
    backend_gate = std::make_shared<Gate>();
    blocker = node->create_subscription<std_msgs::msg::UInt32>("/issue6/backend", 1,
        [&](std_msgs::msg::UInt32::ConstSharedPtr) {
          auto lock = HumbleOutputTestAccess::backendLock(*node);
          backend_gate->pause();
        });
    backend_executor.add_node(node);
    if (coupled) HumbleOutputTestAccess::coupleOdometry(*node, backend_executor);
    else node->startLiveOutput();
    backend_thread = std::thread([&] {
      while (!backend_stopping) backend_executor.spin_once(50ms);
    });
    block_input = observer->create_publisher<std_msgs::msg::UInt32>("/issue6/backend", 1);
    ASSERT_TRUE(until([&] { return block_input->get_subscription_count() == 1; }));
    block_input->publish(std_msgs::msg::UInt32{});
    ASSERT_TRUE(backend_gate->entered());
  }
  std::shared_ptr<ControlledNode> node;
  std::shared_ptr<rclcpp::Node> observer;
  rclcpp::executors::SingleThreadedExecutor collector, backend_executor;
  std::thread backend_thread;
  std::atomic<bool> backend_stopping{false};
  std::shared_ptr<Gate> backend_gate;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr input_odom;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr input_cloud;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr block_input;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr body_sub, world_sub;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub;
  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr blocker;
  std::vector<nav_msgs::msg::Odometry> odometry;
  std::vector<Json::Value> statuses;
  std::vector<sensor_msgs::msg::PointCloud2> body, world;
  std::vector<nav_msgs::msg::Path> paths;
  std::vector<geometry_msgs::msg::TransformStamped> transforms;
};

TEST_F(HumbleLiveOutput, SameProductionSequenceMatchesSharedCoreInAllModes) {
  worldSubscriber();
  for (auto mode : {CoreRunMode::MAPPING, CoreRunMode::LOCALIZATION, CoreRunMode::MAP_EXTENSION}) {
    SCOPED_TRACE(coreRunModeName(mode));
    HumbleOutputTestAccess::configure(*node, mode);
    core::RealtimeOutput reference(mode, "map", 8);
    auto output = trusted();
    auto anchor = raw(epoch - 10);
    reference.acceptBackend(toCoreOdometrySource(anchor), (epoch - 9) * 1e-9, output);
    seed();
    const auto path_count = paths.size();
    auto source = raw();
    const auto expected = reference.observeOdometry(toCoreOdometrySource(source), epoch + 1000, true);
    ASSERT_TRUE(expected.pose);
    const auto count = odometry.size();
    HumbleOutputTestAccess::odom(*node, source);
    ASSERT_TRUE(until([&] { return odometry.size() == count + 1 && !statuses.empty(); }));
    EXPECT_DOUBLE_EQ(odometry.back().pose.pose.position.x, expected.pose->pose.translation().x());
    EXPECT_EQ(toCoreTimeStamp(odometry.back().header.stamp).nsec, epoch);
    EXPECT_EQ(toCoreTimeStamp(transforms.back().header.stamp).nsec, epoch);
    EXPECT_DOUBLE_EQ(odometry.back().twist.twist.linear.x, 1.5);
    EXPECT_DOUBLE_EQ(odometry.back().twist.covariance[0], 0.25);
    auto actual_status = statuses.back(), expected_status = json(expected.status_json);
    actual_status["projection"].removeMember("session_id");
    expected_status["projection"].removeMember("session_id");
    EXPECT_EQ(actual_status, expected_status);
    EXPECT_EQ(paths.size(), path_count);
    const auto clouds = world.size();
    HumbleOutputTestAccess::cloud(*node, scan(epoch + 1), source);
    ASSERT_TRUE(until([&] { return world.size() == clouds + 1; }));
    EXPECT_EQ(toCoreTimeStamp(world.back().header.stamp).nsec, epoch + 1);
    pcl::PointCloud<pcl::PointXYZI> points;
    pcl::fromROSMsg(world.back(), points);
    ASSERT_EQ(points.size(), 1U);
    EXPECT_FLOAT_EQ(points.front().x, 13.0F);
    EXPECT_EQ(statuses.back()["projection"]["odom_stamp_nsec"].asInt64(), epoch);
    EXPECT_EQ(statuses.back()["projection"]["cloud_stamp_nsec"].asInt64(), epoch + 1);
    auto compare_live = [&](nav_msgs::msg::Odometry input, bool publishable) {
      const auto wanted = reference.observeOdometry(toCoreOdometrySource(input), epoch + 1000, publishable);
      const auto before = odometry.size();
      HumbleOutputTestAccess::odom(*node, input);
      drain();
      EXPECT_EQ(odometry.size(), before + (wanted.pose ? 1U : 0U));
      if (wanted.pose && odometry.size() > before) {
        EXPECT_DOUBLE_EQ(odometry.back().pose.pose.position.x, wanted.pose->pose.translation().x());
      }
      auto observed = statuses.back(), expected_json = json(wanted.status_json);
      observed["projection"].removeMember("session_id");
      expected_json["projection"].removeMember("session_id");
      EXPECT_EQ(observed, expected_json);
    };
    auto update = [&](int64_t stamp, core::BackendOutput value) {
      auto input = raw(stamp);
      auto copy = value;
      reference.acceptBackend(toCoreOdometrySource(input), (stamp + 1) * 1e-9, copy);
      HumbleOutputTestAccess::backend(*node, input, value, stamp + 1);
      drain();
    };
    update(epoch + 2, trusted(50.0));
    compare_live(raw(epoch + 4, 3.0), true);
    update(epoch + 5, core::BackendOutput{});
    compare_live(raw(epoch + 7, 4.0), true);
    auto invalid = raw(epoch + 8);
    invalid.twist.twist.linear.x = std::numeric_limits<double>::quiet_NaN();
    compare_live(invalid, false);
    update(epoch + 9, trusted(20.0));
    compare_live(raw(epoch + 10), true);
    if (mode == CoreRunMode::LOCALIZATION) {
      // Published snapshots must retain their own history when the node reuses
      // its accumulated path for subsequent publications.
      ASSERT_EQ(paths[path_count - 1].poses.size(), 1U);
      ASSERT_EQ(paths.back().poses.size(), 3U);
      EXPECT_EQ(paths.back().header.frame_id, "map");
      EXPECT_DOUBLE_EQ(paths.back().poses[0].pose.position.x, 12.0);
      EXPECT_DOUBLE_EQ(paths.back().poses[1].pose.position.x, 50.0);
      EXPECT_DOUBLE_EQ(paths.back().poses[2].pose.position.x, 20.0);
      EXPECT_EQ(toCoreTimeStamp(paths.back().poses[0].header.stamp).nsec, epoch - 9);
      EXPECT_EQ(toCoreTimeStamp(paths.back().header.stamp).nsec, epoch + 10);
    }

  }
}

TEST_F(HumbleLiveOutput, BackendDataLockDoesNotDelayRawOdometry) {
  seed();
  blockBackend();
  input_odom->publish(raw());
  ASSERT_TRUE(until([&] { return odometry.size() == 1; }));
  EXPECT_EQ(toCoreTimeStamp(odometry.back().header.stamp).nsec, epoch);
  EXPECT_EQ(paths.size(), 1U);
}

TEST_F(HumbleLiveOutput, CoupledExecutorControlDetectsDelayAndKeepsOnlyNewestOdom) {
  seed();
  blockBackend(true);
  for (int i = 0; i < 20; ++i) input_odom->publish(raw(epoch + i));
  EXPECT_FALSE(until([&] { return !odometry.empty(); }, 100ms));
  backend_gate->open();
  ASSERT_TRUE(until([&] { return !odometry.empty(); }));
  drain();
  ASSERT_EQ(odometry.size(), 1U);
  EXPECT_EQ(toCoreTimeStamp(odometry.back().header.stamp).nsec, epoch + 19);
}

TEST_F(HumbleLiveOutput, ExactCloudWorkerDoesNotDelayRawOdometry) {
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  input_odom->publish(raw(epoch + 1));
  ASSERT_TRUE(until([&] { return odometry.size() == 2; }));
  EXPECT_TRUE(world.empty());
  EXPECT_EQ(toCoreTimeStamp(odometry.back().header.stamp).nsec, epoch + 1);
}

TEST_F(HumbleLiveOutput, ApproximateCloudSubscribersRunOnTheirOwnWorker) {
  HumbleOutputTestAccess::configure(*node, CoreRunMode::LOCALIZATION, false);
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  input_odom->publish(raw(epoch + 1));
  ASSERT_TRUE(until([&] { return odometry.size() == 2; }));
}

TEST_F(HumbleLiveOutput, InvalidationDuringConversionCancelsWorldPublication) {
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  auto invalid = raw(epoch + 1);
  invalid.twist.covariance[0] = std::numeric_limits<double>::quiet_NaN();
  input_odom->publish(invalid);
  ASSERT_TRUE(until([&] { return !statuses.empty() && statuses.back()["reason"] == "invalid_twist_or_body_frame"; }));
  EXPECT_FALSE(statuses.back()["projection"]["available"].asBool());
  node->conversion_gate->open();
  drain();
  EXPECT_TRUE(world.empty());
  HumbleOutputTestAccess::backend(*node, raw(epoch + 2), trusted(20.0), epoch + 2);
  input_odom->publish(raw(epoch + 3));
  input_cloud->publish(scan(epoch + 3));
  ASSERT_TRUE(until([&] { return world.size() == 1; }));
  EXPECT_EQ(statuses.back()["projection"]["generation"].asUInt64(), 1U);
}

TEST_F(HumbleLiveOutput, LateCorrectionCannotRewriteFrozenCloudGeometry) {
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  HumbleOutputTestAccess::backend(*node, raw(epoch + 1), trusted(50.0), epoch + 2);
  node->conversion_gate->open();
  ASSERT_TRUE(until([&] { return world.size() == 1; }));
  pcl::PointCloud<pcl::PointXYZI> points;
  pcl::fromROSMsg(world.back(), points);
  ASSERT_EQ(points.size(), 1U);
  EXPECT_FLOAT_EQ(points.front().x, 13.0F);
}

TEST_F(HumbleLiveOutput, ChangedSourceCorrectionCancelsReservedCloud) {
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  auto source = raw(epoch + 1);
  source.header.frame_id = "new_continuous";
  HumbleOutputTestAccess::backend(*node, source, trusted(20.0), epoch + 2);
  node->conversion_gate->open();
  drain();
  EXPECT_TRUE(world.empty());
}

TEST_F(HumbleLiveOutput, BodyOnlyStillPublishesAssociationWithoutConversion) {
  seed();
  HumbleOutputTestAccess::odom(*node, raw());
  HumbleOutputTestAccess::cloud(*node, scan(epoch + 1), raw());
  ASSERT_TRUE(until([&] { return !body.empty() && !statuses.empty() && statuses.back()["projection"]["kind"] == "cloud"; }));
  EXPECT_EQ(node->conversions, 0);
  EXPECT_TRUE(world.empty());
  EXPECT_EQ(statuses.back()["projection"]["cloud_stamp_nsec"].asInt64(), epoch + 1);
  auto wrong = scan(epoch + 2);
  wrong.header.frame_id = "wrong_body";
  HumbleOutputTestAccess::cloud(*node, wrong, raw(epoch + 2));
  drain();
  EXPECT_EQ(body.size(), 1U);
}

TEST_F(HumbleLiveOutput, InvalidAndNonIncreasingSourcesNeverCreateFreshPoseOrTf) {
  for (const auto& kind : {"zero", "future", "regressed", "pose", "rotation", "twist", "covariance", "source", "body", "duplicate"}) {
    SCOPED_TRACE(kind);
    HumbleOutputTestAccess::configure(*node, CoreRunMode::LOCALIZATION);
    seed();
    HumbleOutputTestAccess::odom(*node, raw());
    drain();
    const auto poses = odometry.size(), tfs = transforms.size();
    auto invalid = raw(epoch + 1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (std::string(kind) == "zero") invalid.header.stamp = rclcpp::Time(int64_t{0});
    if (std::string(kind) == "future") invalid.header.stamp = rclcpp::Time(epoch + 1001);
    if (std::string(kind) == "regressed") invalid.header.stamp = rclcpp::Time(epoch - 1);
    if (std::string(kind) == "pose") invalid.pose.pose.position.x = nan;
    if (std::string(kind) == "rotation") invalid.pose.pose.orientation.w = 0.0;
    if (std::string(kind) == "twist") invalid.twist.twist.linear.x = nan;
    if (std::string(kind) == "covariance") invalid.twist.covariance[0] = nan;
    if (std::string(kind) == "source") invalid.header.frame_id = "wrong";
    if (std::string(kind) == "body") invalid.child_frame_id = "wrong";
    if (std::string(kind) == "duplicate") invalid = raw();
    HumbleOutputTestAccess::odom(*node, invalid);
    drain();
    EXPECT_EQ(odometry.size(), poses);
    EXPECT_EQ(transforms.size(), tfs);
  }
}

TEST_F(HumbleLiveOutput, ShutdownJoinsConversionBeforeSavingAndSuppressesLatePublish) {
  seed();
  worldSubscriber();
  node->conversion_gate = std::make_shared<Gate>();
  node->startLiveOutput();
  input_odom->publish(raw());
  input_cloud->publish(scan());
  ASSERT_TRUE(node->conversion_gate->entered());
  HumbleOutputTestAccess::enableSave(*node, "/tmp/unused_issue6_save", false);
  auto shutting_down = std::async(std::launch::async, [&] { node->shutdown(); });
  EXPECT_EQ(shutting_down.wait_for(50ms), std::future_status::timeout);
  EXPECT_FALSE(node->saved);
  node->conversion_gate->open();
  ASSERT_EQ(shutting_down.wait_for(2s), std::future_status::ready);
  shutting_down.get();
  EXPECT_TRUE(node->saved);
  drain();
  EXPECT_TRUE(world.empty());
}

TEST_F(HumbleLiveOutput, ShutdownWritesRealMapAfterWorkersStop) {
  auto real = std::make_shared<N3MappingNode>("map_extension_legacy_compatibility");
  HumbleOutputTestAccess::configure(*real, CoreRunMode::MAPPING);
  const auto directory = std::filesystem::temp_directory_path() /
      ("n3_issue6_map_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  HumbleOutputTestAccess::enableSave(*real, directory.string(), true);
  pcl::PointCloud<pcl::PointXYZI> points;
  for (int i = 0; i < 120; ++i) points.push_back(pcl::PointXYZI{
      static_cast<float>(i % 12) * 0.1F, static_cast<float>(i / 12) * 0.1F,
      static_cast<float>(i % 5) * 0.05F, 1.0F});
  sensor_msgs::msg::PointCloud2 cloud;
  pcl::toROSMsg(points, cloud);
  cloud.header = scan().header;
  HumbleOutputTestAccess::process(*real, cloud, raw());
  real->startLiveOutput();
  real->shutdown();
  EXPECT_TRUE(std::filesystem::exists(directory / "n3map.pbstream"));
  EXPECT_TRUE(std::filesystem::exists(directory / "global_map.pcd"));
  real.reset();
  // Only artifacts created by this fixture are removed.
  std::filesystem::remove_all(directory);
}

}  // namespace
}  // namespace n3mapping
