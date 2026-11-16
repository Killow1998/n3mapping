#include "n3mapping/n3map_proto_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace n3mapping {
namespace {

bool setError(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

bool addChecked(uint64_t value, uint64_t &total, uint64_t limit,
                const std::string &message, std::string *error) {
  if (value > limit || total > limit - value)
    return setError(error, message);
  total += value;
  return true;
}

bool validateResourceCounts(const N3Map &map_proto,
                            const PbstreamResourceLimits &limits,
                            std::string *error) {
  if (map_proto.keyframes_size() > static_cast<int>(limits.max_keyframes)) {
    return setError(error, "pbstream has too many keyframes");
  }
  if (map_proto.edges_size() > static_cast<int>(limits.max_edges)) {
    return setError(error, "pbstream has too many edges");
  }
  if (map_proto.dense_optimized_trajectory_size() >
      static_cast<int>(limits.max_dense_trajectory_poses)) {
    return setError(error, "pbstream has too many dense trajectory poses");
  }

  uint64_t total_points = 0;
  uint64_t total_descriptor_values = 0;
  for (int i = 0; i < map_proto.keyframes_size(); ++i) {
    const auto &keyframe = map_proto.keyframes(i);
    if (keyframe.has_cloud()) {
      const uint64_t encoded_points =
          (static_cast<uint64_t>(keyframe.cloud().points_size()) + 3u) / 4u;
      const uint64_t counted_points = std::max(
          static_cast<uint64_t>(keyframe.cloud().num_points()), encoded_points);
      if (!addChecked(counted_points, total_points, limits.max_total_points,
                      "pbstream has too many total points", error)) {
        return false;
      }
    }
    if (keyframe.has_sc_descriptor()) {
      if (!addChecked(
              static_cast<uint64_t>(keyframe.sc_descriptor().values_size()),
              total_descriptor_values, limits.max_total_descriptor_values,
              "pbstream has too many descriptor values", error)) {
        return false;
      }
    }
    if (keyframe.has_rhpd_descriptor()) {
      if (!addChecked(
              static_cast<uint64_t>(keyframe.rhpd_descriptor().values_size()),
              total_descriptor_values, limits.max_total_descriptor_values,
              "pbstream has too many descriptor values", error)) {
        return false;
      }
    }
  }
  return true;
}

}

PbstreamResourceLimits defaultPbstreamResourceLimits() {
  return PbstreamResourceLimits{};
}

bool readN3MapProtoFromFile(const std::string &filepath, N3Map *map_proto,
                            std::string *error) {
  return readN3MapProtoFromFile(filepath, defaultPbstreamResourceLimits(),
                                map_proto, error);
}

bool readN3MapProtoFromFile(const std::string &filepath,
                            const PbstreamResourceLimits &limits,
                            N3Map *map_proto, std::string *error) {
  if (!map_proto)
    return setError(error, "null map proto output");
  map_proto->Clear();
  if (limits.max_file_bytes == 0 || limits.max_keyframes == 0 ||
      limits.max_edges == 0 || limits.max_dense_trajectory_poses == 0 ||
      limits.max_total_points == 0 || limits.max_total_descriptor_values == 0 ||
      limits.protobuf_recursion_limit <= 0) {
    return setError(error, "invalid pbstream resource limits");
  }

  std::error_code ec;
  const auto status = std::filesystem::status(filepath, ec);
  if (ec || !std::filesystem::exists(status))
    return setError(error, "pbstream does not exist");
  if (!std::filesystem::is_regular_file(status))
    return setError(error, "pbstream is not a regular file");

  const auto file_size = std::filesystem::file_size(filepath, ec);
  if (ec)
    return setError(error, "failed to stat pbstream");
  if (file_size > limits.max_file_bytes)
    return setError(error, "pbstream exceeds file size limit");
  if (limits.max_file_bytes >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return setError(error, "pbstream byte limit exceeds protobuf API limit");
  }

  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open())
    return setError(error, "failed to open pbstream");

  google::protobuf::io::IstreamInputStream zero_copy_input(&ifs);
  google::protobuf::io::CodedInputStream coded_input(&zero_copy_input);
  coded_input.SetTotalBytesLimit(static_cast<int>(limits.max_file_bytes));
  coded_input.SetRecursionLimit(limits.protobuf_recursion_limit);
  if (!map_proto->ParseFromCodedStream(&coded_input) ||
      !coded_input.ConsumedEntireMessage()) {
    map_proto->Clear();
    return setError(error, "failed to parse pbstream");
  }
  if (!validateResourceCounts(*map_proto, limits, error)) {
    map_proto->Clear();
    return false;
  }
  return true;
}

}
