#include "n3mapping/n3map_nav_resource_reader.h"

#include <unordered_set>

#include "n3mapping/n3map_proto_utils.h"
#include "n3map.pb.h"

namespace n3mapping {
namespace {

bool setError(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

}  // namespace

bool readN3NavResource(const std::string& pbstream_path,
                       N3NavResource* out,
                       std::string* error) {
    return readN3NavResource(pbstream_path, N3NavReaderOptions{}, out, error);
}

bool readN3NavResource(const std::string& pbstream_path,
                       const N3NavReaderOptions& options,
                       N3NavResource* out,
                       std::string* error) {
    if (!out) return setError(error, "null output resource");

    N3Map map_proto;
    if (!readN3MapProtoFromFile(pbstream_path, &map_proto, error)) return false;

    std::vector<ParsedKeyframeProto> parsed_keyframes;
    std::unordered_set<int64_t> keyframe_ids;
    PbstreamKeyframeParseOptions keyframe_parse_options;
    keyframe_parse_options.policy = PbstreamLoadPolicy::STRICT;
    keyframe_parse_options.parse_descriptors = false;
    if (!parseKeyframesFromProto(map_proto, keyframe_parse_options,
                                 &parsed_keyframes, &keyframe_ids, error)) {
        return false;
    }

    N3NavResource resource;
    const PbstreamMetadata metadata = extractPbstreamMetadata(map_proto.metadata());
    resource.version = metadata.version;
    resource.map_frame = metadata.map_frame;
    resource.body_frame = metadata.body_frame;
    resource.dense_trajectory_source = metadata.dense_trajectory_source;
    resource.dense_trajectory_degraded = metadata.dense_trajectory_degraded;
    resource.nav_cloud_filter_applied = metadata.nav_cloud_filter_applied;
    resource.nav_cloud_filter_policy = metadata.nav_cloud_filter_policy;
    resource.descriptors_recomputed_from_filtered_cloud =
        metadata.descriptors_recomputed_from_filtered_cloud;
    resource.nav_filter_raw_points = metadata.nav_filter_raw_points;
    resource.nav_filter_kept_points = metadata.nav_filter_kept_points;
    resource.nav_filter_removed_points = metadata.nav_filter_removed_points;

    resource.keyframes.reserve(parsed_keyframes.size());
    for (const auto& parsed : parsed_keyframes) {
        N3NavKeyframe keyframe;
        keyframe.id = parsed.id;
        keyframe.timestamp = parsed.timestamp;
        keyframe.pose_odom = parsed.pose_odom;
        keyframe.pose_optimized = parsed.pose_optimized;
        keyframe.cloud = parsed.cloud;

        resource.optimized_poses[keyframe.id] = keyframe.pose_optimized;
        resource.keyframes.push_back(std::move(keyframe));
    }

    PbstreamLoadOptions load_options;
    load_options.policy = PbstreamLoadPolicy::STRICT;
    load_options.allow_keyframe_fallback_dense = options.allow_keyframe_fallback;
    core::DenseTrajectoryMetadata dense_metadata;
    if (!parseDenseTrajectoryFromProto(map_proto, parsed_keyframes, load_options,
                                       &resource.dense_optimized_trajectory,
                                       &dense_metadata, error)) {
        return false;
    }
    if (resource.dense_optimized_trajectory.empty() &&
        map_proto.dense_optimized_trajectory_size() == 0 &&
        !options.allow_keyframe_fallback) {
        return setError(error, "pbstream_missing_dense_trajectory");
    }
    resource.dense_trajectory_source = dense_metadata.source;
    resource.dense_trajectory_degraded = dense_metadata.degraded;
    resource.has_native_dense_trajectory =
        !resource.dense_optimized_trajectory.empty() &&
        resource.dense_trajectory_source == "native" &&
        !resource.dense_trajectory_degraded;
    resource.dense_trajectory_from_keyframe_fallback =
        resource.dense_trajectory_source == "keyframe_fallback" ||
        resource.dense_trajectory_source == "mixed_keyframe_fallback_and_high_rate";

    *out = std::move(resource);
    return true;
}

}  // namespace n3mapping
