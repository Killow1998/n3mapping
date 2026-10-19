#include "n3mapping/relocalization_performance.h"
#include <cmath>
#include <json/json.h>

namespace n3mapping {
namespace {
Json::Value phaseJson(const DescriptorPhaseCost &phase) {
  Json::Value value;
  value["wall_ms"] = phase.wall_ms;
  value["thread_cpu_ms"] = phase.thread_cpu_ms;
  value["calls"] = Json::UInt64(phase.calls);
  return value;
}
} // namespace

Json::Value relocalizationPerformanceJson(const RelocalizationPerformance &cost) {
  Json::Value value;
  const auto timing = [&](const char *name, double ms) {
    if (std::isfinite(ms) && ms >= 0.0) value[name] = ms;
  };
  timing("total_ms", cost.total_ms);
  timing("process_cpu_ms", cost.process_cpu_ms);
  timing("query_build_ms", cost.query_build_ms);
  timing("free_space_ms", cost.free_space_ms);
  timing("directional_check_ms", cost.directional_check_ms);
  value["directional_checked"] = cost.directional_checked;
  value["directional_min_support"] = std::isfinite(cost.directional_min_support)
      ? Json::Value(cost.directional_min_support) : Json::Value();
  timing("diagnostics_ms", cost.diagnostics_ms);
  timing("source_prepare_ms", cost.source_prepare_ms);
  timing("visibility_prepare_ms", cost.visibility_prepare_ms);
  timing("target_build_ms", cost.target_build_ms);
  timing("target_prepare_ms", cost.target_prepare_ms);
  timing("align_ms", cost.align_ms);
  timing("fixed_pose_ms", cost.fixed_pose_ms);
  timing("visibility_ms", cost.visibility_ms);
  value["source_preparations"] = Json::UInt64(cost.source_preparations);
  value["visibility_preparations"] = Json::UInt64(cost.visibility_preparations);
  value["target_builds"] = Json::UInt64(cost.target_builds);
  value["target_preparations"] = Json::UInt64(cost.target_preparations);
  value["alignments"] = Json::UInt64(cost.alignments);
  value["rejected_alignments"] = Json::UInt64(cost.rejected_alignments);
  value["fixed_pose_evaluations"] = Json::UInt64(cost.fixed_pose_evaluations);
  value["visibility_evaluations"] = Json::UInt64(cost.visibility_evaluations);
  value["candidates"] = Json::UInt64(cost.candidates);
  value["hypotheses"] = Json::UInt64(cost.hypotheses);
  value["cache_hits"] = Json::UInt64(cost.cache_hits);
  value["cache_misses"] = Json::UInt64(cost.cache_misses);
  value["cache_bytes"] = Json::UInt64(cost.cache_bytes);
  value["cache_entries"] = Json::UInt64(cost.cache_entries);
  value["effective_cache_max_bytes"] = Json::UInt64(cost.effective_cache_max_bytes);
  value["effective_cache_max_entries"] = Json::UInt64(cost.effective_cache_max_entries);
  value["cpu_scope"] = "whole_process_during_search_including_workers_and_background";
  value["align_scope"] = "sum_of_call_wall_times_may_overlap_across_candidates";
  auto &descriptor = value["descriptor"];
  descriptor["cpu_scope"] = "calling_thread_serial_descriptor_work";
  // Total contains the other descriptor phases; do not add it to them.
  descriptor["total"] = phaseJson(cost.descriptor.total);
  descriptor["generation"] = phaseJson(cost.descriptor.generation);
  descriptor["index_rebuild"] = phaseJson(cost.descriptor.index_rebuild);
  descriptor["coarse"] = phaseJson(cost.descriptor.coarse);
  descriptor["place"] = phaseJson(cost.descriptor.place);
  descriptor["heading"] = phaseJson(cost.descriptor.heading);
  descriptor["input_points"] = Json::UInt64(cost.descriptor.input_points);
  descriptor["place_values"] = Json::UInt64(cost.descriptor.place_values);
  descriptor["heading_values"] = Json::UInt64(cost.descriptor.heading_values);
  descriptor["indexed_keyframes"] = Json::UInt64(cost.descriptor.indexed_keyframes);
  descriptor["coarse_comparisons"] = Json::UInt64(cost.descriptor.coarse_comparisons);
  descriptor["place_comparisons"] = Json::UInt64(cost.descriptor.place_comparisons);
  descriptor["heading_comparisons"] = Json::UInt64(cost.descriptor.heading_comparisons);
  descriptor["returned_candidates"] = Json::UInt64(cost.descriptor.returned_candidates);
  return value;
}
} // namespace n3mapping
