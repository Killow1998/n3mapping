#include "n3mapping/core/n3mapping_session.h"

namespace n3mapping {
namespace core {

N3MappingSession::N3MappingSession(const Config& config)
  : config_(config)
  , keyframe_manager_(config_)
  , point_cloud_matcher_(config_)
  , loop_detector_(config_)
  , loop_closure_manager_(config_)
  , graph_optimizer_(config_)
  , map_serializer_(config_)
  , world_localizing_(config_, keyframe_manager_, loop_detector_, point_cloud_matcher_)
  , mapping_resuming_(config_,
                      keyframe_manager_,
                      loop_detector_,
                      point_cloud_matcher_,
                      graph_optimizer_,
                      map_serializer_,
                      world_localizing_)
{
}

}  // namespace core
}  // namespace n3mapping
