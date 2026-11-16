#pragma once

#include <string>

#include "n3mapping/config.h"
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_trial.h"

namespace n3mapping {

struct SubmapGraphTrialRuntimeResult {
    bool checkpoint = false;
    bool log_enqueued = false;
    std::string output_path;
    SubmapGraphTrialDiagnostics diagnostics;
};

bool isSubmapGraphTrialCheckpointContext(const std::string& context);

std::string resolveSubmapGraphTrialPath(const Config& config);

SubmapGraphTrialRuntimeResult runSubmapGraphTrialCheckpoint(
    const SubmapGraphSnapshot& snapshot,
    const Config& config,
    const std::string& runtime_source,
    const std::string& context);

}
