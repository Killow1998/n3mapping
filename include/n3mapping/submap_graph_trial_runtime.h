// Low-frequency runtime checkpoints for the no-writeback submap graph trial.
#pragma once

#include <string>

#include "n3mapping/config.h"
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_trial.h"

namespace n3mapping {

struct SubmapGraphTrialRuntimeResult {
    bool checkpoint = false;
    bool persisted = false;
    std::string output_path;
    SubmapGraphTrialDiagnostics diagnostics;
};

// Trial solves are intentionally restricted to low-frequency lifecycle
// boundaries. A normal graph/keyframe refresh is never a checkpoint.
bool isSubmapGraphTrialCheckpointContext(const std::string& context);

std::string resolveSubmapGraphTrialPath(const Config& config);

// Evaluates an immutable snapshot and appends one JSONL record when shadow
// submaps are enabled and context is an approved checkpoint. Persistence is
// observational: failure is returned to the caller but never changes graph or
// map lifecycle decisions.
SubmapGraphTrialRuntimeResult runSubmapGraphTrialCheckpoint(
    const SubmapGraphSnapshot& snapshot,
    const Config& config,
    const std::string& runtime_source,
    const std::string& context);

}  // namespace n3mapping
