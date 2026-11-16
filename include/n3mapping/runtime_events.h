#pragma once
#include <cstdint>
#include <string>
namespace n3mapping {
struct CoreLoopClosureResult;
namespace core { struct BackendOutput; struct OdometryEmission; }

class RuntimeEvents {
public:
  void backend(const core::BackendOutput& output, std::int64_t stamp, double ms);
  void live(const core::OdometryEmission& emission);
  void loop(const CoreLoopClosureResult& result);
private:
  std::string backend_state_, live_state_;
};
}
