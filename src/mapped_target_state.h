#pragma once
#include "mapped_target.h"
#include "background_worker.h"
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <future>

namespace n3mapping::registration_detail {
struct AtlasMapping::Impl {
  boost::interprocess::file_mapping file;
  boost::interprocess::mapped_region region;
  std::vector<MappedTargetLevel> levels;
  std::vector<std::array<double, 6>> active_key;
  std::vector<std::array<double, 6>> pending_key;
  std::future<void> pending;
  // Join before releasing the mapping used by the submitted task.
  BackgroundWorker worker;
};
} // namespace n3mapping::registration_detail
