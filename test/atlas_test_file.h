#pragma once

#include "../src/atlas_container.h"
#include "localization_atlas.pb.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace n3mapping::test {
struct AtlasTestFile {
  atlas_codec::ContainerHeader header;
  std::vector<atlas_codec::SectionEntry> sections;
  std::string bytes;

  explicit AtlasTestFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
    std::memcpy(&header, bytes.data(), sizeof(header));
    sections.resize(header.sections);
    std::memcpy(sections.data(), bytes.data() + sizeof(header), sections.size() * sizeof(sections[0]));
  }

  atlas_codec::SectionEntry& section(atlas_codec::SectionTag tag, unsigned flags = 0) {
    return *std::find_if(sections.begin(), sections.end(), [&](const auto& section) {
      return section.tag == tag && section.flags == flags;
    });
  }

  bool readMetadata(LocalizationAtlasData& data) {
    const auto& entry = section(atlas_codec::SectionTag::Metadata);
    return data.ParseFromArray(bytes.data() + entry.offset, entry.size);
  }

  void replaceSection(atlas_codec::SectionTag tag, const std::string& content) {
    auto& entry = section(tag);
    const auto old_end = atlas_codec::align8(entry.offset + entry.size);
    const auto new_end = atlas_codec::align8(entry.offset + content.size());
    const bool last = &entry == &sections.back();
    auto replacement = content;
    if (!last) replacement.resize(new_end - entry.offset, '\0');
    bytes.replace(entry.offset, last ? entry.size : old_end - entry.offset, replacement);
    for (auto& following : sections)
      if (following.offset > entry.offset)
        following.offset = following.offset - old_end + new_end;
    entry.size = content.size();
  }

  bool write(const std::filesystem::path& path) {
    std::memcpy(bytes.data() + sizeof(header), sections.data(), sections.size() * sizeof(sections[0]));
    return bool(std::ofstream(path, std::ios::binary).write(bytes.data(), bytes.size()));
  }

  bool writeMetadata(const std::filesystem::path& path, const LocalizationAtlasData& data) {
    replaceSection(atlas_codec::SectionTag::Metadata, data.SerializeAsString());
    return write(path);
  }
};
}
