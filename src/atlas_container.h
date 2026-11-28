#pragma once

#include <array>
#include <cstdint>

namespace n3mapping::atlas_codec {
inline constexpr std::array<char, 8> kMagic = {'N', '3', 'A', 'T', 'L', 'A', 'S', '5'};
inline constexpr std::size_t kMaxSections = 6;
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

enum class SectionTag : std::uint32_t {
  Metadata = 1, GlobalPoints, PlaneLevel, RefineLevel, FreeSpace
};

struct ContainerHeader {
  std::array<char, 8> magic = kMagic;
  std::uint64_t sections = 0;
};

struct SectionEntry {
  SectionTag tag;
  std::uint32_t flags;
  std::uint64_t offset;
  std::uint64_t size;
};

struct LevelHeader {
  double resolution;
  std::uint64_t points;
  std::uint64_t explicit_covariance;
  std::uint64_t search_bytes;
};

static_assert(sizeof(ContainerHeader) == 16);
static_assert(sizeof(SectionEntry) == 24);
static_assert(sizeof(LevelHeader) == 32);

inline constexpr std::uint64_t align8(std::uint64_t size) {
  return (size + 7) & ~std::uint64_t{7};
}
}
