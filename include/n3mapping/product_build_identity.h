// Compile-time identity for Product V1 executables.
#pragma once

#include <cstddef>
#include <sstream>
#include <string>

#ifndef N3MAPPING_PRODUCT_COMMIT
#define N3MAPPING_PRODUCT_COMMIT "UNVERIFIED"
#endif

#ifndef N3MAPPING_PRODUCT_PROFILE_SHA256
#define N3MAPPING_PRODUCT_PROFILE_SHA256 "UNVERIFIED"
#endif

#ifndef N3MAPPING_PRODUCT_BUILD_VERIFIED
#define N3MAPPING_PRODUCT_BUILD_VERIFIED 0
#endif

#ifndef N3MAPPING_PRODUCT_BUILD_TYPE
#define N3MAPPING_PRODUCT_BUILD_TYPE "UNVERIFIED"
#endif

#ifndef N3MAPPING_PRODUCT_RESEARCH_TOOLS
#define N3MAPPING_PRODUCT_RESEARCH_TOOLS "UNVERIFIED"
#endif

namespace n3mapping {

struct ProductBuildIdentity {
  std::string commit;
  std::string product_profile_sha256;
  std::string build_type;
  std::string research_tools;
  bool verified = false;
};

inline bool isLowerHex(const std::string &value, std::size_t size) {
  if (value.size() != size) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

inline ProductBuildIdentity productBuildIdentity() {
  ProductBuildIdentity identity{
      N3MAPPING_PRODUCT_COMMIT,
      N3MAPPING_PRODUCT_PROFILE_SHA256,
      N3MAPPING_PRODUCT_BUILD_TYPE,
      N3MAPPING_PRODUCT_RESEARCH_TOOLS,
      N3MAPPING_PRODUCT_BUILD_VERIFIED != 0,
  };
  identity.verified =
      identity.verified && isLowerHex(identity.commit, 40) &&
      isLowerHex(identity.product_profile_sha256, 64) &&
      identity.build_type == "Release" &&
      identity.research_tools == "OFF";
  return identity;
}

inline std::string productBuildIdentityJson() {
  const auto identity = productBuildIdentity();
  std::ostringstream stream;
  stream << "{\"schema\":\"n3mapping_product_build_identity_v2\","
         << "\"commit\":\"" << identity.commit << "\","
         << "\"product_profile_sha256\":\""
         << identity.product_profile_sha256 << "\","
         << "\"build_type\":\"" << identity.build_type << "\","
         << "\"research_tools\":\"" << identity.research_tools << "\","
         << "\"verified\":" << (identity.verified ? "true" : "false")
         << "}";
  return stream.str();
}

inline bool productBuildIdentityRequested(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "--build-identity-json") {
      return true;
    }
  }
  return false;
}

}  // namespace n3mapping
