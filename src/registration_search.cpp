#include "registration_search.h"

namespace n3mapping::registration_detail {

template std::size_t TypedCorrespondenceSearch<NativeTreeView>::nearest(
    const Eigen::Vector4d&, std::size_t*, double*, std::size_t) const;
template std::size_t TypedCorrespondenceSearch<MappedTreeView>::nearest(
    const Eigen::Vector4d&, std::size_t*, double*, std::size_t) const;
template void TypedCorrespondenceSearch<NativeTreeView>::visit<true>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
template void TypedCorrespondenceSearch<NativeTreeView>::visit<false>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
template void TypedCorrespondenceSearch<MappedTreeView>::visit<true>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
template void TypedCorrespondenceSearch<MappedTreeView>::visit<false>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
}
