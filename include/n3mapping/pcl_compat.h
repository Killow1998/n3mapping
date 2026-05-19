#pragma once

#include <utility>

#if __has_include(<pcl/memory.h>)
#include <pcl/memory.h>
#elif __has_include(<pcl/make_shared.h>)
#include <pcl/make_shared.h>
#else
#include <pcl/point_cloud.h>

namespace pcl {

template <typename T, typename... Args>
typename T::Ptr make_shared(Args&&... args)
{
    return typename T::Ptr(new T(std::forward<Args>(args)...));
}

}  // namespace pcl
#endif
