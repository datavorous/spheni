#pragma once

#include <cstddef>
#include <span>

namespace spheni::math {

inline long long vector_count(std::span<const float> vectors,
                              int dim) noexcept {
  return static_cast<long long>(vectors.size() / static_cast<std::size_t>(dim));
}

} // namespace spheni::math
