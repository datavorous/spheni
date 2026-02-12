#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace spheni::math::clustering {
class KMeans {
public:
  KMeans(int k, int dim, std::uint32_t seed = 42, int max_iters = 25);

  std::vector<float> fit(std::span<const float> vectors);
  std::vector<int> predict(std::span<const float> vectors,
                           std::vector<float> centroids) const;

private:
  int k_;
  int dim_;
  int max_iters_;
  std::uint32_t seed_;
};
} // namespace spheni::math::clustering
