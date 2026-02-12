#include "math/kmeans.h"
#include "math/kernels.h"
#include <cstdint>
#include <algorithm>
#include <limits>
#include <random>

namespace spheni::math::clustering {
KMeans::KMeans(int k, int dim, int max_iters, std::uint32_t seed)
    : k_(k), dim_(dim), max_iters_(max_iters), seed_(seed) {}

std::vector<float> KMeans::fit(std::span<const float> vectors) {
  long long n = vectors.size() / dim_;

  if (n < k_) {
    std::vector<float> centroids(vectors.begin(), vectors.end());
    centroids.resize(k_ * dim_, 0.0f);
    return centroids;
  }

  std::vector<float> centroids(k_ * dim_);
  std::vector<bool> chosen(n, false);
  std::mt19937 rng(seed_);

  std::uniform_int_distribution<long long> dist(0, n - 1);
  long long first_idx = dist(rng);
  std::copy_n(vectors.data() + first_idx * dim_, dim_, centroids.data());

  chosen[first_idx] = true;

  for (int c = 1; c < k_; c++) {

    std::vector<float> min_distances(n, std::numeric_limits<float>::max());

    for (long long i = 0; i < n; i++) {
      if (chosen[i])
        continue;

      const float *vec = vectors.data() + i * dim_;
      for (int j = 0; j < c; j++) {
        float dist =
            math::kernels::l2_squared(vec, centroids.data() + j * dim_, dim_);
        min_distances[i] = std::min(min_distances[i], dist);
      }
    }

    float sum = 0.0f;
    for (long long i = 0; i < n; i++) {
      if (!chosen[i]) {
        sum += min_distances[i];
      }
    }

    std::uniform_real_distribution<float> real_dist(0.0f, sum);
    float threshold = real_dist(rng);
    float cumsum = 0.0f;

    for (long long i = 0; i < n; i++) {
      if (chosen[i])
        continue;

      cumsum += min_distances[i];
      if (cumsum >= threshold) {
        std::copy_n(vectors.data() + i * dim_, dim_,
                    centroids.data() + c * dim_);
        chosen[i] = true;
        break;
      }
    }
  }

  std::vector<int> assignments(n);
  for (int iter = 0; iter < max_iters_; iter++) {

    assignments = predict(vectors, centroids);

    std::vector<float> new_centroids(k_ * dim_, 0.0f);
    std::vector<int> counts(k_, 0);

    for (long long i = 0; i < n; i++) {
      int cluster = assignments[i];
      counts[cluster]++;
      const float *vec = vectors.data() + i * dim_;
      float *centroid = new_centroids.data() + cluster * dim_;
      for (int d = 0; d < dim_; d++) {
        centroid[d] += vec[d];
      }
    }

    for (int c = 0; c < k_; c++) {
      if (counts[c] > 0) {
        float *centroid = new_centroids.data() + c * dim_;
        for (int d = 0; d < dim_; d++) {
          centroid[d] /= counts[c];
        }
      } else {
        // reinitialize to a random point
        long long idx = dist(rng);
        std::copy_n(vectors.data() + idx * dim_, dim_,
                    new_centroids.data() + c * dim_);
      }
    }

    centroids = std::move(new_centroids);
  }
  return centroids;
}

std::vector<int> KMeans::predict(std::span<const float> vectors,
                                 std::vector<float> centroids) const {
  long long n = vectors.size() / dim_;
  std::vector<int> assignments(n);

  for (long long i = 0; i < n; i++) {
    const float *vec = vectors.data() + i * dim_;
    float min_dist = std::numeric_limits<float>::max();
    int best_cluster = 0;

    for (int c = 0; c < k_; c++) {
      const float *centroid = centroids.data() + c * dim_;
      float dist = math::kernels::l2_squared(vec, centroid, dim_);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = c;
      }
    }
    assignments[i] = best_cluster;
  }
  return assignments;
}
} // namespace spheni::math::clustering
