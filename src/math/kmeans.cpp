#include "math/kmeans.h"
#include "math/math.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>

namespace spheni::math::clustering {
KMeans::KMeans(int k, int dim, int max_iters)
    : k_(k), dim_(dim), max_iters_(max_iters) {}

std::vector<float> KMeans::fit(std::span<const float> vectors) {
        const int n = vectors.size() / dim_;
        assert(n >= k_);

        std::vector<float> centroids(k_ * dim_);
        std::vector<bool> chosen(n, false);
        std::mt19937 rng(42);

        std::uniform_int_distribution<int> dist(0, n - 1);
        const int first_idx = dist(rng);
        std::copy_n(vectors.data() + first_idx * dim_, dim_, centroids.data());

        chosen[first_idx] = true;

        for (int c = 1; c < k_; ++c) {
                std::vector<float> min_distances(
                    n, std::numeric_limits<float>::max());

                for (int i = 0; i < n; ++i) {
                        if (chosen[i])
                                continue;

                        const float *vec = vectors.data() + i * dim_;
                        for (int j = 0; j < c; ++j) {
                                const float d = math::kernels::l2_squared(
                                    vec, centroids.data() + j * dim_, dim_);
                                min_distances[i] =
                                    std::min(min_distances[i], d);
                        }
                }

                float sum = 0.0f;
                for (int i = 0; i < n; ++i) {
                        if (!chosen[i]) {
                                sum += min_distances[i];
                        }
                }

                std::uniform_real_distribution<float> real_dist(0.0f, sum);
                const float threshold = real_dist(rng);
                float cumsum = 0.0f;
                int pick = 0;

                for (int i = 0; i < n; ++i) {
                        if (chosen[i])
                                continue;

                        cumsum += min_distances[i];
                        if (cumsum >= threshold) {
                                pick = i;
                                break;
                        }
                }
                std::copy_n(vectors.data() + pick * dim_, dim_,
                            centroids.data() + c * dim_);
                chosen[pick] = true;
        }

        std::vector<int> assignments(n);
        for (int iter = 0; iter < max_iters_; ++iter) {
                assignments = predict(vectors, centroids);

                std::vector<float> new_centroids(k_ * dim_, 0.0f);
                std::vector<int> counts(k_, 0);

                for (int i = 0; i < n; ++i) {
                        const int cluster = assignments[i];
                        ++counts[cluster];
                        const float *vec = vectors.data() + i * dim_;
                        float *centroid = new_centroids.data() + cluster * dim_;
                        for (int d = 0; d < dim_; ++d) {
                                centroid[d] += vec[d];
                        }
                }

                for (int c = 0; c < k_; ++c) {
                        float *centroid = new_centroids.data() + c * dim_;
                        if (counts[c] > 0) {
                                for (int d = 0; d < dim_; ++d) {
                                        centroid[d] /= counts[c];
                                }
                        } else {
                                const int idx = dist(rng);
                                std::copy_n(vectors.data() + idx * dim_, dim_,
                                            centroid);
                        }
                }

                centroids = std::move(new_centroids);
        }
        return centroids;
}

std::vector<int> KMeans::predict(std::span<const float> vectors,
                                 std::span<const float> centroids) const {
        const int n = vectors.size() / dim_;
        std::vector<int> assignments(n);

        for (int i = 0; i < n; ++i) {
                const float *vec = vectors.data() + i * dim_;
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;

                for (int c = 0; c < k_; ++c) {
                        const float *centroid = centroids.data() + c * dim_;
                        const float d =
                            math::kernels::l2_squared(vec, centroid, dim_);
                        if (d < min_dist) {
                                min_dist = d;
                                best_cluster = c;
                        }
                }
                assignments[i] = best_cluster;
        }
        return assignments;
}
} // namespace spheni::math::clustering
