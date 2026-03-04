#include "math/kmeans.h"
#include "math/math.h"
#include "math/topk.h"
#include "spheni.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace spheni {

IVFIndex::IVFIndex(const IVFSpec &spec) : spec_(spec) {
  cells_.reserve(spec_.nlist);
  for (int i = 0; i < spec_.nlist; i++)
    cells_.emplace_back(spec_);
}

bool IVFIndex::should_normalize() const {
  return spec_.normalize && spec_.metric == Metric::Cosine;
}

int IVFIndex::nearest_centroid(const float *vec) const {
  float best = __INT_MAX__;
  int idx = 0;
  for (int c = 0; c < spec_.nlist; c++) {
    float d = math::kernels::l2_squared(vec, centroids_.data() + c * spec_.dim,
                                        spec_.dim);
    if (d < best) {
      best = d;
      idx = c;
    }
  }
  return idx;
}

void IVFIndex::train(std::span<const long long> ids,
                     std::span<const float> vecs) {
  const int dim = spec_.dim;
  math::clustering::KMeans kmeans(spec_.nlist, dim);
  centroids_ = kmeans.fit(vecs);

  const int n = vecs.size() / dim;
  const auto assignments = kmeans.predict(vecs, centroids_);
  for (int i = 0; i < n; i++) {
    cells_[assignments[i]].add(
        std::span<const long long>(&ids[i], 1),
        std::span<const float>(vecs.data() + i * dim, dim));
    ++ntotal_;
  }
  trained_ = true;
}

void IVFIndex::add(std::span<const long long> ids,
                   std::span<const float> vecs) {
  assert(trained_);
  const int dim = spec_.dim;
  const int n = vecs.size() / dim;
  const bool normalize_inputs = should_normalize();

  std::vector<float> tmp;
  if (normalize_inputs)
    tmp.resize(dim);

  for (int i = 0; i < n; i++) {
    const float *src = vecs.data() + i * dim;
    const float *v = src;
    if (normalize_inputs) {
      tmp.assign(src, src + dim);
      math::kernels::normalize(tmp.data(), dim);
      v = tmp.data();
    }
    cells_[nearest_centroid(v)].add(std::span<const long long>(&ids[i], 1),
                                    std::span<const float>(v, dim));
    ++ntotal_;
  }
}

std::vector<Hit> IVFIndex::search(std::span<const float> query, int k) const {
  const int dim = spec_.dim;
  const bool normalize_query = should_normalize();
  std::vector<float> tmp;
  const float *q = query.data();
  if (normalize_query) {
    tmp.assign(query.begin(), query.end());
    math::kernels::normalize(tmp.data(), dim);
    q = tmp.data();
  }

  std::vector<std::pair<float, int>> dists(spec_.nlist);
  for (int c = 0; c < spec_.nlist; c++)
    dists[c] = {math::kernels::l2_squared(q, centroids_.data() + c * dim, dim),
                c};

  const int nprobe = std::min(spec_.nprobe, spec_.nlist);
  std::partial_sort(dists.begin(), dists.begin() + nprobe, dists.end());

  math::TopK topk(k);
  for (int p = 0; p < nprobe; p++) {
    auto hits =
        cells_[dists[p].second].search(std::span<const float>(q, dim), k);
    for (auto &h : hits)
      topk.push(h.id, h.score);
  }
  return topk.take_sorted();
}

} // namespace spheni
