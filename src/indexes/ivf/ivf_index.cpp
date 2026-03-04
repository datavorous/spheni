#include "indexes/ivf/ivf_index.h"
#include "math/kernels.h"
#include "math/kmeans.h"
#include "math/topk.h"
#include "math/utils.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace spheni {

IVFIndex::IVFIndex(const IndexSpec &spec) : spec_(spec) {
  IndexSpec cell_spec = spec;
  cell_spec.kind = IndexKind::Flat;
  cell_spec.nlist = 0;
  cells_.reserve(static_cast<std::size_t>(spec_.nlist));
  for (int i = 0; i < spec_.nlist; ++i) {
    cells_.emplace_back(cell_spec);
  }
}

bool IVFIndex::should_normalize() const {
  return spec_.normalize && spec_.metric == Metric::Cosine;
}

int IVFIndex::nearest_centroid(const float *vec) const {
  float best = std::numeric_limits<float>::max();
  int idx = 0;
  for (int c = 0; c < spec_.nlist; ++c) {
    const float d = math::kernels::l2_squared(
        vec, centroids_.data() + c * spec_.dim, spec_.dim);
    if (d < best) {
      best = d;
      idx = c;
    }
  }
  return idx;
}

void IVFIndex::train() {
  const int dim = spec_.dim;
  math::clustering::KMeans kmeans(spec_.nlist, spec_.dim);
  centroids_ = kmeans.fit(pending_vecs_);

  const auto assignments = kmeans.predict(pending_vecs_, centroids_);
  for (std::size_t i = 0; i < assignments.size(); ++i) {
    if (pending_ids_[i] < 0) {
      continue;
    }
    cells_[assignments[i]].add(
        std::span<const long long>(&pending_ids_[i], 1),
        std::span<const float>(pending_vecs_.data() + i * dim,
                               static_cast<std::size_t>(dim)));
  }

  trained_ = true;
  pending_vecs_.clear();
  pending_ids_.clear();
}

void IVFIndex::add(std::span<const long long> ids,
                   std::span<const float> vectors) {
  const int dim = spec_.dim;
  const long long n = math::vector_count(vectors, dim);

  if (!trained_) {
    pending_vecs_.insert(pending_vecs_.end(), vectors.begin(), vectors.end());
    pending_ids_.insert(pending_ids_.end(), ids.begin(), ids.end());
    for (const auto id : ids) {
      if (id >= 0) {
        ++ntotal_;
      }
    }
    return;
  }

  const bool normalize_inputs = should_normalize();
  std::vector<float> normalized;
  if (normalize_inputs) {
    normalized.resize(static_cast<std::size_t>(dim));
  }

  for (long long i = 0; i < n; ++i) {
    if (ids[i] < 0) {
      continue;
    }

    const float *src = vectors.data() + i * dim;
    const float *centroid_input = src;
    if (normalize_inputs) {
      normalized.assign(src, src + dim);
      math::kernels::normalize(normalized.data(), dim);
      centroid_input = normalized.data();
    }

    cells_[nearest_centroid(centroid_input)].add(
        std::span<const long long>(&ids[i], 1),
        std::span<const float>(src, static_cast<std::size_t>(dim)));
    ++ntotal_;
  }
}

std::vector<SearchHit> IVFIndex::search(std::span<const float> query,
                                        const SearchParams &params) const {
  const int dim = spec_.dim;
  const float *q = query.data();
  std::vector<float> normalized_query;
  if (should_normalize()) {
    normalized_query.assign(query.begin(), query.end());
    math::kernels::normalize(normalized_query.data(), dim);
    q = normalized_query.data();
  }
  const std::span<const float> query_span(q, static_cast<std::size_t>(dim));

  std::vector<std::pair<float, int>> dists(spec_.nlist);
  for (int c = 0; c < spec_.nlist; ++c) {
    dists[c] = {math::kernels::l2_squared(q, centroids_.data() + c * dim, dim),
                c};
  }

  const int nprobe = std::min(params.nprobe, spec_.nlist);
  std::partial_sort(dists.begin(), dists.begin() + nprobe, dists.end());

  math::TopK topk(params.k);

#ifdef _OPENMP
  std::vector<math::TopK> locals(omp_get_max_threads(), math::TopK(params.k));
#pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < nprobe; ++p) {
    auto cell_hits = cells_[dists[p].second].search(query_span, params);
    for (auto &h : cell_hits) {
      locals[omp_get_thread_num()].push(h.id, h.score);
    }
  }
  for (auto &local : locals) {
    for (auto &h : local.take_sorted()) {
      topk.push(h.id, h.score);
    }
  }
#else
  for (int p = 0; p < nprobe; ++p) {
    auto cell_hits = cells_[dists[p].second].search(query_span, params);
    for (auto &h : cell_hits) {
      topk.push(h.id, h.score);
    }
  }
#endif

  return topk.take_sorted();
}

} // namespace spheni
