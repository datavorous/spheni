#include "indexes/flat/flat_index.h"
#include "math/kernels.h"
#include "math/topk.h"
#include "math/utils.h"
#include <cstddef>

namespace spheni {

FlatIndex::FlatIndex(const IndexSpec &spec) : spec_(spec) {}

bool FlatIndex::should_normalize() const {
  return spec_.normalize && spec_.metric == Metric::Cosine;
}

float FlatIndex::score_f32(const float *q, const float *v) const {
  switch (spec_.metric) {
  case Metric::Cosine:
    return math::kernels::dot(q, v, spec_.dim);
  case Metric::L2:
    return -math::kernels::l2_squared(q, v, spec_.dim);
  }
  std::terminate();
}

void FlatIndex::add(std::span<const long long> ids,
                    std::span<const float> vecs) {
  const int d = spec_.dim;
  const long long n = math::vector_count(vecs, d);

  ids_.insert(ids_.end(), ids.begin(), ids.end());
  vectors_.reserve(vectors_.size() + vecs.size());

  if (!should_normalize()) {
    vectors_.insert(vectors_.end(), vecs.begin(), vecs.end());
    return;
  }

  std::vector<float> tmp(static_cast<std::size_t>(d));
  for (long long i = 0; i < n; ++i) {
    const float *src = vecs.data() + i * d;
    std::copy(src, src + d, tmp.begin());
    math::kernels::normalize(tmp.data(), d);
    vectors_.insert(vectors_.end(), tmp.begin(), tmp.end());
  }
}

std::vector<SearchHit> FlatIndex::search(std::span<const float> query,
                                         const SearchParams &params) const {
  std::vector<float> normalized_query;
  const float *q = query.data();
  if (should_normalize()) {
    normalized_query.assign(query.begin(), query.end());
    math::kernels::normalize(normalized_query.data(), spec_.dim);
    q = normalized_query.data();
  }

  math::TopK topk(params.k);
  const long long n = static_cast<long long>(ids_.size());
  for (long long i = 0; i < n; ++i) {
    topk.push(ids_[i], score_f32(q, vectors_.data() + i * spec_.dim));
  }
  return topk.take_sorted();
}
} // namespace spheni
