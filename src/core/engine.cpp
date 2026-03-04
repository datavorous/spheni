// src/core/engine.cpp
#include "spheni/engine.h"
#include "math/utils.h"
#include <algorithm>
#include <numeric>

namespace spheni {

Engine::Engine(IndexSpec spec) : index_(make_index(spec)) {}

void Engine::add(std::span<const float> vectors) {
  const auto n = math::vector_count(vectors, index_->dim());
  std::vector<long long> ids(static_cast<std::size_t>(n));
  std::iota(ids.begin(), ids.end(), next_id_);
  next_id_ += n;
  index_->add(ids, vectors);
}

void Engine::add(std::span<const long long> ids,
                 std::span<const float> vectors) {
  index_->add(ids, vectors);
  if (auto it = std::max_element(ids.begin(), ids.end()); it != ids.end())
    next_id_ = std::max(next_id_, *it + 1);
}

std::vector<SearchHit> Engine::search(std::span<const float> query,
                                      int k) const {
  return search(query, SearchParams{k});
}

std::vector<SearchHit> Engine::search(std::span<const float> query,
                                      SearchParams p) const {
  return index_->search(query, p);
}

std::vector<std::vector<SearchHit>>
Engine::search_batch(std::span<const float> queries, SearchParams p) const {
  const int d = index_->dim();
  const auto n = math::vector_count(queries, d);

  std::vector<std::vector<SearchHit>> out;
  out.reserve(static_cast<std::size_t>(n));

  for (long long i = 0; i < n; ++i) {
    out.push_back(
        index_->search(queries.subspan(static_cast<std::size_t>(i * d),
                                       static_cast<std::size_t>(d)),
                       p));
  }
  return out;
}

} // namespace spheni
