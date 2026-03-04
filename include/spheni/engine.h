#pragma once

#include "spheni/spheni.h"
#include <memory>
#include <span>
#include <vector>

namespace spheni {
class Engine {
public:
  explicit Engine(IndexSpec spec);

  void add(std::span<const float> vectors);
  void add(std::span<const long long> ids, std::span<const float> vectors);

  std::vector<SearchHit> search(std::span<const float> query, int k) const;
  std::vector<SearchHit> search(std::span<const float> query,
                                SearchParams p) const;

  std::vector<std::vector<SearchHit>>
  search_batch(std::span<const float> queries, SearchParams p) const;

  void train() { index_->train(); }
  long long size() const { return index_->size(); }
  int dim() const { return index_->dim(); }

private:
  std::unique_ptr<Index> index_;
  long long next_id_ = 0;
};
} // namespace spheni
