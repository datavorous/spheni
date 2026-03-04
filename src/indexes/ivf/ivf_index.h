#pragma once
#include "indexes/flat/flat_index.h"
#include <vector>

namespace spheni {
class IVFIndex : public Index {
public:
  explicit IVFIndex(const IndexSpec &spec);

  void train() override;
  void add(std::span<const long long> ids,
           std::span<const float> vectors) override;
  std::vector<SearchHit> search(std::span<const float> query,
                                const SearchParams &params) const override;
  const IndexSpec &spec() const override { return spec_; }
  long long size() const override { return ntotal_; }
  int dim() const override { return spec_.dim; }

private:
  bool should_normalize() const;
  int nearest_centroid(const float *vec) const;

  IndexSpec spec_;
  std::vector<float> centroids_;
  std::vector<FlatIndex> cells_;
  long long ntotal_ = 0;
  bool trained_ = false;

  std::vector<float> pending_vecs_;
  std::vector<long long> pending_ids_;
};
} // namespace spheni
