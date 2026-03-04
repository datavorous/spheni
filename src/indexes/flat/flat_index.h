#pragma once
#include "spheni/spheni.h"
#include <vector>

namespace spheni {

class FlatIndex : public Index {
public:
  explicit FlatIndex(const IndexSpec &spec);

  void add(std::span<const long long> ids,
           std::span<const float> vectors) override;
  std::vector<SearchHit> search(std::span<const float> query,
                                const SearchParams &params) const override;

  const IndexSpec &spec() const override { return spec_; }

  long long size() const override {
    return static_cast<long long>(ids_.size());
  }

  int dim() const override { return spec_.dim; }

private:
  bool should_normalize() const;
  float score_f32(const float *q, const float *v) const;

  IndexSpec spec_;
  std::vector<float> vectors_;
  std::vector<long long> ids_;
};
} // namespace spheni
