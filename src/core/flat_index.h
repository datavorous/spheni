#pragma once

#include "spheni/spheni.h"
#include <vector>
#include <span>

namespace spheni {

class FlatIndex : public Index {
public:
    explicit FlatIndex(const IndexSpec& spec);
    
    void add(std::span<const long long> ids, std::span<const float> vectors) override;
    
    std::vector<SearchHit> search(std::span<const float> query, const SearchParams& params) const override;
    
    long long size() const override { return ids_.size(); }
    int dim() const override { return spec_.dim; }
    
private:
    IndexSpec spec_;

    std::vector<float> vectors_;    
    std::vector<long long> ids_;
    float compute_score(const float* query, const float* db_vec) const;
    
};

}
