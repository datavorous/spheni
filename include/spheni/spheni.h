#pragma once

#include <memory>
#include <span>
#include <vector>

namespace spheni {

enum class Metric {
    Cosine,
    L2
};

enum class IndexKind {
    Flat
};

struct IndexSpec {
    int dim;
    
    Metric metric;
    bool normalize;
    IndexKind kind;

    IndexSpec(int d, Metric m, IndexKind k, bool norm = true): dim(d), metric(m), normalize(norm), kind(k) {}
};

struct SearchParams {
    int k;
    SearchParams(int k_) : k(k_) {}
};

struct SearchHit {
    long long id;
    float score;
    SearchHit(long long id_, float score_) : id(id_), score(score_) {}
};


class Index {
public:
    virtual ~Index() = default;
    virtual void add(std::span<const long long> ids, std::span<const float> vectors) = 0;
    
    // returns k hits sorted by score descending
    virtual std::vector<SearchHit> search(std::span<const float> query, const SearchParams& params) const = 0;
    virtual long long size() const = 0;
    virtual int dim() const = 0;
};

std::unique_ptr<Index> make_index(const IndexSpec& spec);
}
