#pragma once

#include "spheni/spheni.h"
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace spheni {
class Engine {
public:
    Engine(const IndexSpec& spec);

    void add(std::span<const float> vectors);
    void add(std::span<const long long> ids, std::span<const float> vectors);

    std::vector<SearchHit> search(std::span<const float> query, int k) const;
    std::vector<SearchHit> search(std::span<const float> query, int k, int nprobe) const;
    std::vector<std::vector<SearchHit>> search_batch(std::span<const float> queries, int k) const;

    long long size() const;
    int dim() const;

    void save(const std::string& path) const;
    static Engine load(const std::string& path);

private:
    std::unique_ptr<Index> index_;
    long long next_id_;
};
}
