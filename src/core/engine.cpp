#include "spheni/engine.h"
#include <algorithm>

namespace spheni {

Engine::Engine(const IndexSpec& spec): index_(make_index(spec)), next_id_(0) {}

void Engine::add(std::span<const float> vectors) {
    int d = index_->dim();
    long long n = vectors.size() / d;
    std::vector<long long> ids(n);

    for (long long i = 0; i < n; i++) {
        ids[i] = next_id_++;
    }

    index_->add(ids, vectors);
}

void Engine::add(std::span<const long long> ids, std::span<const float> vectors) {
    index_->add(ids, vectors);

    auto max_it = std::max_element(ids.begin(), ids.end());

    if (max_it != ids.end()) {
        long long next = *max_it + 1;
        if (next > next_id_) {
            next_id_ = next;
        }
    }
}

std::vector<SearchHit> Engine::search(std::span<const float> query, int k) const {
    SearchParams params(k);
    return index_->search(query, params);
}

std::vector<std::vector<SearchHit>> Engine::search_batch(std::span<const float> queries, int k) const {
    int d = index_->dim();
    long long n = queries.size() / d;
    std::vector<std::vector<SearchHit>> results;

    for (long long i = 0; i < n; ++i) {
        std::span<const float> query(queries.data() + i * d, d);
        results.push_back(search(query, k));
    }

    return results;
}

long long Engine::size() const { return index_->size(); }
int Engine::dim() const { return index_->dim(); }

}
