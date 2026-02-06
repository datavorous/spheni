#include "core/flat_index.h"
#include "core/serialize.h"
#include "utils/kernels.h"
#include "utils/topk.h"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace spheni {

FlatIndex::FlatIndex(const IndexSpec& spec): spec_(spec) {}

void FlatIndex::add(std::span<const long long> ids, std::span<const float> vectors) {

    long long existing = ids_.size();
    long long offset = existing * spec_.dim;
    vectors_.resize(offset + vectors.size());
    std::copy(vectors.begin(), vectors.end(), vectors_.begin() + offset);
    
    // normalize if req and cosine
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
        for (std::size_t i = 0; i < ids.size(); i++) {
            float* vec = vectors_.data() + (existing + i) * spec_.dim;
            kernels::normalize(vec, spec_.dim);
        }
    }
    
    ids_.insert(ids_.end(), ids.begin(), ids.end());
}

float FlatIndex::compute_score(const float* query, const float* db_vec) const {
    switch (spec_.metric) {
        case Metric::Cosine:
            return kernels::dot(query, db_vec, spec_.dim);
            
        case Metric::L2:
            return -kernels::l2_squared(query, db_vec, spec_.dim);
    }
    
    return 0.0f;
}

std::vector<SearchHit> FlatIndex::search(
    std::span<const float> query,
    const SearchParams& params) const {
    
    long long num_vectors = ids_.size();
    std::vector<float> query_copy;
    const float* query_ptr = query.data();
    
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
        query_copy.assign(query.begin(), query.end());
        kernels::normalize(query_copy.data(), spec_.dim);
        query_ptr = query_copy.data();
    }
    
    TopK topk(params.k);
    
    for (long long i = 0; i < num_vectors; i++) {
        const float* db_vec = vectors_.data() + i * spec_.dim;
        float score = compute_score(query_ptr, db_vec);
        topk.push(ids_[i], score);
    }
    
    return topk.sorted_results();
}

void FlatIndex::save_state(std::ostream& out) const {
    if (spec_.dim <= 0) {
        throw std::runtime_error("FlatIndex::save_state: invalid dimension.");
    }
    if (vectors_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
        throw std::runtime_error("FlatIndex::save_state: vector size mismatch.");
    }
    if (vectors_.size() / static_cast<std::size_t>(spec_.dim) != ids_.size()) {
        throw std::runtime_error("FlatIndex::save_state: ids size mismatch.");
    }

    io::write_vector(out, vectors_);
    io::write_vector(out, ids_);
}

void FlatIndex::load_state(std::istream& in) {
    vectors_ = io::read_vector<float>(in);
    ids_ = io::read_vector<long long>(in);

    if (spec_.dim <= 0) {
        throw std::runtime_error("FlatIndex::load_state: invalid dimension.");
    }
    if (vectors_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
        throw std::runtime_error("FlatIndex::load_state: vector size mismatch.");
    }
    if (vectors_.size() / static_cast<std::size_t>(spec_.dim) != ids_.size()) {
        throw std::runtime_error("FlatIndex::load_state: ids size mismatch.");
    }
}

}
