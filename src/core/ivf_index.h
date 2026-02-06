#pragma once

#include "spheni/spheni.h"
#include <iosfwd>
#include <vector>
#include <span>

namespace spheni {
    class IVFIndex : public Index {
    public:
        IVFIndex(const IndexSpec& spec);
        void add(std::span<const long long> ids, std::span<const float> vectors) override;
        std::vector<SearchHit> search(std::span<const float> query, const SearchParams& params) const override;
        void train();

        long long size() const override { return total_vectors_; }
        int dim() const override { return spec_.dim; }

        const IndexSpec& spec() const { return spec_; }
        void save_state(std::ostream& out) const;
        void load_state(std::istream& in);

    private:
        IndexSpec spec_;

        std::vector<float> centroids_;
        std::vector<std::vector<float>> cluster_vectors_;
        std::vector<std::vector<long long>> cluster_ids_;


        long long total_vectors_;
        bool is_trained_;


        std::vector<float> untrained_vectors_;
        std::vector<long long> untrained_ids_;

        int find_nearest_centroid(const float* vector) const;
        float compute_score(const float* query, const float* vector) const;
    };
}
