#include "math/math.h"
#include "math/pq.h"
#include "math/topk.h"
#include "spheni.h"

#include <cassert>
#include <stdexcept>

namespace spheni {

PQFlatIndex::PQFlatIndex(const PQFlatSpec &spec) : spec_(spec) {
        pq_ = std::make_unique<math::ProductQuantizer>(spec_.dim, spec_.M,
                                                       spec_.ksub);
}

bool PQFlatIndex::should_normalize() const {
        return spec_.normalize; // && spec_.metric == Metric::Cosine;
}

void PQFlatIndex::train(std::span<const float> vecs) {
        const int n = vecs.size() / spec_.dim;
        const bool norm = should_normalize();

        if (!norm)
                pq_->train(vecs);
        else {
                std::vector<float> tmp(vecs.begin(), vecs.end());
                for (int i = 0; i < n; i++)
                        math::kernels::normalize(tmp.data() + i * spec_.dim,
                                                 spec_.dim);
                pq_->train(std::span<const float>(tmp.data(), tmp.size()));
        }
        trained_ = true;
}

void PQFlatIndex::add(std::span<const long long> ids,
                      std::span<const float> vecs) {
        assert(trained_);
        const int n = vecs.size() / spec_.dim;
        const bool norm = should_normalize();

        ids_.insert(ids_.end(), ids.begin(), ids.end());

        if (!norm) {
                auto batch = pq_->encode(vecs);
                codes_.insert(codes_.end(), batch.begin(), batch.end());
                return;
        }

        std::vector<float> tmp(spec_.dim);
        for (int i = 0; i < n; i++) {
                const float *src = vecs.data() + i * spec_.dim;
                std::copy(src, src + spec_.dim, tmp.begin());
                math::kernels::normalize(tmp.data(), spec_.dim);
                auto code = pq_->encode_one(tmp.data());
                codes_.insert(codes_.end(), code.begin(), code.end());
        }
        // printf(">> codes_.size() is %zu\n", codes_.size());
}

std::vector<Hit> PQFlatIndex::search(std::span<const float> query,
                                     int k) const {
        const bool norm = should_normalize();
        std::vector<float> tmp;
        const float *q = query.data();
        if (norm) {
                tmp.assign(query.begin(), query.end());
                math::kernels::normalize(tmp.data(), spec_.dim);
                q = tmp.data();
        }

        auto table = pq_->precompute_table(q);
        const int M = pq_->M();
        math::TopK topk(k);
        // printf("ids_.size()=%zu codes_.size()=%zu M=%d expected_codes=%zu\n",
        // ids_.size(), codes_.size(), M, ids_.size() * M);

        for (int i = 0; i < (int)(ids_.size()); i++) {
                float approx =
                    -pq_->approx_distance(table, codes_.data() + i * M);
                topk.push(ids_[i], approx);
        }
        return topk.take_sorted();
}
} // namespace spheni