#include "math/math.h"
#include "math/topk.h"
#include "spheni.h"

namespace spheni {

FlatIndex::FlatIndex(const Spec &spec) : spec_(spec) {}

bool FlatIndex::should_normalize() const {
        return spec_.normalize && spec_.metric == Metric::Cosine;
}

float FlatIndex::score_f32(const float *q, const float *v) const {
        switch (spec_.metric) {
        case Metric::Cosine:
                return math::kernels::dot(q, v, spec_.dim);
        case Metric::L2:
                return -math::kernels::l2_squared(q, v, spec_.dim);
        }
}

void FlatIndex::add(std::span<const long long> ids,
                    std::span<const float> vecs) {
        const int d = spec_.dim;
        const int n = vecs.size() / d;
        const bool normalize_inputs = should_normalize();

        ids_.insert(ids_.end(), ids.begin(), ids.end());

        if (!normalize_inputs) {
                vecs_.insert(vecs_.end(), vecs.begin(), vecs.end());
                return;
        }

        std::vector<float> tmp(d);
        for (int i = 0; i < n; i++) {
                const float *src = vecs.data() + i * d;
                std::copy(src, src + d, tmp.begin());
                math::kernels::normalize(tmp.data(), d);
                vecs_.insert(vecs_.end(), tmp.begin(), tmp.end());
        }
}

std::vector<Hit> FlatIndex::search(std::span<const float> query, int k) const {
        const bool normalize_query = should_normalize();
        std::vector<float> tmp;
        const float *q = query.data();
        if (normalize_query) {
                tmp.assign(query.begin(), query.end());
                math::kernels::normalize(tmp.data(), spec_.dim);
                q = tmp.data();
        }

        math::TopK topk(k);
        for (int i = 0; i < (int)ids_.size(); i++)
                topk.push(ids_[i], score_f32(q, vecs_.data() + i * spec_.dim));
        return topk.take_sorted();
}

} // namespace spheni
