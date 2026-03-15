#include "math/kmeans.h"
#include "math/math.h"
#include "math/pq.h"
#include "math/topk.h"
#include "spheni.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace spheni {

IVFPQIndex::IVFPQIndex(const IVFPQSpec &spec) : spec_(spec) {
        pq_ = std::make_unique<math::ProductQuantizer>(spec_.dim, spec_.M,
                                                       spec_.ksub);
        cells_.resize(spec_.nlist);
}

IVFPQIndex::~IVFPQIndex() = default;

bool IVFPQIndex::should_normalize() const { return spec_.normalize; }

int IVFPQIndex::nearest_centroid(const float *vec) const {
        float best = std::numeric_limits<float>::max();
        int idx = 0;
        for (int c = 0; c < spec_.nlist; ++c) {
                float d = math::kernels::l2_squared(
                    vec, centroids_.data() + c * spec_.dim, spec_.dim);
                if (d < best) {
                        best = d;
                        idx = c;
                }
        }
        return idx;
}

void IVFPQIndex::train(std::span<const float> vecs) {
        const int n = vecs.size() / spec_.dim;
        const int dim = spec_.dim;
        const bool norm = should_normalize();

        std::vector<float> work;
        std::span<const float> train_vecs = vecs;
        if (norm) {
                work.assign(vecs.begin(), vecs.end());
                for (int i = 0; i < n; i++)
                        math::kernels::normalize(work.data() + i * dim, dim);
                train_vecs = std::span<const float>(work.data(), work.size());
        }
        math::clustering::KMeans coarse_km(spec_.nlist, dim);
        centroids_ = coarse_km.fit(train_vecs);

        auto assignments = coarse_km.predict(train_vecs, centroids_);
        std::vector<float> residuals(n * dim);
        for (int i = 0; i < n; i++) {
                const float *vec = train_vecs.data() + i * dim;
                const float *centroid =
                    centroids_.data() + assignments[i] * dim;
                float *res = residuals.data() + i * dim;
                for (int d = 0; d < dim; d++)
                        res[d] = vec[d] - centroid[d];
        }
        pq_->train(std::span<const float>(residuals.data(), residuals.size()));
        trained_ = true;
}

void IVFPQIndex::add(std::span<const long long> ids,
                     std::span<const float> vecs) {
        assert(trained_);
        const int n = vecs.size() / spec_.dim;
        const int dim = spec_.dim;
        const bool norm = should_normalize();

        std::vector<float> temp(dim);
        std::vector<float> residual(dim);

        for (int i = 0; i < n; i++) {
                const float *src = vecs.data() + i * dim;
                if (norm) {
                        std::copy(src, src + dim, temp.begin());
                        math::kernels::normalize(temp.data(), dim);
                        src = temp.data();
                }

                const int cell_index = nearest_centroid(src);
                const float *centroid = centroids_.data() + cell_index * dim;

                for (int d = 0; d < dim; d++)
                        residual[d] = src[d] - centroid[d];

                auto code = pq_->encode_one(residual.data());

                Cell &cell = cells_[cell_index];
                cell.ids.push_back(ids[i]);
                cell.codes.insert(cell.codes.end(), code.begin(), code.end());
                ntotal_++;
        }
}

std::vector<Hit> IVFPQIndex::search(std::span<const float> query, int k) const {
        const int dim = spec_.dim;
        const bool norm = should_normalize();
        const int M = pq_->M();
        const int nprobe = std::min(spec_.nprobe, spec_.nlist);

        std::vector<float> temp;
        const float *q = query.data();
        if (norm) {
                temp.assign(query.begin(), query.end());
                math::kernels::normalize(temp.data(), dim);
                q = temp.data();
        }

        std::vector<std::pair<float, int>> cell_dists(spec_.nlist);
        for (int c = 0; c < spec_.nlist; c++)
                cell_dists[c] = {math::kernels::l2_squared(
                                     q, centroids_.data() + c * dim, dim),
                                 c};

        std::partial_sort(cell_dists.begin(), cell_dists.begin() + nprobe,
                          cell_dists.end());
        // auto table = pq_->compute_distance_table(q);
        math::TopK topk(k);

        std::vector<float> residual(dim);

        for (int p = 0; p < nprobe; p++) {
                const int cell_index = cell_dists[p].second;
                const Cell &cell = cells_[cell_index];
                if (cell.ids.empty())
                        continue;

                const float *centroid = centroids_.data() + cell_index * dim;
                for (int d = 0; d < dim; d++)
                        residual[d] = q[d] - centroid[d];

                auto table = pq_->precompute_table(residual.data());

                const int cell_size = (int)cell.ids.size();
                for (int i = 0; i < cell_size; i++) {
                        float approx = -pq_->approx_distance(
                            table, cell.codes.data() + i * M);
                        topk.push(cell.ids[i], approx);
                }
        }
        return topk.take_sorted();
}

size_t IVFPQIndex::compressed_bytes() const {
        size_t total = 0;
        for (const auto &cell : cells_)
                total += cell.codes.size();
        return total;
}

size_t IVFPQIndex::uncompressed_bytes() const {
        return (size_t)ntotal_ * spec_.dim * sizeof(float);
}

} // namespace spheni