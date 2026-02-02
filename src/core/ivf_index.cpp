#include "core/ivf_index.h"
#include "utils/kmeans.h"
#include "utils/kernels.h"
#include "utils/topk.h"
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace spheni {

IVFIndex::IVFIndex(const IndexSpec& spec): spec_(spec), total_vectors_(0), is_trained_(false) {
    cluster_vectors_.resize(spec.nlist);
    cluster_ids_.resize(spec.nlist);
}

// TODO: finish this tomorrow
void IVFIndex::train() {
    if (is_trained_ || untrained_vectors_.empty()) {
        return;
    }

    long long n = untrained_vectors_.size() / spec_.dim;
    if (n<spec_.nlist) { return; }


    clustering::KMeans kmeans(spec_.nlist, spec_.dim);
    centroids_ = kmeans.fit(untrained_vectors_);

    auto assignments = kmeans.predict(untrained_vectors_, centroids_);

    for(std::size_t i = 0; i < assignments.size(); i++) {
        int cluster = assignments[i];
        const float* vec = untrained_vectors_.data() + i * spec_.dim;
        if (untrained_ids_[i] < 0) {
            continue;
        }
        cluster_vectors_[cluster].insert(
            cluster_vectors_[cluster].end(),
            vec,
            vec + spec_.dim);

        cluster_ids_[cluster].push_back(untrained_ids_[i]);
    }

    is_trained_ = true;
    untrained_vectors_.clear();
    untrained_ids_.clear();
}


void IVFIndex::add(std::span<const long long> ids, std::span<const float> vectors) {
    long long n = vectors.size() / spec_.dim;
    long long nonneg = 0;
    for (long long i = 0; i < n; ++i) {
        if (ids[i] >= 0) {
            ++nonneg;
        }
    }

    if (!is_trained_) {
        untrained_vectors_.insert(untrained_vectors_.end(), vectors.begin(), vectors.end());
        untrained_ids_.insert(untrained_ids_.end(), ids.begin(), ids.end());
        total_vectors_ += nonneg;
        train();
        return;
    }

    std::vector<float> vecs_copy(vectors.begin(), vectors.end());

    if (spec_.normalize && spec_.metric == Metric::Cosine) {
        for(long long i = 0; i < n; i++) {
            float* vec = vecs_copy.data() + i * spec_.dim;
            kernels::normalize(vec, spec_.dim);
        }
    }


    for(long long i = 0; i < n; i++) {
        if (ids[i] < 0) {
            continue;
        }
        const float* vec = vecs_copy.data() + i * spec_.dim;
        int cluster = find_nearest_centroid(vec);

        cluster_vectors_[cluster].insert(
            cluster_vectors_[cluster].end(),
            vec,
            vec + spec_.dim);

        cluster_ids_[cluster].push_back(ids[i]);
    }
    total_vectors_ += nonneg;
}


int IVFIndex::find_nearest_centroid(const float* vector) const {
    float min_dist = std::numeric_limits<float>::max();
    int best_cluster = 0;

    for(int c = 0; c < spec_.nlist; c++) {
        const float* centroid = centroids_.data() + c * spec_.dim;
        float dist = kernels::l2_squared(vector, centroid, spec_.dim);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }
    return best_cluster;
}



float IVFIndex::compute_score(const float* query, const float* db_vec) const {
    switch (spec_.metric) {
        case Metric::Cosine:
            return kernels::dot(query, db_vec, spec_.dim);
        case Metric::L2:
            return -kernels::l2_squared(query, db_vec, spec_.dim);
        default:
            return 0.0f;
    }
}


std::vector<SearchHit> IVFIndex::search(std::span<const float> query, const SearchParams& params) const {
    if (!is_trained_) {

        TopK topk(params.k);
        long long n = untrained_vectors_.size() / spec_.dim;

        std::vector<float> query_copy(query.begin(), query.end());
        if (spec_.normalize && spec_.metric == Metric::Cosine) {
            kernels::normalize(query_copy.data(), spec_.dim);
        }
        for(long long i = 0; i < n; i++) {
            if (untrained_ids_[i] < 0) {
                continue;
            }
            const float* vec = untrained_vectors_.data() + i * spec_.dim;
            float score = compute_score(query_copy.data(), vec);
            topk.push(untrained_ids_[i], score);
        }
        return topk.sorted_results();
    }


    std::vector<float> query_copy(query.begin(), query.end());
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
        kernels::normalize(query_copy.data(), spec_.dim);
    }
    const float *query_ptr = query_copy.data();

    std::vector<std::pair<float, int>> centroid_dists(spec_.nlist);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int c = 0; c < spec_.nlist; c++) {
        const float* centroid = centroids_.data() + c * spec_.dim;
        float dist = kernels::l2_squared(query_ptr, centroid, spec_.dim);
        centroid_dists[c] = {dist, c};
    }

    std::partial_sort(
        centroid_dists.begin(),
        centroid_dists.begin() + std::min(params.nprobe, spec_.nlist),
        centroid_dists.end());

    const int nprobe = std::min(params.nprobe, spec_.nlist);
    TopK topk(params.k);

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    std::vector<TopK> local_topks;
    local_topks.reserve(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        local_topks.emplace_back(params.k);
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        TopK& local_topk = local_topks[tid];
#pragma omp for schedule(dynamic)
        for(int p = 0; p < nprobe; p++) {
            int cluster = centroid_dists[p].second;
            long long cluster_size = cluster_ids_[cluster].size();

            for(long long i = 0; i < cluster_size; i++) {
                const float* vec = cluster_vectors_[cluster].data() + i * spec_.dim;
                float score = compute_score(query_ptr, vec);
                local_topk.push(cluster_ids_[cluster][i], score);
            }
        }
    }

    for (auto& local_topk : local_topks) {
        auto local_results = local_topk.sorted_results();
        for (const auto& hit : local_results) {
            topk.push(hit.id, hit.score);
        }
    }
#else
    for(int p = 0; p < nprobe; p++) {
        int cluster = centroid_dists[p].second;
        long long cluster_size = cluster_ids_[cluster].size();

        for(long long i = 0; i < cluster_size; i++) {
            const float* vec = cluster_vectors_[cluster].data() + i * spec_.dim;
            float score = compute_score(query_ptr, vec);
            topk.push(cluster_ids_[cluster][i], score);
        }
    }
#endif

    return topk.sorted_results();
}


}
