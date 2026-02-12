#include "indexes/ivf/ivf_index.h"
#include "io/serialize.h"
#include "math/kernels.h"
#include "math/kmeans.h"
#include "math/topk.h"
#include "storage/quantization.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace spheni {

IVFIndex::IVFIndex(const IndexSpec &spec)
    : spec_(spec), total_vectors_(0), is_trained_(false) {
  cluster_vectors_.resize(spec.nlist);
  cluster_vectors_i8_.resize(spec.nlist);
  cluster_scales_.resize(spec.nlist);
  cluster_ids_.resize(spec.nlist);
}

void IVFIndex::train() {
  // training is explicit and allowed only once per instance.
  if (is_trained_) {
    throw std::runtime_error("IVFIndex::train: already trained.");
  }
  if (untrained_vectors_.empty()) {
    throw std::runtime_error("IVFIndex::train: no vectors to train on.");
  }

  long long n = untrained_vectors_.size() / spec_.dim;
  if (n < spec_.nlist) {
    throw std::runtime_error("IVFIndex::train: not enough vectors to train.");
  }

  math::clustering::KMeans KMeans(spec_.nlist, spec_.dim, spec_.seed);
  centroids_ = kmeans.fit(untrained_vectors_);

  auto assignments = kmeans.predict(untrained_vectors_, centroids_);

  for (std::size_t i = 0; i < assignments.size(); i++) {
    int cluster = assignments[i];
    const float *vec = untrained_vectors_.data() + i * spec_.dim;
    if (untrained_ids_[i] < 0) {
      continue;
    }
    if (spec_.storage == StorageType::F32) {
      cluster_vectors_[cluster].insert(cluster_vectors_[cluster].end(), vec,
                                       vec + spec_.dim);
    } else if (spec_.storage == StorageType::INT8) {
      quantization::quantize_vector(vec, spec_.dim,
                                    cluster_vectors_i8_[cluster],
                                    cluster_scales_[cluster]);
    } else {
      throw std::runtime_error("IVFIndex::train: unsupported storage type.");
    }

    cluster_ids_[cluster].push_back(untrained_ids_[i]);
  }

  is_trained_ = true;
  untrained_vectors_.clear();
  untrained_ids_.clear();
}

void IVFIndex::add(std::span<const long long> ids,
                   std::span<const float> vectors) {
  long long n = vectors.size() / spec_.dim;
  long long nonneg = 0;
  for (long long i = 0; i < n; ++i) {
    if (ids[i] >= 0) {
      ++nonneg;
    }
  }

  if (!is_trained_) {
    untrained_vectors_.insert(untrained_vectors_.end(), vectors.begin(),
                              vectors.end());
    untrained_ids_.insert(untrained_ids_.end(), ids.begin(), ids.end());
    total_vectors_ += nonneg;
    return;
  }

  for (long long i = 0; i < n; i++) {
    if (ids[i] < 0) {
      continue;
    }
    const float *vec = vectors.data() + i * spec_.dim;
    std::vector<float> vec_copy;
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
      vec_copy.assign(vec, vec + spec_.dim);
      math::kernels::normalize(vec_copy.data(), spec_.dim);
      vec = vec_copy.data();
    }
    int cluster = find_nearest_centroid(vec);

    if (spec_.storage == StorageType::F32) {
      cluster_vectors_[cluster].insert(cluster_vectors_[cluster].end(), vec,
                                       vec + spec_.dim);
    } else if (spec_.storage == StorageType::INT8) {
      quantization::quantize_vector(vec, spec_.dim,
                                    cluster_vectors_i8_[cluster],
                                    cluster_scales_[cluster]);
    } else {
      throw std::runtime_error("IVFIndex::add: unsupported storage type.");
    }

    cluster_ids_[cluster].push_back(ids[i]);
  }
  total_vectors_ += nonneg;
}

int IVFIndex::find_nearest_centroid(const float *vector) const {
  float min_dist = std::numeric_limits<float>::max();
  int best_cluster = 0;

  for (int c = 0; c < spec_.nlist; c++) {
    const float *centroid = centroids_.data() + c * spec_.dim;
    float dist = math::kernels::l2_squared(vector, centroid, spec_.dim);
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = c;
    }
  }
  return best_cluster;
}

float IVFIndex::compute_score(const float *query, const float *db_vec) const {
  switch (spec_.metric) {
  case Metric::Cosine:
    return math::kernels::dot(query, db_vec, spec_.dim);
  case Metric::L2:
    return -math::kernels::l2_squared(query, db_vec, spec_.dim);
  default:
    return 0.0f;
  }
}

static float compute_score_int8(const float *query, const std::int8_t *db_vec,
                                float scale, int dim, Metric metric) {
  switch (metric) {
  case Metric::Cosine: {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float v = scale * static_cast<float>(db_vec[i]);
      sum += query[i] * v;
    }
    return sum;
  }
  case Metric::L2: {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float v = scale * static_cast<float>(db_vec[i]);
      float diff = v - query[i];
      sum += diff * diff;
    }
    return -sum;
  }
  default:
    return 0.0f;
  }
}

std::vector<SearchHit> IVFIndex::search(std::span<const float> query,
                                        const SearchParams &params) const {
  if (!is_trained_) {
    throw std::runtime_error(
        "IVFIndex::search: index not trained. Call Engine::train().");
  }

  std::vector<float> query_copy(query.begin(), query.end());
  if (spec_.normalize && spec_.metric == Metric::Cosine) {
    math::kernels::normalize(query_copy.data(), spec_.dim);
  }
  const float *query_ptr = query_copy.data();

  std::vector<std::pair<float, int>> centroid_dists(spec_.nlist);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int c = 0; c < spec_.nlist; c++) {
    const float *centroid = centroids_.data() + c * spec_.dim;
    float dist = math::kernels::l2_squared(query_ptr, centroid, spec_.dim);
    centroid_dists[c] = {dist, c};
  }

  std::partial_sort(centroid_dists.begin(),
                    centroid_dists.begin() +
                        std::min(params.nprobe, spec_.nlist),
                    centroid_dists.end());

  const int nprobe = std::min(params.nprobe, spec_.nlist);
  math::TopK topk(params.k);

#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
  std::vector<math::TopK> local_topks;
  local_topks.reserve(max_threads);
  for (int t = 0; t < max_threads; ++t) {
    local_topks.emplace_back(params.k);
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    math::TopK &local_topk = local_topks[tid];
#pragma omp for schedule(dynamic)
    for (int p = 0; p < nprobe; p++) {
      int cluster = centroid_dists[p].second;
      long long cluster_size = cluster_ids_[cluster].size();

      for (long long i = 0; i < cluster_size; i++) {
        float score = 0.0f;
        if (spec_.storage == StorageType::F32) {
          const float *vec = cluster_vectors_[cluster].data() + i * spec_.dim;
          score = compute_score(query_ptr, vec);
        } else if (spec_.storage == StorageType::INT8) {
          const std::int8_t *vec =
              cluster_vectors_i8_[cluster].data() + i * spec_.dim;
          float scale = cluster_scales_[cluster][static_cast<std::size_t>(i)];
          score = compute_score_int8(query_ptr, vec, scale, spec_.dim,
                                     spec_.metric);
        } else {
          throw std::runtime_error(
              "IVFIndex::search: unsupported storage type.");
        }
        local_topk.push(cluster_ids_[cluster][i], score);
      }
    }
  }

  for (auto &local_topk : local_topks) {
    auto local_results = local_topk.sorted_results();
    for (const auto &hit : local_results) {
      topk.push(hit.id, hit.score);
    }
  }
#else
  for (int p = 0; p < nprobe; p++) {
    int cluster = centroid_dists[p].second;
    long long cluster_size = cluster_ids_[cluster].size();

    for (long long i = 0; i < cluster_size; i++) {
      float score = 0.0f;
      if (spec_.storage == StorageType::F32) {
        const float *vec = cluster_vectors_[cluster].data() + i * spec_.dim;
        score = compute_score(query_ptr, vec);
      } else if (spec_.storage == StorageType::INT8) {
        const std::int8_t *vec =
            cluster_vectors_i8_[cluster].data() + i * spec_.dim;
        float scale = cluster_scales_[cluster][static_cast<std::size_t>(i)];
        score =
            compute_score_int8(query_ptr, vec, scale, spec_.dim, spec_.metric);
      } else {
        throw std::runtime_error("IVFIndex::search: unsupported storage type.");
      }
      topk.push(cluster_ids_[cluster][i], score);
    }
  }
#endif

  return topk.sorted_results();
}

void IVFIndex::save_state(std::ostream &out) const {
  if (spec_.dim <= 0) {
    throw std::runtime_error("IVFIndex::save_state: invalid dimension.");
  }
  if (static_cast<int>(cluster_vectors_.size()) != spec_.nlist ||
      static_cast<int>(cluster_vectors_i8_.size()) != spec_.nlist ||
      static_cast<int>(cluster_scales_.size()) != spec_.nlist ||
      static_cast<int>(cluster_ids_.size()) != spec_.nlist) {
    throw std::runtime_error(
        "IVFIndex::save_state: cluster list size mismatch.");
  }
  if (is_trained_) {
    std::size_t expected = static_cast<std::size_t>(spec_.nlist) *
                           static_cast<std::size_t>(spec_.dim);
    if (centroids_.size() != expected) {
      throw std::runtime_error("IVFIndex::save_state: centroid size mismatch.");
    }
  } else {
    if (!centroids_.empty()) {
      throw std::runtime_error(
          "IVFIndex::save_state: centroids present before training.");
    }
    for (int c = 0; c < spec_.nlist; ++c) {
      if (!cluster_vectors_[c].empty() || !cluster_vectors_i8_[c].empty() ||
          !cluster_scales_[c].empty() || !cluster_ids_[c].empty()) {
        throw std::runtime_error(
            "IVFIndex::save_state: clusters present before training.");
      }
    }
  }

  io::write_bool(out, is_trained_);
  io::write_pod(out, total_vectors_);
  io::write_vector(out, centroids_);

  io::write_pod(out, static_cast<std::uint64_t>(cluster_vectors_.size()));
  for (int c = 0; c < spec_.nlist; ++c) {
    const auto &ids = cluster_ids_[c];
    if (spec_.storage == StorageType::F32) {
      const auto &vecs = cluster_vectors_[c];
      if (vecs.size() % static_cast<std::size_t>(spec_.dim) != 0) {
        throw std::runtime_error(
            "IVFIndex::save_state: cluster vector size mismatch.");
      }
      if (vecs.size() / static_cast<std::size_t>(spec_.dim) != ids.size()) {
        throw std::runtime_error(
            "IVFIndex::save_state: cluster ids size mismatch.");
      }
      io::write_vector(out, vecs);
    } else if (spec_.storage == StorageType::INT8) {
      const auto &vecs = cluster_vectors_i8_[c];
      const auto &scales = cluster_scales_[c];
      if (vecs.size() % static_cast<std::size_t>(spec_.dim) != 0) {
        throw std::runtime_error(
            "IVFIndex::save_state: cluster vector size mismatch.");
      }
      if (vecs.size() / static_cast<std::size_t>(spec_.dim) != ids.size()) {
        throw std::runtime_error(
            "IVFIndex::save_state: cluster ids size mismatch.");
      }
      if (scales.size() != ids.size()) {
        throw std::runtime_error(
            "IVFIndex::save_state: cluster scales size mismatch.");
      }
      io::write_vector(out, vecs);
      io::write_vector(out, scales);
    } else {
      throw std::runtime_error(
          "IVFIndex::save_state: unsupported storage type.");
    }
    io::write_vector(out, ids);
  }

  io::write_vector(out, untrained_vectors_);
  io::write_vector(out, untrained_ids_);
}

void IVFIndex::load_state(std::istream &in) {
  if (spec_.dim <= 0) {
    throw std::runtime_error("IVFIndex::load_state: invalid dimension.");
  }

  is_trained_ = io::read_bool(in);
  total_vectors_ = io::read_pod<long long>(in);
  centroids_ = io::read_vector<float>(in);

  std::uint64_t cluster_count = io::read_pod<std::uint64_t>(in);
  if (cluster_count != static_cast<std::uint64_t>(spec_.nlist)) {
    throw std::runtime_error("IVFIndex::load_state: cluster count mismatch.");
  }
  cluster_vectors_.assign(spec_.nlist, {});
  cluster_vectors_i8_.assign(spec_.nlist, {});
  cluster_scales_.assign(spec_.nlist, {});
  cluster_ids_.assign(spec_.nlist, {});

  for (int c = 0; c < spec_.nlist; ++c) {
    if (spec_.storage == StorageType::F32) {
      cluster_vectors_[c] = io::read_vector<float>(in);
    } else if (spec_.storage == StorageType::INT8) {
      cluster_vectors_i8_[c] = io::read_vector<std::int8_t>(in);
      cluster_scales_[c] = io::read_vector<float>(in);
    } else {
      throw std::runtime_error(
          "IVFIndex::load_state: unsupported storage type.");
    }
    cluster_ids_[c] = io::read_vector<long long>(in);

    if (spec_.storage == StorageType::F32) {
      if (cluster_vectors_[c].size() % static_cast<std::size_t>(spec_.dim) !=
          0) {
        throw std::runtime_error(
            "IVFIndex::load_state: cluster vector size mismatch.");
      }
      if (cluster_vectors_[c].size() / static_cast<std::size_t>(spec_.dim) !=
          cluster_ids_[c].size()) {
        throw std::runtime_error(
            "IVFIndex::load_state: cluster ids size mismatch.");
      }
    } else if (spec_.storage == StorageType::INT8) {
      if (cluster_vectors_i8_[c].size() % static_cast<std::size_t>(spec_.dim) !=
          0) {
        throw std::runtime_error(
            "IVFIndex::load_state: cluster vector size mismatch.");
      }
      if (cluster_vectors_i8_[c].size() / static_cast<std::size_t>(spec_.dim) !=
          cluster_ids_[c].size()) {
        throw std::runtime_error(
            "IVFIndex::load_state: cluster ids size mismatch.");
      }
      if (cluster_scales_[c].size() != cluster_ids_[c].size()) {
        throw std::runtime_error(
            "IVFIndex::load_state: cluster scales size mismatch.");
      }
    }
  }

  untrained_vectors_ = io::read_vector<float>(in);
  untrained_ids_ = io::read_vector<long long>(in);

  if (untrained_vectors_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
    throw std::runtime_error(
        "IVFIndex::load_state: untrained vector size mismatch.");
  }
  if (untrained_vectors_.size() / static_cast<std::size_t>(spec_.dim) !=
      untrained_ids_.size()) {
    throw std::runtime_error(
        "IVFIndex::load_state: untrained ids size mismatch.");
  }

  if (is_trained_) {
    std::size_t expected = static_cast<std::size_t>(spec_.nlist) *
                           static_cast<std::size_t>(spec_.dim);
    if (centroids_.size() != expected) {
      throw std::runtime_error("IVFIndex::load_state: centroid size mismatch.");
    }
  } else {
    if (!centroids_.empty()) {
      throw std::runtime_error(
          "IVFIndex::load_state: centroids present before training.");
    }
    for (int c = 0; c < spec_.nlist; ++c) {
      if (!cluster_vectors_[c].empty() || !cluster_vectors_i8_[c].empty() ||
          !cluster_scales_[c].empty() || !cluster_ids_[c].empty()) {
        throw std::runtime_error(
            "IVFIndex::load_state: clusters present before training.");
      }
    }
  }

  long long nonneg = 0;
  for (long long id : untrained_ids_) {
    if (id >= 0) {
      ++nonneg;
    }
  }
  for (int c = 0; c < spec_.nlist; ++c) {
    for (long long id : cluster_ids_[c]) {
      if (id >= 0) {
        ++nonneg;
      }
    }
  }
  if (nonneg != total_vectors_) {
    throw std::runtime_error(
        "IVFIndex::load_state: total vector count mismatch.");
  }
}

} // namespace spheni
