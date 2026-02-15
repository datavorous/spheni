#include "indexes/flat/flat_index.h"
#include "io/serialize.h"
#include "math/kernels.h"
#include "math/topk.h"
#include "storage/quantization.h"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace spheni {

FlatIndex::FlatIndex(const IndexSpec &spec) : spec_(spec) {}

void FlatIndex::add(std::span<const long long> ids,
                    std::span<const float> vectors) {
  long long existing = ids_.size();
  long long n = vectors.size() / spec_.dim;

  if (spec_.storage == StorageType::F32) {
    long long offset = existing * spec_.dim;
    vectors_.resize(offset + vectors.size());
    std::copy(vectors.begin(), vectors.end(), vectors_.begin() + offset);

    // normalize if req and cosine
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
      for (long long i = 0; i < n; i++) {
        float *vec = vectors_.data() + (existing + i) * spec_.dim;
        math::kernels::normalize(vec, spec_.dim);
      }
    }
  } else if (spec_.storage == StorageType::INT8) {
    if (spec_.metric == Metric::Haversine) {
      throw std::runtime_error(
          "FlatIndex::add: Haversine not supported with INT8.");
    }

    std::vector<float> vec_copy;
    if (spec_.normalize && spec_.metric == Metric::Cosine) {
      vec_copy.assign(vectors.begin(), vectors.end());
    }

    for (long long i = 0; i < n; i++) {
      const float *vec = vectors.data() + i * spec_.dim;
      if (spec_.normalize && spec_.metric == Metric::Cosine) {
        float *vec_mut = vec_copy.data() + i * spec_.dim;
        math::kernels::normalize(vec_mut, spec_.dim);
        vec = vec_mut;
      }
      quantization::quantize_vector(vec, spec_.dim, vectors_i8_, scales_);
    }
  } else {
    throw std::runtime_error("FlatIndex::add: unsupported storage type.");
  }

  ids_.insert(ids_.end(), ids.begin(), ids.end());
}

float FlatIndex::compute_score(const float *query, const float *db_vec) const {
  switch (spec_.metric) {
  case Metric::Cosine:
    return math::kernels::dot(query, db_vec, spec_.dim);

  case Metric::L2:
    return -math::kernels::l2_squared(query, db_vec, spec_.dim);

  case Metric::Haversine:
    return -math::kernels::haversine(query, db_vec, spec_.dim);
  }

  return 0.0f;
}

float FlatIndex::compute_score_int8(const float *query,
                                    const std::int8_t *db_vec,
                                    float scale) const {
  switch (spec_.metric) {
  case Metric::Cosine: {
    float sum = 0.0f;
    for (int i = 0; i < spec_.dim; ++i) {
      float v = scale * static_cast<float>(db_vec[i]);
      sum += query[i] * v;
    }
    return sum;
  }
  case Metric::L2: {
    float sum = 0.0f;
    for (int i = 0; i < spec_.dim; ++i) {
      float v = scale * static_cast<float>(db_vec[i]);
      float diff = v - query[i];
      sum += diff * diff;
    }
    return -sum;
  }
  case Metric::Haversine:
    throw std::runtime_error(
        "FlatIndex::compute_score_int8: Haversine not supported with INT8.");
  }
  return 0.0f;
}

std::vector<SearchHit> FlatIndex::search(std::span<const float> query,
                                         const SearchParams &params) const {

  long long num_vectors = ids_.size();
  std::vector<float> query_copy;
  const float *query_ptr = query.data();

  if (spec_.normalize && spec_.metric == Metric::Cosine) {
    query_copy.assign(query.begin(), query.end());
    math::kernels::normalize(query_copy.data(), spec_.dim);
    query_ptr = query_copy.data();
  }

  math::TopK topk(params.k);

  if (spec_.storage == StorageType::F32) {
    for (long long i = 0; i < num_vectors; i++) {
      const float *db_vec = vectors_.data() + i * spec_.dim;
      float score = compute_score(query_ptr, db_vec);
      topk.push(ids_[i], score);
    }
  } else if (spec_.storage == StorageType::INT8) {
    for (long long i = 0; i < num_vectors; i++) {
      const std::int8_t *db_vec = vectors_i8_.data() + i * spec_.dim;
      float scale = scales_[static_cast<std::size_t>(i)];
      float score = compute_score_int8(query_ptr, db_vec, scale);
      topk.push(ids_[i], score);
    }
  } else {
    throw std::runtime_error("FlatIndex::search: unsupported storage type.");
  }

  return topk.sorted_results();
}

void FlatIndex::save_state(std::ostream &out) const {
  if (spec_.dim <= 0) {
    throw std::runtime_error("FlatIndex::save_state: invalid dimension.");
  }
  if (spec_.storage == StorageType::F32) {
    if (vectors_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
      throw std::runtime_error("FlatIndex::save_state: vector size mismatch.");
    }
    if (vectors_.size() / static_cast<std::size_t>(spec_.dim) != ids_.size()) {
      throw std::runtime_error("FlatIndex::save_state: ids size mismatch.");
    }
    io::write_vector(out, vectors_);
  } else if (spec_.storage == StorageType::INT8) {
    if (vectors_i8_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
      throw std::runtime_error("FlatIndex::save_state: vector size mismatch.");
    }
    if (vectors_i8_.size() / static_cast<std::size_t>(spec_.dim) !=
        ids_.size()) {
      throw std::runtime_error("FlatIndex::save_state: ids size mismatch.");
    }
    if (scales_.size() != ids_.size()) {
      throw std::runtime_error("FlatIndex::save_state: scales size mismatch.");
    }
    io::write_vector(out, vectors_i8_);
    io::write_vector(out, scales_);
  } else {
    throw std::runtime_error(
        "FlatIndex::save_state: unsupported storage type.");
  }
  io::write_vector(out, ids_);
}

void FlatIndex::load_state(std::istream &in) {
  if (spec_.storage == StorageType::F32) {
    vectors_ = io::read_vector<float>(in);
  } else if (spec_.storage == StorageType::INT8) {
    vectors_i8_ = io::read_vector<std::int8_t>(in);
    scales_ = io::read_vector<float>(in);
  } else {
    throw std::runtime_error(
        "FlatIndex::load_state: unsupported storage type.");
  }
  ids_ = io::read_vector<long long>(in);

  if (spec_.dim <= 0) {
    throw std::runtime_error("FlatIndex::load_state: invalid dimension.");
  }
  if (spec_.storage == StorageType::F32) {
    if (vectors_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
      throw std::runtime_error("FlatIndex::load_state: vector size mismatch.");
    }
    if (vectors_.size() / static_cast<std::size_t>(spec_.dim) != ids_.size()) {
      throw std::runtime_error("FlatIndex::load_state: ids size mismatch.");
    }
  } else if (spec_.storage == StorageType::INT8) {
    if (vectors_i8_.size() % static_cast<std::size_t>(spec_.dim) != 0) {
      throw std::runtime_error("FlatIndex::load_state: vector size mismatch.");
    }
    if (vectors_i8_.size() / static_cast<std::size_t>(spec_.dim) !=
        ids_.size()) {
      throw std::runtime_error("FlatIndex::load_state: ids size mismatch.");
    }
    if (scales_.size() != ids_.size()) {
      throw std::runtime_error("FlatIndex::load_state: scales size mismatch.");
    }
  } else {
    throw std::runtime_error(
        "FlatIndex::load_state: unsupported storage type.");
  }
}

} // namespace spheni
