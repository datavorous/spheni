#pragma once
#include <memory>
#include <span>
#include <vector>

namespace spheni::math {
class ProductQuantizer;
}

namespace spheni {

enum class Metric { Cosine, L2 };

struct Spec {
        int dim;
        Metric metric = Metric::Cosine;
        bool normalize = true;
};

struct IVFSpec : Spec {
        int nlist = 0;
        int nprobe = 1;
};

struct PQFlatSpec : Spec {
        int M = 8;
        int ksub = 256;
};

struct IVFPQSpec : Spec {
        int nlist = 256;
        int nprobe = 8;
        int M = 8;
        int ksub = 256;
};

struct Hit {
        long long id;
        float score;
};

class FlatIndex {
      public:
        explicit FlatIndex(const Spec &spec);
        void add(std::span<const long long> ids, std::span<const float> vecs);
        std::vector<Hit> search(std::span<const float> query, int k) const;
        long long size() const { return ids_.size(); }

      private:
        Spec spec_;
        std::vector<long long> ids_;
        std::vector<float> vecs_;
        bool should_normalize() const;
        float score_f32(const float *q, const float *v) const;
};

class IVFIndex {
      public:
        explicit IVFIndex(const IVFSpec &spec);
        void train(std::span<const long long> ids,
                   std::span<const float> vectors);
        void add(std::span<const long long> ids, std::span<const float> vecs);
        std::vector<Hit> search(std::span<const float> query, int k) const;
        long long size() const { return ntotal_; }

      private:
        IVFSpec spec_;
        std::vector<float> centroids_;
        std::vector<FlatIndex> cells_;
        long long ntotal_ = 0;
        bool trained_ = false;
        bool should_normalize() const;
        int nearest_centroid(const float *vec) const;
};

class PQFlatIndex {
      public:
        explicit PQFlatIndex(const PQFlatSpec &spec);
        ~PQFlatIndex();

        void train(std::span<const float> vecs);
        void add(std::span<const long long> ids, std::span<const float> vecs);
        std::vector<Hit> search(std::span<const float> query, int k) const;
        long long size() const { return ids_.size(); }

        size_t compressed_bytes() const { return codes_.size(); }
        size_t uncompressed_bytes() const {
                return ids_.size() * spec_.dim * sizeof(float);
        }

      private:
        PQFlatSpec spec_;
        std::unique_ptr<math::ProductQuantizer> pq_;
        std::vector<long long> ids_;
        std::vector<uint8_t> codes_;
        bool trained_ = false;
        bool should_normalize() const;
};

class IVFPQIndex {
      public:
        explicit IVFPQIndex(const IVFPQSpec &spec);
        ~IVFPQIndex();
        void train(std::span<const float> vecs);
        void add(std::span<const long long> ids, std::span<const float> vecs);
        std::vector<Hit> search(std::span<const float> query, int k) const;
        long long size() const { return ntotal_; }

        size_t compressed_bytes() const;
        size_t uncompressed_bytes() const;

      private:
        IVFPQSpec spec_;
        std::unique_ptr<math::ProductQuantizer> pq_;
        std::vector<float> centroids_;
        struct Cell {
                std::vector<long long> ids;
                std::vector<uint8_t> codes;
        };
        std::vector<Cell> cells_;

        long long ntotal_ = 0;
        bool trained_ = false;
        bool should_normalize() const;
        int nearest_centroid(const float *vec) const;
};
} // namespace spheni
