#pragma once

#include "kmeans.h"
#include "math.h"
#include <cassert>
#include <cstdint>
#include <math.h>
#include <span>
#include <vector>

namespace spheni::math {
class ProductQuantizer {
      public:
        ProductQuantizer(int dim, int M, int ksub = 256)
            : dim_(dim), M_(M), ksub_(ksub), dsub_(dim / M) {
                assert(dim % M == 0);
                assert(ksub <= 256);
        }
        int M() const { return M_; }
        int ksub() const { return ksub_; }
        int dsub() const { return dsub_; }
        int dim() const { return dim_; }
        bool trained() const { return trained_; }

        float approx_distance(const std::vector<float> &table,
                              const uint8_t *code) const {
                float d = 0;
                for (int m = 0; m < M_; m++)
                        d += table[m * ksub_ + code[m]];
                return d;
        }
        void train(std::span<const float> vecs) {
                const int n = vecs.size() / dim_;
                codebooks_.resize(M_ * ksub_ * dsub_);
                std::vector<float> sub(n * dsub_);

                for (int m = 0; m < M_; m++) {
                        for (int i = 0; i < n; i++) {
                                const float *src =
                                    vecs.data() + i * dim_ + m * dsub_;
                                std::copy(src, src + dsub_,
                                          sub.data() + i * dsub_);
                        }

                        clustering::KMeans km(ksub_, dsub_);
                        auto cb = km.fit(
                            std::span<const float>(sub.data(), sub.size()));
                        float *dst = codebooks_.data() + m * ksub_ * dsub_;
                        std::copy(cb.begin(), cb.end(), dst);
                }
                trained_ = true;
        }

        std::vector<uint8_t> encode_one(const float *vec) const {
                assert(trained_);
                std::vector<uint8_t> code(M_);
                for (int m = 0; m < M_; m++) {
                        const float *sub = vec + m * dsub_;
                        const float *cb = codebooks_.data() + m * ksub_ * dsub_;
                        code[m] = nearest_centroid(sub, cb);
                }
                return code;
        }

        std::vector<uint8_t> encode(std::span<const float> vecs) const {
                const int n = vecs.size() / dim_;
                std::vector<uint8_t> codes(n * M_);
                for (int i = 0; i < n; i++)
                        encode_one_into(vecs.data() + i * dim_,
                                        codes.data() + i * M_);
                return codes;
        }

        std::vector<float> precompute_table(const float *query) const {
                assert(trained_);
                std::vector<float> table(M_ * ksub_);
                for (int m = 0; m < M_; m++) {
                        const float *qsub = query + m * dsub_;
                        const float *cb = codebooks_.data() + m * ksub_ * dsub_;
                        float *row = table.data() + m * ksub_;
                        for (int k = 0; k < ksub_; k++)
                                // row[k] = kernels::dot(qsub, cb + k * dsub_,
                                // dsub_);
                                row[k] = kernels::l2_squared(
                                    qsub, cb + k * dsub_, dsub_);
                }
                return table;
        }

      private:
        int dim_, M_, ksub_, dsub_;
        std::vector<float> codebooks_;
        bool trained_ = false;

        uint8_t nearest_centroid(const float *sub, const float *cb) const {
                float best = std::numeric_limits<float>::max();
                uint8_t index = 0;
                for (int k = 0; k < ksub_; k++) {
                        float d =
                            kernels::l2_squared(sub, cb + k * dsub_, dsub_);
                        if (d < best) {
                                best = d;
                                // https://en.cppreference.com/w/cpp/language/static_cast.html
                                index = static_cast<uint8_t>(k);
                        }
                }
                return index;
        }
        void encode_one_into(const float *vec, uint8_t *code) const {
                for (int m = 0; m < M_; m++) {
                        const float *sub = vec + m * dsub_;
                        const float *cb = codebooks_.data() + m * ksub_ * dsub_;
                        code[m] = nearest_centroid(sub, cb);
                }
        }
};
} // namespace spheni::math