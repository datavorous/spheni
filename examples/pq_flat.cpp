#include "spheni.h"
#include <cstdio>
#include <numeric>
#include <span>
#include <vector>

int main() {
        const int dim = 128;
        const int n = 10000;
        const int M = 16;

        std::vector<float> vecs(n * dim);
        for (auto &x : vecs)
                x = (float)rand() / RAND_MAX;

        std::vector<long long> ids(n);
        std::iota(ids.begin(), ids.end(), 0);

        spheni::PQFlatSpec spec;
        spec.dim = dim;
        spec.M = M;
        spec.ksub = 256;
        spec.metric = spheni::Metric::Cosine;
        spec.normalize = true;

        spheni::PQFlatIndex pq(spec);
        pq.train(std::span<const float>(vecs));
        pq.add(ids, std::span<const float>(vecs));

        size_t original = n * dim * sizeof(float);
        size_t compressed = pq.compressed_bytes();

        printf("Vectors: %d x dim %d\n", n, dim);
        printf("Original: %zu KB\n", original / 1024);
        printf("Compressed: %zu KB\n", compressed / 1024);
        printf("Ratio: %.1fx\n", (float)original / compressed);

        return 0;
}