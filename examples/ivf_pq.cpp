#include "spheni.h"
#include <cstdio>
#include <numeric>
#include <vector>

int main() {
        const int dim = 128;
        const int n = 10000;

        std::vector<float> vecs(n * dim);
        for (auto &x : vecs)
                x = (float)rand() / RAND_MAX;

        std::vector<long long> ids(n);
        std::iota(ids.begin(), ids.end(), 0);

        spheni::IVFPQSpec spec;
        spec.dim = dim;
        spec.metric = spheni::Metric::Cosine;
        spec.normalize = true;
        spec.nlist = 100;
        spec.nprobe = 10;
        spec.M = 16;
        spec.ksub = 256;

        spheni::IVFPQIndex index(spec);

        printf("Training on %d vectors\n", n);
        index.train(vecs);

        printf("Adding vectors to index\n");
        index.add(ids, vecs);

        size_t original = n * dim * sizeof(float);
        size_t compressed = index.compressed_bytes();

        printf("\n> IVFPQ Metrics\n");
        printf("Original: %zu KB\n", original / 1024);
        printf("Compressed: %zu KB\n", compressed / 1024);
        printf("Ratio: %.1fx\n", (float)original / compressed);

        float query[128];
        for (int i = 0; i < 128; ++i)
                query[i] = (float)rand() / RAND_MAX;

        auto hits = index.search(query, 5);
        printf("\nTop 5 Results:\n");
        for (const auto &h : hits) {
                printf("ID: %lld | Score: %.4f\n", h.id, h.score);
        }

        return 0;
}