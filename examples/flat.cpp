#include "spheni.h"
#include <iostream>

int main() {
        spheni::Spec spec;
        spec.dim = 3;
        spec.metric = spheni::Metric::Cosine;
        spec.normalize = true;

        spheni::FlatIndex index(spec);

        long long ids[] = {0, 1, 2};
        float vecs[] = {
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        };
        index.add(ids, vecs);

        float query[] = {1.0f, 0.2f, 0.0f};
        auto hits = index.search(query, 3);

        for (const auto &h : hits) {
                std::cout << h.id << " " << h.score << "\n";
        }
        return 0;
}
