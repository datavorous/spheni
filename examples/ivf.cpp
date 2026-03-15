#include "spheni.h"
#include <iostream>

int main() {
        spheni::IVFSpec spec{{3, spheni::Metric::Cosine, true}, 2, 1};
        spheni::IVFIndex index(spec);

        long long ids[] = {0, 1, 2};
        float vecs[] = {
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        };
        index.train(ids, vecs);

        float query[] = {1.0f, 0.2f, 0.0f};
        auto hits = index.search(query, 3);

        for (const auto &h : hits) {
                std::cout << h.id << " " << h.score << "\n";
        }
        return 0;
}
