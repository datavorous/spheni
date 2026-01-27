#pragma once

#include <vector>
#include <span>

namespace spheni {
    namespace clustering {
        class KMeans {
        public:
            KMeans(int k, int dim, int max_iters = 25);

            std::vector<float> fit(std::span<const float> vectors);
            std::vector<int> predict(std::span<const float> vectors, std::vector<float> centroids) const;

        private:
            int k_;
            int dim_;
            int max_iters_;
        };
    }
}