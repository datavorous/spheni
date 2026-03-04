#include "spheni/engine.h"

#include <array>
#include <iostream>

int main() {
  spheni::IndexSpec spec{.dim = 3,
                         .normalize = true,
                         .nlist = 2,
                         .metric = spheni::Metric::Cosine,
                         .kind = spheni::IndexKind::IVF};

  spheni::Engine index(spec);

  float base[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  index.add(base);

  long long ids[] = {42, 43, 44};
  float more[] = {1, 1, 0, 1, 0, 1, 0, 1, 1};

  index.add(ids, more);

  index.train();

  std::array<float, 3> q{1, 0.1f, 0};

  auto hits = index.search(q, {.k = 3, .nprobe = 2});

  for (auto &h : hits)
    std::cout << h.id << " " << h.score << "\n";

  float batch[] = {1, 0, 0, 0, 0, 1};

  auto res = index.search_batch(batch, {.k = 2, .nprobe = 2});

  std::cout << "first batch id: " << res[0][0].id << "\n";
}
