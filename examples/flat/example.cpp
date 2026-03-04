#include "spheni/engine.h"

#include <array>
#include <iostream>

int main() {
  spheni::IndexSpec spec{.dim = 3,
                         .normalize = true,
                         .metric = spheni::Metric::Cosine,
                         .kind = spheni::IndexKind::Flat};

  spheni::Engine index(spec);

  float base[] = {1, 0, 0, 0, 1, 0};

  index.add(base);

  long long ids[] = {42, 43};
  float more[] = {0, 0, 1, 1, 1, 0};

  index.add(ids, more);

  std::array<float, 3> q{1, 0.1f, 0};

  auto hits = index.search(q, 3);

  for (auto &h : hits)
    std::cout << h.id << " " << h.score << "\n";

  float batch[] = {1, 0, 0, 0, 0, 1};

  auto res = index.search_batch(batch, {.k = 2});

  std::cout << "first batch id: " << res[0][0].id << "\n";
}
