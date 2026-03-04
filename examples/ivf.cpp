#include "spheni.h"

#include <array>
#include <iostream>

int main() {
  spheni::IVFSpec spec{{3, spheni::Metric::Cosine, true}, 2, 2};

  spheni::IVFIndex index(spec);

  long long ids[] = {0, 1, 2, 42, 43, 44};
  float vecs[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1};
  index.train(ids, vecs);

  std::array<float, 3> q{1, 0.1f, 0};

  auto hits = index.search(q, 3);

  for (auto &h : hits)
    std::cout << h.id << " " << h.score << "\n";
}
