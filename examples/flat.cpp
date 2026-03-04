#include "spheni.h"

#include <array>
#include <iostream>

int main() {
  spheni::Spec spec{
      .dim = 3, .metric = spheni::Metric::Cosine, .normalize = true};

  spheni::FlatIndex index(spec);

  long long ids[] = {0, 1, 42, 43};
  float vecs[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0};
  index.add(ids, vecs);

  std::array<float, 3> q{1, 0.1f, 0};

  auto hits = index.search(q, 3);

  for (auto &h : hits)
    std::cout << h.id << " " << h.score << "\n";
}
