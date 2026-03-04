#pragma once

#include "spheni/spheni.h"
#include <queue>
#include <vector>

namespace spheni::math {

class TopK {
public:
  explicit TopK(int k);

  void push(long long id, float score);

  std::vector<SearchHit> take_sorted();

private:
  int k_;
  struct WorseScore {
    bool operator()(const SearchHit &a, const SearchHit &b) const {
      return a.score > b.score;
    }
  };
  std::priority_queue<SearchHit, std::vector<SearchHit>, WorseScore> heap_;
};

} // namespace spheni::math
