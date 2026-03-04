#pragma once

#include "spheni.h"
#include <queue>
#include <vector>

namespace spheni::math {

class TopK {
public:
  explicit TopK(int k) : k_(k) {}

  void push(long long id, float score) {
    if (heap_.size() < static_cast<std::size_t>(k_)) {
      heap_.emplace(id, score);
    } else if (score > heap_.top().score) {
      heap_.pop();
      heap_.emplace(id, score);
    }
  }

  std::vector<Hit> take_sorted() {
    std::vector<Hit> results(heap_.size());
    for (auto it = results.rbegin(); it != results.rend(); ++it) {
      *it = heap_.top();
      heap_.pop();
    }
    return results;
  }

private:
  int k_;
  struct WorseScore {
    bool operator()(const Hit &a, const Hit &b) const {
      return a.score > b.score;
    }
  };
  std::priority_queue<Hit, std::vector<Hit>, WorseScore> heap_;
};

} // namespace spheni::math
