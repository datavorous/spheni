#include "math/topk.h"
#include <algorithm>

namespace spheni::math {

TopK::TopK(int k) : k_(k) { heap_ = decltype(heap_)(); }

void TopK::push(long long id, float score) {
  if (heap_.size() < static_cast<std::size_t>(k_)) {
    heap_.emplace(id, score);
  } else if (score > heap_.top().score) {
    heap_.pop();
    heap_.emplace(id, score);
  }
}

std::vector<SearchHit> TopK::sorted_results() {
  std::vector<SearchHit> results;

  while (!heap_.empty()) {
    results.push_back(heap_.top());
    heap_.pop();
  }

  std::reverse(results.begin(), results.end());
  return results;
}

} // namespace spheni::math
