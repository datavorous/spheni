#include "math/topk.h"

namespace spheni::math {

TopK::TopK(int k) : k_(k) {}

void TopK::push(long long id, float score) {
  if (k_ <= 0) {
    return;
  }
  if (heap_.size() < static_cast<std::size_t>(k_)) {
    heap_.emplace(id, score);
  } else if (score > heap_.top().score) {
    heap_.pop();
    heap_.emplace(id, score);
  }
}

std::vector<SearchHit> TopK::take_sorted() {
  std::vector<SearchHit> results(heap_.size());
  for (auto it = results.rbegin(); it != results.rend(); ++it) {
    *it = heap_.top();
    heap_.pop();
  }
  return results;
}

} // namespace spheni::math
