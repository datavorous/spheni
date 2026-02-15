#include "indexes/flat/flat_index.h"
#include "indexes/ivf/ivf_index.h"
#include "spheni/spheni.h"
#include <stdexcept>

namespace spheni {
std::unique_ptr<Index> make_index(const IndexSpec &spec) {
  // switch(spec.kind) { case IndexKind::Flat: continue; }
  // return std::make_unique<FlatIndex>(spec);

  if (spec.metric == Metric::Haversine) {
    if (spec.dim != 2) {
      throw std::invalid_argument("Haversine metric requires dim=2");
    }
    if (spec.kind == IndexKind::IVF) {
      throw std::invalid_argument(
          "Haversine metric does not support IVF index");
    }
    if (spec.normalize) {
      throw std::invalid_argument(
          "Haversine metric does not support normalization");
    }
    if (spec.storage == StorageType::INT8) {
      throw std::invalid_argument(
          "Haversine metric does not support INT8 storage");
    }
  }

  switch (spec.kind) {
  case IndexKind::Flat:
    return std::make_unique<FlatIndex>(spec);
  case IndexKind::IVF:
    return std::make_unique<IVFIndex>(spec);
  default:
    throw std::invalid_argument("Unknown IndexKind");
  }
}
} // namespace spheni
