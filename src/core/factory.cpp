#include "indexes/flat/flat_index.h"
#include "indexes/ivf/ivf_index.h"
#include "spheni/spheni.h"
#include <memory>
#include <stdexcept>

namespace spheni {
std::unique_ptr<Index> make_index(const IndexSpec &spec) {
  switch (spec.kind) {
  case IndexKind::Flat:
    return std::make_unique<FlatIndex>(spec);
  case IndexKind::IVF:
    return std::make_unique<IVFIndex>(spec);
  }
  throw std::runtime_error("unknown IndexKind");
}
} // namespace spheni
