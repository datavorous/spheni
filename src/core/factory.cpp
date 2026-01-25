#include "spheni/spheni.h"
#include "core/flat_index.h"
#include <stdexcept>

namespace spheni {
    std::unique_ptr<Index> make_index(const IndexSpec& spec) {
        // switch(spec.kind) { case IndexKind::Flat: continue; }
        return std::make_unique<FlatIndex>(spec);
    }
}
