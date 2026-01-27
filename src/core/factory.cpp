#include "spheni/spheni.h"
#include "core/flat_index.h"
#include "core/ivf_index.h"
#include <stdexcept>

namespace spheni {
    std::unique_ptr<Index> make_index(const IndexSpec& spec) {
        // switch(spec.kind) { case IndexKind::Flat: continue; }
        // return std::make_unique<FlatIndex>(spec);

        switch(spec.kind) {
            case IndexKind::Flat:
                return std::make_unique<FlatIndex>(spec);
            case IndexKind::IVF:
                return std::make_unique<IVFIndex>(spec);
            default:
                throw std::invalid_argument("Unknown IndexKind");
        }
    }
}
