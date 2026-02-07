#include "spheni/engine.h"
#include "core/flat_index.h"
#include "core/ivf_index.h"
#include "core/serialize.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace spheni {

Engine::Engine(const IndexSpec& spec): index_(make_index(spec)), next_id_(0) {}

void Engine::add(std::span<const float> vectors) {
    int d = index_->dim();
    long long n = vectors.size() / d;
    std::vector<long long> ids(n);

    for (long long i = 0; i < n; i++) {
        ids[i] = next_id_++;
    }

    index_->add(ids, vectors);
}

void Engine::add(std::span<const long long> ids, std::span<const float> vectors) {
    index_->add(ids, vectors);

    auto max_it = std::max_element(ids.begin(), ids.end());

    if (max_it != ids.end()) {
        long long next = *max_it + 1;
        if (next > next_id_) {
            next_id_ = next;
        }
    }
}

std::vector<SearchHit> Engine::search(std::span<const float> query, int k) const {
    SearchParams params(k);
    return index_->search(query, params);
}

std::vector<SearchHit> Engine::search(std::span<const float> query, int k, int nprobe) const {
    SearchParams params(k, nprobe);
    return index_->search(query, params);
}

std::vector<std::vector<SearchHit>> Engine::search_batch(std::span<const float> queries, int k) const {
    int d = index_->dim();
    long long n = queries.size() / d;
    std::vector<std::vector<SearchHit>> results;

    for (long long i = 0; i < n; ++i) {
        std::span<const float> query(queries.data() + i * d, d);
        results.push_back(search(query, k, 1));
    }

    return results;
}

void Engine::train() {
    IVFIndex* ivf = dynamic_cast<IVFIndex*>(index_.get());
    if (!ivf) {
        throw std::runtime_error("Engine::train: only IVF index supports training.");
    }
    ivf->train();
}

long long Engine::size() const { return index_->size(); }
int Engine::dim() const { return index_->dim(); }

void Engine::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Engine::save: failed to open file.");
    }

    const IndexSpec* spec = nullptr;
    const FlatIndex* flat = dynamic_cast<const FlatIndex*>(index_.get());
    const IVFIndex* ivf = dynamic_cast<const IVFIndex*>(index_.get());

    if (flat) {
        spec = &flat->spec();
    } else if (ivf) {
        spec = &ivf->spec();
    } else {
        throw std::runtime_error("Engine::save: unsupported index type.");
    }
    if (flat && spec->kind != IndexKind::Flat) {
        throw std::runtime_error("Engine::save: flat index kind mismatch.");
    }
    if (ivf && spec->kind != IndexKind::IVF) {
        throw std::runtime_error("Engine::save: IVF index kind mismatch.");
    }

    io::write_pod(out, static_cast<std::int32_t>(spec->dim));
    io::write_pod(out, static_cast<std::int32_t>(spec->metric));
    io::write_pod(out, static_cast<std::int32_t>(spec->kind));
    io::write_pod(out, static_cast<std::int32_t>(spec->storage));
    io::write_bool(out, spec->normalize);

    io::write_pod(out, static_cast<std::int32_t>(spec->nlist));
    io::write_pod(out, next_id_);

    // index state is written after the spec so load() can rebuild the exact in-memory layout.
    if (flat) {
        flat->save_state(out);
        return;
    }
    if (ivf) {
        ivf->save_state(out);
        return;
    }
}

Engine Engine::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Engine::load: failed to open file.");
    }

    std::int32_t dim = io::read_pod<std::int32_t>(in);
    std::int32_t metric_raw = io::read_pod<std::int32_t>(in);
    std::int32_t kind_raw = io::read_pod<std::int32_t>(in);
    std::int32_t storage_raw = io::read_pod<std::int32_t>(in);
    bool normalize = io::read_bool(in);
    std::int32_t nlist = io::read_pod<std::int32_t>(in);
    long long next_id = io::read_pod<long long>(in);

    if (dim <= 0) {
        throw std::runtime_error("Engine::load: invalid dimension.");
    }
    if (metric_raw < 0 || metric_raw > 1) {
        throw std::runtime_error("Engine::load: invalid metric.");
    }
    if (kind_raw < 0 || kind_raw > 1) {
        throw std::runtime_error("Engine::load: invalid index kind.");
    }
    if (storage_raw < 0 || storage_raw > 1) {
        throw std::runtime_error("Engine::load: invalid storage type.");
    }
    if (kind_raw == static_cast<std::int32_t>(IndexKind::IVF) && nlist <= 0) {
        throw std::runtime_error("Engine::load: invalid IVF nlist.");
    }

    IndexSpec spec(dim,
                   static_cast<Metric>(metric_raw),
                   static_cast<IndexKind>(kind_raw),
                   nlist,
                   static_cast<StorageType>(storage_raw),
                   normalize);

    Engine engine(spec);
    engine.next_id_ = next_id;

    if (spec.kind == IndexKind::Flat) {
        FlatIndex* flat = dynamic_cast<FlatIndex*>(engine.index_.get());
        if (!flat) {
            throw std::runtime_error("Engine::load: flat index cast failed.");
        }
        flat->load_state(in);
        return engine;
    }

    if (spec.kind == IndexKind::IVF) {
        IVFIndex* ivf = dynamic_cast<IVFIndex*>(engine.index_.get());
        if (!ivf) {
            throw std::runtime_error("Engine::load: IVF index cast failed.");
        }
        ivf->load_state(in);
        return engine;
    }

    throw std::runtime_error("Engine::load: unsupported index kind.");
}

}
