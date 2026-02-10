# C++ API Reference

This document describes the public C++ API exposed by Spheni.

## Header

```cpp
#include "spheni/engine.h"
```

The public types are declared in:

- `include/spheni/spheni.h`
- `include/spheni/engine.h`

## Enums

### `spheni::Metric`

- `Metric::Cosine`
- `Metric::L2`

### `spheni::IndexKind`

- `IndexKind::Flat`
- `IndexKind::IVF`

### `spheni::StorageType`

- `StorageType::F32`
- `StorageType::INT8`

## Structs

### `IndexSpec`

Fields:

- `dim` (`int`): vector dimension
- `metric` (`Metric`)
- `normalize` (`bool`): if true and metric is cosine, vectors/queries are normalized at index time/query time where relevant
- `kind` (`IndexKind`)
- `storage` (`StorageType`)
- `nlist` (`int`): IVF cluster count

Constructors:

```cpp
IndexSpec(int dim, Metric metric, IndexKind kind, bool normalize = true);
IndexSpec(int dim, Metric metric, IndexKind kind, StorageType storage, bool normalize = true);
IndexSpec(int dim, Metric metric, IndexKind kind, int nlist, bool normalize = true);
IndexSpec(int dim, Metric metric, IndexKind kind, int nlist, StorageType storage, bool normalize = true);
```

### `SearchParams`

Fields:

- `k` (`int`)
- `nprobe` (`int`): IVF probe count

Constructors:

```cpp
SearchParams(int k); // nprobe defaults to 1
SearchParams(int k, int nprobe);
```

### `SearchHit`

Fields:

- `id` (`long long`)
- `score` (`float`): higher is better (cosine or negative L2)

## Class

### `Engine`

Constructor:

```cpp
Engine(const IndexSpec& spec);
```

Methods:

```cpp
void add(std::span<const float> vectors);
void add(std::span<const long long> ids, std::span<const float> vectors);

std::vector<SearchHit> search(std::span<const float> query, int k) const;
std::vector<SearchHit> search(std::span<const float> query, int k, int nprobe) const;

std::vector<std::vector<SearchHit>> search_batch(std::span<const float> queries, int k) const;
std::vector<std::vector<SearchHit>> search_batch(std::span<const float> queries, int k, int nprobe) const;

void train();

long long size() const;
int dim() const;

void save(const std::string& path) const;
static Engine load(const std::string& path);
```

Behavior notes:

- `add(vectors)` auto-assigns IDs starting from `0`.
- IVF requires explicit `train()` before `search()`.
- `search_batch` expects a flat row-major buffer of shape `(n, dim)`.

## Data Layout Expectations

Because the C++ API takes `std::span`, pass contiguous row-major buffers:

- Base vectors: `n * dim` floats
- Query: `dim` floats
- Query batch: `n * dim` floats
- IDs: `n` signed 64-bit values (`long long`)

## Minimal Example

```cpp
#include "spheni/engine.h"
#include <iostream>
#include <vector>

int main() {
    spheni::IndexSpec spec(
        4,
        spheni::Metric::L2,
        spheni::IndexKind::Flat,
        spheni::StorageType::F32,
        false
    );

    spheni::Engine engine(spec);

    std::vector<float> base = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    engine.add(base);

    std::vector<float> query = {0.2f, 0.7f, 0.1f, 0.0f};
    auto hits = engine.search(query, 2);

    for (const auto& h : hits) {
        std::cout << "id=" << h.id << " score=" << h.score << "\n";
    }
}
```

## IVF Example

```cpp
#include "spheni/engine.h"
#include <vector>

int main() {
    spheni::IndexSpec spec(
        4,
        spheni::Metric::L2,
        spheni::IndexKind::IVF,
        8, // nlist
        spheni::StorageType::F32,
        false
    );

    spheni::Engine engine(spec);
    std::vector<float> base(8 * 4, 0.1f);
    engine.add(base);
    engine.train();

    std::vector<float> query = {0.1f, 0.2f, 0.3f, 0.4f};
    auto hits = engine.search(query, 3, 2); // nprobe=2
}
```
