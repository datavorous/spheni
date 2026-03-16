# Spheni API Reference

This document describes the public C++ API exposed by `include/spheni.h`.

> [!IMPORTANT]
> This document was generated with the aid of Github Copilot. 

## Header and Namespace

Include the library with:

```cpp
#include "spheni.h"
```

All public types live in the `spheni` namespace.

The API uses `std::span` for vector inputs. In practice this means:

- `ids` is a contiguous sequence of `long long` identifiers.
- `vecs` is a contiguous sequence of `float` values laid out row-major as `n * dim`.
- `query` is a single contiguous vector of length `dim`.

## Core Types

### `enum class Metric`

Controls how vectors are scored during search.

- `Metric::Cosine`: similarity is based on dot product. Higher scores are better.
- `Metric::L2`: similarity is exposed as negative squared L2 distance. Values closer to zero are better, and higher scores are still better because the distance is negated.

### `struct Spec`

Base configuration shared by the non-partitioned indexes.

```cpp
struct Spec {
        int dim;
        Metric metric = Metric::Cosine;
        bool normalize = true;
};
```

Fields:

- `dim`: dimensionality of every vector.
- `metric`: scoring mode used during search.
- `normalize`: whether vectors and queries should be normalized before indexing and search.

Normalization notes:

- For cosine search, enabling normalization is usually the intended configuration.
- In the current implementation, `FlatIndex`, `PQFlatIndex`, and `IVFPQIndex` normalize whenever `normalize == true`, regardless of metric.
- `IVFIndex` only normalizes when `metric == Metric::Cosine` and `normalize == true`.

### `struct IVFSpec : Spec`

Configuration for inverted-file search.

```cpp
struct IVFSpec : Spec {
        int nlist = 0;
        int nprobe = 1;
};
```

Fields:

- `nlist`: number of coarse clusters.
- `nprobe`: number of clusters searched per query.

### `struct PQFlatSpec : Spec`

Configuration for flat product-quantized search.

```cpp
struct PQFlatSpec : Spec {
        int M = 8;
        int ksub = 256;
};
```

Fields:

- `M`: number of subquantizers.
- `ksub`: number of centroids per subspace.

Constraints:

- `dim` must be divisible by `M`.
- `ksub` must be `<= 256` because encoded codes are stored as bytes.

### `struct IVFPQSpec : Spec`

Configuration for IVF with residual product quantization.

```cpp
struct IVFPQSpec : Spec {
        int nlist = 256;
        int nprobe = 8;
        int M = 8;
        int ksub = 256;
};
```

Fields combine IVF and PQ settings:

- `nlist`: number of coarse clusters.
- `nprobe`: number of clusters searched per query.
- `M`: number of PQ subquantizers.
- `ksub`: number of centroids per subspace.

### `struct Hit`

Single search result.

```cpp
struct Hit {
        long long id;
        float score;
};
```

Search results are returned as `std::vector<Hit>` sorted from best to worst score.

## Index Types

### `class FlatIndex`

Exact brute-force search over stored vectors.

```cpp
explicit FlatIndex(const Spec &spec);
void add(std::span<const long long> ids, std::span<const float> vecs);
std::vector<Hit> search(std::span<const float> query, int k) const;
long long size() const;
```

Behavior:

- No training step is required.
- `add()` appends ids and vectors to the existing index.
- If normalization is enabled, vectors are normalized on insert and queries are normalized at search time.
- For `Metric::Cosine`, scores are dot products.
- For `Metric::L2`, scores are `-l2_squared(query, vector)`.

Use when:

- You want exact results.
- Dataset size is small enough that full scan latency and memory cost are acceptable.

Example:

```cpp
spheni::Spec spec;
spec.dim = 3;
spec.metric = spheni::Metric::Cosine;
spec.normalize = true;

spheni::FlatIndex index(spec);
index.add(ids, vecs);
auto hits = index.search(query, 10);
```

### `class IVFIndex`

Approximate search using a coarse inverted file and exact search within selected cells.

```cpp
explicit IVFIndex(const IVFSpec &spec);
void train(std::span<const long long> ids, std::span<const float> vectors);
void add(std::span<const long long> ids, std::span<const float> vecs);
std::vector<Hit> search(std::span<const float> query, int k) const;
long long size() const;
```

Lifecycle:

1. Construct with an `IVFSpec`.
2. Call `train(...)` once on training data.
3. Optionally call `add(...)` to insert more vectors after training.
4. Call `search(...)`.

Behavior:

- `train()` performs coarse k-means over the provided vectors, assigns each vector to its nearest centroid, and inserts the provided `(id, vector)` pairs into the corresponding cells.
- `train()` is not just model fitting; it also populates the index with the training vectors.
- `add()` requires the index to be trained first.
- `search()` ranks centroids by L2 distance to the query, probes the best `min(nprobe, nlist)` cells, and merges their top results.
- Each cell uses `FlatIndex` internally.

Operational notes:

- With cosine search, normalization is applied only when both `metric == Metric::Cosine` and `normalize == true`.
- With L2 search, IVF currently does not normalize even if `normalize` is set.
- `size()` counts vectors inserted during both `train()` and later `add()` calls.

Use when:

- You want a speed/recall trade-off.
- You can pay a training cost up front.

Example:

```cpp
spheni::IVFSpec spec{{128, spheni::Metric::Cosine, true}, 256, 8};
spheni::IVFIndex index(spec);

index.train(train_ids, train_vecs);
index.add(extra_ids, extra_vecs);
auto hits = index.search(query, 10);
```

### `class PQFlatIndex`

Flat search over product-quantized codes.

```cpp
explicit PQFlatIndex(const PQFlatSpec &spec);
~PQFlatIndex();

void train(std::span<const float> vecs);
void add(std::span<const long long> ids, std::span<const float> vecs);
std::vector<Hit> search(std::span<const float> query, int k) const;
long long size() const;

size_t compressed_bytes() const;
size_t uncompressed_bytes() const;
```

Lifecycle:

1. Construct with a `PQFlatSpec`.
2. Call `train(...)` on representative vectors.
3. Call `add(...)`.
4. Call `search(...)`.

Behavior:

- `train()` learns PQ codebooks for `M` subspaces.
- `add()` encodes each inserted vector into `M` bytes of PQ code and stores the ids separately.
- `search()` uses asymmetric distance computation: the query remains in float form, database vectors are compared through their PQ codes.
- Returned scores are negative approximate distances, so higher is better.

Storage helpers:

- `compressed_bytes()`: number of bytes used by PQ codes only.
- `uncompressed_bytes()`: estimated raw float storage as `size() * dim * sizeof(float)`.

Operational notes:

- `add()` requires prior training.
- If `normalize == true`, training vectors, inserted vectors, and queries are normalized before PQ operations.
- The current distance table uses L2-style subspace distances even when `metric == Metric::Cosine`, so treat cosine mode here as normalized approximate search rather than exact cosine scoring.

Use when:

- Memory footprint matters more than exact recall.
- You want a compact single-list index without IVF partitioning.

Example:

```cpp
spheni::PQFlatSpec spec;
spec.dim = 128;
spec.M = 16;
spec.ksub = 256;
spec.metric = spheni::Metric::Cosine;
spec.normalize = true;

spheni::PQFlatIndex index(spec);
index.train(train_vecs);
index.add(ids, vecs);
auto hits = index.search(query, 10);
```

### `class IVFPQIndex`

Approximate search using coarse inverted lists plus residual product quantization inside each list.

```cpp
explicit IVFPQIndex(const IVFPQSpec &spec);
~IVFPQIndex();

void train(std::span<const float> vecs);
void add(std::span<const long long> ids, std::span<const float> vecs);
std::vector<Hit> search(std::span<const float> query, int k) const;
long long size() const;

size_t compressed_bytes() const;
size_t uncompressed_bytes() const;
```

Lifecycle:

1. Construct with an `IVFPQSpec`.
2. Call `train(...)` on representative vectors.
3. Call `add(...)`.
4. Call `search(...)`.

Behavior:

- `train()` learns IVF centroids using k-means, computes residuals relative to the assigned centroid, and trains the PQ codebooks on those residuals.
- Unlike `IVFIndex`, `train()` does not insert ids or vectors into the searchable structure.
- `add()` assigns each vector to its nearest centroid, computes its residual, PQ-encodes that residual, and stores the code in the corresponding cell.
- `search()` probes the nearest `min(nprobe, nlist)` cells, computes a query residual per probed cell, and scores stored codes with asymmetric distance computation.
- Returned scores are negative approximate distances, so higher is better.

Storage helpers:

- `compressed_bytes()`: total bytes used by PQ codes across all cells.
- `uncompressed_bytes()`: estimated raw float storage as `size() * dim * sizeof(float)`.

Operational notes:

- `add()` requires prior training.
- If `normalize == true`, training vectors, inserted vectors, and queries are normalized before centroid assignment and residual computation.
- As with `PQFlatIndex`, approximate scoring is based on PQ distance tables rather than an exact cosine dot product path.

Use when:

- You need the strongest compression in the current API.
- You are willing to trade recall for memory efficiency and speed.

Example:

```cpp
spheni::IVFPQSpec spec;
spec.dim = 128;
spec.metric = spheni::Metric::Cosine;
spec.normalize = true;
spec.nlist = 256;
spec.nprobe = 8;
spec.M = 16;
spec.ksub = 256;

spheni::IVFPQIndex index(spec);
index.train(train_vecs);
index.add(ids, vecs);
auto hits = index.search(query, 10);
```

## Input Shape Expectations

The API does not perform explicit argument validation on shape compatibility. Callers should ensure:

- `vecs.size()` is a multiple of `dim`.
- `ids.size()` matches the number of vectors represented by `vecs.size() / dim`.
- `query.size() == dim`.
- IVF and PQ parameters are valid for the selected dimensionality.

Several internal checks currently rely on `assert`, so invalid input may abort in debug builds and may produce undefined behavior in release builds.

## Choosing an Index

Use `FlatIndex` when exactness matters most.

Use `IVFIndex` when you want an ANN index with exact scoring inside selected clusters.

Use `PQFlatIndex` when memory reduction is the main goal and you can accept approximate scoring.

Use `IVFPQIndex` when you need the best compression and scalable approximate search in the current API.
