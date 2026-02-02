<p align="center">
  <img src="media/spheni.png" width="700">
</p>

<h2 align="center">Spheni</h2>

<p align="center">
  A minimal, CPU-first, in-memory vector search library in C++.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

## Index

1. [Overview](#overview)
2. [Features](#features)
3. [Status](#status)
4. [Installation](#installation)
5. [Examples](#minimal-example)
6. [Benchmarks](#benchmarks)
7. [Design Notes](#design-notes)
8. [Roadmap](#roadmap)
9. [References](#references)

## Overview

The goal of Spheni is not feature breadth, but a clean reference implementation of modern vector search primitives with transparent performance characteristics.

It currently provides:

1. Exact search via **Flat index**
2. Approximate nearest neighbor search via **IVF-Flat** (OpenMP batch-parallel)
3. Cosine similarity and L2 distance
4. Top-K queries with predictable tail latency
5. Simple, explicit API (`Engine`, `IndexSpec`, `SearchParams`)
6. Single-node, CPU-only execution

This project is early-stage and intentionally minimal.

## Features

### Indexes
1. **Flat (exact)**: Brute-force scan with Top-K heap selection.
2. **IVF-Flat (ANN)**: K-means clustering + inverted lists with `nprobe` search.

### Metrics
1. **Cosine similarity** (dot product on L2-normalized vectors)
2. **L2 distance** (implemented as negative squared L2 for ranking)

Higher score is always better.

### Core capabilities
1. Automatic ID assignment or user-provided 64-bit IDs
2. Batch add and batch search
3. Optional vector normalization (for cosine)
4. Deterministic IVF training (fixed RNG seed by default)
5. Batch-parallel query execution via OpenMP

## Status

Spheni is usable for experimentation and benchmarking, but not production-ready.

Current limitations:

- No persistence (save/load)
- No SIMD kernels
- No deletion or updates
- Limited parameter validation
- IVF uses brute-force centroid assignment

These are deliberate omissions in the current early-stage release.

## Installation

Build the static library:

```bash
bash build_spheni.sh
```

This produces:

```bash
build/libspheni.a
```

Then include headers from `spheni/` and link against `libspheni.a`.


## Minimal Example

```cpp
#include "spheni/engine.h"
#include <vector>
#include <span>
#include <iostream>

int main() {
    // 3D vectors, L2 metric, Flat index, no normalization
    spheni::IndexSpec spec(3, spheni::Metric::L2, spheni::IndexKind::Flat, false);
    spheni::Engine engine(spec);

    std::vector<float> data = {
        1,0,0,
        0,1,0,
        0,0,1
    };

    engine.add(std::span<const float>(data));

    std::vector<float> query = {0.1f, 0.9f, 0.0f};

    auto hits = engine.search(std::span<const float>(query), 1);

    std::cout << "ID: " << hits[0].id
              << " Score: " << hits[0].score << std::endl;
}
```

## IVF Usage

```cpp
spheni::IndexSpec spec(
    128,
    spheni::Metric::L2,
    spheni::IndexKind::IVF,
    false,
    /* nlist = */ 256
);

spheni::Engine engine(spec);

// add vectors (training triggers automatically once n >= nlist)
engine.add(vectors);

spheni::SearchParams params;
params.k = 10;
params.nprobe = 16;

auto hits = engine.search(query, params);
```

### IVF behavior

1. Training happens automatically once `added_vectors >= nlist`
2. Before training, searches fall back to flat scan
3. After training:

  1. Query -> nearest centroids
  2. Scan `nprobe` inverted lists
  3. Rank using Top-K heap

## Benchmarks

Tested on:

* Intel i7-8650U (4C/8T)
* GCC 13 (`-O3 -march=native`)
* 50k vectors (SIFT1M subset)
* Dimension: 128
* Metric: L2
* Single-threaded

### Flat (exact)

| Recall@10 | Mean    | p99     |
| --------- | ------- | ------- |
| 99.0%     | 5.27 ms | 5.75 ms |

### IVF-256

| nprobe | Recall | Mean     | Speedup |
| ------ | ------ | -------- | ------- |
| 16     | 97.1%  | 0.416 ms | 12.6x   |

### IVF-256 (OpenMP, batch queries)

| Recall@10 | Mean     | p99     | Throughput |
| --------- | -------- | ------- | ---------- |
| 97.11%    | 0.176 ms | 0.287 ms| 5,710 QPS  |


Full report: [`docs/benchmarks.md`](docs/benchmarks.md)

These results demonstrate the expected recall/latency trade-off of IVF on CPU.

## Design Notes

* Scores are always "higher is better"
* Cosine uses normalized vectors + dot product
* L2 uses negative squared distance
* Top-K uses a min-heap (`O(N log K)`)
* IVF training uses k-means++ initialization with deterministic seed

See:

1. [`docs/benchmarks.md`](docs/benchmarks.md)
2. [`docs/v0.1.md`](docs/v0.1.md)
3. [`docs/v0.2.md`](docs/v0.2.md)
4. [`docs/v0.3.md`](docs/v0.3.md)

## Roadmap

Short term:

- [ ] Parameter validation + explicit error handling
- [ ] Expose IVF training state
- [ ] Improve IVF memory locality
- [ ] Flat index optimizations

Longer term:

- [ ] SIMD kernels
- [x] Multithreading (just query search)
- [ ] Persistence
- [ ] Quantized storage (INT8)
- [ ] Additional ANN structures

## References

1. FAISS: [ArXiv Paper](https://arxiv.org/pdf/2401.08281)
2. [Near Neighbor Search in Large Metric Spaces](https://vldb.org/conf/1995/P574.PDF)
3. [The Binary Vector as the Basis of an Inverted Index File](https://ital.corejournals.org/index.php/ital/article/view/8961/8080)

## License

Apache 2.0