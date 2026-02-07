<p align="center">
  <img src="media/spheni.png" width="700">
</p>

<h2 align="center">Spheni</h2>

<p align="center">
  A tiny CPU-first, in-memory vector search library in C++ with Python bindings.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://discord.gg/XPrAs44vdH"><img src="https://img.shields.io/discord/1463893045731921934?label=Discord&logo=discord" alt="Discord"></a>
</p>

## Index

1. [Overview](#overview)
2. [Features](#features)
3. [Applications](#applications)
4. [Getting Started](#getting-started)
5. [Examples](#examples)
6. [Benchmarks](#benchmarks)
7. [Status](#status)
8. [Roadmap](#roadmap)
9. [License](#license)
10. [Disclosure](#disclosure)

## Overview

Spheni is a C++ library with Python bindings to search for points in space that are close to a given query point. The aim is to build-and-document the [architectural](docs/arch/) and [performance improvements](docs/benchmarks/benchmarks.md) over time.

## Features

1. Indexes: Flat, IVF
2. Metrics: Cosine, L2
3. Storage: F32, INT8
4. Ops: add, search, search_batch, train, save, load

Check out the [API Reference](docs/wiki/python_api.md) (Python) for full details.

## Applications

### Semantic Image Search

Spheni manages the low-level indexing and storage of CLIP-generated embeddings to enable vector similarity calculations. It compares the mathematical representation of a text query against the indexed image vectors to find the best semantic matches.

![demo gif](media/image_search.gif)

### Semantic `grep`

It retrieves relevant lines based on meaning rather than exact keywords.
It embeds text once and uses Spheni for fast, offline vector search.

![sphgrep](media/sphgrep.png)

## Getting Started

Git clone and navigate into the root directory.
Have CMake, pybind11 and OpenMP installed.

Build from the repo root:

```bash
./build_spheni.sh --python --install ./dist
```

and then link with your C++/Python projects.

Check out the [full guide](docs/wiki/building.md).

## Examples

C++:

```cpp
#include "spheni/engine.h"
#include <vector>

int main() {
    spheni::IndexSpec spec(3, spheni::Metric::L2, spheni::IndexKind::Flat, false);
    spheni::Engine engine(spec);
    std::vector<float> data = {1,0,0, 0,1,0, 0,0,1};
    engine.add(data);
    std::vector<float> query = {0.1f, 0.9f, 0.0f};
    auto hits = engine.search(query, 1);
}
```

Python:

```python
import numpy as np
import spheni

spec = spheni.IndexSpec(4, spheni.Metric.L2, spheni.IndexKind.Flat)
engine = spheni.Engine(spec)

base = np.random.rand(10, 4).astype(np.float32)
engine.add(base)

query = np.random.rand(4).astype(np.float32)
results = engine.search(query, 3)

for hit in results:
    print(f"ID: {hit.id}, Score: {hit.score}")
```

## Benchmarks

IVF achieves ~97% Recall@10 with ~12x higher throughput than brute force and stable tail latency.
INT8 quantization reduces memory by ~73% with negligible accuracy loss, and OpenMP parallelism adds ~2.4x more throughput.


Read the full [benchmark report](docs/benchmarks/benchmarks.md).

## Status

Spheni is usable for experimentation and benchmarking, but not production-ready.

Current limitations:

- No SIMD kernels
- No deletion or updates
- Limited parameter validation
- IVF uses brute-force centroid assignment

## Roadmap

Short term:

- [ ] Parameter validation + explicit error handling [partially done]
- [x] Expose IVF training state
- [ ] Improve IVF memory locality
- [ ] Flat index optimizations

Longer term:

- [ ] SIMD kernels
- [x] Multithreading (just query search)
- [x] Persistence
- [X] Quantized storage (INT8 : Scalar Quantization)
- [ ] Additional ANN structures

## References

1. FAISS: [ArXiv Paper](https://arxiv.org/pdf/2401.08281)
2. [Near Neighbor Search in Large Metric Spaces](https://vldb.org/conf/1995/P574.PDF)
3. [The Binary Vector as the Basis of an Inverted Index File](https://ital.corejournals.org/index.php/ital/article/view/8961/8080)

## License

Apache 2.0

## Disclosure

I couldn't find any solid resources showing how to structure vector search lib end-to-end, so I relied on a few community discussions (viz. this [reddit thread](https://www.reddit.com/r/Database/comments/1nyigk3/how_hard_would_it_be_to_create_a_vector_db_from/)) and read some ANN literature from the 20th century.  
I also used Claude in a "whiteboard" mode to reason about design decisions. Serialization, bindings, and exception-handling code were Codex-generated and then manually reviewed.
