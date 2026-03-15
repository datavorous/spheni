<p align="center">
  <img src="media/banner.png" width="700px">
</p>

<h2 align="center">Spheni</h2>

<p align="center">
  A lightweight vector search library focused on <b>memory-efficient</b> approximate nearest neighbor search.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://discord.gg/XPrAs44vdH"><img src="https://img.shields.io/discord/1463893045731921934?label=Discord&logo=discord" alt="Discord"></a>
</p>

## Index

1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Benchmarks](#benchmarks)
5. [Roadmap](#roadmap)
6. [References](#references)

## Overview

The goal of Spheni is to be focused on memory efficiency rather than accuracy.  
It implements *inverted indexing*, *residual quantization*, and *product quantization* to reduce memory usage while supporting fast similarity search. 

With [**Cohere 1M Embeddings**](https://huggingface.co/datasets/makneeee/cohere_medium_1m), we achieve a **81x** compression (2.93GB to 35.8MB) with IVF-PQ, essentially storing `~27M vectors/GB` as compared to our previous `~349K vectors/GB` with `~70%` recall@10.

## Features

- **Indexes**: Flat, IVF, FlatPQ, IVF-PQ
- **Metrics**: Cosine similarity, L2 distance
- **Operations**: `train`, `add`, `search`

## Getting Started

### Build

Requirements:

- CMake >= 3.15
- A C++20 compiler (GCC/Clang)

Quick build

```bash
chmod +x build.sh
./build.sh
```

After building, this repository produces `build/libspheni.a`. You only need the public header (`include/spheni.h`) and the static library (`libspheni.a`) to consume Spheni in another project.

Usage 

```bash
g++ -std=c++20 -O3 main.cpp -I /path/to/spheni/include /path/to/libspheni.a -o main
```

### Example

Check out `examples/` folder for more.

```cpp
#include "spheni.h"
#include <iostream>

int main() {
        // define
        spheni::IVFSpec spec{{3, spheni::Metric::Cosine, true}, 2, 1};
        spheni::IVFIndex index(spec);

        long long ids[] = {0, 1, 2};
        float vecs[] = {
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        };
        // train
        index.train(ids, vecs);

        float query[] = {1.0f, 0.2f, 0.0f};
        // search
        auto hits = index.search(query, 3);

        for (const auto &h : hits) {
                std::cout << h.id << " " << h.score << "\n";
        }
        return 0;
}
```
## Benchmarks

Report WIP (for this `rewrite`)  
[Legacy Report](docs/legacy/benchmarks/benchmarks.md) is also available.

## Roadmap

- [ ] Implement `save`/`load` for seralized data
- [ ] Implement multithreading with OpenMP wherever applicable
- [ ] Implement OPQ (tough for me)
- [ ] SIMD vectorizations

## References

**Papers**

1. FAISS: [ArXiv Paper](https://arxiv.org/pdf/2401.08281)
2. [Near Neighbor Search in Large Metric Spaces](https://vldb.org/conf/1995/P574.PDF)
3. [The Binary Vector as the Basis of an Inverted Index File](https://ital.corejournals.org/index.php/ital/article/view/8961/8080)
4. [Product Quantization for Nearest Neighbour Search](https://inria.hal.science/inria-00514462v2/document)

**Blogs**

1. [A Bhayani - Product Quantization](https://arpitbhayani.me/blogs/product-quantization/)
2. [W Lin - Building a high recall vector database serving 1 billion embeddings from a single machine](https://blog.wilsonl.in/corenn/#product-quantization)
