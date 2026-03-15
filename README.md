<p align="center">
  <img src="media/banner.png" width="700px">
</p>

<h2 align="center">Spheni</h2>

<p align="center">
  A lightweight vector search library focused on memory-efficient approximate nearest neighbor search.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://discord.gg/XPrAs44vdH"><img src="https://img.shields.io/discord/1463893045731921934?label=Discord&logo=discord" alt="Discord"></a>
</p>

## Overview

Spheni is a C++ library for approximate nearest neighbor search over high-dimensional vectors.

It implements *inverted indexing*, *residual quantization*, and *product quantization* to reduce memory usage while supporting fast similarity search.

## Features

- **Indexes**: Flat, IVF, FlatPQ, IVF-PQ
- **Metrics**: Cosine similarity, L2 distance
- **Operations**: `train`, `add`, `search`

## Build

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

## Example

```cpp
#include "spheni.h"
#include <iostream>

int main() {
        spheni::IVFSpec spec{{3, spheni::Metric::Cosine, true}, 2, 1};
        spheni::IVFIndex index(spec);

        long long ids[] = {0, 1, 2};
        float vecs[] = {
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        };
        index.train(ids, vecs);

        float query[] = {1.0f, 0.2f, 0.0f};
        auto hits = index.search(query, 3);

        for (const auto &h : hits) {
                std::cout << h.id << " " << h.score << "\n";
        }
        return 0;
}
```
