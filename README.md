<p align="center">
    <img src="media/spheni.png" size="100">
</p>

<p align="center">
    <b>Spheni - An in-memory Vector Search Engine</b>.<br/>
    [WORK IN PROGRESS]
</p>

<p align="center">
  <a href="https://github.com/datavorous/spheni/blob/master/LICENSE" target="_blank">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
</p>

> [!CAUTION]
> This is still a work-in-progress, and not ready for use.
> Aim is to implement ANN algorithms and quanitzation techniques.

## Minimal Example

Build a static library (`libspheni.a`) using `bash build_spheni.sh` and use it.

```cpp
#include "spheni/engine.h"
#include <iostream>
#include <vector>
#include <span>

int main() {
    // index specs: 3 dimensions, L2 distance, Flat index, normalization
    spheni::IndexSpec spec(3, spheni::Metric::L2, spheni::IndexKind::Flat, false);
    spheni::Engine engine(spec);

    // some dummy data (3 vectors of 3 dimensions)
    std::vector<float> data = {1.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 1.0f};

    engine.add(std::span<const float>(data));

    // search for a vector close to the second entry [0, 1, 0]
    std::vector<float> query = {0.1f, 0.9f, 0.0f};
    auto results = engine.search(std::span<const float>(query), 1);

    std::cout << "Top Hit ID: " << results[0].id << " Score: " << results[0].score << std::endl;
    // output: Top Hit ID: 1 Score: -0.02

    return 0;
}
```

## Todo

- [ ] Flat Index Optimization
    - [ ] reducing the latency using memory locality, SIMD acceleration, and threading
    - [ ] replace heapTopK, precompute L2 norms, write SIMD kernels
- [ ] IVF-Flat (ANN) implementation
    - [ ] kmeans training, and nprobe search
- [ ] memory reduction (scalar INT8 first). benchmark: FP32 Flat, INT8 Flat, IVF + INT8
- [ ] add save and load

additionally benchmark and plot recall vs latency in each step.

## Benchmarks

### SIFT1M Benchmark

### Flat Index

| Category        | Metric          | Value        |
|-----------------|-----------------|-------------|
| Dataset         | Base vectors    | 1,000,000   |
|                 | Query vectors   | 10,000      |
|                 | Dimension       | 128         |
| Loading         | Load time       | ~420 ms      |
| Indexing        | Build time      | ~260 ms      |
|                 | Indexed vectors| 1,000,000   |
| Configuration  | k               | 10          |
|                 | Max queries     | 100         |
|                 | Warmup queries | 5           |
| Accuracy        | Recall@10       | 99.90%      |
| Latency (ms)   | p50             | ~104.5     |
|                 | p95             | ~105.2     |
|                 | p99             | ~106.8     |
|                 | Max             | ~108.4     |
|                 | Mean            | ~104.5     |
| Performance     | Throughput     | 9.6 QPS     |
| Build Flags     | Compiler        | g++ -O3 C++20 |


## References

1. [In Search of the History of the Vector Database](https://sw2.beehiiv.com/p/search-history-vector-database)
2. [A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge](https://arxiv.org/pdf/2310.11703)
3. [Near Neighbor Search in Large Metric Spaces](https://vldb.org/conf/1995/P574.PDF)
4. [The Binary Vector as the Basis of an Inverted Index File](https://ital.corejournals.org/index.php/ital/article/view/8961/8080)
5. [A vector space model for automatic indexing](https://dl.acm.org/doi/epdf/10.1145/361219.361220)
6. [MicroNN: An On-device Disk-resident Updatable Vector Database](https://arxiv.org/pdf/2504.05573)
7. [The FAISS Library](https://arxiv.org/pdf/2401.08281)
