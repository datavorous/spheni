# Benchmark Report

**Date:** January 26, 2026

**Hardware:** Intel Core i7-8650U (4C/8T, 1.9GHz - 4.2GHz, 15W TDP)

**Compiler:** GCC 13.x (`-O3 -std=c++20 -march=native`)

## 1. Experimental Design

This benchmark measures the operational efficiency of the Spheni IVF index against a brute-force FLAT baseline.

### Control Constraints

* **Vector Dimension:** 128 (Float32)
* **Metric:** L2 (Euclidean) Distance
* **Dataset Scale:** 50,000 vectors (Subset of SIFT1M)
* **Execution:** Single-threaded pinning (sequential queries)

> [!NOTE]
> While 1M vectors were loaded into memory, the index was limited to 50,000 vectors. This allows for rapid iteration and high-precision tail latency profiling on a mobile-class U-series CPU without thermal throttling skewing the results.

## 2. Baseline: FLAT Index (Brute Force)

The FLAT baseline defines the maximum accuracy and hardware-bound performance floor.

| Metric | Value |
| --- | --- |
| **Recall@10** | 99.00% |
| **Mean Latency** | **5.271 ms** |
| **p95 Latency** | 5.552 ms |
| **p99 Latency** | 5.746 ms |
| **Throughput** | 189.7 QPS |
| **Build Time** | 16 ms |

## 3. Candidate: IVF Index Performance

The Inverted File (IVF) index utilizes Voronoi-based spatial partitioning. We evaluate two clustering scales (`nlist`) across a sweep of search breadths (`nprobe`).

### nlist = 100

*Training: ~7 seconds | Addition: ~0.5 seconds*

| nprobe | Recall % | Mean (ms) | p99 (ms) | Throughput (QPS) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 54.42% | 0.076 | 0.138 | 13,170.0 | **69.3x** |
| 4 | 87.86% | 0.245 | 0.341 | 4,086.1 | **21.5x** |
| **8 (Elbow)** | **95.93%** | **0.458** | **0.611** | **2,185.0** | **11.5x** |
| 16 | 98.70% | 0.879 | 1.060 | 1,137.6 | **6.0x** |

### nlist = 256

*Training: ~35 seconds | Addition: ~1.2 seconds*

| nprobe | Recall % | Mean (ms) | p99 (ms) | Throughput (QPS) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 49.78% | 0.061 | 0.109 | 16,337.2 | **86.4x** |
| 8 | 90.66% | 0.232 | 0.325 | 4,318.2 | **22.7x** |
| **16 (Elbow)** | **97.11%** | **0.416** | **0.554** | **2,405.4** | **12.6x** |
| 64 | 99.00% | 1.405 | 1.621 | 711.6 | **3.7x** |

#### Visuals: IVF Performance

![Recall/latency trade-off](../media/graphs/ivf_recall_latency_tradeoff.png)
*Recall vs latency for IVF configurations (lower is better latency).* 

![QPS vs recall sweep](../media/graphs/qps_vs_recall.png)
*Throughput (QPS) against recall across nprobe sweep.*

![Search fraction efficiency](../media/graphs/search_fraction_efficiency.png)
*Fraction of clusters searched vs achieved recall; highlights efficiency gains as nprobe scales.*

## 4. Key Findings

### Optimal Performance Envelope

The **IVF-256 / nprobe-16** configuration remains the production recommendation. It achieves **97.1% Recall** while sustaining **2,400+ QPS**, representing a **12.6x performance gain** over brute force with minimal tail-latency variance (p99 is within 0.14ms of the mean).

### Training vs. Search Efficiency

Increasing `nlist` from 100 to 256 increased training time by **5x** (7s to 35s) due to the  complexity of K-Means. However, it improved search throughput at high recall levels by approximately **11%**, proving that denser clustering pays dividends in query-heavy environments.

### Tail Latency Analysis

The p99 latencies are remarkably stable across all IVF tests, typically staying within **1.2x–1.5x** of the mean. This suggests high code quality with minimal branch mispredictions or memory alignment issues during the inverted list scans.

#### Visuals: Tail Latency

![Tail latency distribution](../media/graphs/latency_distribution.png)
*CDF of per-query latency showing tight p99 spread.*

## 5. Parallel Batch Query Execution (OpenMP)

**Date:** February 2, 2026

Enabling OpenMP batch query parallelism (OMP=4) on the IVF-256 / nprobe-16 configuration increases throughput from `2,405 QPS` to `5,710 QPS` (`2.37x`) while preserving Recall@10 at `97.11%`. Mean per-query latency over a fixed batch workload improves from `0.416 ms` to `0.176 ms` (`2.36x`), and p99 tail latency reduces from `0.554 ms` to `0.287 ms` (`1.93x`).

Parallelism is applied across independent queries, without modifying the underlying index or search logic.

All values are averaged over 5 runs.