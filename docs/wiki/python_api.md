# Python API Reference

This document describes the Python bindings exposed by the `spheni` module.

## Enums

### `spheni.Metric`
- `Metric.Cosine`
- `Metric.L2`

### `spheni.IndexKind`
- `IndexKind.Flat`
- `IndexKind.IVF`

### `spheni.StorageType`
- `StorageType.F32`
- `StorageType.INT8`

## Classes

### `IndexSpec`
Constructor overloads:
- `IndexSpec(dim, metric, kind, normalize=True)`
- `IndexSpec(dim, metric, kind, storage, normalize=True)`
- `IndexSpec(dim, metric, kind, nlist, normalize=True)`
- `IndexSpec(dim, metric, kind, nlist, storage, normalize=True)`

Fields:
- `dim` (int): vector dimension
- `metric` (Metric)
- `normalize` (bool): if true and metric is Cosine, vectors are L2-normalized on add/query
- `kind` (IndexKind)
- `storage` (StorageType)
- `nlist` (int): IVF cluster count (only used for IVF)

### `SearchParams`
Constructors:
- `SearchParams(k)`
- `SearchParams(k, nprobe)`

Fields:
- `k` (int)
- `nprobe` (int): IVF probe count (defaults to 1 if not provided)

### `SearchHit`
Constructor:
- `SearchHit(id, score)`

Fields:
- `id` (int)
- `score` (float): higher is better (Cosine or negative L2)

### `Engine`
Constructor:
- `Engine(spec)`

Methods:
- `add(vectors)`
  - `vectors`: `numpy.ndarray` float32, shape `(n, dim)`, C-contiguous
  - auto-assigns integer ids starting from 0
- `add(ids, vectors)`
  - `ids`: `numpy.ndarray` int64, shape `(n,)`
  - `vectors`: `numpy.ndarray` float32, shape `(n, dim)`, C-contiguous
- `train()`
  - Required for IVF before search
- `search(query, k)`
  - `query`: `numpy.ndarray` float32, shape `(dim,)`, C-contiguous
  - returns `List[SearchHit]` (sorted by descending score)
- `search(query, k, nprobe)`
  - IVF search with explicit `nprobe`
- `search_batch(queries, k)`
  - `queries`: `numpy.ndarray` float32, shape `(n, dim)`, C-contiguous
  - returns `List[List[SearchHit]]`
- `search_batch(queries, k, nprobe)`
  - IVF batch search with explicit `nprobe`
- `save(path)`
  - `path`: string
- `load(path)` (static)
  - returns `Engine`

## Minimal example

```python
import numpy as np
import spheni

spec = spheni.IndexSpec(
    4, spheni.Metric.L2, spheni.IndexKind.Flat, spheni.StorageType.INT8, False
)
engine = spheni.Engine(spec)

base = np.random.rand(10, 4).astype(np.float32)
engine.add(base)

query = np.random.rand(4).astype(np.float32)
hits = engine.search(query, 3)
print([(h.id, h.score) for h in hits])
```
