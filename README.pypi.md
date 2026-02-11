# spheni

A tiny CPU-first, in-memory vector search library in C++ with Python bindings.

- Source: https://github.com/datavorous/spheni
- License: Apache-2.0

## Install

```bash
pip install spheni
```

## Quick Example

```python
import numpy as np
import spheni

spec = spheni.IndexSpec(4, spheni.Metric.L2, spheni.IndexKind.Flat)
engine = spheni.Engine(spec)

base = np.random.rand(10, 4).astype(np.float32)
engine.add(base)

query = np.random.rand(4).astype(np.float32)
hits = engine.search(query, 3)

for h in hits:
    print(h.id, h.score)
```

## Features

- Indexes: Flat, IVF
- Metrics: Cosine, L2
- Storage: F32, INT8
- Operations: add, search, search_batch, train, save, load

## Build From Source

```bash
python -m pip wheel . --no-deps -w dist
```

Then install the wheel from `dist/`.

## Links

- Full README: https://github.com/datavorous/spheni/blob/master/README.md
- Python API: https://github.com/datavorous/spheni/blob/master/docs/wiki/python_api.md
- Build Guide: https://github.com/datavorous/spheni/blob/master/docs/wiki/building.md