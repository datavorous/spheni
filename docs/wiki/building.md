# Spheni: Build & Run Guide

This guide covers compiling Spheni, running examples, and using it from other folders.

## Prerequisites

1. CMake
2. A C++20 compiler (e.g., `g++` or `clang++`)
3. OpenMP support for your compiler (optional but recommended)
4. Python 3 if you want the Python module
5. `pybind11` (CMake package) if you want the Python module
6. For wheel builds: `pip install build scikit-build-core`

If OpenMP is not available, the build will still work but without parallel speedups.

Python command launcher note:
- Linux: use `python3` if `python` is not available.
- Windows: use `py` (for example, `py -m pip install pybind11`).

### Note for `pybind11`

- System package manager (recommended for CMake `find_package`)
  - Ubuntu/Debian: `sudo apt-get install pybind11-dev`
- Or via pip:
  - `python -m pip install pybind11`

When installed, CMake will pick it up automatically during `./build_spheni.sh --python`.

## Build Script Configurations
`build_spheni.sh` supports:

1. `--python` to build the Python module.
2. `--install <prefix>` to install headers/libs to a portable prefix (e.g., `./dist` or `/opt/spheni`).

```bash
# C++ library only
./build_spheni.sh

# C++ + Python module
./build_spheni.sh --python

# Build and install to a portable folder
./build_spheni.sh --python --install ./dist
```

## Development Build

```bash
# in repo root
./build_spheni.sh --python
```

### Enable local CPU tuning (`-march=native`)

For local/source builds on your own machine, you can enable native CPU tuning:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPHENI_BUILD_PYTHON=ON -DSPHENI_ENABLE_MARCH_NATIVE=ON
cmake --build build
```

Do not use this for redistributable binaries/wheels, since `-march=native` can make artifacts incompatible on other CPUs.

This produces:
1. `build/libspheni.a`
2. `build/_core*.so` (Python module)

### Run a C++ example

```bash
g++ -O3 -std=c++20 -Iinclude /path/to/app.cpp build/libspheni.a -o app -fopenmp
./app
```

### Run a Python example

```bash
PYTHONPATH=build python /path/to/script.py
```

## Portable Build

Create a portable install folder:
```bash
./build_spheni.sh --python --install ./dist
```

You can copy `dist/` to any compatible machine or project.

### C++ Usage (outside the repo)

```bash
g++ -O3 -std=c++20 -I/path/to/dist/include /path/to/app.cpp /path/to/dist/lib/libspheni.a -o app -fopenmp
```

### Python Usage (outside the repo)

```bash
PYTHONPATH=/path/to/dist/lib/spheni python /path/to/your_script.py
```

Example snippet:

```python
import sys
sys.path.append("./dist/lib/spheni")

import numpy as np
import spheni

spec = spheni.IndexSpec(
    4, 
    spheni.Metric.L2, 
    spheni.IndexKind.Flat, 
    spheni.StorageType.INT8, 
    False
)
engine = spheni.Engine(spec)

base = np.random.rand(10, 4).astype(np.float32)
engine.add(base)

query = np.random.rand(4).astype(np.float32)
results = engine.search(query, 3)

for hit in results:
    print(f"ID: {hit.id}, Distance: {hit.score}")
```

## Build a Python Wheel (PEP 427)

From repo root:

```bash
python -m pip install --upgrade pip
python -m pip wheel . --no-deps -w dist
```

Wheel builds keep `SPHENI_ENABLE_MARCH_NATIVE=OFF` by default for portability.

This generates a wheel in `dist/`, for example:

```bash
dist/spheni-0.1.0-cp312-cp312-linux_x86_64.whl
```

Install and test:

```bash
python -m pip install dist/spheni-0.1.0-*.whl
python -c "import spheni; print(spheni.Engine)"
```
