# Spheni

Small in-memory vector search library in C++.

## Build

```bash
./build.sh
```

This builds:

- `build/libspheni.a`
- `build/example`

## Minimal usage

See [example.cpp](example.cpp).

Run it after build:

```bash
./build/example
```

## Layout

- `include/spheni/`: public headers
- `src/core/`: engine and factory
- `src/indexes/`: flat and IVF indexes
- `src/math/`: kernels, kmeans, top-k
- `src/io/`: serialization helpers
