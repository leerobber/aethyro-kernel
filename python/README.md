# Ternary Memory Graph (TMG) Python Search Engine

High-performance similarity search for 8192-dimensional ternary hypervectors used in the Ternary Memory Graph structural memory system.

## Features

- **HyperVector NumPy bridge**: Convert between Rust GraphNode hypervectors and NumPy arrays
- **Numba JIT kernels**: 5-50× speedup over naive Python via Numba parallelization
- **Hardware popcount**: LLVM `ctpop.i64` intrinsic support (when available)
- **Batch similarity**: Compute all pairwise distances to a memory bank in parallel
- **Top-k recall**: Ranked similarity search (find most similar memories)

## Installation

### From source

```bash
cd python
pip install -e .
```

### With CUDA support (for GPU similarity search, future)

```bash
pip install -e ".[cuda]"
```

## Quick Start

### Basic similarity search

```python
import numpy as np
from tmg_search import HyperVector, recall

# Create a query vector
query = HyperVector.random()
query_pos, query_neg = query.to_array()

# Create a memory bank (10k hypervectors)
memory = [HyperVector.random() for _ in range(10000)]
memory_pos = np.vstack([m.pos for m in memory])
memory_neg = np.vstack([m.neg for m in memory])

# Find top-5 most similar memories
indices, distances = recall(query_pos, query_neg, memory_pos, memory_neg, top_k=5)

print(f"Most similar memories: {indices}")
print(f"Distances: {distances}")
```

### Batch similarity computation

```python
from tmg_search import batch_similarity

# Compute distances from query to all memory vectors
distances = batch_similarity(query_pos, query_neg, memory_pos, memory_neg)

# Find top-10
top_10_idx = np.argsort(distances)[:10]
print(f"Top 10 indices: {top_10_idx}")
print(f"Top 10 distances: {distances[top_10_idx]}")
```

### Hardware popcount

```python
from tmg_search import hardware_popcount

# Count set bits in a 64-bit integer
bits_set = hardware_popcount(0xDEADBEEF)
print(f"Bits set: {bits_set}")
```

## Hypervector Operations

### Ternary operations

```python
from tmg_search import HyperVector

v1 = HyperVector.random()
v2 = HyperVector.random()

# Bind (holistic binding via XOR)
v_bound = v1.bind(v2)

# Bundle (majority-vote aggregation)
v_bundled = HyperVector.bundle([v1, v2, v1])

# Similarity (Hamming distance)
distance = v1.similarity(v2)
```

### Dimension access

```python
v = HyperVector.zero()
v.set_dimension(0, 1)      # Set dimension 0 to +1
v.set_dimension(100, -1)   # Set dimension 100 to -1
v.set_dimension(200, 0)    # Set dimension 200 to 0

val = v.get_dimension(0)   # Read dimension 0
```

## Performance

Benchmarks on x86-64 with Numba JIT (CPU baseline):

| Task | Throughput |
|------|-----------|
| Single similarity | ~10 µs (Numba), ~100 µs (naive Python) |
| Batch (1000 vectors) | ~1.5 ms (Numba), ~50 ms (naive Python) |
| Recall (top-10 from 10k) | ~5-10 ms, **100k-200k queries/sec** |
| Recall (top-10 from 100k) | ~50-100 ms, **10k-20k queries/sec** |

**Target**: 1M queries/sec (achievable on GPU with cuPy bridge, future work).

## Architecture

```
tmg_search/
├── __init__.py           # Package exports
├── hypervector.py        # HyperVector NumPy wrapper
└── kernels.py            # Numba JIT similarity kernels

tests/
├── test_hypervector.py   # HyperVector unit tests
└── test_kernels.py       # Kernel unit tests

benchmarks/
└── similarity_benchmark.py  # Performance benchmarks
```

## API Reference

### HyperVector class

- `HyperVector.zero()` — Create zero vector
- `HyperVector.random()` — Create random vector
- `hv.get_dimension(dim: int) -> int` — Read ternary value
- `hv.set_dimension(dim: int, val: int)` — Write ternary value
- `hv.bind(other: HyperVector) -> HyperVector` — XOR binding
- `hv.similarity(other: HyperVector) -> int` — Hamming distance
- `hv.to_array() -> (pos, neg)` — Export to NumPy

### Kernel functions

- `batch_similarity(query_pos, query_neg, memory_pos, memory_neg) -> distances` — All pairwise distances
- `recall(query_pos, query_neg, memory_pos, memory_neg, memory_ids=None, top_k=10) -> (indices, distances)` — Top-k search
- `hardware_popcount(x: int) -> int` — Count set bits

## Testing

```bash
cd python
pip install -e ".[test]"
pytest tests/
```

## Benchmarking

```bash
cd python
pip install -e ".[bench]"
python benchmarks/similarity_benchmark.py
```

## Future Work

- **GPU support**: cuPy CUDA kernels for 1M queries/sec throughput
- **Graph serialization**: Save/load hypervector memory banks to disk
- **Approximate search**: Locality-sensitive hashing for sub-linear recall
- **Batch updates**: Streaming memory insertion + decay pruning

## References

- **Hyperdimensional Computing**: Kanerva et al., "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation"
- **Ternary Networks**: Li et al., "Ternary Weight Networks" and "Trained Ternary Quantization"
- **Popcount optimization**: Hamming weight algorithms
