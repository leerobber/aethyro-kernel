"""
Ternary Memory Graph (TMG) Python Search Engine

Provides NumPy bridge for HyperVectors and Numba JIT-compiled similarity
kernels for batch hypervector recall (finding the most similar memories).

Core operations:
- from_hypervector() / to_hypervector() — bidirectional conversion with Rust
- recall(query_vec, memory_vectors, top_k) — ranked similarity search
- batch_similarity() — compute all pairwise distances

Performance target: 1M similarity queries/sec on single GPU.
"""

__version__ = "0.1.0"

from .hypervector import HyperVector, from_array, to_array
from .kernels import recall, batch_similarity, hardware_popcount

__all__ = [
    "HyperVector",
    "from_array",
    "to_array",
    "recall",
    "batch_similarity",
    "hardware_popcount",
]
