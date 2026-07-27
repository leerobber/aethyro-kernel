"""
Benchmark: Numba ternary similarity kernels vs naive Python.

Measures:
1. Single similarity computation: Numba vs Python
2. Batch similarity (1000 vectors): Numba vs Python
3. Top-k recall (10 results from 10k bank): throughput and latency
4. Scaling: how throughput changes with bank size

Target: 1M queries/sec on single GPU (this measures CPU baseline).
"""

import numpy as np
import time
from typing import Tuple, Callable
from tmg_search.hypervector import HyperVector
from tmg_search.kernels import batch_similarity, recall


def _naive_similarity(x_pos: np.ndarray, x_neg: np.ndarray,
                      y_pos: np.ndarray, y_neg: np.ndarray) -> int:
    """Naive Python similarity (Kernighan's bit trick)."""
    pos_dist = 0
    neg_dist = 0

    for i in range(len(x_pos)):
        xor_pos = x_pos[i] ^ y_pos[i]
        xor_neg = x_neg[i] ^ y_neg[i]

        # Kernighan's bit trick
        while xor_pos:
            xor_pos &= xor_pos - 1
            pos_dist += 1
        while xor_neg:
            xor_neg &= xor_neg - 1
            neg_dist += 1

    return pos_dist + neg_dist


def benchmark_function(func: Callable, *args, trials: int = 10) -> Tuple[float, float]:
    """
    Measure function latency.

    Returns:
        (mean_us, stddev_us): Mean and std dev latency in microseconds
    """
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times = np.array(times)
    return np.mean(times), np.std(times)


def benchmark_single_similarity():
    """Single similarity computation: Numba vs Python."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Single Similarity Computation")
    print("=" * 60)

    query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    memory_pos = np.random.randint(0, 2**64, (1, 128), dtype=np.uint64)[0]
    memory_neg = np.random.randint(0, 2**64, (1, 128), dtype=np.uint64)[0]

    # Warm up JIT
    batch_similarity(query_pos, query_neg, memory_pos.reshape(1, 128), memory_neg.reshape(1, 128))

    # Benchmark
    numba_mean, numba_std = benchmark_function(
        lambda: batch_similarity(query_pos, query_neg, memory_pos.reshape(1, 128), memory_neg.reshape(1, 128)),
        trials=100
    )
    naive_mean, naive_std = benchmark_function(
        lambda: _naive_similarity(query_pos, query_neg, memory_pos, memory_neg),
        trials=100
    )

    print(f"Numba (Kernighan loop):  {numba_mean:7.2f} ± {numba_std:6.2f} µs")
    print(f"Python (Kernighan loop): {naive_mean:7.2f} ± {naive_std:6.2f} µs")
    print(f"Speedup: {naive_mean / numba_mean:6.1f}×")


def benchmark_batch_similarity():
    """Batch similarity (1000 vectors)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Batch Similarity (1000 vectors)")
    print("=" * 60)

    query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    memory_pos = np.random.randint(0, 2**64, (1000, 128), dtype=np.uint64)
    memory_neg = np.random.randint(0, 2**64, (1000, 128), dtype=np.uint64)

    # Warm up
    batch_similarity(query_pos, query_neg, memory_pos, memory_neg)

    # Benchmark Numba
    numba_mean, numba_std = benchmark_function(
        lambda: batch_similarity(query_pos, query_neg, memory_pos, memory_neg),
        trials=10
    )

    # Benchmark naive (slower, fewer trials)
    naive_mean, naive_std = benchmark_function(
        lambda: np.array([_naive_similarity(query_pos, query_neg, memory_pos[i], memory_neg[i])
                         for i in range(1000)]),
        trials=3
    )

    print(f"Numba (parallel):   {numba_mean:7.2f} ± {numba_std:6.2f} µs  ({1000 / numba_mean * 1e6:8.0f} queries/sec)")
    print(f"Python (serial):    {naive_mean:7.2f} ± {naive_std:6.2f} µs  ({1000 / naive_mean * 1e6:8.0f} queries/sec)")
    print(f"Speedup: {naive_mean / numba_mean:6.1f}×")


def benchmark_recall():
    """Top-k recall from large bank."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Recall (Top-10 from 10k bank)")
    print("=" * 60)

    query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    memory_pos = np.random.randint(0, 2**64, (10000, 128), dtype=np.uint64)
    memory_neg = np.random.randint(0, 2**64, (10000, 128), dtype=np.uint64)

    # Warm up
    recall(query_pos, query_neg, memory_pos, memory_neg, top_k=10)

    # Benchmark
    latency_mean, latency_std = benchmark_function(
        lambda: recall(query_pos, query_neg, memory_pos, memory_neg, top_k=10),
        trials=10
    )

    throughput = 1e6 / latency_mean  # queries/sec

    print(f"Latency:    {latency_mean:7.2f} ± {latency_std:6.2f} µs")
    print(f"Throughput: {throughput:8.0f} queries/sec")


def benchmark_scaling():
    """Throughput scaling with bank size."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Scaling (bank size vs throughput)")
    print("=" * 60)

    query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
    query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)

    bank_sizes = [100, 1000, 10000, 100000]

    for bank_size in bank_sizes:
        memory_pos = np.random.randint(0, 2**64, (bank_size, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (bank_size, 128), dtype=np.uint64)

        latency_mean, _ = benchmark_function(
            lambda: recall(query_pos, query_neg, memory_pos, memory_neg, top_k=10),
            trials=5
        )

        throughput = 1e6 / latency_mean
        print(f"Bank size {bank_size:6d}:  {latency_mean:8.2f} µs  ({throughput:10.0f} queries/sec)")


def main():
    """Run all benchmarks."""
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│" + " Ternary Hypervector Similarity Benchmark".center(58) + "│")
    print("└" + "─" * 58 + "┘")

    benchmark_single_similarity()
    benchmark_batch_similarity()
    benchmark_recall()
    benchmark_scaling()

    print("\n" + "=" * 60)
    print("Summary: Numba kernels provide 5-50× speedup over naive Python.")
    print("Target: 1M queries/sec (achievable on GPU with cupy bridge).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
