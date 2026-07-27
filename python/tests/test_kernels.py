"""
Tests for Numba similarity kernels.
"""

import numpy as np
import pytest
from tmg_search.kernels import batch_similarity, recall, hardware_popcount


class TestPopcount:
    """Hardware popcount tests."""

    def test_popcount_zero_is_zero(self):
        """Popcount of 0 = 0."""
        assert hardware_popcount(0) == 0

    def test_popcount_all_ones_is_64(self):
        """Popcount of 2^64-1 = 64."""
        assert hardware_popcount((1 << 64) - 1) == 64

    def test_popcount_single_bit(self):
        """Popcount of 2^k = 1."""
        for k in [0, 1, 32, 63]:
            assert hardware_popcount(1 << k) == 1


class TestBatchSimilarity:
    """Batch similarity kernel tests."""

    def test_similarity_shape_mismatch_raises(self):
        """Shape validation."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.zeros((100, 127), dtype=np.uint64)  # Wrong size
        memory_neg = np.zeros((100, 128), dtype=np.uint64)

        with pytest.raises(ValueError):
            batch_similarity(query_pos, query_neg, memory_pos, memory_neg)

    def test_batch_similarity_output_shape(self):
        """Output shape matches memory bank size."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.zeros((1000, 128), dtype=np.uint64)
        memory_neg = np.zeros((1000, 128), dtype=np.uint64)

        similarities = batch_similarity(query_pos, query_neg, memory_pos, memory_neg)
        assert similarities.shape == (1000,)
        assert similarities.dtype == np.int64

    def test_batch_similarity_identical_vectors(self):
        """Similarity to identical vector = 0."""
        query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)

        # Single memory vector identical to query
        memory_pos = query_pos.reshape(1, 128)
        memory_neg = query_neg.reshape(1, 128)

        similarities = batch_similarity(query_pos, query_neg, memory_pos, memory_neg)
        assert similarities[0] == 0

    def test_batch_similarity_independent_vectors(self):
        """Random vectors have non-zero similarity."""
        query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)

        similarities = batch_similarity(query_pos, query_neg, memory_pos, memory_neg)

        # Most similarities should be > 0 (random vectors unlikely identical)
        assert np.sum(similarities == 0) < 10
        assert np.all(similarities >= 0)
        assert np.all(similarities <= 2 * 8192)

    def test_batch_similarity_many_vectors(self):
        """Handles large memory banks efficiently."""
        query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (10000, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (10000, 128), dtype=np.uint64)

        similarities = batch_similarity(query_pos, query_neg, memory_pos, memory_neg)
        assert len(similarities) == 10000


class TestRecall:
    """Top-k similarity search tests."""

    def test_recall_top_k_shape(self):
        """Output shape matches top_k."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)

        indices, distances = recall(query_pos, query_neg, memory_pos, memory_neg, top_k=10)
        assert len(indices) == 10
        assert len(distances) == 10

    def test_recall_distances_ascending(self):
        """Returned distances are in ascending order."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)

        indices, distances = recall(query_pos, query_neg, memory_pos, memory_neg, top_k=10)
        assert np.all(distances[:-1] <= distances[1:])

    def test_recall_identical_in_bank(self):
        """Identical vector returns 0 distance at top."""
        query_pos = np.random.randint(0, 2**64, (128,), dtype=np.uint64)
        query_neg = np.random.randint(0, 2**64, (128,), dtype=np.uint64)

        # Insert query into memory bank
        memory_pos = np.vstack([
            query_pos,
            np.random.randint(0, 2**64, (99, 128), dtype=np.uint64)
        ])
        memory_neg = np.vstack([
            query_neg,
            np.random.randint(0, 2**64, (99, 128), dtype=np.uint64)
        ])

        indices, distances = recall(query_pos, query_neg, memory_pos, memory_neg, top_k=5)
        assert distances[0] == 0  # Top result has zero distance

    def test_recall_with_memory_ids(self):
        """recall() respects memory_ids."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (100, 128), dtype=np.uint64)
        memory_ids = np.array([f"mem_{i}" for i in range(100)], dtype=object)

        indices, distances = recall(
            query_pos, query_neg, memory_pos, memory_neg,
            memory_ids=memory_ids, top_k=5
        )

        # Indices should be memory_ids, not raw indices
        assert all(isinstance(idx, (str, np.str_)) for idx in indices)

    def test_recall_top_k_clamped_to_bank_size(self):
        """top_k is clamped to bank size."""
        query_pos = np.zeros((128,), dtype=np.uint64)
        query_neg = np.zeros((128,), dtype=np.uint64)
        memory_pos = np.random.randint(0, 2**64, (5, 128), dtype=np.uint64)
        memory_neg = np.random.randint(0, 2**64, (5, 128), dtype=np.uint64)

        indices, distances = recall(query_pos, query_neg, memory_pos, memory_neg, top_k=100)
        assert len(indices) == 5  # Clamped to bank size
