"""
Tests for HyperVector NumPy bridge.
"""

import numpy as np
import pytest
from tmg_search.hypervector import HyperVector, from_array, to_array


class TestHyperVector:
    """HyperVector basic operations."""

    def test_zero_vector_is_all_zeros(self):
        """Zero vector has all dimensions = 0."""
        hv = HyperVector.zero()
        for i in range(0, HyperVector.DIM, 100):
            assert hv.get_dimension(i) == 0

    def test_random_has_correct_shape(self):
        """Random vector has shape (128,)."""
        hv = HyperVector.random()
        assert hv.pos.shape == (128,)
        assert hv.neg.shape == (128,)
        assert hv.pos.dtype == np.uint64
        assert hv.neg.dtype == np.uint64

    def test_set_and_get_dimension(self):
        """Set/get dimension roundtrip."""
        hv = HyperVector.zero()
        hv.set_dimension(0, 1)
        hv.set_dimension(1, -1)
        hv.set_dimension(63, 1)
        hv.set_dimension(64, -1)
        hv.set_dimension(8191, 1)

        assert hv.get_dimension(0) == 1
        assert hv.get_dimension(1) == -1
        assert hv.get_dimension(63) == 1
        assert hv.get_dimension(64) == -1
        assert hv.get_dimension(8191) == 1
        assert hv.get_dimension(2) == 0

    def test_similarity_self_is_zero(self):
        """Self-similarity = 0."""
        hv = HyperVector.random()
        assert hv.similarity(hv) == 0

    def test_similarity_opposite_is_max(self):
        """Opposite vectors have max distance."""
        v1 = HyperVector.zero()
        v2 = HyperVector.zero()

        for i in range(HyperVector.DIM):
            v1.set_dimension(i, 1)
            v2.set_dimension(i, -1)

        # Max distance = 2 * DIM (all bits different in both slices)
        assert v1.similarity(v2) == 2 * HyperVector.DIM

    def test_bind_xor_property(self):
        """Bind is XOR: (A bind B) bind B = A."""
        v1 = HyperVector.random()
        v2 = HyperVector.random()

        v12 = v1.bind(v2)
        v12_b = v12.bind(v2)

        # v12_b should equal v1 (XOR is self-inverse)
        for i in range(0, HyperVector.DIM, 100):
            assert v12_b.get_dimension(i) == v1.get_dimension(i)

    def test_bundle_majority(self):
        """Bundle uses majority vote."""
        v1 = HyperVector.zero()
        v2 = HyperVector.zero()
        v3 = HyperVector.zero()

        # Set dimension 0 to: +1, +1, -1 → majority is +1
        v1.set_dimension(0, 1)
        v2.set_dimension(0, 1)
        v3.set_dimension(0, -1)

        bundled = HyperVector.bundle([v1, v2, v3])
        assert bundled.get_dimension(0) == 1

    def test_from_array_to_array_roundtrip(self):
        """Conversion roundtrip preserves data."""
        hv = HyperVector.random()
        pos, neg = to_array(hv)

        hv2 = from_array(pos, neg)
        assert np.array_equal(hv2.pos, hv.pos)
        assert np.array_equal(hv2.neg, hv.neg)

    def test_copy_is_independent(self):
        """Copy creates independent instance."""
        hv1 = HyperVector.random()
        hv2 = hv1.copy()

        hv2.set_dimension(0, 1)
        assert hv1.get_dimension(0) != 1

    def test_invalid_dimension_raises(self):
        """Out-of-bounds dimension raises IndexError."""
        hv = HyperVector.zero()
        with pytest.raises(IndexError):
            hv.get_dimension(HyperVector.DIM)
        with pytest.raises(IndexError):
            hv.set_dimension(-1, 1)

    def test_invalid_ternary_value_raises(self):
        """Invalid ternary value raises ValueError."""
        hv = HyperVector.zero()
        with pytest.raises(ValueError):
            hv.set_dimension(0, 2)
