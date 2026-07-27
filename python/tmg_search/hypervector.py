"""
HyperVector NumPy bridge.

Converts between Rust HyperVector (bit-sliced {-1,0,+1}) and NumPy for
batch processing. Maintains the 8192-dimensional ternary representation.
"""

import numpy as np
from typing import Tuple


class HyperVector:
    """
    8192-dimensional ternary hypervector with bit-sliced encoding.

    Attributes:
        pos (np.ndarray): Positive bit slice, shape (128,), dtype uint64
        neg (np.ndarray): Negative bit slice, shape (128,), dtype uint64
    """

    DIM = 8192
    WORDS = DIM // 64

    def __init__(self, pos: np.ndarray = None, neg: np.ndarray = None):
        """
        Initialize from bit slices.

        Args:
            pos: Positive slice (128 uint64 words), or None for zero vector
            neg: Negative slice (128 uint64 words), or None for zero vector
        """
        if pos is None:
            self.pos = np.zeros(self.WORDS, dtype=np.uint64)
        else:
            self.pos = np.asarray(pos, dtype=np.uint64)
            if self.pos.shape != (self.WORDS,):
                raise ValueError(f"pos shape must be ({self.WORDS},), got {self.pos.shape}")

        if neg is None:
            self.neg = np.zeros(self.WORDS, dtype=np.uint64)
        else:
            self.neg = np.asarray(neg, dtype=np.uint64)
            if self.neg.shape != (self.WORDS,):
                raise ValueError(f"neg shape must be ({self.WORDS},), got {self.neg.shape}")

    @staticmethod
    def zero() -> "HyperVector":
        """Create a zero vector (all dimensions = 0)."""
        return HyperVector()

    @staticmethod
    def random() -> "HyperVector":
        """Create a random hypervector with uniform ternary distribution."""
        pos = np.random.randint(0, 2**64, size=HyperVector.WORDS, dtype=np.uint64)
        neg = np.random.randint(0, 2**64, size=HyperVector.WORDS, dtype=np.uint64)
        return HyperVector(pos, neg)

    def get_dimension(self, dim: int) -> int:
        """Get value of a single dimension {-1, 0, +1}."""
        if not 0 <= dim < self.DIM:
            raise IndexError(f"dimension {dim} out of range [0, {self.DIM})")
        word_idx = dim // 64
        bit_idx = dim % 64
        pos_bit = (self.pos[word_idx] >> bit_idx) & 1
        neg_bit = (self.neg[word_idx] >> bit_idx) & 1
        if pos_bit and not neg_bit:
            return 1
        elif neg_bit and not pos_bit:
            return -1
        else:
            return 0

    def set_dimension(self, dim: int, val: int):
        """Set a single dimension to a ternary value."""
        if not 0 <= dim < self.DIM:
            raise IndexError(f"dimension {dim} out of range [0, {self.DIM})")
        if val not in [-1, 0, 1]:
            raise ValueError(f"dimension value must be in {{-1, 0, +1}}, got {val}")

        word_idx = dim // 64
        bit_idx = dim % 64
        mask = np.uint64(1) << bit_idx

        if val == 1:
            self.pos[word_idx] |= mask
            self.neg[word_idx] &= ~mask
        elif val == -1:
            self.pos[word_idx] &= ~mask
            self.neg[word_idx] |= mask
        else:
            self.pos[word_idx] &= ~mask
            self.neg[word_idx] &= ~mask

    def bind(self, other: "HyperVector") -> "HyperVector":
        """Element-wise XOR (holistic binding)."""
        return HyperVector(
            self.pos ^ other.pos,
            self.neg ^ other.neg
        )

    @staticmethod
    def bundle(vectors: list["HyperVector"]) -> "HyperVector":
        """Majority-vote aggregation across multiple hypervectors."""
        if not vectors:
            return HyperVector.zero()

        pos_sum = np.zeros(HyperVector.WORDS, dtype=np.uint64)
        neg_sum = np.zeros(HyperVector.WORDS, dtype=np.uint64)

        # Sum all bit positions
        for vec in vectors:
            pos_sum += np.unpackbits(vec.pos)
            neg_sum += np.unpackbits(vec.neg)

        # Threshold: more than half the vectors have this bit set
        threshold = len(vectors) // 2 + 1

        # Reconstruct from majority-vote bits
        result_pos = np.packbits((pos_sum >= threshold).astype(np.uint8))
        result_neg = np.packbits((neg_sum >= threshold).astype(np.uint8))

        return HyperVector(result_pos, result_neg)

    def similarity(self, other: "HyperVector") -> int:
        """Ternary Hamming distance (0 = identical, 16384 = opposite)."""
        pos_xor = self.pos ^ other.pos
        neg_xor = self.neg ^ other.neg
        pos_dist = sum(bin(x).count('1') for x in pos_xor)
        neg_dist = sum(bin(x).count('1') for x in neg_xor)
        return pos_dist + neg_dist

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export as (pos, neg) tuple for external use."""
        return self.pos.copy(), self.neg.copy()

    def copy(self) -> "HyperVector":
        """Create a deep copy."""
        return HyperVector(self.pos.copy(), self.neg.copy())


def from_array(pos: np.ndarray, neg: np.ndarray) -> HyperVector:
    """Create HyperVector from (pos, neg) bit slices."""
    return HyperVector(pos, neg)


def to_array(hv: HyperVector) -> Tuple[np.ndarray, np.ndarray]:
    """Export HyperVector as (pos, neg) tuple."""
    return hv.to_array()
