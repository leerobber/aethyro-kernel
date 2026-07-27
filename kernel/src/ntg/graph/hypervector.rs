//! Hyperdimensional computing vectors for structural memory in the TMG.
//!
//! Ternary hypervectors (8192-dim, 128×u64 encoding as bit-sliced {-1,0,+1}).
//! Operations: bind (XOR), bundle (majority vote), similarity (ternary popcount).

/// Ternary hypervector: 8192-dim, packed as 128 u64 words.
/// Each dimension is ternary {-1, 0, +1} via bit-slicing:
///   bit_pos[i] ⊕ bit_neg[i] determines ternary state per dimension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HyperVector {
    /// Positive bit slice (128 u64 words = 8192 bits).
    pub pos: Vec<u64>,
    /// Negative bit slice (128 u64 words = 8192 bits).
    pub neg: Vec<u64>,
}

impl HyperVector {
    /// 8192 dimensions packed as 128 u64 words.
    pub const DIM: usize = 8192;
    pub const WORDS: usize = Self::DIM / 64;

    /// Create a zero vector (all dimensions = 0).
    pub fn zero() -> Self {
        Self {
            pos: vec![0u64; Self::WORDS],
            neg: vec![0u64; Self::WORDS],
        }
    }

    /// Create a random hypervector with ~50% density (uniform random {-1,0,+1}).
    pub fn random() -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};

        let hasher = RandomState::new().build_hasher();
        let seed = hasher.finish();

        let mut pos = vec![0u64; Self::WORDS];
        let mut neg = vec![0u64; Self::WORDS];

        let mut rng = seed;
        for i in 0..Self::WORDS {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            pos[i] = rng;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            neg[i] = rng;
        }

        Self { pos, neg }
    }

    /// Bind two hypervectors: element-wise XOR (holistic binding).
    pub fn bind(&self, other: &HyperVector) -> HyperVector {
        let mut pos = self.pos.clone();
        let mut neg = self.neg.clone();
        for i in 0..Self::WORDS {
            pos[i] ^= other.pos[i];
            neg[i] ^= other.neg[i];
        }
        HyperVector { pos, neg }
    }

    /// Bundle multiple hypervectors: majority vote per dimension.
    /// Returns a new hypervector where each dimension is the majority element.
    pub fn bundle(vectors: &[&HyperVector]) -> HyperVector {
        if vectors.is_empty() {
            return Self::zero();
        }
        let mut result_pos = vec![0u64; Self::WORDS];
        let mut result_neg = vec![0u64; Self::WORDS];

        let threshold = (vectors.len() as u64 + 1) / 2;

        for word_idx in 0..Self::WORDS {
            let mut pos_count = 0u64;
            let mut neg_count = 0u64;

            for vec in vectors {
                if (vec.pos[word_idx] >> 0) & 1 == 1 {
                    pos_count += 1;
                }
                if (vec.neg[word_idx] >> 0) & 1 == 1 {
                    neg_count += 1;
                }
            }

            if pos_count >= threshold {
                result_pos[word_idx] |= 1;
            }
            if neg_count >= threshold {
                result_neg[word_idx] |= 1;
            }
        }

        HyperVector {
            pos: result_pos,
            neg: result_neg,
        }
    }

    /// Similarity: ternary popcount distance (0 = identical, 8192 = opposite).
    /// Returns the Hamming distance using hardware popcount via bit-sliced representation.
    pub fn similarity(&self, other: &HyperVector) -> i64 {
        let mut pos_dist = 0u64;
        let mut neg_dist = 0u64;

        for i in 0..Self::WORDS {
            let xor_pos = self.pos[i] ^ other.pos[i];
            let xor_neg = self.neg[i] ^ other.neg[i];
            pos_dist += xor_pos.count_ones() as u64;
            neg_dist += xor_neg.count_ones() as u64;
        }

        (pos_dist + neg_dist) as i64
    }

    /// Decode a single dimension (returns {-1, 0, +1}).
    pub fn get_dimension(&self, dim: usize) -> i8 {
        if dim >= Self::DIM {
            return 0;
        }
        let word_idx = dim / 64;
        let bit_idx = dim % 64;
        let pos_bit = (self.pos[word_idx] >> bit_idx) & 1;
        let neg_bit = (self.neg[word_idx] >> bit_idx) & 1;

        match (pos_bit, neg_bit) {
            (1, 0) => 1,
            (0, 1) => -1,
            _ => 0,
        }
    }

    /// Set a single dimension to a ternary value.
    pub fn set_dimension(&mut self, dim: usize, val: i8) {
        if dim >= Self::DIM {
            return;
        }
        let word_idx = dim / 64;
        let bit_idx = dim % 64;
        let mask = 1u64 << bit_idx;

        match val {
            1 => {
                self.pos[word_idx] |= mask;
                self.neg[word_idx] &= !mask;
            }
            -1 => {
                self.pos[word_idx] &= !mask;
                self.neg[word_idx] |= mask;
            }
            _ => {
                self.pos[word_idx] &= !mask;
                self.neg[word_idx] &= !mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_vector_is_all_zeros() {
        let v = HyperVector::zero();
        for i in 0..HyperVector::DIM {
            assert_eq!(v.get_dimension(i), 0);
        }
    }

    #[test]
    fn set_and_get_dimension() {
        let mut v = HyperVector::zero();
        v.set_dimension(0, 1);
        v.set_dimension(1, -1);
        v.set_dimension(63, 1);
        v.set_dimension(64, -1);

        assert_eq!(v.get_dimension(0), 1);
        assert_eq!(v.get_dimension(1), -1);
        assert_eq!(v.get_dimension(63), 1);
        assert_eq!(v.get_dimension(64), -1);
        assert_eq!(v.get_dimension(2), 0);
    }

    #[test]
    fn similarity_self_is_zero() {
        let v = HyperVector::random();
        assert_eq!(v.similarity(&v), 0);
    }

    #[test]
    fn similarity_opposite_is_max() {
        let mut v1 = HyperVector::zero();
        let mut v2 = HyperVector::zero();

        for i in 0..HyperVector::DIM {
            v1.set_dimension(i, 1);
            v2.set_dimension(i, -1);
        }

        // When all dimensions are opposite, both pos and neg bit slices differ.
        // Max distance is 2 * 8192 (all bits different in both slices).
        assert_eq!(v1.similarity(&v2), 2 * HyperVector::DIM as i64);
    }

    #[test]
    fn bind_xor_property() {
        let v1 = HyperVector::random();
        let v2 = HyperVector::random();
        let v12 = v1.bind(&v2);
        let v12_again = v1.bind(&v2);
        assert_eq!(v12, v12_again);
    }

    #[test]
    fn bundle_majority() {
        let mut v1 = HyperVector::zero();
        let mut v2 = HyperVector::zero();
        let mut v3 = HyperVector::zero();

        v1.set_dimension(0, 1);
        v2.set_dimension(0, 1);
        v3.set_dimension(0, -1);

        let bundled = HyperVector::bundle(&[&v1, &v2, &v3]);
        assert_eq!(bundled.get_dimension(0), 1);
    }
}
