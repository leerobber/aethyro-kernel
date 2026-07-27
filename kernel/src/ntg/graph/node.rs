//! Weighted compute node for the native ternary runtime.
//!
//! Distinct from the structural [`super::Node`] (label/signal topology).
//! `GraphNode` carries sparse bit-sliced weights used by
//! [`crate::ntg::runtime::Runtime::forward_native_parallel`].
//! Extended with optional HyperVector for structural memory (TMG phase).

use crate::ntg::error::NtgError;
use crate::ntg::ledger::TamperEvidentLedger;
use crate::ntg::storage::SparseBitSlicedTernary;
use super::hypervector::HyperVector;

/// One layer-side compute unit: dense sequential `id` + sparse ternary weights.
///
/// Ingest contract: within a layer, `node.id == index` (see `tools/ingest.py`
/// and [`crate::ntg::runtime::validate_sequential_ids`]).
///
/// Extended with optional HyperVector for Ternary Memory Graph (TMG):
/// stores structural memory as high-dimensional ternary vectors.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: usize,
    pub weights: SparseBitSlicedTernary,
    /// Optional hypervector for structural memory; none until assigned.
    pub hypervector: Option<HyperVector>,
}

impl GraphNode {
    pub fn new(id: usize, weight_len: usize) -> Self {
        Self {
            id,
            weights: SparseBitSlicedTernary::new(weight_len),
            hypervector: None,
        }
    }

    pub fn with_weights(id: usize, weights: SparseBitSlicedTernary) -> Self {
        Self {
            id,
            weights,
            hypervector: None,
        }
    }

    /// Warm-start from a dense ternary weight slice (e.g. CalibModel.weights).
    /// Phase 5 prep: calib → GraphNode without re-encoding.
    pub fn from_ternary_weights(id: usize, weights: &[i8]) -> Self {
        Self::with_weights(id, SparseBitSlicedTernary::from_slice(weights))
    }

    /// Attach a hypervector for structural memory.
    pub fn with_hypervector(mut self, hv: HyperVector) -> Self {
        self.hypervector = Some(hv);
        self
    }

    /// Get the hypervector, creating a zero vector if none exists.
    pub fn hypervector_or_zero(&self) -> HyperVector {
        self.hypervector.clone().unwrap_or_else(HyperVector::zero)
    }

    /// Ledgered weight update; may trigger lazy sparse compact.
    pub fn update_weight_and_adapt(
        &mut self,
        idx: usize,
        val: i8,
        ledger: &mut TamperEvidentLedger,
    ) -> Result<u64, NtgError> {
        if idx >= self.weights.len() {
            return Err(NtgError::IndexOutOfBounds {
                index: idx,
                len: self.weights.len(),
            });
        }
        if !(-1..=1).contains(&val) {
            return Err(NtgError::InvalidTernaryValue(val));
        }
        self.weights.apply_structural_mutation(idx, val, ledger)
    }

    /// Sparsity fitness hook for Phase 3 Reflexive Fitness Evaluators.
    pub fn fitness_signal(&self) -> f32 {
        self.weights.fitness_signal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_weight_logs_to_ledger() -> Result<(), NtgError> {
        let mut node = GraphNode::new(0, 128);
        let mut ledger = TamperEvidentLedger::new(None)?;
        node.update_weight_and_adapt(0, 1, &mut ledger)?;
        assert_eq!(node.weights.get(0), 1);
        ledger.verify_full_ledger()?;
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_weight() {
        let mut node = GraphNode::new(0, 8);
        let mut ledger = TamperEvidentLedger::new(None).unwrap();
        assert!(node.update_weight_and_adapt(99, 1, &mut ledger).is_err());
    }

    #[test]
    fn from_ternary_weights_preserves_nonzero() {
        let w = vec![1i8, 0, -1, 0, 1];
        let node = GraphNode::from_ternary_weights(3, &w);
        assert_eq!(node.id, 3);
        assert_eq!(node.weights.get(0), 1);
        assert_eq!(node.weights.get(2), -1);
        assert_eq!(node.weights.get(4), 1);
    }

    #[test]
    fn hypervector_or_zero_returns_zero_when_none() {
        let node = GraphNode::new(0, 128);
        let hv = node.hypervector_or_zero();
        assert_eq!(hv.similarity(&hv), 0);
    }

    #[test]
    fn with_hypervector_attaches_and_returns() {
        let hv = HyperVector::random();
        let node = GraphNode::new(0, 128).with_hypervector(hv.clone());
        assert_eq!(node.hypervector.as_ref().unwrap(), &hv);
    }
}
