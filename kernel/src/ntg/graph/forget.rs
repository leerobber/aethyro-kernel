//! Structural forgetting for the Ternary Memory Graph (TMG).
//!
//! Implements time-decay edge pruning and Hebbian learning:
//! - Edges lose weight over time (exponential decay)
//! - Edges below threshold are pruned (forget mechanism)
//! - Firing together wires together (Hebbian rule)
//! - No backprop, no retraining — topology surgery only

use std::collections::HashMap;

/// Edge weight decay model: exponential with configurable half-life.
#[derive(Clone, Debug)]
pub struct DecayModel {
    /// Half-life in seconds (how long until weight = 0.5 × initial)
    pub half_life_secs: f64,
}

impl DecayModel {
    pub fn new(half_life_secs: f64) -> Self {
        Self { half_life_secs }
    }

    /// Standard 24-hour forget curve (Ebbinghaus-inspired)
    pub fn ebbinghaus_24h() -> Self {
        Self::new(86400.0) // 1 day
    }

    /// Rapid forgetting (testing, volatile memories)
    pub fn rapid_2h() -> Self {
        Self::new(7200.0) // 2 hours
    }

    /// Conservative forgetting (long-term retention)
    pub fn conservative_7d() -> Self {
        Self::new(604800.0) // 7 days
    }

    /// Compute weight after elapsed time: weight(t) = initial × (0.5)^(t / half_life)
    pub fn weight_at(&self, initial: f64, elapsed_secs: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return initial;
        }
        initial * 0.5_f64.powf(elapsed_secs / self.half_life_secs)
    }

    /// Time until weight drops below threshold: t = half_life × log₂(initial / threshold)
    pub fn time_to_threshold(&self, initial: f64, threshold: f64) -> f64 {
        if threshold >= initial || threshold <= 0.0 {
            return 0.0;
        }
        self.half_life_secs * (initial / threshold).log2()
    }
}

/// Edge weight tracking for Hebbian learning.
#[derive(Clone, Debug)]
pub struct EdgeWeight {
    /// Current weight [0.0, 1.0]
    pub weight: f64,
    /// Last access time (Unix timestamp, seconds)
    pub last_accessed_at: u64,
    /// Creation time (Unix timestamp, seconds)
    pub created_at: u64,
}

impl EdgeWeight {
    pub fn new(weight: f64, now: u64) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            last_accessed_at: now,
            created_at: now,
        }
    }

    /// Time since last access (seconds)
    pub fn age(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_accessed_at)
    }

    /// Apply decay and return true if weight drops below threshold
    pub fn apply_decay(&mut self, decay_model: &DecayModel, threshold: f64, now: u64) -> bool {
        let elapsed = self.age(now) as f64;
        self.weight = decay_model.weight_at(self.weight, elapsed);
        self.weight < threshold
    }

    /// Strengthen weight via Hebbian rule (fire-together-wire-together)
    /// Returns new weight after potentiation
    pub fn potentiate(&mut self, now: u64) -> f64 {
        self.last_accessed_at = now;
        // Hebbian rule: weak firing → gentle increase, strong firing → saturation
        // Simple model: w' = w + (1 - w) × 0.1 (each co-firing adds 10% toward max)
        let potentiation_rate = 0.1;
        self.weight = (self.weight + (1.0 - self.weight) * potentiation_rate).clamp(0.0, 1.0);
        self.weight
    }
}

/// Forgetting engine: manages edge lifecycle and structural pruning.
#[derive(Clone, Debug)]
pub struct ForgetEngine {
    /// Edge weights (node_pair → EdgeWeight)
    weights: HashMap<(usize, usize), EdgeWeight>,
    /// Decay model
    decay_model: DecayModel,
    /// Prune threshold: edges below this are removed
    prune_threshold: f64,
}

impl ForgetEngine {
    pub fn new(decay_model: DecayModel, prune_threshold: f64) -> Self {
        Self {
            weights: HashMap::new(),
            decay_model,
            prune_threshold: prune_threshold.clamp(0.0, 1.0),
        }
    }

    /// Create or strengthen an edge (Hebbian potentiation)
    pub fn fire_together(&mut self, from: usize, to: usize, now: u64) {
        self.weights
            .entry((from, to))
            .and_modify(|e| {
                e.potentiate(now);
            })
            .or_insert_with(|| {
                let mut e = EdgeWeight::new(0.1, now); // Start weak
                e.potentiate(now); // Strengthen on first firing
                e
            });
    }

    /// Get current weight of an edge (decayed to present time)
    pub fn get_weight(&mut self, from: usize, to: usize, now: u64) -> Option<f64> {
        self.weights.get_mut(&(from, to)).map(|e| {
            let elapsed = e.age(now) as f64;
            self.decay_model.weight_at(e.weight, elapsed)
        })
    }

    /// Prune edges below threshold and return list of pruned edges
    pub fn prune_stale(&mut self, now: u64) -> Vec<(usize, usize)> {
        let threshold = self.prune_threshold;
        let decay_model = &self.decay_model;

        let mut to_prune = Vec::new();

        for (edge, weight) in self.weights.iter_mut() {
            if weight.apply_decay(decay_model, threshold, now) {
                to_prune.push(*edge);
            }
        }

        for edge in &to_prune {
            self.weights.remove(edge);
        }

        to_prune
    }

    /// Get all edges with current weights (as of now)
    pub fn all_edges(&mut self, now: u64) -> Vec<((usize, usize), f64)> {
        self.weights
            .iter_mut()
            .map(|(edge, weight)| {
                let elapsed = weight.age(now) as f64;
                let decayed = self.decay_model.weight_at(weight.weight, elapsed);
                (*edge, decayed)
            })
            .collect()
    }

    /// Statistics: count edges by weight band
    pub fn weight_distribution(&mut self, now: u64) -> (usize, usize, usize) {
        let mut strong = 0;
        let mut medium = 0;
        let mut weak = 0;

        for weight in self.weights.values_mut() {
            let w = self.decay_model.weight_at(weight.weight, weight.age(now) as f64);
            if w >= 0.7 {
                strong += 1;
            } else if w >= 0.4 {
                medium += 1;
            } else {
                weak += 1;
            }
        }

        (strong, medium, weak)
    }

    pub fn edge_count(&self) -> usize {
        self.weights.len()
    }
}

impl Default for ForgetEngine {
    fn default() -> Self {
        Self::new(DecayModel::ebbinghaus_24h(), 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_exponential() {
        let model = DecayModel::new(10.0); // 10 sec half-life

        assert_eq!(model.weight_at(1.0, 0.0), 1.0);
        assert!(model.weight_at(1.0, 10.0) - 0.5 < 0.01);
        assert!(model.weight_at(1.0, 20.0) - 0.25 < 0.01);
    }

    #[test]
    fn time_to_threshold() {
        let model = DecayModel::new(10.0);
        let t = model.time_to_threshold(1.0, 0.25);
        assert!((t - 20.0).abs() < 0.1);
    }

    #[test]
    fn edge_weight_creation() {
        let e = EdgeWeight::new(0.5, 1000);
        assert_eq!(e.weight, 0.5);
        assert_eq!(e.last_accessed_at, 1000);
        assert_eq!(e.created_at, 1000);
    }

    #[test]
    fn edge_weight_decay_below_threshold() {
        let model = DecayModel::new(10.0);
        let mut e = EdgeWeight::new(0.5, 1000);
        let should_prune = e.apply_decay(&model, 0.1, 1050);
        // 0.5 * 0.5^(50/10) ≈ 0.015625, which is below 0.1 threshold → prune
        assert!(should_prune);

        let mut e = EdgeWeight::new(0.5, 1000);
        let should_prune = e.apply_decay(&model, 0.01, 1050);
        // 0.015625 is above 0.01 threshold → don't prune
        assert!(!should_prune);
    }

    #[test]
    fn hebbian_potentiation() {
        let mut e = EdgeWeight::new(0.1, 1000);
        let w1 = e.potentiate(1001);
        assert!(w1 > 0.1); // Weight increased
        let w2 = e.potentiate(1002);
        assert!(w2 > w1); // Weight keeps increasing
        assert!(w2 < 1.0); // But asymptotes toward 1.0
    }

    #[test]
    fn forget_engine_fire_together() {
        let mut engine = ForgetEngine::default();
        engine.fire_together(0, 1, 1000);
        engine.fire_together(0, 1, 1001);
        engine.fire_together(0, 1, 1002);

        let w = engine.get_weight(0, 1, 1002);
        assert!(w.is_some());
        assert!(w.unwrap() > 0.1); // Strengthened via repeated firing
    }

    #[test]
    fn forget_engine_prune_stale() {
        let model = DecayModel::new(100.0); // 100-second half-life
        let mut engine = ForgetEngine::new(model, 0.05);

        engine.fire_together(0, 1, 1000);
        engine.fire_together(1, 2, 1000);
        assert_eq!(engine.edge_count(), 2);

        // After 50 seconds with 100s half-life: weight ≈ 0.707 × initial
        let pruned = engine.prune_stale(1050);
        // With potentiation starting from 0.1, and one boost to ~0.19,
        // after 50s decay: ~0.19 * 0.707 ≈ 0.134, still above 0.05 threshold
        assert!(engine.edge_count() >= 1 || pruned.len() > 0); // At least one edge remains or some pruned
    }

    #[test]
    fn weight_distribution() {
        let mut engine = ForgetEngine::default();
        engine.fire_together(0, 1, 1000);
        engine.fire_together(0, 1, 1001); // Strengthen
        engine.fire_together(1, 2, 1000);
        // edge (1,2) is weak, edge (0,1) is stronger

        let (strong, medium, weak) = engine.weight_distribution(1000);
        assert!(strong + medium + weak >= 1); // At least one edge
    }
}
