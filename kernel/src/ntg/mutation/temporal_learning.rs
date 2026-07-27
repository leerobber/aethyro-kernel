//! Temporal learning: track how mutation effectiveness changes over time.
//!
//! Phase 6.10 temporal component tracks:
//! 1. Time-series effectiveness: Does a mutation stay effective or degrade?
//! 2. Phase-specific patterns: Which mutations work best early vs late in optimization?
//! 3. Temporal decay: How fast does novelty value decay with repetition?
//! 4. Seasonal patterns: Do certain mutations work better in certain efficiency regimes?

use super::adaptive::Domain;
use super::rules::MutationRuleKind;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Time-series snapshot of mutation effectiveness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalMutationRecord {
    /// Cycle when this mutation was evaluated
    pub cycle: u64,
    /// Domain it was applied in
    pub domain: Domain,
    /// The mutation kind
    pub mutation: MutationRuleKind,
    /// Efficiency before mutation
    pub efficiency_before: f64,
    /// Efficiency after mutation
    pub efficiency_after: f64,
    /// Was it accepted?
    pub accepted: bool,
    /// Baseline efficiency at time of mutation
    pub baseline_efficiency: f64,
}

/// Phase classification based on efficiency trajectory.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OptimizationPhase {
    /// Early phase: low baseline efficiency, high improvement potential
    Early,
    /// Mid phase: moderate efficiency, mutations have moderate effect
    Mid,
    /// Late phase: high efficiency, improvements harder to find
    Late,
    /// Plateau: efficiency stagnant, need exploration
    Plateau,
}

impl OptimizationPhase {
    /// Classify phase based on baseline efficiency and improvement trend.
    pub fn from_efficiency(baseline: f64, recent_improvement: f64) -> Self {
        match (baseline, recent_improvement) {
            (b, _) if b < 0.4 => OptimizationPhase::Early,
            (b, imp) if b >= 0.8 && imp < 0.01 => OptimizationPhase::Plateau,
            (b, _) if b >= 0.8 => OptimizationPhase::Late,
            _ => OptimizationPhase::Mid,
        }
    }
}

/// Temporal pattern for a mutation in a specific phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Mutation kind
    pub mutation_desc: String,
    /// Phase when this pattern was observed
    pub phase: OptimizationPhase,
    /// Effectiveness in this phase (0.0-1.0)
    pub effectiveness: f64,
    /// How many times observed in this phase
    pub observations: u32,
    /// Time-decay coefficient (how quickly novelty fades)
    pub decay_rate: f64,
}

/// Temporal learning engine: tracks effectiveness over time.
pub struct TemporalLearningEngine {
    /// History of all mutations with timestamps
    records: Vec<TemporalMutationRecord>,
    /// Per-mutation, per-phase effectiveness data
    phase_effectiveness: HashMap<(String, OptimizationPhase), TemporalPattern>,
    /// Early/mid/late phase effectiveness summary
    phase_summary: HashMap<OptimizationPhase, f64>,
    /// Current optimization phase
    current_phase: OptimizationPhase,
}

impl TemporalLearningEngine {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            phase_effectiveness: HashMap::new(),
            phase_summary: HashMap::new(),
            current_phase: OptimizationPhase::Early,
        }
    }

    /// Record a mutation evaluation with timing information.
    pub fn record_mutation(
        &mut self,
        cycle: u64,
        domain: Domain,
        mutation: MutationRuleKind,
        efficiency_before: f64,
        efficiency_after: f64,
        accepted: bool,
        baseline_efficiency: f64,
    ) {
        let record = TemporalMutationRecord {
            cycle,
            domain,
            mutation: mutation.clone(),
            efficiency_before,
            efficiency_after,
            accepted,
            baseline_efficiency,
        };

        self.records.push(record.clone());

        // Update phase classification
        let improvement = efficiency_after - efficiency_before;
        self.current_phase = OptimizationPhase::from_efficiency(baseline_efficiency, improvement);

        // Update phase-specific effectiveness
        let mutation_desc = mutation.description();
        let key = (mutation_desc.clone(), self.current_phase.clone());

        let pattern = self.phase_effectiveness.entry(key).or_insert(TemporalPattern {
            mutation_desc,
            phase: self.current_phase.clone(),
            effectiveness: 0.0,
            observations: 0,
            decay_rate: 0.1,
        });

        pattern.observations += 1;
        if accepted {
            pattern.effectiveness = (pattern.effectiveness * (pattern.observations - 1) as f64 + 1.0)
                / pattern.observations as f64;
        } else {
            pattern.effectiveness = (pattern.effectiveness * (pattern.observations - 1) as f64)
                / pattern.observations as f64;
        }
    }

    /// Get best mutations for a specific optimization phase.
    pub fn best_mutations_for_phase(
        &self,
        phase: OptimizationPhase,
        top_k: usize,
    ) -> Vec<(String, f64)> {
        let mut mutations: Vec<_> = self
            .phase_effectiveness
            .iter()
            .filter(|(key, _)| key.1 == phase)
            .map(|(key, pattern)| (key.0.clone(), pattern.effectiveness))
            .collect();

        mutations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        mutations.into_iter().take(top_k).collect()
    }

    /// Measure temporal decay of mutation effectiveness.
    /// Returns decay rate (0.0-1.0) where 1.0 means no decay, 0.0 means rapid decay.
    pub fn mutation_temporal_decay(&self, mutation_desc: &str) -> f64 {
        let records: Vec<_> = self
            .records
            .iter()
            .filter(|r| r.mutation.description() == mutation_desc)
            .collect();

        if records.len() < 2 {
            return 0.5; // neutral if insufficient data
        }

        // Compare effectiveness in early vs recent cycles
        let early_cycles = records.iter().take(records.len() / 2);
        let late_cycles = records.iter().skip(records.len() / 2);

        let early_success = early_cycles.filter(|r| r.accepted).count() as f64 / (records.len() / 2) as f64;
        let late_success = late_cycles.filter(|r| r.accepted).count() as f64 / (records.len() / 2) as f64;

        if early_success == 0.0 {
            0.0
        } else {
            late_success / early_success
        }
    }

    /// Get summary statistics for a phase.
    pub fn phase_statistics(&self, phase: OptimizationPhase) -> (u32, f64, f64) {
        let phase_records: Vec<_> = self
            .records
            .iter()
            .filter(|r| OptimizationPhase::from_efficiency(r.baseline_efficiency, r.efficiency_after - r.efficiency_before) == phase)
            .collect();

        if phase_records.is_empty() {
            return (0, 0.0, 0.0);
        }

        let count = phase_records.len() as u32;
        let success_rate = phase_records.iter().filter(|r| r.accepted).count() as f64 / count as f64;
        let avg_improvement = phase_records
            .iter()
            .map(|r| (r.efficiency_after - r.efficiency_before).max(0.0))
            .sum::<f64>()
            / count as f64;

        (count, success_rate, avg_improvement)
    }

    /// Generate temporal learning report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Temporal Learning Report ===\n");
        report.push_str(&format!("Total mutations recorded: {}\n", self.records.len()));
        report.push_str(&format!("Current phase: {:?}\n", self.current_phase));

        report.push_str("\nPhase-specific effectiveness:\n");
        for phase in &[OptimizationPhase::Early, OptimizationPhase::Mid, OptimizationPhase::Late, OptimizationPhase::Plateau] {
            let (count, success, improvement) = self.phase_statistics(phase.clone());
            if count > 0 {
                report.push_str(&format!(
                    "  {:?}: {} mutations, {:.1}% success, {:.3} avg improvement\n",
                    phase,
                    count,
                    success * 100.0,
                    improvement
                ));
            }
        }

        report.push_str("\nMutation decay analysis (top 5):\n");
        let mut decays: Vec<_> = self
            .records
            .iter()
            .map(|r| (r.mutation.description(), 1.0))
            .collect();
        decays.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (mutation, _) in decays.iter().take(5) {
            let decay = self.mutation_temporal_decay(mutation);
            report.push_str(&format!("  {}: {:.2} retention\n", mutation, decay));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_record_stores_mutation_data() {
        let mut engine = TemporalLearningEngine::new();
        engine.record_mutation(1, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.5, 0.6, true, 0.5);
        assert_eq!(engine.records.len(), 1);
        assert_eq!(engine.records[0].cycle, 1);
        assert_eq!(engine.records[0].accepted, true);
    }

    #[test]
    fn temporal_phase_classification() {
        assert_eq!(OptimizationPhase::from_efficiency(0.2, 0.1), OptimizationPhase::Early);
        assert_eq!(OptimizationPhase::from_efficiency(0.9, 0.001), OptimizationPhase::Plateau);
        assert_eq!(OptimizationPhase::from_efficiency(0.85, 0.02), OptimizationPhase::Late);
        assert_eq!(OptimizationPhase::from_efficiency(0.5, 0.05), OptimizationPhase::Mid);
    }

    #[test]
    fn temporal_phase_statistics() {
        let mut engine = TemporalLearningEngine::new();
        engine.record_mutation(1, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.2, 0.3, true, 0.2);
        engine.record_mutation(2, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.3, 0.35, true, 0.3);

        let (count, success, improvement) = engine.phase_statistics(OptimizationPhase::Early);
        assert!(count > 0);
        assert!(success > 0.0);
        assert!(improvement > 0.0);
    }

    #[test]
    fn temporal_best_mutations_for_phase() {
        let mut engine = TemporalLearningEngine::new();
        for i in 0..3 {
            engine.record_mutation(i, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.2, 0.3, true, 0.2);
        }

        let best = engine.best_mutations_for_phase(OptimizationPhase::Early, 5);
        assert!(!best.is_empty());
    }

    #[test]
    fn temporal_decay_calculation() {
        let mut engine = TemporalLearningEngine::new();
        for i in 0..4 {
            let accepted = i < 2; // First half accepted, second half rejected
            engine.record_mutation(i, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.5, 0.6, accepted, 0.5);
        }

        let decay = engine.mutation_temporal_decay("add_node(label='test')");
        assert!(decay >= 0.0 && decay <= 1.0);
    }

    #[test]
    fn temporal_report_is_readable() {
        let mut engine = TemporalLearningEngine::new();
        engine.record_mutation(1, Domain::Generic, MutationRuleKind::AddNode { label: "test".to_string() }, 0.5, 0.6, true, 0.5);

        let report = engine.report();
        assert!(report.contains("Temporal Learning Report"));
        assert!(report.contains("Total mutations recorded"));
    }
}
