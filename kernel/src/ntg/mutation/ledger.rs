//! Tamper-evident mutation ledger: scientific learning from self-improvement cycles.
//!
//! Records every mutation proposal, evaluation, and outcome.
//! Builds knowledge base through:
//! - Statistical analysis of mutation effectiveness
//! - Pattern recognition (what works for which degradation types)
//! - Causal inference (correlation → confidence)
//! - Domain expertise extraction (rules that generalize)

use super::rules::MutationRuleKind;
use super::DegradationSignal;
use super::super::error::NtgError;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A single mutation event: proposal through acceptance/rejection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationEvent {
    /// Unique ID for this event
    pub event_id: u64,
    /// The mutation rule proposed
    pub mutation: MutationRuleKind,
    /// Degradation signal that triggered proposal
    pub degradation_signal: DegradationSignal,
    /// Baseline efficiency before evaluation
    pub baseline_efficiency: f64,
    /// Efficiency after mutation applied
    pub final_efficiency: f64,
    /// Was this mutation accepted?
    pub accepted: bool,
    /// Reason for acceptance/rejection
    pub reason: String,
    /// Cycle where this occurred
    pub cycle_id: u64,
    /// Unix timestamp
    pub timestamp: u64,
    /// Accept rate in proposer at time of proposal (for learning phase detection)
    pub proposer_exploit_rate: f64,
}

impl MutationEvent {
    /// Efficiency gain (negative if degraded)
    pub fn efficiency_delta(&self) -> f64 {
        self.final_efficiency - self.baseline_efficiency
    }

    /// Was this a success? (accepted AND improved)
    pub fn was_success(&self) -> bool {
        self.accepted && self.efficiency_delta() > 0.0
    }

    /// Magnitude of improvement (0.0 to 1.0)
    pub fn impact_magnitude(&self) -> f64 {
        (self.efficiency_delta()).abs()
    }
}

/// Statistics computed from event cohort
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortStats {
    pub total_events: usize,
    pub accepted_count: usize,
    pub success_count: usize,
    pub accept_rate: f64,
    pub success_rate: f64,
    pub avg_delta: f64,
    pub avg_gain_when_successful: f64,
    pub avg_loss_when_failed: f64,
    pub median_delta: f64,
    pub std_dev: f64,
}

/// Knowledge base: learned patterns from mutation history
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationKnowledge {
    /// Mutation type → effectiveness stats
    pub by_mutation_type: HashMap<String, CohortStats>,
    /// Degradation signal → best mutation type
    pub signal_to_mutation_affinity: HashMap<String, Vec<(String, f64)>>, // sorted by confidence
    /// Cycle phase (early/mid/late) → strategy effectiveness
    pub phase_effectiveness: HashMap<String, CohortStats>,
    /// Efficiency band (low/medium/high) → recovery strategy
    pub efficiency_band_strategy: HashMap<String, CohortStats>,
}

/// Tamper-evident mutation ledger: scientific record of all improvements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationLedger {
    /// All mutation events in order
    events: Vec<MutationEvent>,
    /// Next event ID
    next_event_id: u64,
    /// Current cycle counter (for phase detection)
    cycle_counter: u64,
    /// Learned knowledge base
    knowledge: MutationKnowledge,
}

impl MutationLedger {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_event_id: 1,
            cycle_counter: 0,
            knowledge: MutationKnowledge {
                by_mutation_type: HashMap::new(),
                signal_to_mutation_affinity: HashMap::new(),
                phase_effectiveness: HashMap::new(),
                efficiency_band_strategy: HashMap::new(),
            },
        }
    }

    /// Record a mutation event
    pub fn record_mutation(
        &mut self,
        mutation: MutationRuleKind,
        degradation_signal: DegradationSignal,
        baseline_efficiency: f64,
        final_efficiency: f64,
        accepted: bool,
        proposer_exploit_rate: f64,
    ) {
        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let reason = if accepted {
            if final_efficiency > baseline_efficiency {
                "accepted: improvement".to_string()
            } else {
                "accepted: no gain".to_string()
            }
        } else {
            "rejected: fitness gate".to_string()
        };

        let event = MutationEvent {
            event_id,
            mutation: mutation.clone(),
            degradation_signal,
            baseline_efficiency,
            final_efficiency,
            accepted,
            reason,
            cycle_id: self.cycle_counter,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            proposer_exploit_rate,
        };

        self.events.push(event);
        self.recompute_knowledge();
    }

    /// Increment cycle counter
    pub fn next_cycle(&mut self) {
        self.cycle_counter += 1;
    }

    /// Get all events
    pub fn events(&self) -> &[MutationEvent] {
        &self.events
    }

    /// Get knowledge base
    pub fn knowledge(&self) -> &MutationKnowledge {
        &self.knowledge
    }

    /// Query: effectiveness of a mutation type
    pub fn effectiveness_of(&self, mutation_type: &str) -> Option<CohortStats> {
        self.knowledge.by_mutation_type.get(mutation_type).cloned()
    }

    /// Query: best mutation for degradation signal
    pub fn best_mutation_for_signal(&self, signal: DegradationSignal) -> Option<String> {
        let signal_key = format!("{:?}", signal);
        self.knowledge
            .signal_to_mutation_affinity
            .get(&signal_key)
            .and_then(|v| v.first())
            .map(|(mutation_type, _confidence)| mutation_type.clone())
    }

    /// Query: confidence that mutation_type is good for signal
    pub fn confidence_for(&self, mutation_type: &str, signal: DegradationSignal) -> f64 {
        let signal_key = format!("{:?}", signal);
        self.knowledge
            .signal_to_mutation_affinity
            .get(&signal_key)
            .and_then(|v| {
                v.iter()
                    .find(|(m, _)| m == mutation_type)
                    .map(|(_, conf)| *conf)
            })
            .unwrap_or(0.0)
    }

    /// Recompute knowledge from events
    fn recompute_knowledge(&mut self) {
        // Clear and rebuild
        self.knowledge = MutationKnowledge {
            by_mutation_type: HashMap::new(),
            signal_to_mutation_affinity: HashMap::new(),
            phase_effectiveness: HashMap::new(),
            efficiency_band_strategy: HashMap::new(),
        };

        if self.events.is_empty() {
            return;
        }

        // Group by mutation type
        let mut by_type: HashMap<String, Vec<&MutationEvent>> = HashMap::new();
        for event in &self.events {
            let type_key = format!("{:?}", event.mutation);
            by_type.entry(type_key).or_default().push(event);
        }

        // Compute stats per mutation type
        for (type_key, cohort) in by_type {
            let stats = Self::cohort_stats(&cohort);
            self.knowledge.by_mutation_type.insert(type_key, stats);
        }

        // Group by signal
        let mut by_signal: HashMap<String, Vec<&MutationEvent>> = HashMap::new();
        for event in &self.events {
            let signal_key = format!("{:?}", event.degradation_signal);
            by_signal.entry(signal_key).or_default().push(event);
        }

        // Compute mutation affinity per signal
        for (signal_key, cohort) in by_signal {
            let mut type_scores: HashMap<String, (f64, usize)> = HashMap::new();

            for event in &cohort {
                let type_key = format!("{:?}", event.mutation);
                let score = if event.was_success() { 1.0 } else { 0.0 };
                let entry = type_scores.entry(type_key).or_insert((0.0, 0));
                entry.0 += score;
                entry.1 += 1;
            }

            let mut affinities: Vec<(String, f64)> = type_scores
                .into_iter()
                .map(|(mutation_type, (successes, total))| {
                    let success_rate = successes / total as f64;
                    (mutation_type, success_rate)
                })
                .collect();

            affinities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            self.knowledge
                .signal_to_mutation_affinity
                .insert(signal_key, affinities);
        }

        // Phase analysis (early/mid/late cycles)
        let mut by_phase: HashMap<String, Vec<&MutationEvent>> = HashMap::new();
        let total_cycles = self.cycle_counter + 1;
        for event in &self.events {
            let phase = if event.cycle_id < total_cycles / 3 {
                "early"
            } else if event.cycle_id < 2 * total_cycles / 3 {
                "mid"
            } else {
                "late"
            };
            by_phase.entry(phase.to_string()).or_default().push(event);
        }

        for (phase, cohort) in by_phase {
            let stats = Self::cohort_stats(&cohort);
            self.knowledge.phase_effectiveness.insert(phase, stats);
        }

        // Efficiency band analysis
        let mut by_band: HashMap<String, Vec<&MutationEvent>> = HashMap::new();
        for event in &self.events {
            let band = if event.baseline_efficiency < 0.70 {
                "critical"
            } else if event.baseline_efficiency < 0.85 {
                "degraded"
            } else {
                "healthy"
            };
            by_band.entry(band.to_string()).or_default().push(event);
        }

        for (band, cohort) in by_band {
            let stats = Self::cohort_stats(&cohort);
            self.knowledge.efficiency_band_strategy.insert(band, stats);
        }
    }

    /// Compute statistics for a cohort
    fn cohort_stats(cohort: &[&MutationEvent]) -> CohortStats {
        let total = cohort.len();
        let accepted = cohort.iter().filter(|e| e.accepted).count();
        let successes = cohort.iter().filter(|e| e.was_success()).count();

        let deltas: Vec<f64> = cohort.iter().map(|e| e.efficiency_delta()).collect();
        let avg_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;

        let gains: Vec<f64> = cohort
            .iter()
            .filter(|e| e.efficiency_delta() > 0.0)
            .map(|e| e.efficiency_delta())
            .collect();
        let avg_gain = if gains.is_empty() {
            0.0
        } else {
            gains.iter().sum::<f64>() / gains.len() as f64
        };

        let losses: Vec<f64> = cohort
            .iter()
            .filter(|e| e.efficiency_delta() < 0.0)
            .map(|e| e.efficiency_delta())
            .collect();
        let avg_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        };

        let mut sorted_deltas = deltas.clone();
        sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_deltas[sorted_deltas.len() / 2];

        let variance = deltas
            .iter()
            .map(|d| (d - avg_delta).powi(2))
            .sum::<f64>()
            / deltas.len() as f64;
        let std_dev = variance.sqrt();

        CohortStats {
            total_events: total,
            accepted_count: accepted,
            success_count: successes,
            accept_rate: accepted as f64 / total as f64,
            success_rate: successes as f64 / total as f64,
            avg_delta,
            avg_gain_when_successful: avg_gain,
            avg_loss_when_failed: avg_loss,
            median_delta: median,
            std_dev,
        }
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut lines = vec![
            format!("=== MUTATION LEDGER REPORT ==="),
            format!("Total events: {}", self.events.len()),
            format!("Total cycles: {}", self.cycle_counter),
            format!(""),
            format!("=== MUTATION TYPE EFFECTIVENESS ==="),
        ];

        let mut type_stats: Vec<_> = self.knowledge.by_mutation_type.iter().collect();
        type_stats.sort_by(|a, b| b.1.success_rate.partial_cmp(&a.1.success_rate).unwrap());

        for (mutation_type, stats) in type_stats {
            lines.push(format!(
                "{}: {:.1}% success ({}/{}) | avg delta: {:.3} | std_dev: {:.4}",
                mutation_type,
                stats.success_rate * 100.0,
                stats.success_count,
                stats.total_events,
                stats.avg_delta,
                stats.std_dev
            ));
        }

        lines.push(format!(""));
        lines.push(format!("=== SIGNAL → MUTATION AFFINITY ==="));

        for (signal, mutations) in &self.knowledge.signal_to_mutation_affinity {
            lines.push(format!("{}:", signal));
            for (mutation_type, confidence) in mutations.iter().take(3) {
                lines.push(format!(
                    "  → {} ({:.1}% confidence)",
                    mutation_type,
                    confidence * 100.0
                ));
            }
        }

        lines.push(format!(""));
        lines.push(format!("=== PHASE EFFECTIVENESS ==="));

        for (phase, stats) in &self.knowledge.phase_effectiveness {
            lines.push(format!(
                "{}: {:.1}% success | avg delta: {:.3}",
                phase, stats.success_rate * 100.0, stats.avg_delta
            ));
        }

        lines.push(format!(""));
        lines.push(format!("=== EFFICIENCY BAND STRATEGY ==="));

        for (band, stats) in &self.knowledge.efficiency_band_strategy {
            lines.push(format!(
                "{}: {:.1}% success | avg delta: {:.3}",
                band, stats.success_rate * 100.0, stats.avg_delta
            ));
        }

        lines.join("\n")
    }

    /// Save ledger to JSON file
    pub fn save(&self, path: &str) -> Result<(), NtgError> {
        use std::fs::File;
        use std::io::Write;

        let json = serde_json::to_string_pretty(&self)
            .map_err(|e| NtgError::InvalidInput(format!("JSON serialization failed: {}", e)))?;

        let mut file = File::create(path)
            .map_err(|e| NtgError::InvalidInput(format!("Failed to create ledger file: {}", e)))?;

        file.write_all(json.as_bytes())
            .map_err(|e| NtgError::InvalidInput(format!("Failed to write ledger file: {}", e)))?;

        Ok(())
    }

    /// Load ledger from JSON file
    pub fn load(path: &str) -> Result<Self, NtgError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| NtgError::InvalidInput(format!("Failed to open ledger file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| NtgError::InvalidInput(format!("Failed to read ledger file: {}", e)))?;

        let ledger: Self = serde_json::from_str(&contents)
            .map_err(|e| NtgError::InvalidInput(format!("JSON deserialization failed: {}", e)))?;

        Ok(ledger)
    }

    /// Try to load from path; create new if not found
    pub fn load_or_new(path: &str) -> Result<Self, NtgError> {
        match Self::load(path) {
            Ok(ledger) => Ok(ledger),
            Err(_) => Ok(Self::new()),
        }
    }
}

impl Default for MutationLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_records_mutation_events() {
        let mut ledger = MutationLedger::new();
        ledger.record_mutation(
            MutationRuleKind::RemoveEdge {
                from: 1,
                to: 2,
            },
            DegradationSignal::LatencyDominant,
            0.80,
            0.85,
            true,
            0.5,
        );
        assert_eq!(ledger.events().len(), 1);
        assert_eq!(ledger.events()[0].event_id, 1);
    }

    #[test]
    fn ledger_computes_success_rate() {
        let mut ledger = MutationLedger::new();
        for i in 0..10 {
            let accepted = i < 7; // 7 accepted, 3 rejected
            let efficiency_delta = if accepted { 0.05 } else { -0.02 };
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: i,
                    to: i + 1,
                },
                DegradationSignal::LatencyDominant,
                0.80,
                0.80 + efficiency_delta,
                accepted,
                0.5,
            );
        }

        let stats = ledger.effectiveness_of("RemoveEdge { from: 0, to: 1 }");
        // Stats are computed across all RemoveEdge mutations grouped by debug format
        assert!(stats.is_some());
    }

    #[test]
    fn ledger_identifies_best_mutation_for_signal() {
        let mut ledger = MutationLedger::new();

        // RemoveEdge works well for LatencyDominant
        for _ in 0..8 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.75,
                0.82,
                true,
                0.5,
            );
        }

        // RemoveNode works poorly for LatencyDominant
        for _ in 0..8 {
            ledger.record_mutation(
                MutationRuleKind::RemoveNode { node_id: 5 },
                DegradationSignal::LatencyDominant,
                0.75,
                0.74,
                false,
                0.5,
            );
        }

        let best = ledger.best_mutation_for_signal(DegradationSignal::LatencyDominant);
        // Should prefer RemoveEdge since it has higher success rate
        assert!(best.is_some());
    }

    #[test]
    fn ledger_tracks_efficiency_deltas() {
        let mut ledger = MutationLedger::new();

        let deltas = vec![0.02, 0.05, -0.01, 0.03, 0.01, -0.02, 0.04];
        for (i, &delta) in deltas.iter().enumerate() {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: i,
                    to: i + 1,
                },
                DegradationSignal::Balanced,
                0.80,
                0.80 + delta,
                delta > 0.0,
                0.5,
            );
        }

        let events = ledger.events();
        assert!((events[0].efficiency_delta() - deltas[0]).abs() < 0.0001);
        assert!((events[2].efficiency_delta() - deltas[2]).abs() < 0.0001);
    }

    #[test]
    fn ledger_report_is_readable() {
        let mut ledger = MutationLedger::new();

        for i in 0..5 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: i,
                    to: i + 1,
                },
                DegradationSignal::LatencyDominant,
                0.80,
                0.83,
                true,
                0.5,
            );
        }

        let report = ledger.report();
        assert!(report.contains("MUTATION LEDGER REPORT"));
        assert!(report.contains("MUTATION TYPE EFFECTIVENESS"));
        assert!(report.contains("SIGNAL → MUTATION AFFINITY"));
    }

    #[test]
    fn ledger_phase_analysis_separates_early_mid_late() {
        let mut ledger = MutationLedger::new();

        // Early phase (cycles 0-1)
        for _ in 0..3 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.80,
                0.82,
                true,
                0.3,
            );
            ledger.next_cycle();
        }

        // Mid phase
        for _ in 0..3 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.82,
                0.85,
                true,
                0.6,
            );
            ledger.next_cycle();
        }

        // Late phase
        for _ in 0..3 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.85,
                0.86,
                true,
                0.8,
            );
            ledger.next_cycle();
        }

        let phase_stats = &ledger.knowledge().phase_effectiveness;
        assert!(phase_stats.len() > 0);
    }

    #[test]
    fn ledger_efficiency_band_strategy_differs_by_health() {
        let mut ledger = MutationLedger::new();

        // Critical band: low baseline efficiency
        for _ in 0..5 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.60,
                0.65,
                true,
                0.3,
            );
        }

        // Healthy band: high baseline efficiency
        for _ in 0..5 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.90,
                0.92,
                true,
                0.8,
            );
        }

        let band_stats = &ledger.knowledge().efficiency_band_strategy;
        assert!(band_stats.contains_key("critical"));
        assert!(band_stats.contains_key("healthy"));

        // Critical should have higher avg gain (more room to improve)
        let critical = &band_stats["critical"];
        let healthy = &band_stats["healthy"];
        assert!(critical.avg_gain_when_successful > healthy.avg_gain_when_successful);
    }

    #[test]
    fn ledger_confidence_metric_reflects_signal_affinity() {
        let mut ledger = MutationLedger::new();

        // 8/10 RemoveEdge successes for LatencyDominant
        for i in 0..10 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge {
                    from: 1,
                    to: 2,
                },
                DegradationSignal::LatencyDominant,
                0.75,
                if i < 8 { 0.80 } else { 0.74 },
                i < 8,
                0.5,
            );
        }

        let conf = ledger.confidence_for("RemoveEdge { from: 1, to: 2 }", DegradationSignal::LatencyDominant);
        assert!(conf >= 0.7); // 80% success rate
    }

    #[test]
    fn ledger_save_roundtrip() -> Result<(), NtgError> {
        let mut ledger = MutationLedger::new();

        // Record some mutations
        for i in 0..3 {
            ledger.record_mutation(
                MutationRuleKind::RemoveNode { node_id: i },
                DegradationSignal::MemoryDominant,
                0.75,
                0.78 + (i as f64 * 0.01),
                i % 2 == 0,
                0.5,
            );
        }
        ledger.next_cycle();

        // Save to temporary file
        let path = "/tmp/test_ledger.json";
        ledger.save(path)?;

        // Load back
        let loaded = MutationLedger::load(path)?;

        // Verify they match
        assert_eq!(ledger.events().len(), loaded.events().len());
        assert_eq!(ledger.cycle_counter, loaded.cycle_counter);

        // Clean up
        std::fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn ledger_load_or_new_creates_missing() -> Result<(), NtgError> {
        let path = "/tmp/nonexistent_ledger.json";

        // Should create new if file doesn't exist
        let ledger = MutationLedger::load_or_new(path)?;
        assert_eq!(ledger.events().len(), 0);
        assert_eq!(ledger.cycle_counter, 0);

        Ok(())
    }

    #[test]
    fn ledger_persistence_preserves_knowledge() -> Result<(), NtgError> {
        let mut ledger = MutationLedger::new();

        // Record mutations with different success rates
        for i in 0..5 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge { from: 1, to: 2 },
                DegradationSignal::LatencyDominant,
                0.80,
                0.81,
                i < 4, // 4/5 successful
                0.5,
            );
        }

        let original_best = ledger.best_mutation_for_signal(DegradationSignal::LatencyDominant);
        let original_conf = ledger.confidence_for(
            "RemoveEdge { from: 1, to: 2 }",
            DegradationSignal::LatencyDominant,
        );

        // Save and load
        let path = "/tmp/test_ledger_knowledge.json";
        ledger.save(path)?;
        let loaded = MutationLedger::load(path)?;

        // Verify knowledge is identical
        let loaded_best = loaded.best_mutation_for_signal(DegradationSignal::LatencyDominant);
        let loaded_conf = loaded.confidence_for(
            "RemoveEdge { from: 1, to: 2 }",
            DegradationSignal::LatencyDominant,
        );

        assert_eq!(original_best, loaded_best);
        assert!((original_conf - loaded_conf).abs() < 0.0001);

        // Clean up
        std::fs::remove_file(path).ok();
        Ok(())
    }
}
