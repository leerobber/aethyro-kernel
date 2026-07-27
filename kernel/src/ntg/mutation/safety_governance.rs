//! Autonomous safety & governance layer (Phase 6.12).
//!
//! Provides:
//! 1. Safety scoring: evaluate mutations against constraints and alignment
//! 2. Behavioral drift detection: identify when system diverges from core objectives
//! 3. Constraint enforcement: hard limits on modification scope and rate
//! 4. Rollback capability: revert to known-good states
//! 5. Audit trail: complete record of all decisions and modifications
//!
//! Critical for self-modifying systems to maintain alignment and prevent divergence.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Safety constraints for autonomous operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyConstraints {
    /// Maximum efficiency gain allowed per mutation (prevents over-optimization)
    pub max_efficiency_jump: f64, // 0.0-1.0
    /// Minimum acceptance rate threshold (prevents stagnation or recklessness)
    pub acceptance_rate_floor: f64,
    pub acceptance_rate_ceiling: f64,
    /// Maximum cumulative efficiency drop allowed across cycle
    pub max_efficiency_drop: f64,
    /// Maximum mutations per cycle
    pub max_mutations_per_cycle: u32,
    /// Behavioral drift threshold (0.0-1.0): deviation from historical patterns
    pub drift_threshold: f64,
    /// Minimum confidence in mutation effect
    pub min_effect_confidence: f64,
    /// Maximum time since last successful rollback (cycles)
    pub rollback_timeout_cycles: u32,
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_efficiency_jump: 0.3,
            acceptance_rate_floor: 0.1,
            acceptance_rate_ceiling: 0.9,
            max_efficiency_drop: 0.2,
            max_mutations_per_cycle: 10,
            drift_threshold: 0.4,
            min_effect_confidence: 0.5,
            rollback_timeout_cycles: 100,
        }
    }
}

/// Decision outcome from safety gate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SafetyDecision {
    Approved,
    Rejected(String), // reason for rejection
    Flagged(String),  // approved but flagged for monitoring
}

/// Safety score components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyScore {
    /// Constraint compliance (0.0-1.0)
    pub constraint_score: f64,
    /// Alignment with historical patterns (0.0-1.0)
    pub alignment_score: f64,
    /// Confidence in predicted effect (0.0-1.0)
    pub confidence_score: f64,
    /// Overall safety score (0.0-1.0)
    pub overall_score: f64,
    /// Why this score was assigned
    pub rationale: String,
}

/// Behavioral state snapshot for drift detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehavioralSnapshot {
    pub cycle: u64,
    pub efficiency: f64,
    pub acceptance_rate: f64,
    pub mutations_accepted: usize,
    pub avg_efficiency_gain_per_mutation: f64,
    pub mutation_types_used: Vec<String>,
    pub domain_focus: String,
}

/// Rollback checkpoint for recovery.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackCheckpoint {
    pub cycle: u64,
    pub timestamp: u64,
    pub efficiency: f64,
    pub reason: String,
    pub safe_state_hash: String,
}

/// Audit log entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    pub cycle: u64,
    pub timestamp: u64,
    pub mutation_id: String,
    pub decision: SafetyDecision,
    pub safety_score: f64,
    pub efficiency_before: f64,
    pub efficiency_after: f64,
    pub constraint_violations: Vec<String>,
}

/// Safety governance engine for autonomous systems.
pub struct SafetyGovernanceEngine {
    constraints: SafetyConstraints,
    behavioral_history: VecDeque<BehavioralSnapshot>,
    checkpoints: Vec<RollbackCheckpoint>,
    audit_log: Vec<AuditEntry>,
    cycle_count: u64,
    last_successful_cycle: u64,
    consecutive_rejections: u32,
}

impl SafetyGovernanceEngine {
    pub fn new(constraints: SafetyConstraints) -> Self {
        Self {
            constraints,
            behavioral_history: VecDeque::with_capacity(100),
            checkpoints: Vec::new(),
            audit_log: Vec::new(),
            cycle_count: 0,
            last_successful_cycle: 0,
            consecutive_rejections: 0,
        }
    }

    /// Evaluate mutation against safety constraints and alignment.
    pub fn evaluate_mutation_safety(
        &self,
        mutation_id: &str,
        efficiency_before: f64,
        efficiency_after: f64,
        confidence: f64,
        mutation_type: &str,
    ) -> SafetyScore {
        let mut violations = Vec::new();

        // Check 1: Efficiency jump constraint
        let efficiency_gain = efficiency_after - efficiency_before;
        let jump_violation = efficiency_gain > self.constraints.max_efficiency_jump;
        if jump_violation {
            violations.push(format!(
                "Efficiency jump too large: {:.2}% (max: {:.2}%)",
                efficiency_gain * 100.0,
                self.constraints.max_efficiency_jump * 100.0
            ));
        }

        // Check 2: Efficiency drop constraint
        let drop_violation = efficiency_gain < -self.constraints.max_efficiency_drop;
        if drop_violation {
            violations.push(format!(
                "Efficiency drop too large: {:.2}% (max: {:.2}%)",
                efficiency_gain.abs() * 100.0,
                self.constraints.max_efficiency_drop * 100.0
            ));
        }

        // Check 3: Confidence threshold
        let confidence_violation = confidence < self.constraints.min_effect_confidence;
        if confidence_violation {
            violations.push(format!(
                "Low confidence: {:.2}% (min: {:.2}%)",
                confidence * 100.0,
                self.constraints.min_effect_confidence * 100.0
            ));
        }

        // Constraint score: decrease for each violation (more severe penalty)
        let constraint_score = 1.0 - (violations.len() as f64 * 0.35).min(1.0);

        // Alignment score: based on alignment with historical patterns
        let alignment_score = self.compute_alignment_score(mutation_type, efficiency_gain);

        // Overall score: weighted combination
        let overall_score = (constraint_score * 0.4 + alignment_score * 0.3 + confidence * 0.3).max(0.0);

        let rationale = if violations.is_empty() {
            format!(
                "All constraints satisfied. Mutation '{}' with {:.1}% efficiency gain.",
                mutation_id,
                efficiency_gain * 100.0
            )
        } else {
            format!(
                "Constraint violations: {}",
                violations.join("; ")
            )
        };

        SafetyScore {
            constraint_score,
            alignment_score,
            confidence_score: confidence,
            overall_score,
            rationale,
        }
    }

    /// Detect behavioral drift from historical patterns.
    pub fn detect_behavioral_drift(&self, current_state: &BehavioralSnapshot) -> (bool, f64, String) {
        if self.behavioral_history.is_empty() {
            return (false, 0.0, "No historical data yet".to_string());
        }

        // Compare current state to recent historical average
        let recent_snapshots: Vec<_> = self.behavioral_history
            .iter()
            .rev()
            .take(10)
            .collect();

        let avg_acceptance_rate = recent_snapshots
            .iter()
            .map(|s| s.acceptance_rate)
            .sum::<f64>() / recent_snapshots.len() as f64;

        let avg_efficiency_gain = recent_snapshots
            .iter()
            .map(|s| s.avg_efficiency_gain_per_mutation)
            .sum::<f64>() / recent_snapshots.len() as f64;

        // Calculate drift as deviation from historical norm
        let acceptance_drift = (current_state.acceptance_rate - avg_acceptance_rate).abs();
        let efficiency_drift = (current_state.avg_efficiency_gain_per_mutation - avg_efficiency_gain).abs();

        let total_drift = (acceptance_drift * 0.5 + efficiency_drift.min(1.0) * 0.5).min(1.0);
        let is_drifting = total_drift > self.constraints.drift_threshold;

        let reason = if is_drifting {
            format!(
                "Behavioral drift detected: acceptance_rate divergence {:.1}%, efficiency_gain divergence {:.1}%",
                acceptance_drift * 100.0,
                efficiency_drift * 100.0
            )
        } else {
            "Behavior within expected range".to_string()
        };

        (is_drifting, total_drift, reason)
    }

    /// Gate mutation acceptance with safety checks.
    pub fn gate_mutation_acceptance(
        &mut self,
        _mutation_id: &str,
        proposed_acceptance: bool,
        safety_score: &SafetyScore,
        _efficiency_before: f64,
        _efficiency_after: f64,
    ) -> SafetyDecision {
        let reason = if safety_score.overall_score < 0.3 {
            format!(
                "Safety score too low: {:.2} < 0.30. {}",
                safety_score.overall_score,
                safety_score.rationale
            )
        } else if safety_score.overall_score < 0.6 && proposed_acceptance {
            // Approved but flagged
            return SafetyDecision::Flagged(format!(
                "Lower confidence: {:.2}. {}",
                safety_score.overall_score,
                safety_score.rationale
            ));
        } else if proposed_acceptance && safety_score.overall_score >= 0.6 {
            // Fully approved
            return SafetyDecision::Approved;
        } else {
            "Proposed rejection by fitness gate".to_string()
        };

        SafetyDecision::Rejected(reason)
    }

    /// Create a rollback checkpoint to recover from catastrophic changes.
    pub fn create_checkpoint(
        &mut self,
        reason: String,
        efficiency: f64,
    ) {
        let checkpoint = RollbackCheckpoint {
            cycle: self.cycle_count,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            efficiency,
            reason,
            safe_state_hash: format!("checkpoint_{}", self.cycle_count),
        };

        self.checkpoints.push(checkpoint);
    }

    /// Record mutation decision in audit log.
    pub fn audit_mutation_decision(
        &mut self,
        mutation_id: String,
        decision: SafetyDecision,
        safety_score: f64,
        efficiency_before: f64,
        efficiency_after: f64,
        constraints_violated: Vec<String>,
    ) {
        let entry = AuditEntry {
            cycle: self.cycle_count,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            mutation_id,
            decision: decision.clone(),
            safety_score,
            efficiency_before,
            efficiency_after,
            constraint_violations: constraints_violated,
        };

        self.audit_log.push(entry);

        // Track consecutive rejections
        if matches!(decision, SafetyDecision::Rejected(_)) {
            self.consecutive_rejections += 1;
        } else {
            self.consecutive_rejections = 0;
            self.last_successful_cycle = self.cycle_count;
        }
    }

    /// Record behavioral snapshot for drift detection.
    pub fn record_behavioral_snapshot(
        &mut self,
        efficiency: f64,
        acceptance_rate: f64,
        mutations_accepted: usize,
        _mutations_evaluated: usize,
        domain: String,
    ) {
        let avg_gain = if mutations_accepted > 0 {
            efficiency / mutations_accepted as f64
        } else {
            0.0
        };

        let snapshot = BehavioralSnapshot {
            cycle: self.cycle_count,
            efficiency,
            acceptance_rate,
            mutations_accepted,
            avg_efficiency_gain_per_mutation: avg_gain,
            mutation_types_used: Vec::new(),
            domain_focus: domain,
        };

        self.behavioral_history.push_back(snapshot);

        // Keep only last 100 snapshots
        while self.behavioral_history.len() > 100 {
            self.behavioral_history.pop_front();
        }
    }

    /// Advance to next cycle.
    pub fn next_cycle(&mut self) {
        self.cycle_count += 1;
    }

    /// Get latest rollback checkpoint.
    pub fn latest_checkpoint(&self) -> Option<&RollbackCheckpoint> {
        self.checkpoints.last()
    }

    /// Check if system needs to initiate rollback.
    pub fn should_rollback(&self) -> bool {
        // Rollback if too many consecutive rejections
        if self.consecutive_rejections > self.constraints.max_mutations_per_cycle * 2 {
            return true;
        }

        // Rollback if acceptance rate dropped too low
        if let Some(latest) = self.behavioral_history.back() {
            if latest.acceptance_rate < self.constraints.acceptance_rate_floor * 0.5 {
                return true;
            }
        }

        false
    }

    /// Generate safety report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Safety & Governance Report (Phase 6.12) ===\n");
        report.push_str(&format!("Cycle count: {}\n", self.cycle_count));
        report.push_str(&format!("Consecutive rejections: {}\n", self.consecutive_rejections));
        report.push_str(&format!("Total decisions: {}\n", self.audit_log.len()));

        let approved = self.audit_log
            .iter()
            .filter(|e| matches!(e.decision, SafetyDecision::Approved))
            .count();
        let rejected = self.audit_log
            .iter()
            .filter(|e| matches!(e.decision, SafetyDecision::Rejected(_)))
            .count();
        let flagged = self.audit_log
            .iter()
            .filter(|e| matches!(e.decision, SafetyDecision::Flagged(_)))
            .count();

        report.push_str(&format!("  Approved: {}\n", approved));
        report.push_str(&format!("  Rejected: {}\n", rejected));
        report.push_str(&format!("  Flagged: {}\n\n", flagged));

        report.push_str(&format!("Checkpoints created: {}\n", self.checkpoints.len()));
        if let Some(latest) = self.latest_checkpoint() {
            report.push_str(&format!(
                "  Latest (cycle {}): {} (efficiency: {:.2})\n",
                latest.cycle, latest.reason, latest.efficiency
            ));
        }

        if let Some(latest) = self.behavioral_history.back() {
            report.push_str(&format!("\nLatest behavioral snapshot:\n"));
            report.push_str(&format!(
                "  Efficiency: {:.2}\n",
                latest.efficiency
            ));
            report.push_str(&format!(
                "  Acceptance rate: {:.1}%\n",
                latest.acceptance_rate * 100.0
            ));
            report.push_str(&format!(
                "  Avg gain per mutation: {:.4}\n",
                latest.avg_efficiency_gain_per_mutation
            ));
        }

        report
    }

    // Private helper: compute alignment score based on mutation type
    fn compute_alignment_score(&self, mutation_type: &str, efficiency_gain: f64) -> f64 {
        // Higher score for mutations aligned with historical success patterns
        // This is a simplified heuristic; in practice, would check against learned patterns
        let base_score: f64 = if efficiency_gain > 0.0 { 0.8 } else { 0.5 };
        let type_bonus: f64 = if mutation_type.contains("removal") || mutation_type.contains("pruning") {
            0.15 // Removal-based mutations historically successful
        } else {
            0.0
        };

        (base_score + type_bonus).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_constraints_default() {
        let constraints = SafetyConstraints::default();
        assert_eq!(constraints.max_efficiency_jump, 0.3);
        assert_eq!(constraints.drift_threshold, 0.4);
    }

    #[test]
    fn evaluate_mutation_safety_approved() {
        let engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        let score = engine.evaluate_mutation_safety(
            "mutation_1",
            1.0,
            1.1,
            0.8,
            "removal",
        );
        assert!(score.overall_score > 0.5);
    }

    #[test]
    fn evaluate_mutation_safety_high_jump() {
        let engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        let score = engine.evaluate_mutation_safety(
            "mutation_1",
            1.0,
            1.5,
            0.8,
            "addition",
        );
        assert!(score.overall_score < 0.8); // Should be flagged/reduced due to large jump
        assert!(score.constraint_score < 1.0); // Constraint violation detected
    }

    #[test]
    fn safety_decision_gate() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        let score = SafetyScore {
            constraint_score: 0.9,
            alignment_score: 0.8,
            confidence_score: 0.85,
            overall_score: 0.85,
            rationale: "All good".to_string(),
        };

        let decision = engine.gate_mutation_acceptance(
            "mut_1",
            true,
            &score,
            1.0,
            1.1,
        );

        assert_eq!(decision, SafetyDecision::Approved);
    }

    #[test]
    fn safety_decision_low_confidence() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        let score = SafetyScore {
            constraint_score: 0.2,
            alignment_score: 0.1,
            confidence_score: 0.1,
            overall_score: 0.15,
            rationale: "Too risky".to_string(),
        };

        let decision = engine.gate_mutation_acceptance(
            "mut_1",
            true,
            &score,
            1.0,
            1.1,
        );

        assert!(matches!(decision, SafetyDecision::Rejected(_)));
    }

    #[test]
    fn behavioral_drift_detection() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());

        // Record historical snapshots
        for i in 0..5 {
            engine.record_behavioral_snapshot(
                0.02 * i as f64,
                0.6,
                3,
                5,
                "Ranking".to_string(),
            );
            engine.next_cycle();
        }

        // Check normal behavior
        let normal = BehavioralSnapshot {
            cycle: 5,
            efficiency: 0.08,
            acceptance_rate: 0.6,
            mutations_accepted: 3,
            avg_efficiency_gain_per_mutation: 0.027,
            mutation_types_used: vec![],
            domain_focus: "Ranking".to_string(),
        };

        let (drifting, drift_amount, _) = engine.detect_behavioral_drift(&normal);
        assert!(!drifting);
        assert!(drift_amount < 0.1);
    }

    #[test]
    fn checkpoint_creation() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        engine.create_checkpoint(
            "Preemptive safety checkpoint".to_string(),
            0.95,
        );

        assert_eq!(engine.checkpoints.len(), 1);
        assert_eq!(engine.checkpoints[0].efficiency, 0.95);
    }

    #[test]
    fn audit_log_tracking() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        engine.audit_mutation_decision(
            "mut_1".to_string(),
            SafetyDecision::Approved,
            0.85,
            1.0,
            1.1,
            vec![],
        );

        assert_eq!(engine.audit_log.len(), 1);
        assert_eq!(engine.consecutive_rejections, 0);
    }

    #[test]
    fn consecutive_rejection_tracking() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());

        for i in 0..3 {
            engine.audit_mutation_decision(
                format!("mut_{}", i),
                SafetyDecision::Rejected("Test rejection".to_string()),
                0.3,
                1.0,
                0.99,
                vec!["constraint_1".to_string()],
            );
        }

        assert_eq!(engine.consecutive_rejections, 3);
    }

    #[test]
    fn rollback_decision() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());

        // Simulate many rejections
        for i in 0..25 {
            engine.audit_mutation_decision(
                format!("mut_{}", i),
                SafetyDecision::Rejected("Test".to_string()),
                0.2,
                1.0,
                0.99,
                vec![],
            );
        }

        assert!(engine.should_rollback());
    }

    #[test]
    fn governance_report() {
        let mut engine = SafetyGovernanceEngine::new(SafetyConstraints::default());
        engine.audit_mutation_decision(
            "mut_1".to_string(),
            SafetyDecision::Approved,
            0.85,
            1.0,
            1.1,
            vec![],
        );
        engine.create_checkpoint("test".to_string(), 0.95);

        let report = engine.report();
        assert!(report.contains("Safety & Governance Report"));
        assert!(report.contains("Checkpoints created: 1"));
    }
}
