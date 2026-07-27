//! Autonomous self-improvement loop controller.
//!
//! Orchestrates:
//! - Monitor health trends
//! - Propose mutations when degradation detected
//! - Evaluate under budget constraints
//! - Accept/reject based on fitness improvement
//! - Log all events to the tamper-evident ledger
//! - Record every mutation to build domain expertise

use super::super::graph::Graph;
use super::super::error::NtgError;
use super::{MutationCycle, SelfModConfig, AdaptiveMutationProposer, DegradationSignal, MutationLedger, Domain};

/// Outcome of a self-improvement cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopOutcome {
    /// No mutations proposed (no degradation detected).
    NoAction,
    /// Cycle ran, mutations evaluated.
    Executed,
    /// Cycle was budget-exhausted mid-evaluation.
    BudgetExhausted,
    /// No mutations accepted (all rejected due to fitness gate).
    NoAcceptance,
    /// One or more mutations accepted and applied.
    Improved,
}

/// Statistics from a single loop iteration.
#[derive(Clone, Debug)]
pub struct LoopStats {
    pub outcome: LoopOutcome,
    pub mutations_proposed: usize,
    pub mutations_evaluated: usize,
    pub mutations_accepted: usize,
    pub baseline_efficiency: f64,
    pub final_efficiency: f64,
    pub budget_consumed_us: u64,
}

impl Default for LoopStats {
    fn default() -> Self {
        Self {
            outcome: LoopOutcome::NoAction,
            mutations_proposed: 0,
            mutations_evaluated: 0,
            mutations_accepted: 0,
            baseline_efficiency: 1.0,
            final_efficiency: 1.0,
            budget_consumed_us: 0,
        }
    }
}

/// Autonomous self-improvement loop controller.
pub struct LoopController {
    /// Configuration for mutation cycles.
    pub config: SelfModConfig,
    /// Adaptive proposer that learns from history.
    pub proposer: AdaptiveMutationProposer,
    /// Cycle count.
    pub cycle_count: u64,
    /// Efficiency baseline (for measuring improvement).
    pub efficiency_baseline: f64,
    /// Tamper-evident ledger: records every mutation for learning and analysis.
    pub ledger: MutationLedger,
    /// Problem domain (for learning domain-specific strategies).
    pub domain: Domain,
}

impl LoopController {
    pub fn new(config: SelfModConfig) -> Self {
        Self {
            config,
            proposer: AdaptiveMutationProposer::new(),
            cycle_count: 0,
            efficiency_baseline: 1.0,
            ledger: MutationLedger::new(),
            domain: Domain::Generic,
        }
    }

    /// Set the problem domain for learning domain-specific strategies
    pub fn set_domain(&mut self, domain: Domain) {
        self.domain = domain;
    }

    /// Run one iteration of the self-improvement loop.
    ///
    /// Returns:
    /// - The outcome of the cycle
    /// - Updated graph (if mutations were accepted)
    /// - Statistics for logging
    pub fn step(
        &mut self,
        graph: &Graph,
        current_efficiency: f64,
        degradation_signal: Option<DegradationSignal>,
    ) -> Result<(Graph, LoopStats), NtgError> {
        self.cycle_count += 1;
        let mut stats = LoopStats::default();
        stats.baseline_efficiency = current_efficiency;

        // Step 1: Check if we should run this cycle.
        let should_run = degradation_signal.is_some() && current_efficiency < self.efficiency_baseline;
        if !should_run {
            return Ok((graph.clone(), stats));
        }

        let signal = degradation_signal.unwrap();

        // Step 2: Propose mutations (with ledger-informed confidence bias and domain awareness).
        let proposals = self.proposer.propose_mutations_with_domain(
            graph,
            signal,
            self.domain,
            self.config.max_mutations_per_cycle,
            &self.ledger,
        )?;
        stats.mutations_proposed = proposals.len();

        // Step 3: Create mutation cycle.
        let baseline_fitness = (
            (current_efficiency * 100.0) as u64,
            (current_efficiency * 1_000_000.0) as u64,
        );

        let mut cycle = MutationCycle::new(self.config.clone(), baseline_fitness)?;

        // Step 4: Evaluate mutations.
        let mut best_graph = graph.clone();
        let mut best_efficiency = current_efficiency;

        for (idx, proposal) in proposals.iter().enumerate() {
            cycle.propose_mutation(proposal.clone())?;
            stats.mutations_evaluated += 1;

            let (new_fitness, elapsed_us) = cycle.evaluate_mutation(graph, idx)?;
            stats.budget_consumed_us += elapsed_us;

            // Compute new efficiency from fitness.
            let new_efficiency = (
                new_fitness.0 as f64 / baseline_fitness.0 as f64
            ).min(1.0) * 0.8 + (
                new_fitness.1 as f64 / baseline_fitness.1 as f64
            ).min(1.0) * 0.2;

            // Step 5: Decide acceptance.
            let was_accepted = cycle.should_accept(new_fitness) && new_efficiency > best_efficiency;
            if was_accepted {
                // Accept this mutation.
                cycle.accept_mutation(idx)?;
                stats.mutations_accepted += 1;
                self.proposer.record_success(proposal.description());

                // Apply to working graph.
                proposal.apply(&mut best_graph)?;
                best_efficiency = new_efficiency;
            }

            // Record this mutation event in the ledger for learning
            self.ledger.record_mutation(
                proposal.kind.clone(),
                signal,
                self.domain,
                current_efficiency,
                new_efficiency,
                was_accepted,
                self.proposer.accept_rate,
            );

            // Check budget.
            if !cycle.within_budget() {
                stats.outcome = LoopOutcome::BudgetExhausted;
                stats.final_efficiency = best_efficiency;
                self.ledger.next_cycle();
                return Ok((best_graph, stats));
            }
        }

        // Step 6: Update accept rate for exploit/explore tuning.
        self.proposer.update_accept_rate(
            stats.mutations_accepted,
            stats.mutations_evaluated,
        );

        // Step 7: Determine outcome.
        stats.final_efficiency = best_efficiency;
        stats.outcome = if stats.mutations_accepted > 0 {
            // Update baseline for next cycle.
            self.efficiency_baseline = best_efficiency;
            LoopOutcome::Improved
        } else if stats.mutations_evaluated > 0 {
            LoopOutcome::NoAcceptance
        } else {
            LoopOutcome::NoAction
        };

        // Step 8: Advance ledger to next cycle
        self.ledger.next_cycle();

        Ok((best_graph, stats))
    }

    /// Get the current learning report from the mutation ledger
    pub fn learning_report(&self) -> String {
        self.ledger.report()
    }

    /// Save the ledger to a file for persistence across sessions
    pub fn save_ledger(&self, path: &str) -> Result<(), NtgError> {
        self.ledger.save(path)
    }

    /// Load the ledger from a file, or create new if not found
    pub fn load_ledger(&mut self, path: &str) -> Result<(), NtgError> {
        self.ledger = MutationLedger::load_or_new(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::graph::NodeKind;

    #[test]
    fn loop_controller_new() {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let controller = LoopController::new(config);
        assert_eq!(controller.cycle_count, 0);
        assert_eq!(controller.efficiency_baseline, 1.0);
    }

    #[test]
    fn loop_controller_step_no_degradation() -> Result<(), NtgError> {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let mut controller = LoopController::new(config);
        let graph = Graph::new();

        let (_, stats) = controller.step(&graph, 0.95, None)?;
        assert_eq!(stats.outcome, LoopOutcome::NoAction);
        Ok(())
    }

    #[test]
    fn loop_controller_step_with_degradation() -> Result<(), NtgError> {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let mut controller = LoopController::new(config);
        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        // Run with degradation signal.
        let (_, stats) = controller.step(
            &graph,
            0.85,
            Some(DegradationSignal::LatencyDominant),
        )?;

        // Should have proposed and evaluated mutations.
        assert!(stats.mutations_proposed > 0);
        assert!(stats.mutations_evaluated > 0);
        Ok(())
    }

    #[test]
    fn loop_stats_default() {
        let stats = LoopStats::default();
        assert_eq!(stats.outcome, LoopOutcome::NoAction);
        assert_eq!(stats.mutations_proposed, 0);
        assert_eq!(stats.baseline_efficiency, 1.0);
    }
}
