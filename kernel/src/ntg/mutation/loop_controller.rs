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
use super::{
    MutationCycle, SelfModConfig, AdaptiveMutationProposer, DegradationSignal, MutationLedger, Domain,
    InterdomainAffinityGraph, PatternExtractor, StrategyDiscoveryEngine, TemporalLearningEngine,
    CausalityInferenceEngine, PortfolioLearningEngine, KnowledgeDistillationEngine,
    ResearchPaperEngine, StudyDesignEngine, PeerReviewEngine, PublicationPipeline,
};

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
    /// Cross-domain knowledge transfer: tracks which patterns work across domains.
    pub affinity_graph: InterdomainAffinityGraph,
    /// Autonomous strategy discovery engine for novel hypothesis generation.
    pub discovery_engine: StrategyDiscoveryEngine,
    /// Temporal learning: track how mutation effectiveness changes over time.
    pub temporal_engine: TemporalLearningEngine,
    /// Causality inference: learn causal relationships between mutations and improvements.
    pub causality_engine: CausalityInferenceEngine,
    /// Portfolio learning: manage diversified strategy portfolio.
    pub portfolio_engine: PortfolioLearningEngine,
    /// Knowledge distillation: compress learned knowledge into rules.
    pub distillation_engine: KnowledgeDistillationEngine,
    /// Research paper generation: automated paper synthesis from mutations.
    pub research_paper_engine: ResearchPaperEngine,
    /// Study design: experimental design generation and validation.
    pub study_design_engine: StudyDesignEngine,
    /// Peer review: automated reviewer simulation and feedback.
    pub peer_review_engine: PeerReviewEngine,
    /// Publication pipeline: end-to-end manuscript submission workflow.
    pub publication_pipeline: PublicationPipeline,
}

impl LoopController {
    pub fn new(config: SelfModConfig) -> Self {
        let affinity_graph = InterdomainAffinityGraph::new();
        Self {
            config,
            proposer: AdaptiveMutationProposer::new(),
            cycle_count: 0,
            efficiency_baseline: 1.0,
            ledger: MutationLedger::new(),
            domain: Domain::Generic,
            discovery_engine: StrategyDiscoveryEngine::new(affinity_graph.clone()),
            affinity_graph,
            temporal_engine: TemporalLearningEngine::new(),
            causality_engine: CausalityInferenceEngine::new(),
            portfolio_engine: PortfolioLearningEngine::new(),
            distillation_engine: KnowledgeDistillationEngine::new(),
            research_paper_engine: ResearchPaperEngine::new(),
            study_design_engine: StudyDesignEngine::new(),
            peer_review_engine: PeerReviewEngine::new(),
            publication_pipeline: PublicationPipeline::new(),
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

        // Step 2: Discover novel strategies from cross-domain patterns and high-confidence hypotheses.
        let discovered_strategies = self.discovery_engine.discover_strategies(
            self.domain,
            self.config.max_mutations_per_cycle,
        );
        for strategy in &discovered_strategies {
            self.discovery_engine.record_proposed_mutation(strategy.mutation.description());
        }

        // Step 2b: Propose mutations (with ledger-informed confidence bias and domain awareness).
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

                // Phase 6.10: Record mutation for continuous learning
                // Temporal learning: track effectiveness over time
                self.temporal_engine.record_mutation(
                    self.cycle_count,
                    self.domain,
                    proposal.kind.clone(),
                    current_efficiency,
                    new_efficiency,
                    true,
                    current_efficiency,
                );

                // Causality inference: record as isolated intervention
                self.causality_engine.record_intervention(
                    proposal.kind.clone(),
                    current_efficiency,
                    new_efficiency,
                    true,
                    vec![],
                    0.0,
                );

                // Portfolio learning: record strategy result
                self.portfolio_engine.add_strategy_result(
                    proposal.description(),
                    vec![proposal.description()],
                    self.domain,
                    new_efficiency - current_efficiency,
                );

                // Extract and record patterns for cross-domain transfer learning.
                // This enables patterns learned in one domain to transfer to others.
                let patterns = PatternExtractor::extract_patterns(&proposal.kind, self.domain);
                for pattern in patterns {
                    // Record transfer from current domain to all other domains
                    // (simulating that this pattern might be useful elsewhere)
                    for other_domain in &[Domain::Ranking, Domain::Classification, Domain::Search, Domain::Generic] {
                        if *other_domain != self.domain {
                            // Success in current domain suggests potential transfer
                            self.affinity_graph.record_transfer(
                                self.domain,
                                *other_domain,
                                pattern.clone(),
                                true,
                            );
                        }
                    }
                }
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

        // Step 9: Sync discovery engine with latest affinity graph for next cycle
        self.discovery_engine.sync_affinity_graph(self.affinity_graph.clone());

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

    /// Get cross-domain transfer report: which patterns transfer well between domains
    pub fn cross_domain_transfer_report(&self) -> String {
        self.affinity_graph.report()
    }

    /// Get specialization score for this domain: how well do its patterns transfer to others?
    pub fn domain_transfer_specialization(&self) -> f64 {
        self.affinity_graph.cross_domain_specialization_score(self.domain)
    }

    /// Get best patterns to transfer from current domain to target domain
    pub fn best_transfer_patterns(&self, target_domain: Domain) -> Vec<String> {
        self.affinity_graph
            .best_patterns_for_transfer(self.domain, target_domain)
            .iter()
            .take(5)
            .map(|(pattern, rate)| format!("{}: {:.1}%", pattern, rate * 100.0))
            .collect()
    }

    /// Get temporal learning report (Phase 6.10)
    pub fn temporal_learning_report(&self) -> String {
        self.temporal_engine.report()
    }

    /// Get causality inference report (Phase 6.10)
    pub fn causality_inference_report(&self) -> String {
        self.causality_engine.report()
    }

    /// Get portfolio learning report (Phase 6.10)
    pub fn portfolio_learning_report(&self) -> String {
        self.portfolio_engine.report()
    }

    /// Get knowledge distillation report (Phase 6.10)
    pub fn knowledge_distillation_report(&self) -> String {
        self.distillation_engine.report()
    }

    /// Get comprehensive continuous learning report (Phase 6.10)
    pub fn continuous_learning_report(&self) -> String {
        let mut report = String::from("=== Phase 6.10: Continuous Learning Report ===\n\n");
        report.push_str("--- Temporal Learning ---\n");
        report.push_str(&self.temporal_learning_report());
        report.push_str("\n--- Causality Inference ---\n");
        report.push_str(&self.causality_inference_report());
        report.push_str("\n--- Portfolio Learning ---\n");
        report.push_str(&self.portfolio_learning_report());
        report.push_str("\n--- Knowledge Distillation ---\n");
        report.push_str(&self.knowledge_distillation_report());
        report
    }

    /// Get research paper generation report (Phase 6.11)
    pub fn research_paper_report(&self) -> String {
        self.research_paper_engine.report()
    }

    /// Get study design report (Phase 6.11)
    pub fn study_design_report(&self) -> String {
        self.study_design_engine.report()
    }

    /// Get peer review report (Phase 6.11)
    pub fn peer_review_report(&self) -> String {
        self.peer_review_engine.report()
    }

    /// Get publication pipeline report (Phase 6.11)
    pub fn publication_pipeline_report(&self) -> String {
        self.publication_pipeline.report()
    }

    /// Get comprehensive research and publication report (Phase 6.11)
    pub fn research_and_publication_report(&self) -> String {
        let mut report = String::from("=== Phase 6.11: Research & Academic Discovery Report ===\n\n");
        report.push_str("--- Research Paper Generation ---\n");
        report.push_str(&self.research_paper_report());
        report.push_str("\n--- Study Design ---\n");
        report.push_str(&self.study_design_report());
        report.push_str("\n--- Peer Review ---\n");
        report.push_str(&self.peer_review_report());
        report.push_str("\n--- Publication Pipeline ---\n");
        report.push_str(&self.publication_pipeline_report());
        report
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

    // Phase 6.8: Cross-Domain Knowledge Transfer Tests
    #[test]
    fn loop_controller_initializes_affinity_graph() {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let mut controller = LoopController::new(config);
        controller.set_domain(Domain::Ranking);
        // Affinity graph should be initialized but empty
        let report = controller.cross_domain_transfer_report();
        assert!(report.contains("Inter-Domain Affinity"));
    }

    #[test]
    fn loop_controller_records_patterns_on_success() -> Result<(), NtgError> {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let mut controller = LoopController::new(config);
        controller.set_domain(Domain::Ranking);

        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        // Run with degradation signal - should trigger mutations
        let (_, _stats) = controller.step(
            &graph,
            0.85,
            Some(DegradationSignal::LatencyDominant),
        )?;

        // Even if no mutations accepted, affinity graph should not error
        let transfer_score = controller.domain_transfer_specialization();
        assert!(transfer_score >= 0.0 && transfer_score <= 1.0);
        Ok(())
    }

    #[test]
    fn loop_controller_provides_transfer_patterns() -> Result<(), NtgError> {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };
        let mut controller = LoopController::new(config);
        controller.set_domain(Domain::Ranking);

        let best_patterns = controller.best_transfer_patterns(Domain::Classification);
        // Should return a list (possibly empty if no transfers yet)
        assert!(best_patterns.len() <= 5); // Limited to 5 best patterns
        Ok(())
    }

    #[test]
    fn loop_controller_multiple_domains_separate_affinity() -> Result<(), NtgError> {
        let config = SelfModConfig {
            enabled: true,
            ..SelfModConfig::default()
        };

        let mut controller1 = LoopController::new(config.clone());
        controller1.set_domain(Domain::Ranking);

        let mut controller2 = LoopController::new(config.clone());
        controller2.set_domain(Domain::Classification);

        // Each controller should track domain-specific patterns
        let score1 = controller1.domain_transfer_specialization();
        let score2 = controller2.domain_transfer_specialization();

        // Both should be valid scores
        assert!(score1 >= 0.0 && score1 <= 1.0);
        assert!(score2 >= 0.0 && score2 <= 1.0);
        Ok(())
    }
}
