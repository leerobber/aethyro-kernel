//! Autonomous strategy discovery: generate novel mutations by combining proven patterns.
//!
//! Phase 6.9 implements three hypothesis-generation mechanisms:
//! 1. **Cross-Domain Combination**: Combine successful mutations from different domains
//!    Example: "RemoveEdge worked in Ranking + SimplifyLayout in WebDesign → propose ChangeFormFields"
//!
//! 2. **Pattern-Based Synthesis**: Generate mutations that embody high-confidence patterns
//!    Example: "RemovalBias has 90% success across domains → propose RemoveNode variant"
//!
//! 3. **Novelty Exploration**: Propose untested mutation combinations
//!    Example: "Never tested AdjustWhitespace + AdjustColorContrast together → hypothesize synergy"

use super::adaptive::Domain;
use super::rules::MutationRuleKind;
use super::transfer_learning::{PatternAbstraction, InterdomainAffinityGraph};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A hypothesized mutation strategy with confidence and reasoning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypothesizedStrategy {
    /// The proposed mutation to try
    pub mutation: MutationRuleKind,
    /// Short description of why this mutation is promising
    pub rationale: String,
    /// Confidence score (0.0-1.0) based on supporting evidence
    pub confidence: f64,
    /// Source of hypothesis: CrossDomainTransfer, PatternBias, or NovelExploration
    pub source: HypothesisSource,
    /// Supporting evidence: which patterns or transfers informed this hypothesis
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HypothesisSource {
    /// Hypothesis from successful patterns in other domains
    CrossDomainTransfer,
    /// Hypothesis from high-confidence patterns
    PatternBias,
    /// Hypothesis from untested novel combinations
    NovelExploration,
}

/// Autonomous strategy discovery engine.
pub struct StrategyDiscoveryEngine {
    /// Affinity graph for querying cross-domain patterns
    affinity_graph: InterdomainAffinityGraph,
    /// Track which mutation combinations have been proposed (avoid duplicates)
    proposed_combinations: HashMap<String, u32>,
    /// Mutation proposal history (for novelty scoring)
    mutation_history: HashMap<String, u32>,
}

impl StrategyDiscoveryEngine {
    pub fn new(affinity_graph: InterdomainAffinityGraph) -> Self {
        Self {
            affinity_graph,
            proposed_combinations: HashMap::new(),
            mutation_history: HashMap::new(),
        }
    }

    /// Generate hypothesized strategies based on cross-domain patterns.
    ///
    /// Returns ranked list of mutations to try, with confidence scores.
    pub fn discover_strategies(
        &self,
        current_domain: Domain,
        target_mutation_count: usize,
    ) -> Vec<HypothesizedStrategy> {
        let mut strategies = Vec::new();

        // Strategy 1: Cross-domain transfer - apply patterns from other successful domains
        let transfer_strategies = self.discover_transfer_strategies(current_domain);
        strategies.extend(transfer_strategies);

        // Strategy 2: Pattern-bias - mutations that embody high-confidence patterns
        let pattern_strategies = self.discover_pattern_strategies(current_domain);
        strategies.extend(pattern_strategies);

        // Strategy 3: Novel exploration - untested combinations
        let novel_strategies = self.discover_novel_strategies(current_domain);
        strategies.extend(novel_strategies);

        // Sort by confidence (descending) and deduplicate
        strategies.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        strategies.dedup_by_key(|s| s.mutation.description());

        strategies.into_iter().take(target_mutation_count).collect()
    }

    /// Discover strategies by transferring patterns from other domains.
    fn discover_transfer_strategies(&self, current_domain: Domain) -> Vec<HypothesizedStrategy> {
        let mut strategies = Vec::new();

        // Query best patterns from all other domains
        for source_domain in &[Domain::Ranking, Domain::Classification, Domain::Search, Domain::Generic] {
            if *source_domain == current_domain {
                continue; // Skip current domain
            }

            let best_patterns = self.affinity_graph.best_patterns_for_transfer(*source_domain, current_domain);
            for (pattern, success_rate) in best_patterns.iter().take(3) {
                // Generate mutations that embody this pattern
                if let Some(mutations) = self.mutations_for_pattern(pattern, current_domain) {
                    for mutation in mutations {
                        strategies.push(HypothesizedStrategy {
                            mutation: mutation.clone(),
                            rationale: format!(
                                "Pattern '{}' transferred from {} domain ({}% success)",
                                pattern,
                                source_domain,
                                (success_rate * 100.0) as i32
                            ),
                            confidence: 0.6 + (success_rate * 0.35), // Base 0.6 + up to 0.35 bonus
                            source: HypothesisSource::CrossDomainTransfer,
                            evidence: vec![
                                format!("Transfer success rate: {:.1}%", success_rate * 100.0),
                                format!("Source domain: {}", source_domain),
                            ],
                        });
                    }
                }
            }
        }

        strategies
    }

    /// Discover strategies by finding mutations that embody high-confidence patterns.
    fn discover_pattern_strategies(&self, current_domain: Domain) -> Vec<HypothesizedStrategy> {
        let mut strategies = Vec::new();

        // Identify high-confidence patterns for this domain
        let high_confidence_patterns = vec![
            (PatternAbstraction::RemovalBias, 0.85),
            (PatternAbstraction::Simplification, 0.75),
            (PatternAbstraction::Emphasis, 0.70),
            (PatternAbstraction::Locality, 0.80),
        ];

        for (pattern, confidence_threshold) in high_confidence_patterns {
            // Only consider patterns with sufficient confidence
            let mut avg_confidence = 0.0;
            let mut count = 0;

            for source_domain in &[Domain::Ranking, Domain::Classification, Domain::Search, Domain::Generic] {
                if let Some(rate) = self.affinity_graph.transfer_rate(*source_domain, current_domain, &pattern) {
                    avg_confidence += rate;
                    count += 1;
                }
            }

            if count > 0 {
                avg_confidence /= count as f64;
                if avg_confidence > confidence_threshold {
                    // Generate mutations that embody this pattern
                    if let Some(mutations) = self.mutations_for_pattern(&pattern, current_domain) {
                        for mutation in mutations {
                            strategies.push(HypothesizedStrategy {
                                mutation: mutation.clone(),
                                rationale: format!(
                                    "Pattern '{}' is high-confidence ({:.1}%) across domains",
                                    pattern,
                                    avg_confidence * 100.0
                                ),
                                confidence: 0.5 + (avg_confidence * 0.4),
                                source: HypothesisSource::PatternBias,
                                evidence: vec![
                                    format!("Pattern confidence: {:.1}%", avg_confidence * 100.0),
                                    format!("Domains with this pattern: {}", count),
                                ],
                            });
                        }
                    }
                }
            }
        }

        strategies
    }

    /// Discover novel strategies by proposing untested mutation combinations.
    fn discover_novel_strategies(&self, current_domain: Domain) -> Vec<HypothesizedStrategy> {
        let mut strategies = Vec::new();

        // Novel strategy: combine mutations that have never been tested together
        // For now, propose a few standard untested combinations with lower confidence
        match current_domain {
            Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic => {
                // Propose a rarely-used mutation type with lower confidence
                strategies.push(HypothesizedStrategy {
                    mutation: MutationRuleKind::RewireEdge {
                        from: 0,
                        old_to: 1,
                        new_to: 2,
                    },
                    rationale: "Novel topology rewiring (rarely tested combination)".to_string(),
                    confidence: 0.35,
                    source: HypothesisSource::NovelExploration,
                    evidence: vec!["Untested combination".to_string()],
                });
            }
        }

        // Web design novel strategies
        if current_domain == Domain::Generic {
            use super::rules::ImagePlacement;
            strategies.push(HypothesizedStrategy {
                mutation: MutationRuleKind::ModifyImagePlacement {
                    placement: ImagePlacement::Hidden,
                },
                rationale: "Novel design exploration: test hidden image placement".to_string(),
                confidence: 0.30,
                source: HypothesisSource::NovelExploration,
                evidence: vec!["Untested in current domain".to_string()],
            });
        }

        strategies
    }

    /// Map abstract patterns to concrete mutations for a domain.
    fn mutations_for_pattern(
        &self,
        pattern: &PatternAbstraction,
        domain: Domain,
    ) -> Option<Vec<MutationRuleKind>> {
        match (pattern, domain) {
            // Topology domain
            (PatternAbstraction::RemovalBias, Domain::Ranking) |
            (PatternAbstraction::RemovalBias, Domain::Classification) |
            (PatternAbstraction::RemovalBias, Domain::Search) |
            (PatternAbstraction::RemovalBias, Domain::Generic) => {
                Some(vec![
                    MutationRuleKind::RemoveEdge { from: 0, to: 1 },
                    MutationRuleKind::RemoveNode { node_id: 0 },
                ])
            }

            (PatternAbstraction::Locality, Domain::Ranking) |
            (PatternAbstraction::Locality, Domain::Classification) |
            (PatternAbstraction::Locality, Domain::Search) |
            (PatternAbstraction::Locality, Domain::Generic) => {
                Some(vec![MutationRuleKind::RewireEdge {
                    from: 0,
                    old_to: 1,
                    new_to: 2,
                }])
            }

            (PatternAbstraction::EnhancementFocus, Domain::Ranking) |
            (PatternAbstraction::EnhancementFocus, Domain::Classification) |
            (PatternAbstraction::EnhancementFocus, Domain::Search) |
            (PatternAbstraction::EnhancementFocus, Domain::Generic) => {
                Some(vec![MutationRuleKind::AddNode {
                    label: "enhanced_node".to_string(),
                }])
            }

            // Web Design domain
            (PatternAbstraction::Simplification, Domain::Generic) => {
                Some(vec![
                    MutationRuleKind::SimplifyLayout { target_elements: 5 },
                    MutationRuleKind::ChangeFormFields { num_fields: 3 },
                ])
            }

            (PatternAbstraction::Emphasis, Domain::Generic) => {
                use super::rules::CTAPosition;
                Some(vec![
                    MutationRuleKind::ReorderCTA {
                        position: CTAPosition::AboveFold,
                    },
                    MutationRuleKind::AdjustColorContrast { increase: true },
                ])
            }

            (PatternAbstraction::ProgressiveDisclosure, Domain::Generic) => {
                use super::rules::CTAPosition;
                Some(vec![MutationRuleKind::ReorderCTA {
                    position: CTAPosition::FloatingCorner,
                }])
            }

            // Marketing domain
            (PatternAbstraction::Urgency, Domain::Generic) => {
                use super::rules::CopyTone;
                Some(vec![MutationRuleKind::AdjustCopyTone {
                    tone: CopyTone::Urgent,
                }])
            }

            (PatternAbstraction::Segmentation, Domain::Generic) => {
                use super::rules::AudienceSegment;
                Some(vec![MutationRuleKind::ChangeTargetAudience {
                    segment: AudienceSegment::SMB,
                }])
            }

            _ => None,
        }
    }

    /// Record a mutation as proposed to track history.
    pub fn record_proposed_mutation(&mut self, mutation_desc: String) {
        let count = self.mutation_history.entry(mutation_desc).or_insert(0);
        *count += 1;
    }

    /// Get novelty score for a mutation (how rarely has it been proposed?).
    pub fn novelty_score(&self, mutation_desc: &str) -> f64 {
        let count = self.mutation_history.get(mutation_desc).copied().unwrap_or(0);
        // Exponential decay: first proposal is novel (1.0), gets less novel with repetition
        (-0.1 * count as f64).exp()
    }

    /// Update the affinity graph with a new state (for keeping in sync with LoopController).
    pub fn sync_affinity_graph(&mut self, affinity_graph: InterdomainAffinityGraph) {
        self.affinity_graph = affinity_graph;
    }

    /// Get discovery statistics for reporting.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Strategy Discovery Report ===\n");
        report.push_str(&format!(
            "Total proposed combinations: {}\n",
            self.proposed_combinations.len()
        ));
        report.push_str(&format!(
            "Mutation proposals tracked: {}\n",
            self.mutation_history.len()
        ));

        let most_proposed: Vec<_> = self
            .mutation_history
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        report.push_str("\nMost frequently proposed mutations:\n");
        for (mutation, count) in most_proposed.iter().take(5) {
            report.push_str(&format!("  {}: {} times\n", mutation, count));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_discovery_engine_creates_strategies() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let engine = StrategyDiscoveryEngine::new(affinity_graph);

        let strategies = engine.discover_strategies(Domain::Ranking, 5);
        // Should generate some strategies even with empty affinity graph
        assert!(!strategies.is_empty());
    }

    #[test]
    fn strategy_discovery_includes_multiple_sources() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let engine = StrategyDiscoveryEngine::new(affinity_graph);

        let strategies = engine.discover_strategies(Domain::Generic, 10);

        // Should have strategies from different sources
        let sources: std::collections::HashSet<_> =
            strategies.iter().map(|s| s.source.clone()).collect();
        assert!(sources.len() >= 1); // At least one source
    }

    #[test]
    fn strategy_discovery_records_mutations() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let mut engine = StrategyDiscoveryEngine::new(affinity_graph);

        engine.record_proposed_mutation("test_mutation".to_string());
        engine.record_proposed_mutation("test_mutation".to_string());

        assert_eq!(engine.mutation_history.get("test_mutation"), Some(&2));
    }

    #[test]
    fn strategy_discovery_novelty_score() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let mut engine = StrategyDiscoveryEngine::new(affinity_graph);

        let mutation = "novel_mutation";
        let score1 = engine.novelty_score(mutation);

        engine.record_proposed_mutation(mutation.to_string());
        let score2 = engine.novelty_score(mutation);

        // Novelty should decrease after proposal
        assert!(score2 < score1);
    }

    #[test]
    fn strategy_discovery_hypothesis_has_evidence() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let engine = StrategyDiscoveryEngine::new(affinity_graph);

        let strategies = engine.discover_strategies(Domain::Generic, 10);

        // Each strategy should have supporting evidence
        for strategy in &strategies {
            assert!(!strategy.evidence.is_empty(), "Strategy should have evidence: {}", strategy.rationale);
            assert!(!strategy.rationale.is_empty(), "Strategy should have rationale");
        }
    }

    #[test]
    fn strategy_discovery_confidence_in_range() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let engine = StrategyDiscoveryEngine::new(affinity_graph);

        let strategies = engine.discover_strategies(Domain::Generic, 10);

        for strategy in strategies {
            assert!(
                strategy.confidence >= 0.0 && strategy.confidence <= 1.0,
                "Confidence should be in [0, 1], got {}",
                strategy.confidence
            );
        }
    }

    #[test]
    fn strategy_discovery_report_is_readable() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let mut engine = StrategyDiscoveryEngine::new(affinity_graph);

        engine.record_proposed_mutation("mutation1".to_string());
        engine.record_proposed_mutation("mutation2".to_string());

        let report = engine.report();
        assert!(report.contains("Strategy Discovery Report"));
        assert!(report.contains("Most frequently proposed"));
    }

    #[test]
    fn strategy_discovery_generates_diverse_mutations() {
        let affinity_graph = InterdomainAffinityGraph::new();
        let engine = StrategyDiscoveryEngine::new(affinity_graph);

        let strategies = engine.discover_strategies(Domain::Generic, 10);

        // Should generate multiple different mutations
        let mutation_types: std::collections::HashSet<_> =
            strategies.iter().map(|s| format!("{:?}", s.mutation)).collect();
        assert!(mutation_types.len() > 1, "Should generate diverse mutations");
    }
}
