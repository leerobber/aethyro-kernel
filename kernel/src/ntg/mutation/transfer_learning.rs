//! Cross-domain knowledge transfer: learn patterns in one domain and apply to others.
//!
//! Key insight: optimization patterns learned in Topology domain transfer to Web Design and
//! Marketing domains. For example:
//! - "Progressive disclosure" (Ranking: RemoveEdge early) → (WebDesign: hide features)
//!   → (Marketing: reveal benefits one at a time)
//! - "Locality principle" (all high-success mutations affect <3 nodes) → (WebDesign: small
//!   incremental UI changes) → (Marketing: focused message)
//!
//! This module tracks:
//! 1. Pattern abstractions (generalized principles from specific mutations)
//! 2. Domain affinity matrix (which patterns work well across domains)
//! 3. Transfer confidence (how often a pattern from Domain X succeeds in Domain Y)

use super::adaptive::Domain;
use super::rules::MutationRuleKind;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A generalized pattern abstraction that applies across domains.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternAbstraction {
    /// Progressive disclosure: reveal features/benefits incrementally.
    ProgressiveDisclosure,
    /// Locality principle: make small, focused changes (affects few nodes/elements).
    Locality,
    /// Removal bias: deletion outperforms addition in this scenario.
    RemovalBias,
    /// Enhancement focus: adding/strengthening outperforms removal.
    EnhancementFocus,
    /// Simplification: reduce complexity (fewer options, clearer paths).
    Simplification,
    /// Emphasis: highlight critical elements over peripheral ones.
    Emphasis,
    /// Urgency: time-sensitive or scarcity messaging.
    Urgency,
    /// Segmentation: tailor approach to specific audience types.
    Segmentation,
    /// Customization: personalized vs generic approach.
    Customization,
    /// Cost-focus: emphasize value/savings vs other benefits.
    CostFocus,
}

impl std::fmt::Display for PatternAbstraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternAbstraction::ProgressiveDisclosure => write!(f, "progressive_disclosure"),
            PatternAbstraction::Locality => write!(f, "locality"),
            PatternAbstraction::RemovalBias => write!(f, "removal_bias"),
            PatternAbstraction::EnhancementFocus => write!(f, "enhancement_focus"),
            PatternAbstraction::Simplification => write!(f, "simplification"),
            PatternAbstraction::Emphasis => write!(f, "emphasis"),
            PatternAbstraction::Urgency => write!(f, "urgency"),
            PatternAbstraction::Segmentation => write!(f, "segmentation"),
            PatternAbstraction::Customization => write!(f, "customization"),
            PatternAbstraction::CostFocus => write!(f, "cost_focus"),
        }
    }
}

/// Mapping from domain-specific mutations to abstract patterns.
pub struct PatternExtractor;

impl PatternExtractor {
    /// Extract abstract patterns from a mutation rule.
    pub fn extract_patterns(mutation: &MutationRuleKind, domain: Domain) -> Vec<PatternAbstraction> {
        match (domain, mutation) {
            // Topology domain patterns
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::RemoveEdge { .. }) => {
                vec![PatternAbstraction::RemovalBias, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::RemoveNode { .. }) => {
                vec![PatternAbstraction::RemovalBias, PatternAbstraction::Simplification]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AddNode { .. }) => {
                vec![PatternAbstraction::EnhancementFocus, PatternAbstraction::Customization]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AddEdge { .. }) => {
                vec![PatternAbstraction::EnhancementFocus, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::RewireEdge { .. }) => {
                vec![PatternAbstraction::Locality, PatternAbstraction::Customization]
            }

            // Web Design domain patterns
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::SimplifyLayout { .. }) => {
                vec![PatternAbstraction::Simplification, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AdjustWhitespace { .. }) => {
                vec![PatternAbstraction::Simplification, PatternAbstraction::Emphasis]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AdjustColorContrast { .. }) => {
                vec![PatternAbstraction::Emphasis, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ReorderCTA { .. }) => {
                vec![PatternAbstraction::Emphasis, PatternAbstraction::ProgressiveDisclosure]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ChangeFormFields { .. }) => {
                vec![PatternAbstraction::Simplification, PatternAbstraction::RemovalBias]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ModifyImagePlacement { .. }) => {
                vec![PatternAbstraction::Emphasis, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AdjustTypographyHierarchy { .. }) => {
                vec![PatternAbstraction::Emphasis, PatternAbstraction::Simplification]
            }

            // Marketing domain patterns
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::RefocusValueProposition { .. }) => {
                vec![PatternAbstraction::Emphasis, PatternAbstraction::Simplification]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AdjustCopyTone { .. }) => {
                vec![PatternAbstraction::Urgency, PatternAbstraction::Customization]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ChangeTargetAudience { .. }) => {
                vec![PatternAbstraction::Segmentation, PatternAbstraction::Customization]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ModifyPricingTier { .. }) => {
                vec![PatternAbstraction::CostFocus, PatternAbstraction::Segmentation]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::ShiftChannelMix { .. }) => {
                vec![PatternAbstraction::Customization, PatternAbstraction::Locality]
            }
            (Domain::Ranking | Domain::Classification | Domain::Search | Domain::Generic, MutationRuleKind::AdjustRetentionStrategy { .. }) => {
                vec![PatternAbstraction::Customization, PatternAbstraction::Emphasis]
            }
        }
    }
}

/// Inter-domain affinity matrix: tracks transfer success rates between domains.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterdomainAffinityGraph {
    /// Affinity scores: (source_domain, target_domain, pattern) → success_rate
    affinities: HashMap<(Domain, Domain, PatternAbstraction), TransferStats>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferStats {
    /// Number of times this pattern transferred from source to target domain
    pub transfer_count: u32,
    /// Number of successful transfers
    pub successful_transfers: u32,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
}

impl InterdomainAffinityGraph {
    pub fn new() -> Self {
        Self {
            affinities: HashMap::new(),
        }
    }

    /// Record a pattern transfer from source domain to target domain.
    pub fn record_transfer(
        &mut self,
        source_domain: Domain,
        target_domain: Domain,
        pattern: PatternAbstraction,
        succeeded: bool,
    ) {
        let key = (source_domain, target_domain, pattern);
        let stats = self.affinities.entry(key).or_insert(TransferStats {
            transfer_count: 0,
            successful_transfers: 0,
            success_rate: 0.0,
        });

        stats.transfer_count += 1;
        if succeeded {
            stats.successful_transfers += 1;
        }

        // Update success rate
        if stats.transfer_count > 0 {
            stats.success_rate = stats.successful_transfers as f64 / stats.transfer_count as f64;
        }
    }

    /// Get transfer success rate for a pattern across domains.
    pub fn transfer_rate(
        &self,
        source_domain: Domain,
        target_domain: Domain,
        pattern: &PatternAbstraction,
    ) -> Option<f64> {
        self.affinities
            .get(&(source_domain, target_domain, pattern.clone()))
            .map(|stats| stats.success_rate)
    }

    /// Get best patterns to transfer from source to target domain (sorted by success rate).
    pub fn best_patterns_for_transfer(
        &self,
        source_domain: Domain,
        target_domain: Domain,
    ) -> Vec<(PatternAbstraction, f64)> {
        let mut patterns: Vec<_> = self
            .affinities
            .iter()
            .filter(|(key, _)| key.0 == source_domain && key.1 == target_domain)
            .map(|(key, stats)| (key.2.clone(), stats.success_rate))
            .collect();

        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        patterns
    }

    /// Get cross-domain specialization score: how well does this domain transfer to others?
    pub fn cross_domain_specialization_score(&self, domain: Domain) -> f64 {
        let mut total_rate = 0.0;
        let mut count = 0;

        for ((src, _tgt, _), stats) in &self.affinities {
            if *src == domain {
                total_rate += stats.success_rate;
                count += 1;
            }
        }

        if count == 0 {
            0.5 // neutral if no transfer data
        } else {
            total_rate / count as f64
        }
    }

    /// Get a summary of affinity statistics.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Inter-Domain Affinity Report ===\n");

        let mut by_transfer: HashMap<(Domain, Domain), Vec<(PatternAbstraction, f64)>> = HashMap::new();
        for ((src, tgt, pattern), stats) in &self.affinities {
            by_transfer
                .entry((*src, *tgt))
                .or_insert_with(Vec::new)
                .push((pattern.clone(), stats.success_rate));
        }

        for ((src, tgt), mut patterns) in by_transfer {
            patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            report.push_str(&format!(
                "\n{} → {}: {} patterns\n",
                src,
                tgt,
                patterns.len()
            ));
            for (pattern, rate) in patterns.iter().take(5) {
                report.push_str(&format!("  {:?}: {:.1}%\n", pattern, rate * 100.0));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rules::MutationRuleKind;

    #[test]
    fn pattern_extractor_topology_mutations() {
        let patterns = PatternExtractor::extract_patterns(
            &MutationRuleKind::RemoveEdge { from: 1, to: 2 },
            Domain::Ranking,
        );
        assert!(patterns.contains(&PatternAbstraction::RemovalBias));
        assert!(patterns.contains(&PatternAbstraction::Locality));
    }

    #[test]
    fn pattern_extractor_web_design_mutations() {
        use super::super::rules::CTAPosition;
        let patterns = PatternExtractor::extract_patterns(
            &MutationRuleKind::ReorderCTA {
                position: CTAPosition::AboveFold,
            },
            Domain::Generic,
        );
        assert!(patterns.contains(&PatternAbstraction::Emphasis));
        assert!(patterns.contains(&PatternAbstraction::ProgressiveDisclosure));
    }

    #[test]
    fn pattern_extractor_marketing_mutations() {
        let patterns = PatternExtractor::extract_patterns(
            &MutationRuleKind::RefocusValueProposition {
                focus_area: "speed".to_string(),
            },
            Domain::Generic,
        );
        assert!(patterns.contains(&PatternAbstraction::Emphasis));
        assert!(patterns.contains(&PatternAbstraction::Simplification));
    }

    #[test]
    fn affinity_graph_records_transfers() {
        let mut graph = InterdomainAffinityGraph::new();

        graph.record_transfer(
            Domain::Ranking,
            Domain::Classification,
            PatternAbstraction::RemovalBias,
            true,
        );
        graph.record_transfer(
            Domain::Ranking,
            Domain::Classification,
            PatternAbstraction::RemovalBias,
            true,
        );
        graph.record_transfer(
            Domain::Ranking,
            Domain::Classification,
            PatternAbstraction::RemovalBias,
            false,
        );

        let rate = graph
            .transfer_rate(Domain::Ranking, Domain::Classification, &PatternAbstraction::RemovalBias)
            .unwrap();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn affinity_graph_best_patterns() {
        let mut graph = InterdomainAffinityGraph::new();

        // RemovalBias: 80% success
        for _ in 0..8 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Search,
                PatternAbstraction::RemovalBias,
                true,
            );
        }
        for _ in 0..2 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Search,
                PatternAbstraction::RemovalBias,
                false,
            );
        }

        // Locality: 60% success
        for _ in 0..6 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Search,
                PatternAbstraction::Locality,
                true,
            );
        }
        for _ in 0..4 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Search,
                PatternAbstraction::Locality,
                false,
            );
        }

        let best = graph.best_patterns_for_transfer(Domain::Ranking, Domain::Search);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].0, PatternAbstraction::RemovalBias);
        assert!(best[0].1 > best[1].1);
    }

    #[test]
    fn affinity_graph_cross_domain_specialization() {
        let mut graph = InterdomainAffinityGraph::new();

        // High success transferring from Ranking
        for _ in 0..9 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Classification,
                PatternAbstraction::RemovalBias,
                true,
            );
        }
        for _ in 0..1 {
            graph.record_transfer(
                Domain::Ranking,
                Domain::Classification,
                PatternAbstraction::RemovalBias,
                false,
            );
        }

        let score = graph.cross_domain_specialization_score(Domain::Ranking);
        assert!(score > 0.8);
    }

    #[test]
    fn affinity_graph_report_is_readable() {
        let mut graph = InterdomainAffinityGraph::new();
        graph.record_transfer(
            Domain::Ranking,
            Domain::Search,
            PatternAbstraction::Locality,
            true,
        );
        graph.record_transfer(
            Domain::Classification,
            Domain::Search,
            PatternAbstraction::Emphasis,
            false,
        );

        let report = graph.report();
        assert!(report.contains("Inter-Domain Affinity"));
        assert!(report.contains("→"));
    }
}
