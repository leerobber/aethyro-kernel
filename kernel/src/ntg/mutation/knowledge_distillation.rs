//! Knowledge distillation: compress learned knowledge into simpler models.
//!
//! Phase 6.10 distillation component:
//! 1. Rule extraction: Convert learned patterns into simple decision rules
//! 2. Meta-patterns: Identify universal principles across domains
//! 3. Compression: Reduce stored knowledge while preserving accuracy
//! 4. Transfer: Enable faster learning in new domains via distilled knowledge

use super::adaptive::Domain;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A distilled rule: simplified pattern extracted from complex knowledge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistilledRule {
    /// Human-readable rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// When to apply this rule (condition)
    pub condition: RuleCondition,
    /// What to do (action)
    pub action: RuleAction,
    /// Confidence in this rule (0.0-1.0)
    pub confidence: f64,
    /// How many observations support this rule
    pub support: u32,
    /// Domains where this rule applies best
    pub applicable_domains: Vec<Domain>,
}

/// Condition for applying a rule.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RuleCondition {
    /// Apply when efficiency is in a certain range
    EfficiencyBand { min: f32, max: f32 },
    /// Apply when specific signal detected
    DegradationSignal(String),
    /// Apply when strategy is in specific phase
    OptimizationPhase(String),
    /// Always apply
    Always,
}

/// Action to take when condition met.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuleAction {
    /// Try this mutation
    ProposeMutation(String),
    /// Use this strategy
    UseStrategy(String),
    /// Explore this domain
    ExploreRegion(String),
    /// Rebalance portfolio
    Rebalance,
}

/// Meta-pattern: universal principle across domains.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaPattern {
    /// Pattern name (e.g., "removal_bias", "locality_principle")
    pub name: String,
    /// How many times observed across domains
    pub observations: u32,
    /// Average effectiveness (0.0-1.0)
    pub effectiveness: f64,
    /// Domains where observed
    pub observed_in_domains: Vec<Domain>,
    /// Unified description
    pub description: String,
}

/// Knowledge distillation engine: extract and compress learned knowledge.
pub struct KnowledgeDistillationEngine {
    /// Extracted rules
    rules: Vec<DistilledRule>,
    /// Meta-patterns (universal principles)
    meta_patterns: HashMap<String, MetaPattern>,
    /// Compression ratio: how much we've compressed
    compression_ratio: f64,
    /// Original knowledge size (for compression measurement)
    original_size: usize,
    /// Compressed knowledge size
    compressed_size: usize,
}

impl KnowledgeDistillationEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            meta_patterns: HashMap::new(),
            compression_ratio: 0.0,
            original_size: 0,
            compressed_size: 0,
        }
    }

    /// Extract a rule from observed patterns.
    pub fn extract_rule(
        &mut self,
        name: String,
        description: String,
        condition: RuleCondition,
        action: RuleAction,
        confidence: f64,
        support: u32,
        domains: Vec<Domain>,
    ) {
        let rule = DistilledRule {
            name,
            description,
            condition,
            action,
            confidence,
            support,
            applicable_domains: domains,
        };

        self.rules.push(rule);
        self.update_compression();
    }

    /// Register a meta-pattern: universal principle across domains.
    pub fn register_meta_pattern(
        &mut self,
        name: String,
        description: String,
        effectiveness: f64,
        domains: Vec<Domain>,
    ) {
        let pattern = MetaPattern {
            name: name.clone(),
            observations: 1,
            effectiveness,
            observed_in_domains: domains,
            description,
        };

        let entry = self.meta_patterns.entry(name).or_insert(pattern);
        entry.observations += 1;
        entry.effectiveness = (entry.effectiveness + effectiveness) / 2.0;
    }

    /// Get rules applicable to current conditions.
    pub fn applicable_rules(
        &self,
        efficiency: f64,
        domain: Domain,
    ) -> Vec<&DistilledRule> {
        self.rules
            .iter()
            .filter(|rule| {
                // Check domain applicability
                if !rule.applicable_domains.contains(&domain) && !rule.applicable_domains.contains(&Domain::Generic) {
                    return false;
                }

                // Check condition
                match &rule.condition {
                    RuleCondition::EfficiencyBand { min, max } => {
                        efficiency >= *min as f64 && efficiency <= *max as f64
                    }
                    RuleCondition::Always => true,
                    _ => false, // Other conditions handled by caller
                }
            })
            .collect()
    }

    /// Get meta-patterns applicable across domains.
    pub fn cross_domain_meta_patterns(&self) -> Vec<&MetaPattern> {
        self.meta_patterns
            .values()
            .filter(|p| p.observed_in_domains.len() >= 2)
            .collect()
    }

    /// Get universal meta-patterns (observed in 3+ domains).
    pub fn universal_patterns(&self) -> Vec<&MetaPattern> {
        self.meta_patterns
            .values()
            .filter(|p| p.observed_in_domains.len() >= 3)
            .collect()
    }

    /// Measure distillation quality: how much knowledge compressed with what accuracy?
    pub fn distillation_quality(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }

        // Quality = (1 - compression_loss) * compression_ratio
        // compression_loss ≈ (1 - average_rule_confidence)
        let avg_confidence = if !self.rules.is_empty() {
            self.rules.iter().map(|r| r.confidence).sum::<f64>() / self.rules.len() as f64
        } else {
            0.0
        };

        let compression_loss = 1.0 - avg_confidence;
        let actual_ratio = if self.original_size > 0 {
            self.compressed_size as f64 / self.original_size as f64
        } else {
            0.0
        };

        (1.0 - compression_loss) * (1.0 - actual_ratio)
    }

    /// Update compression metrics.
    fn update_compression(&mut self) {
        // Simplified: each rule ≈ 100 bytes, original knowledge from ledger
        self.compressed_size = self.rules.len() * 100 + self.meta_patterns.len() * 80;

        // Rough estimate: original would need ~10x more to store full knowledge
        self.original_size = self.compressed_size * 10;

        self.compression_ratio = 1.0 - (self.compressed_size as f64 / self.original_size as f64);
    }

    /// Generate distillation report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Knowledge Distillation Report ===\n");
        report.push_str(&format!(
            "Extracted rules: {}\n",
            self.rules.len()
        ));
        report.push_str(&format!(
            "Meta-patterns: {}\n",
            self.meta_patterns.len()
        ));
        report.push_str(&format!(
            "Compression ratio: {:.1}%\n",
            self.compression_ratio * 100.0
        ));
        report.push_str(&format!(
            "Distillation quality: {:.2}\n",
            self.distillation_quality()
        ));

        report.push_str("\nCross-domain meta-patterns:\n");
        for pattern in self.cross_domain_meta_patterns().iter().take(5) {
            report.push_str(&format!(
                "  {}: {:.1}% effective, {} domains\n",
                pattern.name,
                pattern.effectiveness * 100.0,
                pattern.observed_in_domains.len()
            ));
        }

        report.push_str("\nUniversal patterns (3+ domains):\n");
        for pattern in self.universal_patterns().iter().take(5) {
            report.push_str(&format!(
                "  {}: {}\n",
                pattern.name,
                pattern.description
            ));
        }

        report.push_str("\nTop rules by confidence:\n");
        let mut sorted_rules = self.rules.clone();
        sorted_rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        for rule in sorted_rules.iter().take(5) {
            report.push_str(&format!(
                "  {}: {:.0}% confidence ({} observations)\n",
                rule.name,
                rule.confidence * 100.0,
                rule.support
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distillation_extract_rule() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.extract_rule(
            "early_phase_removal".to_string(),
            "In early phase, removal works well".to_string(),
            RuleCondition::EfficiencyBand { min: 0.0, max: 0.4 },
            RuleAction::ProposeMutation("RemoveEdge".to_string()),
            0.85,
            100,
            vec![Domain::Generic],
        );

        assert_eq!(engine.rules.len(), 1);
    }

    #[test]
    fn distillation_meta_pattern() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.register_meta_pattern(
            "removal_bias".to_string(),
            "Removal outperforms addition in early phases".to_string(),
            0.80,
            vec![Domain::Ranking, Domain::Classification],
        );

        assert!(engine.meta_patterns.contains_key("removal_bias"));
    }

    #[test]
    fn distillation_applicable_rules() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.extract_rule(
            "rule1".to_string(),
            "Test".to_string(),
            RuleCondition::EfficiencyBand { min: 0.0, max: 0.5 },
            RuleAction::ProposeMutation("Test".to_string()),
            0.8,
            50,
            vec![Domain::Generic],
        );

        let applicable = engine.applicable_rules(0.3, Domain::Generic);
        assert_eq!(applicable.len(), 1);
    }

    #[test]
    fn distillation_cross_domain_patterns() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.register_meta_pattern(
            "pattern1".to_string(),
            "Test".to_string(),
            0.8,
            vec![Domain::Ranking, Domain::Classification],
        );

        let cross = engine.cross_domain_meta_patterns();
        assert!(cross.len() > 0);
    }

    #[test]
    fn distillation_universal_patterns() {
        let mut engine = KnowledgeDistillationEngine::new();

        // Register a single pattern with 3+ domains to make it universal
        engine.register_meta_pattern(
            "universal".to_string(),
            "A truly universal pattern".to_string(),
            0.8,
            vec![Domain::Ranking, Domain::Classification, Domain::Search],
        );

        let universal = engine.universal_patterns();
        assert!(universal.len() > 0);
        assert_eq!(universal[0].observed_in_domains.len(), 3);
    }

    #[test]
    fn distillation_quality_score() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.extract_rule(
            "rule1".to_string(),
            "Test".to_string(),
            RuleCondition::Always,
            RuleAction::Rebalance,
            0.9,
            50,
            vec![Domain::Generic],
        );

        let quality = engine.distillation_quality();
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn distillation_report_is_readable() {
        let mut engine = KnowledgeDistillationEngine::new();
        engine.extract_rule(
            "rule1".to_string(),
            "Test".to_string(),
            RuleCondition::Always,
            RuleAction::Rebalance,
            0.8,
            50,
            vec![Domain::Generic],
        );

        let report = engine.report();
        assert!(report.contains("Knowledge Distillation Report"));
        assert!(report.contains("Extracted rules"));
    }
}
