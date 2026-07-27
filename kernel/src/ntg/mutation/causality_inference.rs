//! Causality inference: infer causal relationships between mutations and improvements.
//!
//! Phase 6.10 causality component builds causal models:
//! 1. Intervention analysis: Does mutation X actually CAUSE improvement or just correlate?
//! 2. Confounding detection: Are improvements caused by mutation or baseline drift?
//! 3. Causal graph: Map which mutations enable others (synergies)
//! 4. Treatment effect estimation: Isolate mutation's true causal effect

use super::adaptive::Domain;
use super::rules::MutationRuleKind;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Causal evidence for a mutation's effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalEffect {
    /// Mutation that was applied
    pub mutation_desc: String,
    /// Domain it was applied in
    pub domain: Domain,
    /// Average treatment effect (ATE): average improvement from this mutation
    pub ate: f64,
    /// Number of independent trials
    pub trial_count: u32,
    /// Confidence in causal effect (0.0-1.0)
    pub confidence: f64,
    /// Are there confounders (baseline drift, other changes)?
    pub has_confounders: bool,
    /// Mutations that interact with this one (synergies)
    pub synergies: Vec<String>,
}

/// Intervention record: what changed, what was the result?
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterventionRecord {
    /// Which mutation was applied
    pub mutation: MutationRuleKind,
    /// Efficiency before intervention
    pub efficiency_before: f64,
    /// Efficiency after intervention
    pub efficiency_after: f64,
    /// Was it the only change (isolated)?
    pub isolated: bool,
    /// Other mutations applied around same time (potential confounders)
    pub concurrent_mutations: Vec<String>,
    /// Baseline trend before intervention (for confounder detection)
    pub baseline_trend: f64,
}

/// Causality inference engine: build causal models of mutations.
pub struct CausalityInferenceEngine {
    /// Intervention history
    interventions: Vec<InterventionRecord>,
    /// Causal effects for each mutation
    causal_effects: HashMap<String, CausalEffect>,
    /// Confounder detection: baseline drift estimates
    baseline_drift: f64,
    /// Causal graph: (mutation1, mutation2) -> synergy score
    synergy_graph: HashMap<(String, String), f64>,
}

impl CausalityInferenceEngine {
    pub fn new() -> Self {
        Self {
            interventions: Vec::new(),
            causal_effects: HashMap::new(),
            baseline_drift: 0.0,
            synergy_graph: HashMap::new(),
        }
    }

    /// Record an intervention with potential confounders.
    pub fn record_intervention(
        &mut self,
        mutation: MutationRuleKind,
        efficiency_before: f64,
        efficiency_after: f64,
        isolated: bool,
        concurrent_mutations: Vec<String>,
        baseline_trend: f64,
    ) {
        let mutation_desc = mutation.description();
        let record = InterventionRecord {
            mutation: mutation.clone(),
            efficiency_before,
            efficiency_after,
            isolated,
            concurrent_mutations: concurrent_mutations.clone(),
            baseline_trend,
        };

        self.interventions.push(record);

        // Update baseline drift estimate (for confounder detection)
        self.baseline_drift = (self.baseline_drift + baseline_trend) / 2.0;

        // Calculate causal effect
        let observed_effect = efficiency_after - efficiency_before;
        let has_confounders = !isolated || !concurrent_mutations.is_empty();

        // Estimate true causal effect by removing baseline drift
        let causal_effect = if has_confounders {
            (observed_effect - self.baseline_drift).max(0.0)
        } else {
            observed_effect
        };

        let effect = self.causal_effects.entry(mutation_desc.clone()).or_insert(CausalEffect {
            mutation_desc: mutation_desc.clone(),
            domain: Domain::Generic,
            ate: 0.0,
            trial_count: 0,
            confidence: 0.0,
            has_confounders,
            synergies: Vec::new(),
        });

        // Update ATE (average treatment effect) with new observation
        effect.ate = (effect.ate * effect.trial_count as f64 + causal_effect) / (effect.trial_count as f64 + 1.0);
        effect.trial_count += 1;

        // Confidence grows with more isolated trials
        if isolated {
            effect.confidence = (effect.trial_count as f64 / (effect.trial_count as f64 + 3.0)).min(1.0);
        } else {
            effect.confidence = (effect.trial_count as f64 / (effect.trial_count as f64 + 5.0)).min(0.8);
        }

        // Track synergies: if this mutation succeeds when others also present, likely synergy
        if !concurrent_mutations.is_empty() && efficiency_after > efficiency_before {
            for concurrent in concurrent_mutations {
                let key = (mutation_desc.clone(), concurrent.clone());
                let synergy = self.synergy_graph.entry(key).or_insert(0.0);
                *synergy = (*synergy + causal_effect) / 2.0;
            }
        }
    }

    /// Get causal effect for a mutation.
    pub fn get_causal_effect(&self, mutation_desc: &str) -> Option<&CausalEffect> {
        self.causal_effects.get(mutation_desc)
    }

    /// Get mutations with strongest causal effects (high confidence).
    pub fn strongest_causal_mutations(&self, top_k: usize) -> Vec<(String, f64, f64)> {
        let mut effects: Vec<_> = self
            .causal_effects
            .iter()
            .filter(|(_, effect)| effect.confidence > 0.5)
            .map(|(desc, effect)| (desc.clone(), effect.ate, effect.confidence))
            .collect();

        effects.sort_by(|a, b| {
            let score_a = a.1 * a.2; // ate * confidence
            let score_b = b.1 * b.2;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        effects.into_iter().take(top_k).collect()
    }

    /// Detect confounding effects: mutations that correlate but lack causal evidence.
    pub fn confounded_mutations(&self) -> Vec<(String, f64)> {
        self.causal_effects
            .iter()
            .filter(|(_, effect)| effect.has_confounders && effect.confidence < 0.6)
            .map(|(desc, effect)| (desc.clone(), effect.ate))
            .collect()
    }

    /// Get synergistic mutation pairs (mutations that work better together).
    pub fn synergies(&self, top_k: usize) -> Vec<((String, String), f64)> {
        let mut pairs: Vec<_> = self
            .synergy_graph
            .iter()
            .map(|(pair, score)| (pair.clone(), *score))
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(top_k).collect()
    }

    /// Estimate causal effect size (Cohen's d-like measure).
    pub fn effect_size(&self, mutation_desc: &str) -> f64 {
        if let Some(effect) = self.causal_effects.get(mutation_desc) {
            effect.ate
        } else {
            0.0
        }
    }

    /// Generate causality report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Causality Inference Report ===\n");
        report.push_str(&format!("Total interventions: {}\n", self.interventions.len()));
        report.push_str(&format!("Estimated baseline drift: {:.4}\n", self.baseline_drift));

        report.push_str("\nStrongest causal mutations (high confidence):\n");
        for (mutation, ate, conf) in self.strongest_causal_mutations(5) {
            report.push_str(&format!(
                "  {}: ATE={:.4}, confidence={:.2}%\n",
                mutation,
                ate,
                conf * 100.0
            ));
        }

        report.push_str("\nMutations with confounding risk:\n");
        for (mutation, ate) in self.confounded_mutations().iter().take(5) {
            report.push_str(&format!("  {}: ATE={:.4} (unconfirmed)\n", mutation, ate));
        }

        report.push_str("\nSynergistic mutation pairs:\n");
        for ((mut1, mut2), score) in self.synergies(5) {
            report.push_str(&format!("  {} + {} = {:.3}\n", mut1, mut2, score));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causality_intervention_recording() {
        let mut engine = CausalityInferenceEngine::new();
        engine.record_intervention(
            MutationRuleKind::AddNode { label: "test".to_string() },
            0.5,
            0.6,
            true,
            vec![],
            0.0,
        );
        assert_eq!(engine.interventions.len(), 1);
    }

    #[test]
    fn causality_ate_calculation() {
        let mut engine = CausalityInferenceEngine::new();
        for i in 0..3 {
            engine.record_intervention(
                MutationRuleKind::AddNode { label: "test".to_string() },
                0.5,
                0.5 + 0.1 * (i as f64 + 1.0),
                true,
                vec![],
                0.0,
            );
        }

        if let Some(effect) = engine.get_causal_effect("add_node(label='test')") {
            assert!(effect.ate > 0.0);
            assert_eq!(effect.trial_count, 3);
        }
    }

    #[test]
    fn causality_confounder_detection() {
        let mut engine = CausalityInferenceEngine::new();
        engine.record_intervention(
            MutationRuleKind::AddNode { label: "confounded".to_string() },
            0.5,
            0.6,
            false, // not isolated
            vec!["other_mutation".to_string()],
            0.05, // baseline drift
        );

        if let Some(effect) = engine.get_causal_effect("add_node(label='confounded')") {
            assert!(effect.has_confounders);
            assert!(effect.confidence < 0.8);
        }
    }

    #[test]
    fn causality_synergy_detection() {
        let mut engine = CausalityInferenceEngine::new();
        engine.record_intervention(
            MutationRuleKind::AddNode { label: "test".to_string() },
            0.5,
            0.7,
            false,
            vec!["RemoveEdge".to_string()],
            0.0,
        );

        let synergies = engine.synergies(5);
        assert!(!synergies.is_empty());
    }

    #[test]
    fn causality_strongest_mutations() {
        let mut engine = CausalityInferenceEngine::new();
        // Record isolated interventions multiple times to build confidence
        for _ in 0..5 {
            engine.record_intervention(
                MutationRuleKind::AddNode { label: "test".to_string() },
                0.5,
                0.7,
                true,
                vec![],
                0.0,
            );
        }

        let strongest = engine.strongest_causal_mutations(5);
        assert!(strongest.len() > 0);
        assert!(strongest[0].2 > 0.5); // confidence should be > 0.5
    }

    #[test]
    fn causality_report_is_readable() {
        let mut engine = CausalityInferenceEngine::new();
        engine.record_intervention(
            MutationRuleKind::AddNode { label: "test".to_string() },
            0.5,
            0.6,
            true,
            vec![],
            0.0,
        );

        let report = engine.report();
        assert!(report.contains("Causality Inference Report"));
        assert!(report.contains("Total interventions"));
    }
}
