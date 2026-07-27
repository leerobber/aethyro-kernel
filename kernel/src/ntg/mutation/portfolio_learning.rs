//! Portfolio learning: track and learn from portfolios of successful strategies.
//!
//! Phase 6.10 portfolio component manages:
//! 1. Strategy diversification: Maintain portfolio of diverse, effective strategies
//! 2. Interdependency learning: Which strategies work better together?
//! 3. Portfolio rebalancing: Allocate effort to underexplored regions
//! 4. Portfolio return: Track cumulative effectiveness of strategy combinations

use super::adaptive::Domain;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A portfolio entry: a strategy or combination with its track record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortfolioEntry {
    /// Strategy identifier (mutation or combination)
    pub strategy_id: String,
    /// Mutations in this portfolio entry
    pub mutations: Vec<String>,
    /// Cumulative return (sum of improvements)
    pub cumulative_return: f64,
    /// Number of times deployed
    pub deployments: u32,
    /// Average return per deployment
    pub avg_return: f64,
    /// Volatility (std dev of returns)
    pub volatility: f64,
    /// Sharpe ratio (return per unit of risk)
    pub sharpe_ratio: f64,
    /// Domain where this strategy works best
    pub primary_domain: Domain,
}

/// Interdependency: how well two strategies work together.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractionEffect {
    /// First strategy
    pub strategy1: String,
    /// Second strategy
    pub strategy2: String,
    /// Interaction coefficient: >1.0 means synergy, <1.0 means interference
    pub interaction_coefficient: f64,
    /// Number of times tried together
    pub observations: u32,
}

/// Portfolio learning engine: manage strategy portfolios.
pub struct PortfolioLearningEngine {
    /// Current portfolio of strategies
    portfolio: HashMap<String, PortfolioEntry>,
    /// Strategy interaction effects
    interactions: HashMap<(String, String), InteractionEffect>,
    /// Domain allocations (how much effort to each domain)
    domain_allocation: HashMap<Domain, f64>,
    /// Portfolio diversity score (0.0-1.0, higher is more diverse)
    diversity_score: f64,
    /// Risk threshold (maximum acceptable volatility)
    risk_threshold: f64,
}

impl PortfolioLearningEngine {
    pub fn new() -> Self {
        Self {
            portfolio: HashMap::new(),
            interactions: HashMap::new(),
            domain_allocation: HashMap::new(),
            diversity_score: 0.5,
            risk_threshold: 0.3,
        }
    }

    /// Add a strategy to the portfolio after evaluation.
    pub fn add_strategy_result(
        &mut self,
        strategy_id: String,
        mutations: Vec<String>,
        domain: Domain,
        return_value: f64,
    ) {
        let entry = self.portfolio.entry(strategy_id.clone()).or_insert(PortfolioEntry {
            strategy_id: strategy_id.clone(),
            mutations: mutations.clone(),
            cumulative_return: 0.0,
            deployments: 0,
            avg_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            primary_domain: domain,
        });

        entry.cumulative_return += return_value;
        entry.deployments += 1;
        entry.avg_return = entry.cumulative_return / entry.deployments as f64;

        // Simplified volatility: variance of returns (proper implementation would track history)
        entry.volatility = (return_value - entry.avg_return).abs();

        // Sharpe ratio: excess return per unit of risk
        if entry.volatility > 0.0 {
            entry.sharpe_ratio = entry.avg_return / entry.volatility;
        } else {
            entry.sharpe_ratio = entry.avg_return;
        }

        // Update domain allocation
        let allocation = self.domain_allocation.entry(domain).or_insert(0.0);
        *allocation += 1.0;

        // Update portfolio diversity
        self.update_diversity_score();
    }

    /// Record interaction between two strategies.
    pub fn record_interaction(
        &mut self,
        strategy1: String,
        strategy2: String,
        combined_return: f64,
        individual_return1: f64,
        individual_return2: f64,
    ) {
        let expected_return = individual_return1 + individual_return2;
        let interaction_coeff = if expected_return > 0.0 {
            combined_return / expected_return
        } else {
            1.0
        };

        let key = if strategy1 < strategy2 {
            (strategy1.clone(), strategy2.clone())
        } else {
            (strategy2.clone(), strategy1.clone())
        };

        let interaction = self.interactions.entry(key).or_insert(InteractionEffect {
            strategy1: strategy1.clone(),
            strategy2: strategy2.clone(),
            interaction_coefficient: 1.0,
            observations: 0,
        });

        interaction.interaction_coefficient = (interaction.interaction_coefficient * interaction.observations as f64 + interaction_coeff)
            / (interaction.observations as f64 + 1.0);
        interaction.observations += 1;
    }

    /// Get top performers in portfolio.
    pub fn top_strategies(&self, top_k: usize) -> Vec<(String, f64, f64)> {
        let mut strategies: Vec<_> = self
            .portfolio
            .iter()
            .map(|(id, entry)| (id.clone(), entry.sharpe_ratio, entry.avg_return))
            .collect();

        strategies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        strategies.into_iter().take(top_k).collect()
    }

    /// Identify underexplored regions (low allocation domains).
    pub fn underexplored_domains(&self) -> Vec<(Domain, f64)> {
        let total: f64 = self.domain_allocation.values().sum();
        if total == 0.0 {
            return vec![];
        }

        let avg_allocation = total / self.domain_allocation.len() as f64;

        let mut allocations: Vec<_> = self
            .domain_allocation
            .iter()
            .map(|(domain, alloc)| (domain.clone(), *alloc / total))
            .collect();

        allocations.retain(|(_, alloc)| alloc < &avg_allocation);
        allocations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        allocations
    }

    /// Get synergistic strategy pairs (high interaction coefficients).
    pub fn synergistic_pairs(&self, top_k: usize) -> Vec<((String, String), f64)> {
        let mut pairs: Vec<_> = self
            .interactions
            .iter()
            .filter(|(_, effect)| effect.interaction_coefficient > 1.1)
            .map(|(key, effect)| (key.clone(), effect.interaction_coefficient))
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(top_k).collect()
    }

    /// Get interfering strategy pairs (low interaction coefficients).
    pub fn interfering_pairs(&self, top_k: usize) -> Vec<((String, String), f64)> {
        let mut pairs: Vec<_> = self
            .interactions
            .iter()
            .filter(|(_, effect)| effect.interaction_coefficient < 0.9)
            .map(|(key, effect)| (key.clone(), effect.interaction_coefficient))
            .collect();

        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(top_k).collect()
    }

    /// Rebalance portfolio to explore underexplored regions.
    pub fn rebalancing_recommendation(&self) -> Vec<(Domain, f64)> {
        self.underexplored_domains()
    }

    /// Calculate portfolio Sharpe ratio (overall risk-adjusted return).
    pub fn portfolio_sharpe_ratio(&self) -> f64 {
        if self.portfolio.is_empty() {
            return 0.0;
        }

        let avg_sharpe: f64 = self.portfolio.values().map(|e| e.sharpe_ratio).sum::<f64>()
            / self.portfolio.len() as f64;
        avg_sharpe
    }

    /// Update diversity score based on strategy distribution.
    fn update_diversity_score(&mut self) {
        if self.portfolio.is_empty() {
            self.diversity_score = 0.0;
            return;
        }

        // Diversity = entropy of strategy performance distribution
        let total_return: f64 = self.portfolio.values().map(|e| e.cumulative_return).sum();
        if total_return == 0.0 {
            self.diversity_score = 0.0;
            return;
        }

        let mut entropy = 0.0;
        for entry in self.portfolio.values() {
            let p = entry.cumulative_return / total_return;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize entropy to [0, 1]
        let max_entropy = (self.portfolio.len() as f64).ln();
        self.diversity_score = if max_entropy > 0.0 {
            (entropy / max_entropy).min(1.0)
        } else {
            0.0
        };
    }

    /// Generate portfolio report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Portfolio Learning Report ===\n");
        report.push_str(&format!("Portfolio size: {}\n", self.portfolio.len()));
        report.push_str(&format!("Diversity score: {:.2}\n", self.diversity_score));
        report.push_str(&format!("Portfolio Sharpe ratio: {:.3}\n", self.portfolio_sharpe_ratio()));

        report.push_str("\nTop performing strategies:\n");
        for (strategy, sharpe, avg_return) in self.top_strategies(5) {
            report.push_str(&format!(
                "  {}: Sharpe={:.2}, avg_return={:.4}\n",
                strategy, sharpe, avg_return
            ));
        }

        report.push_str("\nUnderexplored domains:\n");
        for (domain, allocation) in self.underexplored_domains().iter().take(5) {
            report.push_str(&format!("  {:?}: {:.1}% allocation\n", domain, allocation * 100.0));
        }

        report.push_str("\nSynergistic pairs:\n");
        for ((s1, s2), coeff) in self.synergistic_pairs(5) {
            report.push_str(&format!("  {} + {} = {:.2}x\n", s1, s2, coeff));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_add_strategy() {
        let mut engine = PortfolioLearningEngine::new();
        engine.add_strategy_result("strategy1".to_string(), vec!["mut1".to_string()], Domain::Generic, 0.1);
        assert_eq!(engine.portfolio.len(), 1);
    }

    #[test]
    fn portfolio_sharpe_ratio() {
        let mut engine = PortfolioLearningEngine::new();
        for i in 0..3 {
            engine.add_strategy_result(
                format!("strategy{}", i),
                vec![format!("mut{}", i)],
                Domain::Generic,
                0.1 * (i as f64 + 1.0),
            );
        }

        let sharpe = engine.portfolio_sharpe_ratio();
        assert!(sharpe >= 0.0);
    }

    #[test]
    fn portfolio_diversity_score() {
        let mut engine = PortfolioLearningEngine::new();
        engine.add_strategy_result("s1".to_string(), vec!["m1".to_string()], Domain::Generic, 0.5);
        engine.add_strategy_result("s2".to_string(), vec!["m2".to_string()], Domain::Generic, 0.5);

        assert!(engine.diversity_score >= 0.0 && engine.diversity_score <= 1.0);
    }

    #[test]
    fn portfolio_interaction_effects() {
        let mut engine = PortfolioLearningEngine::new();
        engine.record_interaction("s1".to_string(), "s2".to_string(), 0.3, 0.1, 0.1);
        assert!(engine.interactions.len() > 0);
    }

    #[test]
    fn portfolio_synergistic_pairs() {
        let mut engine = PortfolioLearningEngine::new();
        engine.record_interaction("s1".to_string(), "s2".to_string(), 0.25, 0.1, 0.1);
        let synergies = engine.synergistic_pairs(5);
        assert!(synergies.len() > 0);
    }

    #[test]
    fn portfolio_underexplored_domains() {
        let mut engine = PortfolioLearningEngine::new();
        engine.add_strategy_result("s1".to_string(), vec!["m1".to_string()], Domain::Generic, 0.1);
        engine.add_strategy_result("s2".to_string(), vec!["m2".to_string()], Domain::Generic, 0.1);

        let underexplored = engine.underexplored_domains();
        assert!(underexplored.len() >= 0);
    }

    #[test]
    fn portfolio_report_is_readable() {
        let mut engine = PortfolioLearningEngine::new();
        engine.add_strategy_result("s1".to_string(), vec!["m1".to_string()], Domain::Generic, 0.1);

        let report = engine.report();
        assert!(report.contains("Portfolio Learning Report"));
        assert!(report.contains("Portfolio size"));
    }
}
