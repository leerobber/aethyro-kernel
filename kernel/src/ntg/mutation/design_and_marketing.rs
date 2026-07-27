//! Domain-specific fitness evaluators for web design and marketing mutations.
//!
//! Unlike topology mutations evaluated against simulated graph performance,
//! web design and marketing mutations are evaluated against real metrics:
//! - Web Design: conversion rate, bounce rate, engagement, time-on-page
//! - Marketing: CTR, impressions, lead quality, revenue per campaign
//!
//! This module provides abstractions for tracking and computing fitness
//! based on domain-specific metrics.

use serde::{Serialize, Deserialize};

/// Real-world metrics for web design optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebDesignMetrics {
    /// Conversion rate (0.0-1.0)
    pub conversion_rate: f64,
    /// Bounce rate (0.0-1.0, lower is better)
    pub bounce_rate: f64,
    /// Average time on page (seconds)
    pub avg_time_on_page: f64,
    /// Engagement score (0-100)
    pub engagement_score: f64,
    /// Scroll depth (0.0-1.0, how far users scroll)
    pub scroll_depth: f64,
}

impl WebDesignMetrics {
    /// Calculate a composite fitness score for web design.
    /// Weights: conversion (40%), engagement (30%), bounce rate (20%), scroll depth (10%)
    pub fn fitness_score(&self) -> f64 {
        (self.conversion_rate * 0.40)
            + (self.engagement_score / 100.0 * 0.30)
            + ((1.0 - self.bounce_rate) * 0.20)
            + (self.scroll_depth * 0.10)
    }
}

/// Real-world metrics for marketing optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketingMetrics {
    /// Click-through rate (0.0-1.0)
    pub ctr: f64,
    /// Impressions/reach for the campaign
    pub impressions: u64,
    /// Cost per acquisition ($)
    pub cost_per_acquisition: f64,
    /// Lead quality score (0-100, subjective)
    pub lead_quality: f64,
    /// Customer lifetime value ($)
    pub customer_lifetime_value: f64,
}

impl MarketingMetrics {
    /// Calculate a composite fitness score for marketing.
    /// Weights: CTR (30%), lead quality (30%), CLV/CPA ratio (40%)
    pub fn fitness_score(&self) -> f64 {
        let clv_to_cpa_ratio = if self.cost_per_acquisition > 0.0 {
            (self.customer_lifetime_value / self.cost_per_acquisition).min(10.0) / 10.0
        } else {
            0.5 // neutral if CPA is unknown
        };

        (self.ctr * 0.30) + (self.lead_quality / 100.0 * 0.30) + (clv_to_cpa_ratio * 0.40)
    }
}

/// Real-world metrics for retention/engagement optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionMetrics {
    /// Monthly churn rate (0.0-1.0, lower is better)
    pub churn_rate: f64,
    /// Monthly active users retention (0.0-1.0)
    pub mau_retention: f64,
    /// Net Promoter Score (-100 to 100)
    pub nps: f64,
    /// Feature adoption rate (0.0-1.0)
    pub feature_adoption_rate: f64,
}

impl RetentionMetrics {
    /// Calculate a composite fitness score for retention.
    /// Weights: churn reduction (35%), MAU retention (35%), NPS (20%), adoption (10%)
    pub fn fitness_score(&self) -> f64 {
        let nps_normalized = (self.nps + 100.0) / 200.0; // normalize -100..100 to 0..1

        ((1.0 - self.churn_rate) * 0.35)
            + (self.mau_retention * 0.35)
            + (nps_normalized * 0.20)
            + (self.feature_adoption_rate * 0.10)
    }
}

/// Unified domain metrics container for multi-domain evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DomainMetrics {
    WebDesign(WebDesignMetrics),
    Marketing(MarketingMetrics),
    Retention(RetentionMetrics),
}

impl DomainMetrics {
    /// Get the fitness score for this domain's metrics (0.0-1.0).
    pub fn fitness_score(&self) -> f64 {
        match self {
            DomainMetrics::WebDesign(metrics) => metrics.fitness_score(),
            DomainMetrics::Marketing(metrics) => metrics.fitness_score(),
            DomainMetrics::Retention(metrics) => metrics.fitness_score(),
        }
    }

    /// Create a baseline metric (neutral/mediocre performance).
    pub fn baseline_web_design() -> Self {
        DomainMetrics::WebDesign(WebDesignMetrics {
            conversion_rate: 0.02,
            bounce_rate: 0.45,
            avg_time_on_page: 60.0,
            engagement_score: 50.0,
            scroll_depth: 0.5,
        })
    }

    /// Create a baseline metric for marketing.
    pub fn baseline_marketing() -> Self {
        DomainMetrics::Marketing(MarketingMetrics {
            ctr: 0.02,
            impressions: 10000,
            cost_per_acquisition: 50.0,
            lead_quality: 50.0,
            customer_lifetime_value: 500.0,
        })
    }

    /// Create a baseline metric for retention.
    pub fn baseline_retention() -> Self {
        DomainMetrics::Retention(RetentionMetrics {
            churn_rate: 0.05,
            mau_retention: 0.95,
            nps: 30.0,
            feature_adoption_rate: 0.6,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_design_fitness_calculation() {
        let metrics = WebDesignMetrics {
            conversion_rate: 0.04,
            bounce_rate: 0.35,
            avg_time_on_page: 120.0,
            engagement_score: 75.0,
            scroll_depth: 0.8,
        };
        let fitness = metrics.fitness_score();
        // Weights: conversion (40%), engagement (30%), bounce (20%), scroll (10%)
        // fitness = 0.04*0.4 + 0.75*0.3 + 0.65*0.2 + 0.8*0.1 = 0.451
        assert!(fitness > 0.4 && fitness < 0.6);
    }

    #[test]
    fn marketing_fitness_calculation() {
        let metrics = MarketingMetrics {
            ctr: 0.04,
            impressions: 50000,
            cost_per_acquisition: 30.0,
            lead_quality: 80.0,
            customer_lifetime_value: 1000.0,
        };
        let fitness = metrics.fitness_score();
        assert!(fitness > 0.6 && fitness < 0.9);
    }

    #[test]
    fn retention_fitness_calculation() {
        let metrics = RetentionMetrics {
            churn_rate: 0.02,
            mau_retention: 0.98,
            nps: 50.0,
            feature_adoption_rate: 0.85,
        };
        let fitness = metrics.fitness_score();
        assert!(fitness > 0.8 && fitness < 1.0);
    }

    #[test]
    fn baseline_web_design_fitness_is_neutral() {
        let baseline = DomainMetrics::baseline_web_design();
        let fitness = baseline.fitness_score();
        assert!(fitness > 0.3 && fitness < 0.7);
    }

    #[test]
    fn baseline_marketing_fitness_is_neutral() {
        let baseline = DomainMetrics::baseline_marketing();
        let fitness = baseline.fitness_score();
        assert!(fitness > 0.3 && fitness < 0.7);
    }

    #[test]
    fn baseline_retention_fitness_is_neutral() {
        let baseline = DomainMetrics::baseline_retention();
        let fitness = baseline.fitness_score();
        assert!(fitness > 0.7 && fitness < 1.0);
    }

    #[test]
    fn improved_web_design_has_higher_fitness_than_baseline() {
        let baseline = WebDesignMetrics {
            conversion_rate: 0.02,
            bounce_rate: 0.45,
            avg_time_on_page: 60.0,
            engagement_score: 50.0,
            scroll_depth: 0.5,
        };

        let improved = WebDesignMetrics {
            conversion_rate: 0.03,
            bounce_rate: 0.35,
            avg_time_on_page: 90.0,
            engagement_score: 70.0,
            scroll_depth: 0.7,
        };

        assert!(improved.fitness_score() > baseline.fitness_score());
    }
}
