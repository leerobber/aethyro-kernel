//! Study design: generate experimental designs for research validation.
//!
//! Phase 6.11 study design component:
//! 1. Experimental design generation: RCT, quasi-experimental, observational designs
//! 2. Control/treatment group setup: balanced randomization and stratification
//! 3. Sample size calculation: power analysis and effect size estimation
//! 4. Protocol generation: detailed methodology and statistical plan
//! 5. Validity threats assessment: internal, external, construct, statistical validity

use super::adaptive::Domain;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Type of experimental design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DesignType {
    /// Randomized controlled trial (gold standard)
    RandomizedControlledTrial,
    /// Quasi-experimental with matched controls
    QuasiExperimental,
    /// Observational with propensity score matching
    Observational,
    /// Within-subjects repeated measures
    RepeatedMeasures,
    /// Multi-arm trial with multiple treatment conditions
    MultiArm { arms: usize },
}

/// Allocation strategy for study arms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Simple randomization (50/50 split)
    SimpleRandomization,
    /// Stratified randomization (balanced within strata)
    StratifiedRandomization { strata: Vec<String> },
    /// Minimization to balance covariates
    MinimizationBalance,
    /// Adaptive allocation (Bayesian response-adaptive)
    AdaptiveAllocation,
}

/// Statistical power analysis result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Effect size (Cohen's d or similar)
    pub effect_size: f64,
    /// Statistical power (1 - beta, typically 0.80-0.90)
    pub power: f64,
    /// Significance level (alpha, typically 0.05)
    pub alpha: f64,
    /// Sample size per arm needed for adequate power
    pub sample_size_per_arm: usize,
    /// Total sample size
    pub total_sample_size: usize,
    /// Confidence interval for effect size (lower, upper)
    pub ci_95: (f64, f64),
}

/// Study protocol: detailed experimental methodology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyProtocol {
    /// Study title
    pub title: String,
    /// Research question
    pub research_question: String,
    /// Hypothesized effect size
    pub hypothesized_effect_size: f64,
    /// Primary outcome measure
    pub primary_outcome: String,
    /// Secondary outcomes
    pub secondary_outcomes: Vec<String>,
    /// Study duration (in cycles/iterations)
    pub duration_cycles: usize,
    /// Inclusion criteria
    pub inclusion_criteria: Vec<String>,
    /// Exclusion criteria
    pub exclusion_criteria: Vec<String>,
    /// Intervention description
    pub intervention_description: String,
    /// Control condition description
    pub control_description: String,
    /// Data collection points (cycle numbers)
    pub assessment_timepoints: Vec<usize>,
}

/// Study arm assignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyArm {
    /// Arm name (e.g., "Treatment", "Control", "Intervention A")
    pub name: String,
    /// Arm description
    pub description: String,
    /// Size of this arm
    pub size: usize,
    /// Whether this is the control condition
    pub is_control: bool,
    /// Treatment/intervention applied in this arm
    pub treatment: String,
}

/// Experimental design specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentalDesign {
    /// Design type
    pub design_type: DesignType,
    /// Problem domain
    pub domain: Domain,
    /// Study arms
    pub arms: Vec<StudyArm>,
    /// Allocation strategy
    pub allocation: AllocationStrategy,
    /// Power analysis results
    pub power_analysis: PowerAnalysis,
    /// Study protocol
    pub protocol: StudyProtocol,
    /// Number of replicates/iterations
    pub replicates: usize,
    /// Planned statistical tests
    pub statistical_tests: Vec<String>,
}

/// Validity threats assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidityThreatAssessment {
    /// Internal validity threats (confounding, selection bias, etc.)
    pub internal_threats: Vec<(String, String)>, // (threat, mitigation)
    /// External validity threats (generalizability)
    pub external_threats: Vec<(String, String)>,
    /// Construct validity threats (measurement, operationalization)
    pub construct_threats: Vec<(String, String)>,
    /// Statistical validity threats (power, assumptions)
    pub statistical_threats: Vec<(String, String)>,
}

/// Study design engine: generate experimental designs.
pub struct StudyDesignEngine {
    /// Generated designs
    designs: Vec<ExperimentalDesign>,
    /// Validity assessments
    validity_assessments: HashMap<String, ValidityThreatAssessment>,
    /// Design history (for learning)
    design_history: Vec<(String, f64)>, // (design_name, estimated_validity_score)
}

impl StudyDesignEngine {
    pub fn new() -> Self {
        Self {
            designs: Vec::new(),
            validity_assessments: HashMap::new(),
            design_history: Vec::new(),
        }
    }

    /// Generate a randomized controlled trial design.
    pub fn generate_rct(
        &mut self,
        title: String,
        domain: Domain,
        hypothesized_effect_size: f64,
        desired_power: f64,
        alpha: f64,
    ) -> ExperimentalDesign {
        // Calculate sample size from power analysis
        let effect_size = hypothesized_effect_size;
        let sample_per_arm = Self::calculate_sample_size(effect_size, desired_power, alpha);

        let power_analysis = PowerAnalysis {
            effect_size,
            power: desired_power,
            alpha,
            sample_size_per_arm: sample_per_arm,
            total_sample_size: sample_per_arm * 2,
            ci_95: (effect_size - 1.96 * (1.0 / (sample_per_arm as f64).sqrt()),
                    effect_size + 1.96 * (1.0 / (sample_per_arm as f64).sqrt())),
        };

        let control_arm = StudyArm {
            name: "Control".to_string(),
            description: "Standard optimization approach".to_string(),
            size: sample_per_arm,
            is_control: true,
            treatment: "Baseline".to_string(),
        };

        let treatment_arm = StudyArm {
            name: "Treatment".to_string(),
            description: "Novel optimization strategy".to_string(),
            size: sample_per_arm,
            is_control: false,
            treatment: "Novel mutation strategy".to_string(),
        };

        let protocol = StudyProtocol {
            title: title.clone(),
            research_question: format!("Does the novel strategy improve optimization in {:?}?", domain),
            hypothesized_effect_size: effect_size,
            primary_outcome: "Optimization efficiency improvement".to_string(),
            secondary_outcomes: vec![
                "Solution quality".to_string(),
                "Convergence speed".to_string(),
            ],
            duration_cycles: 100,
            inclusion_criteria: vec![
                "Valid graph representation".to_string(),
                "Optimization metric available".to_string(),
            ],
            exclusion_criteria: vec![
                "Incomplete data".to_string(),
                "Corrupted optimization state".to_string(),
            ],
            intervention_description: "Apply novel mutation strategy with adaptive control".to_string(),
            control_description: "Apply standard optimization without novel strategies".to_string(),
            assessment_timepoints: vec![10, 25, 50, 75, 100],
        };

        let design = ExperimentalDesign {
            design_type: DesignType::RandomizedControlledTrial,
            domain,
            arms: vec![control_arm, treatment_arm],
            allocation: AllocationStrategy::SimpleRandomization,
            power_analysis,
            protocol,
            replicates: 3,
            statistical_tests: vec![
                "Independent t-test".to_string(),
                "Mann-Whitney U test".to_string(),
                "Effect size calculation (Cohen's d)".to_string(),
            ],
        };

        self.designs.push(design.clone());
        design
    }

    /// Generate a multi-arm trial design (comparing multiple treatments).
    pub fn generate_multi_arm(
        &mut self,
        title: String,
        domain: Domain,
        treatments: Vec<String>,
        hypothesized_effect_size: f64,
        desired_power: f64,
        alpha: f64,
    ) -> ExperimentalDesign {
        let num_arms = treatments.len() + 1; // +1 for control

        // Adjust sample size for multiple comparisons
        let base_sample = Self::calculate_sample_size(hypothesized_effect_size, desired_power, alpha);
        let adjusted_sample = ((base_sample as f64) * (num_arms as f64 / 2.0).sqrt()) as usize;

        let mut arms = vec![StudyArm {
            name: "Control".to_string(),
            description: "Standard baseline".to_string(),
            size: adjusted_sample,
            is_control: true,
            treatment: "Baseline".to_string(),
        }];

        for (idx, treatment) in treatments.iter().enumerate() {
            arms.push(StudyArm {
                name: format!("Treatment {}", idx + 1),
                description: format!("Experimental treatment: {}", treatment),
                size: adjusted_sample,
                is_control: false,
                treatment: treatment.clone(),
            });
        }

        let power_analysis = PowerAnalysis {
            effect_size: hypothesized_effect_size,
            power: desired_power,
            alpha,
            sample_size_per_arm: adjusted_sample,
            total_sample_size: adjusted_sample * num_arms,
            ci_95: (hypothesized_effect_size - 0.2, hypothesized_effect_size + 0.2),
        };

        let protocol = StudyProtocol {
            title: title.clone(),
            research_question: format!("Which treatment is most effective in {:?}?", domain),
            hypothesized_effect_size,
            primary_outcome: "Relative treatment effectiveness".to_string(),
            secondary_outcomes: vec!["Safety".to_string(), "Robustness".to_string()],
            duration_cycles: 100,
            inclusion_criteria: vec!["Valid optimization setup".to_string()],
            exclusion_criteria: vec!["Failed setups".to_string()],
            intervention_description: format!("Compare {} treatment conditions", treatments.len()),
            control_description: "Standard optimization approach".to_string(),
            assessment_timepoints: vec![25, 50, 75, 100],
        };

        let design = ExperimentalDesign {
            design_type: DesignType::MultiArm { arms: num_arms },
            domain,
            arms,
            allocation: AllocationStrategy::StratifiedRandomization {
                strata: vec!["Domain".to_string()],
            },
            power_analysis,
            protocol,
            replicates: 2,
            statistical_tests: vec![
                "One-way ANOVA".to_string(),
                "Kruskal-Wallis test".to_string(),
                "Post-hoc pairwise comparisons".to_string(),
            ],
        };

        self.designs.push(design.clone());
        design
    }

    /// Calculate required sample size using simple power calculation.
    fn calculate_sample_size(effect_size: f64, power: f64, alpha: f64) -> usize {
        // Simplified formula: n = (z_alpha + z_beta)^2 * 2 * sigma^2 / d^2
        // Using standard normal approximation
        let z_alpha = Self::quantile_normal(1.0 - alpha / 2.0); // two-tailed
        let z_beta = Self::quantile_normal(power);

        let n = ((z_alpha + z_beta).powi(2) * 2.0) / (effect_size.max(0.1).powi(2));
        (n.ceil() as usize).max(20) // Minimum 20 per arm
    }

    /// Approximation of inverse normal CDF (quantile).
    fn quantile_normal(p: f64) -> f64 {
        if p < 0.5 {
            -Self::quantile_normal(1.0 - p)
        } else if p < 0.5000001 {
            0.0
        } else if p < 0.8413 {
            ((p - 0.5) * 8.0).sqrt()
        } else {
            2.807 * (-((-2.0 * (p - 0.5).ln()).sqrt())).exp()
        }
    }

    /// Assess validity threats in a design.
    pub fn assess_validity(&mut self, design: &ExperimentalDesign) -> ValidityThreatAssessment {
        let mut assessment = ValidityThreatAssessment {
            internal_threats: vec![],
            external_threats: vec![],
            construct_threats: vec![],
            statistical_threats: vec![],
        };

        // Internal validity threats (design-type specific)
        match &design.design_type {
            DesignType::RandomizedControlledTrial => {
                assessment.internal_threats.push((
                    "Selection bias".to_string(),
                    "Random allocation minimizes selection bias".to_string(),
                ));
            }
            DesignType::QuasiExperimental => {
                assessment.internal_threats.push((
                    "Selection bias (HIGH RISK)".to_string(),
                    "Use matching or regression adjustment".to_string(),
                ));
            }
            DesignType::Observational => {
                assessment.internal_threats.push((
                    "Confounding (HIGH RISK)".to_string(),
                    "Use propensity score matching or adjustment".to_string(),
                ));
            }
            _ => {}
        }

        // External validity threats
        assessment.external_threats.push((
            "Limited domain generalization".to_string(),
            format!("Test in multiple domains beyond {:?}", design.domain),
        ));

        // Construct validity
        assessment.construct_threats.push((
            "Measurement reliability".to_string(),
            "Use validated optimization metrics".to_string(),
        ));

        // Statistical validity
        let min_recommended_n = 30;
        if design.power_analysis.sample_size_per_arm < min_recommended_n {
            assessment.statistical_threats.push((
                "Insufficient sample size".to_string(),
                format!("Increase to minimum {} per arm", min_recommended_n),
            ));
        }

        self.validity_assessments.insert(
            design.protocol.title.clone(),
            assessment.clone(),
        );

        assessment
    }

    /// Calculate overall design validity score (0.0-1.0).
    pub fn design_validity_score(&self, design: &ExperimentalDesign) -> f64 {
        let mut score = 1.0;

        // Deduct for design weakness
        let weakness = match &design.design_type {
            DesignType::RandomizedControlledTrial => 0.0,
            DesignType::QuasiExperimental => 0.1,
            DesignType::Observational => 0.2,
            DesignType::RepeatedMeasures => 0.05,
            DesignType::MultiArm { .. } => 0.05,
        };
        score -= weakness;

        // Deduct for low power
        if design.power_analysis.power < 0.80 {
            score -= (0.80 - design.power_analysis.power) * 0.2;
        }

        // Deduct for small sample size
        if design.power_analysis.sample_size_per_arm < 30 {
            score -= 0.1;
        }

        score.max(0.0).min(1.0)
    }

    /// Generate study protocol document.
    pub fn generate_protocol_document(&self, design: &ExperimentalDesign) -> String {
        let mut doc = String::new();
        doc.push_str(&format!("=== Study Protocol: {} ===\n\n", design.protocol.title));

        doc.push_str("RESEARCH QUESTION:\n");
        doc.push_str(&format!("{}\n\n", design.protocol.research_question));

        doc.push_str("STUDY DESIGN:\n");
        doc.push_str(&format!("{:?}\n\n", design.design_type));

        doc.push_str("SAMPLE SIZE & POWER:\n");
        doc.push_str(&format!(
            "Effect size: {:.2}\n",
            design.power_analysis.effect_size
        ));
        doc.push_str(&format!(
            "Power: {:.1}%\n",
            design.power_analysis.power * 100.0
        ));
        doc.push_str(&format!(
            "Sample per arm: {}\n",
            design.power_analysis.sample_size_per_arm
        ));
        doc.push_str(&format!(
            "Total N: {}\n\n",
            design.power_analysis.total_sample_size
        ));

        doc.push_str("STUDY ARMS:\n");
        for arm in &design.arms {
            doc.push_str(&format!("  - {}: n={} ({})\n", arm.name, arm.size, arm.description));
        }
        doc.push_str("\n");

        doc.push_str("PRIMARY OUTCOME:\n");
        doc.push_str(&format!("{}\n\n", design.protocol.primary_outcome));

        doc.push_str("SECONDARY OUTCOMES:\n");
        for outcome in &design.protocol.secondary_outcomes {
            doc.push_str(&format!("  - {}\n", outcome));
        }
        doc.push_str("\n");

        doc.push_str("ASSESSMENT TIMEPOINTS:\n");
        doc.push_str(&format!("Cycles: {:?}\n\n", design.protocol.assessment_timepoints));

        doc.push_str("STATISTICAL TESTS:\n");
        for test in &design.statistical_tests {
            doc.push_str(&format!("  - {}\n", test));
        }

        doc
    }

    /// Generate study design report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Study Design Report ===\n");
        report.push_str(&format!("Designs generated: {}\n", self.designs.len()));

        if !self.designs.is_empty() {
            let avg_validity: f64 = self.designs.iter()
                .map(|d| self.design_validity_score(d))
                .sum::<f64>() / self.designs.len() as f64;

            report.push_str(&format!("Average design validity: {:.2}\n", avg_validity));

            report.push_str("\nRecent designs:\n");
            for design in self.designs.iter().rev().take(3) {
                let validity = self.design_validity_score(design);
                report.push_str(&format!(
                    "  \"{}\": {:?}, n={}, validity={:.2}\n",
                    design.protocol.title,
                    design.design_type,
                    design.power_analysis.total_sample_size,
                    validity
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn study_design_generate_rct() {
        let mut engine = StudyDesignEngine::new();
        let design = engine.generate_rct(
            "Test Study".to_string(),
            Domain::Ranking,
            0.5,
            0.80,
            0.05,
        );

        assert_eq!(design.arms.len(), 2);
        assert_eq!(design.protocol.title, "Test Study");
        assert!(design.power_analysis.sample_size_per_arm >= 20);
    }

    #[test]
    fn study_design_multi_arm() {
        let mut engine = StudyDesignEngine::new();
        let design = engine.generate_multi_arm(
            "Multi-arm Study".to_string(),
            Domain::Classification,
            vec!["Strategy A".to_string(), "Strategy B".to_string()],
            0.4,
            0.85,
            0.05,
        );

        assert_eq!(design.arms.len(), 3); // control + 2 treatments
        assert!(design.power_analysis.total_sample_size > 0);
    }

    #[test]
    fn study_design_sample_size_calculation() {
        // Larger effect size should require smaller sample
        let n1 = StudyDesignEngine::calculate_sample_size(0.8, 0.80, 0.05);
        let n2 = StudyDesignEngine::calculate_sample_size(0.5, 0.80, 0.05);
        assert!(n1 < n2);
    }

    #[test]
    fn study_design_validity_assessment() {
        let mut engine = StudyDesignEngine::new();
        let design = engine.generate_rct(
            "Validity Test".to_string(),
            Domain::Search,
            0.6,
            0.90,
            0.05,
        );

        let assessment = engine.assess_validity(&design);
        assert!(!assessment.internal_threats.is_empty());
        assert!(!assessment.external_threats.is_empty());
    }

    #[test]
    fn study_design_validity_score() {
        let mut engine = StudyDesignEngine::new();
        let design = engine.generate_rct(
            "Score Test".to_string(),
            Domain::Generic,
            0.7,
            0.85,
            0.05,
        );

        let score = engine.design_validity_score(&design);
        assert!(score >= 0.0 && score <= 1.0);
        assert!(score > 0.7); // RCT with good power should score high
    }

    #[test]
    fn study_design_protocol_document() {
        let mut engine = StudyDesignEngine::new();
        let design = engine.generate_rct(
            "Protocol Doc Test".to_string(),
            Domain::Ranking,
            0.5,
            0.80,
            0.05,
        );

        let doc = engine.generate_protocol_document(&design);
        assert!(doc.contains("Study Protocol"));
        assert!(doc.contains("RESEARCH QUESTION"));
        assert!(doc.contains("SAMPLE SIZE"));
    }

    #[test]
    fn study_design_report_is_readable() {
        let mut engine = StudyDesignEngine::new();
        engine.generate_rct(
            "Report Test 1".to_string(),
            Domain::Classification,
            0.6,
            0.80,
            0.05,
        );
        engine.generate_rct(
            "Report Test 2".to_string(),
            Domain::Search,
            0.5,
            0.85,
            0.05,
        );

        let report = engine.report();
        assert!(report.contains("Study Design Report"));
        assert!(report.contains("Designs generated: 2"));
        assert!(report.contains("Average design validity"));
    }
}
