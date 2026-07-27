//! Peer review simulation: automated reviewer personas and feedback generation.
//!
//! Phase 6.11 peer review component:
//! 1. Reviewer personas: diverse expert profiles with different perspectives
//! 2. Critical feedback generation: constructive criticism aligned to reviewer expertise
//! 3. Acceptance/rejection decision: aggregate reviewer scores and comments
//! 4. Revision recommendations: prioritized list of improvements
//! 5. Rebuttal handling: track author responses to reviewer concerns

use serde::{Serialize, Deserialize};

/// Reviewer expertise area.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ExpertiseArea {
    MethodologyRigour,
    StatisticalValidity,
    NoveltyAndSignificance,
    Reproducibility,
    GeneralizationAndTransfer,
    PracticalApplicability,
}

/// Reviewer tone preference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReviewerTone {
    /// Harsh, critical, demanding high standards
    Critical,
    /// Balanced, fair, constructive
    Balanced,
    /// Supportive, encouraging, highlights strengths
    Supportive,
}

/// Review score for a specific criterion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriterionScore {
    pub criterion: String,
    pub score: f64, // 1.0-5.0
    pub justification: String,
}

/// Reviewer feedback on a submission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewerFeedback {
    /// Reviewer ID/persona
    pub reviewer_id: String,
    /// Overall assessment score (1.0-5.0)
    pub overall_score: f64,
    /// Recommendation: "Accept", "Minor Revisions", "Major Revisions", "Reject"
    pub recommendation: String,
    /// Detailed comments
    pub comments: String,
    /// Criterion scores
    pub criterion_scores: Vec<CriterionScore>,
    /// Major concerns (bullets)
    pub major_concerns: Vec<String>,
    /// Minor suggestions (bullets)
    pub minor_suggestions: Vec<String>,
    /// Confidence in assessment (0.0-1.0)
    pub confidence: f64,
}

/// Reviewer persona: distinct expert with expertise and preferences.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewerPersona {
    pub id: String,
    pub name: String,
    pub expertise: ExpertiseArea,
    pub tone: ReviewerTone,
    /// Acceptance threshold (mean score needed to recommend accept/minor revisions)
    pub acceptance_threshold: f64,
    /// Typical harshness (affects scoring distribution)
    pub harshness: f64, // 0.0-1.0
}

/// Revision recommendation: a prioritized improvement suggestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevisionRecommendation {
    pub category: String, // "Methods", "Results", "Discussion", etc.
    pub priority: String, // "Critical", "Important", "Minor"
    pub suggestion: String,
    pub justification: String,
}

/// Rebuttal: author's response to reviewer comments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rebuttal {
    pub reviewer_id: String,
    pub original_concern: String,
    pub author_response: String,
    pub changes_made: Vec<String>,
    pub resolved: bool,
}

/// Review round results: aggregated feedback from all reviewers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewRoundResults {
    pub round_number: u32,
    pub reviews: Vec<ReviewerFeedback>,
    pub mean_score: f64,
    pub consensus_recommendation: String,
    pub revision_recommendations: Vec<RevisionRecommendation>,
    pub rebuttals: Vec<Rebuttal>,
}

/// Peer review engine: simulate peer review process.
pub struct PeerReviewEngine {
    /// Collection of reviewer personas
    reviewers: Vec<ReviewerPersona>,
    /// Review history
    review_history: Vec<ReviewRoundResults>,
    /// Acceptance rate (for learning)
    acceptance_rate: f64,
}

impl PeerReviewEngine {
    pub fn new() -> Self {
        Self {
            reviewers: Self::initialize_reviewers(),
            review_history: Vec::new(),
            acceptance_rate: 0.35,
        }
    }

    /// Initialize a diverse set of reviewer personas.
    fn initialize_reviewers() -> Vec<ReviewerPersona> {
        vec![
            ReviewerPersona {
                id: "R1".to_string(),
                name: "Methods Expert".to_string(),
                expertise: ExpertiseArea::MethodologyRigour,
                tone: ReviewerTone::Critical,
                acceptance_threshold: 3.8,
                harshness: 0.8,
            },
            ReviewerPersona {
                id: "R2".to_string(),
                name: "Statistician".to_string(),
                expertise: ExpertiseArea::StatisticalValidity,
                tone: ReviewerTone::Balanced,
                acceptance_threshold: 3.5,
                harshness: 0.5,
            },
            ReviewerPersona {
                id: "R3".to_string(),
                name: "Innovation Evaluator".to_string(),
                expertise: ExpertiseArea::NoveltyAndSignificance,
                tone: ReviewerTone::Supportive,
                acceptance_threshold: 3.2,
                harshness: 0.3,
            },
            ReviewerPersona {
                id: "R4".to_string(),
                name: "Reproducibility Advocate".to_string(),
                expertise: ExpertiseArea::Reproducibility,
                tone: ReviewerTone::Balanced,
                acceptance_threshold: 3.6,
                harshness: 0.6,
            },
        ]
    }

    /// Generate review from a specific reviewer persona.
    pub fn generate_review(
        &self,
        reviewer: &ReviewerPersona,
        paper_quality: f64,
        novelty: f64,
        rigor: f64,
    ) -> ReviewerFeedback {
        // Adjust scores based on reviewer's expertise and harshness
        let base_score = (paper_quality * 0.4 + novelty * 0.3 + rigor * 0.3) * 5.0;
        let harshness_adjustment = (1.0 - reviewer.harshness) * 0.5; // -0.5 to 0
        let score_adjustment = match reviewer.expertise {
            ExpertiseArea::MethodologyRigour => rigor * 0.3,
            ExpertiseArea::StatisticalValidity => rigor * 0.4 - 0.2,
            ExpertiseArea::NoveltyAndSignificance => novelty * 0.4 - 0.1,
            ExpertiseArea::Reproducibility => rigor * 0.25 - 0.15,
            ExpertiseArea::GeneralizationAndTransfer => novelty * 0.2 - 0.1,
            ExpertiseArea::PracticalApplicability => paper_quality * 0.3 - 0.1,
        };

        let overall_score = (base_score + score_adjustment + harshness_adjustment)
            .max(1.0)
            .min(5.0);

        // Generate recommendation
        let recommendation = if overall_score >= reviewer.acceptance_threshold {
            if overall_score >= 4.2 {
                "Accept".to_string()
            } else {
                "Minor Revisions".to_string()
            }
        } else if overall_score >= 3.0 {
            "Major Revisions".to_string()
        } else {
            "Reject".to_string()
        };

        // Generate criterion scores
        let criterion_scores = vec![
            CriterionScore {
                criterion: "Scientific Rigor".to_string(),
                score: (rigor * 5.0 * (1.0 - reviewer.harshness * 0.2))
                    .max(1.0)
                    .min(5.0),
                justification: "Methods appear well-designed and assumptions stated".to_string(),
            },
            CriterionScore {
                criterion: "Novelty & Significance".to_string(),
                score: (novelty * 5.0 + 0.5).max(1.0).min(5.0),
                justification: "Contribution advances understanding in the field".to_string(),
            },
            CriterionScore {
                criterion: "Clarity & Presentation".to_string(),
                score: (paper_quality * 5.0 * 0.9).max(1.0).min(5.0),
                justification: "Writing is generally clear with minor improvements needed".to_string(),
            },
            CriterionScore {
                criterion: "Significance of Findings".to_string(),
                score: ((paper_quality + novelty) / 2.0 * 5.0).max(1.0).min(5.0),
                justification: "Results have meaningful implications".to_string(),
            },
        ];

        // Generate concerns and suggestions
        let major_concerns = if overall_score < 3.5 {
            vec![
                "Limited experimental validation across domains".to_string(),
                "Insufficient statistical power in some analyses".to_string(),
                "Unclear generalization beyond tested scenarios".to_string(),
            ]
        } else {
            vec![]
        };

        let minor_suggestions = vec![
            "Expand discussion of limitations".to_string(),
            "Add more implementation details for reproducibility".to_string(),
            "Include sensitivity analysis for key assumptions".to_string(),
        ];

        let confidence = (rigor + 0.2).min(1.0);

        let comments = format!(
            "This paper presents {} novel approaches to mutation-based optimization. \
            The experimental design is {}, though concerns remain about {}. \
            Overall, the work makes a {} contribution to the field.",
            if novelty > 0.7 { "highly" } else { "moderately" },
            if rigor > 0.7 { "solid" } else { "adequate" },
            if rigor < 0.6 { "statistical power and generalization" } else { "practical applicability" },
            if overall_score >= 4.0 { "significant" } else if overall_score >= 3.5 { "moderate" } else { "limited" }
        );

        ReviewerFeedback {
            reviewer_id: reviewer.id.clone(),
            overall_score,
            recommendation,
            comments,
            criterion_scores,
            major_concerns,
            minor_suggestions,
            confidence,
        }
    }

    /// Conduct a full review round with all reviewers.
    pub fn review_paper(
        &mut self,
        paper_quality: f64,
        novelty: f64,
        rigor: f64,
    ) -> ReviewRoundResults {
        let mut reviews = Vec::new();
        let round_number = (self.review_history.len() + 1) as u32;

        // Collect reviews from all reviewers
        for reviewer in &self.reviewers {
            let review = self.generate_review(reviewer, paper_quality, novelty, rigor);
            reviews.push(review);
        }

        // Calculate mean score and consensus
        let mean_score: f64 = reviews.iter().map(|r| r.overall_score).sum::<f64>() / reviews.len() as f64;

        let accept_count = reviews.iter()
            .filter(|r| r.recommendation == "Accept")
            .count();
        let minor_count = reviews.iter()
            .filter(|r| r.recommendation == "Minor Revisions")
            .count();

        let consensus_recommendation = if accept_count >= 3 {
            "Accept".to_string()
        } else if accept_count >= 2 || minor_count >= 2 {
            "Minor Revisions".to_string()
        } else if reviews.iter().all(|r| r.recommendation == "Reject") {
            "Reject".to_string()
        } else {
            "Major Revisions".to_string()
        };

        // Generate revision recommendations based on reviewer feedback
        let mut revision_recommendations = Vec::new();
        for review in &reviews {
            for concern in &review.major_concerns {
                revision_recommendations.push(RevisionRecommendation {
                    category: "Methods & Analysis".to_string(),
                    priority: "Critical".to_string(),
                    suggestion: concern.clone(),
                    justification: format!("Raised by {}", review.reviewer_id),
                });
            }
            for suggestion in review.minor_suggestions.iter().take(1) {
                revision_recommendations.push(RevisionRecommendation {
                    category: "Presentation".to_string(),
                    priority: "Minor".to_string(),
                    suggestion: suggestion.clone(),
                    justification: format!("Suggested by {}", review.reviewer_id),
                });
            }
        }

        let result = ReviewRoundResults {
            round_number,
            reviews,
            mean_score,
            consensus_recommendation,
            revision_recommendations,
            rebuttals: Vec::new(),
        };

        self.review_history.push(result.clone());
        result
    }

    /// Process author rebuttal to reviewer concerns.
    pub fn process_rebuttal(
        &mut self,
        round_number: u32,
        rebuttal: Rebuttal,
    ) {
        if let Some(round) = self.review_history.iter_mut().find(|r| r.round_number == round_number) {
            round.rebuttals.push(rebuttal);
        }
    }

    /// Determine if paper is ready for publication after review rounds.
    pub fn publication_readiness(&self) -> (bool, f64) {
        if self.review_history.is_empty() {
            return (false, 0.0);
        }

        let latest = self.review_history.last().unwrap();

        // Ready if: Accept recommendation AND high mean score
        let is_ready = latest.consensus_recommendation == "Accept" && latest.mean_score >= 3.8;

        // Readiness score: 0.0-1.0
        let readiness = if latest.consensus_recommendation == "Accept" {
            latest.mean_score / 5.0
        } else if latest.consensus_recommendation == "Minor Revisions" {
            (latest.mean_score / 5.0) * 0.7
        } else if latest.consensus_recommendation == "Major Revisions" {
            (latest.mean_score / 5.0) * 0.3
        } else {
            0.0
        };

        (is_ready, readiness)
    }

    /// Get review history for tracking improvements.
    pub fn review_history(&self) -> &[ReviewRoundResults] {
        &self.review_history
    }

    /// Generate peer review report.
    pub fn report(&self) -> String {
        let mut report = String::from("=== Peer Review Report ===\n");
        report.push_str(&format!("Review rounds completed: {}\n", self.review_history.len()));

        if !self.review_history.is_empty() {
            let avg_score: f64 = self.review_history.iter()
                .map(|r| r.mean_score)
                .sum::<f64>() / self.review_history.len() as f64;

            report.push_str(&format!("Average review score: {:.2}/5.0\n", avg_score));

            let latest = self.review_history.last().unwrap();
            report.push_str(&format!(
                "Latest recommendation: {}\n",
                latest.consensus_recommendation
            ));

            report.push_str("\nReview progression:\n");
            for round in self.review_history.iter().rev().take(3) {
                report.push_str(&format!(
                    "  Round {}: {:.2}/5.0 - {}\n",
                    round.round_number,
                    round.mean_score,
                    round.consensus_recommendation
                ));
            }

            if !latest.revision_recommendations.is_empty() {
                report.push_str("\nCritical revisions needed:\n");
                for rec in latest.revision_recommendations.iter()
                    .filter(|r| r.priority == "Critical")
                    .take(3) {
                    report.push_str(&format!("  - {}\n", rec.suggestion));
                }
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_review_generate_review() {
        let engine = PeerReviewEngine::new();
        let reviewer = engine.reviewers[0].clone();
        let review = engine.generate_review(&reviewer, 0.8, 0.7, 0.9);

        assert!(!review.comments.is_empty());
        assert!(!review.recommendation.is_empty());
        assert!(review.overall_score >= 1.0 && review.overall_score <= 5.0);
    }

    #[test]
    fn peer_review_multiple_reviewers() {
        let mut engine = PeerReviewEngine::new();
        let results = engine.review_paper(0.85, 0.75, 0.80);

        assert_eq!(results.reviews.len(), 4); // 4 reviewers
        assert!(results.mean_score >= 1.0 && results.mean_score <= 5.0);
    }

    #[test]
    fn peer_review_recommendation_logic() {
        let mut engine = PeerReviewEngine::new();

        // High quality paper should get accept/minor
        let good_results = engine.review_paper(0.9, 0.85, 0.90);
        assert!(
            good_results.consensus_recommendation == "Accept"
                || good_results.consensus_recommendation == "Minor Revisions"
        );
    }

    #[test]
    fn peer_review_revision_recommendations() {
        let mut engine = PeerReviewEngine::new();
        let results = engine.review_paper(0.6, 0.5, 0.5);

        assert!(!results.revision_recommendations.is_empty());
    }

    #[test]
    fn peer_review_process_rebuttal() {
        let mut engine = PeerReviewEngine::new();
        let _ = engine.review_paper(0.7, 0.6, 0.7);

        let rebuttal = Rebuttal {
            reviewer_id: "R1".to_string(),
            original_concern: "Limited statistical power".to_string(),
            author_response: "We expanded the sample size in the revised version".to_string(),
            changes_made: vec!["Increased n from 100 to 150".to_string()],
            resolved: true,
        };

        engine.process_rebuttal(1, rebuttal);
        assert!(!engine.review_history.is_empty());
        assert!(!engine.review_history[0].rebuttals.is_empty());
    }

    #[test]
    fn peer_review_publication_readiness() {
        let mut engine = PeerReviewEngine::new();
        engine.review_paper(0.9, 0.85, 0.90);

        let (ready, score) = engine.publication_readiness();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn peer_review_report_is_readable() {
        let mut engine = PeerReviewEngine::new();
        engine.review_paper(0.8, 0.7, 0.8);

        let report = engine.report();
        assert!(report.contains("Peer Review Report"));
        assert!(report.contains("Review rounds completed"));
        assert!(report.contains("Average review score"));
    }
}
