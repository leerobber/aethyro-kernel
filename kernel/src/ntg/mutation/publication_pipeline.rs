//! Publication pipeline: end-to-end manuscript submission and publication workflow.
//!
//! Phase 6.11 publication component:
//! 1. Manuscript preparation: format and structure for submission
//! 2. Journal selection: target venues based on scope and impact
//! 3. Submission tracking: timeline and status monitoring
//! 4. Revision cycles: rounds of review and author response
//! 5. Publication metrics: acceptance rates, impact tracking

use super::adaptive::Domain;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Publication venue (journal/conference).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicationVenue {
    pub name: String,
    pub type_: String, // "Journal" or "Conference"
    pub impact_factor: f64, // 0.0-10.0
    pub acceptance_rate: f64, // 0.0-1.0
    pub review_timeline_weeks: u32,
    pub scope: Vec<String>, // relevant research areas
    pub tier: String, // "Top", "Mid", "Specialty"
}

/// Manuscript status throughout the publication pipeline.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ManuscriptStatus {
    Draft,
    ReadyForSubmission,
    Submitted,
    UnderReview,
    Rejected,
    MajorRevisions,
    MinorRevisions,
    Accepted,
    Published,
}

/// Submission tracking entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmissionRecord {
    pub submission_date: String,
    pub venue: PublicationVenue,
    pub status: ManuscriptStatus,
    pub editor_comment: String,
}

/// Revision cycle: one round of review and response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevisionCycle {
    pub cycle_number: u32,
    pub status_before: ManuscriptStatus,
    pub reviewer_feedback_count: usize,
    pub major_issues_addressed: u32,
    pub minor_issues_addressed: u32,
    pub changes_summary: String,
    pub status_after: ManuscriptStatus,
}

/// Manuscript metadata and publication history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manuscript {
    pub title: String,
    pub authors: Vec<String>,
    pub domain: Domain,
    pub abstract_text: String,
    pub keywords: Vec<String>,
    pub word_count: usize,
    pub current_status: ManuscriptStatus,
    pub submission_history: Vec<SubmissionRecord>,
    pub revision_cycles: Vec<RevisionCycle>,
    /// Citation count (tracks impact)
    pub citation_count: u32,
    /// Download/view count
    pub download_count: u32,
}

/// Publication outlet: selected venue for submission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicationOutlet {
    pub venue: PublicationVenue,
    pub suitability_score: f64, // 0.0-1.0
    pub recommendation_reason: String,
}

/// Publication pipeline: manage entire manuscript workflow.
pub struct PublicationPipeline {
    /// All available venues
    venues: Vec<PublicationVenue>,
    /// Manuscripts in the pipeline
    manuscripts: Vec<Manuscript>,
    /// Venue selections (for each manuscript)
    venue_selections: HashMap<String, Vec<PublicationOutlet>>,
    /// Publication metrics
    total_submitted: u32,
    total_accepted: u32,
}

impl PublicationPipeline {
    pub fn new() -> Self {
        Self {
            venues: Self::initialize_venues(),
            manuscripts: Vec::new(),
            venue_selections: HashMap::new(),
            total_submitted: 0,
            total_accepted: 0,
        }
    }

    /// Initialize a collection of target publication venues.
    fn initialize_venues() -> Vec<PublicationVenue> {
        vec![
            PublicationVenue {
                name: "Nature Machine Intelligence".to_string(),
                type_: "Journal".to_string(),
                impact_factor: 8.5,
                acceptance_rate: 0.12,
                review_timeline_weeks: 12,
                scope: vec![
                    "Machine learning".to_string(),
                    "Optimization".to_string(),
                    "Autonomous systems".to_string(),
                ],
                tier: "Top".to_string(),
            },
            PublicationVenue {
                name: "ICML (International Conference on Machine Learning)".to_string(),
                type_: "Conference".to_string(),
                impact_factor: 7.0,
                acceptance_rate: 0.22,
                review_timeline_weeks: 8,
                scope: vec![
                    "Machine learning".to_string(),
                    "Statistical learning".to_string(),
                ],
                tier: "Top".to_string(),
            },
            PublicationVenue {
                name: "Journal of Machine Learning Research".to_string(),
                type_: "Journal".to_string(),
                impact_factor: 6.2,
                acceptance_rate: 0.25,
                review_timeline_weeks: 10,
                scope: vec!["Machine learning".to_string()],
                tier: "Top".to_string(),
            },
            PublicationVenue {
                name: "Optimization Letters".to_string(),
                type_: "Journal".to_string(),
                impact_factor: 3.5,
                acceptance_rate: 0.35,
                review_timeline_weeks: 8,
                scope: vec!["Optimization".to_string(), "Algorithms".to_string()],
                tier: "Mid".to_string(),
            },
            PublicationVenue {
                name: "ACM Transactions on Evolutionary Learning and Optimization".to_string(),
                type_: "Journal".to_string(),
                impact_factor: 4.2,
                acceptance_rate: 0.40,
                review_timeline_weeks: 12,
                scope: vec!["Evolutionary algorithms".to_string(), "Mutation strategies".to_string()],
                tier: "Mid".to_string(),
            },
        ]
    }

    /// Create a new manuscript in the pipeline.
    pub fn create_manuscript(
        &mut self,
        title: String,
        authors: Vec<String>,
        domain: Domain,
        abstract_text: String,
        keywords: Vec<String>,
        word_count: usize,
    ) -> Manuscript {
        let manuscript = Manuscript {
            title,
            authors,
            domain,
            abstract_text,
            keywords,
            word_count,
            current_status: ManuscriptStatus::Draft,
            submission_history: Vec::new(),
            revision_cycles: Vec::new(),
            citation_count: 0,
            download_count: 0,
        };

        self.manuscripts.push(manuscript.clone());
        manuscript
    }

    /// Select best venues for a manuscript based on its content.
    pub fn select_venues(&mut self, manuscript_title: &str, num_venues: usize) -> Vec<PublicationOutlet> {
        let mut ranked_venues = Vec::new();

        for venue in &self.venues {
            // Calculate suitability based on scope match and tier
            let scope_match = if venue.scope.contains(&"Machine learning".to_string())
                || venue.scope.contains(&"Optimization".to_string()) {
                0.9
            } else {
                0.5
            };

            let tier_score = match venue.tier.as_str() {
                "Top" => 1.0,
                "Mid" => 0.7,
                "Specialty" => 0.6,
                _ => 0.5,
            };

            // Higher acceptance rate makes it more suitable for first submission
            let acceptance_bonus = (venue.acceptance_rate * 0.5) + 0.5;

            let suitability = (scope_match * 0.4 + tier_score * 0.4 + acceptance_bonus * 0.2).min(1.0);

            ranked_venues.push((venue.clone(), suitability));
        }

        // Sort by suitability and take top N
        ranked_venues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<PublicationOutlet> = ranked_venues
            .into_iter()
            .take(num_venues)
            .map(|(venue, score)| PublicationOutlet {
                venue,
                suitability_score: score,
                recommendation_reason: format!(
                    "Good match for this manuscript based on scope and acceptance rate"
                ),
            })
            .collect();

        self.venue_selections.insert(manuscript_title.to_string(), selected.clone());
        selected
    }

    /// Submit manuscript to a venue.
    pub fn submit_manuscript(
        &mut self,
        manuscript_title: &str,
        venue: &PublicationVenue,
    ) -> Result<(), String> {
        let manuscript_count = self.manuscripts.len();
        if let Some(manuscript) = self.manuscripts.iter_mut().find(|m| m.title == manuscript_title) {
            if manuscript.current_status != ManuscriptStatus::Draft
                && manuscript.current_status != ManuscriptStatus::ReadyForSubmission {
                return Err("Manuscript not in draft or ready status".to_string());
            }

            manuscript.submission_history.push(SubmissionRecord {
                submission_date: format!("2026-07-{}", 20 + manuscript_count as u32),
                venue: venue.clone(),
                status: ManuscriptStatus::Submitted,
                editor_comment: "Manuscript received and assigned to editor".to_string(),
            });

            manuscript.current_status = ManuscriptStatus::UnderReview;
            self.total_submitted += 1;
            Ok(())
        } else {
            Err("Manuscript not found".to_string())
        }
    }

    /// Record a revision cycle (review round + author response).
    pub fn add_revision_cycle(
        &mut self,
        manuscript_title: &str,
        major_issues: u32,
        minor_issues: u32,
        changes_summary: String,
        new_status: ManuscriptStatus,
    ) -> Result<(), String> {
        if let Some(manuscript) = self.manuscripts.iter_mut().find(|m| m.title == manuscript_title) {
            let cycle_number = (manuscript.revision_cycles.len() + 1) as u32;

            manuscript.revision_cycles.push(RevisionCycle {
                cycle_number,
                status_before: manuscript.current_status.clone(),
                reviewer_feedback_count: major_issues as usize + minor_issues as usize,
                major_issues_addressed: major_issues,
                minor_issues_addressed: minor_issues,
                changes_summary,
                status_after: new_status.clone(),
            });

            manuscript.current_status = new_status;

            if manuscript.current_status == ManuscriptStatus::Accepted {
                self.total_accepted += 1;
            }

            Ok(())
        } else {
            Err("Manuscript not found".to_string())
        }
    }

    /// Calculate manuscript readiness for publication.
    pub fn publication_readiness_score(&self, manuscript_title: &str) -> f64 {
        if let Some(manuscript) = self.manuscripts.iter().find(|m| m.title == manuscript_title) {
            match &manuscript.current_status {
                ManuscriptStatus::Published => 1.0,
                ManuscriptStatus::Accepted => 0.95,
                ManuscriptStatus::MinorRevisions => 0.7,
                ManuscriptStatus::MajorRevisions => 0.4,
                ManuscriptStatus::UnderReview => 0.5,
                ManuscriptStatus::Submitted => 0.5,
                ManuscriptStatus::ReadyForSubmission => 0.3,
                ManuscriptStatus::Draft => 0.0,
                ManuscriptStatus::Rejected => 0.0,
            }
        } else {
            0.0
        }
    }

    /// Estimate likelihood of acceptance at a venue.
    pub fn estimate_acceptance_likelihood(&self, manuscript: &Manuscript, venue: &PublicationVenue) -> f64 {
        // Base: venue's acceptance rate
        let mut likelihood = venue.acceptance_rate;

        // Adjust based on word count (longer papers have lower acceptance in top venues)
        if manuscript.word_count > 8000 && venue.tier == "Top" {
            likelihood *= 0.85;
        }

        // Adjust based on manuscript revision history
        let prior_rejections = manuscript
            .submission_history
            .iter()
            .filter(|s| s.status == ManuscriptStatus::Rejected)
            .count();

        if prior_rejections > 0 {
            likelihood *= (1.0 - (0.1 * prior_rejections as f64)).max(0.3);
        }

        likelihood.min(1.0)
    }

    /// Get publication metrics.
    pub fn publication_metrics(&self) -> (u32, u32, f64) {
        let acceptance_rate = if self.total_submitted > 0 {
            self.total_accepted as f64 / self.total_submitted as f64
        } else {
            0.0
        };

        (self.total_submitted, self.total_accepted, acceptance_rate)
    }

    /// Generate publication pipeline report.
    pub fn report(&self) -> String {
        let (total_submitted, total_accepted, acceptance_rate) = self.publication_metrics();

        let mut report = String::from("=== Publication Pipeline Report ===\n");
        report.push_str(&format!("Manuscripts in pipeline: {}\n", self.manuscripts.len()));
        report.push_str(&format!("Total submitted: {}\n", total_submitted));
        report.push_str(&format!("Total accepted: {}\n", total_accepted));
        report.push_str(&format!("Acceptance rate: {:.1}%\n\n", acceptance_rate * 100.0));

        report.push_str("Manuscript statuses:\n");
        let mut status_counts: HashMap<String, u32> = HashMap::new();
        for manuscript in &self.manuscripts {
            let status_str = format!("{:?}", manuscript.current_status);
            *status_counts.entry(status_str).or_insert(0) += 1;
        }

        for (status, count) in &status_counts {
            report.push_str(&format!("  {}: {}\n", status, count));
        }

        if !self.manuscripts.is_empty() {
            report.push_str("\nRecent manuscripts:\n");
            for manuscript in self.manuscripts.iter().rev().take(3) {
                report.push_str(&format!(
                    "  \"{}\": {:?}, {} revisions, {} citations\n",
                    manuscript.title,
                    manuscript.current_status,
                    manuscript.revision_cycles.len(),
                    manuscript.citation_count
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
    fn publication_create_manuscript() {
        let mut pipeline = PublicationPipeline::new();
        pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author 1".to_string()],
            Domain::Ranking,
            "Abstract text".to_string(),
            vec!["optimization".to_string()],
            5000,
        );

        assert_eq!(pipeline.manuscripts.len(), 1);
        assert_eq!(
            pipeline.manuscripts[0].current_status,
            ManuscriptStatus::Draft
        );
    }

    #[test]
    fn publication_select_venues() {
        let mut pipeline = PublicationPipeline::new();
        let outlets = pipeline.select_venues("Test Paper", 3);

        assert_eq!(outlets.len(), 3);
        for outlet in outlets {
            assert!(outlet.suitability_score >= 0.0 && outlet.suitability_score <= 1.0);
        }
    }

    #[test]
    fn publication_submit_manuscript() {
        let mut pipeline = PublicationPipeline::new();
        pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author".to_string()],
            Domain::Classification,
            "Abstract".to_string(),
            vec![],
            4000,
        );

        let venue = pipeline.venues[0].clone();
        let result = pipeline.submit_manuscript("Test Paper", &venue);

        assert!(result.is_ok());
        assert_eq!(
            pipeline.manuscripts[0].current_status,
            ManuscriptStatus::UnderReview
        );
    }

    #[test]
    fn publication_revision_cycle() {
        let mut pipeline = PublicationPipeline::new();
        pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author".to_string()],
            Domain::Search,
            "Abstract".to_string(),
            vec![],
            5000,
        );

        let venue = pipeline.venues[0].clone();
        let _ = pipeline.submit_manuscript("Test Paper", &venue);
        let result = pipeline.add_revision_cycle(
            "Test Paper",
            3,
            2,
            "Addressed all reviewer comments".to_string(),
            ManuscriptStatus::MinorRevisions,
        );

        assert!(result.is_ok());
        assert!(!pipeline.manuscripts[0].revision_cycles.is_empty());
    }

    #[test]
    fn publication_readiness_score() {
        let mut pipeline = PublicationPipeline::new();
        pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author".to_string()],
            Domain::Generic,
            "Abstract".to_string(),
            vec![],
            3000,
        );

        let score = pipeline.publication_readiness_score("Test Paper");
        assert_eq!(score, 0.0); // Draft status = 0.0

        let venue = pipeline.venues[0].clone();
        let _ = pipeline.submit_manuscript("Test Paper", &venue);
        let score_after = pipeline.publication_readiness_score("Test Paper");
        assert_eq!(score_after, 0.5); // Under review = 0.5
    }

    #[test]
    fn publication_acceptance_likelihood() {
        let mut pipeline = PublicationPipeline::new();
        let manuscript = pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author".to_string()],
            Domain::Ranking,
            "Abstract".to_string(),
            vec![],
            5000,
        );

        let venue = pipeline.venues[0].clone();
        let likelihood = pipeline.estimate_acceptance_likelihood(&manuscript, &venue);
        assert!(likelihood >= 0.0 && likelihood <= 1.0);
    }

    #[test]
    fn publication_pipeline_report_is_readable() {
        let mut pipeline = PublicationPipeline::new();
        pipeline.create_manuscript(
            "Test Paper".to_string(),
            vec!["Author".to_string()],
            Domain::Classification,
            "Abstract".to_string(),
            vec![],
            4000,
        );

        let report = pipeline.report();
        assert!(report.contains("Publication Pipeline Report"));
        assert!(report.contains("Manuscripts in pipeline"));
        assert!(report.contains("Acceptance rate"));
    }
}
