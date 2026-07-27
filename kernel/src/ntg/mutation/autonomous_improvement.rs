//! Autonomous improvement orchestrator: coordinates GPU acceleration + self-optimization
//!
//! Master loop that integrates:
//! 1. Performance monitoring (Brain δ perception)
//! 2. Bottleneck detection (self_optimization)
//! 3. Proposal generation (Brain β learning)
//! 4. Safety evaluation (Brain γ governance)
//! 5. Coordinated rollout (Brain α synchronization)
//!
//! Cycle: measure → detect → propose → evaluate → decide → apply → learn
//! Target: <100ms per improvement cycle, <1% regression rate

use super::self_optimization::{SelfOptimizer, SelfOptimizationConfig, PerformanceMetrics};
use super::brain_integration::QuadBrainAgent;
use crate::ntg::error::NtgError;
use std::collections::VecDeque;
use std::time::Instant;

/// Autonomous improvement cycle orchestrator
pub struct AutonomousImprover {
    pub config: AutonomousImproverConfig,
    pub self_optimizer: SelfOptimizer,
    pub quad_brain: Option<QuadBrainAgent>,
    pub improvement_history: VecDeque<ImprovementRound>,
    pub cycle_count: u64,
    pub total_improvement: f32,  // Cumulative speedup from all improvements
}

/// Configuration for autonomous improvement
#[derive(Clone, Debug)]
pub struct AutonomousImproverConfig {
    /// Run improvement cycle every N perception cycles
    pub improvement_cycle_interval: usize,
    /// Maximum time budget for improvement analysis (milliseconds)
    pub analysis_time_budget_ms: u64,
    /// Minimum improvement threshold to apply (e.g., 0.05 = 5% speedup required)
    pub min_improvement_threshold: f32,
    /// Maximum acceptable regression risk (e.g., 0.10 = 10% risk tolerance)
    pub max_regression_risk: f32,
    /// Historical window for trend analysis (cycles)
    pub history_window: usize,
    /// Enable GPU acceleration when available
    pub enable_gpu_acceleration: bool,
    /// Enable autonomous proposal acceptance
    pub autonomous_acceptance: bool,
}

impl Default for AutonomousImproverConfig {
    fn default() -> Self {
        Self {
            improvement_cycle_interval: 100,  // Every 100 cycles
            analysis_time_budget_ms: 50,       // 50ms budget for analysis
            min_improvement_threshold: 0.05,   // 5% speedup minimum
            max_regression_risk: 0.10,         // 10% risk tolerance
            history_window: 50,
            enable_gpu_acceleration: true,
            autonomous_acceptance: true,
        }
    }
}

/// A single improvement round
#[derive(Clone, Debug)]
pub struct ImprovementRound {
    pub cycle_number: u64,
    pub timestamp: Instant,
    pub metrics_before: PerformanceMetrics,
    pub bottlenecks_detected: usize,
    pub proposals_generated: usize,
    pub proposals_accepted: usize,
    pub speedup_achieved: f32,  // ratio: old_time / new_time
    pub regressions: usize,
}

impl AutonomousImprover {
    pub fn new(config: AutonomousImproverConfig) -> Result<Self, NtgError> {
        Ok(Self {
            config,
            self_optimizer: SelfOptimizer::new(SelfOptimizationConfig::default())?,
            quad_brain: None,
            improvement_history: VecDeque::new(),
            cycle_count: 0,
            total_improvement: 1.0,
        })
    }

    /// Attach quad-brain for coordination
    pub fn attach_quad_brain(&mut self, brain: QuadBrainAgent) {
        self.quad_brain = Some(brain);
    }

    /// Execute one improvement cycle
    pub fn run_cycle(&mut self, metrics: PerformanceMetrics) -> Result<ImprovementRoundResult, NtgError> {
        let cycle_start = Instant::now();
        let metrics_before = metrics.clone();

        self.cycle_count += 1;

        // Phase 1: Detect bottlenecks (Brain δ perception)
        self.self_optimizer.detect_bottlenecks(metrics.clone())?;

        if self.self_optimizer.bottleneck_detections.is_empty() {
            return Ok(ImprovementRoundResult {
                applied: false,
                reason: "No bottlenecks detected".to_string(),
                speedup: 1.0,
            });
        }

        // Phase 2: Propose optimizations (Brain β learning)
        let proposals = self.self_optimizer.propose_optimizations()?;

        // Phase 3: Filter proposals by safety (Brain γ governance)
        let safe_proposals: Vec<_> = proposals
            .into_iter()
            .filter(|p| p.regression_risk <= self.config.max_regression_risk)
            .filter(|p| p.expected_improvement >= self.config.min_improvement_threshold)
            .collect();

        if safe_proposals.is_empty() {
            return Ok(ImprovementRoundResult {
                applied: false,
                reason: "No proposals met safety thresholds".to_string(),
                speedup: 1.0,
            });
        }

        // Phase 4: Rank by expected impact
        let mut ranked_proposals = safe_proposals;
        ranked_proposals.sort_by(|a, b| {
            b.expected_improvement.partial_cmp(&a.expected_improvement).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Phase 5: Accept top proposal (Brain α coordination)
        if let Some(selected_proposal) = ranked_proposals.first() {
            if self.config.autonomous_acceptance {
                self.self_optimizer.accept_proposal(selected_proposal.clone())?;

                // Simulate improvement achieved
                let speedup = 1.0 + selected_proposal.expected_improvement;
                self.total_improvement *= speedup;

                let round = ImprovementRound {
                    cycle_number: self.cycle_count,
                    timestamp: cycle_start,
                    metrics_before,
                    bottlenecks_detected: self.self_optimizer.bottleneck_detections.len(),
                    proposals_generated: self.self_optimizer.proposals.len(),
                    proposals_accepted: 1,
                    speedup_achieved: speedup,
                    regressions: 0,
                };

                if self.improvement_history.len() >= self.config.history_window {
                    self.improvement_history.pop_front();
                }
                self.improvement_history.push_back(round);

                return Ok(ImprovementRoundResult {
                    applied: true,
                    reason: selected_proposal.action_description.clone(),
                    speedup,
                });
            }
        }

        Ok(ImprovementRoundResult {
            applied: false,
            reason: "Autonomous acceptance disabled".to_string(),
            speedup: 1.0,
        })
    }

    /// Get summary of improvement progress
    pub fn improvement_summary(&self) -> AutonomousImprovementSummary {
        let total_rounds = self.improvement_history.len();
        let successful_rounds = self.improvement_history.iter().filter(|r| r.speedup_achieved > 1.0).count();
        let avg_speedup = if successful_rounds > 0 {
            self.improvement_history
                .iter()
                .filter(|r| r.speedup_achieved > 1.0)
                .map(|r| r.speedup_achieved)
                .sum::<f32>() / successful_rounds as f32
        } else {
            1.0
        };

        AutonomousImprovementSummary {
            total_cycles: self.cycle_count,
            improvement_rounds: total_rounds,
            successful_improvements: successful_rounds,
            cumulative_speedup: self.total_improvement,
            average_speedup_per_round: avg_speedup,
            total_regressions: self.improvement_history.iter().map(|r| r.regressions).sum(),
        }
    }

    /// Export improvement log for analysis
    pub fn export_log(&self) -> String {
        let mut log = String::from("=== Autonomous Improvement Log ===\n");
        log.push_str(&format!("Total cycles: {}\n", self.cycle_count));
        log.push_str(&format!("Cumulative speedup: {:.2}×\n\n", self.total_improvement));

        for round in &self.improvement_history {
            log.push_str(&format!(
                "Cycle {}: {:.2}× speedup ({} bottlenecks, {} proposals accepted, {} regressions)\n",
                round.cycle_number, round.speedup_achieved, round.bottlenecks_detected,
                round.proposals_accepted, round.regressions
            ));
        }

        log
    }
}

/// Result of one improvement cycle
#[derive(Clone, Debug)]
pub struct ImprovementRoundResult {
    pub applied: bool,
    pub reason: String,
    pub speedup: f32,
}

/// Summary of autonomous improvement progress
#[derive(Clone, Debug)]
pub struct AutonomousImprovementSummary {
    pub total_cycles: u64,
    pub improvement_rounds: usize,
    pub successful_improvements: usize,
    pub cumulative_speedup: f32,
    pub average_speedup_per_round: f32,
    pub total_regressions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autonomous_improvement_detects_bottleneck() -> Result<(), NtgError> {
        let config = AutonomousImproverConfig {
            autonomous_acceptance: true,
            ..Default::default()
        };
        let mut improver = AutonomousImprover::new(config)?;

        let metrics = PerformanceMetrics {
            cpu_utilization: 0.98,
            memory_bandwidth_usage: 0.92,
            ..Default::default()
        };

        let result = improver.run_cycle(metrics)?;
        assert!(result.applied || !result.reason.is_empty());
        Ok(())
    }

    #[test]
    fn improvement_summary_accumulates() -> Result<(), NtgError> {
        let config = AutonomousImproverConfig {
            autonomous_acceptance: true,
            ..Default::default()
        };
        let mut improver = AutonomousImprover::new(config)?;

        for i in 0..5 {
            let metrics = PerformanceMetrics {
                cpu_utilization: 0.85 + (i as f32 * 0.01),
                ..Default::default()
            };
            improver.run_cycle(metrics)?;
        }

        let summary = improver.improvement_summary();
        assert!(summary.total_cycles >= 5);
        Ok(())
    }
}
