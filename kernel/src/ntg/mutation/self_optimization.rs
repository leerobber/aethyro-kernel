//! Self-optimization engine: agents autonomously detect and fix bottlenecks
//!
//! Phase 7: Autonomous bottleneck detection and performance-driven refactoring
//! - Brain δ identifies latency/memory hotspots via perception
//! - Brain β proposes micro-mutations (kernel tuning, parallelization changes)
//! - Brain γ evaluates proposals against safety constraints
//! - Brain α coordinates acceptance and rollout
//!
//! Cycle: measure → analyze → propose → evaluate → decide → apply → learn

use super::ledger::MutationLedger;
use super::evaluator::FitnessEvaluator;
use crate::ntg::error::NtgError;

/// Bottleneck type identified by self-optimization
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckType {
    /// CPU utilization > 95%
    ComputeOverload,
    /// Memory bandwidth saturation
    MemoryBandwidth,
    /// GPU-CPU synchronization stalls
    SyncLatency,
    /// Task scheduling contention
    SchedulingContention,
    /// Thread pool exhaustion
    ThreadPoolExhaustion,
    /// Lock contention in critical sections
    LockContention,
    /// Cache miss rate > threshold
    CacheMisses,
    /// I/O wait time significant
    IoWait,
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::ComputeOverload => write!(f, "compute_overload"),
            BottleneckType::MemoryBandwidth => write!(f, "memory_bandwidth"),
            BottleneckType::SyncLatency => write!(f, "sync_latency"),
            BottleneckType::SchedulingContention => write!(f, "scheduling_contention"),
            BottleneckType::ThreadPoolExhaustion => write!(f, "thread_pool_exhaustion"),
            BottleneckType::LockContention => write!(f, "lock_contention"),
            BottleneckType::CacheMisses => write!(f, "cache_misses"),
            BottleneckType::IoWait => write!(f, "io_wait"),
        }
    }
}

/// Performance metric snapshot for bottleneck analysis
#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    pub cpu_utilization: f32,      // 0.0-1.0
    pub memory_bandwidth_usage: f32, // 0.0-1.0
    pub gpu_utilization: f32,       // 0.0-1.0 (if available)
    pub cache_hit_rate: f32,        // 0.0-1.0
    pub sync_latency_us: u64,       // GPU-CPU sync time in microseconds
    pub task_queue_depth: usize,    // Pending tasks
    pub lock_wait_time_us: u64,     // Time spent waiting on locks
    pub io_wait_us: u64,            // I/O blocking time
    pub cycle_time_us: u64,         // Total cycle duration
    pub throughput_ops_per_sec: f32, // Operations per second
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_bandwidth_usage: 0.0,
            gpu_utilization: 0.0,
            cache_hit_rate: 1.0,
            sync_latency_us: 0,
            task_queue_depth: 0,
            lock_wait_time_us: 0,
            io_wait_us: 0,
            cycle_time_us: 0,
            throughput_ops_per_sec: 0.0,
        }
    }
}

/// Identified bottleneck with severity
#[derive(Clone, Debug)]
pub struct BottleneckDetection {
    pub bottleneck_type: BottleneckType,
    /// Severity 0.0-1.0; higher = more urgent
    pub severity: f32,
    /// Metric supporting this detection
    pub metric_value: f32,
    /// Threshold that triggered detection
    pub threshold: f32,
}

/// Proposed optimization action
#[derive(Clone, Debug)]
pub struct OptimizationProposal {
    pub id: String,
    pub bottleneck: BottleneckDetection,
    /// Action description (e.g., "increase thread pool size to 16")
    pub action_description: String,
    /// Confidence in improvement (0.0-1.0)
    pub expected_improvement: f32,
    /// Risk of regression (0.0-1.0)
    pub regression_risk: f32,
    /// Resource cost to apply (memory/cpu for recompilation, etc)
    pub resource_cost: u64,
}

/// Self-optimization engine: coordinates autonomous improvement
pub struct SelfOptimizer {
    pub config: SelfOptimizationConfig,
    pub performance_history: Vec<PerformanceMetrics>,
    pub bottleneck_detections: Vec<BottleneckDetection>,
    pub proposals: Vec<OptimizationProposal>,
    pub accepted_proposals: Vec<OptimizationProposal>,
    pub fitness_evaluator: FitnessEvaluator,
    pub mutation_ledger: MutationLedger,
}

/// Configuration for self-optimization
#[derive(Clone, Debug)]
pub struct SelfOptimizationConfig {
    /// Compute overload threshold (e.g., 0.95 = 95% CPU)
    pub compute_threshold: f32,
    /// Memory bandwidth saturation threshold
    pub bandwidth_threshold: f32,
    /// Cache miss rate threshold (0.0-1.0)
    pub cache_miss_threshold: f32,
    /// Sync latency threshold in microseconds
    pub sync_latency_threshold_us: u64,
    /// Lock contention threshold (microseconds)
    pub lock_contention_threshold_us: u64,
    /// Confidence threshold for accepting proposals
    pub acceptance_threshold: f32,
    /// Maximum proposals per cycle
    pub max_proposals_per_cycle: usize,
    /// Historical window for trend analysis
    pub history_window_cycles: usize,
}

impl Default for SelfOptimizationConfig {
    fn default() -> Self {
        Self {
            compute_threshold: 0.95,
            bandwidth_threshold: 0.90,
            cache_miss_threshold: 0.30,
            sync_latency_threshold_us: 1000,
            lock_contention_threshold_us: 500,
            acceptance_threshold: 0.75,
            max_proposals_per_cycle: 5,
            history_window_cycles: 20,
        }
    }
}

impl SelfOptimizer {
    pub fn new(config: SelfOptimizationConfig) -> Result<Self, NtgError> {
        Ok(Self {
            config,
            performance_history: Vec::new(),
            bottleneck_detections: Vec::new(),
            proposals: Vec::new(),
            accepted_proposals: Vec::new(),
            fitness_evaluator: FitnessEvaluator::new(),
            mutation_ledger: MutationLedger::new(),
        })
    }

    /// Analyze current performance metrics and detect bottlenecks
    pub fn detect_bottlenecks(&mut self, metrics: PerformanceMetrics) -> Result<(), NtgError> {
        self.performance_history.push(metrics.clone());

        // Keep history window
        if self.performance_history.len() > self.config.history_window_cycles {
            self.performance_history.remove(0);
        }

        self.bottleneck_detections.clear();

        // Check each bottleneck condition
        if metrics.cpu_utilization > self.config.compute_threshold {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::ComputeOverload,
                severity: (metrics.cpu_utilization - self.config.compute_threshold) / (1.0 - self.config.compute_threshold),
                metric_value: metrics.cpu_utilization,
                threshold: self.config.compute_threshold,
            });
        }

        if metrics.memory_bandwidth_usage > self.config.bandwidth_threshold {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::MemoryBandwidth,
                severity: (metrics.memory_bandwidth_usage - self.config.bandwidth_threshold) / (1.0 - self.config.bandwidth_threshold),
                metric_value: metrics.memory_bandwidth_usage,
                threshold: self.config.bandwidth_threshold,
            });
        }

        if metrics.cache_hit_rate < (1.0 - self.config.cache_miss_threshold) {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::CacheMisses,
                severity: (1.0 - metrics.cache_hit_rate) / self.config.cache_miss_threshold,
                metric_value: 1.0 - metrics.cache_hit_rate,
                threshold: self.config.cache_miss_threshold,
            });
        }

        if metrics.sync_latency_us > self.config.sync_latency_threshold_us {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::SyncLatency,
                severity: (metrics.sync_latency_us as f32 / self.config.sync_latency_threshold_us as f32).min(1.0),
                metric_value: metrics.sync_latency_us as f32,
                threshold: self.config.sync_latency_threshold_us as f32,
            });
        }

        if metrics.lock_wait_time_us > self.config.lock_contention_threshold_us {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::LockContention,
                severity: (metrics.lock_wait_time_us as f32 / self.config.lock_contention_threshold_us as f32).min(1.0),
                metric_value: metrics.lock_wait_time_us as f32,
                threshold: self.config.lock_contention_threshold_us as f32,
            });
        }

        if metrics.task_queue_depth > 100 {
            self.bottleneck_detections.push(BottleneckDetection {
                bottleneck_type: BottleneckType::ThreadPoolExhaustion,
                severity: (metrics.task_queue_depth as f32 / 1000.0).min(1.0),
                metric_value: metrics.task_queue_depth as f32,
                threshold: 100.0,
            });
        }

        Ok(())
    }

    /// Propose optimizations for detected bottlenecks
    pub fn propose_optimizations(&mut self) -> Result<Vec<OptimizationProposal>, NtgError> {
        self.proposals.clear();

        for detection in &self.bottleneck_detections {
            let proposal = match detection.bottleneck_type {
                BottleneckType::ComputeOverload => {
                    OptimizationProposal {
                        id: format!("opt_compute_{}", self.proposals.len()),
                        bottleneck: detection.clone(),
                        action_description: "Enable CUDA GPU acceleration for Brain δ embeddings".to_string(),
                        expected_improvement: 0.5,  // 50× speedup on perception
                        regression_risk: 0.05,       // 5% risk (well-tested CUDA kernels)
                        resource_cost: 50_000_000,   // 50MB VRAM for kernel state
                    }
                }
                BottleneckType::MemoryBandwidth => {
                    OptimizationProposal {
                        id: format!("opt_memory_{}", self.proposals.len()),
                        bottleneck: detection.clone(),
                        action_description: "Reduce data movement via fusion optimization".to_string(),
                        expected_improvement: 0.3,
                        regression_risk: 0.10,
                        resource_cost: 1_000_000,
                    }
                }
                BottleneckType::LockContention => {
                    OptimizationProposal {
                        id: format!("opt_lock_{}", self.proposals.len()),
                        bottleneck: detection.clone(),
                        action_description: "Switch to lock-free concurrent data structures".to_string(),
                        expected_improvement: 0.4,
                        regression_risk: 0.15,
                        resource_cost: 500_000,
                    }
                }
                BottleneckType::SyncLatency => {
                    OptimizationProposal {
                        id: format!("opt_sync_{}", self.proposals.len()),
                        bottleneck: detection.clone(),
                        action_description: "Overlap GPU/CPU computation via streams".to_string(),
                        expected_improvement: 0.35,
                        regression_risk: 0.08,
                        resource_cost: 100_000,
                    }
                }
                BottleneckType::ThreadPoolExhaustion => {
                    OptimizationProposal {
                        id: format!("opt_threads_{}", self.proposals.len()),
                        bottleneck: detection.clone(),
                        action_description: "Increase thread pool size and use work-stealing queue".to_string(),
                        expected_improvement: 0.25,
                        regression_risk: 0.12,
                        resource_cost: 10_000_000,  // More thread stacks
                    }
                }
                _ => continue,
            };

            if self.proposals.len() < self.config.max_proposals_per_cycle {
                self.proposals.push(proposal);
            }
        }

        Ok(self.proposals.clone())
    }

    /// Evaluate proposal feasibility and expected impact
    pub fn evaluate_proposal(&self, proposal: &OptimizationProposal) -> Result<f32, NtgError> {
        // Score based on:
        // - Expected improvement magnitude
        // - Regression risk (inverted)
        // - Resource cost (normalized)

        let improvement_score = proposal.expected_improvement * 0.6;  // 60% weight
        let safety_score = (1.0 - proposal.regression_risk) * 0.3;     // 30% weight
        let efficiency_score = (1.0 - (proposal.resource_cost as f32 / 100_000_000.0)).max(0.0) * 0.1; // 10% weight

        let total_score = improvement_score + safety_score + efficiency_score;
        Ok(total_score)
    }

    /// Accept a proposal and log it for rollout
    pub fn accept_proposal(&mut self, proposal: OptimizationProposal) -> Result<(), NtgError> {
        self.accepted_proposals.push(proposal);
        // Note: Full ledger integration would require adapting to MutationEvent struct
        // For now, proposals are tracked in accepted_proposals
        Ok(())
    }

    /// Summarize optimization cycle
    pub fn cycle_summary(&self) -> OptimizationCycleSummary {
        OptimizationCycleSummary {
            bottlenecks_detected: self.bottleneck_detections.len(),
            proposals_generated: self.proposals.len(),
            proposals_accepted: self.accepted_proposals.len(),
            critical_bottlenecks: self.bottleneck_detections.iter()
                .filter(|b| b.severity > 0.8)
                .count(),
        }
    }
}

/// Summary of a single optimization cycle
#[derive(Clone, Debug)]
pub struct OptimizationCycleSummary {
    pub bottlenecks_detected: usize,
    pub proposals_generated: usize,
    pub proposals_accepted: usize,
    pub critical_bottlenecks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottleneck_detection_compute_overload() -> Result<(), NtgError> {
        let mut optimizer = SelfOptimizer::new(SelfOptimizationConfig::default())?;

        let metrics = PerformanceMetrics {
            cpu_utilization: 0.98,
            ..Default::default()
        };

        optimizer.detect_bottlenecks(metrics)?;

        assert_eq!(optimizer.bottleneck_detections.len(), 1);
        assert_eq!(optimizer.bottleneck_detections[0].bottleneck_type, BottleneckType::ComputeOverload);
        Ok(())
    }

    #[test]
    fn proposal_generation() -> Result<(), NtgError> {
        let mut optimizer = SelfOptimizer::new(SelfOptimizationConfig::default())?;

        let metrics = PerformanceMetrics {
            cpu_utilization: 0.98,
            memory_bandwidth_usage: 0.92,
            lock_wait_time_us: 1500,
            ..Default::default()
        };

        optimizer.detect_bottlenecks(metrics)?;
        let proposals = optimizer.propose_optimizations()?;

        assert!(proposals.len() >= 2);  // Should propose for multiple bottlenecks
        Ok(())
    }

    #[test]
    fn proposal_evaluation() -> Result<(), NtgError> {
        let optimizer = SelfOptimizer::new(SelfOptimizationConfig::default())?;

        let proposal = OptimizationProposal {
            id: "test_proposal".to_string(),
            bottleneck: BottleneckDetection {
                bottleneck_type: BottleneckType::ComputeOverload,
                severity: 0.9,
                metric_value: 0.98,
                threshold: 0.95,
            },
            action_description: "Test optimization".to_string(),
            expected_improvement: 0.8,
            regression_risk: 0.05,
            resource_cost: 1_000_000,
        };

        let score = optimizer.evaluate_proposal(&proposal)?;
        assert!(score > 0.5);  // Should be a high-scoring proposal
        Ok(())
    }
}
