//! Health monitoring and self-healing for NanoKeymaster.
//!
//! Detects topology degradation and triggers:
//! - Edge pruning (remove w < threshold)
//! - Hypervector compression (drop low-confidence intents)
//! - Decay acceleration (speed up forgetting under load)
//! - Conservative routing (fallback when both latency & memory high)

use std::collections::VecDeque;

/// Performance snapshot: latency and memory at a point in time
#[derive(Clone, Debug)]
pub struct PerformanceSnapshot {
    pub latency_us: u64,
    pub memory_bytes: usize,
    pub timestamp: u64,
}

/// Trend signal: is the system improving, stable, or degrading?
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrendSignal {
    Improving,
    Stable,
    Degrading,
}

/// Health monitor: tracks performance and detects when healing is needed
#[derive(Clone, Debug)]
pub struct HealthMonitor {
    /// Baseline latency established after warmup (microseconds)
    baseline_latency: u64,
    /// Baseline memory at startup (bytes)
    baseline_memory: usize,
    /// Efficiency loss threshold: trigger healing if efficiency drops >N% (e.g., 0.10 = 10%)
    degradation_threshold: f64,
    /// Window size: track last N calls for trend analysis
    call_window: usize,
    /// Recent performance measurements
    recent_metrics: VecDeque<PerformanceSnapshot>,
    /// Call counter: incremented each request
    call_count: u64,
    /// Whether system is currently in conservative mode
    pub in_conservative_mode: bool,
}

impl HealthMonitor {
    pub fn new(baseline_latency: u64, baseline_memory: usize) -> Self {
        Self {
            baseline_latency,
            baseline_memory,
            degradation_threshold: 0.10, // 10% efficiency drop = alarm
            call_window: 50,
            recent_metrics: VecDeque::with_capacity(50),
            call_count: 0,
            in_conservative_mode: false,
        }
    }

    /// Record a measurement (called after each request)
    pub fn sample(&mut self, latency_us: u64, memory_bytes: usize) {
        self.call_count += 1;
        let snapshot = PerformanceSnapshot {
            latency_us,
            memory_bytes,
            timestamp: self.call_count,
        };

        self.recent_metrics.push_back(snapshot);
        if self.recent_metrics.len() > self.call_window {
            self.recent_metrics.pop_front();
        }
    }

    /// Compute current efficiency: weighted average of latency + memory health
    /// Returns 1.0 = perfect, 0.0 = completely degraded
    pub fn current_efficiency(&self) -> f64 {
        if self.recent_metrics.is_empty() {
            return 1.0;
        }

        let avg_latency = self.recent_metrics.iter().map(|s| s.latency_us).sum::<u64>()
            as f64
            / self.recent_metrics.len() as f64;
        let avg_memory = self.recent_metrics.iter().map(|s| s.memory_bytes).sum::<usize>()
            as f64
            / self.recent_metrics.len() as f64;

        // Efficiency: 80% latency, 20% memory
        // Higher is better (lower latency/memory = higher efficiency)
        // Returns 1.0 when at baseline, decreases as metrics degrade
        let latency_ratio = (self.baseline_latency as f64 / (avg_latency + 1.0).max(1.0)).min(1.0);
        let memory_ratio = (self.baseline_memory as f64 / (avg_memory + 1.0).max(1.0)).min(1.0);

        latency_ratio * 0.8 + memory_ratio * 0.2
    }

    /// Analyze trend: are we improving, stable, or degrading?
    pub fn efficiency_trend(&self) -> TrendSignal {
        if self.recent_metrics.len() < 10 {
            return TrendSignal::Stable;
        }

        // Split into halves and compare average efficiency
        let mid = self.recent_metrics.len() / 2;
        let first_half = self.recent_metrics.iter().take(mid);
        let second_half = self.recent_metrics.iter().skip(mid);

        let first_avg_latency = first_half.clone().map(|s| s.latency_us).sum::<u64>() as f64
            / (mid as f64).max(1.0);
        let second_avg_latency =
            second_half.clone().map(|s| s.latency_us).sum::<u64>() as f64
                / ((self.recent_metrics.len() - mid) as f64).max(1.0);

        if second_avg_latency > first_avg_latency * 1.05 {
            // Latency increased by >5%
            TrendSignal::Degrading
        } else if second_avg_latency < first_avg_latency * 0.95 {
            // Latency decreased by >5%
            TrendSignal::Improving
        } else {
            TrendSignal::Stable
        }
    }

    /// Should healing be triggered?
    /// Returns true if efficiency drops below (baseline - threshold)
    pub fn should_trigger_healing(&self) -> bool {
        if self.recent_metrics.len() < 10 {
            return false;
        }

        let current_eff = self.current_efficiency();
        // Trigger if efficiency falls below (1.0 - degradation_threshold)
        current_eff < (1.0 - self.degradation_threshold)
    }

    /// Check if we should enter conservative mode (both latency AND memory high)
    pub fn should_enter_conservative_mode(&self) -> bool {
        if self.recent_metrics.is_empty() {
            return false;
        }

        let avg_latency = self.recent_metrics.iter().map(|s| s.latency_us).sum::<u64>()
            as f64
            / self.recent_metrics.len() as f64;
        let avg_memory = self.recent_metrics.iter().map(|s| s.memory_bytes).sum::<usize>()
            as f64
            / self.recent_metrics.len() as f64;

        let latency_high = avg_latency > self.baseline_latency as f64 * 1.3; // 30% over baseline
        let memory_high = avg_memory > self.baseline_memory as f64 * 1.2; // 20% over baseline

        latency_high && memory_high
    }

    pub fn set_conservative_mode(&mut self, enabled: bool) {
        self.in_conservative_mode = enabled;
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    pub fn recent_metrics_snapshot(&self) -> Vec<PerformanceSnapshot> {
        self.recent_metrics.iter().cloned().collect()
    }

    /// Get baseline latency (microseconds)
    pub fn baseline_latency_us(&self) -> u64 {
        self.baseline_latency
    }

    /// Get baseline memory (bytes)
    pub fn baseline_memory_bytes(&self) -> usize {
        self.baseline_memory
    }

    /// Get current average latency (microseconds)
    pub fn current_latency_us(&self) -> u64 {
        if self.recent_metrics.is_empty() {
            self.baseline_latency
        } else {
            self.recent_metrics.iter().map(|s| s.latency_us).sum::<u64>()
                / self.recent_metrics.len() as u64
        }
    }

    /// Get current average memory (bytes)
    pub fn current_memory_bytes(&self) -> usize {
        if self.recent_metrics.is_empty() {
            self.baseline_memory
        } else {
            self.recent_metrics.iter().map(|s| s.memory_bytes).sum::<usize>()
                / self.recent_metrics.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_tracks_baseline() {
        let monitor = HealthMonitor::new(100, 1000);
        assert_eq!(monitor.baseline_latency, 100);
        assert_eq!(monitor.baseline_memory, 1000);
    }

    #[test]
    fn efficiency_is_high_when_metrics_match_baseline() {
        let mut monitor = HealthMonitor::new(100, 1000);
        for _ in 0..50 {
            monitor.sample(100, 1000);
        }
        let eff = monitor.current_efficiency();
        assert!(eff > 0.95);
    }

    #[test]
    fn efficiency_drops_when_latency_degrades() {
        let mut monitor = HealthMonitor::new(100, 1000);
        for _ in 0..50 {
            monitor.sample(200, 1000); // 2x baseline latency
        }
        let eff = monitor.current_efficiency();
        // 2x latency: ratio = 100/200 = 0.5; eff = 0.5*0.8 + 1.0*0.2 = 0.6
        assert!(eff < 0.7 && eff > 0.5);
    }

    #[test]
    fn trend_detects_degradation() {
        let mut monitor = HealthMonitor::new(100, 1000);
        // First half: good latency
        for _ in 0..10 {
            monitor.sample(100, 1000);
        }
        // Second half: degraded
        for _ in 0..10 {
            monitor.sample(150, 1000);
        }
        assert_eq!(monitor.efficiency_trend(), TrendSignal::Degrading);
    }

    #[test]
    fn should_trigger_healing_when_efficiency_low() {
        let mut monitor = HealthMonitor::new(100, 1000);
        for _ in 0..50 {
            monitor.sample(500, 5000); // Very degraded
        }
        assert!(monitor.should_trigger_healing());
    }

    #[test]
    fn conservative_mode_triggered_when_both_high() {
        let mut monitor = HealthMonitor::new(100, 1000);
        for _ in 0..20 {
            monitor.sample(150, 1300); // Both 30%+ over baseline
        }
        assert!(monitor.should_enter_conservative_mode());
    }
}
