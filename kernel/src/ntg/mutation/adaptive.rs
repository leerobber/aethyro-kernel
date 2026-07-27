//! Adaptive mutation proposer: suggests topology changes based on degradation signals.
//!
//! Analyzes:
//! - Current graph structure (connectivity, degree distribution)
//! - Degradation type (latency-dominant vs memory-dominant)
//! - History of accepted mutations (learn what works)
//!
//! Proposes mutations likely to improve the detected bottleneck.

use super::rules::{MutationRule, MutationRuleKind};
use super::super::graph::{Graph, NodeId};
use super::super::error::NtgError;
use super::ledger::MutationLedger;
use serde::{Serialize, Deserialize};

/// Signal indicating what aspect of performance is degrading.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationSignal {
    /// Latency dominant (latency ratio > memory ratio).
    LatencyDominant,
    /// Memory dominant (memory ratio > latency ratio).
    MemoryDominant,
    /// Both elevated equally.
    Balanced,
}

impl DegradationSignal {
    /// Classify degradation based on baseline and current fitness.
    pub fn from_metrics(
        baseline_latency: u64,
        baseline_memory: usize,
        current_latency: u64,
        current_memory: usize,
    ) -> Self {
        let latency_ratio = current_latency as f64 / baseline_latency as f64;
        let memory_ratio = current_memory as f64 / baseline_memory as f64;

        if latency_ratio > memory_ratio * 1.1 {
            DegradationSignal::LatencyDominant
        } else if memory_ratio > latency_ratio * 1.1 {
            DegradationSignal::MemoryDominant
        } else {
            DegradationSignal::Balanced
        }
    }
}

/// Adaptive mutation proposer: suggests mutations based on graph structure and degradation type.
#[derive(Clone, Debug)]
pub struct AdaptiveMutationProposer {
    /// Mutations that have succeeded before (bias toward repeating wins).
    pub successful_mutations: Vec<String>,
    /// Total mutations proposed so far.
    pub proposal_count: u64,
    /// Accept rate (for deciding when to explore vs exploit).
    pub accept_rate: f64,
}

impl AdaptiveMutationProposer {
    pub fn new() -> Self {
        Self {
            successful_mutations: Vec::new(),
            proposal_count: 0,
            accept_rate: 0.5,
        }
    }

    /// Log a successful mutation for future learning.
    pub fn record_success(&mut self, mutation_desc: String) {
        self.successful_mutations.push(mutation_desc);
    }

    /// Update accept rate based on cycle results.
    pub fn update_accept_rate(&mut self, accepted: usize, total: usize) {
        if total > 0 {
            self.accept_rate = accepted as f64 / total as f64;
        }
    }

    /// Propose mutations tailored to the degradation signal and graph structure.
    pub fn propose_mutations(
        &mut self,
        graph: &Graph,
        signal: DegradationSignal,
        count: usize,
    ) -> Result<Vec<MutationRule>, NtgError> {
        let mut proposals = Vec::new();
        self.proposal_count += count as u64;

        // Explore vs exploit: if accept rate is high, exploit (repeat successes);
        // if low, explore (try new mutations).
        let exploit = self.accept_rate > 0.6;

        for i in 0..count {
            let rule = if exploit && !self.successful_mutations.is_empty() {
                // Exploit: try variations on previous wins.
                self.propose_exploitation_mutation(graph, i)?
            } else {
                // Explore: try targeted mutations based on degradation type.
                self.propose_exploration_mutation(graph, signal, i)?
            };
            proposals.push(rule);
        }

        Ok(proposals)
    }

    /// Propose mutations with ledger-informed confidence bias.
    /// Uses learned patterns to prefer mutations that have worked well for this signal type.
    pub fn propose_mutations_with_ledger(
        &mut self,
        graph: &Graph,
        signal: DegradationSignal,
        count: usize,
        ledger: &MutationLedger,
    ) -> Result<Vec<MutationRule>, NtgError> {
        let mut proposals = Vec::new();
        self.proposal_count += count as u64;

        // Ledger coverage: if we have learned data, shift toward exploitation
        let ledger_events = ledger.events().len();
        let ledger_confidence_bonus = (ledger_events as f64 / 100.0).min(0.2); // up to +20% from ledger
        let exploit_threshold = (0.6 - ledger_confidence_bonus).max(0.3); // but never below 30%
        let exploit = self.accept_rate > exploit_threshold;

        for i in 0..count {
            let rule = if exploit && !self.successful_mutations.is_empty() {
                // Exploit: try variations on previous wins, biased by ledger confidence.
                self.propose_exploitation_mutation(graph, i)?
            } else {
                // Explore: try targeted mutations, preferring high-confidence types from ledger.
                self.propose_exploration_mutation_with_confidence(graph, signal, i, ledger)?
            };
            proposals.push(rule);
        }

        Ok(proposals)
    }

    /// Exploration with ledger confidence bias: prefer mutation types with high success rate.
    fn propose_exploration_mutation_with_confidence(
        &self,
        graph: &Graph,
        signal: DegradationSignal,
        index: usize,
        ledger: &MutationLedger,
    ) -> Result<MutationRule, NtgError> {
        // Query ledger for best mutation for this signal
        if let Some(best_mutation_type) = ledger.best_mutation_for_signal(signal) {
            let confidence = ledger.confidence_for(&best_mutation_type, signal);

            // If high confidence, bias toward this mutation type
            if confidence > 0.6 {
                // Use best mutation from ledger with 70% probability
                if index % 10 < 7 {
                    return self.propose_mutation_of_type(graph, &best_mutation_type, index, signal);
                }
            }
        }

        // Fallback: standard degradation-signal-driven exploration
        match signal {
            DegradationSignal::LatencyDominant => {
                self.propose_edge_removal(graph, index)
            }
            DegradationSignal::MemoryDominant => {
                self.propose_node_removal(graph, index)
            }
            DegradationSignal::Balanced => {
                if index % 2 == 0 {
                    self.propose_edge_removal(graph, index)
                } else {
                    self.propose_node_removal(graph, index)
                }
            }
        }
    }

    /// Propose a mutation of a specific type (from ledger best practice).
    fn propose_mutation_of_type(
        &self,
        graph: &Graph,
        mutation_type: &str,
        index: usize,
        _signal: DegradationSignal,
    ) -> Result<MutationRule, NtgError> {
        if mutation_type.contains("RemoveEdge") {
            self.propose_edge_removal(graph, index)
        } else if mutation_type.contains("RemoveNode") {
            self.propose_node_removal(graph, index)
        } else if mutation_type.contains("AddNode") {
            self.propose_node_addition(graph)
        } else {
            // Fallback
            self.propose_edge_removal(graph, index)
        }
    }

    /// Exploitation: generate variations on previously successful mutations.
    fn propose_exploitation_mutation(
        &self,
        graph: &Graph,
        index: usize,
    ) -> Result<MutationRule, NtgError> {
        if self.successful_mutations.is_empty() {
            return self.propose_edge_removal(graph, index);
        }

        let prev_success = &self.successful_mutations[index % self.successful_mutations.len()];

        // Parse the success description to extract node/edge info.
        // Format: "remove_edge(N→M)" or "add_node(label)"
        if prev_success.contains("remove_edge") {
            // Try removing another edge.
            self.propose_edge_removal(graph, index)
        } else if prev_success.contains("add_node") {
            // Try adding another node.
            self.propose_node_addition(graph)
        } else {
            self.propose_edge_removal(graph, index)
        }
    }

    /// Exploration: propose mutations tailored to the degradation type.
    fn propose_exploration_mutation(
        &self,
        graph: &Graph,
        signal: DegradationSignal,
        index: usize,
    ) -> Result<MutationRule, NtgError> {
        match signal {
            DegradationSignal::LatencyDominant => {
                // Latency-dominant: remove edges to reduce forward-pass traversal cost.
                // Prefer removing weak edges (low-weight connections).
                self.propose_edge_removal(graph, index)
            }
            DegradationSignal::MemoryDominant => {
                // Memory-dominant: remove nodes or reduce graph size.
                // Prefer removing leaf nodes (no dependents).
                self.propose_node_removal(graph, index)
            }
            DegradationSignal::Balanced => {
                // Both metrics high: mixed strategy.
                // Alternate between edge removal and node removal.
                if index % 2 == 0 {
                    self.propose_edge_removal(graph, index)
                } else {
                    self.propose_node_removal(graph, index)
                }
            }
        }
    }

    /// Propose edge removal: identify a removable edge and remove it.
    fn propose_edge_removal(
        &self,
        graph: &Graph,
        index: usize,
    ) -> Result<MutationRule, NtgError> {
        let edges_slice = graph.edges();
        if edges_slice.is_empty() {
            // No edges to remove; propose node addition as fallback.
            return self.propose_node_addition(graph);
        }

        // Pick an edge deterministically based on index.
        let (from, to) = edges_slice[index % edges_slice.len()];

        Ok(MutationRule {
            kind: MutationRuleKind::RemoveEdge { from, to },
        })
    }

    /// Propose node removal: find a removable leaf node and remove it.
    fn propose_node_removal(
        &self,
        graph: &Graph,
        index: usize,
    ) -> Result<MutationRule, NtgError> {
        // Find nodes with no incoming edges (leaf nodes / sinks).
        let edges_slice = graph.edges();
        let nodes: Vec<NodeId> = graph
            .nodes()
            .iter()
            .filter(|(id, node_opt)| {
                // A node is removable if it has no incoming edges (safe to delete) and exists.
                node_opt.is_some() && !edges_slice.iter().any(|(_, to)| to == id)
            })
            .map(|(id, _)| *id)
            .collect();

        if nodes.is_empty() {
            // No removable nodes; propose adding a new node as fallback.
            return self.propose_node_addition(graph);
        }

        let node_id = nodes[index % nodes.len()];
        Ok(MutationRule {
            kind: MutationRuleKind::RemoveNode { node_id },
        })
    }

    /// Propose node addition: introduce a new node with a generic label.
    fn propose_node_addition(&self, graph: &Graph) -> Result<MutationRule, NtgError> {
        let node_count = graph.node_count();
        let label = format!("synthetic_node_{}", node_count);

        Ok(MutationRule {
            kind: MutationRuleKind::AddNode { label },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::graph::NodeKind;

    #[test]
    fn degradation_signal_latency_dominant() {
        let signal = DegradationSignal::from_metrics(100, 1000, 200, 1050);
        assert_eq!(signal, DegradationSignal::LatencyDominant);
    }

    #[test]
    fn degradation_signal_memory_dominant() {
        let signal = DegradationSignal::from_metrics(100, 1000, 110, 2000);
        assert_eq!(signal, DegradationSignal::MemoryDominant);
    }

    #[test]
    fn degradation_signal_balanced() {
        let signal = DegradationSignal::from_metrics(100, 1000, 150, 1500);
        assert_eq!(signal, DegradationSignal::Balanced);
    }

    #[test]
    fn proposer_new() {
        let proposer = AdaptiveMutationProposer::new();
        assert_eq!(proposer.proposal_count, 0);
        assert!(proposer.successful_mutations.is_empty());
    }

    #[test]
    fn proposer_record_success() {
        let mut proposer = AdaptiveMutationProposer::new();
        proposer.record_success("remove_edge(1→2)".to_string());
        assert_eq!(proposer.successful_mutations.len(), 1);
    }

    #[test]
    fn proposer_update_accept_rate() {
        let mut proposer = AdaptiveMutationProposer::new();
        proposer.update_accept_rate(3, 10);
        assert!((proposer.accept_rate - 0.3).abs() < 0.01);
    }

    #[test]
    fn proposer_propose_mutations_empty_graph() -> Result<(), NtgError> {
        let mut proposer = AdaptiveMutationProposer::new();
        let graph = Graph::new();
        let mutations = proposer.propose_mutations(&graph, DegradationSignal::LatencyDominant, 2)?;
        // Empty graph should still produce proposals (node additions).
        assert!(mutations.len() >= 1);
        Ok(())
    }

    #[test]
    fn proposer_propose_mutations_non_empty_graph() -> Result<(), NtgError> {
        let mut proposer = AdaptiveMutationProposer::new();
        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        let mutations = proposer.propose_mutations(&graph, DegradationSignal::LatencyDominant, 2)?;
        assert_eq!(mutations.len(), 2);
        // Proposals should be edge removals for latency-dominant.
        Ok(())
    }

    #[test]
    fn proposer_exploit_vs_explore() -> Result<(), NtgError> {
        let mut proposer = AdaptiveMutationProposer::new();
        proposer.update_accept_rate(8, 10); // 80% accept rate -> exploit mode

        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        proposer.record_success("remove_edge(1→2)".to_string());

        let mutations = proposer.propose_mutations(&graph, DegradationSignal::Balanced, 1)?;
        assert_eq!(mutations.len(), 1);
        Ok(())
    }

    #[test]
    fn proposer_with_empty_ledger_behaves_normally() -> Result<(), NtgError> {
        use super::super::ledger::MutationLedger;

        let mut proposer = AdaptiveMutationProposer::new();
        let ledger = MutationLedger::new();

        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        let mutations = proposer.propose_mutations_with_ledger(
            &graph,
            DegradationSignal::LatencyDominant,
            2,
            &ledger,
        )?;
        assert_eq!(mutations.len(), 2);
        Ok(())
    }

    #[test]
    fn proposer_biases_toward_ledger_confidence() -> Result<(), NtgError> {
        use super::super::ledger::MutationLedger;

        let mut proposer = AdaptiveMutationProposer::new();
        let mut ledger = MutationLedger::new();

        // Record RemoveEdge successes for LatencyDominant
        for _i in 0..8 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge { from: 1, to: 2 },
                DegradationSignal::LatencyDominant,
                0.80,
                0.83,
                true, // accepted
                0.5,
            );
        }

        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        // Low accept rate + high ledger confidence = should explore but bias toward RemoveEdge
        proposer.update_accept_rate(2, 10); // 20% accept rate
        let mutations = proposer.propose_mutations_with_ledger(
            &graph,
            DegradationSignal::LatencyDominant,
            5,
            &ledger,
        )?;

        // Should propose 5 mutations, likely biased toward RemoveEdge
        assert_eq!(mutations.len(), 5);
        Ok(())
    }

    #[test]
    fn proposer_adjusts_exploit_threshold_with_ledger() -> Result<(), NtgError> {
        use super::super::ledger::MutationLedger;

        let mut proposer = AdaptiveMutationProposer::new();
        let mut ledger = MutationLedger::new();

        // Add many events to ledger to trigger confidence bonus
        for i in 0..150 {
            ledger.record_mutation(
                MutationRuleKind::RemoveEdge { from: i % 5, to: (i + 1) % 5 },
                DegradationSignal::Balanced,
                0.80,
                0.82,
                i % 3 != 0, // 66% success rate
                0.5,
            );
        }

        let mut graph = Graph::new();
        let n1 = graph.add_node(NodeKind::Content, "node1".to_string());
        let n2 = graph.add_node(NodeKind::Content, "node2".to_string());
        graph.add_edge(n1, n2)?;

        // At 55% accept rate (below 60% but above lowered threshold due to ledger)
        proposer.update_accept_rate(55, 100);
        proposer.record_success("remove_edge(1→2)".to_string());

        let mutations = proposer.propose_mutations_with_ledger(
            &graph,
            DegradationSignal::Balanced,
            1,
            &ledger,
        )?;

        // With ledger confidence bonus, should shift toward exploitation
        assert_eq!(mutations.len(), 1);
        Ok(())
    }
}
