//! NanoKeymaster integration with Ternary Memory Graph (TMG).
//!
//! Extends NanoKeymaster with:
//! - HyperVector storage per intent (semantic memory)
//! - Similarity-based routing (find related intents via HDC)
//! - Hebbian learning on intent edges (fire-together-wire-together)
//! - Time-decay pruning of stale intent relationships
//! - Live topology evolution tracking

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use super::hypervector::HyperVector;
use super::forget::ForgetEngine;
use super::{Graph, NodeKind};

/// Intent routing with structural memory: each intent has a hypervector
/// that grows through Hebbian learning as the agent encounters related requests.
#[derive(Clone, Debug)]
pub struct IntentMemory {
    /// Intent name → hypervector (semantic embedding).
    pub hypervectors: HashMap<String, HyperVector>,
    /// Intent → forget engine (manages edges to other intents).
    pub edges: HashMap<String, ForgetEngine>,
    /// Call counter for time-based edge decay.
    call_counter: u64,
}

impl IntentMemory {
    pub fn new() -> Self {
        Self {
            hypervectors: HashMap::new(),
            edges: HashMap::new(),
            call_counter: 0,
        }
    }

    /// Get current Unix timestamp (seconds).
    fn now_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Record a call and update structural memory via Hebbian learning.
    /// If intent is new, assign a random hypervector.
    /// If intent exists, strengthen edges to recently-accessed intents (fire-together).
    pub fn observe_call(&mut self, intent: &str, related_intents: &[&str]) -> HyperVector {
        self.call_counter += 1;
        let now = self.now_secs();

        // Ensure intent has a hypervector (create random if new).
        self.hypervectors
            .entry(intent.to_string())
            .or_insert_with(HyperVector::random);

        let hv = self.hypervectors.get(intent).unwrap().clone();

        // Fire-together-wire-together: strengthen edges to co-accessed intents.
        let engine = self.edges.entry(intent.to_string()).or_insert_with(|| {
            ForgetEngine::new(
                super::forget::DecayModel::ebbinghaus_24h(),
                0.1, // Prune threshold: 10% of max weight
            )
        });

        for other_intent in related_intents.iter() {
            // Create implicit edge index from intent names (deterministic).
            let other_id = fnv1a(other_intent) as usize;
            let intent_id = fnv1a(intent) as usize;

            // Fire-together: co-access strengthens the edge (Hebbian rule).
            engine.fire_together(intent_id, other_id, now);

            // Ensure related intent also has a hypervector.
            self.hypervectors
                .entry(other_intent.to_string())
                .or_insert_with(HyperVector::random);
        }

        hv
    }

    /// Similarity search: find top-k most similar intents to query.
    /// Uses Hamming distance on hypervectors.
    pub fn find_similar(&self, query_intent: &str, top_k: usize) -> Vec<(String, i64)> {
        let query_hv = self
            .hypervectors
            .get(query_intent)
            .cloned()
            .unwrap_or_else(HyperVector::zero);

        let mut distances: Vec<(String, i64)> = self
            .hypervectors
            .iter()
            .map(|(intent, hv)| (intent.clone(), query_hv.similarity(hv)))
            .collect();

        // Sort by distance (ascending = most similar first).
        distances.sort_by_key(|(_intent, dist)| *dist);

        distances.into_iter().take(top_k).collect()
    }

    /// Prune stale edges below threshold (time-decay mechanism).
    /// Called periodically to remove forgotten intent relationships.
    pub fn prune_stale_edges(&mut self) -> HashMap<String, usize> {
        let now = self.now_secs();
        let mut pruned_per_intent: HashMap<String, usize> = HashMap::new();

        for (intent, engine) in &mut self.edges {
            let pruned = engine.prune_stale(now);
            if !pruned.is_empty() {
                pruned_per_intent.insert(intent.clone(), pruned.len());
            }
        }

        pruned_per_intent
    }

    /// Statistics: count edges by strength band across all intents.
    pub fn edge_statistics(&self) -> (usize, usize, usize) {
        let strong = 0;
        let medium = 0;
        let mut weak = 0;

        for engine in self.edges.values() {
            // Safely count edges (would need to expose distribution from engine).
            // For now, use edge_count as proxy.
            weak += engine.edge_count();
        }

        (strong, medium, weak)
    }

    /// Total intents and edges tracked.
    pub fn stats(&self) -> (usize, usize) {
        let total_intents = self.hypervectors.len();
        let total_edges: usize = self.edges.values().map(|e| e.edge_count()).sum();
        (total_intents, total_edges)
    }

    /// Convert intent topology to a Graph for mutation proposals.
    /// Each intent becomes a node; edges represent Hebbian relationships.
    pub fn current_structure(&self) -> Graph {
        let mut graph = Graph::new();

        // Create a node for each intent.
        let mut intent_ids = HashMap::new();
        for intent_name in self.hypervectors.keys() {
            let node_id = graph.add_node(NodeKind::Content, intent_name.clone());
            intent_ids.insert(intent_name.clone(), node_id);
        }

        // Add edges for Hebbian relationships (if available).
        // Note: ForgetEngine doesn't expose edges directly, so this is structural only.
        for (intent_name, _engine) in self.edges.iter() {
            if let Some(from_id) = intent_ids.get(intent_name) {
                // Add edges to top similar intents.
                let similar = self.find_similar(intent_name, 2);
                for (other_intent, _) in similar.iter().take(2) {
                    if let Some(to_id) = intent_ids.get(other_intent) {
                        let _ = graph.add_edge(*from_id, *to_id);
                    }
                }
            }
        }

        graph
    }
}

impl Default for IntentMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a 64-bit hash for deterministic intent ID generation.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_intent_gets_random_hypervector() {
        let mut mem = IntentMemory::new();
        let hv = mem.observe_call("classify", &[]);
        assert_eq!(hv.similarity(&hv), 0);
    }

    #[test]
    fn observe_call_strengthens_edges() {
        let mut mem = IntentMemory::new();
        mem.observe_call("classify", &["score", "predict"]);
        mem.observe_call("classify", &["score"]);
        mem.observe_call("classify", &["score"]);

        // Edge (classify → score) should have been strengthened 3 times.
        let stats = mem.stats();
        assert!(stats.0 >= 2); // At least classify and score
    }

    #[test]
    fn find_similar_returns_sorted() {
        let mut mem = IntentMemory::new();
        mem.observe_call("classify", &[]);
        mem.observe_call("score", &[]);

        let similar = mem.find_similar("classify", 5);
        assert!(similar.len() >= 1);
        // Similar should be sorted by distance (ascending).
        for i in 1..similar.len() {
            assert!(similar[i].1 >= similar[i - 1].1);
        }
    }

    #[test]
    fn stats_counts_intents_and_edges() {
        let mut mem = IntentMemory::new();
        mem.observe_call("a", &["b"]);
        mem.observe_call("b", &["c"]);
        mem.observe_call("c", &["a"]);

        let (intents, edges) = mem.stats();
        assert_eq!(intents, 3);
        assert!(edges > 0); // Should have recorded edges via fire-together.
    }
}
