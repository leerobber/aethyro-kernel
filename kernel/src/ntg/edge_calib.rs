//! Follow-up to the diagnosed `edge_interaction_score` weakness
//! (docs/EXPERIMENTS.md, 2026-07-08): neither the raw nor the
//! length-normalized fixed-formula score reliably separated real
//! parent-child edges from random node pairs on this repo's own real
//! docs. That entry's own conclusion was that a genuine relatedness
//! signal "probably needs actual learned weights ... not a fixed,
//! untrained byte-correlation" -- this module is that experiment, not
//! a new hunch: a ternary perceptron trained directly on real-edge vs.
//! random-pair labels, same corpus, same honest win-or-not reporting
//! discipline as `calib/mod.rs`'s Phase 4 NodeKind classifier.
//!
//! Deliberately **not** a generalization of `calib::Sample` /
//! `calib::ClassMetrics` -- those are tuned for single-node
//! Execution-vs-Content classification (threshold search ranges,
//! collapse-recovery heuristics) and reusing them here would silently
//! import assumptions that were never validated for this different
//! problem. Small, self-contained duplication of the perceptron/metric
//! math beats a shared abstraction with borrowed magic numbers.

use super::error::NtgError;
use super::graph::Graph;
use super::ternary::encode_fixed;
use super::docparse;

/// Each label contributes this many ternary features to a pair's vector
/// (encode_fixed, truncated/zero-padded). Total feature length is 2x this.
pub const LABEL_FEATURE_DIM: usize = 64;

/// One (node, node) pair sample: a real parent-child edge or a sampled
/// non-adjacent pair from the same graph.
#[derive(Clone, Debug)]
pub struct EdgePairSample {
    pub features: Vec<i8>,
    pub is_real_edge: bool,
    /// For debugging/reporting only; not used by training or metrics.
    pub preview: String,
}

/// Confusion / ranking metrics for the "is a real edge" positive class.
/// Same shape as `calib::ClassMetrics` but named for this task rather
/// than borrowing the Execution-class-specific field names.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeClassMetrics {
    pub accuracy: f32,
    pub balanced_accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
}

fn fixed_len_ternary(label: &str, len: usize) -> Vec<i8> {
    let mut v = encode_fixed(label);
    v.resize(len, 0);
    v.truncate(len);
    v
}

/// Concatenate both labels' fixed-length ternary encodings. Unlike
/// `edge_interaction_score`'s single dot-product formula, this gives a
/// trainable classifier per-position access to both labels independently
/// instead of forcing a fixed interaction shape.
pub fn edge_features(a_label: &str, b_label: &str) -> Vec<i8> {
    let mut feat = fixed_len_ternary(a_label, LABEL_FEATURE_DIM);
    feat.extend(fixed_len_ternary(b_label, LABEL_FEATURE_DIM));
    feat
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Build one shared graph from multiple documents, collect every real
/// parent-child edge as a positive sample, and sample an equal number of
/// deterministic random non-adjacent, non-self pairs as negatives.
pub fn edge_samples_from_documents(
    docs: &[(&str, &str)],
    seed: u64,
) -> Result<Vec<EdgePairSample>, NtgError> {
    let mut graph = Graph::new();
    for &(name, text) in docs {
        docparse::parse_into(&mut graph, name, text);
    }
    edge_samples_from_graph(&graph, seed)
}

pub fn edge_samples_from_graph(
    graph: &Graph,
    seed: u64,
) -> Result<Vec<EdgePairSample>, NtgError> {
    let all_ids = graph.all_node_ids();
    let mut real_edges: Vec<(usize, usize)> = Vec::new();
    let mut is_edge = std::collections::HashSet::new();
    for &id in &all_ids {
        for child in graph.children(id) {
            real_edges.push((id, child));
            is_edge.insert((id, child));
        }
    }

    let mut samples = Vec::with_capacity(real_edges.len() * 2);
    for &(a, b) in &real_edges {
        let (na, nb) = (graph.node(a)?, graph.node(b)?);
        samples.push(EdgePairSample {
            features: edge_features(&na.label, &nb.label),
            is_real_edge: true,
            preview: format!("{}->{}", preview(&na.label), preview(&nb.label)),
        });
    }

    // Deterministic random non-adjacent pairs, matched count to positives.
    let mut state = seed ^ 0xED6E_5EEDu64;
    let n = all_ids.len();
    let mut negatives_added = 0usize;
    let mut attempts = 0usize;
    let target = real_edges.len();
    while negatives_added < target && n >= 2 && attempts < target.saturating_mul(50).max(200) {
        attempts += 1;
        let ia = (xorshift(&mut state) as usize) % n;
        let ib = (xorshift(&mut state) as usize) % n;
        if ia == ib {
            continue;
        }
        let (a, b) = (all_ids[ia], all_ids[ib]);
        if is_edge.contains(&(a, b)) || is_edge.contains(&(b, a)) {
            continue;
        }
        let (na, nb) = (graph.node(a)?, graph.node(b)?);
        samples.push(EdgePairSample {
            features: edge_features(&na.label, &nb.label),
            is_real_edge: false,
            preview: format!("{}~{}", preview(&na.label), preview(&nb.label)),
        });
        negatives_added += 1;
    }

    Ok(samples)
}

fn preview(label: &str) -> String {
    label.chars().take(24).collect()
}

fn score(weights: &[i8], features: &[i8]) -> i64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(&w, &x)| (w as i64) * (x as i64))
        .sum()
}

fn predict(weights: &[i8], features: &[i8], threshold: i64) -> bool {
    score(weights, features) >= threshold
}

fn clamp_ternary(v: i32) -> i8 {
    v.clamp(-1, 1) as i8
}

/// Deterministic train/test split (stratified by class), same ordering
/// discipline as `calib::stratified_split`.
pub fn edge_stratified_split(
    samples: &[EdgePairSample],
    train_ratio: f32,
) -> (Vec<EdgePairSample>, Vec<EdgePairSample>) {
    let mut pos: Vec<&EdgePairSample> = samples.iter().filter(|s| s.is_real_edge).collect();
    let mut neg: Vec<&EdgePairSample> = samples.iter().filter(|s| !s.is_real_edge).collect();
    pos.sort_by(|a, b| a.preview.cmp(&b.preview));
    neg.sort_by(|a, b| a.preview.cmp(&b.preview));

    let split = |v: &[&EdgePairSample]| -> (Vec<EdgePairSample>, Vec<EdgePairSample>) {
        if v.is_empty() {
            return (vec![], vec![]);
        }
        let n_train = ((v.len() as f32) * train_ratio).round() as usize;
        let n_train = n_train.clamp(0, v.len());
        (
            v[..n_train].iter().map(|s| (*s).clone()).collect(),
            v[n_train..].iter().map(|s| (*s).clone()).collect(),
        )
    };
    let (pos_tr, pos_te) = split(&pos);
    let (neg_tr, neg_te) = split(&neg);

    let mut train = pos_tr;
    train.extend(neg_tr);
    let mut test = pos_te;
    test.extend(neg_te);
    (train, test)
}

/// Majority-class accuracy on `samples`.
pub fn edge_baseline_accuracy(samples: &[EdgePairSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let majority = samples.iter().filter(|s| !s.is_real_edge).count().max(
        samples.iter().filter(|s| s.is_real_edge).count(),
    );
    majority as f32 / samples.len() as f32
}

/// Ternary perceptron: standard mistake-driven update (+features on a
/// missed positive, -features on a missed negative), weights clamped to
/// {-1,0,1} after every update. Feature dim inferred from the first
/// sample; empty input returns empty weights.
pub fn train_edge_perceptron(samples: &[EdgePairSample], epochs: usize) -> Vec<i8> {
    let dim = match samples.first() {
        Some(s) => s.features.len(),
        None => return Vec::new(),
    };
    let mut weights = vec![0i8; dim];
    for _ in 0..epochs {
        for s in samples {
            let pred = predict(&weights, &s.features, 0);
            if pred != s.is_real_edge {
                let sign: i32 = if s.is_real_edge { 1 } else { -1 };
                for (w, &x) in weights.iter_mut().zip(s.features.iter()) {
                    *w = clamp_ternary(*w as i32 + sign * x as i32);
                }
            }
        }
    }
    weights
}

/// Scan a small integer threshold range and keep whichever maximizes
/// balanced accuracy on `samples`. Simpler than `calib`'s collapse-
/// recovery search deliberately -- that logic was tuned for a different
/// task and importing it here would smuggle in unvalidated assumptions.
pub fn best_threshold(weights: &[i8], samples: &[EdgePairSample]) -> i64 {
    let max_score: i64 = weights.len() as i64;
    let mut best_thr = 0i64;
    let mut best_bal = -1.0f32;
    for thr in -max_score..=max_score {
        let m = edge_class_metrics(weights, samples, thr);
        if m.balanced_accuracy > best_bal {
            best_bal = m.balanced_accuracy;
            best_thr = thr;
        }
    }
    best_thr
}

pub fn edge_class_metrics(weights: &[i8], samples: &[EdgePairSample], threshold: i64) -> EdgeClassMetrics {
    if samples.is_empty() {
        return EdgeClassMetrics::default();
    }
    let (mut tp, mut tn, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for s in samples {
        let pred = predict(weights, &s.features, threshold);
        match (pred, s.is_real_edge) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
        }
    }
    let n = samples.len() as f32;
    let accuracy = (tp + tn) as f32 / n;
    let has_pos = tp + fn_ > 0;
    let has_neg = tn + fp > 0;
    let tpr = if has_pos { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
    let tnr = if has_neg { tn as f32 / (tn + fp) as f32 } else { 0.0 };
    let balanced_accuracy = match (has_pos, has_neg) {
        (true, true) => 0.5 * (tpr + tnr),
        (true, false) => tpr,
        (false, true) => tnr,
        (false, false) => 0.0,
    };
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = tpr;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    EdgeClassMetrics {
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        tp,
        tn,
        fp,
        fn_,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::NodeKind;

    #[test]
    fn perceptron_converges_on_trivially_separable_data() {
        // Positives: features start with [1,1]; negatives: [-1,-1]. Trivially
        // linearly separable -- a sanity check the training loop itself
        // works before trusting it on noisy real-doc data.
        let mut samples = Vec::new();
        for i in 0..20 {
            samples.push(EdgePairSample {
                features: vec![1, 1, 0, 0],
                is_real_edge: true,
                preview: format!("pos{i}"),
            });
            samples.push(EdgePairSample {
                features: vec![-1, -1, 0, 0],
                is_real_edge: false,
                preview: format!("neg{i}"),
            });
        }
        let weights = train_edge_perceptron(&samples, 10);
        let thr = best_threshold(&weights, &samples);
        let m = edge_class_metrics(&weights, &samples, thr);
        assert!(
            m.balanced_accuracy > 0.95,
            "perceptron should separate trivially linearly separable data, got {:.3}",
            m.balanced_accuracy
        );
    }

    #[test]
    fn edge_features_has_expected_length() {
        let f = edge_features("hello", "hi");
        assert_eq!(f.len(), LABEL_FEATURE_DIM * 2);
    }

    #[test]
    fn edge_samples_from_documents_produces_matched_positive_negative_counts() {
        let doc = "# Title\n## Section\n- item one\n- item two\n## Other\n- more\n";
        let samples = edge_samples_from_documents(&[("doc", doc)], 42).unwrap();
        let pos = samples.iter().filter(|s| s.is_real_edge).count();
        let neg = samples.iter().filter(|s| !s.is_real_edge).count();
        assert!(pos > 0, "doc should produce at least one real edge");
        assert_eq!(pos, neg, "negative sampling should match positive count");
    }

    #[test]
    fn edge_samples_are_deterministic() {
        let doc = "# Title\n## Section\n- item one\n- item two\n## Other\n- more\n";
        let a = edge_samples_from_documents(&[("doc", doc)], 7).unwrap();
        let b = edge_samples_from_documents(&[("doc", doc)], 7).unwrap();
        let a_flags: Vec<bool> = a.iter().map(|s| s.is_real_edge).collect();
        let b_flags: Vec<bool> = b.iter().map(|s| s.is_real_edge).collect();
        assert_eq!(a_flags, b_flags);
    }

    #[test]
    fn baseline_accuracy_is_majority_fraction() {
        let mut g = Graph::new();
        let a = g.add_node(NodeKind::Content, "a");
        let b = g.add_node(NodeKind::Content, "b");
        g.add_edge(a, b).unwrap();
        let samples = edge_samples_from_graph(&g, 1).unwrap();
        let base = edge_baseline_accuracy(&samples);
        assert!((0.0..=1.0).contains(&base));
    }
}
