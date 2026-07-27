//! Does a *trained* classifier separate real parent-child edges from
//! random node pairs any better than the fixed, untrained
//! `normalized_edge_interaction_score` formula did?
//!
//! docs/EXPERIMENTS.md (2026-07-08) found the fixed formula does not --
//! on this repo's real docs, real edges and random pairs land within one
//! std of each other after removing the length confound -- and named the
//! obvious follow-up as "not yet tried": a genuine relatedness signal
//! probably needs actual learned weights. This binary runs that
//! follow-up and reports the honest result either way, same discipline
//! as `phase4_calib` / `density_bench` / `gemm_bench`.
//!
//! Run:
//!   cargo run --release --bin edge_relatedness_bench
//!
//! Corpus: this repo's own real ADRs + DESIGN.md + ROADMAP.md (identical
//! set to `tests/self_parse.rs` and the original diagnosis), so results
//! are reproducible from the checked-in docs, not an external dataset.

use ntg_kernel::ntg::edge_calib::{
    best_threshold, edge_baseline_accuracy, edge_class_metrics, edge_stratified_split,
    train_edge_perceptron,
};
use ntg_kernel::ntg::interaction::normalized_edge_interaction_score;
use ntg_kernel::ntg::docparse;
use ntg_kernel::ntg::graph::Graph;

const ADR_0001: &str = include_str!("../../../docs/architecture/0001-vision-and-pivot.md");
const ADR_0002: &str =
    include_str!("../../../docs/architecture/0002-safety-rails-for-self-modification.md");
const ADR_0003: &str = include_str!("../../../docs/architecture/0003-sis-frontend.md");
const DESIGN: &str = include_str!("../../../docs/DESIGN.md");
const ROADMAP: &str = include_str!("../../../docs/ROADMAP.md");

const EPOCHS: usize = 25;
const SEED: u64 = 0x0ED9_CA11u64;

/// Fixed-formula baseline: threshold-sweep `normalized_edge_interaction_score`
/// itself as a single-feature classifier, on the *same* real graph and the
/// *same* test split's pairs, for a fair trained-vs-untrained comparison.
fn fixed_formula_metrics(graph: &Graph, pairs: &[(usize, usize, bool)]) -> (f32, f32) {
    // Score every pair once.
    let scores: Vec<(f32, bool)> = pairs
        .iter()
        .filter_map(|&(a, b, is_edge)| {
            normalized_edge_interaction_score(graph, a, b)
                .ok()
                .map(|s| (s, is_edge))
        })
        .collect();
    if scores.is_empty() {
        return (0.0, 0.0);
    }
    // Sweep thresholds over observed score range, maximize balanced accuracy.
    let mut candidates: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
    candidates.push(f32::NEG_INFINITY);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut best_bal = -1.0f32;
    let mut best_acc = 0.0f32;
    for &thr in &candidates {
        let (mut tp, mut tn, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
        for &(s, is_edge) in &scores {
            let pred = s >= thr;
            match (pred, is_edge) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
            }
        }
        let has_pos = tp + fn_ > 0;
        let has_neg = tn + fp > 0;
        let tpr = if has_pos { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
        let tnr = if has_neg { tn as f32 / (tn + fp) as f32 } else { 0.0 };
        let bal = match (has_pos, has_neg) {
            (true, true) => 0.5 * (tpr + tnr),
            (true, false) => tpr,
            (false, true) => tnr,
            (false, false) => 0.0,
        };
        if bal > best_bal {
            best_bal = bal;
            best_acc = (tp + tn) as f32 / scores.len() as f32;
        }
    }
    (best_acc, best_bal.max(0.0))
}

fn main() {
    let docs: Vec<(&str, &str)> = vec![
        ("adr-0001", ADR_0001),
        ("adr-0002", ADR_0002),
        ("adr-0003", ADR_0003),
        ("design", DESIGN),
        ("roadmap", ROADMAP),
    ];

    let mut graph = Graph::new();
    for &(name, text) in &docs {
        docparse::parse_into(&mut graph, name, text);
    }
    let samples = ntg_kernel::ntg::edge_calib::edge_samples_from_graph(&graph, SEED)
        .expect("real docs parse into a valid graph");

    let (train, test) = edge_stratified_split(&samples, 0.8);
    let weights = train_edge_perceptron(&train, EPOCHS);
    let threshold = best_threshold(&weights, &train);

    let train_metrics = edge_class_metrics(&weights, &train, threshold);
    let test_metrics = edge_class_metrics(&weights, &test, threshold);
    let base_acc = edge_baseline_accuracy(&test);

    // Rebuild (node_a, node_b, is_edge) id-pairs matching `test` for the
    // fixed-formula comparison (edge_calib only keeps encoded features,
    // not raw ids, so recompute independently here from the same graph).
    let mut real_edges = Vec::new();
    for id in graph.all_node_ids() {
        for child in graph.children(id) {
            real_edges.push((id, child, true));
        }
    }
    let mut state = SEED ^ 0xED6E_5EEDu64;
    let is_edge_set: std::collections::HashSet<(usize, usize)> =
        real_edges.iter().map(|&(a, b, _)| (a, b)).collect();
    let n = graph.all_node_ids().len();
    let ids = graph.all_node_ids();
    let mut random_pairs = Vec::new();
    let mut attempts = 0usize;
    while random_pairs.len() < real_edges.len() && attempts < real_edges.len() * 50 + 200 {
        attempts += 1;
        let ia = (xorshift(&mut state) as usize) % n.max(1);
        let ib = (xorshift(&mut state) as usize) % n.max(1);
        if ia == ib {
            continue;
        }
        let (a, b) = (ids[ia], ids[ib]);
        if is_edge_set.contains(&(a, b)) || is_edge_set.contains(&(b, a)) {
            continue;
        }
        random_pairs.push((a, b, false));
    }
    let mut all_pairs = real_edges;
    all_pairs.extend(random_pairs);
    let (fixed_acc, fixed_bal) = fixed_formula_metrics(&graph, &all_pairs);

    println!("# edge_relatedness_bench");
    println!(
        "corpus: 5 real docs, {} nodes, {} real edges, epochs={EPOCHS}",
        graph.node_count(),
        samples.iter().filter(|s| s.is_real_edge).count()
    );
    println!();
    println!("| classifier | split | accuracy | balanced accuracy | precision | recall | f1 |");
    println!("|---|---|---:|---:|---:|---:|---:|");
    println!(
        "| majority baseline | test | {:.3} | 0.500 | - | - | - |",
        base_acc
    );
    println!(
        "| fixed normalized_edge_interaction_score (threshold-swept) | test (approx, resampled negatives) | {:.3} | {:.3} | - | - | - |",
        fixed_acc, fixed_bal
    );
    println!(
        "| trained ternary perceptron (this experiment) | train | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
        train_metrics.accuracy, train_metrics.balanced_accuracy, train_metrics.precision, train_metrics.recall, train_metrics.f1
    );
    println!(
        "| trained ternary perceptron (this experiment) | test | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
        test_metrics.accuracy, test_metrics.balanced_accuracy, test_metrics.precision, test_metrics.recall, test_metrics.f1
    );

    println!();
    let delta = test_metrics.balanced_accuracy - 0.5;
    println!(
        "delta_vs_chance_balanced_accuracy: {:+.3} ({})",
        delta,
        if delta > 0.05 { "meaningful separation" } else { "no meaningful separation -- consistent with the 2026-07-08 fixed-formula finding" }
    );

    println!();
    println!("## JSON");
    println!(
        r#"{{"n_nodes":{},"n_real_edges":{},"n_train":{},"n_test":{},"epochs":{EPOCHS},"baseline_accuracy":{:.4},"fixed_formula_accuracy":{:.4},"fixed_formula_balanced_accuracy":{:.4},"trained_test_accuracy":{:.4},"trained_test_balanced_accuracy":{:.4},"trained_test_f1":{:.4}}}"#,
        graph.node_count(),
        samples.iter().filter(|s| s.is_real_edge).count(),
        train.len(),
        test.len(),
        base_acc,
        fixed_acc,
        fixed_bal,
        test_metrics.accuracy,
        test_metrics.balanced_accuracy,
        test_metrics.f1
    );
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
