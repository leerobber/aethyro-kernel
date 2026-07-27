//! End-to-end multi-layer forward-pass benchmark ("GEMM-scale").
//!
//! `density_bench` and `ld_simd_bench` measure isolated dot products.
//! Nothing before this exercised `Runtime::forward_native_parallel` across
//! more than 1-2 toy nodes, so there was no evidence the parallel/threaded
//! scheduling path holds up (in correctness or in wall-clock) once a layer
//! has hundreds to low-thousands of nodes chained across several layers.
//! This does both: builds real multi-node, multi-layer sparse ternary
//! networks and times the actual `Runtime` forward chain, then checks every
//! layer's output bit-for-bit against a single-threaded serial reference
//! that calls the same `SparseBitSlicedTernary::ternary_matmul` primitive
//! node-by-node (no threading, no chunking).
//!
//! Layer wiring convention: a layer of `n` nodes emits an activation tensor
//! of length `n * 64`, one meaningful value at bit-offset 0 of chunk `id`
//! per node (see `Runtime::forward_native_parallel`). So a downstream
//! layer's weight vectors are built at length `prev_layer_len * 64`, with
//! any nonzero mass placed only at those `logical_idx * 64` chunk-0
//! positions -- placing it elsewhere would just multiply against
//! structurally-zero bits and silently do nothing.
//!
//! Run:
//!   cargo run --release --bin gemm_bench

use ntg_kernel::ntg::graph::GraphNode;
use ntg_kernel::{Runtime, SparseBitSlicedTernary};
use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 15;
const NUM_LAYERS: usize = 3;
const THRESHOLD: i64 = 1;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Build a sparse ternary vector of `logical_dim` slots, each mapped to
/// physical bit-offset `slot * 64` (see module doc), nonzero with
/// probability `density`.
fn make_strided_sparse(logical_dim: usize, density: f32, seed: u64) -> SparseBitSlicedTernary {
    let mut s = seed;
    let mut t = SparseBitSlicedTernary::new(logical_dim * 64);
    let threshold = ((density as f64) * (u64::MAX as f64)) as u64;
    for slot in 0..logical_dim {
        if xorshift(&mut s) < threshold {
            let val: i8 = if xorshift(&mut s) & 1 == 0 { 1 } else { -1 };
            t.set(slot * 64, val);
        }
    }
    t.compute_density();
    t
}

/// One layer: `layer_len` nodes, each a strided-sparse weight vector over
/// `input_logical_dim` slots (see [`make_strided_sparse`]).
fn make_layer(layer_len: usize, input_logical_dim: usize, density: f32, seed: u64) -> Vec<GraphNode> {
    (0..layer_len)
        .map(|id| {
            GraphNode::with_weights(
                id,
                make_strided_sparse(input_logical_dim, density, seed ^ (id as u64).wrapping_mul(0x9E37_79B9)),
            )
        })
        .collect()
}

/// Single-threaded reference forward for one layer: identical primitive
/// (`ternary_matmul`, take first block) to `Runtime::forward_native_parallel`,
/// just called serially with no chunking/threading.
fn golden_layer_forward(layer: &[GraphNode], input: &SparseBitSlicedTernary) -> SparseBitSlicedTernary {
    let mut blocks = Vec::new();
    for node in layer {
        let node_out = SparseBitSlicedTernary::ternary_matmul(&node.weights, input, THRESHOLD);
        if let Some((_, block)) = node_out.blocks.first() {
            if !block.is_empty() {
                blocks.push((node.id as u32, *block));
            }
        }
    }
    let mut out = SparseBitSlicedTernary::with_capacity(layer.len() * 64, blocks.len());
    out.blocks = blocks;
    out.compute_density();
    out
}

fn golden_chain_forward(
    layers: &[Vec<GraphNode>],
    input: &SparseBitSlicedTernary,
) -> Vec<SparseBitSlicedTernary> {
    let mut activations = input.clone();
    let mut outputs = Vec::with_capacity(layers.len());
    for layer in layers {
        activations = golden_layer_forward(layer, &activations);
        outputs.push(activations.clone());
    }
    outputs
}

fn runtime_chain_forward(
    rt: &Runtime,
    input: &SparseBitSlicedTernary,
) -> Result<Vec<SparseBitSlicedTernary>, ntg_kernel::NtgError> {
    let mut activations = input.clone();
    let mut outputs = Vec::with_capacity(rt.layers.len());
    for layer_idx in 0..rt.layers.len() {
        activations = rt.forward_native_parallel(layer_idx, &activations, THRESHOLD)?;
        outputs.push(activations.clone());
    }
    Ok(outputs)
}

fn tensors_match(a: &SparseBitSlicedTernary, b: &SparseBitSlicedTernary) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_blocks: Vec<_> = a.blocks.iter().filter(|(_, b)| !b.is_empty()).collect();
    let mut b_blocks: Vec<_> = b.blocks.iter().filter(|(_, b)| !b.is_empty()).collect();
    a_blocks.sort_by_key(|&&(c, _)| c);
    b_blocks.sort_by_key(|&&(c, _)| c);
    a_blocks == b_blocks
}

fn median_ns(samples: &mut [u128]) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

struct Config {
    label: &'static str,
    layer_len: usize,
    input_density: f32,
}

fn run_config(cfg: &Config) -> (f64, bool, Vec<f32>) {
    let seed = 0xC0FF_EE00u64 ^ cfg.layer_len as u64 ^ (cfg.input_density.to_bits() as u64);

    // Layer 0 takes an "external" input of the same logical width as its
    // own node count, purely to keep every layer in this bench square.
    let input = make_strided_sparse(cfg.layer_len, cfg.input_density, seed);

    let mut layers = Vec::with_capacity(NUM_LAYERS);
    let mut prev_logical_dim = cfg.layer_len;
    for l in 0..NUM_LAYERS {
        layers.push(make_layer(
            cfg.layer_len,
            prev_logical_dim,
            cfg.input_density,
            seed ^ ((l as u64) << 32),
        ));
        prev_logical_dim = cfg.layer_len;
    }

    let mut rt = Runtime::new();
    for layer in &layers {
        rt.push_layer(layer.clone()).expect("sequential ids by construction");
    }

    // Correctness oracle, computed once, outside the timing loop.
    let golden_outputs = golden_chain_forward(&layers, &input);
    let runtime_outputs = runtime_chain_forward(&rt, &input).expect("shape-matched by construction");
    let all_match = golden_outputs.len() == runtime_outputs.len()
        && golden_outputs
            .iter()
            .zip(runtime_outputs.iter())
            .all(|(g, r)| tensors_match(g, r));
    let per_layer_density: Vec<f32> = runtime_outputs.iter().map(|t| t.density()).collect();

    let mut samples = Vec::with_capacity(ITERS);
    for i in 0..(WARMUP + ITERS) {
        let t0 = Instant::now();
        let out = runtime_chain_forward(&rt, black_box(&input)).expect("shape-matched by construction");
        let elapsed = t0.elapsed().as_nanos();
        black_box(&out);
        if i >= WARMUP {
            samples.push(elapsed);
        }
    }

    (median_ns(&mut samples) / 1000.0, all_match, per_layer_density)
}

fn main() {
    println!("# gemm_bench");
    println!(
        "layers={NUM_LAYERS} warmup={WARMUP} iters={ITERS} threshold={THRESHOLD} (median wall-clock)"
    );
    println!("Each row is a full {NUM_LAYERS}-layer forward chain, not a single dot product.");
    println!();
    println!("| shape | nodes/layer | input density | chain µs | nodes/sec | per-layer output density | matches serial reference |");
    println!("|---|---:|---:|---:|---:|---|:---:|");

    let configs = [
        Config { label: "small (256x256x3)", layer_len: 256, input_density: 0.05 },
        Config { label: "small (256x256x3)", layer_len: 256, input_density: 0.20 },
        Config { label: "small (256x256x3)", layer_len: 256, input_density: 0.60 },
        Config { label: "gemm (1024x1024x3)", layer_len: 1024, input_density: 0.05 },
        Config { label: "gemm (1024x1024x3)", layer_len: 1024, input_density: 0.20 },
        Config { label: "gemm (1024x1024x3)", layer_len: 1024, input_density: 0.60 },
    ];

    let mut rows = Vec::new();
    for cfg in &configs {
        let (chain_us, matches, densities) = run_config(cfg);
        let nodes_per_sec = (cfg.layer_len * NUM_LAYERS) as f64 / (chain_us / 1_000_000.0);
        let density_str = densities
            .iter()
            .map(|d| format!("{:.3}", d))
            .collect::<Vec<_>>()
            .join(" -> ");
        println!(
            "| {} | {} | {:.0}% | {:.2} | {:.0} | {} | {} |",
            cfg.label,
            cfg.layer_len,
            cfg.input_density * 100.0,
            chain_us,
            nodes_per_sec,
            density_str,
            if matches { "yes" } else { "NO" }
        );
        rows.push((cfg, chain_us, nodes_per_sec, matches));
    }

    println!();
    println!("## JSON");
    print!("[");
    for (i, (cfg, chain_us, nodes_per_sec, matches)) in rows.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(
            r#"{{"shape":"{}","layer_len":{},"num_layers":{},"input_density":{},"chain_us":{:.4},"nodes_per_sec":{:.1},"matches_serial_reference":{}}}"#,
            cfg.label, cfg.layer_len, NUM_LAYERS, cfg.input_density, chain_us, nodes_per_sec, matches
        );
    }
    println!("]");

    if rows.iter().any(|(_, _, _, matches)| !*matches) {
        eprintln!("ERROR: parallel forward diverged from serial reference at GEMM scale");
        std::process::exit(1);
    }
}
