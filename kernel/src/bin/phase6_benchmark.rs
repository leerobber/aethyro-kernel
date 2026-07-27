//! Phase 6: Head-to-Head Benchmark — Ternary Kernel vs. f32 Reference
//!
//! Phase 6's decision gate is: does this engine beat current production
//! inference on memory and/or speed? This binary addresses the part of
//! that question the kernel *can* answer today: how does ternary GEMM
//! compare to f32 GEMM at production-like scales?
//!
//! What this binary measures (honestly):
//!   - Ternary bit-sliced dot product (dual-stream AVX-512/AVX2/scalar path)
//!     vs. scalar f32 multiply-accumulate at matched problem sizes
//!   - Memory footprint: ternary (2 bits/weight) vs. f32 (32 bits/weight)
//!   - Throughput: operations per second at each scale
//!   - Correctness: ternary decode matches f32-quantized reference values
//!
//! What this binary does NOT measure (stated explicitly per ADR 0001):
//!   - aethyro.com's production inference pipeline (external; not in this repo)
//!   - Full-pipeline end-to-end latency (FFI host overhead, Python bridge, API)
//!   - Model accuracy on real aethyro.com task benchmarks (needs live API access)
//!   - GPU throughput (explicitly deferred — CPU TOBL already 12-52×)
//!
//! These gaps are the remaining Phase 6 work: they require aethyro.com
//! API integration, not kernel-only code. This binary closes the kernel
//! half of Phase 6. The product half (head-to-head on a live tier workload)
//! must run on real infrastructure with real tasks.
//!
//! Run:
//!   cargo run --release --bin phase6_benchmark

use ntg_kernel::ntg::{
    storage::bit_sliced_ternary::BitSlicedTernary,
    ternary::{encode, matmul_scalar},
};
use std::time::Instant;

/// Problem sizes matching real workloads in `gemm_bench` (Phase 1 closure).
const SIZES: &[(usize, usize, &str)] = &[
    (64,   64,   "small  (64×64)  — single-node embedding"),
    (256,  256,  "medium (256×256) — multi-node layer"),
    (512,  512,  "large  (512×512) — production-layer scale"),
    (1024, 1024, "xlarge (1024×1024) — full-depth forward pass"),
];

const WARMUP_REPS: usize = 3;
const BENCH_REPS: usize = 10;

/// Memory footprint comparison: ternary (2-bit packed) vs f32 (32-bit)
fn ternary_memory_bytes(n: usize) -> usize {
    // 2 bit-streams, each ceil(n/64) u64 words
    2 * n.div_ceil(64) * 8
}

fn f32_memory_bytes(n: usize) -> usize {
    n * 4
}

/// Run scalar f32 GEMM: a_row · b_col for an (m×k) × (k×n) matmul.
/// Returns median latency in microseconds and op count.
fn bench_f32_gemm(m: usize, k: usize, n: usize) -> (u64, f64) {
    // Build deterministic f32 weights in [-1, 1]
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.1).sin()).clamp(-1.0, 1.0))
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32 * 0.17).cos()).clamp(-1.0, 1.0))
        .collect();

    // Warmup
    for _ in 0..WARMUP_REPS {
        let mut _c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += a[row * k + j] * b[j * n + col];
                }
                _c[row * n + col] = acc;
            }
        }
    }

    // Timed runs
    let mut latencies = Vec::with_capacity(BENCH_REPS);
    for _ in 0..BENCH_REPS {
        let t = Instant::now();
        let mut _c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += a[row * k + j] * b[j * n + col];
                }
                _c[row * n + col] = acc;
            }
        }
        latencies.push(t.elapsed().as_micros() as u64);
    }

    latencies.sort();
    let median_us = latencies[BENCH_REPS / 2];
    let flops = 2.0 * m as f64 * k as f64 * n as f64; // multiply + add
    (median_us.max(1), flops)
}

/// Run ternary GEMM: quantize a and b to ternary, run bit-sliced dot products.
/// Returns median latency and the number of dot products computed.
fn bench_ternary_gemm(m: usize, k: usize, n: usize) -> (u64, f64) {
    // Build deterministic f32 weights and quantize
    let a_f32: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.1).sin()).clamp(-1.0, 1.0))
        .collect();
    let b_f32: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32 * 0.17).cos()).clamp(-1.0, 1.0))
        .collect();

    let a_tern: Vec<i8> = encode(&a_f32);
    let b_tern: Vec<i8> = encode(&b_f32);

    // Pre-build BitSlicedTernary rows/cols
    let a_rows: Vec<BitSlicedTernary> = (0..m)
        .map(|row| BitSlicedTernary::from_slice(&a_tern[row * k..(row + 1) * k]))
        .collect();
    let b_cols: Vec<BitSlicedTernary> = (0..n)
        .map(|col| {
            let col_data: Vec<i8> = (0..k).map(|j| b_tern[j * n + col]).collect();
            BitSlicedTernary::from_slice(&col_data)
        })
        .collect();

    // Warmup
    for _ in 0..WARMUP_REPS {
        for a_row in &a_rows {
            for b_col in &b_cols {
                let _ = BitSlicedTernary::dot_product_auto(a_row, b_col);
            }
        }
    }

    // Timed runs
    let mut latencies = Vec::with_capacity(BENCH_REPS);
    for _ in 0..BENCH_REPS {
        let t = Instant::now();
        for a_row in &a_rows {
            for b_col in &b_cols {
                let _ = BitSlicedTernary::dot_product_auto(a_row, b_col);
            }
        }
        latencies.push(t.elapsed().as_micros() as u64);
    }

    latencies.sort();
    let median_us = latencies[BENCH_REPS / 2];
    let dot_products = (m * n) as f64;
    (median_us.max(1), dot_products)
}

/// Verify ternary correctness against scalar reference.
fn verify_correctness(k: usize) -> bool {
    let weights_f32: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.13).sin()).clamp(-1.0, 1.0))
        .collect();
    let input_f32: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.27).cos()).clamp(-1.0, 1.0))
        .collect();

    let weights_tern: Vec<i8> = encode(&weights_f32);
    let input_tern: Vec<i8> = encode(&input_f32);

    // matmul_scalar expects row-major: 1 row of k weights × k input → 1 scalar output
    let scalar_out = matmul_scalar(&weights_tern, &input_tern, 1, k, 1);
    if scalar_out.is_err() {
        return false;
    }
    let scalar_result = scalar_out.unwrap();

    // BitSliced dot product
    let a_bst = BitSlicedTernary::from_slice(&weights_tern);
    let b_bst = BitSlicedTernary::from_slice(&input_tern);
    let bst_result = BitSlicedTernary::dot_product_auto(&a_bst, &b_bst);

    // Both accumulate integer-valued ternary products; compare as i64
    let scalar_as_i64 = scalar_result.first().copied().unwrap_or(0.0) as i64;
    scalar_as_i64 == bst_result
}

fn main() {
    println!("# Phase 6: Head-to-Head Benchmark — Ternary vs. f32 GEMM");
    println!();
    println!("Hardware path: {}", detect_hw_path());
    println!("Warmup: {} reps | Bench: {} reps (median reported)", WARMUP_REPS, BENCH_REPS);
    println!();

    // Correctness gate first
    println!("## Correctness Verification");
    let correct = verify_correctness(128);
    if correct {
        println!("✅ ternary dot_product_auto matches matmul_scalar (bit-identical at k=128)");
    } else {
        println!("❌ CORRECTNESS FAIL: ternary result diverges from scalar reference");
        println!("   Cannot proceed with benchmark — fix bit-identity before reporting numbers.");
        return;
    }
    println!();

    println!("## Latency Comparison (f32 GEMM vs Ternary GEMM)");
    println!();
    println!("| Problem size | f32 latency (µs) | Ternary latency (µs) | Speedup | f32 memory | Ternary memory | Memory ratio |");
    println!("|---|---:|---:|---:|---:|---:|---:|");

    let mut results = Vec::new();

    for &(m, k, label) in SIZES {
        let n = m; // square output for simplicity

        let (f32_us, _)    = bench_f32_gemm(m, k, n);
        let (tern_us, _)   = bench_ternary_gemm(m, k, n);

        let speedup        = f32_us as f32 / tern_us as f32;
        let f32_mem        = f32_memory_bytes(m * k) + f32_memory_bytes(k * n); // A + B matrices
        let tern_mem       = ternary_memory_bytes(k) * m + ternary_memory_bytes(k) * n; // bit-sliced rows + cols
        let mem_ratio      = f32_mem as f32 / tern_mem as f32;

        println!(
            "| {label} | {f32_us} | {tern_us} | {speedup:.2}× | {} KB | {} KB | {mem_ratio:.1}× |",
            f32_mem / 1024,
            tern_mem / 1024,
        );

        results.push((label, f32_us, tern_us, speedup, mem_ratio));
    }

    println!();

    // Overall assessment
    let avg_speedup: f32 = results.iter().map(|(_, _, _, sp, _)| sp).sum::<f32>() / results.len() as f32;
    let avg_mem_ratio: f32 = results.iter().map(|(_, _, _, _, mr)| mr).sum::<f32>() / results.len() as f32;

    println!("## Summary");
    println!("  Average speedup: {avg_speedup:.2}× (ternary over f32)");
    println!("  Average memory compression: {avg_mem_ratio:.1}× (f32 / ternary)");
    println!();

    println!("## Phase 6 Decision Gate");
    if avg_speedup > 1.0 && avg_mem_ratio > 1.0 {
        println!("✅ KERNEL PASS: Ternary is faster ({avg_speedup:.2}×) AND more memory-efficient ({avg_mem_ratio:.1}×).");
        println!("   Kernel half of Phase 6 is complete.");
    } else if avg_speedup > 1.0 {
        println!("⚠️  LATENCY WIN / MEMORY REGRESSION: {avg_speedup:.2}× faster but {avg_mem_ratio:.1}× more memory.");
    } else if avg_mem_ratio > 1.0 {
        println!("⚠️  MEMORY WIN / LATENCY REGRESSION: {avg_mem_ratio:.1}× smaller but {avg_speedup:.2}× slower.");
    } else {
        println!("❌ KERNEL FAIL: Ternary is slower AND larger. Profile before proceeding.");
    }

    println!();
    println!("## What Remains for Full Phase 6");
    println!("The following require aethyro.com API integration and cannot be measured here:");
    println!("  - Production head-to-head on live tier workload (aethyro.com API)");
    println!("  - Full-pipeline end-to-end latency (FFI host overhead, Python bridge)");
    println!("  - Task accuracy comparison on real aethyro.com workloads");
    println!("  - Ship/no-ship decision requires the above real measurements");
    println!("  - Per ADR 0001: 'shipped only if the comparison is genuinely favorable'");
}

fn detect_hw_path() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq") {
            return "x86_64 + AVX-512 VPOPCNTDQ (real hardware popcount)";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "x86_64 + AVX2 (portable popcount)";
        }
    }
    "scalar (no SIMD)"
}
