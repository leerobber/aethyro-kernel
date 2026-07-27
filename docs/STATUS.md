# Aethyro NTG Engine — Project Status Report

**As of:** 2026-07-27  
**Capability version:** 10 (`ternary_capability()` — Phase 5 runtime calib supported)  
**Build:** `cargo test` + `cargo build --release` + `cargo clippy -- -D warnings` all green on host  
**Authority:** This document is the single source of truth for “where the project is.” The
`BUILD_STATUS.md` / `BREAKTHROUGH_SUMMARY.md` / `PHASE3_SUMMARY.md` stub files that used to
sit at repo root and redirect here were deleted 2026-07-26 (zero unique content, pure
redirect clutter) — this file and [ROADMAP.md](ROADMAP.md) are now the only status docs.

**2026-07-26 cleanup pass:** the repo previously carried the *entire* pre-pivot genomics
research tree despite the README claiming a "clean kernel-only extract." Audited actual
`use` dependencies (not just doc claims) and found `ntg::mutation::multi_axis` (Rung 2
fitness) genuinely depends on `genomic::sovereign_brain` — that part is real, tested,
load-bearing and was kept. Everything else in `genomic/` (disease-detection agents,
evolution/phenotype/quality-control/extended-validation pipelines, the epigenetic-engine
demo, `vitascale/`) had zero reachability from the kernel or from CI and was deleted, along
with the 10 one-off demo binaries that exclusively exercised it. Net: 9 `genomic/` modules
+ 1 subdirectory + 10 `bin/` targets removed, `genomic/` mod docs rewritten to state its
real, narrow scope. See git history for anything that needs recovering.

---

## 1. Executive verdict

| Dimension | Assessment |
|-----------|------------|
| **Maturity** | Pre-alpha **research kernel** — substantial foundations, not a product |
| **Correctness posture** | Strong: Result-based APIs, bit-identity tests, ledger tamper tests, ADR rails tested |
| **Performance posture** | Improving: real AVX-512 VPOPCNTDQ kernel now live (5.9-7.1× over portable bit-sliced, 43-52× over scalar, measured 2026-07-27) on top of the earlier bit-sliced ~12× vs i8 scalar. Real NEON kernels also now implemented (both `matmul_neon_inner` and TOBL's `tobl_dot_neon`), verified via QEMU aarch64 emulation (no real ARM CI/hardware yet). Still no end-to-end vs aethyro.com |
| **Safety posture** | Phase 3 rails implemented and tested; self-mod **off by default** |
| **Product readiness** | **Not ready** — no production-benchmark win/loss vs aethyro.com inference |
| **Primary risk** | Docs and marketing language outrunning measurements; dual storage stacks need a clear “canonical path” story |

**Bottom line:** The engine has a real, tested stack through **Phase 5**: ternary storage → graph/SIS → ledger/self-mod → calibration → **precision push + CalibModel→GraphNode production scoring**. **Doctorate schooling** (`ntg_school`, [docs/schooling/](schooling/)) runs study+exam on real data for Phases 0–5 with a **75% pass bar** and full redo on fail — multi-run notebooks under `docs/schooling/runs/`. What it does **not** have is Phase 6 integration (WASM/FFI host product path), aethyro.com head-to-head, or GPU (explicitly deferred — CPU TOBL already 12–20×).

### Phase gates (policy 2026-07-09, certificates filed)

| Question | Answer |
|----------|--------|
| Soft advance without certificates? | **Forbidden** — see [PHASE_GATE_PROTOCOL.md](PHASE_GATE_PROTOCOL.md) |
| Phases 0–5 COMPLETE certificates? | **YES** — `docs/phases/PHASE_{0,1,2,3,4,5}_COMPLETE.md` |
| May Phase 6 begin? | **YES** — Integration (host load of frozen models + production compare) |

**Process:** after each phase COMPLETE, deep-dive is in the certificate;
do not start N+1 until N is certified.

---

## 2. Evidence base (this audit)

| Check | Result |
|-------|--------|
| `cargo test` (kernel) | 313 unit + 33 integration = 346, all green (capability v10; calib model/sparse/compare tests included) |
| `cargo build --release` | Success (`libntg_kernel.{so,rlib}`, `phase4_calib`, benches) |
| `cargo clippy -- -D warnings` | Clean (exit 0) |
| CI | `.github/workflows/ci.yml`: test (debug + `--release`) + phase4 smoke + model roundtrip + density_bench + gemm_bench + edge_relatedness_bench + clippy |
| Host hardware (audit machine) | x86_64 with AVX2 + AVX-512F/VPOPCNTDQ advertised — VPOPCNTDQ now actually used by `bit_sliced_avx512` |

---

## 3. Architecture as implemented

```
┌──────────────────────────────────────────────────────────────┐
│  tools/ingest.py  — sequential GraphNode.id contract         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────┴─────────────────────────────────┐
│  Runtime (ntg/runtime.rs) + AccelManager (ntg/accel.rs)      │
│  forward_native_parallel → density-selected AccelDevice      │
│  GraphNode.weights: SparseBitSlicedTernary                   │
└────────────────────────────┬─────────────────────────────────┘
┌────────────────────────────┴─────────────────────────────────┐
│  Mutation engine (ntg/mutation/) — OFF by default            │
│  BudgetTracker + FitnessEvaluator + MutationRule             │
└────────────────────────────┬─────────────────────────────────┘
┌────────────────────────────┴─────────────────────────────────┐
│  TamperEvidentLedger (ntg/ledger/) — SHA-256 chain + sign    │
│  StateSlotStore lineage, ExecutionTrace, SignedEntry         │
└────────────────────────────┬─────────────────────────────────┘
┌────────────────────────────┴─────────────────────────────────┐
│  Graph (ntg/graph/) — structural Node + edges + adj_list     │
│  docparse / pathparse / fsevents / leafsignal / interaction  │
└────────────────────────────┬─────────────────────────────────┘
┌────────────────────────────┴─────────────────────────────────┐
│  Storage family (ntg/storage/ + packed.rs + ternary.rs)      │
│  i8 scalar · PackedTernary · BitSliced · SparseBitSliced     │
│  SIMD dispatcher + TOBL FFI (ntg/simd/, ntg/ffi/)            │
└──────────────────────────────────────────────────────────────┘
```

### Module map (kernel)

| Path | Role | Maturity |
|------|------|----------|
| `ntg/ternary.rs` | Scalar golden reference, `encode` / `encode_fixed`, `matmul_scalar` | **Solid** |
| `ntg/packed.rs` | Early 2-bit packing (4 values/byte) | Solid; overlaps storage PackedTernary |
| `ntg/storage/packed_ternary.rs` | Cache-aligned 2-bit/u64 packing + density/generation | Solid |
| `ntg/storage/bit_sliced_ternary.rs` | Dual-stream pos/neg, 64-wide popcount dot | Solid |
| `ntg/storage/sparse_bit_sliced_ternary.rs` | Flat COO sparse, tombstones, matmul/residual, ledger mutations | Solid |
| `ntg/storage/tobl_kernel.rs` | TOBL dot dispatch (scalar / AVX2 / real NEON) | Bit-identity tested on all three paths |
| `ntg/storage/bit_sliced_avx512.rs` | Real AVX-512 VPOPCNTDQ dual-stream popcount, 8 words/instruction | Tested bit-identical to portable path; 5.9-7.1× measured |
| `ntg/simd/*` | Runtime SIMD path selection + AVX2/NEON wrappers | Bit-identity OK; AVX2 matmul is correctness path, NEON verified via QEMU |
| `ntg/ffi/*` | C ABI matmul + TOBL handles + OpStats | Solid boundary tests |
| `ntg/graph/*` | Structural graph + `GraphNode` weights | Solid; adj_list added |
| `ntg/docparse`, `pathparse`, `fsevents`, `leafsignal` | SIS front-end (ADR 0003 partial) | Solid within scope |
| `ntg/interaction.rs` | Edge interaction scores | Kept with known empirical limits |
| `ntg/chain.rs` | Non-crypto hash chain (legacy / change detect) | Honest scope |
| `ntg/ledger/*` | Crypto ledger + slots + replay | Solid for Phase 3 rails |
| `ntg/mutation/*` (incl. `multi_axis.rs`, Rung 2) | Self-mod rules, budget, fitness, 4-axis sovereign fitness | Solid; disabled default |
| `ntg/runtime.rs` + `accel.rs` | Native parallel forward + device select | Solid API; HW kernels shared |
| `genomic/*` (14 files) | VCF → LD → haplotype-block → chromosome-brain → `SovereignBrain` chain feeding `ntg::mutation::multi_axis`'s `biological_consistency` axis. **Not** a general genomics pipeline — narrowly scoped to this one consumer since the 2026-07-26 cleanup removed everything else (disease-detection agents, evolution/phenotype/QC pipelines, `vitascale/`) that had zero reachability from the kernel or CI | Tested (own unit tests + `multi_axis` integration); real VCF parsing, not synthetic-only |
| `tools/ingest.py` | Layer node-ID contract | Solid |

---

## 4. Phase status (honest)

### Phase 0 — Repo / process — **DONE**
- Private proprietary LICENSE, CI, ADRs, CONTRIBUTING.

### Phase 1 — Ternary core — **MOSTLY DONE**
| Item | Status |
|------|--------|
| 1.1 Scalar reference + tests | ✅ |
| 1.2 Packed storage | ✅ (two implementations: `packed.rs` + `storage/packed_ternary.rs`) |
| Bit-sliced dense + sparse COO | ✅ |
| SIMD dispatch + bit-identity tests | ✅ |
| Native runtime + accel selection | ✅ |
| Recorded micro-bench deltas vs scalar (dots) | ✅ EXPERIMENTS.md 2026-07-09 |
| End-to-end / GEMM / production deltas | ✅ **DONE 2026-07-27** — `gemm_bench`: real multi-layer forward chain, proven bit-identical to a serial reference (not just claimed) up to 1024 nodes/layer × 3 layers |
| True AVX-512 multi-block popcnt kernels | ✅ **DONE 2026-07-27** — `bit_sliced_avx512::dot_product_avx512`, real `_mm512_popcnt_epi64`, wired into `dot_product_auto`/`bit_sliced_dot_fast` |
| NEON full path | ✅ **DONE 2026-07-27** — real `vmull_s8`/`vaddlvq_s16` in `matmul_neon_inner` and `tobl_dot_neon`; verified via QEMU aarch64 emulation (no ARM CI/hardware yet, flagged honestly) |

### Phase 2 — Graph + SIS — **STRUCTURALLY DONE; GAPS REMAIN**
| Item | Status |
|------|--------|
| Graph CRUD, topo order, forward_pass, fingerprint | ✅ |
| adj_list O(degree) children | ✅ |
| Doc / path parse, fs-event pure layer, leaf signal | ✅ |
| Self-parse dogfood tests | ✅ |
| GraphNode + sparse weights for native runtime | ✅ |
| Lazy glyph / PIXEL-lite | ❌ design only (ADR 0003) |
| Forward-pass overhead vs static baseline (measured) | ✅ **DONE 2026-07-09** — `graph_overhead_bench`: ~10.5× on an 8-node graph (sub-microsecond absolute cost either way; not a ternary TOBL cost, pure structural/topo overhead) — see EXPERIMENTS.md. This table had gone stale claiming it was still open; corrected 2026-07-27. |
| Edge interaction as structural relatedness | ⚠️ **partially explored 2026-07-27** — fixed formula still empirically weak (2026-07-08); a trained ternary perceptron on the same real-doc corpus beats chance by +0.073 balanced accuracy and beats the fixed formula, but only reaches 0.573 test balanced accuracy (0.635 train) — a real, modest signal that learning helps, not a working relatedness detector yet. See EXPERIMENTS.md. |

### Phase 3 — Self-mod + ledger — **DONE for ADR 0002 rails**
| Rail | Status |
|------|--------|
| 1 Off by default | ✅ tested |
| 2 Bounded budget | ✅ tested |
| 3 Auto rollback on regression | ✅ tested |
| 4 Deterministic replay | ✅ tested |
| 5 Every mutation ledger-logged | ✅ tested |
| SHA-256 chain + signed entries | ✅ |
| StateSlot lineage (1-based parent pointers) | ✅ |

**Not claimed:** production mmap ChronosLedger file format parity, multi-agent orchestration, live Reflexive Fitness critics driving topology at scale.

### Phase 4 — **COMPLETE** (2026-07-09)
- Certificate: `docs/phases/PHASE_4_COMPLETE.md`
- Real docs WIN: bal_acc≈0.61, rec≈0.25, fp≈12; self-mod off by default
- Optional `--self-mod` probe ledgered, fitness may reject

### Phase 5 — **COMPLETE** (2026-07-09)
- Certificate: `docs/phases/PHASE_5_COMPLETE.md`
- Precision calib WIN: bal≈0.70, F1≈0.28 (up from Phase 4's bal≈0.61, F1≈0.18)
- CalibModel → GraphNode production path, CPU parallel batch scoring
- GPU explicitly re-scoped to Phase 6+ (64-d tensors don't justify PCIe transfer cost yet)

### Phase 6–8 — **NOT STARTED**
WASM target, aethyro.com head-to-head, and the ship/no-ship product
decision remain later — see [ROADMAP.md](ROADMAP.md) Phase 6.

---

## 5. Testing proof matrix

| Suite | Count | What it proves |
|-------|------:|----------------|
| Unit (`--lib`) | 313 | Core algorithms, ledger, mutation (incl. Rung 2 multi-axis), storage (incl. AVX-512 vs. portable bit-identity), runtime, graph, accel, `genomic/` sovereign-brain chain, edge-relatedness perceptron |
| `phase1_2_3_simd_ffi` | 11 | SIMD parity, FFI, OpStats |
| `phase1_2_3_storage_integration` | 10 | PackedTernary + TOBL + ledger glue |
| `phase3_integration` | 7 | All five ADR 0002 rails end-to-end |
| `self_parse` | 3 | Real repo docs parse without panic |
| `sovereign_integration` | 2 | LTM motif activation, train/prune acceptance bias |
| **Total** | **346** | |

### Known test honesty notes
- Sparse `ternary_matmul` is **chunk-level score → ±1 gate**, not full dense GEMM.
- AVX2 dense matmul currently prioritizes **scalar-correct bit-identity** over peak SIMD throughput.
- AccelDevice variants share the same sparse kernel today; selection is policy + observability, not separate HW implementations.
- `edge_interaction_score` tests prove self-similarity / edit sensitivity, **not** structural edge discovery on real docs.

---

## 6. Documentation inventory (organized)

| Document | Role after this audit |
|----------|------------------------|
| **[STATUS.md](STATUS.md)** (this file) | **Current truth** — read first |
| [ROADMAP.md](ROADMAP.md) | Phased checklist (kept in sync with STATUS) |
| [DESIGN.md](DESIGN.md) | Target architecture (updated layer diagram) |
| [LITERATURE.md](LITERATURE.md) | Novelty claims + sources |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Measured wins / non-wins |
| [architecture/](architecture/) | ADRs 0001–0007 |
| [PHASE1_2_3_*.md](PHASE1_2_3_IMPLEMENTATION.md) | Historical implementation notes (may lag STATUS) |
| `kernel/FFI_INTEGRATION.md`, `TOBL_FFI_REFERENCE.md` | FFI operator docs |
| `CONTRIBUTING.md` | Engineering rules |

The root `BUILD_STATUS.md` / `BREAKTHROUGH_SUMMARY.md` / `PHASE3_SUMMARY.md` redirect
stubs were deleted 2026-07-26 — they carried no content beyond "see STATUS.md."

---

## 7. Critical gaps & recommended next work (agency priority)

### Resolved since the last audit
1. ~~Micro-bench suite~~ **DONE 2026-07-09** — `cargo run --release --bin density_bench`
2. ~~Log numbers in EXPERIMENTS.md~~ **DONE** — see EXPERIMENTS.md "density micro-bench"
   - Bit-sliced ~12× vs scalar i8 at N=262144
   - Sparse best at 1% (~20×); at 10–50% random fill loses to bit-sliced
3. ~~Canonical storage ADR~~ **DONE** — [ADR 0005](architecture/0005-canonical-ternary-storage.md)
   names the legacy `ntg::packed::PackedTernary` vs. canonical
   `ntg::storage::*` roles explicitly; kept both, not unified, by
   deliberate decision (legacy stays for backward tests/demos).
4. ~~CI clippy debt~~ **DONE 2026-07-26** — `cargo clippy -- -D warnings` clean.
5. ~~Repo scope matched its README claim~~ **DONE 2026-07-26** — pre-pivot
   genomics bloat (9 modules, `vitascale/`, 10 demo binaries) removed;
   `genomic/` is now honestly scoped to the Rung 2 sovereign-brain chain.
6. ~~Real AVX-512 VPOPCNTDQ kernels~~ **DONE 2026-07-27** —
   `ntg::storage::bit_sliced_avx512::dot_product_avx512` (8 words/512
   elements per `_mm512_popcnt_epi64`), wired into `runtime::bit_sliced_dot_fast`
   via `BitSlicedTernary::dot_product_auto` so detected hardware is actually
   used, not just reported. 5 new tests prove bit-identity against the
   portable reference across the SIMD-boundary sizes. Measured: 5.9-7.1×
   over the already-fast portable bit-sliced path, 43-52× over scalar — see
   EXPERIMENTS.md "Real AVX-512 VPOPCNTDQ kernel."
7. ~~Real NEON kernels~~ **DONE 2026-07-27** — both `ntg::simd::neon::matmul_neon_inner`
   (`vmull_s8` + `vaddlvq_s16`) and `ntg::storage::tobl_kernel::tobl_dot_neon`
   (same unpack-then-multiply approach as the AVX2 path, at NEON's native
   8-lane width). No ARM CI runner exists, so verified by cross-compiling
   to `aarch64-unknown-linux-gnu` and running the full test suite under
   `qemu-aarch64-static` user-mode emulation — genuine instruction-level
   execution, not real silicon, flagged honestly rather than claimed as
   hardware-tested. Also found and fixed a pre-existing latent bug in
   `neon_matches_scalar` (wrong `NtgError` scope) that had silently never
   compiled before, since nothing had ever actually built this crate's
   aarch64-gated code until now.
8. ~~End-to-end/GEMM-scale benchmark~~ **DONE 2026-07-27** — `cargo run
   --release --bin gemm_bench`: real multi-node, multi-layer sparse
   ternary forward chains (up to 1024 nodes/layer × 3 layers), timed
   end-to-end through `Runtime::forward_native_parallel` and checked
   bit-for-bit at every layer against a single-threaded serial reference
   using the same primitive. Also surfaced and fixed a real
   `cargo test --release`-only failure (`test_observability_metrics`
   asserted wall-clock cycles are always nonzero; false on fast hardware)
   that plain debug-mode `cargo test` — what CI runs — never caught. See
   EXPERIMENTS.md "First multi-layer forward benchmark."
9. ~~Whether learned weights beat the fixed edge-interaction formula~~
   **ANSWERED 2026-07-27, partial win** — `cargo run --release --bin
   edge_relatedness_bench` / `ntg::edge_calib`: a trained ternary
   perceptron on real-edge-vs-random-pair labels beats chance by +0.073
   test balanced accuracy and beats the fixed formula, but only reaches
   0.573 (0.635 train) — real signal, not yet a working relatedness
   detector. See EXPERIMENTS.md "Does learning the edge-relatedness
   weights beat the fixed formula?" The Phase 2 row above is marked
   partial, not done, on purpose.

### P1 — Engineering hardening (still open)
1. Wire OpStats + device name into ledger entries on forward.
2. A wall-clock-budget test (`calib::tests::self_mod_probe_enabled_logs_ledger`,
   5ms budget) was observed to fail intermittently under QEMU aarch64
   emulation with the full suite running in parallel (never on native
   x86_64, and not reproducible on repeat aarch64 runs either) — noted,
   not fixed, since it never touches the real CI environment (native
   x86_64) and isn't reliably reproducible to debug further right now.

### P2 — Phase 6 entry (Phases 0–5 all certified; see §1)
1. Freeze a model artifact (`phase4_calib --write-model`), load it in an
   external host / WASM or FFI consumer.
2. Real workload head-to-head vs aethyro.com inference (memory + latency).
3. Report outcome honestly regardless of win/loss (same discipline as
   every prior phase certificate).

### Explicit non-goals (now)
- Product marketing claims of "40% cycle reduction" until measured.
- Legal/Healthcare vertical code.
- Claiming ChronosLedger full file format compatibility.
- Re-litigating the genomics-scope cleanup without a concrete new
  consumer — if something else needs `agents.rs`/`domain_agents.rs`/etc.
  back, pull it from git history with a stated reason, don't silently
  restore unreachable code "just in case."

---

## 8. Capability snapshot (v10, from `ternary_capability()` in `lib.rs`)

```
scalar_supported: true
packed_supported: true
simd_supported: true          // dispatcher present; not "all paths peak HW"
graph_supported: true
doc_path_parsing_supported: true
forward_pass_supported: true  // structural LeafSignal aggregate
fingerprint_supported: true
edge_interaction_score_supported: true  // narrow properties only
chain_log_supported: true
bit_sliced_supported: true
sparse_bit_sliced_supported: true
native_parallel_forward_supported: true
phase4_calibration_supported: true
phase5_runtime_calib_supported: true
version: 10
```

---

## 9. Sign-off

| Role lens | Conclusion |
|-----------|------------|
| **Research agency** | Credible experimental platform with unusually honest non-win documentation (edge scoring). |
| **Systems eng** | Kernel modular, test-heavy, replay/safety conscious; perf path incomplete. |
| **Product** | Do not ship efficiency claims yet. |
| **Security / sovereignty** | Ledger + off-by-default self-mod is the right shape; needs threat model review before air-gap product claims. |

**Status classification:** `RESEARCH-READY / PRE-PRODUCT`  
**Merge readiness of local tree:** code green locally; docs consolidated here; commit & CI push still required.
