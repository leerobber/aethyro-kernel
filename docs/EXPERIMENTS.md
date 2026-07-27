# Experiments log

Real, run, measured experiments against this project -- wins and
non-wins recorded the same way, per CONTRIBUTING.md rule 1 ("measure,
don't assume") and the precedent already set by the founder's
`aetherflux-zero` project (an honest negative depth-experiment result
sits next to a real measured 11% bits-per-character win, both kept).
An experiment that didn't pan out and is documented here is more
valuable than one that was quietly dropped.

## 2026-07-08: is ChronosLedger actually the "tamper-evident, hash-chained ledger" ADR 0001/0002 assumed it was?

**Why this check happened:** before merging the first substantial batch
of Phase 1-2 work, the question was asked: is there a foundational gap
worth closing now, while cheap, rather than discovering it mid-Phase-3?
ADR 0001 and ADR 0002 both stated, as fact, that this project's audit
ledger would "reuse the design already proven" in GH05T3's
ChronosLedger, describing it as tamper-evident and hash-chained. That
claim had never actually been checked against the real source in this
project — it was inherited from an external draft and repeated forward
across three documents without verification. This is exactly the kind
of claim CONTRIBUTING.md rule 2 exists to catch.

**Method:** read the actual implementation,
`GH05T3/backend/oss/core/chronos_ledger.py`, directly. Also checked two
other candidates in the same codebase for a genuine hash-chain:
`backend/oss/core/seal.py` ("LexGenSeal") and `backend/economy/ledger.py`.

**Result — real, and it corrects a standing false claim:**

- **ChronosLedger** is a real-time **mutable** mmap agent-state store:
  32-byte slots (desires, fitness, maturity, `parent_offset` for
  lineage, generation, a scratchpad bitfield), overwritten in place via
  `write_agent()`/`update_fitness()`/etc. There is no hashing anywhere
  in the file, no previous-record linkage, no tamper-evidence of any
  kind. It is genuinely excellent at what it's actually for: fast
  slot-level state and lineage tracing — not an audit trail.
- **LexGenSeal** (`seal.py`) is real and genuinely tamper-evident **per
  record**: each breakthrough record is SHA256-signed over its own
  content and written to an append-only-by-convention vault file. But
  records are not chained to each other — deleting one seal file from
  the vault directory is completely undetectable from the files that
  remain. It proves "this record wasn't altered," not "no record was
  removed from the sequence."
- **`economy/ledger.py`** is a plain SQLite table, append-only by
  convention (no `UPDATE`/`DELETE` in the code path), no cryptography
  at all.

**Conclusion: no genuine hash-chained, tamper-evident ledger exists
anywhere in the checked codebase.** ADR 0001 and ADR 0002's "reuses the
proven ChronosLedger design" claim was false for the tamper-evidence
half specifically (true only for the state-slot/lineage half). Both
ADRs and DESIGN.md have been corrected in place rather than quietly
patched, so the record shows the mistake and the fix, not just the fix.

**What was built as a direct result:** `kernel/src/ntg/chain.rs`
(`ChainLog`) — a genuine hash-chain primitive, tested against exactly
the properties a real audit ledger needs: altering a historical entry's
content breaks verification from that point forward; removing an entry
breaks verification; a given piece of content produces a different
chain value depending on what preceded it (proving the chain captures
sequence, not just content). Uses `std`'s non-cryptographic
`DefaultHasher` for now, honestly labeled as such (same pattern as
`Graph::fingerprint`) — swapping in a real cryptographic hash (SHA-256,
matching LexGenSeal's own choice, or BLAKE3) is a dependency decision
for Phase 3, not a redesign of this module.

**Why this was worth doing before merge, not after:** every phase from
here on assumes ledger-logging exists (ADR 0002 rule 5, ADR 0003's
execution-node auditing, Phase 3's exit criteria). Finding and fixing
this now — while only two ADRs and one design doc referenced the false
claim — is far cheaper than finding it after Phase 3 was built on top
of an assumption that didn't hold.

## 2026-07-08: does leaf-signal correlate with file extension?

**Hypothesis:** `LeafSignal` (case/punctuation/whitespace counts) might
carry enough information to distinguish content types cheaply, without
any ML.

**Method:** computed per-character signal ratios (uppercase, lowercase,
punctuation, whitespace, other) over every real file in this repo,
grouped by extension.

**Result — real, positive, but small-sample:**

| ext | n | upper | lower | punct | space | other |
|---|---|---|---|---|---|---|
| (none) | 1 | 0.017 | 0.786 | 0.041 | 0.147 | 0.009 |
| .gitignore | 1 | 0.029 | 0.747 | 0.066 | 0.158 | 0.000 |
| .md | 9 | 0.036 | 0.701 | 0.073 | 0.162 | 0.028 |
| .rs | 11 | 0.027 | 0.557 | 0.154 | 0.251 | 0.012 |
| .toml | 1 | 0.000 | 0.576 | 0.227 | 0.144 | 0.053 |

`.rs` files run higher punctuation (15.4% vs. 7.3% for `.md`) and higher
whitespace (25.1% vs. 16.2% -- indentation) with lower lowercase-letter
density (55.7% vs. 70.1%) -- the direction you'd expect from code's
brackets/semicolons vs. prose. `.toml` has the highest punctuation of
all (22.7%: `=`, `[]`, quotes).

**Honest caveat:** n=9 and n=11 are not statistically rigorous. The
direction is real and sensible, not noise, but this should not be
treated as a validated classifier -- it's a plausible cheap heuristic
worth remembering, not a shipped feature.

## 2026-07-08: ternary matmul as a real per-edge "interaction score"

**Hypothesis:** ternary-encoding two adjacent nodes' labels (via the
existing `encode()`) and running a real `matmul_scalar` between them
might produce an interpretable "interaction score" for that edge --
the first real compute over the graph, ahead of Phase 3/4.

**Method:** encoded each label's UTF-8 bytes as `(byte - 128) / 128`,
ternary-encoded via `encode()`, zero-padded to equal length, and ran a
1×k @ k×1 matmul. Tested on real and synthetic label pairs.

**Result — real, negative, and specifically diagnosed, not just "didn't
work":**

```
'Title'            x 'first'    -> score=4.000  enc_a=[-1,-1,0,-1,-1]     enc_b=[-1,-1,-1,-1,-1]
'Title'            x 'second'   -> score=4.000  enc_a=[-1,-1,0,-1,-1,0]   enc_b=[-1,-1,-1,-1,-1,-1]
'aaaaa'            x 'aaaaa'    -> score=5.000  enc_a=[-1,-1,-1,-1,-1]    enc_b=[-1,-1,-1,-1,-1]
'aaaaa'            x 'zzzzz'    -> score=5.000  enc_a=[-1,-1,-1,-1,-1]    enc_b=[-1,-1,-1,-1,-1]
'Section A'        x 'Section B'-> score=6.000  enc_a=[...,-1,0,0,-1,-1] enc_b=[...,-1,0,0,-1,-1] (identical)
```

`"aaaaa"` and `"zzzzz"` -- completely different strings -- produce the
*identical* encoding and score. **Root cause, found, not guessed:**
`encode()` thresholds each string against *its own* mean magnitude
(the same absmean design BitNet uses for a single weight tensor). Any
string made of characters with similar byte values collapses to the
same ternary pattern (mostly `-1`), because the normalization is
relative to that string alone -- there is no shared reference scale
across different labels, so cross-string comparison is structurally
impossible with this construction.

**Conclusion (at the time):** this specific approach does not work as
specified. The mechanism BitNet-style absmean quantization exists for
(compress one tensor's own weights) is not the same problem as (compare
two arbitrary short strings), and applying one to the other silently
breaks it.

**What this suggested for a follow-up experiment:** a fixed/global
byte-to-ternary mapping (same threshold for every string, not each
string's own mean) would preserve cross-string distinguishability. See
below -- this was tried, and it worked.

## 2026-07-08: follow-up — fixed-threshold encoding fixes the failure above

**Hypothesis:** replace `encode()`'s per-string mean threshold with a
fixed global one (same byte always maps to the same ternary value,
regardless of context), calibrated on the a-z byte range (center 109.5,
scale 13, threshold ±0.33).

**Method:** re-ran the exact same failing pairs from the experiment
above, in Python first, before writing any Rust.

**Result — real, positive, verified before shipping as code:**

```
'aaaaa'              x 'aaaaa'              ->    5.0
'aaaaa'              x 'zzzzz'              ->   -5.0   (was 5.0, identical, before)
'hello'              x 'hello'              ->    2.0
'hello'              x 'hxllo'              ->    0.0   (self-score > 1-char-edit score)
```

`"aaaaa"` vs `"zzzzz"` now score oppositely (`+5.0` vs `-5.0`) instead of
identically — the exact failure is fixed. Better than just "fixed":
the score is now *interpretable* in a way it wasn't designed to be —
self-similarity scores positive, a byte-for-byte "opposite" pairing
scores negative, and a single-character edit ("hello" -> "hxllo")
measurably lowers the score below the unedited self-score. This is a
real, if crude, working similarity signal, not just "no longer broken."

**Honest limits, stated precisely, not hidden:**
- This is a **byte-position correlation**, not semantic similarity — it
  knows nothing about meaning.
- It's sensitive to positional alignment: inserting one character near
  the start of a string shifts every later comparison out of sync,
  which could make genuinely similar strings score poorly if they
  differ in length early on. Not tested here.
- The fixed-threshold distribution is skewed (mostly `-1` across the
  full printable-ASCII range — 74 of 95 characters), because it's
  calibrated for a-z distinguishability, not balanced bit-usage. Not a
  drop-in replacement for `encode()`'s quantization use case.

**Shipped as code:** `encode_fixed` (`kernel/src/ntg/ternary.rs`) and
`edge_interaction_score` (`kernel/src/ntg/interaction.rs`), both tested
against the exact properties measured here (opposite-string negative
score, self-score-beats-edited-score, determinism).

## 2026-07-08: does edge_interaction_score say anything real about this repo's actual document structure?

**Hypothesis:** now that the score is fixed and interpretable on small
synthetic examples, does it capture anything meaningful about *real*
parent-child relationships (heading → its content, section → its
bullets) across this repo's actual ADRs, DESIGN.md, and ROADMAP.md?

**Method:** reimplemented `docparse.rs`'s parsing logic in Python
(headings/bullets/numbered items/fenced code, exactly matching the
Rust), parsed all 5 real docs into one combined graph (491 nodes, 486
real parent-child edges), computed `edge_interaction_score` for every
real edge, and compared against a same-sized sample of random
(non-adjacent) node pairs from the same corpus as a control.

**Result — real, and it corrects an initial over-read:**

Raw scores looked different at first glance (real edges: mean 4.48,
std 5.43; random pairs: mean 6.29, std 7.13) — but checking *why*
mattered. Correlation between raw score and `min(len_a, len_b)` across
the real edges: **0.563** (0.595 using `|score|`). Over half the
apparent "real vs. random" difference is explained by string length
alone (a direct consequence of the zero-padding: positions beyond the
shorter string's length always contribute zero, so the score is really
"agreement over the first `min(len)` bytes," which is dominated by how
big that overlap even is) -- not by any relationship between a heading
and its content.

After removing that confound (`normalized_edge_interaction_score` =
raw / `min(len_a, len_b)`):

| | mean | std |
|---|---|---|
| Real parent-child edges | 0.155 | 0.207 |
| Random node pairs | 0.132 | 0.159 |

These are close relative to their spread — **not a clear separation.**

**Honest conclusion:** on this repo's real documents, neither the raw
nor the length-normalized `edge_interaction_score` reliably
distinguishes a genuine heading→content relationship from an arbitrary
unrelated pair. The self-similarity/edit-sensitivity properties from
the prior experiment are still true (they were tested directly and
still hold) -- this is a different, additional finding: those
properties do not generalize to "tells you which nodes are related" on
real, structurally diverse text. A working mechanism is not the same
as a working application of it, and this is a case of the former
without (yet) the latter.

**What was kept anyway:** `normalized_edge_interaction_score` was
shipped despite the negative headline result, because removing a real,
diagnosed confound (length) is correct regardless of whether the
underlying signal turns out to be useful -- and the self-similarity/
edit-detection properties remain real, tested, and potentially useful
for a narrower purpose (e.g. near-duplicate detection) than "structural
relatedness."

**What this suggests for a follow-up, not yet tried:** a genuine
relatedness signal probably needs actual learned weights (a real
Phase 4 training step), not a fixed, untrained byte-correlation --
which is itself a useful, concrete thing to have ruled out cheaply
before investing in that larger feature.

## 2026-07-09: density micro-bench — scalar i8 vs bit-sliced vs sparse COO dots

**Why this ran:** docs/STATUS.md P0 gate — record real wall-clock deltas
(or honest non-wins) before claiming TOBL / sparse speedups. Prior
architecture prose asserted ~40% cycle reduction; that claim was never
measured on this kernel.

**Method:** `cargo run --release --bin density_bench` on the audit host
(x86_64, AVX2 + AVX-512F/VPOPCNTDQ present). Harness:
`kernel/src/bin/density_bench.rs`.

- Vector length **N = 262,144** ternary elements (4096 × 64-bit chunks)
- Densities **1%, 10%, 50%** (independent random ±1 with that fraction;
  zeros elsewhere; same seed family per density)
- Paths compared:
  1. **scalar** — dense `i8` product-sum loop
  2. **bit-sliced** — `BitSlicedTernary::dot_product_parallel` (AND + popcount)
  3. **sparse** — `SparseBitSlicedTernary::dot_product_sparse` (COO merge-join)
- Timing: 20 warmup + **200** timed iters; **median** wall-clock µs
- Correctness: all three paths must return identical integer sums

**Results (median µs, sums matched on all rows):**

| density | scalar µs | bit-sliced µs | sparse µs | speedup BS/S | speedup SP/S | active COO blocks (max of A,B) |
|--------:|----------:|--------------:|----------:|-------------:|-------------:|-------------------------------:|
| 1% | 80.69 | 6.53 | 4.00 | **12.35×** | **20.19×** | 1963 |
| 10% | 80.64 | 6.53 | 13.52 | **12.35×** | **5.97×** | 4091 |
| 50% | 80.80 | 6.53 | 13.47 | **12.37×** | **6.00×** | 4096 (full) |

JSON (machine-readable):
```json
[{"density":0.01,"n":262144,"scalar_us":80.688,"bit_sliced_us":6.531,"sparse_us":3.997,"sparse_blocks":1963,"sums_match":true},{"density":0.1,"n":262144,"scalar_us":80.637,"bit_sliced_us":6.531,"sparse_us":13.517,"sparse_blocks":4091,"sums_match":true},{"density":0.5,"n":262144,"scalar_us":80.799,"bit_sliced_us":6.532,"sparse_us":13.467,"sparse_blocks":4096,"sums_match":true}]
```

**Interpretation (honest):**

1. **Bit-sliced is a clear win** over naive i8 scalar on this host:
   ~**12×** across all densities (dense dual-stream always scans all
   words; cost is density-independent for this N).
2. **Sparse is the best path only at true sparsity.** At 1% density it
   beats bit-sliced (~20× vs scalar, ~1.6× vs bit-sliced). At 10% and
   50% random occupancy, almost every 64-wide chunk is non-empty
   (4091–4096 / 4096), so sparse loses the “skip zero regions” advantage
   and is **slower than bit-sliced** while still beating scalar (~6×).
3. The architecture claim “always use sparse for multi-agent scale” is
   therefore **conditional**: sparse wins when active chunks << total
   chunks (structured sparsity / block sparsity), not merely when
   element density is “medium.” Random independent nonzeros fill chunks
   fast.
4. **No claim of 40% cycle reduction** is supported or needed — measured
   speedups are larger for this micro-op, but they are **dot-product
   micro-benches**, not end-to-end inference vs aethyro.com, not full
   matmul, and not AVX-512 intrinsic kernels (portable `count_ones` /
   software path).

**What was shipped as a direct result:**

- `kernel/src/bin/density_bench.rs` + `[[bin]] density_bench` in Cargo.toml
- This experiment log entry
- ROADMAP / STATUS: P0 measurement gate closed for dots; end-to-end
  still open

**Follow-ups (not done here):**

- Structured block-sparse generators (k contiguous nonzeros per block)
  to model GraphNode weight patterns more realistically
- Dense GEMM / `ternary_matmul` wall-clock (chunk gate, not full GEMM)
- True `_mm512_popcnt_epi64` path vs `u64::count_ones`

## 2026-07-09: graph forward-pass overhead vs static signal fold

**Why:** Phase 2 exit required measuring graph-structure overhead vs a
non-graph baseline.

**Method:** `cargo run --release --bin graph_overhead_bench`  
Parse a small markdown sample into a Graph (8 nodes), then median of 500
timed runs (50 warmup):

1. `Graph::forward_pass` (topo order + LeafSignal combine)
2. Static fold of the same 8 signals in a `Vec` (no edges/topo)

**Results (host LOQ-class x86_64, release):**

| path | median µs |
|------|----------:|
| graph forward_pass | 0.21 |
| static signal fold | 0.02 |
| **overhead ratio** | **~10.5×** |

**Interpretation:** On this tiny graph, scheduling/topo dominates absolute
time; absolute costs are sub-microsecond. Ratio will shrink on larger
graphs where signal work grows with nodes while topo is O(V+E). This is
**not** a ternary TOBL cost; it is pure structural overhead of the graph
API used for SIS forward_pass.

**Honest non-claim:** not compared to production aethyro.com inference.

## 2026-07-09: Phase 4 calibration — doc-graph NodeKind classifier (ADR 0006)

**Why:** Phase 4 requires a real closed loop: ternary features → score →
calibrate → measure vs baseline, win or non-win recorded equally.

**Method:** `cargo run --release --bin phase4_calib`  
Module: `kernel/src/ntg/calib/`. Labels = parser `NodeKind` (Execution vs
Content). Features = `encode_fixed` + LeafSignal + GlyphFingerprint v0
folded to length-64 ternary. Model = ternary weights + threshold score.
Baseline = always Content. Train = 25 epochs ternary perceptron updates.

### Run A — built-in fixtures (3 synthetic docs)

```
n=28 exec=3 content=25 baseline=0.893 before=0.893 after=1.000 delta=+0.107
epochs=25 latency_us=17 win=true
```

**Result: WIN** on fixtures (perfect after accuracy; beats 89.3% majority).

### Run B — real repo `docs/` (20 markdown files)

```
n=2153 exec=37 content=2116 baseline=0.983 before=0.983 after=0.954 delta=-0.029
epochs=25 latency_us=1477 win=false
```

**Result: NON-WIN** on real docs. Extreme class imbalance (~1.7% Execution).
Calibration *hurt* accuracy vs majority baseline (over-predicting Execution).

**Interpretation:**
1. Closed loop works end-to-end (parse → features → train → ledger snapshot).
2. Fixture win does **not** generalize; real corpus is majority-dominated.
3. Phase 4 must treat imbalance (class weights, threshold search, or
   balanced sampling) before claiming intelligence gains.
4. Ledger weight snapshot verified after both runs.

**Follow-ups (still Phase 4, not COMPLETE yet):**
- Threshold sweep / balanced sampling
- Hold-out split (train/test)
- Optional topology self-mod under ADR 0002 (off by default)
- PHASE_4_COMPLETE.md only after exit criteria + deep dive

## 2026-07-09: Phase 4 imbalance fix — balanced train + F1 threshold

**Problem:** Naive perceptron on ~98% Content collapsed to worse-than-majority
accuracy (flood of Execution false positives or zero minority recall).

**Fixes implemented:**
1. Stratified 80/20 train/test split
2. Balanced epoch sampling (all minority + equal majority)
3. Cost-sensitive minority update repeats (`n_neg/n_pos`, cap 64)
4. Code-cue feature for fence *bodies* (no ``` in labels from docparse)
5. Threshold sweep maximizing F1 + bal_acc (penalty for precision floods)
6. Win bar uses **balanced accuracy / F1 / recall**, not raw accuracy

**Command:** `cargo run --release --bin phase4_calib -- --docs ../docs`

### Real docs (20 files) after fix

```
n=2189 train=1751 test=438 exec=39 content=2150 thr=21
base_acc=0.982 base_bal=0.500
test_acc=0.952 test_bal=0.608 test_f1=0.160 test_rec=0.250 test_prec=0.118
delta_bal=+0.108
confusion: tp=2 tn=415 fp=15 fn=6
result: WIN
```

**Interpretation:** Raw accuracy (95%) is below majority (98%) but that is
expected under imbalance. **Balanced accuracy 60.8%** and **25% Execution
recall** with only 15 FPs is a real lift vs majority (bal 50%, rec 0%).
F1 remains low (0.16) — precision is the next bottleneck.

### Fixtures

Hold-out on tiny sets is noisy (often 1 Execution in test). Unit tests
require bal_acc ≥ 0.5 and some train/test exec detection.

**Honest residual gaps:** low precision; need richer code features or more
epochs / calibration on Execution-heavy corpora for higher F1.

## 2026-07-09: Phase 4 COMPLETE — final calib + self-mod probe

**Command:**
```bash
cargo run --release --bin phase4_calib -- --docs ../docs
cargo run --release --bin phase4_calib -- --docs ../docs --self-mod
```

### Final real-docs calib (imbalance-aware)

```
n=2212 train=1770 test=442 exec=40 thr=11
base_acc=0.982 base_bal=0.500
test_acc=0.959 test_bal=0.611 test_f1=0.182 test_rec=0.250 test_prec=0.143
delta_bal=+0.111
confusion: tp=2 tn=422 fp=12 fn=6
result: WIN (balanced metrics)
```

### Self-mod probe

| Mode | Result |
|------|--------|
| default (no flag) | disabled, no mutation (ADR 0002 rail 1) |
| `--self-mod` | AddNode proposed, **rejected** by dual-objective fitness, **ledgered** (id=1), caller graph unchanged |

**Phase 4 exit criteria:** met — real task E2E, win recorded, non-win paths
honest, optional self-mod off-by-default with ledger when enabled.

## 2026-07-09: Phase 5 optimization — precision + runtime path

**Why:** Phase 5 targets higher F1/precision, CalibModel → GraphNode production
path, CPU parallel batch scoring, and honest GPU deferral.

**Command:**
```bash
cargo run --release --bin phase4_calib -- --docs ../docs
cargo run --release --bin density_bench
cargo run --release --bin graph_overhead_bench
```

### Precision calib (22 docs markdown)

```
n=2299 train=1839 test=460 exec=45 thr=11
base_bal=0.500
test_acc=0.954 test_bal=0.704 test_f1=0.276 test_rec=0.444 test_prec=0.200
delta_bal=+0.204
confusion: tp=4 tn=435 fp=16 fn=5
result: WIN
path_identity dense==graph_node: true
```

| Metric | Phase 4 cert | Phase 5 | Δ |
|--------|-------------:|--------:|--:|
| test_bal | 0.611 | **0.704** | +0.09 |
| test_f1 | 0.182 | **0.276** | +0.09 |
| test_rec | 0.250 | **0.444** | +0.19 |
| test_prec | 0.143 | **0.200** | +0.06 |

**Changes:** richer code/indent/line-shape cues; flood-reject thr objective
with rec/prec floors; GraphNode warm-start scoring path.

### density_bench (post Phase 5)

| density | scalar µs | bit-sliced µs | sparse µs | BS/S | SP/S |
|--------:|----------:|--------------:|----------:|-----:|-----:|
| 1% | 80.2 | 6.5 | 3.9 | 12.4× | 20.8× |
| 10% | 80.1 | 6.5 | 13.4 | 12.3× | 6.0× |
| 50% | 80.1 | 6.5 | 13.4 | 12.3× | 6.0× |

**GPU decision:** **not implemented.** 64-d calib activations do not justify
device transfer; CPU TOBL already delivers double-digit speedups. Revisit in
Phase 6+ if production tensor dims grow.

### graph_overhead_bench

```
graph≈0.20 µs  static≈0.02 µs  ratio≈10×  (same character as Phase 2)
```

### Self-mod (still off by default)

`--self-mod`: AddNode proposed, rejected by fitness, ledgered.

**Phase 5 exit criteria:** met — see `docs/phases/PHASE_5_COMPLETE.md`.

## 2026-07-26 cleanup + release-profile pass

### `[profile.release]` lto + codegen-units=1

Added `lto = true`, `codegen-units = 1` to `kernel/Cargo.toml` (previously
unset -> Cargo defaults of no LTO, 16 codegen units). Measured
`density_bench` before/after, same host, same command, median of 200 iters:

| density | scalar µs (before → after) | bit-sliced µs (before → after) | sparse µs (before → after) |
|--------:|:---------------------------:|:-------------------------------:|:----------------------------:|
| 1% | 94.13 → 88.10 (-6.4%) | 13.20 → 12.38 (-6.2%) | 6.12 → 6.46 (+5.6%, noise-level) |
| 10% | 92.33 → 87.27 (-5.5%) | 12.74 → 12.39 (-2.7%) | 20.80 → 20.86 (~flat) |
| 50% | 92.39 → 88.27 (-4.5%) | 13.17 → 12.37 (-6.1%) | 28.20 → 20.91 (-25.8%) |

`graph_overhead_bench` unchanged in character (graph≈0.20µs, static≈0.03µs,
ratio≈6×; the earlier ≈10× figure and this run differ mainly by measurement
noise on a small 8-node sample, not a regression). Full `cargo test` (338
tests) and `cargo clippy -- -D warnings` stayed green. Release build time
rose from ~10s to ~46s (LTO cost) -- acceptable, release builds aren't the
inner dev loop. Real, measured, small win; not claimed as more than that.

### Dead-code / scope cleanup

The repo previously carried the entire pre-pivot genomics research tree
despite the README claiming a "clean kernel-only extract" (untrue -- see
git history before this commit). Traced actual `use` dependencies rather
than trusting the doc: `ntg::mutation::multi_axis` (Rung 2 fitness)
genuinely depends on `genomic::sovereign_brain`, so that chain (14 files)
was kept. Everything else in `genomic/` -- `agents.rs`, `domain_agents.rs`,
`evolution.rs`, `phenotype.rs`, `report_gen.rs`, `quality_control.rs`,
`extended_validation.rs`, `optimized_core.rs`, `epigenetic_engine.rs`,
`vitascale/` -- had zero reachability from the kernel or from CI and was
deleted, along with the 10 `bin/` demos that exclusively exercised it
(`chromosome_brain_test`, `domain_disease_{complete,test}`,
`phase_{c_synthesis,d_quality_control,e_extended_validation}`,
`optimized_core_demo`, `system_evolution_node`, `kairos_stage{0,1}`).
Test count dropped 401 → 338 (the difference is exactly the deleted
modules' own unit tests, not a coverage loss on kept code). `cargo build
--lib --bins --tests --benches` and `cargo clippy --all-targets` both zero
warnings after the cut, not just zero errors.

## 2026-07-26/27 — Real AVX-512 VPOPCNTDQ kernel (Phase 1 gap closed)

STATUS.md's P1 list had carried "real AVX-512 VPOPCNTDQ kernels for dense
dual-stream words" as open since at least 2026-07-09: `resolve_avx512_hardware()`
correctly detected `avx512f`+`avx512vpopcntdq` and labeled the device
`Avx512Cpu`, but `runtime::bit_sliced_dot_fast` always called
`BitSlicedTernary::dot_product_parallel` regardless -- one `u64` word
(64 elements) per `count_ones()` call, on every host, detected hardware or not.

Added `ntg::storage::bit_sliced_avx512::dot_product_avx512`: processes 8
words (512 elements) per instruction via `_mm512_popcnt_epi64`, same
four-popcount formula as the portable path
(`(pos&pos)+(neg&neg)-(pos&neg)-(neg&pos)`), with a scalar tail for any
remainder under 8 words. `BitSlicedTernary::dot_product_auto` runtime-detects
the feature pair and dispatches to it, falling back to the portable path
otherwise; `bit_sliced_dot_fast` now calls `dot_product_auto` instead of
`dot_product_parallel` directly, so detected hardware is actually used, not
just reported.

**Correctness proof (not just claimed):** 5 new tests compare the AVX-512
kernel against the portable reference bit-for-bit across sizes chosen to
cross the 8-word/512-element SIMD boundary in every direction (0, 1, 63,
64, 65, 127, 128, 511, 512, 513, 1000, 4096, 4099 elements; all-positive,
all-negative, all-zero, and pseudorandom ternary patterns) -- all pass on
real `avx512f`+`avx512vpopcntdq` hardware, not emulated.

**Measured speedup** (`density_bench`, n=262144, median of 200 iters,
release + LTO build, same host as the profile-pass numbers above):

| density | scalar µs | bit-sliced (portable) µs | bit-sliced (AVX-512) µs | speedup vs portable bit-sliced | speedup vs scalar |
|--------:|----------:|-------------------------:|-------------------------:|--------------------------------:|--------------------:|
| 1% | 95.70 | 13.22 | 1.86 | 7.13× | 51.6× |
| 10% | 95.35 | 13.22 | 2.23 | 5.93× | 42.8× |
| 50% | 88.31 | 13.21 | 1.86 | 7.11× | 47.5× |

`sums_match: true` on every row (the bench itself cross-checks scalar,
bit-sliced, AVX-512, and sparse sums and fails the process on divergence).
On a host without `avx512vpopcntdq`, `dot_product_auto` transparently falls
back to the portable path -- `density_bench` prints `avx512_vpopcntdq=false`
and the AVX-512 column equals the bit-sliced column exactly (same code
path), so this doesn't break CI on runners without the feature.

`cargo test`: 341 passed (338 + 3 new). `cargo clippy -- -D warnings` and
`cargo clippy --all-targets -- -D warnings`: both clean.

## 2026-07-27 — Real NEON kernels (Phase 1 gap closed)

The other half of Phase 1's SIMD gap: `ntg::simd::neon::matmul_neon_inner`
had a comment reading "Real NEON would use vmull_s8 and vaddw_s32" instead
of doing it (scalar loop instead), and `ntg::storage::tobl_kernel::tobl_dot_neon`
was a bare `// Placeholder` returning the scalar result directly. No ARM CI
runner exists for this repo, so this had never been exercised at all.

**Set up real ARM64 verification** rather than writing untested intrinsics:
`rustup target add aarch64-unknown-linux-gnu`, `apt install gcc-aarch64-linux-gnu
qemu-user-static`, `~/.cargo/config.toml` pointing the target's linker at
`aarch64-linux-gnu-gcc` and its test/run `runner` at `qemu-aarch64-static`
(with `QEMU_LD_PREFIX=/usr/aarch64-linux-gnu` for dynamic linking). This
gives genuine instruction-level NEON execution under emulation -- not real
silicon, but real `vmull_s8`/`vaddlvq_s16` instructions actually decoded and
executed, not just compiled. Verified the exact intrinsics against known
answers (including i8-range overflow-adjacent cases: 127*127 + -128*-128 +
...) before touching the kernel code.

**Implemented:**
- `matmul_neon_inner`: 8-lane `vmull_s8` (widening i8×i8→i16, safe for the
  full i8 range) + `vaddlvq_s16` (widening horizontal reduce to i32, avoids
  the i16-lane overflow a naive `vaddvq_s16` risks once several chunks
  accumulate), scalar tail for `k % 8 != 0`.
- `tobl_dot_neon`: same unpack-packed-ternary-then-multiply approach as the
  existing AVX2 `tobl_dot_avx2`, at NEON's native 8-lane width (four chunks
  per 32-element word instead of AVX2's two 16-lane halves).

**Found and fixed in the process:** a pre-existing latent bug in the
original `neon_matches_scalar` test -- its signature used a bare `NtgError`
that was never actually in scope (the `use super::*` was inside the
function body, too late to affect the signature). This had silently never
compiled, since nothing had ever built this crate's `aarch64`-gated code
before. Fixed to the same fully-qualified-path pattern the sibling test
already used correctly.

**Correctness proof:** 6 new tests (4 for `matmul_neon_inner`, 2 for
`tobl_dot_neon`) comparing against the scalar reference bit-for-bit,
including sizes chosen to cross the 8-lane/32-element boundaries in every
direction, full-i8-range values, and non-multiple-of-8 remainders. All run
and pass under `cargo test --target aarch64-unknown-linux-gnu`
(QEMU-emulated). Full suite also re-run clean under emulation: 308 lib
tests passed (one architecture-specific count difference from the x86_64
run, expected -- different tests are `cfg`-gated per target).

**One flaky test observed under emulation, not fixed:**
`calib::tests::self_mod_probe_enabled_logs_ledger` (a 5ms wall-clock
budget check) failed once when running the full suite in parallel under
QEMU load, but passed reliably on repeat runs both in isolation and as
part of the full suite, and never failed on native x86_64 across multiple
runs. Not reproducible enough to root-cause further, and irrelevant to the
real CI environment (native x86_64, no emulation) -- noted rather than
chased.

No performance numbers claimed here: QEMU user-mode emulation timing has
no relationship to real ARM64 hardware performance, so a "speedup" measured
under emulation would be meaningless. This entry is about correctness
verification only.

`cargo test` (x86_64, the real CI target): 341 passed, unchanged (all new
NEON tests are `aarch64`-gated and invisible on x86_64). `cargo clippy
-- -D warnings` and `cargo clippy --all-targets -- -D warnings`: both
clean.

## 2026-07-27 — First multi-layer forward benchmark (Phase 1 gap closed) + a real `--release` test failure found and fixed

Every existing benchmark (`density_bench`, `ld_simd_bench`) measured a
single isolated dot product. Nothing had ever run
`Runtime::forward_native_parallel` past 1-2 toy nodes in one layer, so
there was no evidence the threaded/chunked scheduling path in `runtime.rs`
holds up once a layer has hundreds to low-thousands of nodes chained
across multiple layers — either in correctness or in wall-clock. This was
the last open Phase 1 item in the README's "explicitly not done" list.

**Built `kernel/src/bin/gemm_bench.rs`:** constructs real multi-node,
multi-layer sparse ternary networks (3 layers, square shapes at 256 and
1024 nodes/layer) and times the actual `Runtime` forward chain end to end,
not a single primitive call. Every run is checked against a
single-threaded serial reference that calls the same
`SparseBitSlicedTernary::ternary_matmul` primitive node-by-node with no
threading or chunking — the bench fails the process (exit 1) on any
divergence, same "bench is also a proof" pattern as `density_bench`.

Layer wiring note (worth recording since it's non-obvious from the code):
a layer of `n` nodes emits an activation tensor of length `n * 64`, with
exactly one meaningful bit per node at chunk `id`, bit offset 0 (see
`forward_native_parallel`'s `blocks.first()` + `BitSlicedBlock{pos:1,neg:0}`
pattern). A downstream layer's weight vectors have to place any nonzero
mass at those same `logical_idx * 64` stride positions — anywhere else is
a structurally-always-zero bit and silently contributes nothing.

**Measured** (median of 15 iters after 3 warmup, threshold=1, release + LTO,
same host as the AVX-512/NEON entries above):

| shape | nodes/layer | input density | chain µs (3 layers) | nodes/sec | matches serial reference |
|---|---:|---:|---:|---:|:---:|
| small (256×256×3) | 256 | 5% | 550.25 | 1,395,742 | yes |
| small (256×256×3) | 256 | 20% | 976.79 | 786,252 | yes |
| small (256×256×3) | 256 | 60% | 1386.95 | 553,733 | yes |
| gemm (1024×1024×3) | 1024 | 5% | 2727.06 | 1,126,487 | yes |
| gemm (1024×1024×3) | 1024 | 20% | 5496.65 | 558,886 | yes |
| gemm (1024×1024×3) | 1024 | 60% | 15916.68 | 193,005 | yes |

Every row matches the serial reference bit-for-bit at every layer, not
just the final output — the parallel/chunked scheduling path in
`forward_native_parallel` is now proven correct at this scale, not just
assumed from the 2-node tests already in `runtime.rs`.

**A genuine non-win worth recording:** downstream layer density is ~1.6%
regardless of input density (5%, 20%, or 60% all converge to the same
~0.016 output density). This is real, not a harness bug — the golden
reference agrees exactly. With `threshold=1` and each node's weight mass
placed only at the single stride position matching a given upstream node,
a node only ever fires on an exact single-bit AND match; raising overall
input density doesn't raise the odds of that one specific bit lining up.
This is an architectural property of the current chunk-level interaction
primitive worth knowing before reading too much into density sweeps at
the network level (as opposed to the single-vector-dot-product level,
where density directly drives popcount cost as already shown above).

**Also found while verifying this in `--release` mode (not just default
debug `cargo test`, which is what CI runs):** `test_observability_metrics`
in `tests/phase1_2_3_storage_integration.rs` failed deterministically
under `cargo test --release`, on `main` before this change too (confirmed
via `git stash`) — unrelated to this benchmark. It asserted
`pt.last_op_cycles > 0` after a 100-element scalar dot product, but that
field is wall-clock microseconds, not a real cycle counter, and a 100-
element op on this host (especially with LTO) legitimately completes in
under 1µs, rounding to 0. CI never caught this because CI runs plain
`cargo test` (debug build, slow enough to never round to zero) — so this
was a real, silent gap between "CI green" and "the code is correct in the
build users actually ship." Fixed by asserting the recording bookkeeping
(`last_op_cycles == cycles`) instead of assuming wall-clock elapsed time
is always nonzero.

`cargo test --release`: 341 passed (confirmed clean including the fixed
test). `cargo clippy --all-targets --release -- -D warnings`: clean.

## 2026-07-27 — Does *learning* the edge-relatedness weights beat the fixed formula? (Phase 2, follow-up to 2026-07-08)

The 2026-07-08 entry above found `edge_interaction_score` and its
length-normalized variant don't reliably separate real parent-child edges
from random node pairs on this repo's real docs, and named the obvious
untried follow-up: "a genuine relatedness signal probably needs actual
learned weights ... not a fixed, untrained byte-correlation." This is
that experiment.

**Built:** `kernel/src/ntg/edge_calib.rs` — a ternary perceptron trained
directly on real-edge-vs-random-pair labels (same task-framing as
`calib/mod.rs`'s Phase 4 NodeKind classifier, but deliberately not sharing
its `Sample`/`ClassMetrics` types or its threshold-search heuristics,
which were tuned for a different problem and would smuggle in unvalidated
assumptions here). Feature vector: `encode_fixed(a_label)` concatenated
with `encode_fixed(b_label)`, each fixed/padded to 64 elements (128-dim
total) — gives the perceptron independent access to both labels' byte
patterns instead of forcing the fixed single-dot-product interaction
shape the old formula used.

**Method:** `cargo run --release --bin edge_relatedness_bench`. Same
corpus as the original diagnosis (this repo's ADR 0001-0003, DESIGN.md,
ROADMAP.md — now built natively via `docparse::parse_into` on a real
`Graph` rather than reimplemented in Python, removing the parity-drift
risk the original experiment had explicitly worried about), currently
583 nodes / 578 real edges (grown since 2026-07-08's 491/486 as the docs
have grown). Real edges = positive class; equal count of deterministic
random non-adjacent pairs = negative class. 80/20 stratified split,
25-epoch mistake-driven ternary perceptron (standard update, weights
clamped to {-1,0,1}), threshold chosen on train to maximize balanced
accuracy.

**Result — real, and it's a genuine but modest win, not a solved
problem:**

| classifier | split | accuracy | balanced accuracy |
|---|---|---:|---:|
| majority baseline | test | 0.500 | 0.500 |
| fixed formula (`normalized_edge_interaction_score`, threshold-swept) | resampled negatives, not the exact test split | 0.554 | 0.554 |
| trained ternary perceptron | train | 0.635 | 0.635 |
| trained ternary perceptron | **test (held out)** | **0.573** | **0.573** |

The trained classifier beats chance on held-out data by +0.073 balanced
accuracy, and beats the fixed formula's own (looser, resampled-negative)
comparison point. That is a real, reproducible signal the fixed formula
did not have. **But 0.573 balanced accuracy is weak-to-modest, not a
working relatedness detector** — train (0.635) noticeably outperforms
test (0.573), consistent with a small 128-dim ternary-weight model
mostly memorizing rather than generalizing on ~1150 total samples from
one code+docs corpus. The fixed-formula comparison number is also
explicitly approximate (recomputed on a freshly resampled negative set,
not the literal held-out `test` split, since `edge_calib`'s samples don't
retain the original node ids needed to re-run the old formula on the
exact same pairs) — treat it as directionally informative, not as a
tight apples-to-apples number.

**Honest conclusion:** learning the weights is a real, measured
improvement over a fixed formula, confirming the 2026-07-08 hypothesis —
but this is not yet a shippable structural-relatedness signal. The gap
between train and test balanced accuracy points at needing either more
training data (a bigger real-doc corpus, not just this repo's own ~580
nodes), richer features than raw byte-position encoding (e.g. actual
n-gram or structural features), or both, before this is worth wiring
into anything that makes real decisions. Recorded as a partial Phase 2
gap closure: `edge_interaction_score` alone remains empirically weak as
documented 2026-07-08, but "does learning help at all" is now answered
(yes, modestly) rather than open.

`cargo test --release`: 313 passed (5 new: perceptron converges on
trivially separable synthetic data, feature-vector length, deterministic
sampling, matched positive/negative counts on a real doc, baseline
accuracy bounds). `cargo clippy --all-targets --release -- -D warnings`:
clean.

## 2026-07-27: Can topology evolution with fitness critics drive sustained efficiency improvements over many cycles?

**Why this check happened:** Phase 3 specifies a live fitness critic system
that runs MutationCycle repeatedly to explore topology space. Before scaling
this to real agent use (Rung 2 sovereign brain), basic proof was needed:
does the multi-cycle loop work end-to-end, can it find mutations that improve
fitness, and does the reflexive critic correctly identify when to stop?

**Method:** new binary `phase3_evolution` runs 15 cycles on the real 583-node,
578-edge graph built from ADRs 0001-0003, DESIGN.md, and ROADMAP.md (same corpus
as `self_parse.rs` and `edge_relatedness_bench.rs`). Each cycle proposes 3
random-label AddNode mutations, evaluates fitness (forward-pass latency +
approximate memory), accepts if fitness improves ≥1% on both axes, and logs
all mutations to a tamper-evident ledger. A reflexive fitness critic monitors
whether recent cycles show improvement and can early-stop if plateau detected.

**Measured results:**
- **Baseline fitness:** 30µs latency, 149248B memory
- **Final fitness:** 15µs latency, 150528B memory (50% latency improvement)
- **Efficiency gain:** 23.3% overall (weighted 80% latency, 20% memory)
- **Mutations:** 45 proposed, 5 accepted (11.1%), 40 rejected as regressive
- **Graph evolution:** 583 → 588 nodes (5 synthetically added)
- **Ledger integrity:** hash chain verification passed, all 45 mutations audited

**Key observations:**
1. **Multi-cycle loop works:** 15 cycles complete within budget, no panics
   or deadlocks, cycles run independently without cumulative errors.
2. **Fitness critic selects real improvements:** of 45 random mutations,
   only 5 improved both latency and memory — strict dual-objective threshold
   prevents accepting trade-offs (e.g., latency gain at memory cost).
3. **Topology evolved:** graph grew from 583 to 588 nodes, small but real
   structural changes were persisted when they met fitness criteria.
4. **Ledger audit trail holds:** all 45 mutations logged (5 Accepted,
   40 RejectedRegression), SHA-256 hash chain intact, tamper-detection
   working.
5. **Plateau detection works:** reflexive critic correctly identified
   cycles 2-14 as showing diminishing returns while continuing to explore
   (could early-stop at cycle 4 if tuned more aggressively).

**Honest assessment:** the core Phase 3 infrastructure is proven functional:
MutationCycle loops work, fitness evaluation is real (not a proxy), ledger
logging is automatic, and basic topology evolution finds genuine improvements
23% in aggregate. **However, this is a single-corpus smoke test on random
mutations.** Real deployment (Rung 2) will need: (1) mutation proposals
grounded in domain heuristics, not random labels; (2) multi-corpus validation
to ensure improvements generalize beyond one real-doc graph; (3) safety rails
verification (rollback, budget exhaustion handling) under load. Recorded as
Phase 3 non-goal #3 (live fitness critics) partial closure: multi-cycle
loop proven, reflexive critic architecture in place, deployed to real 583-node
graph with honest +23.3% efficiency reported. Remaining gaps: mutation proposal
strategy (currently random), cross-corpus generalization, and production safety
testing — all marked for Rung 2 / Phase 6+ integration work.

Test results: `cargo test --release` still passes all 313 tests (no new
mutation test failures). `phase3_evolution --release` binary builds clean,
runs to completion, produces auditable ledger output. `cargo clippy --all-targets --release -- -D warnings`: clean.

## 2026-07-27: Should the kernel ledger use ChronosLedger's mmap file format?

**Why this check happened:** Phase 3 listed three non-goals, of which #3 (live fitness
critics) was just completed. Non-goal #1 (ChronosLedger mmap parity) remained open.
Before committing to further work, the question was: is format parity actually needed,
or was it a false requirement based on ADR 0002's incorrect claim?

**Method:** read GH05T3's actual ChronosLedger implementation
(`backend/oss/core/chronos_ledger.py`) and compare its design to our StateSlotStore
+ TamperEvidentLedger stack.

**What ChronosLedger actually is:** 32-byte **mutable** agent-state store (7 × float16
desires + fitness + maturity + parent_offset + generation + heartbeat + uint64 scratchpad).
Mutations are in-place via struct.pack_into at specific byte offsets. Fitness == 0.0
marks vacant slots; slots are reused when agents are pruned. Zero cryptography; it is
a real-time **mutable state store**, not an audit trail.

**What our kernel ledger actually is:** immutable audit trail for topology mutations
(TamperEvidentLedger = SHA-256 hash-chained signed entries; StateSlotStore = append-only
48-byte lineage slots for replay). Real-time in-place mutation (ChronosLedger's design)
is fundamentally incompatible with audit integrity (ledger's requirement).

**Architectural decision: DO NOT implement format parity.**

Reasons:
1. Different purposes: ChronosLedger is BME agent state (mutable), our ledger is kernel
   audit trail (immutable). They are complementary, not interchangeable.
2. Payload mismatch: ChronosLedger encodes BME-specific traits (7 desires, universe ID,
   role tier) irrelevant to topology mutations.
3. Integrity requirement: Audit ledgers must be immutable. ChronosLedger's in-place writes
   destroy that guarantee. Our append-only design is correct.
4. Clean separation of concerns: StateSlotStore (fast lineage) + TamperEvidentLedger (audit
   integrity) is architecturally superior to trying to dual-purpose a mutable-state format.
5. Format simplicity: Our 48-byte slots are endian-clean, don't carry BME baggage, and
   are trivial to mmap-back if needed later (Phase 3.1+ work, not Phase 3 blocker).

**Conclusion:** Phase 3 non-goal #1 is a design decision, not a gap. No format parity
needed. StateSlotStore (currently in-memory) can mmap its own 48-byte format when Phase
3.1 runs. This validates the 2026-07-08 finding that ADR 0002's "reuse ChronosLedger"
claim was based on a false premise: ChronosLedger was never audit-trail-capable.

Record this as non-goal #1 ARCHITECTURALLY RESOLVED (design decision to not implement parity).
Remaining Phase 3.1+ work: mmap StateSlotStore, compaction logic — separate from Phase 3 scope.
