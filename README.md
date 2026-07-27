# aethyro-kernel

The Aethyro NTG (Neural Ternary Graph) Engine: a ternary-weight,
self-evolving-graph-topology inference engine, wrapped in a
tamper-evident audit ledger, engineered for air-gapped / sovereign edge
deployment.

This started as an extract of the kernel from
[aethyro-ntg](https://github.com/leerobber/aethyro-ntg), with a round of
correctness/soundness fixes applied (FFI unsound-pointer bugs, a ledger
audit-integrity bug, an O(n²) hot path in the graph scheduler, two
corrupted identifiers, and the CI-breaking clippy errors that had crept
back in).

**Correction (2026-07-26):** this repo originally shipped the *entire*
`aethyro-ntg` tree despite claiming otherwise here — including the full
pre-pivot disease-detection genomics pipeline, which had zero
reachability from the kernel. That's now actually fixed: `genomic/` is
down to the 14 files that `ntg::mutation::multi_axis`'s Rung 2
`biological_consistency` fitness axis genuinely depends on (a real VCF
→ LD → haplotype-block → chromosome-brain → `SovereignBrain` chain), not
a general-purpose genomics library. Everything else in that tree —
disease-detection agents, evolution/phenotype/quality-control/
extended-validation pipelines, `vitascale/`, and the 10 demo binaries
that only exercised those — was deleted as dead weight. If this claim
ever drifts from reality again, trust `find kernel/src -name '*.rs' |
xargs grep` over this paragraph.

**Current status (authoritative):**  
→ **[docs/STATUS.md](docs/STATUS.md)** — full research-agency report, test
proof matrix, gaps, and next priorities.  
→ **[docs/ROADMAP.md](docs/ROADMAP.md)** — phased gates.

**As of 2026-07-26:** pre-alpha research kernel, **capability v10**,
Phase 0–5 COMPLETE (calib + precision + GraphNode warm-start path), 338
tests green, clippy clean. Not benchmarked against production
aethyro.com inference; no GTM decision.

## What's actually new (precise claim)

See [docs/architecture/0001-vision-and-pivot.md](docs/architecture/0001-vision-and-pivot.md)
and [docs/LITERATURE.md](docs/LITERATURE.md). Ternary quantization and
dynamic graph topology each have prior art. This project's bet is the
*combination* of both inside a deterministic-replay, ledger-audited
safety envelope for fully air-gapped deployment.

## Where this fits with aethyro.com

[aethyro.com](https://aethyro.com) is live (Personal, CPA, Dev, Research).
This engine's first intended target is an efficiency upgrade on hardware
those tiers already run — not a new vertical sales motion.

## Quick start

```bash
cd kernel
cargo test
cargo build --release
```

From repo root (tests + benches + calib + schooling):

```bash
./tools/dev.sh check    # test + clippy
./tools/dev.sh model    # train → artifacts/models + eval + predict
./tools/dev.sh model-ab # A/B two epoch settings
./tools/dev.sh school   # doctorate study+exam phases 0–5 (75% gate, notebooks)
```

Schooling notebooks: [docs/schooling/](docs/schooling/) — real data only, fail <75% full redo.

Optional layer ingest contract check:

```bash
echo '{"layers":[{"nodes":[{"id":0},{"id":1}]}]}' | python3 tools/ingest.py
```

## Repository layout

| Path | Purpose |
|------|---------|
| `kernel/` | Rust crate: ternary core, storage, graph, ledger, mutation, runtime, calib |
| `kernel/src/genomic/` | VCF → LD → haplotype-block → `SovereignBrain` chain feeding Rung 2 fitness (`ntg::mutation::multi_axis`) — not a general genomics pipeline |
| `tools/ingest.py` | Sequential `GraphNode.id` contract for native forward |
| `tools/dev.sh` | One-shot test / calib / model / bench workflows |
| `artifacts/models/` | Local CalibModel dumps (`dev.sh model`; not required in git) |
| `docs/STATUS.md` | **Where the project is** (read first) |
| `docs/ROADMAP.md` | Phased build plan and open gates |
| `docs/PHASE5_PREP.md` | Pre-positioned Phase 5 hooks |
| `docs/DESIGN.md` | Technical architecture |
| `docs/LITERATURE.md` | Sourced novelty grounding |
| `docs/EXPERIMENTS.md` | Measured wins and non-wins |
| `docs/architecture/` | ADRs 0001–0007 |
| `kernel/FFI_*.md`, `TOBL_FFI_REFERENCE.md` | C ABI notes |

## Implemented stack (summary)

1. **Ternary core** — scalar golden `matmul_scalar`, encoding  
2. **Storage** — packed 2-bit, dual-stream bit-sliced, sparse COO  
3. **SIMD / TOBL / FFI** — runtime dispatch, C ABI, OpStats  
4. **Graph + SIS** — topology, doc/path parse, fs-event pure layer, adj_list  
5. **Native runtime** — `forward_native_parallel` + density-based `AccelManager`  
6. **Ledger + self-mod** — SHA-256 chain, budgets, fitness, **off by default**
7. **Rung 2 multi-axis fitness** — task/structure/biology/safety scoring over `SovereignBrain`, real VCF-derived biological-consistency signal

## Explicitly not done

- Full AVX-512 VPOPCNTDQ kernels (detect yes, full kernels no)  
- GPU/NPU (re-scoped: CPU TOBL 12–20×; revisit at large tensors)  
- Phase 6 integration / product head-to-head vs aethyro.com  
- Self-mod enabled by default (stays off)
- Lazy PIXEL-lite glyph fingerprints (ADR 0003 design only)

## Engineering principles

See [CONTRIBUTING.md](CONTRIBUTING.md). Rule: **measure, don't assume**;
docs and CI green before calling a phase done.

## License

MIT — see [LICENSE](LICENSE).
