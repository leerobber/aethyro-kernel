//! Genomic data plumbing behind the sovereign-brain Rung 2 fitness axis.
//!
//! This is not a general-purpose genomics pipeline: it's the minimal real
//! VCF -> LD -> haplotype-block -> chromosome-brain -> sovereign-brain chain
//! that `ntg::mutation::multi_axis` needs for its `biological_consistency`
//! fitness axis (see that module's docs for how it's used). Everything here
//! is reachable from `sovereign_brain::SovereignBrain` or from the
//! `sovereign_campaign` / `sovereign_brain_demo` binaries; disconnected
//! exploratory modules from the project's earlier disease-detection phase
//! (agents/domain_agents/evolution/phenotype/report_gen/quality_control/
//! extended_validation/optimized_core/epigenetic_engine/vitascale) were
//! removed as dead weight -- see git history if any of that is wanted back.

pub mod bitsliced_genotypes;
pub mod vcf_stream;
pub mod ld_compute;
pub mod haplotype_blocks;
pub mod chromosome_brain;
pub mod synthesis;
pub mod validation;
pub mod real_pipeline;
pub mod sovereign_brain;
pub mod sovereign_fitness;
pub mod language_organ;
pub mod organ;
pub mod selection_loop;
pub mod sovereign_persist;

pub use bitsliced_genotypes::BitstreamGenotypes;
pub use vcf_stream::{VcfParser, VcfChromosome, SnpRecord};
pub use ld_compute::{LdComputer, LdMatrix, LdPair};
pub use haplotype_blocks::{BlockDetector, HaplotypeBlock, BlockStatistics, compute_block_statistics};
pub use chromosome_brain::{ChromosomeBrain, ChromosomeId, NeuronId, Synapse, GenomicNeuron, KairosState, BrainSummary, EmbeddingLayer, init_chromosome_brain};
pub use synthesis::{Genome, GenomeSampler, HaplotypePool};
pub use validation::{GenomeComparator, ReferenceGenome, SyntheticGenome, ValidationResults, PowerAnalysis};
pub use real_pipeline::{RealChromosomeData, build_real_chromosome, snp_key};
pub use sovereign_brain::{
    SovereignBrain, WorkingSet, LtmMotif, LtmStats, GlobalNeuronRef, StructuralMetrics,
    ConsolidateReport,
};
pub use sovereign_fitness::{
    SovereignFitnessContext, reference_from_brain, synthetic_from_brain, ld_coverage,
};
pub use language_organ::{LanguageOrgan, fixture_docs};
pub use organ::Organ;
pub use selection_loop::{run_selection_loop, format_summary, LoopSummary, LoopStepRecord};
pub use sovereign_persist::{save_snapshot, load_snapshot_into, SnapshotReport};
