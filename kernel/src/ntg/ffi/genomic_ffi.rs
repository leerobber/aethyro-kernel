//! NTG Genomic FFI: C-compatible interface for OmniSynth-X
//!
//! Exposes genomic operator to FFI for Python/C integration
//! Handles: LD computation, PRS scoring, LD clustering

use crate::ntg::operators::genomic::GenomicOperator;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global genomic operation counter
static GENOMIC_OP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Opaque handle for C code (wraps GenomicOperator)
pub struct GenomicHandle {
    inner: GenomicOperator,
}

// =====================================================================
// LIFECYCLE
// =====================================================================

/// Create new GenomicOperator (C interface)
///
/// # Safety
/// Caller must call `ntg_genomic_drop` to free allocated memory.
#[no_mangle]
pub extern "C" fn ntg_genomic_new(
    num_individuals: usize,
    num_snps: usize,
) -> *mut GenomicHandle {
    let handle = Box::new(GenomicHandle {
        inner: GenomicOperator::new(num_individuals, num_snps),
    });
    Box::into_raw(handle)
}

/// Destroy GenomicOperator (C interface)
///
/// # Safety
/// `handle` must be either null or a pointer previously returned by
/// `ntg_genomic_new` that has not already been passed to `ntg_genomic_drop`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_drop(handle: *mut GenomicHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// =====================================================================
// GENOTYPE I/O
// =====================================================================

/// Set genotype value
/// val: 0=ref/ref, 1=ref/alt, 2=alt/alt, 3=missing
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_set(
    handle: *mut GenomicHandle,
    snp_idx: usize,
    ind_idx: usize,
    val: u8,
) -> i32 {
    if handle.is_null() {
        return -1; // EINVAL
    }

    unsafe {
        if snp_idx >= (*handle).inner.num_snps || ind_idx >= (*handle).inner.num_individuals {
            return -2; // out of bounds
        }
        (*handle).inner.set(snp_idx, ind_idx, val);
        0
    }
}

/// Get genotype value
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_get(
    handle: *const GenomicHandle,
    snp_idx: usize,
    ind_idx: usize,
) -> i32 {
    if handle.is_null() {
        return -1;
    }

    unsafe {
        if snp_idx >= (*handle).inner.num_snps || ind_idx >= (*handle).inner.num_individuals {
            return -2;
        }
        (*handle).inner.get(snp_idx, ind_idx) as i32
    }
}

// =====================================================================
// STATISTICS
// =====================================================================

/// Compute means and standard deviations
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_compute_statistics(handle: *mut GenomicHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }

    unsafe {
        (*handle).inner.compute_statistics();
        GENOMIC_OP_COUNT.fetch_add(1, Ordering::Relaxed);
        0
    }
}

/// Get mean for a SNP
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_get_mean(handle: *const GenomicHandle, snp_idx: usize) -> f64 {
    if handle.is_null() {
        return 0.0;
    }

    unsafe {
        if snp_idx >= (*handle).inner.num_snps {
            return 0.0;
        }
        // Explicit `&`, not needless: an implicit autoref through this raw
        // pointer deref is a rustc-denied lint (dangerous_implicit_autorefs).
        #[allow(clippy::needless_borrow)]
        (&(*handle).inner.means).get(snp_idx).copied().unwrap_or(0.0)
    }
}

/// Get std dev for a SNP
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_get_std_dev(handle: *const GenomicHandle, snp_idx: usize) -> f64 {
    if handle.is_null() {
        return 0.0;
    }

    unsafe {
        if snp_idx >= (*handle).inner.num_snps {
            return 0.0;
        }
        // Explicit `&`, not needless: an implicit autoref through this raw
        // pointer deref is a rustc-denied lint (dangerous_implicit_autorefs).
        #[allow(clippy::needless_borrow)]
        (&(*handle).inner.std_devs).get(snp_idx).copied().unwrap_or(1.0)
    }
}

// =====================================================================
// LINKAGE DISEQUILIBRIUM
// =====================================================================

/// Compute LD matrix (returns into pre-allocated buffer)
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`. `out` must point to at least (num_snps * num_snps)
/// f64 values, valid for writes.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_compute_ld(
    handle: *mut GenomicHandle,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return -1;
    }

    unsafe {
        let expected_len = (*handle).inner.num_snps * (*handle).inner.num_snps;
        if out_len < expected_len {
            return -2; // buffer too small
        }

        let ld_matrix = (*handle).inner.compute_ld_matrix();

        for (i, val) in ld_matrix.iter().enumerate() {
            if i < out_len {
                *out.add(i) = *val;
            }
        }

        GENOMIC_OP_COUNT.fetch_add(1, Ordering::Relaxed);
        ld_matrix.len() as i32
    }
}

/// Get single LD value (r)
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_get_ld(
    handle: *mut GenomicHandle,
    snp_i: usize,
    snp_j: usize,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }

    unsafe {
        if snp_i >= (*handle).inner.num_snps || snp_j >= (*handle).inner.num_snps {
            return 0.0;
        }

        let ld_matrix = (*handle).inner.compute_ld_matrix();
        ld_matrix[snp_i * (*handle).inner.num_snps + snp_j]
    }
}

// =====================================================================
// POLYGENIC RISK SCORES
// =====================================================================

/// Compute PRS for all individuals
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`. `weights` must point to at least `weights_len` f64
/// values, valid for reads. `out` must point to at least `out_len` f64
/// values, valid for writes.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_compute_prs(
    handle: *const GenomicHandle,
    weights: *const f64,
    weights_len: usize,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    if handle.is_null() || weights.is_null() || out.is_null() {
        return -1;
    }

    unsafe {
        if weights_len != (*handle).inner.num_snps {
            return -2; // dimension mismatch
        }
        if out_len < (*handle).inner.num_individuals {
            return -3; // output buffer too small
        }

        let weights_slice = std::slice::from_raw_parts(weights, weights_len);
        let prs = (*handle).inner.compute_prs(weights_slice);

        for (i, val) in prs.iter().enumerate() {
            if i < out_len {
                *out.add(i) = *val;
            }
        }

        GENOMIC_OP_COUNT.fetch_add(1, Ordering::Relaxed);
        prs.len() as i32
    }
}

// =====================================================================
// METADATA
// =====================================================================

/// Get number of individuals
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_num_individuals(handle: *const GenomicHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    unsafe { (*handle).inner.num_individuals }
}

/// Get number of SNPs
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_num_snps(handle: *const GenomicHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    unsafe { (*handle).inner.num_snps }
}

/// Get missing data rate (0.0 to 1.0) - stub implementation
#[no_mangle]
pub extern "C" fn ntg_genomic_missing_rate(_handle: *const GenomicHandle) -> f64 {
    // TODO: implement missing rate estimation in GenomicOperator
    0.0
}

/// Get total genomic operation count (for profiling)
#[no_mangle]
pub extern "C" fn ntg_genomic_op_count() -> u64 {
    GENOMIC_OP_COUNT.load(Ordering::Relaxed)
}

// =====================================================================
// UTILITY
// =====================================================================

/// Bulk load genotypes from array
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`. `genotypes` must point to (num_snps *
/// num_individuals) u8 values, in row-major order (SNP-major), valid for
/// reads.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_load_bulk(
    handle: *mut GenomicHandle,
    genotypes: *const u8,
    num_genotypes: usize,
) -> i32 {
    if handle.is_null() || genotypes.is_null() {
        return -1;
    }

    unsafe {
        let expected = (*handle).inner.num_snps * (*handle).inner.num_individuals;
        if num_genotypes != expected {
            return -2; // size mismatch
        }

        let geno_slice = std::slice::from_raw_parts(genotypes, num_genotypes);
        let mut idx = 0;

        for snp in 0..(*handle).inner.num_snps {
            for ind in 0..(*handle).inner.num_individuals {
                if idx < geno_slice.len() {
                    (*handle).inner.set(snp, ind, geno_slice[idx]);
                    idx += 1;
                }
            }
        }

        GENOMIC_OP_COUNT.fetch_add(1, Ordering::Relaxed);
        0
    }
}

/// Export all genotypes (for serialization)
///
/// # Safety
/// `handle` must be either null or a valid, live pointer returned by
/// `ntg_genomic_new`. `out` must point to at least (num_snps *
/// num_individuals) u8 values, valid for writes.
#[no_mangle]
pub unsafe extern "C" fn ntg_genomic_export_bulk(
    handle: *const GenomicHandle,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return -1;
    }

    unsafe {
        let expected = (*handle).inner.num_snps * (*handle).inner.num_individuals;
        if out_len < expected {
            return -2; // buffer too small
        }

        let mut idx = 0;
        for snp in 0..(*handle).inner.num_snps {
            for ind in 0..(*handle).inner.num_individuals {
                let val = (*handle).inner.get(snp, ind);
                *out.add(idx) = val;
                idx += 1;
            }
        }

        idx as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genomic_ffi_new_drop() {
        unsafe {
            let handle = ntg_genomic_new(10, 5);
            assert!(!handle.is_null());
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_set_get_roundtrip() {
        unsafe {
            let handle = ntg_genomic_new(10, 5);
            assert_eq!(ntg_genomic_set(handle, 0, 0, 2), 0);
            assert_eq!(ntg_genomic_get(handle, 0, 0), 2);
            assert_eq!(ntg_genomic_set(handle, 4, 9, 1), 0);
            assert_eq!(ntg_genomic_get(handle, 4, 9), 1);
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_set_get_out_of_bounds() {
        unsafe {
            let handle = ntg_genomic_new(10, 5);
            assert_eq!(ntg_genomic_set(handle, 5, 0, 1), -2);
            assert_eq!(ntg_genomic_get(handle, 0, 10), -2);
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_statistics_and_metadata() {
        unsafe {
            let handle = ntg_genomic_new(100, 4);
            for snp in 0..4 {
                for ind in 0..100 {
                    ntg_genomic_set(handle, snp, ind, ((snp + ind) % 3) as u8);
                }
            }
            assert_eq!(ntg_genomic_compute_statistics(handle), 0);

            // Every SNP's mean should be pulled toward [0, 2] genotype range.
            for snp in 0..4 {
                let mean = ntg_genomic_get_mean(handle, snp);
                assert!((0.0..=2.0).contains(&mean));
                assert!(ntg_genomic_get_std_dev(handle, snp) >= 0.0);
            }

            assert_eq!(ntg_genomic_num_individuals(handle), 100);
            assert_eq!(ntg_genomic_num_snps(handle), 4);

            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_ld_matrix_matches_get_ld() {
        unsafe {
            let handle = ntg_genomic_new(50, 3);
            for snp in 0..3 {
                for ind in 0..50 {
                    ntg_genomic_set(handle, snp, ind, ((snp * 2 + ind) % 3) as u8);
                }
            }

            let mut buf = vec![0.0f64; 3 * 3];
            let written = ntg_genomic_compute_ld(handle, buf.as_mut_ptr(), buf.len());
            assert_eq!(written, 9);

            // A SNP always perfectly correlates with itself.
            assert!((buf[0] - 1.0).abs() < 1e-9);

            // ntg_genomic_get_ld must agree with the bulk matrix for the same pair.
            let direct = ntg_genomic_get_ld(handle, 1, 2);
            assert!((direct - buf[3 + 2]).abs() < 1e-9);

            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_compute_ld_rejects_small_buffer() {
        unsafe {
            let handle = ntg_genomic_new(20, 3);
            let mut buf = vec![0.0f64; 2]; // too small for 3x3
            assert_eq!(ntg_genomic_compute_ld(handle, buf.as_mut_ptr(), buf.len()), -2);
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_prs_scoring() {
        unsafe {
            let handle = ntg_genomic_new(20, 3);
            for ind in 0..20 {
                ntg_genomic_set(handle, 0, ind, 1);
                ntg_genomic_set(handle, 1, ind, 2);
                ntg_genomic_set(handle, 2, ind, 0);
            }

            let weights = [0.5f64, 1.0, 2.0];
            let mut out = vec![0.0f64; 20];
            let written = ntg_genomic_compute_prs(
                handle,
                weights.as_ptr(),
                weights.len(),
                out.as_mut_ptr(),
                out.len(),
            );
            assert_eq!(written, 20);
            // genotype 1 * 0.5 + genotype 2 * 1.0 + genotype 0 * 2.0 = 2.5
            assert!((out[0] - 2.5).abs() < 1e-9);

            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_prs_rejects_dimension_mismatch() {
        unsafe {
            let handle = ntg_genomic_new(10, 3);
            let weights = [1.0f64, 2.0]; // wrong length (2 vs 3 SNPs)
            let mut out = vec![0.0f64; 10];
            let result = ntg_genomic_compute_prs(
                handle,
                weights.as_ptr(),
                weights.len(),
                out.as_mut_ptr(),
                out.len(),
            );
            assert_eq!(result, -2);
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_bulk_load_export_roundtrip() {
        unsafe {
            let handle = ntg_genomic_new(4, 3);
            // Row-major (SNP-major): snp0=[0,1,2,3], snp1=[1,1,1,1], snp2=[2,0,2,0]
            let genotypes: [u8; 12] = [0, 1, 2, 3, 1, 1, 1, 1, 2, 0, 2, 0];
            assert_eq!(
                ntg_genomic_load_bulk(handle, genotypes.as_ptr(), genotypes.len()),
                0
            );

            let mut out = vec![0u8; 12];
            let written = ntg_genomic_export_bulk(handle, out.as_mut_ptr(), out.len());
            assert_eq!(written, 12);
            assert_eq!(&out[..], &genotypes[..]);

            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_bulk_load_rejects_size_mismatch() {
        unsafe {
            let handle = ntg_genomic_new(4, 3);
            let genotypes: [u8; 5] = [0, 1, 2, 3, 1]; // should be 12
            assert_eq!(
                ntg_genomic_load_bulk(handle, genotypes.as_ptr(), genotypes.len()),
                -2
            );
            ntg_genomic_drop(handle);
        }
    }

    #[test]
    fn genomic_ffi_null_handle_is_safe() {
        unsafe {
            let null: *mut GenomicHandle = std::ptr::null_mut();
            assert_eq!(ntg_genomic_set(null, 0, 0, 1), -1);
            assert_eq!(ntg_genomic_get(null, 0, 0), -1);
            assert_eq!(ntg_genomic_compute_statistics(null), -1);
            assert_eq!(ntg_genomic_get_mean(null, 0), 0.0);
            assert_eq!(ntg_genomic_get_std_dev(null, 0), 0.0);
            assert_eq!(ntg_genomic_num_individuals(null), 0);
            assert_eq!(ntg_genomic_num_snps(null), 0);
            ntg_genomic_drop(null); // must not panic
        }
    }

    #[test]
    fn genomic_ffi_op_count_increments() {
        unsafe {
            let before = ntg_genomic_op_count();
            let handle = ntg_genomic_new(5, 2);
            ntg_genomic_compute_statistics(handle);
            assert!(ntg_genomic_op_count() > before);
            ntg_genomic_drop(handle);
        }
    }
}
