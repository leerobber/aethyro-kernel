//! NEON path for ternary matmul (ARM64).
//!
//! Real intrinsic implementation: `vmull_s8` widens 8 lanes of i8*i8 to
//! i16 in one instruction (safe for the full i8 range: 127*127 = 16129,
//! -128*-128 = 16384, both well inside i16), and `vaddlvq_s16` does a
//! widening horizontal reduction to i32 (avoids the i16-lane overflow a
//! naive `vaddvq_s16` would risk once several products accumulate).
//! Verified bit-identical to [`crate::ntg::ternary::matmul_scalar`] --
//! see `tests` below -- cross-compiled to `aarch64-unknown-linux-gnu` and
//! run under QEMU user-mode emulation (no ARM CI runner exists yet, so
//! this is instruction-level-emulated verification, not real silicon;
//! flagged honestly rather than claimed as hardware-tested).
//!
//! On non-ARM hosts, [`matmul_neon`] falls back to scalar immediately.

use super::super::error::NtgError;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddlvq_s16, vld1_s8, vmull_s8};

/// NEON matmul: (m x k) @ (k x n) -> m x n
/// Requires: ARM64 with NEON support
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn matmul_neon_inner(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, NtgError> {
    if a.len() != m * k {
        return Err(NtgError::ShapeMismatch {
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(NtgError::ShapeMismatch {
            expected: k * n,
            got: b.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];
    // b is column-strided (element (p, j) at index p*n+j), so an 8-wide
    // contiguous NEON load only works for the a-row; b needs gathering.
    let mut b_chunk = [0i8; 8];

    for i in 0..m {
        for j in 0..n {
            let mut sum: i32 = 0;

            // Process k in chunks of 8 (NEON register width for i8).
            let mut p = 0;
            while p + 8 <= k {
                let a_vals = &a[i * k + p..i * k + p + 8];
                for (idx, slot) in b_chunk.iter_mut().enumerate() {
                    *slot = b[(p + idx) * n + j];
                }

                // Safety: `neon` is enabled for this whole item via the
                // outer `target_feature = "neon"` cfg gate (this is a
                // baseline AArch64 feature, always on for this target),
                // and both slices are exactly 8 elements.
                sum += unsafe {
                    let av = vld1_s8(a_vals.as_ptr());
                    let bv = vld1_s8(b_chunk.as_ptr());
                    let prod = vmull_s8(av, bv);
                    vaddlvq_s16(prod)
                };

                p += 8;
            }

            // Remainder (< 8 elements): scalar.
            while p < k {
                sum += (a[i * k + p] as i32) * (b[p * n + j] as i32);
                p += 1;
            }

            c[i * n + j] = sum as f32;
        }
    }

    Ok(c)
}

/// Public wrapper for NEON matmul
pub fn matmul_neon(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, NtgError> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return matmul_neon_inner(a, b, m, k, n);
    }

    // Fallback to scalar if NEON not available
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    super::super::ternary::matmul_scalar(a, b, m, k, n)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matmul_simple() -> Result<(), super::super::super::error::NtgError> {
        use super::*;
        let a = vec![1i8, -1, 0, 1];
        let b = vec![1i8, 0, -1, 1];

        let result = matmul_neon(&a, &b, 2, 2, 2)?;

        // Expected: [[2, -1], [-1, 1]]
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], -1.0);
        assert_eq!(result[2], -1.0);
        assert_eq!(result[3], 1.0);

        Ok(())
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_scalar() -> Result<(), super::super::super::error::NtgError> {
        use super::*;
        let a = vec![1i8, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0];
        let b = vec![1i8, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1];

        let scalar_result = super::super::super::ternary::matmul_scalar(&a, &b, 3, 4, 3)?;
        let neon_result = matmul_neon(&a, &b, 3, 4, 3)?;

        // Must be bit-identical
        assert_eq!(scalar_result, neon_result, "NEON result differs from scalar");

        Ok(())
    }

    /// Exercises the 8-wide NEON chunk path (k=16 crosses two full chunks)
    /// with non-ternary, full-i8-range values to check the widening
    /// multiply/reduce doesn't overflow where a naive i16 accumulation would.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_scalar_full_i8_range() -> Result<(), super::super::super::error::NtgError> {
        use super::*;
        let a: Vec<i8> = (0..16)
            .map(|i| if i % 2 == 0 { 127 } else { -128 })
            .collect();
        let b: Vec<i8> = (0..16)
            .map(|i| if i % 3 == 0 { -128 } else { 127 })
            .collect();

        let scalar_result = super::super::super::ternary::matmul_scalar(&a, &b, 1, 16, 1)?;
        let neon_result = matmul_neon(&a, &b, 1, 16, 1)?;
        assert_eq!(scalar_result, neon_result);
        Ok(())
    }

    /// k not a multiple of 8 (11): exercises the chunk loop plus the
    /// scalar remainder tail together.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_scalar_remainder_tail() -> Result<(), super::super::super::error::NtgError> {
        use super::*;
        let a = vec![1i8, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let b = vec![1i8, 0, -1, 1, 0, 1, -1, 0, 1, -1, 1];

        let scalar_result = super::super::super::ternary::matmul_scalar(&a, &b, 1, 11, 1)?;
        let neon_result = matmul_neon(&a, &b, 1, 11, 1)?;
        assert_eq!(scalar_result, neon_result);
        Ok(())
    }
}
