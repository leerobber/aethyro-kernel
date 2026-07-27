//! Real AVX-512 VPOPCNTDQ kernel for [`super::bit_sliced_ternary::BitSlicedTernary`]
//! dot products.
//!
//! Closes the gap STATUS.md flagged: hardware detection for AVX-512
//! (`ntg::accel::resolve_avx512_hardware`) existed, but nothing actually
//! used it -- every dot product ran the same portable `u64::count_ones()`
//! loop regardless of what the host advertised. This processes 8 `u64`
//! words (512 bits, 512 ternary elements) per instruction using
//! `_mm512_popcnt_epi64`, instead of one word (64 elements) at a time.
//!
//! Same math as [`super::bit_sliced_ternary::BitSlicedTernary::dot_product_parallel`]:
//! for each word, `(pos&pos).popcount() + (neg&neg).popcount() -
//! (pos&neg).popcount() - (neg&pos).popcount()`. Verified bit-identical
//! against that portable reference in `tests` below, across sizes that
//! cross the 8-word (512-element) SIMD boundary in every direction.

use super::bit_sliced_ternary::BitSlicedTernary;

/// Dot product using `_mm512_popcnt_epi64`, 8 words/512 elements per instruction.
///
/// # Safety
/// Caller must ensure the host supports `avx512f` and `avx512vpopcntdq`
/// (e.g. via `is_x86_feature_detected!`); this is not checked internally
/// since `#[target_feature]` functions cannot safely be called otherwise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn dot_product_avx512(a: &BitSlicedTernary, b: &BitSlicedTernary) -> i64 {
    use std::arch::x86_64::*;

    let words = a.pos_bits.len().min(b.pos_bits.len());
    let chunks = words / 8;
    let remainder_start = chunks * 8;

    let mut acc = _mm512_setzero_si512();

    for c in 0..chunks {
        let off = c * 8;
        let ap = _mm512_loadu_si512(a.pos_bits.as_ptr().add(off) as *const __m512i);
        let an = _mm512_loadu_si512(a.neg_bits.as_ptr().add(off) as *const __m512i);
        let bp = _mm512_loadu_si512(b.pos_bits.as_ptr().add(off) as *const __m512i);
        let bn = _mm512_loadu_si512(b.neg_bits.as_ptr().add(off) as *const __m512i);

        let pp = _mm512_popcnt_epi64(_mm512_and_si512(ap, bp));
        let mm = _mm512_popcnt_epi64(_mm512_and_si512(an, bn));
        let pm = _mm512_popcnt_epi64(_mm512_and_si512(ap, bn));
        let mp = _mm512_popcnt_epi64(_mm512_and_si512(an, bp));

        let pos_matches = _mm512_add_epi64(pp, mm);
        let neg_matches = _mm512_add_epi64(pm, mp);
        let delta = _mm512_sub_epi64(pos_matches, neg_matches);
        acc = _mm512_add_epi64(acc, delta);
    }

    let mut total = _mm512_reduce_add_epi64(acc);

    // Tail: fewer than 8 words left. Same four-popcount formula, scalar.
    for i in remainder_start..words {
        let pp = (a.pos_bits[i] & b.pos_bits[i]).count_ones() as i64;
        let mm = (a.neg_bits[i] & b.neg_bits[i]).count_ones() as i64;
        let pm = (a.pos_bits[i] & b.neg_bits[i]).count_ones() as i64;
        let mp = (a.neg_bits[i] & b.pos_bits[i]).count_ones() as i64;
        total += (pp + mm) - (pm + mp);
    }

    total
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn avx512_available() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
    }

    fn check_matches_portable(vals_a: &[i8], vals_b: &[i8]) {
        let a = BitSlicedTernary::from_slice(vals_a);
        let b = BitSlicedTernary::from_slice(vals_b);
        let portable = BitSlicedTernary::dot_product_parallel(&a, &b);
        if avx512_available() {
            let avx512 = unsafe { dot_product_avx512(&a, &b) };
            assert_eq!(
                avx512,
                portable,
                "AVX-512 result diverged from portable popcount at len={}",
                vals_a.len()
            );
        }
    }

    fn ternary_pattern(len: usize, seed: u64) -> Vec<i8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift, deterministic across runs
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match state % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                }
            })
            .collect()
    }

    #[test]
    fn matches_portable_across_word_boundaries() {
        // 0, <1 word, exactly 1 word, >1 word, exactly one SIMD chunk (8
        // words = 512 elements), one chunk + remainder, several chunks.
        for &len in &[0usize, 1, 63, 64, 65, 127, 128, 511, 512, 513, 1000, 4096, 4099] {
            let a = ternary_pattern(len, 0x1234_5678_9abc_def0 ^ len as u64);
            let b = ternary_pattern(len, 0x0fed_cba9_8765_4321 ^ (len as u64).wrapping_mul(7));
            check_matches_portable(&a, &b);
        }
    }

    #[test]
    fn matches_portable_all_same_sign() {
        let a = vec![1i8; 600];
        let b = vec![1i8; 600];
        check_matches_portable(&a, &b);

        let a = vec![-1i8; 600];
        let b = vec![1i8; 600];
        check_matches_portable(&a, &b);
    }

    #[test]
    fn matches_portable_all_zero() {
        let a = vec![0i8; 600];
        let b = vec![0i8; 600];
        check_matches_portable(&a, &b);
    }
}
