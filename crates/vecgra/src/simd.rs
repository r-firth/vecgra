#[inline]
pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "dot-product dimensions differ");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 baseline and the length check
        // above guarantees every vector load can read from both slices.
        return unsafe { dot_neon(left, right) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection establishes AVX2 support and the
            // length check above guarantees both slices have equal bounds.
            return unsafe { dot_avx2(left, right) };
        }
    }

    #[allow(unreachable_code)]
    dot_portable(left, right)
}

#[inline]
pub(crate) fn dot_f16(left: &[f32], right_le_bytes: &[u8]) -> f32 {
    assert_eq!(
        left.len().checked_mul(2),
        Some(right_le_bytes.len()),
        "F16 dot-product dimensions differ"
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // SAFETY: runtime detection establishes FP16 support and the
            // function checks slice bounds before every unaligned load.
            return unsafe { dot_f16_neon(left, right_le_bytes) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: both required instruction-set extensions were detected.
            return unsafe { dot_f16_avx2(left, right_le_bytes) };
        }
    }

    dot_f16_table(left, right_le_bytes)
}

#[inline]
fn dot_f16_table(left: &[f32], right_le_bytes: &[u8]) -> f32 {
    use std::sync::OnceLock;
    static DECODE: OnceLock<Box<[f32]>> = OnceLock::new();
    let decode = DECODE.get_or_init(|| {
        (0..=u16::MAX)
            .map(crate::codec::f16_to_f32)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });

    // Independent accumulators hide most of the table-lookup latency and keep
    // the F16 checkpoint compact without forcing a full F32 hydration.
    let mut sums = [0.0f32; 8];
    let chunks = left.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        let byte = index * 2;
        for lane in 0..8 {
            let bits = u16::from_le_bytes([
                right_le_bytes[byte + lane * 2],
                right_le_bytes[byte + lane * 2 + 1],
            ]);
            sums[lane] += left[index + lane] * decode[bits as usize];
        }
    }
    let mut result =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for (index, value) in left.iter().enumerate().skip(chunks * 8) {
        let byte = index * 2;
        let bits = u16::from_le_bytes([right_le_bytes[byte], right_le_bytes[byte + 1]]);
        result += value * decode[bits as usize];
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn dot_f16_neon(left: &[f32], right_le_bytes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    use std::arch::asm;

    let mut index = 0;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    while index + 8 <= left.len() {
        // SAFETY: the loop bounds cover eight F32 and eight F16 inputs;
        // AArch64 vector loads permit unaligned addresses.
        let bits0 = unsafe { vld1_u16(right_le_bytes.as_ptr().add(index * 2).cast()) };
        // SAFETY: the same loop bound covers the second four-value load.
        let bits1 = unsafe { vld1_u16(right_le_bytes.as_ptr().add((index + 4) * 2).cast()) };
        let right0: float32x4_t;
        let right1: float32x4_t;
        // SAFETY: Stable Rust does not expose `vcvt_f32_f16`; these register-
        // only instructions are the exact AArch64 FCVTL operations used by
        // that intrinsic and do not access memory or the stack.
        unsafe {
            asm!(
                "fcvtl {right0:v}.4s, {bits0:v}.4h",
                "fcvtl {right1:v}.4s, {bits1:v}.4h",
                right0 = out(vreg) right0,
                right1 = out(vreg) right1,
                bits0 = in(vreg) bits0,
                bits1 = in(vreg) bits1,
                options(pure, nomem, nostack)
            );
        }
        // SAFETY: the loop bound guarantees the first four readable floats.
        let left0 = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        // SAFETY: the same loop bound covers the second four-value load.
        let left1 = unsafe { vld1q_f32(left.as_ptr().add(index + 4)) };
        sum0 = vfmaq_f32(sum0, left0, right0);
        sum1 = vfmaq_f32(sum1, left1, right1);
        index += 8;
    }
    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));
    if index < left.len() {
        result += dot_f16_table(&left[index..], &right_le_bytes[index * 2..]);
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_f16_avx2(left: &[f32], right_le_bytes: &[u8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut index = 0;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    while index + 16 <= left.len() {
        // SAFETY: the loop bound covers 16 F16 inputs and the intrinsic
        // explicitly permits unaligned loads.
        let half0 =
            unsafe { _mm_loadu_si128(right_le_bytes.as_ptr().add(index * 2).cast::<__m128i>()) };
        // SAFETY: the same loop bound covers the second eight-value load.
        let half1 = unsafe {
            _mm_loadu_si128(
                right_le_bytes
                    .as_ptr()
                    .add((index + 8) * 2)
                    .cast::<__m128i>(),
            )
        };
        let right0 = _mm256_cvtph_ps(half0);
        let right1 = _mm256_cvtph_ps(half1);
        // SAFETY: the loop bound covers the first eight F32 inputs and this
        // intrinsic explicitly permits unaligned loads.
        let left0 = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        // SAFETY: the same loop bound covers the second eight-value load.
        let left1 = unsafe { _mm256_loadu_ps(left.as_ptr().add(index + 8)) };
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(left0, right0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(left1, right1));
        index += 16;
    }
    let sum = _mm256_add_ps(sum0, sum1);
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` has space for all eight values and the store permits
    // unaligned output.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().sum();
    if index < left.len() {
        result += dot_f16_table(&left[index..], &right_le_bytes[index * 2..]);
    }
    result
}

#[inline]
fn dot_portable(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0f32; 4];
    let chunks = left.len() / 4;
    for index in 0..chunks {
        let offset = index * 4;
        sums[0] += left[offset] * right[offset];
        sums[1] += left[offset + 1] * right[offset + 1];
        sums[2] += left[offset + 2] * right[offset + 2];
        sums[3] += left[offset + 3] * right[offset + 3];
    }
    let mut result = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for index in chunks * 4..left.len() {
        result += left[index] * right[index];
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut index = 0;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    while index + 8 <= left.len() {
        // SAFETY: the loop bounds guarantee eight readable floats per slice.
        let l0 = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        // SAFETY: the same loop bound applies to `right`.
        let r0 = unsafe { vld1q_f32(right.as_ptr().add(index)) };
        // SAFETY: the same loop bound covers the second four-value load.
        let l1 = unsafe { vld1q_f32(left.as_ptr().add(index + 4)) };
        // SAFETY: the same loop bound covers the second load from `right`.
        let r1 = unsafe { vld1q_f32(right.as_ptr().add(index + 4)) };
        sum0 = vfmaq_f32(sum0, l0, r0);
        sum1 = vfmaq_f32(sum1, l1, r1);
        index += 8;
    }
    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));
    while index < left.len() {
        result += left[index] * right[index];
        index += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut index = 0;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    while index + 16 <= left.len() {
        // SAFETY: the loop bounds guarantee sixteen readable floats per slice;
        // unaligned loads accept Vec's ordinary alignment.
        let l0 = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        // SAFETY: the same loop bound applies to `right`.
        let r0 = unsafe { _mm256_loadu_ps(right.as_ptr().add(index)) };
        // SAFETY: the same loop bound covers the second eight-value load.
        let l1 = unsafe { _mm256_loadu_ps(left.as_ptr().add(index + 8)) };
        // SAFETY: the same loop bound covers the second load from `right`.
        let r1 = unsafe { _mm256_loadu_ps(right.as_ptr().add(index + 8)) };
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(l0, r0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(l1, r1));
        index += 16;
    }
    let sum = _mm256_add_ps(sum0, sum1);
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` has space for all eight lanes and unaligned stores are
    // explicitly supported.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().sum();
    while index < left.len() {
        result += left[index] * right[index];
        index += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_portable_for_awkward_lengths() {
        for length in 0..67 {
            let left: Vec<_> = (0..length).map(|i| i as f32 * 0.125 - 2.0).collect();
            let right: Vec<_> = (0..length).map(|i| i as f32 * -0.25 + 3.0).collect();
            let expected = dot_portable(&left, &right);
            let actual = dot(&left, &right);
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-5);
        }
    }

    #[test]
    fn f16_dot_matches_decoded_f32() {
        for length in 0..67 {
            let left: Vec<_> = (0..length).map(|i| i as f32 * 0.03125 - 1.0).collect();
            let values: Vec<_> = (0..length).map(|i| i as f32 * -0.0625 + 2.0).collect();
            let mut encoded = Vec::with_capacity(length * 2);
            let mut decoded = Vec::with_capacity(length);
            for value in values {
                let bits = crate::codec::f32_to_f16(value);
                encoded.extend_from_slice(&bits.to_le_bytes());
                decoded.push(crate::codec::f16_to_f32(bits));
            }
            let expected = dot(&left, &decoded);
            let actual = dot_f16(&left, &encoded);
            assert!((actual - expected).abs() <= expected.abs().max(1.0) * 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "dot-product dimensions differ")]
    fn dot_rejects_mismatched_dimensions_in_release_builds() {
        let _ = dot(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "F16 dot-product dimensions differ")]
    fn f16_dot_rejects_mismatched_dimensions_in_release_builds() {
        let _ = dot_f16(&[1.0, 2.0], &[0, 0]);
    }
}
