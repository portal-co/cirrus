//! Radix-2 two-adic discrete Fourier transform over KoalaBear.
//!
//! The convention mirrors the pinned upstream Plonky3 `Radix2DFTSmallBatch`:
//! treating each column of a row-major matrix as polynomial coefficients,
//! `dft` evaluates it over the unique multiplicative subgroup of order
//! `height`, so `output[j] = sum_i input[i] * g^{i*j}` with
//! `g = KoalaBear::two_adic_generator(log2(height))`. No coset shift and no
//! normalization factor are applied. Extension-field batches use the same
//! base-field twiddles via `mul_base`.

use alloc::vec::Vec;

use crate::{KoalaBear, QuinticExtension};

/// Largest two-adic subgroup supported by the KoalaBear field (`2^24 | p - 1`).
pub const KOALABEAR_TWO_ADICITY: usize = 24;

/// Generator of the unique KoalaBear multiplicative subgroup of order
/// `2^bits`, matching upstream's `TwoAdicField::two_adic_generator`.
///
/// With `g = 3` a primitive root, this is `g^{(p - 1) / 2^bits}`.
pub fn two_adic_generator(bits: usize) -> KoalaBear {
    assert!(
        bits <= KOALABEAR_TWO_ADICITY,
        "KoalaBear two-adicity is {KOALABEAR_TWO_ADICITY}"
    );
    KoalaBear::from_u64(3).pow((crate::KOALABEAR_MODULUS as u64 - 1) >> bits)
}

/// Why a DFT request is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DftError {
    /// The matrix height is not a positive power of two within the field
    /// two-adicity.
    InvalidHeight,
    /// The value count is not `height * width`.
    InvalidWidth,
}

impl core::fmt::Display for DftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeight => {
                f.write_str("DFT height must be a power of two within two-adicity")
            }
            Self::InvalidWidth => f.write_str("value count must equal height times width"),
        }
    }
}

impl core::error::Error for DftError {}

fn checked_log_height(height: usize) -> Result<usize, DftError> {
    if height == 0 || !height.is_power_of_two() {
        return Err(DftError::InvalidHeight);
    }
    let log = height.ilog2() as usize;
    if log > KOALABEAR_TWO_ADICITY {
        return Err(DftError::InvalidHeight);
    }
    Ok(log)
}

/// In-place radix-2 decimation-in-time FFT over one coefficient slice.
///
/// After the call, `values[j] = sum_i input[i] * g^{i*j}` where `g` is the
/// generator of the subgroup of order `values.len()`.
fn fft_in_place_base(values: &mut [KoalaBear], generator: KoalaBear) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let bits = n.ilog2();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            values.swap(i, j);
        }
    }
    // Butterfly stages.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // `step` is the primitive `len`-th root: g^{n/len}.
        let step = generator.pow((n / len) as u64);
        for chunk in values.chunks_exact_mut(len) {
            let mut w = KoalaBear::ONE;
            for j in 0..half {
                let (lo, hi) = (chunk[j], chunk[j + half] * w);
                chunk[j] = lo + hi;
                chunk[j + half] = lo - hi;
                w = w * step;
            }
        }
        len <<= 1;
    }
}

/// The same butterfly for extension-field values with base-field twiddles.
fn fft_in_place_ext(values: &mut [QuinticExtension], generator: KoalaBear) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    let bits = n.ilog2();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            values.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = generator.pow((n / len) as u64);
        for chunk in values.chunks_exact_mut(len) {
            let mut w = KoalaBear::ONE;
            for j in 0..half {
                let (lo, hi) = (chunk[j], chunk[j + half].mul_base(w));
                chunk[j] = lo + hi;
                chunk[j + half] = lo - hi;
                w = w * step;
            }
        }
        len <<= 1;
    }
}

/// Compute the batch DFT of a row-major matrix in place: each of the
/// `width` columns is transformed independently over the subgroup of order
/// `height`.
///
/// `values` holds `height * width` entries in row-major order.
pub fn dft_batch_base(
    values: &mut [KoalaBear],
    height: usize,
    width: usize,
) -> Result<(), DftError> {
    let log = checked_log_height(height)?;
    if width == 0 || values.len() != height * width {
        return Err(DftError::InvalidWidth);
    }
    let generator = two_adic_generator(log);
    let mut column = alloc::vec![KoalaBear::ZERO; height];
    for c in 0..width {
        for (r, slot) in column.iter_mut().enumerate() {
            *slot = values[r * width + c];
        }
        fft_in_place_base(&mut column, generator);
        for (r, &v) in column.iter().enumerate() {
            values[r * width + c] = v;
        }
    }
    Ok(())
}

/// Extension-field batch DFT with base-field twiddles, matching upstream's
/// `dft_algebra_batch`.
pub fn dft_batch_ext(
    values: &mut [QuinticExtension],
    height: usize,
    width: usize,
) -> Result<(), DftError> {
    let log = checked_log_height(height)?;
    if width == 0 || values.len() != height * width {
        return Err(DftError::InvalidWidth);
    }
    let generator = two_adic_generator(log);
    let mut column = alloc::vec![QuinticExtension::ZERO; height];
    for c in 0..width {
        for (r, slot) in column.iter_mut().enumerate() {
            *slot = values[r * width + c];
        }
        fft_in_place_ext(&mut column, generator);
        for (r, &v) in column.iter().enumerate() {
            values[r * width + c] = v;
        }
    }
    Ok(())
}

/// Convenience wrapper: DFT of a single base-field vector.
pub fn dft_base(values: &[KoalaBear]) -> Result<Vec<KoalaBear>, DftError> {
    let mut out = values.to_vec();
    dft_batch_base(&mut out, values.len(), 1)?;
    Ok(out)
}
