//! Poseidon hash and compression profiles used by upstream Spartan-WHIR.

use crate::{
    KoalaBear, POSEIDON2_WIDTH_16, POSEIDON2_WIDTH_24, Poseidon2KoalaBear16, Poseidon2KoalaBear24,
};

/// Digest length in base-field elements.
pub const POSEIDON_DIGEST_ELEMENTS: usize = 8;
/// Rate of the upstream Poseidon2-24 field-element sponge.
pub const POSEIDON_FIELD_HASH_RATE: usize = 16;
/// Rate of the upstream Poseidon2-16 duplex challenger.
pub const POSEIDON_CHALLENGER_RATE: usize = 8;

/// One Poseidon digest.
pub type PoseidonDigest = [KoalaBear; POSEIDON_DIGEST_ELEMENTS];

/// Hash fixed-length field-element data with the upstream
/// `PaddingFreeSponge<Poseidon24, 24, 16, 8>` profile.
///
/// This is overwrite-mode and has no variable-length padding. Callers must
/// bind lengths and row layouts outside this function whenever variable-length
/// data is possible.
pub fn poseidon_hash_fixed(input: &[KoalaBear]) -> PoseidonDigest {
    let mut state = [KoalaBear::ZERO; POSEIDON2_WIDTH_24];
    let mut chunks = input.chunks_exact(POSEIDON_FIELD_HASH_RATE);
    for chunk in &mut chunks {
        state[..POSEIDON_FIELD_HASH_RATE].copy_from_slice(chunk);
        Poseidon2KoalaBear24::permute_mut(&mut state);
    }
    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        state[..remainder.len()].copy_from_slice(remainder);
        Poseidon2KoalaBear24::permute_mut(&mut state);
    }
    state[..POSEIDON_DIGEST_ELEMENTS]
        .try_into()
        .expect("digest slice length")
}

/// Compress two digests with the upstream
/// `TruncatedPermutation<Poseidon16, 2, 8, 16>` node compressor.
pub fn poseidon_compress2(input: [PoseidonDigest; 2]) -> PoseidonDigest {
    let mut state = [KoalaBear::ZERO; POSEIDON2_WIDTH_16];
    state[..POSEIDON_DIGEST_ELEMENTS].copy_from_slice(&input[0]);
    state[POSEIDON_DIGEST_ELEMENTS..2 * POSEIDON_DIGEST_ELEMENTS].copy_from_slice(&input[1]);
    Poseidon2KoalaBear16::permute_mut(&mut state);
    state[..POSEIDON_DIGEST_ELEMENTS]
        .try_into()
        .expect("digest slice length")
}
