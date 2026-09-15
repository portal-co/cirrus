//! Independent scalar Poseidon2 permutation for the upstream KoalaBear profile.
//!
//! This implements the width-16 and width-24 configurations used by the pinned
//! Spartan-WHIR Poseidon engine. It is intentionally scalar and allocation
//! free; SIMD and packed implementations are later performance work.

use crate::KoalaBear;
use crate::poseidon2_constants::*;

/// Poseidon2 width used for the upstream duplex challenger and node compressor.
pub const POSEIDON2_WIDTH_16: usize = 16;
/// Poseidon2 width used for the upstream field-element sponge.
pub const POSEIDON2_WIDTH_24: usize = 24;
/// Number of initial and terminal full rounds on each side.
pub const POSEIDON2_HALF_FULL_ROUNDS: usize = 4;
/// Width-16 partial-round count.
pub const POSEIDON2_PARTIAL_ROUNDS_16: usize = 20;
/// Width-24 partial-round count.
pub const POSEIDON2_PARTIAL_ROUNDS_24: usize = 23;

/// Scalar width-16 Poseidon2 permutation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Poseidon2KoalaBear16;

/// Scalar width-24 Poseidon2 permutation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Poseidon2KoalaBear24;

impl Poseidon2KoalaBear16 {
    /// Apply the full Poseidon2 permutation in place.
    pub fn permute_mut(state: &mut [KoalaBear; POSEIDON2_WIDTH_16]) {
        external_initial(state, &KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL);
        for &constant in &KOALABEAR_POSEIDON2_RC_16_INTERNAL {
            state[0] = (state[0] + constant).cube();
            internal_linear_layer(state, internal_layer_16);
        }
        external_terminal(state, &KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL);
    }
}

impl Poseidon2KoalaBear24 {
    /// Apply the full Poseidon2 permutation in place.
    pub fn permute_mut(state: &mut [KoalaBear; POSEIDON2_WIDTH_24]) {
        external_initial(state, &KOALABEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL);
        for &constant in &KOALABEAR_POSEIDON2_RC_24_INTERNAL {
            state[0] = (state[0] + constant).cube();
            internal_linear_layer(state, internal_layer_24);
        }
        external_terminal(state, &KOALABEAR_POSEIDON2_RC_24_EXTERNAL_FINAL);
    }
}

fn external_initial<const WIDTH: usize>(
    state: &mut [KoalaBear; WIDTH],
    constants: &[[KoalaBear; WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
) {
    mds_light_permutation(state);
    external_terminal(state, constants);
}

fn external_terminal<const WIDTH: usize>(
    state: &mut [KoalaBear; WIDTH],
    constants: &[[KoalaBear; WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
) {
    for round in constants {
        for (value, constant) in state.iter_mut().zip(round) {
            *value = (*value + *constant).cube();
        }
        mds_light_permutation(state);
    }
}

impl KoalaBear {
    const fn cube(self) -> Self {
        self.square().mul(self)
    }
}

/// Multiply by the Poseidon2 4x4 MDS matrix:
/// `[2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]`.
fn apply_mat4(x: &mut [KoalaBear; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    x[3] = t01233 + x[0].double();
    x[1] = t01123 + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

fn mds_light_permutation<const WIDTH: usize>(state: &mut [KoalaBear; WIDTH]) {
    assert!(WIDTH % 4 == 0);
    for chunk in state.chunks_exact_mut(4) {
        apply_mat4(chunk.try_into().expect("four-element chunk"));
    }
    let mut sums = [KoalaBear::ZERO; 4];
    for (index, value) in state.iter().enumerate() {
        sums[index % 4] = sums[index % 4] + *value;
    }
    for (index, value) in state.iter_mut().enumerate() {
        *value = *value + sums[index % 4];
    }
}

fn internal_linear_layer<const WIDTH: usize>(
    state: &mut [KoalaBear; WIDTH],
    internal_layer: fn(&mut [KoalaBear; WIDTH], KoalaBear),
) {
    let mut part_sum = KoalaBear::ZERO;
    for value in &state[1..] {
        part_sum = part_sum + *value;
    }
    let full_sum = part_sum + state[0];
    state[0] = part_sum - state[0];
    internal_layer(state, full_sum);
}

fn internal_layer_16(state: &mut [KoalaBear; 16], sum: KoalaBear) {
    state[1] = state[1] + sum;
    state[2] = state[2].double() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = state[4].double() + state[4] + sum;
    state[5] = state[5].double().double() + sum;
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].double() + state[7]);
    state[8] = sum - state[8].double().double();
    state[9] = state[9].div_2exp_u64(8) + sum;
    state[10] = state[10].div_2exp_u64(3) + sum;
    state[11] = state[11].div_2exp_u64(24) + sum;
    state[12] = sum - state[12].div_2exp_u64(8);
    state[13] = sum - state[13].div_2exp_u64(3);
    state[14] = sum - state[14].div_2exp_u64(4);
    state[15] = sum - state[15].div_2exp_u64(24);
}

fn internal_layer_24(state: &mut [KoalaBear; 24], sum: KoalaBear) {
    state[1] = state[1] + sum;
    state[2] = state[2].double() + sum;
    state[3] = state[3].halve() + sum;
    state[4] = state[4].double() + state[4] + sum;
    state[5] = state[5].double().double() + sum;
    state[6] = sum - state[6].halve();
    state[7] = sum - (state[7].double() + state[7]);
    state[8] = sum - state[8].double().double();
    state[9] = state[9].div_2exp_u64(8) + sum;
    state[10] = state[10].div_2exp_u64(2) + sum;
    state[11] = state[11].div_2exp_u64(3) + sum;
    state[12] = state[12].div_2exp_u64(4) + sum;
    state[13] = state[13].div_2exp_u64(5) + sum;
    state[14] = state[14].div_2exp_u64(6) + sum;
    state[15] = state[15].div_2exp_u64(24) + sum;
    state[16] = sum - state[16].div_2exp_u64(8);
    state[17] = sum - state[17].div_2exp_u64(3);
    state[18] = sum - state[18].div_2exp_u64(4);
    state[19] = sum - state[19].div_2exp_u64(5);
    state[20] = sum - state[20].div_2exp_u64(6);
    state[21] = sum - state[21].div_2exp_u64(7);
    state[22] = sum - state[22].div_2exp_u64(9);
    state[23] = sum - state[23].div_2exp_u64(24);
}
