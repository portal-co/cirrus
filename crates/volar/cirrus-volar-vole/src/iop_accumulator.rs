//! `volar-iop` implementation of the verifier AND-gate hook.
//!
//! This provider intentionally folds one explicitly selected VOLE lane. Its
//! challenge callback must be transcript-bound by the caller; a gate index is
//! not itself a cryptographic challenge source.

use core::marker::PhantomData;

use hybrid_array::{Array, ArraySize};
use volar_iop::{
    field::{Beta8, Beta16, Beta32, Beta64, Ext, Gf128},
    fold::{IopAccumulator, fold_gate},
};
use volar_spec::{
    field::Galois,
    vole::{Delta, Q},
};

use crate::VoleVerifierHook;

/// Canonically embeds a VOLE scalar into Volar's `GF(2^128)` IOP field.
pub trait IopLift {
    /// Embed `self` as the tower field's base component.
    fn iop_lift(&self) -> Gf128;
}

impl IopLift for Galois {
    fn iop_lift(&self) -> Gf128 {
        let gf16 = Ext::<Galois, Beta8>::new(*self, Galois(0));
        let gf32 = Ext::<_, Beta16>::new(gf16, Default::default());
        let gf64 = Ext::<_, Beta32>::new(gf32, Default::default());
        Ext::<_, Beta64>::new(gf64, Default::default())
    }
}

/// Folds one selected lane of each verifier AND into Volar's IOP accumulator.
///
/// The challenge callback is called once per successful gate observation. The
/// selected lane is deliberately explicit: this does not aggregate parallel
/// VOLE lanes.
pub struct IopAccumulatorHook<T, C> {
    lane: usize,
    challenge: C,
    accumulator: IopAccumulator<Gf128>,
    marker: PhantomData<fn() -> T>,
}

impl<T, C> IopAccumulatorHook<T, C> {
    /// Construct a lane-specific accumulator hook.
    pub fn new(lane: usize, challenge: C) -> Self {
        Self {
            lane,
            challenge,
            accumulator: IopAccumulator::fresh(),
            marker: PhantomData,
        }
    }

    /// Return the lane selected for folding.
    pub fn lane(&self) -> usize {
        self.lane
    }

    /// Borrow the current fixed-size accumulator.
    pub fn accumulator(&self) -> &IopAccumulator<Gf128> {
        &self.accumulator
    }

    /// Consume the hook and return its accumulator.
    pub fn into_accumulator(self) -> IopAccumulator<Gf128> {
        self.accumulator
    }
}

impl<N, T, C> VoleVerifierHook<N, T> for IopAccumulatorHook<T, C>
where
    N: ArraySize,
    T: IopLift,
    C: FnMut(usize) -> Gf128,
{
    fn on_and(
        &mut self,
        gate_index: usize,
        delta: &Delta<N, T>,
        q_a: &Q<N, T>,
        q_b: &Q<N, T>,
        q_c: &Q<N, T>,
        hat: &Array<T, N>,
    ) {
        assert!(self.lane < N::USIZE, "IOP accumulator lane is out of range");
        let lane = self.lane;
        self.accumulator = fold_gate(
            core::mem::take(&mut self.accumulator),
            q_a.q[lane].iop_lift(),
            q_b.q[lane].iop_lift(),
            q_c.q[lane].iop_lift(),
            delta.delta[lane].iop_lift(),
            hat[lane].iop_lift(),
            (self.challenge)(gate_index),
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use cipher::consts::U1;
    use cirrus_core::ContextWithBitAnd;
    use hybrid_array::Array;
    use volar_iop::{field::Field, fold::and_check_r1cs};
    use volar_spec::vole::Delta;

    use super::*;
    use crate::VoleVerifierContext;

    #[test]
    fn verifier_context_folds_the_observed_and_gate() {
        let delta = Galois(7);
        let q_a = Galois(3);
        let q_b = Galois(5);
        let q_c = Galois(11);
        // Characteristic two: Vhat = Kc * Delta + Ka * Kb.
        let hat = q_c * delta + q_a * q_b;
        let hook = IopAccumulatorHook::new(0, |_| Gf128::ONE);
        let mut context = VoleVerifierContext {
            delta: Delta {
                delta: Array::<Galois, U1>::from_fn(|_| delta),
            },
            hats: vec![Array::<Galois, U1>::from_fn(|_| hat)].into_iter(),
            hook,
            gate_index: 0,
        };
        let output = context
            .bitand(
                Q {
                    q: Array::<Galois, U1>::from_fn(|_| q_a),
                },
                Q {
                    q: Array::<Galois, U1>::from_fn(|_| q_b),
                },
            )
            .unwrap();

        assert_eq!(output.q[0], q_c);
        assert_eq!(context.gate_index, 1);
        let (witness, error, u) = context.hook.accumulator().witness().unwrap();
        assert!(and_check_r1cs::<Gf128>().is_satisfied_relaxed(witness, error, u));
    }
}
