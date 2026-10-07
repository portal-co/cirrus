// @ai: assisted
//! Cut-and-choose for malicious-garbler security, built **generically as a
//! wrapper** over the cirrus `Context` — the shape of the old, scrapped
//! `cirrus-vole` wrapper (`Vole<Wrapped>`), but for garbled circuits.
//!
//! The generic piece is [`Broadcast`]: a `Context` that wraps **N** inner
//! `Context`s and runs them **in parallel** (each Boolean op is broadcast to all
//! N). On top of it sits the cut-and-choose protocol:
//!
//! 1. **Garble N copies** of a `Program`, each with an independent `GlobalSecret`
//!    and label chain derived from a per-copy seed. The garbler **commits** to
//!    each copy's table stream + input/output label bases up front.
//! 2. **Challenge**: the evaluator picks a random open subset.
//! 3. **Open**: for each opened copy the garbler reveals the seed; the evaluator
//!    re-garbles deterministically and checks the tables match the commitment.
//! 4. **Use**: the remaining copies are evaluated; their outputs must **agree**.
//!
//! A malicious garbler that corrupts a copy is caught when that copy is opened
//! (table mismatch) or when the use set disagrees.

use alloc::vec::Vec;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue, HasError,
};

// ============================================================================
// Broadcast — run N Contexts in parallel (the generic wrapper)
// ============================================================================

/// A `Context` wrapping N inner `Context`s, broadcasting every Boolean op.
///
/// The wire is `Vec<inner>` — one inner wire per copy — mirroring the old
/// `cirrus-vole` wrapper's `(inner, inner)`, generalized to N. This is the
/// "run N `Context`s in parallel" primitive the cut-and-choose harness drives.
pub struct Broadcast<C>(pub Vec<C>);

impl<C: HasError> HasError for Broadcast<C> {
    type Error = C::Error;
}

impl<Val, C: ContextWithValue<Val>> ContextWithValue<Val> for Broadcast<C> {
    type Wrapped = Vec<C::Wrapped>;
}

impl<Val: Clone, C: ContextWithCreate<Val>> ContextWithCreate<Val> for Broadcast<C> {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        self.0.iter_mut().map(|c| c.create(val.clone())).collect()
    }
}

macro_rules! broadcast_binop {
    ($trait:ident, $method:ident, $assign:ident) => {
        impl<Val, C: $trait<Val>> $trait<Val> for Broadcast<C> {
            fn $method(
                &mut self,
                a: Self::Wrapped,
                b: Self::Wrapped,
            ) -> Result<Self::Wrapped, Self::Error> {
                self.0
                    .iter_mut()
                    .zip(a)
                    .zip(b)
                    .map(|((c, a), b)| c.$method(a, b))
                    .collect()
            }
            fn $assign(
                &mut self,
                a: &mut Self::Wrapped,
                b: Self::Wrapped,
            ) -> Result<(), Self::Error> {
                for ((c, a), b) in self.0.iter_mut().zip(a.iter_mut()).zip(b) {
                    c.$assign(a, b)?;
                }
                Ok(())
            }
        }
    };
}
broadcast_binop!(ContextWithBitAnd, bitand, bitand_assign);
broadcast_binop!(ContextWithBitOr, bitor, bitor_assign);
broadcast_binop!(ContextWithBitXor, bitxor, bitxor_assign);

impl<Val, C: ContextWithMux<Val>> ContextWithMux<Val> for Broadcast<C>
where
    C: ContextWithValue<bool>,
{
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        self.0
            .iter_mut()
            .zip(cond)
            .zip(then)
            .zip(r#else)
            .map(|(((c, k), t), e)| c.mux(k, t, e))
            .collect()
    }
}
