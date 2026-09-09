//! Runtime observation hooks for verifier-side VOLE AND gates.
//!
//! A hook observes the exact values used by [`crate::VoleVerifierContext`] to
//! derive an AND output. It cannot alter that output or the hat transcript.

use hybrid_array::{Array, ArraySize};
use volar_spec::vole::{Delta, Q};

/// Observes a successfully replayed verifier-side AND gate.
///
/// `gate_index` is zero based and follows the verifier's hat-consumption
/// order. Implementations must not retain the supplied references.
pub trait VoleVerifierHook<N: ArraySize, T> {
    /// Record the gate relation `q_a * q_b + hat = q_c * delta`.
    fn on_and(
        &mut self,
        gate_index: usize,
        delta: &Delta<N, T>,
        q_a: &Q<N, T>,
        q_b: &Q<N, T>,
        q_c: &Q<N, T>,
        hat: &Array<T, N>,
    );
}

/// The default verifier hook, which records nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopVoleVerifierHook;

impl<N: ArraySize, T> VoleVerifierHook<N, T> for NoopVoleVerifierHook {
    fn on_and(
        &mut self,
        _gate_index: usize,
        _delta: &Delta<N, T>,
        _q_a: &Q<N, T>,
        _q_b: &Q<N, T>,
        _q_c: &Q<N, T>,
        _hat: &Array<T, N>,
    ) {
    }
}
