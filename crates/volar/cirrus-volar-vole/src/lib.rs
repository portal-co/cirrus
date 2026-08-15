#![no_std]
#![warn(missing_docs)]

//! QuickSilver-style VOLE-ZK `cirrus_core::Context` backend pair, built on
//! `volar_spec::vole::{prove, setup}`'s online-phase AND-gate primitives.
//!
//! # Scope
//!
//! This is a same-process/ideal-functionality SIMULATION of both roles (see
//! `volar_spec::vole::setup::vole_commit_bit`'s own doc: "In a real
//! protocol the prover and verifier would each see only one half; this
//! single-process API is used to drive the ideal functionality from
//! tests."). It validates the arithmetization/soundness logic; it is NOT a
//! real 2-party networked protocol.
//!
//! `Op::Create(bool)` (a compile-time-known constant, never a witness --
//! see this workspace's other backends for the same convention) maps to a
//! deterministic constant share on both sides, needing no correlated
//! randomness. Genuine witnesses (`program.inputs`) must instead be
//! pre-committed via `volar_spec::vole::setup::vole_commit_bit` against a
//! shared `IdealCot`, splitting the returned `(Vope, Q)` pair into the
//! prover's and verifier's separate input vectors, before calling
//! `cirrus_recompile_rt::execute` -- the VOLE analogue of how
//! `cirrus-r1cs-backend`'s `ProgramCircuit` pre-allocates
//! `Boolean::new_witness` for its own `program.inputs`.

use core::{
    convert::Infallible,
    fmt,
    ops::{Add, Mul},
};

use cipher::consts::U1;
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue, HasError, Pusher,
};
use hybrid_array::{Array, ArraySize};
use volar_spec::{
    field::Invert,
    vole::{Delta, Q, VoleArray, Vope, prove::vole_and_prover_step, setup::derive_and_q},
};

/// Prover-side context: `Wrapped = Vope<N, T, U1>`. Streams one `hat` per
/// AND gate to [`Self::hats`], in circuit order -- the transcript the
/// verifier must consume in the same order via [`VoleVerifierContext`].
pub struct VoleProverContext<'a, 'b, N: VoleArray<T>, T> {
    /// Ordered streaming destination for one `hat` per AND gate.
    pub hats: &'a mut (dyn Pusher<Array<T, N>> + 'b),
    /// Lifts a known-constant bit to the field `T`, used only by
    /// `Op::Create`. Genuine witnesses never go through this -- they are
    /// pre-committed via `vole_commit_bit` before `execute` runs.
    pub bit_to_t: fn(bool) -> T,
}

impl<N: VoleArray<T>, T> HasError for VoleProverContext<'_, '_, N, T> {
    type Error = Infallible;
}
impl<N: VoleArray<T>, T> ContextWithValue<bool> for VoleProverContext<'_, '_, N, T> {
    type Wrapped = Vope<N, T, U1>;
}
impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreate<bool> for VoleProverContext<'_, '_, N, T> {
    fn create(&mut self, val: bool) -> Result<Vope<N, T, U1>, Infallible> {
        let t = (self.bit_to_t)(val);
        Ok(Vope {
            u: Array::<Array<T, N>, U1>::from_fn(|_| Array::<T, N>::from_fn(|_| t.clone())),
            v: Array::<T, N>::from_fn(|_| T::default()),
        })
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Default> ContextWithBitXor<bool>
    for VoleProverContext<'_, '_, N, T>
{
    fn bitxor(&mut self, a: Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<Vope<N, T, U1>, Infallible> {
        // Free: `Vope`'s own `Add` impl (volar_spec::vole::vope).
        Ok(a + b)
    }
    fn bitxor_assign(&mut self, a: &mut Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<(), Infallible> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithBitAnd<bool>
    for VoleProverContext<'_, '_, N, T>
{
    fn bitand(&mut self, a: Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<Vope<N, T, U1>, Infallible> {
        let (c, hat) = vole_and_prover_step(a, b);
        self.hats.push(hat);
        Ok(c)
    }
    fn bitand_assign(&mut self, a: &mut Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<(), Infallible> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithBitOr<bool>
    for VoleProverContext<'_, '_, N, T>
{
    fn bitor(&mut self, a: Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<Vope<N, T, U1>, Infallible> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }
    fn bitor_assign(&mut self, a: &mut Vope<N, T, U1>, b: Vope<N, T, U1>) -> Result<(), Infallible> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithMux<bool>
    for VoleProverContext<'_, '_, N, T>
{
    fn mux(
        &mut self,
        cond: Vope<N, T, U1>,
        then: Vope<N, T, U1>,
        r#else: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let diff = self.bitxor(then, r#else.clone())?;
        let masked = self.bitand(cond, diff)?;
        self.bitxor(r#else, masked)
    }
}

/// An error while replaying a `hat` transcript on the verifier side.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoleVerifyError {
    /// An AND gate required a `hat` after the transcript iterator ended.
    HatExhausted,
}
impl fmt::Display for VoleVerifyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HatExhausted => formatter.write_str("hat transcript iterator is exhausted"),
        }
    }
}
impl core::error::Error for VoleVerifyError {}

/// Verifier-side context: `Wrapped = Q<N, T>`. Pulls one `hat` per AND gate
/// from [`Self::hats`]. [`derive_and_q`] propagates every intermediate
/// AND-gate share unconditionally -- it never rejects. The actual
/// QuickSilver soundness check (`vole_and_verifier_check`) is a
/// caller-driven step performed only at claimed/revealed output wires, not
/// part of the per-gate `Context` operations -- see this crate's own
/// round-trip test.
pub struct VoleVerifierContext<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> {
    /// The verifier's secret global offset.
    pub delta: Delta<N, T>,
    /// The ordered source of `hat` values, one per AND gate.
    pub hats: I,
}

impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> HasError for VoleVerifierContext<N, T, I> {
    type Error = VoleVerifyError;
}
impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> ContextWithValue<bool>
    for VoleVerifierContext<N, T, I>
{
    type Wrapped = Q<N, T>;
}
impl<N: ArraySize, T: Clone + Default, I: Iterator<Item = Array<T, N>>> ContextWithCreate<bool>
    for VoleVerifierContext<N, T, I>
{
    fn create(&mut self, val: bool) -> Result<Q<N, T>, VoleVerifyError> {
        Ok(Q {
            q: if val {
                self.delta.delta.clone()
            } else {
                Array::<T, N>::from_fn(|_| T::default())
            },
        })
    }
}
impl<N: ArraySize, T: Clone + Add<Output = T>, I: Iterator<Item = Array<T, N>>> ContextWithBitXor<bool>
    for VoleVerifierContext<N, T, I>
{
    fn bitxor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        // `Q` has no dedicated `Add` impl; this pointwise construction is
        // the same one volar_spec::vole::bridge's own verifier helpers
        // compute inline.
        Ok(Q {
            q: Array::<T, N>::from_fn(|i| a.q[i].clone() + b.q[i].clone()),
        })
    }
    fn bitxor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
> ContextWithBitAnd<bool> for VoleVerifierContext<N, T, I>
{
    fn bitand(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let hat = self.hats.next().ok_or(VoleVerifyError::HatExhausted)?;
        Ok(derive_and_q(&self.delta, &a, &b, &hat))
    }
    fn bitand_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
> ContextWithBitOr<bool> for VoleVerifierContext<N, T, I>
{
    fn bitor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }
    fn bitor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}
impl<
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
> ContextWithMux<bool> for VoleVerifierContext<N, T, I>
{
    fn mux(&mut self, cond: Q<N, T>, then: Q<N, T>, r#else: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let diff = self.bitxor(then, r#else.clone())?;
        let masked = self.bitand(cond, diff)?;
        self.bitxor(r#else, masked)
    }
}
