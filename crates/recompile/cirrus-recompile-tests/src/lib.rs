//! Two extra `cirrus-recompile-rt` pinned backends, defined here rather than
//! in `cirrus-recompile-rt` itself, as a worked example that
//! `define_pinned_backend!` is genuinely usable from *any* crate: this one
//! depends on `cirrus-recompile-rt` and `cirrus-garbled-circuit` like any
//! other downstream consumer would, and calls
//! `cirrus_recompile_rt::define_pinned_backend!` the exact same way
//! `cirrus-recompile-rt` calls it internally for its own `plaintext`
//! backend.
//!
//! [`gc_backend::GcBackend`] and [`eval_backend::EvalBackend`] wrap
//! `cirrus-garbled-circuit`'s `GC` and `Evaluator`, adding the two
//! operations a garbler/evaluator's own API deliberately omits (`create`,
//! `mux`) so each satisfies the same [`cirrus_core`] `Context` trait bounds
//! every pinned backend needs. Neither wrapper changes `GC`/`Evaluator`'s
//! own semantics; a caller who wants direct access to the wrapped value can
//! always reach it through the public `gc`/`evaluator` field.
//!
//! See `tests/multi_backend.rs` for these backends actually compiled into
//! and run from real, `rustc`-built Rust source via
//! `cirrus_rust_codegen::BackendTarget`.

use core::array;
use core::convert::Infallible;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue, HasError,
};
use cirrus_garbled_circuit::{EvaluationError, Evaluator, GC, GarblingRecord, Label};
use digest::Digest;

/// Byte width of a garbled-wire label these backends use.
pub const LABEL_BYTES: usize = 16;
/// The digest these backends derive fresh wire labels with.
pub type LabelDigest = sha2::Sha256;

/// The garbler pinned backend: wraps [`GC`], adding wire-label
/// materialization ([`ContextWithCreate`]) and Boolean-select
/// ([`ContextWithMux`]).
///
/// A fresh wire's zero-label is derived by chaining `GC`'s own digest seed
/// -- the exact mechanism `GC`'s AND-gate output labels already use (see
/// `GC`'s `ContextWithMul<Bit>` impl) -- so `create` behaves like any other
/// wire-label derivation `GC` performs internally, not an ad hoc scheme
/// layered on top. `mux` is the same free-XOR identity
/// `cirrus_ert_core::select_word` already uses (`else ^ (cond & (then ^
/// else))`), so it needs no dedicated garbled primitive either.
pub struct GcBackend<'a, 'b, D: Digest, const N: usize> {
    /// The wrapped garbler.
    pub gc: GC<'a, 'b, D, N>,
}

impl<'a, 'b, D: Digest, const N: usize> GcBackend<'a, 'b, D, N> {
    /// Wrap a garbler.
    pub fn new(gc: GC<'a, 'b, D, N>) -> Self {
        Self { gc }
    }

    fn fresh_label(&mut self) -> Label<N> {
        self.gc.seed = D::digest(&self.gc.seed);
        let seed = self.gc.seed.clone();
        Label::new(array::from_fn(|i| seed[i]))
    }
}

impl<D: Digest, const N: usize> HasError for GcBackend<'_, '_, D, N> {
    type Error = Infallible;
}

impl<D: Digest, const N: usize> ContextWithValue<bool> for GcBackend<'_, '_, D, N> {
    type Wrapped = Label<N>;
}

impl<D: Digest, const N: usize> ContextWithCreate<bool> for GcBackend<'_, '_, D, N> {
    fn create(&mut self, _val: bool) -> Result<Label<N>, Infallible> {
        // A garbler tracks a wire through its zero-label only (see
        // `Label`'s docs); the actual bit is never visible to it, so the
        // value passed here is immaterial -- any fresh, unused label is a
        // valid new wire.
        Ok(self.fresh_label())
    }
}

impl<D: Digest, const N: usize> ContextWithBitAnd<bool> for GcBackend<'_, '_, D, N> {
    fn bitand(&mut self, a: Label<N>, b: Label<N>) -> Result<Label<N>, Infallible> {
        ContextWithBitAnd::bitand(&mut self.gc, a, b)
    }
    fn bitand_assign(&mut self, a: &mut Label<N>, b: Label<N>) -> Result<(), Infallible> {
        *a = self.bitand(*a, b)?;
        Ok(())
    }
}

impl<D: Digest, const N: usize> ContextWithBitOr<bool> for GcBackend<'_, '_, D, N> {
    fn bitor(&mut self, a: Label<N>, b: Label<N>) -> Result<Label<N>, Infallible> {
        ContextWithBitOr::bitor(&mut self.gc, a, b)
    }
    fn bitor_assign(&mut self, a: &mut Label<N>, b: Label<N>) -> Result<(), Infallible> {
        *a = self.bitor(*a, b)?;
        Ok(())
    }
}

impl<D: Digest, const N: usize> ContextWithBitXor<bool> for GcBackend<'_, '_, D, N> {
    fn bitxor(&mut self, a: Label<N>, b: Label<N>) -> Result<Label<N>, Infallible> {
        ContextWithBitXor::bitxor(&mut self.gc, a, b)
    }
    fn bitxor_assign(&mut self, a: &mut Label<N>, b: Label<N>) -> Result<(), Infallible> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}

impl<D: Digest, const N: usize> ContextWithMux<bool> for GcBackend<'_, '_, D, N> {
    fn mux(
        &mut self,
        cond: Label<N>,
        then: Label<N>,
        r#else: Label<N>,
    ) -> Result<Label<N>, Infallible> {
        let diff = ContextWithBitXor::bitxor(&mut self.gc, then, r#else)?;
        let masked = ContextWithBitAnd::bitand(&mut self.gc, cond, diff)?;
        ContextWithBitXor::bitxor(&mut self.gc, r#else, masked)
    }
}

/// A `Box<dyn Iterator<...>>` alias, spelled out once so
/// `cirrus_recompile_rt::define_pinned_backend!` (which needs a plain
/// `Ty<'lifetimes>` shape) has a single concrete generic parameter to
/// substitute rather than a raw `dyn` type.
pub type BoxedRecords<'a, const N: usize> =
    std::boxed::Box<dyn Iterator<Item = GarblingRecord<N>> + 'a>;

/// The evaluator pinned backend: wraps [`Evaluator`], adding
/// [`ContextWithMux`] (the same identity [`GcBackend`] uses) and a
/// placeholder [`ContextWithCreate`] impl that exists only to satisfy
/// `cirrus_recompile_rt::define_pinned_backend!`'s uniform bound across
/// every backend.
///
/// An evaluator cannot manufacture *any* wire's label from nothing -- it
/// only ever receives one via a garbler-revealed value or the garbling-table
/// stream -- so every slot an `Op::Create` would touch must instead be
/// supplied externally by the caller, exactly like this crate's convention
/// for `program.inputs` slots (see `cirrus_recompile_rt::execute`'s doc
/// comment). A program meant to run against this backend must therefore
/// mark every such constant slot as an input too, not just its genuine
/// secret inputs.
pub struct EvalBackend<'a, const N: usize> {
    /// The wrapped evaluator.
    pub evaluator: Evaluator<BoxedRecords<'a, N>, N>,
}

impl<'a, const N: usize> EvalBackend<'a, N> {
    /// Wrap an evaluator.
    pub fn new(evaluator: Evaluator<BoxedRecords<'a, N>, N>) -> Self {
        Self { evaluator }
    }
}

impl<const N: usize> HasError for EvalBackend<'_, N> {
    type Error = EvaluationError;
}

impl<const N: usize> ContextWithValue<bool> for EvalBackend<'_, N> {
    type Wrapped = [u8; N];
}

impl<const N: usize> ContextWithCreate<bool> for EvalBackend<'_, N> {
    fn create(&mut self, _val: bool) -> Result<[u8; N], EvaluationError> {
        Ok([0; N])
    }
}

impl<const N: usize> ContextWithBitAnd<bool> for EvalBackend<'_, N> {
    fn bitand(&mut self, a: [u8; N], b: [u8; N]) -> Result<[u8; N], EvaluationError> {
        ContextWithBitAnd::bitand(&mut self.evaluator, a, b)
    }
    fn bitand_assign(&mut self, a: &mut [u8; N], b: [u8; N]) -> Result<(), EvaluationError> {
        *a = self.bitand(*a, b)?;
        Ok(())
    }
}

impl<const N: usize> ContextWithBitOr<bool> for EvalBackend<'_, N> {
    fn bitor(&mut self, a: [u8; N], b: [u8; N]) -> Result<[u8; N], EvaluationError> {
        ContextWithBitOr::bitor(&mut self.evaluator, a, b)
    }
    fn bitor_assign(&mut self, a: &mut [u8; N], b: [u8; N]) -> Result<(), EvaluationError> {
        *a = self.bitor(*a, b)?;
        Ok(())
    }
}

impl<const N: usize> ContextWithBitXor<bool> for EvalBackend<'_, N> {
    fn bitxor(&mut self, a: [u8; N], b: [u8; N]) -> Result<[u8; N], EvaluationError> {
        ContextWithBitXor::bitxor(&mut self.evaluator, a, b)
    }
    fn bitxor_assign(&mut self, a: &mut [u8; N], b: [u8; N]) -> Result<(), EvaluationError> {
        *a = self.bitxor(*a, b)?;
        Ok(())
    }
}

impl<const N: usize> ContextWithMux<bool> for EvalBackend<'_, N> {
    fn mux(
        &mut self,
        cond: [u8; N],
        then: [u8; N],
        r#else: [u8; N],
    ) -> Result<[u8; N], EvaluationError> {
        let diff = ContextWithBitXor::bitxor(&mut self.evaluator, then, r#else)?;
        let masked = ContextWithBitAnd::bitand(&mut self.evaluator, cond, diff)?;
        ContextWithBitXor::bitxor(&mut self.evaluator, r#else, masked)
    }
}

// Defined by calling `cirrus-recompile-rt`'s exported macro exactly as an
// external, unrelated crate would -- this is the "usable by any crate"
// demonstration; nothing here is special-cased for being in this workspace.
cirrus_recompile_rt::define_pinned_backend!(pub mod gc_backend for ['a, 'b] GcBackend<'a, 'b, LabelDigest, LABEL_BYTES> as "gc");
cirrus_recompile_rt::define_pinned_backend!(pub mod eval_backend for ['a] EvalBackend<'a, LABEL_BYTES> as "eval");

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::Pusher;

    struct VecPusher<T>(Vec<T>);
    impl<T> Pusher<T> for VecPusher<T> {
        fn push(&mut self, x: T) {
            self.0.push(x);
        }
    }

    /// Garble, then evaluate, a small circuit via [`GcBackend`]/
    /// [`EvalBackend`] directly through `cirrus_recompile_rt::execute` (not
    /// yet through any code-generating backend -- that is
    /// `tests/multi_backend.rs`'s job) -- the baseline sanity check that
    /// these two wrapper `Context`s round-trip correctly at all before any
    /// backend lowers a `Program` against them.
    #[test]
    fn garble_then_evaluate_round_trips_through_the_wrapper_backends() {
        let mut recorder = cirrus_recompile_core::Recorder::new();
        let zero = recorder.create(false).unwrap();
        let one = recorder.create(true).unwrap();
        let a = recorder.create(false).unwrap();
        let b = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
        let mux = ContextWithMux::mux(&mut recorder, a, one, zero).unwrap();
        let program = recorder.finish(vec![zero, one, a, b], vec![and, or, xor, mux]);

        for &(av, bv) in &[(false, false), (false, true), (true, false), (true, true)] {
            let plaintext = cirrus_recompile_core::interpret(&program, &[false, true, av, bv]);

            let mut tables = VecPusher(Vec::new());
            let mut gc_backend = GcBackend::new(GC::<LabelDigest, LABEL_BYTES> {
                queue: &mut tables,
                seed: Default::default(),
                delta: Default::default(),
            });
            gc_backend.gc.delta[0] = 1;
            // The garbler's own labels for the input wires are picked by
            // the caller (see `Label`'s docs: a garbler never learns the
            // true bit, so any label is a valid "zero" reference) -- here,
            // directly, rather than through `Op::Create`, exactly like a
            // real ERT program's data inputs (see `program.inputs`).
            let garbler_zero = Label::new([0x11; LABEL_BYTES]);
            let garbler_one = Label::new([0x22; LABEL_BYTES]);
            let garbler_a = Label::new([0x33; LABEL_BYTES]);
            let garbler_b = Label::new([0x44; LABEL_BYTES]);
            let garbled = cirrus_recompile_rt::execute(
                &mut gc_backend,
                &program,
                &[garbler_zero, garbler_one, garbler_a, garbler_b],
            )
            .unwrap();

            let selected = |zero_label: Label<LABEL_BYTES>, bit: bool| -> [u8; LABEL_BYTES] {
                if bit {
                    array::from_fn(|i| zero_label.zero_label()[i] ^ gc_backend.gc.delta[i])
                } else {
                    zero_label.zero_label()
                }
            };
            let mut eval_backend = EvalBackend::new(Evaluator::new(Box::new(
                tables.0.into_iter().map(GarblingRecord::Table),
            )));
            let evaluated = cirrus_recompile_rt::execute(
                &mut eval_backend,
                &program,
                &[
                    selected(garbler_zero, false),
                    selected(garbler_one, true),
                    selected(garbler_a, av),
                    selected(garbler_b, bv),
                ],
            )
            .expect("the matching garbling table is available for every AND gate");

            for (i, &plaintext_bit) in plaintext.iter().enumerate() {
                assert_eq!(
                    evaluated[i],
                    selected(garbled[i], plaintext_bit),
                    "output {i} mismatch for inputs ({av}, {bv})"
                );
            }
        }
    }
}
