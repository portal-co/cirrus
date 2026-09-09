#![no_std]
#![warn(missing_docs)]

//! Half-gate garbled-circuit `cirrus_core::Context` backend built on
//! `volar_spec::garble`'s per-gate primitives.
//!
//! Mirrors `cirrus_garbled_circuit`'s `GC`/`Evaluator` streaming/pull-based
//! shape (one [`GarbleTable`] pushed per AND gate, XOR free), but delegates
//! every cryptographic step -- AND-table generation, AND evaluation,
//! free-XOR -- to `volar_spec::garble` instead of this workspace's own
//! baseline four-row-table construction. This validates that `volar-spec`
//! can be called live, as a `Context`, from any of cirrus's existing
//! interpreters (`cirrus-ert`, `cirrus-llvm-frontend`) instead of only
//! through `volar-weaver`'s printed-Rust-source-then-`rustc` path.

use core::{convert::Infallible, fmt, marker::PhantomData};

extern crate alloc;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithStorage, ContextWithValue, HasError, Pusher, StorageAddressBit,
};
use digest::{Digest, array::Array};
use volar_spec::{
    garble::{Eval, Garble, GarbleTable, GlobalSecret, GramOutput, gram_decode_label, gram_regarble},
    vole::VoleArray,
};

/// The garbler-side context: streams one [`GarbleTable`] per AND gate to
/// [`Self::queue`], in circuit order.
pub struct VolarGarbleBackend<'a, 'b, D: Digest, N: VoleArray<u8>> {
    /// Ordered streaming destination for one [`GarbleTable`] per AND gate.
    pub queue: &'a mut (dyn Pusher<GarbleTable<N>> + 'b),
    /// The garbler's free-XOR global secret (Delta).
    pub secret: GlobalSecret<N>,
    /// Digest state chained to derive each fresh wire's false-label.
    pub seed: Array<u8, D::OutputSize>,
    marker: PhantomData<D>,
}

impl<'a, 'b, D: Digest, N: VoleArray<u8>> VolarGarbleBackend<'a, 'b, D, N> {
    /// Wrap a table sink and a global secret.
    pub fn new(queue: &'a mut (dyn Pusher<GarbleTable<N>> + 'b), secret: GlobalSecret<N>) -> Self {
        Self {
            queue,
            secret,
            seed: Default::default(),
            marker: PhantomData,
        }
    }

    fn fresh_label(&mut self) -> Garble<N> {
        self.seed = D::digest(&self.seed);
        let seed = self.seed.clone();
        Garble {
            base: Array::<u8, N>::from_fn(|i| seed[i]),
        }
    }
}

impl<D: Digest, N: VoleArray<u8>> HasError for VolarGarbleBackend<'_, '_, D, N> {
    type Error = Infallible;
}
impl<D: Digest, N: VoleArray<u8>> ContextWithValue<bool> for VolarGarbleBackend<'_, '_, D, N> {
    type Wrapped = Garble<N>;
}
impl<D: Digest, N: VoleArray<u8>> ContextWithCreate<bool> for VolarGarbleBackend<'_, '_, D, N> {
    fn create(&mut self, _val: bool) -> Result<Garble<N>, Infallible> {
        // A garbler's `Garble<N>` tracks a wire through its false-label only
        // (see `volar_spec::garble::Garble`'s field doc); the true bit is
        // never visible to it, so `_val` is immaterial -- the same
        // rationale `GcBackend::create` documents in cirrus-recompile-tests.
        Ok(self.fresh_label())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitXor<bool> for VolarGarbleBackend<'_, '_, D, N> {
    fn bitxor(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        Ok(Garble {
            base: Array::<u8, N>::from_fn(|i| a.base[i] ^ b.base[i]),
        })
    }
    fn bitxor_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitAnd<bool> for VolarGarbleBackend<'_, '_, D, N> {
    fn bitand(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        let table = self.secret.gen_and_table::<D>(&a, &b);
        let result = a.and_result::<D>(&b);
        self.queue.push(table);
        Ok(result)
    }
    fn bitand_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitOr<bool> for VolarGarbleBackend<'_, '_, D, N> {
    fn bitor(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }
    fn bitor_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithMux<bool> for VolarGarbleBackend<'_, '_, D, N> {
    fn mux(
        &mut self,
        cond: Garble<N>,
        then: Garble<N>,
        r#else: Garble<N>,
    ) -> Result<Garble<N>, Infallible> {
        let diff = self.bitxor(then, r#else.clone())?;
        let masked = self.bitand(cond, diff)?;
        self.bitxor(r#else, masked)
    }
}

impl<D: Digest, N: VoleArray<u8>> ContextWithStorage<bool> for VolarGarbleBackend<'_, '_, D, N> {
    type Storage = [Garble<N>];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
    ) -> Result<Garble<N>, Self::Error> {
        Ok(storage[concrete_storage_index(address)].clone())
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
        value: Garble<N>,
    ) -> Result<(), Self::Error> {
        storage[concrete_storage_index(address)] = value;
        Ok(())
    }
}

/// A constant-carrying garbler backend.
///
/// [`VolarGarbleBackend::create`] returns a *fresh* label from the digest
/// chain on every call, which is correct for a woven circuit (no internal
/// constants) but wrong for the ERT RV32 machine, whose ABI/ALU build
/// constant words from a *single* `zero`/`one` wire pair: each `create`
/// must return the *same* label for the same constant or the machine's
/// one-zero-wire invariant breaks. `VcGarbleBackend` caches the const-0 /
/// const-1 false-labels (derived once from the caller-supplied bases) and
/// returns them consistently. It is the garbler-side counterpart of
/// [`VcEvalBackend`]: the garbler's const-0 base is the all-zero label (so
/// the evaluator's const-0 `Eval` is the all-zero label too, identity
/// under free-XOR), and const-1 is a distinct revealed base.
pub struct VcGarbleBackend<'a, 'b, D: Digest, N: VoleArray<u8>> {
    /// The underlying gate garbler streaming tables to `queue`.
    inner: VolarGarbleBackend<'a, 'b, D, N>,
    /// The cached constant-0 / constant-1 wire false-labels.
    zero_label: Option<Garble<N>>,
    one_label: Option<Garble<N>>,
}

impl<'a, 'b, D: Digest, N: VoleArray<u8>> VcGarbleBackend<'a, 'b, D, N> {
    /// Wrap a table sink, a global secret, and the two constant-wire bases.
    /// `zero_base` should be the all-zero base so the evaluator's const-0
    /// label is the all-zero `Eval` (free-XOR identity).
    pub fn new(
        queue: &'a mut (dyn Pusher<GarbleTable<N>> + 'b),
        secret: GlobalSecret<N>,
        zero_base: Garble<N>,
        one_base: Garble<N>,
    ) -> Self {
        Self {
            inner: VolarGarbleBackend::new(queue, secret),
            zero_label: Some(zero_base),
            one_label: Some(one_base),
        }
    }

    /// The global secret (for deriving the evaluator's input labels).
    pub fn secret(&self) -> &GlobalSecret<N> {
        &self.inner.secret
    }

    fn zero_label(&mut self) -> Garble<N> {
        self.zero_label.clone().unwrap()
    }

    fn one_label(&mut self) -> Garble<N> {
        self.one_label.clone().unwrap()
    }
}

impl<D: Digest, N: VoleArray<u8>> HasError for VcGarbleBackend<'_, '_, D, N> {
    type Error = Infallible;
}
impl<D: Digest, N: VoleArray<u8>> ContextWithValue<bool> for VcGarbleBackend<'_, '_, D, N> {
    type Wrapped = Garble<N>;
}
impl<D: Digest, N: VoleArray<u8>> ContextWithCreate<bool> for VcGarbleBackend<'_, '_, D, N> {
    fn create(&mut self, val: bool) -> Result<Garble<N>, Infallible> {
        Ok(if val { self.one_label() } else { self.zero_label() })
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitXor<bool> for VcGarbleBackend<'_, '_, D, N> {
    fn bitxor(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        self.inner.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitAnd<bool> for VcGarbleBackend<'_, '_, D, N> {
    fn bitand(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        self.inner.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithBitOr<bool> for VcGarbleBackend<'_, '_, D, N> {
    fn bitor(&mut self, a: Garble<N>, b: Garble<N>) -> Result<Garble<N>, Infallible> {
        self.inner.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut Garble<N>, b: Garble<N>) -> Result<(), Infallible> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, N: VoleArray<u8>> ContextWithMux<bool> for VcGarbleBackend<'_, '_, D, N> {
    fn mux(
        &mut self,
        cond: Garble<N>,
        then: Garble<N>,
        r#else: Garble<N>,
    ) -> Result<Garble<N>, Infallible> {
        self.inner.mux(cond, then, r#else)
    }
}

/// An error while replaying a [`GarbleTable`] stream: the record iterator
/// ended before every AND gate was evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VolarEvalError {
    /// An AND operation required a table after the record iterator ended.
    Exhausted,
}
impl fmt::Display for VolarEvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("garble-table record iterator is exhausted"),
        }
    }
}
impl core::error::Error for VolarEvalError {}

/// The evaluator-side context: pulls one [`GarbleTable`] per AND gate from
/// [`Self::tables`], none for XOR.
pub struct VolarEvalBackend<D: Digest, I, N: VoleArray<u8>>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    /// The ordered source of AND-gate tables, one per AND gate.
    pub tables: I,
    marker: PhantomData<D>,
}

impl<D: Digest, I, N: VoleArray<u8>> VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    /// Construct an evaluator that pulls ordered tables from `tables`.
    pub fn new(tables: I) -> Self {
        Self {
            tables,
            marker: PhantomData,
        }
    }

    fn next_table(&mut self) -> Result<GarbleTable<N>, VolarEvalError> {
        self.tables.next().ok_or(VolarEvalError::Exhausted)
    }
}

impl<D: Digest, I, N: VoleArray<u8>> HasError for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    type Error = VolarEvalError;
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithValue<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    type Wrapped = Eval<N>;
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithCreate<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    fn create(&mut self, _val: bool) -> Result<Eval<N>, VolarEvalError> {
        // An evaluator cannot manufacture any wire's label from nothing --
        // it only ever receives one via a garbler-revealed value. Every
        // slot an `Op::Create` would touch must instead be supplied
        // externally by the caller as a `program.inputs` entry, exactly
        // like `EvalBackend::create`'s documented convention.
        Ok(Eval::zero())
    }
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithBitXor<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    fn bitxor(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        // `volar_spec::garble` provides `impl BitXor<Eval<N>> for Eval<N>`
        // directly.
        Ok(a ^ b)
    }
    fn bitxor_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithBitAnd<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    fn bitand(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        let table = self.next_table()?;
        // Operand order must match the garbler's `gen_and_table(&a, &b)`
        // call for the same gate.
        Ok(a.and_via_table::<D>(&b, &table))
    }
    fn bitand_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithBitOr<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    fn bitor(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }
    fn bitor_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}
impl<D: Digest, I, N: VoleArray<u8>> ContextWithMux<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    fn mux(
        &mut self,
        cond: Eval<N>,
        then: Eval<N>,
        r#else: Eval<N>,
    ) -> Result<Eval<N>, VolarEvalError> {
        let diff = self.bitxor(then, r#else.clone())?;
        let masked = self.bitand(cond, diff)?;
        self.bitxor(r#else, masked)
    }
}

/// A constant-carrying evaluator backend for the embedded vc executor (D3).
///
/// `VolarEvalBackend` collapses both `create(false)` and `create(true)` to
/// an all-zero `Eval` — the right choice for a woven circuit (which never
/// fabricates constants internally) but wrong for the ERT RV32 machine,
/// whose ALU and ABI build constant words from the `zero`/`one` ABI wires
/// and `ContextWithCreate`. With both constants equal, every constant
/// computation silently evaluates to 0.
///
/// `VcEvalBackend` fixes this by sourcing the constant-0 and constant-1
/// wire labels from a caller-supplied iterator — the garbler's constant-
/// wire labels, delivered at session setup (public constants, so they may
/// be revealed in the clear). The ERT machine uses exactly one constant-0
/// and one constant-1 wire per run, so a two-label supply suffices.
///
/// Gate semantics are identical to `VolarEvalBackend` (XOR free, AND via a
/// streamed table, OR/mux derived); only the constant wiring differs. The
/// garbler builds its half of the circuit treating `const0`/`const1` as
/// ordinary input wires, so every AND table already accounts for them.
pub struct VcEvalBackend<D: Digest, I, N: VoleArray<u8>, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    /// The underlying gate evaluator.
    inner: VolarEvalBackend<D, I, N>,
    /// Ordered constant labels: the first `next()` is the const-0 wire, the
    /// second the const-1 wire. Subsequent `create` calls reuse them.
    consts: C,
    /// The resolved constant-0 / constant-1 wire labels, once pulled.
    zero_label: Option<Eval<N>>,
    one_label: Option<Eval<N>>,
}

impl<D: Digest, I, N: VoleArray<u8>, C> VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    /// Construct a constant-carrying evaluator pulling AND tables from
    /// `tables` and the two constant-wire labels from `consts`.
    pub fn new(tables: I, consts: C) -> Self {
        Self {
            inner: VolarEvalBackend::new(tables),
            consts,
            zero_label: None,
            one_label: None,
        }
    }

    /// The constant-0 wire label, pulling it from the supply on first use.
    fn zero_label(&mut self) -> Eval<N> {
        if self.zero_label.is_none() {
            self.zero_label = Some(self.consts.next().unwrap_or_else(Eval::zero));
        }
        self.zero_label.clone().unwrap()
    }

    /// The constant-1 wire label, pulling it from the supply on first use.
    fn one_label(&mut self) -> Eval<N> {
        if self.one_label.is_none() {
            // Ensure const-0 was consumed first so the supply order holds.
            let _ = self.zero_label();
            self.one_label = Some(self.consts.next().unwrap_or_else(Eval::zero));
        }
        self.one_label.clone().unwrap()
    }
}

impl<D: Digest, I, N: VoleArray<u8>, C> HasError for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    type Error = VolarEvalError;
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithValue<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    type Wrapped = Eval<N>;
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithCreate<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    fn create(&mut self, val: bool) -> Result<Eval<N>, VolarEvalError> {
        Ok(if val { self.one_label() } else { self.zero_label() })
    }
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithBitXor<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    fn bitxor(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        self.inner.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        self.inner.bitxor_assign(a, b)
    }
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithBitAnd<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    fn bitand(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        self.inner.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        self.inner.bitand_assign(a, b)
    }
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithBitOr<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    fn bitor(&mut self, a: Eval<N>, b: Eval<N>) -> Result<Eval<N>, VolarEvalError> {
        self.inner.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut Eval<N>, b: Eval<N>) -> Result<(), VolarEvalError> {
        self.inner.bitor_assign(a, b)
    }
}
impl<D: Digest, I, N: VoleArray<u8>, C> ContextWithMux<bool> for VcEvalBackend<D, I, N, C>
where
    I: Iterator<Item = GarbleTable<N>>,
    C: Iterator<Item = Eval<N>>,
{
    fn mux(
        &mut self,
        cond: Eval<N>,
        then: Eval<N>,
        r#else: Eval<N>,
    ) -> Result<Eval<N>, VolarEvalError> {
        self.inner.mux(cond, then, r#else)
    }
}

impl<D: Digest, I, N: VoleArray<u8>> ContextWithStorage<bool> for VolarEvalBackend<D, I, N>
where
    I: Iterator<Item = GarbleTable<N>>,
{
    type Storage = [Eval<N>];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Eval<N>>],
    ) -> Result<Eval<N>, Self::Error> {
        Ok(storage[concrete_storage_index(address)].clone())
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Eval<N>>],
        value: Eval<N>,
    ) -> Result<(), Self::Error> {
        storage[concrete_storage_index(address)] = value;
        Ok(())
    }
}

fn concrete_storage_index<W>(address: &[StorageAddressBit<W>]) -> usize {
    address
        .iter()
        .enumerate()
        .fold(0usize, |index, (bit, address_bit)| {
            if address_bit
                .known
                .expect("ERT storage addresses are concrete")
            {
                index
                    | (1usize
                        .checked_shl(bit as u32)
                        .expect("ERT storage address exceeds usize width"))
            } else {
                index
            }
        })
}

// ============================================================================
// GRAM action host (interpreter-side garbled RAM access gadget)
// ============================================================================

/// The result of one GRAM action call: one bit per `num_bits`, each delivered
/// per its [`GramOutput`] mode.
#[derive(Clone)]
pub enum GramActionResult<N: VoleArray<u8>> {
    /// All result bits are cleartext (the [`GramOutput::Cleartext`] gadget):
    /// the evaluator decoded them and learns the values. Used for the ORAM
    /// `begin` leaf index and the tree path read.
    Cleartext(alloc::vec::Vec<bool>),
    /// Result bits are re-garbled to fresh labels (the [`GramOutput::Regarble`]
    /// gadget): the evaluator receives labels it cannot read. Used for ORAM
    /// `process`/`evict` bucket data that must stay secret.
    Regarble(alloc::vec::Vec<Eval<N>>),
}

/// Interpreter-side GRAM action host: executes one action of the garbled RAM
/// access sub-protocol against evaluator-held labels, mirroring the volar
/// garble weaver's cleartext-read / re-garble gadget (see `MPC_PLAN.md`
/// workstream A).
///
/// The host is the evaluator's trusted local party: it holds each wire's
/// false-label (`base`) so it can [`gram_decode_label`] the action argument
/// labels into plaintext, run the ORAM client logic (position-map lookup,
/// stash scan, eviction) over them, and return each result bit per its
/// [`GramOutput`] mode. This is the live-`Context` counterpart of the woven
/// evaluator's host extern call — same label-level operations, shared through
/// `volar_spec::garble`.
pub struct GramActionHost<N: VoleArray<u8>> {
    secret: GlobalSecret<N>,
}

impl<N: VoleArray<u8>> GramActionHost<N> {
    /// Construct a host from the garbler's [`GlobalSecret`]. The host needs
    /// the secret only to re-garble result bits; decoding uses each wire's
    /// false-label.
    pub fn new(secret: GlobalSecret<N>) -> Self {
        Self { secret }
    }

    /// Decode each action argument label to its plaintext bit. `args[i]` is
    /// the evaluator's label for arg wire `i`, `bases[i]` that wire's
    /// false-label.
    pub fn decode_args(args: &[Eval<N>], bases: &[Garble<N>]) -> alloc::vec::Vec<bool> {
        args.iter()
            .zip(bases)
            .map(|(label, base)| gram_decode_label(label, base))
            .collect()
    }

    /// Package a host-computed result bit per its [`GramOutput`] mode:
    /// cleartext (the evaluator learns it) or re-garbled under `base`.
    pub fn deliver(&self, mode: GramOutput, base: &Garble<N>, bit: bool) -> GramActionResult<N> {
        match mode {
            GramOutput::Cleartext => GramActionResult::Cleartext(alloc::vec![bit]),
            GramOutput::Regarble => {
                GramActionResult::Regarble(alloc::vec![gram_regarble(&self.secret, base, bit)])
            }
        }
    }
}

// ============================================================================
// GRAM ORAM host (interpreter-side full ORAM access over labels)
// ============================================================================

/// The result of one full interpreter-side ORAM access: the block's read
/// data as re-garbled labels (secret), ready to flow back into the circuit.
#[derive(Clone)]
pub struct GramOramRead<N: VoleArray<u8>> {
    /// The accessed block's data bits, re-garbled to fresh labels (one per
    /// data bit, `8 * B` of them).
    pub data_labels: alloc::vec::Vec<Eval<N>>,
    /// The cleartext old-leaf index the access touched (data-independent —
    /// the evaluator is allowed to learn it, per the GRAM output contract).
    pub old_leaf: u64,
}

/// Interpreter-side full ORAM access over garbled labels: the cirrus
/// counterpart of volar-vc's `OramHost` + `OramHostShim`, combining the
/// label layer ([`GramActionHost`]) with the shared bit-level driver
/// ([`volar_oram::bit_host::OramHost`]).
///
/// Where the woven evaluator calls a host extern once per action, the cirrus
/// interpreter drives the whole begin → process → evict×2 sequence inline
/// over the evaluator's labels: it decodes the address label to plaintext,
/// runs the ORAM client against the evaluator-hosted [`OramTree`], and
/// re-garbles the resulting read-data bits to fresh labels.
///
/// The host holds the garbler's [`GlobalSecret`] (for re-garbling) and the
/// shared bit-level driver. The per-result-wire bases are supplied by a
/// `base_for(i) -> Garble<N>` closure — the same deterministic base supply
/// the garbler used for those wires (the increment-4 embedder contract).
///
/// `Z` is the bucket size, `B` the block size in bytes; `levels` the tree
/// depth.
pub struct GramOramHost<N: VoleArray<u8>, const Z: usize, const B: usize> {
    /// Label layer: decode arg labels, re-garble result bits.
    action_host: GramActionHost<N>,
    /// Shared bit-level ORAM client driver (label-free).
    driver: volar_oram::bit_host::OramHost<Z, B>,
}

impl<N: VoleArray<u8>, const Z: usize, const B: usize> GramOramHost<N, Z, B> {
    /// Construct a host from the garbler's secret, for a tree of `levels`
    /// levels addressing `num_addrs` blocks.
    pub fn new(secret: GlobalSecret<N>, levels: usize, num_addrs: u64) -> Self {
        Self {
            action_host: GramActionHost::new(secret),
            driver: volar_oram::bit_host::OramHost::new(levels, num_addrs),
        }
    }

    /// Borrow the shared driver's ORAM client (e.g. to inspect the stash).
    pub fn client(&self) -> &volar_oram::OramClient<Z, B> {
        self.driver.client()
    }

    /// Run one full ORAM access over labels against the evaluator-hosted
    /// `tree`, returning the accessed block's data re-garbled to labels.
    ///
    /// - `addr_labels` / `addr_bases`: the evaluator's labels for the 64
    ///   address input bits (LSB-first) and their false-labels. The host
    ///   decodes them to a plaintext `u64` address — the address is a
    ///   client-secret input that becomes data-independent once the position
    ///   map is consulted, so the host may learn it (per the GRAM output
    ///   contract for `begin`).
    /// - `write`: `Some(bits)` writes the supplied `8 * B` data bits into the
    ///   cell; `None` is a read. (In the full gadget the write-data bits
    ///   arrive as labels too and are decoded here; the harness passes them
    ///   pre-decoded.)
    /// - `base_for(i)`: the false-label for the `i`-th read-data result bit.
    /// - `rng`: position-map leaf-assignment randomness.
    ///
    /// The tree is evaluator-hosted: the host reads/writes buckets in the
    /// clear at the (cleartext) leaves the client logic selects.
    pub fn access(
        &mut self,
        tree: &mut volar_oram::OramTree<Z, B>,
        addr_labels: &[Eval<N>],
        addr_bases: &[Garble<N>],
        write: Option<&[bool]>,
        base_for: &mut dyn FnMut(usize) -> Garble<N>,
        rng: &mut dyn FnMut() -> u64,
    ) -> GramOramRead<N> {
        use volar_oram::bit_host::OramHost as Drv;

        // 1. Decode the address labels to a plaintext u64 (LSB-first).
        assert_eq!(
            addr_labels.len(),
            64,
            "GramOramHost: address is 64 bits"
        );
        let addr_bits_decoded = GramActionHost::<N>::decode_args(addr_labels, addr_bases);
        let mut addr = 0u64;
        for (i, b) in addr_bits_decoded.iter().enumerate() {
            if *b {
                addr |= 1u64 << i;
            }
        }
        let mut addr_bits = alloc::vec::Vec::new();
        Drv::<Z, B>::push_u64(&mut addr_bits, addr, 64);

        // begin: addr → old_leaf.
        let leaf_bits = self
            .driver
            .begin(&addr_bits, rng)
            .expect("GramOramHost: begin");
        let mut off = 0usize;
        let old_leaf = Drv::<Z, B>::take_u64(&leaf_bits, &mut off, 64);

        // 2. Read the tree path at old_leaf, flatten to bits.
        let path = tree.read_path(old_leaf);
        let mut path_bits = alloc::vec::Vec::new();
        self.driver.push_path(&mut path_bits, &path);

        // 3. process: path ‖ data ‖ is_write → wb_path ‖ read_data ‖ e1 ‖ e2.
        let mut proc_args = path_bits.clone();
        let wd: alloc::vec::Vec<bool> = match write {
            Some(w) => w.to_vec(),
            None => alloc::vec![false; 8 * B],
        };
        proc_args.extend_from_slice(&wd);
        proc_args.push(write.is_some());
        let proc_out = self.driver.process(&proc_args).expect("GramOramHost: process");

        let path_bits_len = path_bits.len();
        let wb_bits = &proc_out[..path_bits_len];
        let mut roff = path_bits_len;
        let mut read_data_bits = alloc::vec::Vec::with_capacity(8 * B);
        for _ in 0..(8 * B) {
            read_data_bits.push(proc_out[roff]);
            roff += 1;
        }
        let evict1 = Drv::<Z, B>::take_u64(&proc_out, &mut roff, 64);
        let evict2 = Drv::<Z, B>::take_u64(&proc_out, &mut roff, 64);

        // 4. Write back the updated path at old_leaf.
        let wb_path = self.driver.take_path(wb_bits, "wb").expect("GramOramHost: wb path");
        tree.write_path(old_leaf, &wb_path);

        // 5. Two eviction passes.
        for evict_leaf in [evict1, evict2] {
            let ep = tree.read_path(evict_leaf);
            let mut ep_bits = alloc::vec::Vec::new();
            self.driver.push_path(&mut ep_bits, &ep);
            let new_ep_bits = self.driver.evict(&ep_bits).expect("GramOramHost: evict");
            let new_ep = self
                .driver
                .take_path(&new_ep_bits, "evict_out")
                .expect("GramOramHost: evict path");
            tree.write_path(evict_leaf, &new_ep);
        }

        // 6. Re-garble the read-data bits to fresh labels.
        let data_labels = read_data_bits
            .iter()
            .enumerate()
            .map(|(i, &bit)| match self.action_host.deliver(GramOutput::Regarble, &base_for(i), bit) {
                GramActionResult::Regarble(labels) => labels[0].clone(),
                GramActionResult::Cleartext(_) => unreachable!("Regarble mode returns labels"),
            })
            .collect();

        GramOramRead {
            data_labels,
            old_leaf,
        }
    }
}

// ============================================================================
// GRAM-backed storage (A1 capstone): sublinear ORAM storage for the
// garbled-circuit contexts
// ============================================================================

use volar_oram::OramTree;

/// A deterministic base supply for ORAM data wires: the `i`-th access's
/// `j`-th data-bit false-label is derived as
/// `H(0xDA || access_index || bit_index)` — a pure function both the garbler
/// and the evaluator-side host compute, mirroring
/// [`Garble::action_result_base`], so the garbler's downstream gates and the
/// host's re-garble agree with no extra communication.
fn gram_data_base<D: Digest, N: VoleArray<u8>>(access: u64, bit: u64) -> Garble<N> {
    let mut d = D::new();
    d.update(&[0xDAu8]);
    d.update(&access.to_le_bytes());
    d.update(&bit.to_le_bytes());
    let hash = d.finalize();
    Garble {
        base: Array::<u8, N>::from_fn(|i| hash[i]),
    }
}

/// GRAM-backed storage for the garbled-circuit contexts: the A1 capstone.
///
/// Replaces the linear-scan `[Garble]`/`[Eval]` slice storage with a
/// **sublinear ORAM** backed by an evaluator-hosted [`OramTree`]. The storage
/// is a pair `(garbler_addr_bases, tree)` where the garbler side tracks the
/// per-address false-labels it assigned and the evaluator side holds the
/// ORAM tree plus its ORAM host.
///
/// The two roles share one context type parameterized on the wire label
/// `W` (`Garble<N>` garbler-side, `Eval<N>` evaluator-side):
///
/// - **Garbler** (`GramStorage<VolarGarbleBackend>`): `storage_read` returns
///   the deterministic ORAM data-wire bases; the address bits are tracked as
///   their false-labels. No tables are emitted for the access itself (the
///   ORAM client logic is the gadget).
/// - **Evaluator** (`GramStorage<VolarEvalBackend>`): `storage_read` /
///   `storage_write` decode the address, run the full ORAM access through
///   [`GramOramHost`], and return the re-garbled read-data labels — which
///   decode correctly against the garbler's deterministic bases.
///
/// This is the interpreter-side counterpart of the volar woven GRAM circuit
/// (`weave_garbler_with_gram` + `weave_evaluator_with_gram`): same label
/// currency, same ORAM access sub-protocol, driven live by the `Context`.
///
/// `Z` is the ORAM bucket size, `B` the block size in bytes (one bit per
/// byte stored, matching the one-bit-per-cell boolar convention).
pub struct GramStorage<'t, C, D: Digest, N: VoleArray<u8>, const Z: usize, const B: usize> {
    /// The wrapped gate context (garbler or evaluator backend).
    pub inner: C,
    /// The ORAM host (evaluator-side; holds the secret for re-garbling).
    pub host: GramOramHost<N, Z, B>,
    /// The evaluator-hosted ORAM tree.
    pub tree: &'t mut OramTree<Z, B>,
    /// The garbler's global secret (re-garbling + address re-encoding).
    secret: GlobalSecret<N>,
    /// Number of ORAM accesses performed (drives the deterministic base
    /// supply).
    access_count: u64,
    marker: PhantomData<D>,
    /// Position-map leaf-assignment randomness (splitmix64 state). A real
    /// RNG is required: reusing a constant leaf breaks ORAM correctness
    /// across cells.
    rng_state: u64,
}

impl<'t, C, D: Digest, N: VoleArray<u8>, const Z: usize, const B: usize>
    GramStorage<'t, C, D, N, Z, B>
{
    /// Wrap a gate context with GRAM storage over `tree`.
    pub fn new(
        inner: C,
        secret: GlobalSecret<N>,
        tree: &'t mut OramTree<Z, B>,
        levels: usize,
        num_addrs: u64,
    ) -> Self {
        Self {
            inner,
            host: GramOramHost::new(secret.clone(), levels, num_addrs),
            tree,
            secret,
            access_count: 0,
            rng_state: 0x9E3779B97F4A7C15,
            marker: PhantomData,
        }
    }

    /// Set the position-map RNG seed (test determinism).
    pub fn with_rng_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Next splitmix64 output for ORAM leaf assignment.
    fn next_rng(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        splitmix64(self.rng_state)
    }

    /// The number of ORAM accesses performed so far. The deterministic
    /// read-data base for the *next* read is
    /// `gram_data_base::<D, N>(self.access_count() + 1, 0)`; tests and the
    /// garbler-side harness use this to derive the matching base.
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// The deterministic base for the `bit`-th wire of access `access`
    /// (`bit` 0 = read-data output, `bit` 1 = write-data input).
    pub fn base_for(access: u64, bit: u64) -> Garble<N> {
        gram_data_base::<D, N>(access, bit)
    }

    /// Split off an independent leaf-assignment stream for one access,
    /// advancing the shared state once. The returned stream owns its state
    /// (no borrow of `self`) so it can be passed to `self.host.access`.
    fn rng_stream(&mut self) -> impl FnMut() -> u64 + 'static {
        let mut s = self.next_rng();
        move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            splitmix64(s)
        }
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// Delegate the gate operations to the wrapped context.
impl<'t, C, D, N, const Z: usize, const B: usize> HasError for GramStorage<'t, C, D, N, Z, B>
where
    C: HasError,
    D: Digest,
    N: VoleArray<u8>,
{
    type Error = C::Error;
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithValue<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithValue<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    type Wrapped = C::Wrapped;
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithCreate<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithCreate<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    fn create(&mut self, val: bool) -> Result<C::Wrapped, C::Error> {
        self.inner.create(val)
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithBitXor<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithBitXor<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    fn bitxor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.inner.bitxor_assign(a, b)
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithBitAnd<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithBitAnd<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    fn bitand(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.inner.bitand_assign(a, b)
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithBitOr<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithBitOr<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    fn bitor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.inner.bitor_assign(a, b)
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithMux<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: ContextWithMux<bool>,
    D: Digest,
    N: VoleArray<u8>,
{
    fn mux(
        &mut self,
        cond: C::Wrapped,
        then: C::Wrapped,
        r#else: C::Wrapped,
    ) -> Result<C::Wrapped, C::Error> {
        self.inner.mux(cond, then, r#else)
    }
}

/// The caller-owned storage space for [`GramStorage`]: one ORAM memory of
/// `num_addrs` blocks. Holds the garbler's per-address false-labels so the
/// garbler side can track addresses as bases; the evaluator side uses only
/// the concrete address index (the tree lives on the context).
///
/// One block = `B` bytes; a storage cell is one bit (boolar convention), so
/// a `storage_read` returns the bit at a concrete (byte, bit) position. To
/// keep the gadget simple, each storage cell maps to one ORAM block whose
/// byte 0 bit 0 carries the value — matching the small-memory GRAM regime
/// the gadget targets.
pub struct GramStorageSpace<N: VoleArray<u8>> {
    /// The garbler's false-label for each storage cell's ORAM data wire
    /// (`num_cells` of them), used garbler-side as the tracked address base
    /// and evaluator-side as the deterministic base the host re-garbles to.
    pub cell_bases: alloc::vec::Vec<Garble<N>>,
}

impl<N: VoleArray<u8>> GramStorageSpace<N> {
    /// Allocate a storage space of `num_cells` one-bit cells, deriving each
    /// cell's base deterministically.
    pub fn new<D: Digest>(num_cells: usize) -> Self {
        Self {
            cell_bases: (0..num_cells)
                .map(|i| gram_data_base::<D, N>(0, i as u64))
                .collect(),
        }
    }
}

impl<'t, 'b, D, N, const Z: usize, const B: usize> ContextWithStorage<bool>
    for GramStorage<'t, VolarGarbleBackend<'t, 'b, D, N>, D, N, Z, B>
where
    D: Digest,
    N: VoleArray<u8>,
{
    type Storage = GramStorageSpace<N>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
    ) -> Result<Garble<N>, Infallible> {
        Ok(self.garbler_read(storage, address))
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
        value: Garble<N>,
    ) -> Result<(), Infallible> {
        self.garbler_write(storage, address, value);
        Ok(())
    }
}

impl<'t, 'b, D, N, const Z: usize, const B: usize> ContextWithStorage<bool>
    for GramStorage<'t, VcGarbleBackend<'t, 'b, D, N>, D, N, Z, B>
where
    D: Digest,
    N: VoleArray<u8>,
{
    type Storage = GramStorageSpace<N>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
    ) -> Result<Garble<N>, Infallible> {
        Ok(self.garbler_read(storage, address))
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Garble<N>>],
        value: Garble<N>,
    ) -> Result<(), Infallible> {
        self.garbler_write(storage, address, value);
        Ok(())
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> ContextWithStorage<bool>
    for GramStorage<'t, C, D, N, Z, B>
where
    C: HasError<Error = VolarEvalError> + ContextWithValue<bool, Wrapped = Eval<N>>,
    D: Digest,
    N: VoleArray<u8>,
{
    type Storage = GramStorageSpace<N>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Eval<N>>],
    ) -> Result<Eval<N>, VolarEvalError> {
        let cell = concrete_storage_index(address);
        let _ = storage;
        self.access_count += 1;
        let access = self.access_count;
        // Run the ORAM access (read) through the host; the data bit is
        // re-garbled to the cell's deterministic base.
        let addr = cell as u64;
        let (addr_labels, addr_bases) = self.encode_addr(addr);
        let base = gram_data_base::<D, N>(access, 0);
        let mut stream = self.rng_stream();
        let this = &mut *self;
        let read = this.host.access(
            &mut *this.tree,
            &addr_labels,
            &addr_bases,
            None,
            &mut |_| base.clone(),
            &mut stream,
        );
        Ok(read.data_labels[0].clone())
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Eval<N>>],
        value: Eval<N>,
    ) -> Result<(), VolarEvalError> {
        let cell = concrete_storage_index(address);
        let _ = storage;
        self.access_count += 1;
        let access = self.access_count;
        let addr = cell as u64;
        let (addr_labels, addr_bases) = self.encode_addr(addr);
        // The write bit is the evaluator's label for `value`; decode it
        // host-side against the value's base. For a machine-driven write the
        // value wire's base is the machine's (e.g. a constant-wire base);
        // the garbler tracks it in `cell_bases[cell]` and the evaluator
        // mirrors that here so both parties decode the same bit.
        let value_base = storage.cell_bases[cell].clone();
        let bit = gram_decode_label(&value, &value_base);
        let mut wb = alloc::vec![false; 8 * B];
        wb[0] = bit;
        let base = gram_data_base::<D, N>(access, 0);
        let mut stream = self.rng_stream();
        self.host.access(
            &mut *self.tree,
            &addr_labels,
            &addr_bases,
            Some(&wb),
            &mut |_| base.clone(),
            &mut stream,
        );
        Ok(())
    }
}

impl<'t, C, D, N, const Z: usize, const B: usize> GramStorage<'t, C, D, N, Z, B>
where
    D: Digest,
    N: VoleArray<u8>,
{
    /// Encode a concrete 64-bit address as labels + bases (the evaluator
    /// learns the address — data-independent in the ORAM begin gadget).
    fn encode_addr(&self, addr: u64) -> (alloc::vec::Vec<Eval<N>>, alloc::vec::Vec<Garble<N>>) {
        let mut labels = alloc::vec::Vec::with_capacity(64);
        let mut bases = alloc::vec::Vec::with_capacity(64);
        for i in 0..64u64 {
            let base = gram_data_base::<D, N>(u64::MAX, i);
            let bit = (addr >> i) & 1 == 1;
            labels.push(self.secret.encode(&base, bit));
            bases.push(base);
        }
        (labels, bases)
    }

    /// Garbler-side read: return the deterministic base the evaluator's
    /// read label is re-garbled to (`gram_data_base(access, 0)`), so the
    /// two parties' output wires share one base. No tables are emitted for
    /// the access (the ORAM client logic is the gadget). The address is
    /// concrete (data-independent in the begin gadget).
    fn garbler_read(
        &mut self,
        storage: &mut GramStorageSpace<N>,
        address: &[StorageAddressBit<Garble<N>>],
    ) -> Garble<N> {
        let _cell = concrete_storage_index(address);
        let _ = storage;
        self.access_count += 1;
        let access = self.access_count;
        gram_data_base::<D, N>(access, 0)
    }

    /// Garbler-side write: record the cell's new base (the written wire's
    /// base), which the evaluator mirrors in `cell_bases[cell]` so its
    /// write-value decode matches. No table is emitted.
    fn garbler_write(
        &mut self,
        storage: &mut GramStorageSpace<N>,
        address: &[StorageAddressBit<Garble<N>>],
        value: Garble<N>,
    ) {
        let cell = concrete_storage_index(address);
        self.access_count += 1;
        storage.cell_bases[cell] = value;
    }
}
