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

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithStorage, ContextWithValue, HasError, Pusher, StorageAddressBit,
};
use digest::{Digest, array::Array};
use volar_spec::{
    garble::{Eval, Garble, GarbleTable, GlobalSecret},
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
