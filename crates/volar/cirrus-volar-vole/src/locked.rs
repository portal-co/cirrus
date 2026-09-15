//! Locked, `&self`-capable VOLE handlers for parallel compatible circuits.
//!
//! Compatible circuits share a field and `Delta`. Mutating operations are
//! serialized: AND hats join one ordered stream, and authenticated storage
//! timestamps, witnesses, and accumulators go through one lock. The verifier
//! must replay **the same interleaving** (lock-acquisition order). The locks
//! do not tag hats or storage accesses per circuit.
//!
//! Circuit-granularity locking (hold a mutex around each `execute`) is the
//! easy sound schedule. Gate-granularity sharing is sound only if both
//! sides observe the same interleaving.

use core::{
    convert::Infallible,
    ops::{Add, Mul},
};

use cipher::consts::U1;
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitAndByRef, ContextWithBitOr, ContextWithBitOrByRef,
    ContextWithBitXor, ContextWithBitXorByRef, ContextWithCreate, ContextWithCreateByRef,
    ContextWithMux, ContextWithMuxByRef, ContextWithStorage, ContextWithStorageByRef,
    ContextWithValue, HasError, Pusher, PusherByRef, StorageAddressBit,
};
use hybrid_array::{Array, ArraySize};
use volar_spec::{
    field::Invert,
    vole::{Delta, Q, VoleArray, Vope, prove::vole_and_prover_step, setup::derive_and_q},
};

use crate::{
    StorageReadWitness, StorageWriteWitness, VoleProverStorage, VoleStorageConfig,
    VoleStorageContextError, VoleStorageError, VoleVerifierStorage, VoleVerifyError,
};

/// A [`Pusher`] that serializes `push` through a spin mutex.
pub struct MutexPusher<P> {
    inner: spin::Mutex<P>,
}

impl<P> MutexPusher<P> {
    /// Wrap an exclusive pusher.
    pub fn new(pusher: P) -> Self {
        Self {
            inner: spin::Mutex::new(pusher),
        }
    }

    /// Recover the inner pusher.
    pub fn into_inner(self) -> P {
        self.inner.into_inner()
    }
}

impl<T, P: Pusher<T>> Pusher<T> for MutexPusher<P> {
    fn push(&mut self, x: T) {
        self.push_by_ref(x);
    }
}

impl<T, P: Pusher<T>> PusherByRef<T> for MutexPusher<P> {
    fn push_by_ref(&self, x: T) {
        self.inner.lock().push(x);
    }
}

/// Shared-reference source of a hat (or other) stream.
pub trait PullerByRef<T> {
    /// Take the next item, or `None` when the stream is exhausted.
    fn next_by_ref(&self) -> Option<T>;
}

/// An iterator that serializes `next` through a spin mutex.
pub struct MutexPuller<I> {
    inner: spin::Mutex<I>,
}

impl<I> MutexPuller<I> {
    /// Wrap an exclusive iterator.
    pub fn new(iter: I) -> Self {
        Self {
            inner: spin::Mutex::new(iter),
        }
    }

    /// Recover the inner iterator.
    pub fn into_inner(self) -> I {
        self.inner.into_inner()
    }
}

impl<I: Iterator> PullerByRef<I::Item> for MutexPuller<I> {
    fn next_by_ref(&self) -> Option<I::Item> {
        self.inner.lock().next()
    }
}

/// Prover-side context that emits AND hats through [`PusherByRef`].
///
/// XOR and `create` are lock-free. AND (and OR/mux, which use AND) push one
/// hat per multiplication through [`Self::hats`].
pub struct LockedVoleProverContext<'a, 'b, N: VoleArray<T>, T> {
    /// Ordered streaming destination for one `hat` per AND gate.
    pub hats: &'a (dyn PusherByRef<Array<T, N>> + Sync + 'b),
    /// Lifts a known-constant bit to the field `T`.
    pub bit_to_t: fn(bool) -> T,
}

impl<N: VoleArray<T>, T> HasError for LockedVoleProverContext<'_, '_, N, T> {
    type Error = Infallible;
}

impl<N: VoleArray<T>, T> ContextWithValue<bool> for LockedVoleProverContext<'_, '_, N, T> {
    type Wrapped = Vope<N, T, U1>;
}

impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreate<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn create(&mut self, val: bool) -> Result<Vope<N, T, U1>, Infallible> {
        self.create_by_ref(val)
    }
}

impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreateByRef<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn create_by_ref(&self, val: bool) -> Result<Vope<N, T, U1>, Infallible> {
        let t = (self.bit_to_t)(val);
        Ok(Vope {
            u: Array::<Array<T, N>, U1>::from_fn(|_| Array::<T, N>::from_fn(|_| t.clone())),
            v: Array::<T, N>::from_fn(|_| T::default()),
        })
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Default> ContextWithBitXor<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitxor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        self.bitxor_by_ref(a, b)
    }
    fn bitxor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        self.bitxor_assign_by_ref(a, b)
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Default> ContextWithBitXorByRef<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitxor_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        Ok(a + b)
    }
    fn bitxor_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        *a = self.bitxor_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default>
    ContextWithBitAnd<bool> for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitand(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        self.bitand_by_ref(a, b)
    }
    fn bitand_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        self.bitand_assign_by_ref(a, b)
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default>
    ContextWithBitAndByRef<bool> for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitand_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let (c, hat) = vole_and_prover_step(a, b);
        self.hats.push_by_ref(hat);
        Ok(c)
    }
    fn bitand_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        *a = self.bitand_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithBitOr<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        self.bitor_by_ref(a, b)
    }
    fn bitor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        self.bitor_assign_by_ref(a, b)
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default>
    ContextWithBitOrByRef<bool> for LockedVoleProverContext<'_, '_, N, T>
{
    fn bitor_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let either = self.bitxor_by_ref(a.clone(), b.clone())?;
        let both = self.bitand_by_ref(a, b)?;
        self.bitxor_by_ref(either, both)
    }
    fn bitor_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        *a = self.bitor_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithMux<bool>
    for LockedVoleProverContext<'_, '_, N, T>
{
    fn mux(
        &mut self,
        cond: Vope<N, T, U1>,
        then: Vope<N, T, U1>,
        r#else: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        self.mux_by_ref(cond, then, r#else)
    }
}

impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default>
    ContextWithMuxByRef<bool> for LockedVoleProverContext<'_, '_, N, T>
{
    fn mux_by_ref(
        &self,
        cond: Vope<N, T, U1>,
        then: Vope<N, T, U1>,
        r#else: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let diff = self.bitxor_by_ref(then, r#else.clone())?;
        let masked = self.bitand_by_ref(cond, diff)?;
        self.bitxor_by_ref(r#else, masked)
    }
}

/// Verifier-side context that pulls AND hats through [`PullerByRef`].
pub struct LockedVoleVerifierContext<'a, 'b, N: ArraySize, T> {
    /// The verifier's secret global offset.
    pub delta: Delta<N, T>,
    /// The ordered source of `hat` values, one per AND gate.
    pub hats: &'a (dyn PullerByRef<Array<T, N>> + Sync + 'b),
}

impl<N: ArraySize, T> HasError for LockedVoleVerifierContext<'_, '_, N, T> {
    type Error = VoleVerifyError;
}

impl<N: ArraySize, T> ContextWithValue<bool> for LockedVoleVerifierContext<'_, '_, N, T> {
    type Wrapped = Q<N, T>;
}

impl<N: ArraySize, T: Clone + Default> ContextWithCreate<bool>
    for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn create(&mut self, val: bool) -> Result<Q<N, T>, VoleVerifyError> {
        self.create_by_ref(val)
    }
}

impl<N: ArraySize, T: Clone + Default> ContextWithCreateByRef<bool>
    for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn create_by_ref(&self, val: bool) -> Result<Q<N, T>, VoleVerifyError> {
        Ok(Q {
            q: if val {
                self.delta.delta.clone()
            } else {
                Array::<T, N>::from_fn(|_| T::default())
            },
        })
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T>> ContextWithBitXor<bool>
    for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitxor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        self.bitxor_by_ref(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        self.bitxor_assign_by_ref(a, b)
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T>> ContextWithBitXorByRef<bool>
    for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitxor_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        Ok(Q {
            q: Array::<T, N>::from_fn(|i| a.q[i].clone() + b.q[i].clone()),
        })
    }
    fn bitxor_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitxor_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithBitAnd<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitand(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        self.bitand_by_ref(a, b)
    }
    fn bitand_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        self.bitand_assign_by_ref(a, b)
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithBitAndByRef<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitand_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let hat = self
            .hats
            .next_by_ref()
            .ok_or(VoleVerifyError::HatExhausted)?;
        Ok(derive_and_q(&self.delta, &a, &b, &hat))
    }
    fn bitand_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitand_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithBitOr<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        self.bitor_by_ref(a, b)
    }
    fn bitor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        self.bitor_assign_by_ref(a, b)
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithBitOrByRef<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn bitor_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let either = self.bitxor_by_ref(a.clone(), b.clone())?;
        let both = self.bitand_by_ref(a, b)?;
        self.bitxor_by_ref(either, both)
    }
    fn bitor_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitor_by_ref(a.clone(), b)?;
        Ok(())
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithMux<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn mux(
        &mut self,
        cond: Q<N, T>,
        then: Q<N, T>,
        r#else: Q<N, T>,
    ) -> Result<Q<N, T>, VoleVerifyError> {
        self.mux_by_ref(cond, then, r#else)
    }
}

impl<N: ArraySize, T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default>
    ContextWithMuxByRef<bool> for LockedVoleVerifierContext<'_, '_, N, T>
{
    fn mux_by_ref(
        &self,
        cond: Q<N, T>,
        then: Q<N, T>,
        r#else: Q<N, T>,
    ) -> Result<Q<N, T>, VoleVerifyError> {
        let diff = self.bitxor_by_ref(then, r#else.clone())?;
        let masked = self.bitand_by_ref(cond, diff)?;
        self.bitxor_by_ref(r#else, masked)
    }
}

/// Locked wrapper around [`VoleProverStorage`].
///
/// `initialize`, `drain`, `finish`, and storage accesses take `&self` and
/// serialize through the mutex so several circuits can share one trace.
pub struct LockedVoleProverStorage<N: VoleArray<T>, T> {
    inner: spin::Mutex<VoleProverStorage<N, T>>,
}

impl<N, T> LockedVoleProverStorage<N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    /// Construct locked storage from caller-owned, ordered witnesses.
    pub fn new(
        zero: Vope<N, T, U1>,
        one: Vope<N, T, U1>,
        config: VoleStorageConfig<T>,
        reads: impl IntoIterator<Item = StorageReadWitness<Vope<N, T, U1>>>,
        writes: impl IntoIterator<Item = StorageWriteWitness<Vope<N, T, U1>>>,
    ) -> Self {
        Self {
            inner: spin::Mutex::new(VoleProverStorage::new(zero, one, config, reads, writes)),
        }
    }

    /// Add an initial live storage entry to the producer accumulator.
    pub fn initialize(
        &self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().initialize(address, value, timestamp)
    }

    /// Drain one surviving authenticated entry into the consume accumulator.
    pub fn drain(
        &self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().drain(address, value, timestamp)
    }

    /// Ensure all witnesses were consumed and open the final accumulator mask.
    pub fn finish(&self) -> Result<Array<T, N>, VoleStorageError> {
        self.inner.lock().finish()
    }

    fn read(
        &self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, VoleStorageError> {
        self.inner.lock().read(address)
    }

    fn write(
        &self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().write(address, value)
    }
}

/// Locked wrapper around [`VoleVerifierStorage`].
pub struct LockedVoleVerifierStorage<N: ArraySize, T> {
    inner: spin::Mutex<VoleVerifierStorage<N, T>>,
}

impl<N, T> LockedVoleVerifierStorage<N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Construct locked verifier storage from caller-owned, ordered witnesses.
    pub fn new(
        zero: Q<N, T>,
        one: Q<N, T>,
        config: VoleStorageConfig<T>,
        reads: impl IntoIterator<Item = StorageReadWitness<Q<N, T>>>,
        writes: impl IntoIterator<Item = StorageWriteWitness<Q<N, T>>>,
    ) -> Self {
        Self {
            inner: spin::Mutex::new(VoleVerifierStorage::new(zero, one, config, reads, writes)),
        }
    }

    /// Add an initial live storage entry to the producer accumulator.
    pub fn initialize(
        &self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().initialize(address, value, timestamp)
    }

    /// Drain one surviving authenticated entry into the consume accumulator.
    pub fn drain(
        &self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().drain(address, value, timestamp)
    }

    /// Consume every witness and verify a prover-supplied final drain opening.
    pub fn finish(&self, opening: &Array<T, N>) -> Result<(), VoleStorageError>
    where
        T: PartialEq,
    {
        self.inner.lock().finish(opening)
    }

    fn read(&self, address: &[StorageAddressBit<Q<N, T>>]) -> Result<Q<N, T>, VoleStorageError> {
        self.inner.lock().read(address)
    }

    fn write(
        &self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
    ) -> Result<(), VoleStorageError> {
        self.inner.lock().write(address, value)
    }
}

/// Storage-capable locked prover adapter.
pub struct LockedVoleProverStorageContext<'a, 'b, N: VoleArray<T>, T> {
    /// The underlying locked VOLE Boolean context.
    pub inner: LockedVoleProverContext<'a, 'b, N, T>,
}

impl<N: VoleArray<T>, T> HasError for LockedVoleProverStorageContext<'_, '_, N, T> {
    type Error = VoleStorageContextError;
}

impl<N: VoleArray<T>, T> ContextWithValue<bool> for LockedVoleProverStorageContext<'_, '_, N, T> {
    type Wrapped = Vope<N, T, U1>;
}

impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreate<bool>
    for LockedVoleProverStorageContext<'_, '_, N, T>
{
    fn create(&mut self, value: bool) -> Result<Vope<N, T, U1>, Self::Error> {
        self.create_by_ref(value)
    }
}

impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreateByRef<bool>
    for LockedVoleProverStorageContext<'_, '_, N, T>
{
    fn create_by_ref(&self, value: bool) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner
            .create_by_ref(value)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitXor<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Default,
{
    fn bitxor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.bitxor_by_ref(a, b)
    }
    fn bitxor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.bitxor_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitXorByRef<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Default,
{
    fn bitxor_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner
            .bitxor_by_ref(a, b)
            .map_err(|error| match error {})
    }
    fn bitxor_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitxor_assign_by_ref(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitAnd<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitand(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.bitand_by_ref(a, b)
    }
    fn bitand_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.bitand_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitAndByRef<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitand_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner
            .bitand_by_ref(a, b)
            .map_err(|error| match error {})
    }
    fn bitand_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitand_assign_by_ref(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitOr<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.bitor_by_ref(a, b)
    }
    fn bitor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.bitor_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitOrByRef<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitor_by_ref(
        &self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner
            .bitor_by_ref(a, b)
            .map_err(|error| match error {})
    }
    fn bitor_assign_by_ref(
        &self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitor_assign_by_ref(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithMux<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn mux(
        &mut self,
        condition: Vope<N, T, U1>,
        then: Vope<N, T, U1>,
        otherwise: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.mux_by_ref(condition, then, otherwise)
    }
}

impl<N, T> ContextWithMuxByRef<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn mux_by_ref(
        &self,
        condition: Vope<N, T, U1>,
        then: Vope<N, T, U1>,
        otherwise: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner
            .mux_by_ref(condition, then, otherwise)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithStorage<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    type Storage = LockedVoleProverStorage<N, T>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.storage_read_by_ref(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.storage_write_by_ref(storage, address, value)
    }
}

impl<N, T> ContextWithStorageByRef<bool> for LockedVoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        storage
            .read(address)
            .map_err(VoleStorageContextError::Storage)
    }

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        storage
            .write(address, value)
            .map_err(VoleStorageContextError::Storage)
    }
}

/// Storage-capable locked verifier adapter.
pub struct LockedVoleVerifierStorageContext<'a, 'b, N: ArraySize, T> {
    /// The underlying locked VOLE Boolean context.
    pub inner: LockedVoleVerifierContext<'a, 'b, N, T>,
}

impl<N: ArraySize, T> HasError for LockedVoleVerifierStorageContext<'_, '_, N, T> {
    type Error = VoleStorageContextError;
}

impl<N: ArraySize, T> ContextWithValue<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T> {
    type Wrapped = Q<N, T>;
}

impl<N, T> ContextWithCreate<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Default,
{
    fn create(&mut self, value: bool) -> Result<Q<N, T>, Self::Error> {
        self.create_by_ref(value)
    }
}

impl<N, T> ContextWithCreateByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Default,
{
    fn create_by_ref(&self, value: bool) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .create_by_ref(value)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T> ContextWithBitXor<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T>,
{
    fn bitxor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.bitxor_by_ref(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.bitxor_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitXorByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T>,
{
    fn bitxor_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitxor_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitxor_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitxor_assign_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T> ContextWithBitAnd<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn bitand(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.bitand_by_ref(a, b)
    }
    fn bitand_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.bitand_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitAndByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn bitand_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitand_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitand_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitand_assign_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T> ContextWithBitOr<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn bitor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.bitor_by_ref(a, b)
    }
    fn bitor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.bitor_assign_by_ref(a, b)
    }
}

impl<N, T> ContextWithBitOrByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn bitor_by_ref(&self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitor_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitor_assign_by_ref(&self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitor_assign_by_ref(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T> ContextWithMux<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn mux(
        &mut self,
        cond: Q<N, T>,
        then: Q<N, T>,
        r#else: Q<N, T>,
    ) -> Result<Q<N, T>, Self::Error> {
        self.mux_by_ref(cond, then, r#else)
    }
}

impl<N, T> ContextWithMuxByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
{
    fn mux_by_ref(
        &self,
        cond: Q<N, T>,
        then: Q<N, T>,
        r#else: Q<N, T>,
    ) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .mux_by_ref(cond, then, r#else)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T> ContextWithStorage<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    type Storage = LockedVoleVerifierStorage<N, T>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
    ) -> Result<Q<N, T>, Self::Error> {
        self.storage_read_by_ref(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
    ) -> Result<(), Self::Error> {
        self.storage_write_by_ref(storage, address, value)
    }
}

impl<N, T> ContextWithStorageByRef<bool> for LockedVoleVerifierStorageContext<'_, '_, N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
    ) -> Result<Q<N, T>, Self::Error> {
        storage
            .read(address)
            .map_err(VoleStorageContextError::Storage)
    }

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
    ) -> Result<(), Self::Error> {
        storage
            .write(address, value)
            .map_err(VoleStorageContextError::Storage)
    }
}
