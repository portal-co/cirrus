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

extern crate alloc;

mod hook;
#[cfg(feature = "iop-accumulator")]
pub mod iop_accumulator;
mod locked;
mod mode_b_relation;
#[cfg(feature = "spartan-whir-adapter")]
mod spartan_whir_adapter;
mod trace_audit;
mod typed;

pub use hook::{NoopVoleVerifierHook, VoleVerifierHook};
pub use locked::{
    LockedVoleProverContext, LockedVoleProverStorage, LockedVoleProverStorageContext,
    LockedVoleVerifierContext, LockedVoleVerifierStorage, LockedVoleVerifierStorageContext,
    MutexPuller, MutexPusher, PullerByRef,
};
pub use mode_b_relation::{
    CircuitId, KOALABEAR_MODULUS, KOALABEAR_QUINTIC_DEGREE, KoalaBearLinearCombination,
    KoalaBearR1cs, KoalaBearR1csRow, LinearCombination, MODE_B_RELATION_VERSION,
    ModeBPublicInstance, ModeBRelation, ModeBRelationError, PRIME_RAM_ADDRESS_BITS,
    PRIME_RAM_FORMAT, PRIME_RAM_LANE_BITS, PRIME_RAM_STORAGE_BITS, PRIME_RAM_TIME_BITS,
    PrimeFieldRamConfig, PrimeRamMaterialization, PrimeRamPermutationChallenges,
    PrimeRamPermutationR1cs, PrimeRamR1cs, PrimeRamRecord, PrimeRamRecordLayout,
    PrimeRamScanLayout, PrimeRamScanRow, PublicBinding, R1csRow, RamAccess, RamAccessKind,
    RamWitness, SpartanWhirMatrixEntry, SpartanWhirR1csShape, StorageRelation, UnifiedR1cs,
    UnifiedR1csWitness,
};
#[cfg(feature = "spartan-whir-adapter")]
pub use spartan_whir_adapter::{SpartanWhirAdapterShape, SpartanWhirAdapterWitness};
pub use trace_audit::{
    BoolarTraceAudit, MemoryAccess, MemoryAccessKind, MemoryPermutationAudit, TraceAuditError,
    TraceDigest, WireAuthPath, WireOpening, commit_boolar_trace,
};
pub use typed::TypedVoleValue;

use alloc::vec::Vec;
use core::{
    convert::Infallible,
    fmt,
    ops::{Add, Mul},
};

use cipher::consts::U1;
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithStorage, ContextWithValue, HasError, Pusher, StorageAddressBit,
};
use hybrid_array::{Array, ArraySize};
use volar_spec::{
    field::Invert,
    vole::{
        Delta, Q, VoleArray, Vope,
        bridge::{
            mem_acc_absorb_q, mem_acc_absorb_vope, mem_drain_check, mem_drain_open, q_scale_const,
            vope_scale_const,
        },
        prove::vole_and_prover_step,
        setup::derive_and_q,
    },
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
impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreate<bool>
    for VoleProverContext<'_, '_, N, T>
{
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
    fn bitxor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        // Free: `Vope`'s own `Add` impl (volar_spec::vole::vope).
        Ok(a + b)
    }
    fn bitxor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default>
    ContextWithBitAnd<bool> for VoleProverContext<'_, '_, N, T>
{
    fn bitand(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let (c, hat) = vole_and_prover_step(a, b);
        self.hats.push(hat);
        Ok(c)
    }
    fn bitand_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}
impl<N: VoleArray<T>, T: Clone + Add<Output = T> + Mul<Output = T> + Default> ContextWithBitOr<bool>
    for VoleProverContext<'_, '_, N, T>
{
    fn bitor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Infallible> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }
    fn bitor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Infallible> {
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
pub struct VoleVerifierContext<
    N: ArraySize,
    T,
    I: Iterator<Item = Array<T, N>>,
    H = NoopVoleVerifierHook,
> {
    /// The verifier's secret global offset.
    pub delta: Delta<N, T>,
    /// The ordered source of `hat` values, one per AND gate.
    pub hats: I,
    /// Optional observer for successfully replayed AND gates.
    pub hook: H,
    /// Number of successfully observed gates.
    pub gate_index: usize,
}

impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>, H> HasError
    for VoleVerifierContext<N, T, I, H>
{
    type Error = VoleVerifyError;
}
impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>, H> ContextWithValue<bool>
    for VoleVerifierContext<N, T, I, H>
{
    type Wrapped = Q<N, T>;
}
impl<N: ArraySize, T: Clone + Default, I: Iterator<Item = Array<T, N>>, H> ContextWithCreate<bool>
    for VoleVerifierContext<N, T, I, H>
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
impl<N: ArraySize, T: Clone + Add<Output = T>, I: Iterator<Item = Array<T, N>>, H>
    ContextWithBitXor<bool> for VoleVerifierContext<N, T, I, H>
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
    H: VoleVerifierHook<N, T>,
> ContextWithBitAnd<bool> for VoleVerifierContext<N, T, I, H>
{
    fn bitand(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, VoleVerifyError> {
        let hat = self.hats.next().ok_or(VoleVerifyError::HatExhausted)?;
        let c = derive_and_q(&self.delta, &a, &b, &hat);
        self.hook
            .on_and(self.gate_index, &self.delta, &a, &b, &c, &hat);
        self.gate_index += 1;
        Ok(c)
    }
    fn bitand_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), VoleVerifyError> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}

/// A caller-supplied predecessor entry for a storage read.
#[derive(Clone)]
pub struct StorageReadWitness<W> {
    /// The authenticated value returned by the read.
    pub value: W,
    /// Timestamp of the entry this read consumes.
    pub predecessor_timestamp: u64,
}

/// A caller-supplied predecessor entry for a storage write.
#[derive(Clone)]
pub struct StorageWriteWitness<W> {
    /// The authenticated value overwritten by the write.
    pub overwritten: W,
    /// Timestamp of the entry this write consumes.
    pub predecessor_timestamp: u64,
}

/// Public coefficients used to pack a symbolic address and absorb memory
/// tuples. The coefficient vector is LSB-first and must exactly match every
/// storage address presented to the backend.
#[derive(Clone)]
pub struct VoleStorageConfig<T> {
    /// Public coefficient for each LSB-first address bit.
    pub address_coefficients: Vec<T>,
    /// Non-zero constant term for each multiset entry.
    pub r0: T,
    /// Public coefficient for the packed address.
    pub r1: T,
    /// Public coefficient for the Boolean value.
    pub r2: T,
    /// Public coefficient for the access timestamp.
    pub r3: T,
    /// Encodes a public access timestamp in the field.
    pub timestamp_to_t: fn(u64) -> T,
}

/// Why an authenticated storage trace could not be consumed or finalized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoleStorageError {
    /// A caller provided a symbolic address with the wrong width.
    AddressWidth {
        /// Width required by the configured address packer.
        expected: usize,
        /// Width supplied by the storage access.
        found: usize,
    },
    /// A read was attempted after its witness stream ended.
    ReadWitnessExhausted,
    /// A write was attempted after its witness stream ended.
    WriteWitnessExhausted,
    /// A predecessor was not earlier than the access consuming it.
    InvalidPredecessorTimestamp {
        /// Timestamp attached to the witness's overwritten entry.
        predecessor: u64,
        /// Timestamp allocated to the storage access.
        access: u64,
    },
    /// Timestamp allocation overflowed.
    TimestampOverflow,
    /// Finalization found unconsumed caller-provided witnesses.
    UnconsumedWitnesses {
        /// Number of remaining read witnesses.
        reads: usize,
        /// Number of remaining write witnesses.
        writes: usize,
    },
    /// The verifier's final multiset opening did not match its accumulators.
    InvalidDrain,
}

impl fmt::Display for VoleStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddressWidth { expected, found } => write!(
                formatter,
                "storage address width {found} does not match {expected}"
            ),
            Self::ReadWitnessExhausted => {
                formatter.write_str("storage read witness stream is exhausted")
            }
            Self::WriteWitnessExhausted => {
                formatter.write_str("storage write witness stream is exhausted")
            }
            Self::InvalidPredecessorTimestamp {
                predecessor,
                access,
            } => write!(
                formatter,
                "storage predecessor timestamp {predecessor} is not earlier than access {access}"
            ),
            Self::TimestampOverflow => formatter.write_str("storage timestamp overflow"),
            Self::UnconsumedWitnesses { reads, writes } => write!(
                formatter,
                "storage witness streams retain {reads} reads and {writes} writes"
            ),
            Self::InvalidDrain => formatter
                .write_str("storage drain opening does not match its authenticated accumulators"),
        }
    }
}

impl core::error::Error for VoleStorageError {}

/// Prover-owned authenticated storage state and its ordered access witnesses.
pub struct VoleProverStorage<N: VoleArray<T>, T> {
    config: VoleStorageConfig<T>,
    reads: Vec<StorageReadWitness<Vope<N, T, U1>>>,
    writes: Vec<StorageWriteWitness<Vope<N, T, U1>>>,
    next_read: usize,
    next_write: usize,
    next_timestamp: u64,
    zero: Vope<N, T, U1>,
    one: Vope<N, T, U1>,
    produce: Vope<N, T, U1>,
    consume: Vope<N, T, U1>,
}

impl<N, T> VoleProverStorage<N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    /// Construct storage from caller-owned, ordered read and write witnesses.
    pub fn new(
        zero: Vope<N, T, U1>,
        one: Vope<N, T, U1>,
        config: VoleStorageConfig<T>,
        reads: impl IntoIterator<Item = StorageReadWitness<Vope<N, T, U1>>>,
        writes: impl IntoIterator<Item = StorageWriteWitness<Vope<N, T, U1>>>,
    ) -> Self {
        Self {
            config,
            reads: reads.into_iter().collect(),
            writes: writes.into_iter().collect(),
            next_read: 0,
            next_write: 0,
            next_timestamp: 1,
            produce: zero.clone(),
            consume: zero.clone(),
            zero,
            one,
        }
    }

    fn timestamp(&mut self) -> Result<u64, VoleStorageError> {
        let timestamp = self.next_timestamp;
        self.next_timestamp = self
            .next_timestamp
            .checked_add(1)
            .ok_or(VoleStorageError::TimestampOverflow)?;
        Ok(timestamp)
    }

    fn address(
        &self,
        bits: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, VoleStorageError> {
        if bits.len() != self.config.address_coefficients.len() {
            return Err(VoleStorageError::AddressWidth {
                expected: self.config.address_coefficients.len(),
                found: bits.len(),
            });
        }
        let mut packed = self.zero.clone();
        for (bit, coefficient) in bits.iter().zip(&self.config.address_coefficients) {
            packed = packed + vope_scale_const(&bit.wire, coefficient);
        }
        Ok(packed)
    }

    fn absorb_produce(&mut self, address: &Vope<N, T, U1>, value: &Vope<N, T, U1>, timestamp: u64) {
        let previous = core::mem::replace(&mut self.produce, self.zero.clone());
        let timestamp = vope_scale_const(&self.one, &(self.config.timestamp_to_t)(timestamp));
        self.produce = mem_acc_absorb_vope(
            previous,
            &self.one,
            address,
            value,
            &timestamp,
            &self.config.r0,
            &self.config.r1,
            &self.config.r2,
            &self.config.r3,
        );
    }

    fn absorb_consume(&mut self, address: &Vope<N, T, U1>, value: &Vope<N, T, U1>, timestamp: u64) {
        let previous = core::mem::replace(&mut self.consume, self.zero.clone());
        let timestamp = vope_scale_const(&self.one, &(self.config.timestamp_to_t)(timestamp));
        self.consume = mem_acc_absorb_vope(
            previous,
            &self.one,
            address,
            value,
            &timestamp,
            &self.config.r0,
            &self.config.r1,
            &self.config.r2,
            &self.config.r3,
        );
    }

    /// Add an initial live storage entry to the producer accumulator.
    ///
    /// Every witness predecessor used by a read or write must originate from
    /// either an earlier access or one such initial entry. The caller drains
    /// the final live entry with [`Self::drain`].
    pub fn initialize(
        &mut self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        self.absorb_produce(&address, &value, timestamp);
        Ok(())
    }

    /// Drain one surviving authenticated entry into the consume accumulator.
    pub fn drain(
        &mut self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        self.absorb_consume(&address, &value, timestamp);
        Ok(())
    }

    /// Ensure all witnesses were consumed and open the final accumulator mask.
    pub fn finish(&self) -> Result<Array<T, N>, VoleStorageError> {
        let reads = self.reads.len().saturating_sub(self.next_read);
        let writes = self.writes.len().saturating_sub(self.next_write);
        if reads != 0 || writes != 0 {
            return Err(VoleStorageError::UnconsumedWitnesses { reads, writes });
        }
        Ok(mem_drain_open(&self.produce, &self.consume))
    }
}

impl<N, T> VoleProverStorage<N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    fn read(
        &mut self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, VoleStorageError> {
        let address = self.address(address)?;
        let witness = self
            .reads
            .get(self.next_read)
            .cloned()
            .ok_or(VoleStorageError::ReadWitnessExhausted)?;
        let timestamp = self.timestamp()?;
        if witness.predecessor_timestamp >= timestamp {
            return Err(VoleStorageError::InvalidPredecessorTimestamp {
                predecessor: witness.predecessor_timestamp,
                access: timestamp,
            });
        }
        self.next_read += 1;
        self.absorb_produce(&address, &witness.value, timestamp);
        self.absorb_consume(&address, &witness.value, witness.predecessor_timestamp);
        Ok(witness.value)
    }

    fn write(
        &mut self,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        let witness = self
            .writes
            .get(self.next_write)
            .cloned()
            .ok_or(VoleStorageError::WriteWitnessExhausted)?;
        let timestamp = self.timestamp()?;
        if witness.predecessor_timestamp >= timestamp {
            return Err(VoleStorageError::InvalidPredecessorTimestamp {
                predecessor: witness.predecessor_timestamp,
                access: timestamp,
            });
        }
        self.next_write += 1;
        self.absorb_produce(&address, &value, timestamp);
        self.absorb_consume(
            &address,
            &witness.overwritten,
            witness.predecessor_timestamp,
        );
        Ok(())
    }
}

/// Verifier-owned authenticated storage state and its ordered access witnesses.
pub struct VoleVerifierStorage<N: ArraySize, T> {
    config: VoleStorageConfig<T>,
    reads: Vec<StorageReadWitness<Q<N, T>>>,
    writes: Vec<StorageWriteWitness<Q<N, T>>>,
    next_read: usize,
    next_write: usize,
    next_timestamp: u64,
    zero: Q<N, T>,
    one: Q<N, T>,
    produce: Q<N, T>,
    consume: Q<N, T>,
}

impl<N, T> VoleVerifierStorage<N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Construct verifier storage from caller-owned, ordered witnesses.
    pub fn new(
        zero: Q<N, T>,
        one: Q<N, T>,
        config: VoleStorageConfig<T>,
        reads: impl IntoIterator<Item = StorageReadWitness<Q<N, T>>>,
        writes: impl IntoIterator<Item = StorageWriteWitness<Q<N, T>>>,
    ) -> Self {
        Self {
            config,
            reads: reads.into_iter().collect(),
            writes: writes.into_iter().collect(),
            next_read: 0,
            next_write: 0,
            next_timestamp: 1,
            produce: zero.clone(),
            consume: zero.clone(),
            zero,
            one,
        }
    }

    fn timestamp(&mut self) -> Result<u64, VoleStorageError> {
        let timestamp = self.next_timestamp;
        self.next_timestamp = self
            .next_timestamp
            .checked_add(1)
            .ok_or(VoleStorageError::TimestampOverflow)?;
        Ok(timestamp)
    }

    fn address(&self, bits: &[StorageAddressBit<Q<N, T>>]) -> Result<Q<N, T>, VoleStorageError> {
        if bits.len() != self.config.address_coefficients.len() {
            return Err(VoleStorageError::AddressWidth {
                expected: self.config.address_coefficients.len(),
                found: bits.len(),
            });
        }
        let mut packed = self.zero.clone();
        for (bit, coefficient) in bits.iter().zip(&self.config.address_coefficients) {
            let scaled = q_scale_const(&bit.wire, coefficient);
            packed = Q {
                q: Array::<T, N>::from_fn(|i| packed.q[i].clone() + scaled.q[i].clone()),
            };
        }
        Ok(packed)
    }

    fn absorb(
        &self,
        accumulator: Q<N, T>,
        address: &Q<N, T>,
        value: &Q<N, T>,
        timestamp: u64,
    ) -> Q<N, T> {
        let timestamp = q_scale_const(&self.one, &(self.config.timestamp_to_t)(timestamp));
        mem_acc_absorb_q(
            accumulator,
            &self.one,
            address,
            value,
            &timestamp,
            &self.config.r0,
            &self.config.r1,
            &self.config.r2,
            &self.config.r3,
        )
    }

    /// Add an initial live storage entry to the producer accumulator.
    ///
    /// Every witness predecessor used by a read or write must originate from
    /// either an earlier access or one such initial entry. The caller drains
    /// the final live entry with [`Self::drain`].
    pub fn initialize(
        &mut self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        self.produce = self.absorb(self.produce.clone(), &address, &value, timestamp);
        Ok(())
    }

    /// Drain one surviving authenticated entry into the consume accumulator.
    pub fn drain(
        &mut self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
        timestamp: u64,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        self.consume = self.absorb(self.consume.clone(), &address, &value, timestamp);
        Ok(())
    }

    /// Consume every witness and verify a prover-supplied final drain opening.
    pub fn finish(&self, opening: &Array<T, N>) -> Result<(), VoleStorageError>
    where
        T: PartialEq,
    {
        let reads = self.reads.len().saturating_sub(self.next_read);
        let writes = self.writes.len().saturating_sub(self.next_write);
        if reads != 0 || writes != 0 {
            return Err(VoleStorageError::UnconsumedWitnesses { reads, writes });
        }
        mem_drain_check(&self.produce, &self.consume, opening)
            .then_some(())
            .ok_or(VoleStorageError::InvalidDrain)
    }
}

/// Error emitted by the storage-capable VOLE context adapters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VoleStorageContextError {
    /// The ordered storage witness trace or drain was invalid.
    Storage(VoleStorageError),
    /// The verifier's AND-hat stream was malformed.
    Verification(VoleVerifyError),
}

impl fmt::Display for VoleStorageContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(error) => error.fmt(formatter),
            Self::Verification(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for VoleStorageContextError {}

/// Storage-capable prover adapter. It preserves the ordinary VOLE gate
/// transcript while routing storage accesses through [`VoleProverStorage`].
pub struct VoleProverStorageContext<'a, 'b, N: VoleArray<T>, T> {
    /// The underlying VOLE Boolean context.
    pub inner: VoleProverContext<'a, 'b, N, T>,
}

impl<N: VoleArray<T>, T> HasError for VoleProverStorageContext<'_, '_, N, T> {
    type Error = VoleStorageContextError;
}

impl<N: VoleArray<T>, T> ContextWithValue<bool> for VoleProverStorageContext<'_, '_, N, T> {
    type Wrapped = Vope<N, T, U1>;
}

impl<N: VoleArray<T>, T: Clone + Default> ContextWithCreate<bool>
    for VoleProverStorageContext<'_, '_, N, T>
{
    fn create(&mut self, value: bool) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner.create(value).map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitXor<bool> for VoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Default,
{
    fn bitxor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner.bitxor(a, b).map_err(|error| match error {})
    }
    fn bitxor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitxor_assign(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitAnd<bool> for VoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitand(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner.bitand(a, b).map_err(|error| match error {})
    }
    fn bitand_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitand_assign(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithBitOr<bool> for VoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn bitor(
        &mut self,
        a: Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        self.inner.bitor(a, b).map_err(|error| match error {})
    }
    fn bitor_assign(
        &mut self,
        a: &mut Vope<N, T, U1>,
        b: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitor_assign(a, b)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithMux<bool> for VoleProverStorageContext<'_, '_, N, T>
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
        self.inner
            .mux(condition, then, otherwise)
            .map_err(|error| match error {})
    }
}

impl<N, T> ContextWithStorage<bool> for VoleProverStorageContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    Vope<N, T, U1>: Add<Output = Vope<N, T, U1>>,
{
    type Storage = VoleProverStorage<N, T>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
    ) -> Result<Vope<N, T, U1>, Self::Error> {
        storage
            .read(address)
            .map_err(VoleStorageContextError::Storage)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Vope<N, T, U1>>],
        value: Vope<N, T, U1>,
    ) -> Result<(), Self::Error> {
        storage
            .write(address, value)
            .map_err(VoleStorageContextError::Storage)
    }
}

/// Storage-capable verifier adapter. It consumes the same storage trace as
/// the prover adapter and keeps AND hats separate from storage operations.
pub struct VoleVerifierStorageContext<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> {
    /// The underlying VOLE Boolean context.
    pub inner: VoleVerifierContext<N, T, I>,
}

impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> HasError
    for VoleVerifierStorageContext<N, T, I>
{
    type Error = VoleStorageContextError;
}

impl<N: ArraySize, T, I: Iterator<Item = Array<T, N>>> ContextWithValue<bool>
    for VoleVerifierStorageContext<N, T, I>
{
    type Wrapped = Q<N, T>;
}

impl<N, T, I> ContextWithCreate<bool> for VoleVerifierStorageContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Default,
    I: Iterator<Item = Array<T, N>>,
{
    fn create(&mut self, value: bool) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .create(value)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T, I> ContextWithBitXor<bool> for VoleVerifierStorageContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Add<Output = T>,
    I: Iterator<Item = Array<T, N>>,
{
    fn bitxor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitxor(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitxor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitxor_assign(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T, I> ContextWithBitAnd<bool> for VoleVerifierStorageContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
{
    fn bitand(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitand(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitand_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitand_assign(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T, I> ContextWithBitOr<bool> for VoleVerifierStorageContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
{
    fn bitor(&mut self, a: Q<N, T>, b: Q<N, T>) -> Result<Q<N, T>, Self::Error> {
        self.inner
            .bitor(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
    fn bitor_assign(&mut self, a: &mut Q<N, T>, b: Q<N, T>) -> Result<(), Self::Error> {
        self.inner
            .bitor_assign(a, b)
            .map_err(VoleStorageContextError::Verification)
    }
}

impl<N, T, I> ContextWithStorage<bool> for VoleVerifierStorageContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
    I: Iterator<Item = Array<T, N>>,
{
    type Storage = VoleVerifierStorage<N, T>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
    ) -> Result<Q<N, T>, Self::Error> {
        storage
            .read(address)
            .map_err(VoleStorageContextError::Storage)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
    ) -> Result<(), Self::Error> {
        storage
            .write(address, value)
            .map_err(VoleStorageContextError::Storage)
    }
}

impl<N, T> VoleVerifierStorage<N, T>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    fn read(
        &mut self,
        address: &[StorageAddressBit<Q<N, T>>],
    ) -> Result<Q<N, T>, VoleStorageError> {
        let address = self.address(address)?;
        let witness = self
            .reads
            .get(self.next_read)
            .cloned()
            .ok_or(VoleStorageError::ReadWitnessExhausted)?;
        let timestamp = self.timestamp()?;
        if witness.predecessor_timestamp >= timestamp {
            return Err(VoleStorageError::InvalidPredecessorTimestamp {
                predecessor: witness.predecessor_timestamp,
                access: timestamp,
            });
        }
        self.next_read += 1;
        self.produce = self.absorb(self.produce.clone(), &address, &witness.value, timestamp);
        self.consume = self.absorb(
            self.consume.clone(),
            &address,
            &witness.value,
            witness.predecessor_timestamp,
        );
        Ok(witness.value)
    }

    fn write(
        &mut self,
        address: &[StorageAddressBit<Q<N, T>>],
        value: Q<N, T>,
    ) -> Result<(), VoleStorageError> {
        let address = self.address(address)?;
        let witness = self
            .writes
            .get(self.next_write)
            .cloned()
            .ok_or(VoleStorageError::WriteWitnessExhausted)?;
        let timestamp = self.timestamp()?;
        if witness.predecessor_timestamp >= timestamp {
            return Err(VoleStorageError::InvalidPredecessorTimestamp {
                predecessor: witness.predecessor_timestamp,
                access: timestamp,
            });
        }
        self.next_write += 1;
        self.produce = self.absorb(self.produce.clone(), &address, &value, timestamp);
        self.consume = self.absorb(
            self.consume.clone(),
            &address,
            &witness.overwritten,
            witness.predecessor_timestamp,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cipher::consts::U1;
    use volar_spec::field::Galois128;

    struct Sink;

    impl<T> Pusher<T> for Sink {
        fn push(&mut self, _: T) {}
    }

    fn field_bit(value: bool) -> Galois128 {
        Galois128(value as u128)
    }

    fn prover_bit(value: bool) -> Vope<U1, Galois128, U1> {
        Vope {
            u: Array::from_fn(|_| Array::from_fn(|_| field_bit(value))),
            v: Array::from_fn(|_| Galois128(0)),
        }
    }

    fn verifier_bit(value: bool, delta: Galois128) -> Q<U1, Galois128> {
        Q {
            q: Array::from_fn(|_| if value { delta } else { Galois128(0) }),
        }
    }

    fn storage_config() -> VoleStorageConfig<Galois128> {
        VoleStorageConfig {
            address_coefficients: alloc::vec![Galois128(7)],
            r0: Galois128(11),
            r1: Galois128(13),
            r2: Galois128(17),
            r3: Galois128(19),
            timestamp_to_t: |timestamp| Galois128(timestamp as u128),
        }
    }

    #[test]
    fn direct_storage_contexts_consume_a_trace_and_verify_its_drain() {
        let delta = Galois128(23);
        let prover_zero = prover_bit(false);
        let prover_one = prover_bit(true);
        let verifier_zero = verifier_bit(false, delta);
        let verifier_one = verifier_bit(true, delta);

        let mut prover_storage = VoleProverStorage::new(
            prover_zero.clone(),
            prover_one.clone(),
            storage_config(),
            [StorageReadWitness {
                value: prover_one.clone(),
                predecessor_timestamp: 1,
            }],
            [StorageWriteWitness {
                overwritten: prover_zero.clone(),
                predecessor_timestamp: 0,
            }],
        );
        let mut sink = Sink;
        let mut prover = VoleProverStorageContext {
            inner: VoleProverContext {
                hats: &mut sink,
                bit_to_t: field_bit,
            },
        };
        let prover_address = [StorageAddressBit {
            wire: prover_one.clone(),
            known: Some(true),
        }];
        prover_storage
            .initialize(&prover_address, prover_zero.clone(), 0)
            .unwrap();
        prover
            .storage_write(&mut prover_storage, &prover_address, prover_one.clone())
            .unwrap();
        prover
            .storage_read(&mut prover_storage, &prover_address)
            .unwrap();
        prover_storage
            .drain(&prover_address, prover_one.clone(), 2)
            .unwrap();
        let opening = prover_storage.finish().unwrap();

        let mut verifier_storage = VoleVerifierStorage::new(
            verifier_zero.clone(),
            verifier_one.clone(),
            storage_config(),
            [StorageReadWitness {
                value: verifier_one.clone(),
                predecessor_timestamp: 1,
            }],
            [StorageWriteWitness {
                overwritten: verifier_zero.clone(),
                predecessor_timestamp: 0,
            }],
        );
        let mut verifier = VoleVerifierStorageContext {
            inner: VoleVerifierContext {
                delta: Delta {
                    delta: Array::from_fn(|_| delta),
                },
                hats: alloc::vec::Vec::new().into_iter(),
                hook: NoopVoleVerifierHook,
                gate_index: 0,
            },
        };
        let verifier_address = [StorageAddressBit {
            wire: verifier_one.clone(),
            known: Some(true),
        }];
        verifier_storage
            .initialize(&verifier_address, verifier_zero.clone(), 0)
            .unwrap();
        verifier
            .storage_write(
                &mut verifier_storage,
                &verifier_address,
                verifier_one.clone(),
            )
            .unwrap();
        verifier
            .storage_read(&mut verifier_storage, &verifier_address)
            .unwrap();
        verifier_storage
            .drain(&verifier_address, verifier_one, 2)
            .unwrap();
        verifier_storage.finish(&opening).unwrap();
    }
}
impl<
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = Array<T, N>>,
    H: VoleVerifierHook<N, T>,
> ContextWithBitOr<bool> for VoleVerifierContext<N, T, I, H>
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
    H: VoleVerifierHook<N, T>,
> ContextWithMux<bool> for VoleVerifierContext<N, T, I, H>
{
    fn mux(
        &mut self,
        cond: Q<N, T>,
        then: Q<N, T>,
        r#else: Q<N, T>,
    ) -> Result<Q<N, T>, VoleVerifyError> {
        let diff = self.bitxor(then, r#else.clone())?;
        let masked = self.bitand(cond, diff)?;
        self.bitxor(r#else, masked)
    }
}
