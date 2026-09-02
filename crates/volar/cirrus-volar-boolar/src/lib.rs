#![no_std]

//! Execute fused Boolar circuits through Cirrus Boolean contexts.
//!
//! The interpreter treats [`BCircuit`] parameters as unknown symbolic bits.
//! Facts derived from Boolar `Zero`/`One` statements are kept internally so
//! storage accesses only branch on address bits that are not statically known.

extern crate alloc;

mod address_trim;
mod sparse;
mod typed;
mod types;

pub use address_trim::{
    trim_storage_addr_width, trim_storage_element_bits, AddressTrimError, WASM_BYTE_ADDRESS_BITS,
};
pub use sparse::{SparseBank, SparseMuxTreeContext, SparseStorageError};
pub use typed::{lower_volar_circuit, TypedLowerError};
pub use types::{lower_volar_types, VolarTypeMap, VolarTypeMapError};

use alloc::{string::String, vec::Vec};
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitAndByRef, ContextWithBitOr, ContextWithBitOrByRef,
    ContextWithBitXor, ContextWithBitXorByRef, ContextWithCreate, ContextWithCreateByRef,
    ContextWithStorage, ContextWithValue, HasError, StorageAddressBit,
};
use lazy_repo::{ChunkCodec, Repository};
use lazy_repo_crypto::{EncryptedScratch, ScratchSlot, ScratchStore};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
    lazy::{BStmtChunk, ChunkLoadError, ChunkedBCircuit, WireSchedule},
};
use volar_ir_common::StorageId;

pub(crate) type Wire<C> = <C as ContextWithValue<bool>>::Wrapped;

/// The Boolean operations required by [`execute`].
pub trait BoolarContext:
    ContextWithCreate<bool> + ContextWithBitAnd<bool> + ContextWithBitOr<bool> + ContextWithBitXor<bool>
{
}

impl<T> BoolarContext for T where
    T: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>
{
}

/// Caller-supplied dispatch for Boolar external primitives.
///
/// The registry is passed to execution rather than embedded in the circuit,
/// so serialized circuits retain only stable source metadata.  Every method
/// is bit-granular and receives the occurrence token emitted by lowering.
/// Action handlers own the conditional storage behavior: they receive both
/// `guard` and `fallback` and must write the selected bit to `cells`.
pub trait ExternalBitRegistry<C>
where
    C: BoolarContext + ContextWithStorage<bool>,
    Wire<C>: Clone,
{
    /// Evaluate one declared oracle result bit.
    fn oracle_bit(
        &mut self,
        context: &mut C,
        name: &str,
        args: &[Wire<C>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Wire<C>, C::Error>;

    /// Evaluate one declared fresh RNG bit.
    fn rng_bit(
        &mut self,
        context: &mut C,
        name: &str,
        bit: usize,
        occurrence: u64,
    ) -> Result<Wire<C>, C::Error>;

    /// Execute one direct action-storage effect.
    #[allow(clippy::too_many_arguments)]
    fn action_store_bit(
        &mut self,
        context: &mut C,
        name: &str,
        guard: Wire<C>,
        args: &[Wire<C>],
        fallback: Wire<C>,
        storage: StorageId,
        lane: LaneId,
        address: &[StorageAddressBit<Wire<C>>],
        bit: usize,
        occurrence: u64,
        cells: &mut C::Storage,
    ) -> Result<(), C::Error>;
}

impl<C> ExternalBitRegistry<C> for ()
where
    C: BoolarContext + ContextWithStorage<bool>,
    Wire<C>: Clone,
{
    fn oracle_bit(
        &mut self,
        _context: &mut C,
        _name: &str,
        _args: &[Wire<C>],
        _bit: usize,
        _occurrence: u64,
    ) -> Result<Wire<C>, C::Error> {
        unreachable!("external dispatch is checked before the empty registry is used")
    }

    fn rng_bit(
        &mut self,
        _context: &mut C,
        _name: &str,
        _bit: usize,
        _occurrence: u64,
    ) -> Result<Wire<C>, C::Error> {
        unreachable!("external dispatch is checked before the empty registry is used")
    }

    fn action_store_bit(
        &mut self,
        _context: &mut C,
        _name: &str,
        _guard: Wire<C>,
        _args: &[Wire<C>],
        _fallback: Wire<C>,
        _storage: StorageId,
        _lane: LaneId,
        _address: &[StorageAddressBit<Wire<C>>],
        _bit: usize,
        _occurrence: u64,
        _cells: &mut C::Storage,
    ) -> Result<(), C::Error> {
        unreachable!("external dispatch is checked before the empty registry is used")
    }
}

/// The ordinary dense-storage implementation of [`ContextWithStorage`].
///
/// Wrap a Boolean context in this adapter when storage should lower to the
/// traditional MUX/demux trees.  The cells remain a separate caller-owned
/// slice (`Storage = [Wire]`); the wrapper only supplies the Boolean gates.
pub struct MuxTreeContext<C> {
    inner: C,
}

impl<C> MuxTreeContext<C> {
    pub fn new(inner: C) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut C {
        &mut self.inner
    }

    pub fn into_inner(self) -> C {
        self.inner
    }
}

impl<C: HasError> HasError for MuxTreeContext<C> {
    type Error = C::Error;
}

impl<C, Val> ContextWithValue<Val> for MuxTreeContext<C>
where
    C: ContextWithValue<Val>,
{
    type Wrapped = C::Wrapped;
}

impl<C> ContextWithCreate<bool> for MuxTreeContext<C>
where
    C: ContextWithCreate<bool>,
{
    fn create(&mut self, value: bool) -> Result<Self::Wrapped, Self::Error> {
        self.inner.create(value)
    }
}

impl<C> ContextWithCreateByRef<bool> for MuxTreeContext<C>
where
    C: ContextWithCreateByRef<bool>,
{
    fn create_by_ref(&self, value: bool) -> Result<Self::Wrapped, Self::Error> {
        self.inner.create_by_ref(value)
    }
}

impl<C> ContextWithBitAnd<bool> for MuxTreeContext<C>
where
    C: ContextWithBitAnd<bool>,
{
    fn bitand(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitand(left, right)
    }

    fn bitand_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitand_assign(left, right)
    }
}

impl<C> ContextWithBitAndByRef<bool> for MuxTreeContext<C>
where
    C: ContextWithBitAndByRef<bool>,
{
    fn bitand_by_ref(
        &self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitand_by_ref(left, right)
    }

    fn bitand_assign_by_ref(
        &self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitand_assign_by_ref(left, right)
    }
}

impl<C> ContextWithBitOr<bool> for MuxTreeContext<C>
where
    C: ContextWithBitOr<bool>,
{
    fn bitor(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitor(left, right)
    }

    fn bitor_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitor_assign(left, right)
    }
}

impl<C> ContextWithBitOrByRef<bool> for MuxTreeContext<C>
where
    C: ContextWithBitOrByRef<bool>,
{
    fn bitor_by_ref(
        &self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitor_by_ref(left, right)
    }

    fn bitor_assign_by_ref(
        &self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitor_assign_by_ref(left, right)
    }
}

impl<C> ContextWithBitXor<bool> for MuxTreeContext<C>
where
    C: ContextWithBitXor<bool>,
{
    fn bitxor(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitxor(left, right)
    }

    fn bitxor_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitxor_assign(left, right)
    }
}

impl<C> ContextWithBitXorByRef<bool> for MuxTreeContext<C>
where
    C: ContextWithBitXorByRef<bool>,
{
    fn bitxor_by_ref(
        &self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner.bitxor_by_ref(left, right)
    }

    fn bitxor_assign_by_ref(
        &self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner.bitxor_assign_by_ref(left, right)
    }
}

/// Caller-owned **dense** storage for one Boolar storage lane.
///
/// `value` is `2^address_bits` cells. WASM/spill lowering emits 32-bit element
/// pointers; Boolar appends bit-index high bits (e.g. 34 wires). Dense fit is
/// [`trim_storage_element_bits`] plus [`trim_storage_addr_width`] at
/// [`WASM_BYTE_ADDRESS_BITS`] (16 MiB). Sparse cell backing (no `2^n`
/// allocation) is the handoff in `sparse-banks.md`.
///
/// The namespace routing is intentionally kept here rather than in
/// [`ContextWithStorage`]: an implementation can therefore use the same
/// storage value for several Volar storage namespaces. `address_bits` is the
/// declared width of the lane's LSB-first Boolean address.
pub struct StorageBank<'a, S: ?Sized> {
    /// Storage namespace selected by the Boolar statement.
    pub storage: StorageId,
    /// Type-disambiguating one-bit storage lane.
    pub lane: LaneId,
    /// The declared width of this lane's symbolic address.
    pub address_bits: usize,
    /// The external storage value used for this lane.
    pub value: &'a mut S,
}

/// The dense storage allocation required for one Boolar storage lane.
///
/// `cells` is always a non-zero power of two and is suitable for directly
/// constructing a [`StorageBank`].  Requirements include lanes referenced
/// only by [`BCircuit::pre_init`], so callers can allocate storage before the
/// first execution rather than discovering a missing bank midway through it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageRequirement {
    pub storage: StorageId,
    pub lane: LaneId,
    pub address_bits: usize,
    pub cells: usize,
}

/// Why a circuit's storage layout cannot be represented by dense banks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StorageLayoutError {
    /// An address has too many bits to form a platform `usize` cell count.
    AddressWidthOverflow { bits: usize },
    /// The same storage lane was used with incompatible address widths.
    ConflictingAddressWidth {
        storage: StorageId,
        lane: LaneId,
        first: usize,
        second: usize,
    },
    /// A pre-initialisation segment cannot be represented in a `usize` bank.
    PreInitOffsetOverflow { storage: StorageId, lane: LaneId },
    /// Static data extends beyond the lane capacity implied by its addresses.
    PreInitOutOfBounds {
        storage: StorageId,
        lane: LaneId,
        offset: u64,
        data_len: usize,
        cells: usize,
    },
}

/// Why Boolar execution could not complete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecuteError<E> {
    /// The caller supplied the wrong number of parameter wires.
    InputArity { expected: usize, found: usize },
    /// A statement or output referred to a value that is not available yet.
    UndefinedVariable { var: IRVarId, available: usize },
    /// More than one caller bank has the same storage-lane key.
    DuplicateStorageBank { storage: StorageId, lane: LaneId },
    /// A circuit storage access had no corresponding caller bank.
    MissingStorageBank { storage: StorageId, lane: LaneId },
    /// The address is too wide to describe a platform `usize` cell count.
    AddressWidthOverflow { bits: usize },
    /// A storage bank's declared address width does not match a statement.
    StorageAddressWidth {
        storage: StorageId,
        lane: LaneId,
        address_bits: usize,
        declared_address_bits: usize,
    },
    /// The circuit's storage accesses and static initialization cannot share
    /// one dense layout.
    StorageLayout(StorageLayoutError),
    /// The circuit uses a Boolar statement that has no Cirrus context meaning.
    UnsupportedStatement,
    /// The circuit needs a source registry but none was passed to execution.
    MissingExternalHandler { kind: &'static str, name: String },
    /// The underlying Cirrus Boolean context rejected an operation.
    Context(E),
}

#[derive(Clone)]
pub(crate) struct Value<W> {
    pub(crate) wire: W,
    pub(crate) known: Option<bool>,
}

impl<C> ContextWithStorage<bool> for MuxTreeContext<C>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    type Storage = [Wire<C>];

    fn storage_read(
        &mut self,
        cells: &mut Self::Storage,
        address: &[StorageAddressBit<Wire<C>>],
    ) -> Result<Wire<C>, Self::Error> {
        if let Some(index) = known_storage_address(address) {
            if let Some(value) = cells.get(index) {
                return Ok(value.clone());
            }
        }
        let address: Vec<_> = address
            .iter()
            .cloned()
            .map(|bit| Value {
                wire: bit.wire,
                known: bit.known,
            })
            .collect();
        let facts = alloc::vec![None; cells.len()];
        read_mux_storage(self, cells, &facts, &address)
            .map(|value| value.wire)
            .map_err(|error| match error {
                ExecuteError::Context(error) => error,
                _ => unreachable!("the MUX tree only reports context errors"),
            })
    }

    fn storage_write(
        &mut self,
        cells: &mut Self::Storage,
        address: &[StorageAddressBit<Wire<C>>],
        value: Wire<C>,
    ) -> Result<(), Self::Error> {
        if let Some(index) = known_storage_address(address) {
            if let Some(cell) = cells.get_mut(index) {
                *cell = value;
                return Ok(());
            }
        }
        let address: Vec<_> = address
            .iter()
            .cloned()
            .map(|bit| Value {
                wire: bit.wire,
                known: bit.known,
            })
            .collect();
        let mut facts = alloc::vec![None; cells.len()];
        let mut canonical_one = None;
        write_mux_storage(
            self,
            cells,
            &mut facts,
            Value {
                wire: value,
                known: None,
            },
            &address,
            &mut canonical_one,
        )
        .map_err(|error| match error {
            ExecuteError::Context(error) => error,
            _ => unreachable!("the MUX tree only reports context errors"),
        })
    }
}

pub(crate) fn known_storage_address<W>(address: &[StorageAddressBit<W>]) -> Option<usize> {
    let mut index = 0usize;
    for (bit, value) in address.iter().enumerate() {
        let value = value.known?;
        if value {
            if bit >= usize::BITS as usize {
                return None;
            }
            index |= 1usize << bit;
        }
    }
    Some(index)
}

/// Derive the dense storage banks needed by `circuit`.
///
/// Every storage statement for one `(storage, lane)` pair must use the same
/// address width. Static initialization is checked against that capacity.
pub fn storage_requirements<P: Clone>(
    circuit: &BCircuit<P>,
) -> Result<Vec<StorageRequirement>, StorageLayoutError> {
    let mut requirements: Vec<(StorageRequirement, bool)> = Vec::new();

    for node in &circuit.stmts {
        let (storage, lane, address_bits) = match &node.kind {
            BIrStmt::StorageRead {
                storage,
                lane,
                addr,
            }
            | BIrStmt::StorageWrite {
                storage,
                lane,
                addr,
                ..
            }
            | BIrStmt::ActionStoreBit {
                storage,
                lane,
                addr,
                ..
            } => (*storage, *lane, addr.len()),
            _ => continue,
        };
        let cells = cells_for_address_bits(address_bits)?;
        if let Some((requirement, has_access)) = requirements
            .iter_mut()
            .find(|(requirement, _)| requirement.storage == storage && requirement.lane == lane)
        {
            if *has_access && requirement.address_bits != address_bits {
                return Err(StorageLayoutError::ConflictingAddressWidth {
                    storage,
                    lane,
                    first: requirement.address_bits,
                    second: address_bits,
                });
            }
            requirement.address_bits = address_bits;
            requirement.cells = cells;
            *has_access = true;
        } else {
            requirements.push((
                StorageRequirement {
                    storage,
                    lane,
                    address_bits,
                    cells,
                },
                true,
            ));
        }
    }

    for segment in &circuit.pre_init {
        let data_len = u64::try_from(segment.data.len()).map_err(|_| {
            StorageLayoutError::PreInitOffsetOverflow {
                storage: segment.storage,
                lane: segment.lane,
            }
        })?;
        let end = segment.offset.checked_add(data_len).ok_or(
            StorageLayoutError::PreInitOffsetOverflow {
                storage: segment.storage,
                lane: segment.lane,
            },
        )?;
        if let Some((requirement, has_access)) = requirements.iter_mut().find(|(requirement, _)| {
            requirement.storage == segment.storage && requirement.lane == segment.lane
        }) {
            if end > requirement.cells as u64 {
                return Err(StorageLayoutError::PreInitOutOfBounds {
                    storage: segment.storage,
                    lane: segment.lane,
                    offset: segment.offset,
                    data_len: segment.data.len(),
                    cells: requirement.cells,
                });
            }
            debug_assert!(*has_access);
        } else {
            let minimum_cells =
                usize::try_from(end).map_err(|_| StorageLayoutError::PreInitOffsetOverflow {
                    storage: segment.storage,
                    lane: segment.lane,
                })?;
            let cells = minimum_cells.max(1).checked_next_power_of_two().ok_or(
                StorageLayoutError::PreInitOffsetOverflow {
                    storage: segment.storage,
                    lane: segment.lane,
                },
            )?;
            requirements.push((
                StorageRequirement {
                    storage: segment.storage,
                    lane: segment.lane,
                    address_bits: cells.trailing_zeros() as usize,
                    cells,
                },
                false,
            ));
        }
    }

    Ok(requirements
        .into_iter()
        .map(|(requirement, _)| requirement)
        .collect())
}

/// Install a circuit's static storage values into fresh caller-owned banks.
///
/// Call this once before repeatedly using [`execute_initialized`] to carry
/// state across transition circuits. [`execute`] is the convenient one-shot
/// version and invokes this automatically.
pub fn initialize_storage<C, P>(
    context: &mut C,
    circuit: &BCircuit<P>,
    banks: &mut [StorageBank<'_, C::Storage>],
) -> Result<(), ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Wire<C>: Clone,
{
    validate_circuit_storage::<C, P>(circuit, banks)?;
    for segment in &circuit.pre_init {
        let bank = find_bank(banks, segment.storage, segment.lane)
            .expect("validated pre-init storage bank");
        let start = usize::try_from(segment.offset).map_err(|_| {
            ExecuteError::StorageLayout(StorageLayoutError::PreInitOffsetOverflow {
                storage: segment.storage,
                lane: segment.lane,
            })
        })?;
        for (offset, bit) in segment.data.iter().copied().enumerate() {
            let address = constant_address(context, start + offset, banks[bank].address_bits)?;
            let value = context.create(bit).map_err(ExecuteError::Context)?;
            context
                .storage_write(banks[bank].value, &address, value)
                .map_err(ExecuteError::Context)?;
        }
    }
    Ok(())
}

/// Execute a fused Boolar circuit through `context`.
///
/// Parameters and caller-supplied storage cells are treated as unknown. The
/// interpreter tracks only values proven by Boolar constants and Boolean
/// operations on them, using those facts to shrink storage MUX/demux trees.
/// Static [`BCircuit::pre_init`] values are installed before execution. For a
/// transition loop, initialize once with [`initialize_storage`] and then use
/// [`execute_initialized`] so writes from earlier transitions persist.
pub fn execute<C, P>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Wire<C>: Clone,
{
    if inputs.len() != circuit.params as usize {
        return Err(ExecuteError::InputArity {
            expected: circuit.params as usize,
            found: inputs.len(),
        });
    }
    initialize_storage(context, circuit, banks)?;
    execute_impl::<C, P, ()>(context, circuit, inputs, banks, None)
}

/// Execute a circuit with named external primitive handlers.
pub fn execute_with_externals<C, P, R>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    externals: &mut R,
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    R: ExternalBitRegistry<C>,
    Wire<C>: Clone,
{
    if inputs.len() != circuit.params as usize {
        return Err(ExecuteError::InputArity {
            expected: circuit.params as usize,
            found: inputs.len(),
        });
    }
    initialize_storage(context, circuit, banks)?;
    execute_impl(context, circuit, inputs, banks, Some(externals))
}

/// Execute a circuit against already-initialized storage.
///
/// This is the persistent-state counterpart to [`execute`]. It deliberately
/// does not replay `pre_init`; callers should use [`initialize_storage`] once
/// for a fresh storage image.
pub fn execute_initialized<C, P>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Wire<C>: Clone,
{
    validate_circuit_storage::<C, P>(circuit, banks)?;
    execute_impl::<C, P, ()>(context, circuit, inputs, banks, None)
}

/// Persistent-storage counterpart to [`execute_with_externals`].
pub fn execute_initialized_with_externals<C, P, R>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    externals: &mut R,
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    R: ExternalBitRegistry<C>,
    Wire<C>: Clone,
{
    if inputs.len() != circuit.params as usize {
        return Err(ExecuteError::InputArity {
            expected: circuit.params as usize,
            found: inputs.len(),
        });
    }
    validate_circuit_storage::<C, P>(circuit, banks)?;
    execute_impl(context, circuit, inputs, banks, Some(externals))
}

fn execute_impl<C, P, R>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    mut externals: Option<&mut R>,
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    R: ExternalBitRegistry<C>,
    Wire<C>: Clone,
{
    let mut values = Vec::with_capacity(circuit.params as usize + circuit.stmts.len());
    values.extend(
        inputs
            .iter()
            .cloned()
            .map(|wire| Value { wire, known: None }),
    );
    let mut canonical_one = None;

    for node in &circuit.stmts {
        let value = match &node.kind {
            BIrStmt::Zero => Value {
                wire: context.create(false).map_err(ExecuteError::Context)?,
                known: Some(false),
            },
            BIrStmt::One => {
                let wire = context.create(true).map_err(ExecuteError::Context)?;
                if canonical_one.is_none() {
                    canonical_one = Some(wire.clone());
                }
                Value {
                    wire,
                    known: Some(true),
                }
            }
            BIrStmt::And(left, right) => apply_and(
                context,
                value_at(&values, *left)?.clone(),
                value_at(&values, *right)?.clone(),
            )?,
            BIrStmt::Or(left, right) => apply_or(
                context,
                value_at(&values, *left)?.clone(),
                value_at(&values, *right)?.clone(),
            )?,
            BIrStmt::Xor(left, right) => apply_xor(
                context,
                value_at(&values, *left)?.clone(),
                value_at(&values, *right)?.clone(),
            )?,
            BIrStmt::Not(input) => apply_not(
                context,
                value_at(&values, *input)?.clone(),
                &mut canonical_one,
            )?,
            BIrStmt::StorageRead {
                storage,
                lane,
                addr,
            } => {
                let address = address_values(&values, addr)?;
                let bank =
                    find_bank(banks, *storage, *lane).ok_or(ExecuteError::MissingStorageBank {
                        storage: *storage,
                        lane: *lane,
                    })?;
                validate_address_width(*storage, *lane, address.len(), banks[bank].address_bits)?;
                let address = storage_address(&address);
                Value {
                    wire: context
                        .storage_read(banks[bank].value, &address)
                        .map_err(ExecuteError::Context)?,
                    known: None,
                }
            }
            BIrStmt::StorageWrite {
                storage,
                lane,
                src,
                addr,
            } => {
                let source = value_at(&values, *src)?.clone();
                let address = address_values(&values, addr)?;
                let bank =
                    find_bank(banks, *storage, *lane).ok_or(ExecuteError::MissingStorageBank {
                        storage: *storage,
                        lane: *lane,
                    })?;
                validate_address_width(*storage, *lane, address.len(), banks[bank].address_bits)?;
                let address = storage_address(&address);
                context
                    .storage_write(banks[bank].value, &address, source.wire)
                    .map_err(ExecuteError::Context)?;
                Value {
                    wire: context.create(false).map_err(ExecuteError::Context)?,
                    known: Some(false),
                }
            }
            BIrStmt::OracleBit {
                name,
                args,
                bit,
                occurrence,
            } => {
                let args = args
                    .iter()
                    .map(|arg| Ok(value_at(&values, *arg)?.wire.clone()))
                    .collect::<Result<Vec<_>, ExecuteError<C::Error>>>()?;
                let handler = externals.as_deref_mut().ok_or_else(|| {
                    ExecuteError::MissingExternalHandler {
                        kind: "oracle",
                        name: name.clone(),
                    }
                })?;
                Value {
                    wire: handler
                        .oracle_bit(context, name, &args, *bit, *occurrence)
                        .map_err(ExecuteError::Context)?,
                    known: None,
                }
            }
            BIrStmt::RngBit {
                name,
                bit,
                occurrence,
            } => {
                let handler = externals.as_deref_mut().ok_or_else(|| {
                    ExecuteError::MissingExternalHandler {
                        kind: "rng",
                        name: name.clone(),
                    }
                })?;
                Value {
                    wire: handler
                        .rng_bit(context, name, *bit, *occurrence)
                        .map_err(ExecuteError::Context)?,
                    known: None,
                }
            }
            BIrStmt::ActionStoreBit {
                name,
                guard,
                args,
                fallback,
                storage,
                lane,
                addr,
                bit,
                occurrence,
            } => {
                let guard = value_at(&values, *guard)?.wire.clone();
                let args = args
                    .iter()
                    .map(|arg| Ok(value_at(&values, *arg)?.wire.clone()))
                    .collect::<Result<Vec<_>, ExecuteError<C::Error>>>()?;
                let fallback = value_at(&values, *fallback)?.wire.clone();
                let address = address_values(&values, addr)?;
                let bank =
                    find_bank(banks, *storage, *lane).ok_or(ExecuteError::MissingStorageBank {
                        storage: *storage,
                        lane: *lane,
                    })?;
                validate_address_width(*storage, *lane, address.len(), banks[bank].address_bits)?;
                let address = storage_address(&address);
                let handler = externals.as_deref_mut().ok_or_else(|| {
                    ExecuteError::MissingExternalHandler {
                        kind: "action",
                        name: name.clone(),
                    }
                })?;
                handler
                    .action_store_bit(
                        context,
                        name,
                        guard,
                        &args,
                        fallback,
                        *storage,
                        *lane,
                        &address,
                        *bit,
                        *occurrence,
                        banks[bank].value,
                    )
                    .map_err(ExecuteError::Context)?;
                Value {
                    wire: context.create(false).map_err(ExecuteError::Context)?,
                    known: Some(false),
                }
            }
            BIrStmt::OracleCall { .. }
            | BIrStmt::OracleProjectedBit { .. }
            | BIrStmt::ActionCall { .. }
            | BIrStmt::ActionBit { .. }
            | BIrStmt::Rng { .. } => return Err(ExecuteError::UnsupportedStatement),
            _ => return Err(ExecuteError::UnsupportedStatement),
        };
        values.push(value);
    }

    circuit
        .outputs
        .iter()
        .map(|output| Ok(value_at(&values, *output)?.wire.clone()))
        .collect()
}

fn validate_circuit_storage<C, P>(
    circuit: &BCircuit<P>,
    banks: &[StorageBank<'_, C::Storage>],
) -> Result<(), ExecuteError<C::Error>>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
{
    validate_banks(banks)?;
    for requirement in storage_requirements(circuit).map_err(ExecuteError::StorageLayout)? {
        let bank = find_bank(banks, requirement.storage, requirement.lane).ok_or(
            ExecuteError::MissingStorageBank {
                storage: requirement.storage,
                lane: requirement.lane,
            },
        )?;
        validate_address_width(
            requirement.storage,
            requirement.lane,
            requirement.address_bits,
            banks[bank].address_bits,
        )?;
    }
    Ok(())
}

fn cells_for_address_bits(bits: usize) -> Result<usize, StorageLayoutError> {
    // Dense `StorageBank` is `2^bits` cells. LLVM fused guests emit 64-bit
    // STACK / VAFFLE_SSA_SPILL addresses (`2^64` does not fit `usize`).
    // Native VOLE / `SparseBank` ignore `cells`; report 0 so layout still
    // names the lane. `AddressWidthOverflow` stays for API consumers that
    // refuse an unrepresentable dense image.
    let Ok(shift) = u32::try_from(bits) else {
        return Ok(0);
    };
    Ok(1usize.checked_shl(shift).unwrap_or(0))
}

fn validate_banks<S: ?Sized, E>(banks: &[StorageBank<'_, S>]) -> Result<(), ExecuteError<E>> {
    for (index, bank) in banks.iter().enumerate() {
        if banks[..index]
            .iter()
            .any(|prior| prior.storage == bank.storage && prior.lane == bank.lane)
        {
            return Err(ExecuteError::DuplicateStorageBank {
                storage: bank.storage,
                lane: bank.lane,
            });
        }
    }
    Ok(())
}

fn value_at<'a, W, E>(
    values: &'a [Value<W>],
    id: IRVarId,
) -> Result<&'a Value<W>, ExecuteError<E>> {
    values
        .get(id.0 as usize)
        .ok_or(ExecuteError::UndefinedVariable {
            var: id,
            available: values.len(),
        })
}

fn address_values<W: Clone, E>(
    values: &[Value<W>],
    address: &[IRVarId],
) -> Result<Vec<Value<W>>, ExecuteError<E>> {
    address
        .iter()
        .map(|id| Ok(value_at(values, *id)?.clone()))
        .collect()
}

fn storage_address<W: Clone>(address: &[Value<W>]) -> Vec<StorageAddressBit<W>> {
    address
        .iter()
        .cloned()
        .map(|bit| StorageAddressBit {
            wire: bit.wire,
            known: bit.known,
        })
        .collect()
}

fn constant_address<C>(
    context: &mut C,
    address: usize,
    address_bits: usize,
) -> Result<Vec<StorageAddressBit<Wire<C>>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    (0..address_bits)
        .map(|bit| {
            let value = (address >> bit) & 1 != 0;
            Ok(StorageAddressBit {
                wire: context.create(value).map_err(ExecuteError::Context)?,
                known: Some(value),
            })
        })
        .collect()
}

fn find_bank<S: ?Sized>(
    banks: &[StorageBank<'_, S>],
    storage: StorageId,
    lane: LaneId,
) -> Option<usize> {
    banks
        .iter()
        .position(|bank| bank.storage == storage && bank.lane == lane)
}

fn validate_address_width<E>(
    storage: StorageId,
    lane: LaneId,
    address_bits: usize,
    declared_address_bits: usize,
) -> Result<(), ExecuteError<E>> {
    if address_bits != declared_address_bits {
        return Err(ExecuteError::StorageAddressWidth {
            storage,
            lane,
            address_bits,
            declared_address_bits,
        });
    }
    Ok(())
}

pub(crate) fn apply_and<C>(
    context: &mut C,
    left: Value<Wire<C>>,
    right: Value<Wire<C>>,
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let known = match (left.known, right.known) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(left), Some(right)) => Some(left & right),
        _ => None,
    };
    Ok(Value {
        wire: context
            .bitand(left.wire, right.wire)
            .map_err(ExecuteError::Context)?,
        known,
    })
}

fn apply_or<C>(
    context: &mut C,
    left: Value<Wire<C>>,
    right: Value<Wire<C>>,
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let known = match (left.known, right.known) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(left), Some(right)) => Some(left | right),
        _ => None,
    };
    Ok(Value {
        wire: context
            .bitor(left.wire, right.wire)
            .map_err(ExecuteError::Context)?,
        known,
    })
}

fn apply_xor<C>(
    context: &mut C,
    left: Value<Wire<C>>,
    right: Value<Wire<C>>,
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let known = match (left.known, right.known) {
        (Some(left), Some(right)) => Some(left ^ right),
        _ => None,
    };
    Ok(Value {
        wire: context
            .bitxor(left.wire, right.wire)
            .map_err(ExecuteError::Context)?,
        known,
    })
}

pub(crate) fn one<C>(
    context: &mut C,
    canonical_one: &mut Option<Wire<C>>,
) -> Result<Wire<C>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    if let Some(value) = canonical_one {
        return Ok(value.clone());
    }
    let value = context.create(true).map_err(ExecuteError::Context)?;
    *canonical_one = Some(value.clone());
    Ok(value)
}

pub(crate) fn apply_not<C>(
    context: &mut C,
    input: Value<Wire<C>>,
    canonical_one: &mut Option<Wire<C>>,
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let known = input.known.map(|value| !value);
    let one = one(context, canonical_one)?;
    Ok(Value {
        wire: context
            .bitxor(input.wire, one)
            .map_err(ExecuteError::Context)?,
        known,
    })
}

pub(crate) fn apply_mux<C>(
    context: &mut C,
    select: Value<Wire<C>>,
    when_zero: Value<Wire<C>>,
    when_one: Value<Wire<C>>,
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let known = match select.known {
        Some(false) => when_zero.known,
        Some(true) => when_one.known,
        None if when_zero.known == when_one.known => when_zero.known,
        None => None,
    };
    let difference = apply_xor(context, when_zero.clone(), when_one)?;
    let masked = apply_and(context, select, difference)?;
    let mut result = apply_xor(context, when_zero, masked)?;
    result.known = known;
    Ok(result)
}

/// Bit positions of `address` that are not statically known.
pub(crate) fn unknown_bits<W>(address: &[Value<W>]) -> Vec<usize> {
    address
        .iter()
        .enumerate()
        .filter_map(|(bit, value)| value.known.is_none().then_some(bit))
        .collect()
}

/// Whether the concrete cell `index` is consistent with `address`'s known
/// bits (an unknown bit is always consistent; a known bit must match).
///
/// Shared by the dense [`candidates`] (which checks every `0..cells`) and
/// [`sparse`]'s live-set scan (which checks only materialized keys).
pub(crate) fn index_matches_known_bits<W>(index: usize, address: &[Value<W>]) -> bool {
    address.iter().enumerate().all(|(bit, value)| {
        if bit >= usize::BITS as usize {
            return value.known != Some(true);
        }
        value
            .known
            .is_none_or(|expected| ((index >> bit) & 1 != 0) == expected)
    })
}

fn candidates<W>(address: &[Value<W>], cells: usize) -> (Vec<usize>, Vec<usize>) {
    let unknown = unknown_bits(address);
    if unknown.is_empty() {
        let mut index = 0usize;
        for (bit, value) in address.iter().enumerate() {
            if value.known == Some(true) {
                if bit >= usize::BITS as usize {
                    return (Vec::new(), unknown);
                }
                index |= 1usize << bit;
            }
        }
        return (
            if index < cells {
                alloc::vec![index]
            } else {
                Vec::new()
            },
            unknown,
        );
    }
    let cells = (0..cells)
        .filter(|&cell| index_matches_known_bits(cell, address))
        .collect();
    (cells, unknown)
}

fn read_mux_storage<C>(
    context: &mut C,
    cells: &[Wire<C>],
    facts: &[Option<bool>],
    address: &[Value<Wire<C>>],
) -> Result<Value<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let (candidates, unknown_bits) = candidates(address, cells.len());
    let mut level: Vec<_> = candidates
        .into_iter()
        .map(|cell| Value {
            wire: cells[cell].clone(),
            known: facts[cell],
        })
        .collect();

    for bit in unknown_bits {
        level = level
            .chunks_exact(2)
            .map(|pair| {
                apply_mux(
                    context,
                    address[bit].clone(),
                    pair[0].clone(),
                    pair[1].clone(),
                )
            })
            .collect::<Result<_, _>>()?;
    }
    Ok(level
        .pop()
        .expect("a power-of-two storage bank always has a candidate"))
}

fn write_mux_storage<C>(
    context: &mut C,
    cells: &mut [Wire<C>],
    facts: &mut [Option<bool>],
    source: Value<Wire<C>>,
    address: &[Value<Wire<C>>],
    canonical_one: &mut Option<Wire<C>>,
) -> Result<(), ExecuteError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let (candidates, unknown_bits) = candidates(address, cells.len());
    if unknown_bits.is_empty() {
        let cell = candidates[0];
        cells[cell] = source.wire;
        facts[cell] = source.known;
        return Ok(());
    }

    let mut selectors = alloc::vec![Value {
        wire: one(context, canonical_one)?,
        known: Some(true),
    }];
    // Expand most-significant unknown dimensions first so the final selector
    // list remains in increasing little-endian cell-address order.
    for bit in unknown_bits.into_iter().rev() {
        let not_bit = apply_not(context, address[bit].clone(), canonical_one)?;
        let previous = core::mem::take(&mut selectors);
        selectors = Vec::with_capacity(previous.len() * 2);
        for prefix in previous {
            selectors.push(apply_and(context, prefix.clone(), not_bit.clone())?);
            selectors.push(apply_and(context, prefix, address[bit].clone())?);
        }
    }

    for (cell, select) in candidates.into_iter().zip(selectors) {
        let old = Value {
            wire: cells[cell].clone(),
            known: facts[cell],
        };
        let next = apply_mux(context, select, old, source.clone())?;
        cells[cell] = next.wire;
        facts[cell] = next.known;
    }
    Ok(())
}

/// Caller-supplied serialization for opaque Cirrus wire values in lazy runs.
pub trait LazyWireCodec<W> {
    /// Codec-specific error.
    type Error;
    /// Encode one wire and its known-Boolean fact.
    fn encode(&self, wire: &W, known: Option<bool>, out: &mut Vec<u8>) -> Result<(), Self::Error>;
    /// Decode one wire and its known-Boolean fact.
    fn decode(&self, bytes: &[u8]) -> Result<(W, Option<bool>), Self::Error>;
}

/// The mandatory authenticated scratch adapter used by chunked Boolar runs.
pub struct AuthenticatedWireScratch<S, C> {
    scratch: EncryptedScratch<S>,
    codec: C,
}

impl<S, C> AuthenticatedWireScratch<S, C> {
    /// Combine caller-defined wire encoding with XChaCha20-Poly1305 scratch.
    #[must_use]
    pub fn new(scratch: EncryptedScratch<S>, codec: C) -> Self {
        Self { scratch, codec }
    }
}

/// Error returned by the bounded chunked interpreter.
#[derive(Debug)]
pub enum LazyExecuteError<ContextError, SourceError, ChunkCodecError, ScratchError, WireCodecError>
{
    /// Existing Boolar execution or storage validation failed.
    Execute(ExecuteError<ContextError>),
    /// Fetching, digest checking, decoding, or range checking a chunk failed.
    Chunk(ChunkLoadError<SourceError, ChunkCodecError>),
    /// AEAD-protected scratch rejected a record.
    Scratch(lazy_repo_crypto::ScratchError<ScratchError>),
    /// Caller-defined wire serialization failed.
    WireCodec(WireCodecError),
    /// The root liveness schedule and streamed circuit disagree.
    InvalidManifest,
}

struct ChunkedValues<'a, W, S, C> {
    schedule: &'a WireSchedule,
    values: Vec<Option<Value<W>>>,
    spilled: Vec<bool>,
    resident: usize,
    capacity: usize,
    scratch: &'a mut AuthenticatedWireScratch<S, C>,
}

impl<'a, W: Clone, S: ScratchStore, C: LazyWireCodec<W>> ChunkedValues<'a, W, S, C> {
    fn new(
        schedule: &'a WireSchedule,
        variable_count: usize,
        inputs: &[W],
        scratch: &'a mut AuthenticatedWireScratch<S, C>,
    ) -> Result<Self, LazyExecuteError<(), (), (), S::Error, C::Error>> {
        if schedule.last_use.len() != variable_count || schedule.release_at.is_empty() {
            return Err(LazyExecuteError::InvalidManifest);
        }
        let capacity = schedule.persistent_wires() as usize;
        if capacity == 0 {
            return Err(LazyExecuteError::InvalidManifest);
        }
        let mut result = Self {
            schedule,
            values: (0..variable_count).map(|_| None).collect(),
            spilled: alloc::vec![false; variable_count],
            resident: 0,
            capacity,
            scratch,
        };
        for (index, wire) in inputs.iter().cloned().enumerate() {
            if schedule.last_use[index] != 0 {
                result.insert(IRVarId(index as u32), Value { wire, known: None })?;
            }
        }
        Ok(result)
    }

    fn scratch_put(
        &mut self,
        variable: IRVarId,
        value: &Value<W>,
    ) -> Result<(), LazyExecuteError<(), (), (), S::Error, C::Error>> {
        let mut bytes = Vec::new();
        self.scratch
            .codec
            .encode(&value.wire, value.known, &mut bytes)
            .map_err(LazyExecuteError::WireCodec)?;
        self.scratch
            .scratch
            .put(ScratchSlot(variable.0), &bytes)
            .map_err(LazyExecuteError::Scratch)
    }

    fn make_room(&mut self) -> Result<(), LazyExecuteError<(), (), (), S::Error, C::Error>> {
        if self.resident < self.capacity {
            return Ok(());
        }
        let index = self
            .values
            .iter()
            .position(Option::is_some)
            .ok_or(LazyExecuteError::InvalidManifest)?;
        let value = self.values[index]
            .take()
            .ok_or(LazyExecuteError::InvalidManifest)?;
        self.scratch_put(IRVarId(index as u32), &value)?;
        self.spilled[index] = true;
        self.resident -= 1;
        Ok(())
    }

    fn insert(
        &mut self,
        variable: IRVarId,
        value: Value<W>,
    ) -> Result<(), LazyExecuteError<(), (), (), S::Error, C::Error>> {
        let index = variable.0 as usize;
        if index >= self.values.len() || self.values[index].is_some() || self.spilled[index] {
            return Err(LazyExecuteError::InvalidManifest);
        }
        self.make_room()?;
        self.values[index] = Some(value);
        self.resident += 1;
        Ok(())
    }

    fn get(
        &mut self,
        variable: IRVarId,
    ) -> Result<Value<W>, LazyExecuteError<(), (), (), S::Error, C::Error>> {
        let index = variable.0 as usize;
        let Some(present) = self.values.get(index) else {
            return Err(LazyExecuteError::InvalidManifest);
        };
        if let Some(value) = present {
            return Ok(value.clone());
        }
        if !self.spilled.get(index).copied().unwrap_or(false) {
            return Err(LazyExecuteError::InvalidManifest);
        }
        let bytes = self
            .scratch
            .scratch
            .take(ScratchSlot(variable.0))
            .map_err(LazyExecuteError::Scratch)?;
        let (wire, known) = self
            .scratch
            .codec
            .decode(&bytes)
            .map_err(LazyExecuteError::WireCodec)?;
        let value = Value { wire, known };
        // `take` removes the encrypted record, so restore it immediately;
        // the returned clone is only a temporary operand wire.
        self.scratch_put(variable, &value)?;
        Ok(value)
    }

    fn release(
        &mut self,
        position: u32,
    ) -> Result<(), LazyExecuteError<(), (), (), S::Error, C::Error>> {
        let variables = self
            .schedule
            .release_at
            .get(position as usize)
            .ok_or(LazyExecuteError::InvalidManifest)?;
        for variable in variables.iter().copied() {
            let index = variable as usize;
            if let Some(value) = self.values.get_mut(index).and_then(Option::take) {
                drop(value);
                self.resident -= 1;
            } else if self.spilled.get(index).copied().unwrap_or(false) {
                self.scratch
                    .scratch
                    .remove(ScratchSlot(variable))
                    .map_err(LazyExecuteError::Scratch)?;
                self.spilled[index] = false;
            }
        }
        Ok(())
    }
}

/// Execute a content-addressed Boolar circuit one decoded statement range at
/// a time. The schedule reserves temporary operand wires, and every remaining
/// live wire above its persistent budget is moved through authenticated scratch.
pub fn execute_chunked<C, P, Source, ChunkSer, Scratch, WireSer>(
    context: &mut C,
    circuit: &ChunkedBCircuit,
    repository: &mut Repository<Source>,
    chunk_codec: &ChunkSer,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    scratch: &mut AuthenticatedWireScratch<Scratch, WireSer>,
) -> Result<
    Vec<Wire<C>>,
    LazyExecuteError<C::Error, Source::Error, ChunkSer::Error, Scratch::Error, WireSer::Error>,
>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Source: lazy_repo::ChunkSource,
    ChunkSer: ChunkCodec<BStmtChunk<P>>,
    Scratch: ScratchStore,
    WireSer: LazyWireCodec<Wire<C>>,
    Wire<C>: Clone,
{
    execute_chunked_impl::<C, P, Source, ChunkSer, Scratch, WireSer, ()>(
        context,
        circuit,
        repository,
        chunk_codec,
        inputs,
        banks,
        scratch,
        None,
    )
}

/// Execute a content-addressed Boolar circuit with caller-supplied external
/// primitive handlers. This is the lazy counterpart of
/// [`execute_with_externals`].
#[allow(clippy::too_many_arguments)]
pub fn execute_chunked_with_externals<C, P, Source, ChunkSer, Scratch, WireSer, R>(
    context: &mut C,
    circuit: &ChunkedBCircuit,
    repository: &mut Repository<Source>,
    chunk_codec: &ChunkSer,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    scratch: &mut AuthenticatedWireScratch<Scratch, WireSer>,
    externals: &mut R,
) -> Result<
    Vec<Wire<C>>,
    LazyExecuteError<C::Error, Source::Error, ChunkSer::Error, Scratch::Error, WireSer::Error>,
>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Source: lazy_repo::ChunkSource,
    ChunkSer: ChunkCodec<BStmtChunk<P>>,
    Scratch: ScratchStore,
    WireSer: LazyWireCodec<Wire<C>>,
    R: ExternalBitRegistry<C>,
    Wire<C>: Clone,
{
    execute_chunked_impl(
        context,
        circuit,
        repository,
        chunk_codec,
        inputs,
        banks,
        scratch,
        Some(externals),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_chunked_impl<C, P, Source, ChunkSer, Scratch, WireSer, R>(
    context: &mut C,
    circuit: &ChunkedBCircuit,
    repository: &mut Repository<Source>,
    chunk_codec: &ChunkSer,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, C::Storage>],
    scratch: &mut AuthenticatedWireScratch<Scratch, WireSer>,
    mut externals: Option<&mut R>,
) -> Result<
    Vec<Wire<C>>,
    LazyExecuteError<C::Error, Source::Error, ChunkSer::Error, Scratch::Error, WireSer::Error>,
>
where
    C: BoolarContext + ContextWithStorage<bool>,
    P: Clone,
    Source: lazy_repo::ChunkSource,
    ChunkSer: ChunkCodec<BStmtChunk<P>>,
    Scratch: ScratchStore,
    WireSer: LazyWireCodec<Wire<C>>,
    R: ExternalBitRegistry<C>,
    Wire<C>: Clone,
{
    if inputs.len() != circuit.params as usize {
        return Err(LazyExecuteError::Execute(ExecuteError::InputArity {
            expected: circuit.params as usize,
            found: inputs.len(),
        }));
    }
    // Reuse the eager static-data installer against a header-only circuit;
    // it needs no statement body to install pre-init segments.
    let header: BCircuit<P> = BCircuit {
        params: circuit.params,
        stmts: Vec::new(),
        pre_init: circuit.pre_init.clone(),
        outputs: Vec::new(),
    };
    initialize_storage(context, &header, banks).map_err(LazyExecuteError::Execute)?;
    let statement_count = circuit.statement_count();
    let variable_count = circuit
        .params
        .checked_add(statement_count)
        .ok_or(LazyExecuteError::InvalidManifest)? as usize;
    let mut values = ChunkedValues::new(&circuit.wire_schedule, variable_count, inputs, scratch)
        .map_err(|error| match error {
            LazyExecuteError::Scratch(error) => LazyExecuteError::Scratch(error),
            LazyExecuteError::WireCodec(error) => LazyExecuteError::WireCodec(error),
            _ => LazyExecuteError::InvalidManifest,
        })?;
    let mut expected_statement = 0_u32;

    for chunk_index in 0..circuit.statement_chunks.len() {
        let decoded = circuit
            .load_statement_chunk(repository, chunk_index, chunk_codec)
            .map_err(LazyExecuteError::Chunk)?;
        if decoded.value.first != expected_statement {
            return Err(LazyExecuteError::InvalidManifest);
        }
        for (offset, node) in decoded.value.stmts.iter().enumerate() {
            let statement_index = decoded
                .value
                .first
                .checked_add(offset as u32)
                .ok_or(LazyExecuteError::InvalidManifest)?;
            let local = |error| match error {
                LazyExecuteError::Scratch(error) => LazyExecuteError::Scratch(error),
                LazyExecuteError::WireCodec(error) => LazyExecuteError::WireCodec(error),
                _ => LazyExecuteError::InvalidManifest,
            };
            let value = match &node.kind {
                BIrStmt::Zero => Value {
                    wire: context
                        .create(false)
                        .map_err(|error| LazyExecuteError::Execute(ExecuteError::Context(error)))?,
                    known: Some(false),
                },
                BIrStmt::One => Value {
                    wire: context
                        .create(true)
                        .map_err(|error| LazyExecuteError::Execute(ExecuteError::Context(error)))?,
                    known: Some(true),
                },
                BIrStmt::And(left, right) => apply_and(
                    context,
                    values.get(*left).map_err(local)?,
                    values.get(*right).map_err(local)?,
                )
                .map_err(LazyExecuteError::Execute)?,
                BIrStmt::Or(left, right) => apply_or(
                    context,
                    values.get(*left).map_err(local)?,
                    values.get(*right).map_err(local)?,
                )
                .map_err(LazyExecuteError::Execute)?,
                BIrStmt::Xor(left, right) => apply_xor(
                    context,
                    values.get(*left).map_err(local)?,
                    values.get(*right).map_err(local)?,
                )
                .map_err(LazyExecuteError::Execute)?,
                BIrStmt::Not(input) => {
                    // A cached `one` would be an unplanned resident wire.
                    // Create and discard it within this scheduled operation.
                    let mut one = None;
                    apply_not(context, values.get(*input).map_err(local)?, &mut one)
                        .map_err(LazyExecuteError::Execute)?
                }
                BIrStmt::StorageRead {
                    storage,
                    lane,
                    addr,
                } => {
                    let address: Vec<_> = addr
                        .iter()
                        .map(|id| values.get(*id).map_err(local))
                        .collect::<Result<_, _>>()?;
                    let bank = find_bank(banks, *storage, *lane).ok_or(
                        LazyExecuteError::Execute(ExecuteError::MissingStorageBank {
                            storage: *storage,
                            lane: *lane,
                        }),
                    )?;
                    validate_address_width(
                        *storage,
                        *lane,
                        address.len(),
                        banks[bank].address_bits,
                    )
                    .map_err(LazyExecuteError::Execute)?;
                    Value {
                        wire: context
                            .storage_read(banks[bank].value, &storage_address(&address))
                            .map_err(|error| {
                                LazyExecuteError::Execute(ExecuteError::Context(error))
                            })?,
                        known: None,
                    }
                }
                BIrStmt::StorageWrite {
                    storage,
                    lane,
                    src,
                    addr,
                } => {
                    let source = values.get(*src).map_err(local)?;
                    let address: Vec<_> = addr
                        .iter()
                        .map(|id| values.get(*id).map_err(local))
                        .collect::<Result<_, _>>()?;
                    let bank = find_bank(banks, *storage, *lane).ok_or(
                        LazyExecuteError::Execute(ExecuteError::MissingStorageBank {
                            storage: *storage,
                            lane: *lane,
                        }),
                    )?;
                    validate_address_width(
                        *storage,
                        *lane,
                        address.len(),
                        banks[bank].address_bits,
                    )
                    .map_err(LazyExecuteError::Execute)?;
                    context
                        .storage_write(banks[bank].value, &storage_address(&address), source.wire)
                        .map_err(|error| LazyExecuteError::Execute(ExecuteError::Context(error)))?;
                    Value {
                        wire: context.create(false).map_err(|error| {
                            LazyExecuteError::Execute(ExecuteError::Context(error))
                        })?,
                        known: Some(false),
                    }
                }
                BIrStmt::OracleBit {
                    name,
                    args,
                    bit,
                    occurrence,
                } => {
                    let args: Vec<_> = args
                        .iter()
                        .map(|id| values.get(*id).map(|value| value.wire).map_err(local))
                        .collect::<Result<_, _>>()?;
                    let handler = externals.as_deref_mut().ok_or_else(|| {
                        LazyExecuteError::Execute(ExecuteError::MissingExternalHandler {
                            kind: "oracle",
                            name: name.clone(),
                        })
                    })?;
                    Value {
                        wire: handler
                            .oracle_bit(context, name, &args, *bit, *occurrence)
                            .map_err(|error| {
                                LazyExecuteError::Execute(ExecuteError::Context(error))
                            })?,
                        known: None,
                    }
                }
                BIrStmt::RngBit {
                    name,
                    bit,
                    occurrence,
                } => {
                    let handler = externals.as_deref_mut().ok_or_else(|| {
                        LazyExecuteError::Execute(ExecuteError::MissingExternalHandler {
                            kind: "rng",
                            name: name.clone(),
                        })
                    })?;
                    Value {
                        wire: handler.rng_bit(context, name, *bit, *occurrence).map_err(
                            |error| LazyExecuteError::Execute(ExecuteError::Context(error)),
                        )?,
                        known: None,
                    }
                }
                BIrStmt::ActionStoreBit {
                    name,
                    guard,
                    args,
                    fallback,
                    storage,
                    lane,
                    addr,
                    bit,
                    occurrence,
                } => {
                    let guard = values.get(*guard).map_err(local)?.wire;
                    let args: Vec<_> = args
                        .iter()
                        .map(|id| values.get(*id).map(|value| value.wire).map_err(local))
                        .collect::<Result<_, _>>()?;
                    let fallback = values.get(*fallback).map_err(local)?.wire;
                    let address: Vec<_> = addr
                        .iter()
                        .map(|id| values.get(*id).map_err(local))
                        .collect::<Result<_, _>>()?;
                    let bank = find_bank(banks, *storage, *lane).ok_or(
                        LazyExecuteError::Execute(ExecuteError::MissingStorageBank {
                            storage: *storage,
                            lane: *lane,
                        }),
                    )?;
                    validate_address_width(
                        *storage,
                        *lane,
                        address.len(),
                        banks[bank].address_bits,
                    )
                    .map_err(LazyExecuteError::Execute)?;
                    let address = storage_address(&address);
                    let handler = externals.as_deref_mut().ok_or_else(|| {
                        LazyExecuteError::Execute(ExecuteError::MissingExternalHandler {
                            kind: "action",
                            name: name.clone(),
                        })
                    })?;
                    handler
                        .action_store_bit(
                            context,
                            name,
                            guard,
                            &args,
                            fallback,
                            *storage,
                            *lane,
                            &address,
                            *bit,
                            *occurrence,
                            banks[bank].value,
                        )
                        .map_err(|error| LazyExecuteError::Execute(ExecuteError::Context(error)))?;
                    Value {
                        wire: context.create(false).map_err(|error| {
                            LazyExecuteError::Execute(ExecuteError::Context(error))
                        })?,
                        known: Some(false),
                    }
                }
                _ => {
                    return Err(LazyExecuteError::Execute(
                        ExecuteError::UnsupportedStatement,
                    ));
                }
            };
            let variable = circuit
                .params
                .checked_add(statement_index)
                .ok_or(LazyExecuteError::InvalidManifest)?;
            values.insert(IRVarId(variable), value).map_err(local)?;
            values
                .release(statement_index.saturating_add(1))
                .map_err(local)?;
            expected_statement = expected_statement.saturating_add(1);
        }
    }
    if expected_statement != statement_count {
        return Err(LazyExecuteError::InvalidManifest);
    }
    circuit
        .outputs
        .iter()
        .map(|output| {
            values
                .get(*output)
                .map(|value| value.wire)
                .map_err(|error| match error {
                    LazyExecuteError::Scratch(error) => LazyExecuteError::Scratch(error),
                    LazyExecuteError::WireCodec(error) => LazyExecuteError::WireCodec(error),
                    _ => LazyExecuteError::InvalidManifest,
                })
        })
        .collect()
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use cirrus_core::{
        ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, HasError,
    };
    use cirrus_recompile_core::{interpret, Recorder};
    use core::convert::Infallible;
    use lazy_repo::{CacheConfig, ContentId, MemorySource};
    use std::{cell::RefCell, collections::BTreeMap};
    use volar_ir::boolar::BIrPreInitSegment;
    use volar_ir::lazy::{chunk_b_circuit, BStmtChunk};
    use volar_ir_common::Node;

    #[test]
    fn bounded_riscv_fixture_lowers_to_initialized_five_bit_memory() {
        use volar_riscv_test_programs::{parse_and_expand, wat_gen::test_program_wat};
        use volar_vaffle_target::{
            import_config::WaffleImportConfig, waffle_lower::lower_waffle_module, VaffleTarget,
        };

        let wasm = wat::parse_str(test_program_wat()).expect("RISC fixture WAT should assemble");
        let module = parse_and_expand(&wasm).expect("RISC fixture should parse and expand");
        let mut target = VaffleTarget::new();
        let errors = lower_waffle_module(
            &module,
            &mut target,
            &WaffleImportConfig::default().with_memory_address_bits(5),
        );
        assert!(errors.is_empty(), "unexpected lowering errors: {errors:?}");
        assert_eq!(
            target.module.pre_init.len(),
            2,
            "code and data memories must be initialized"
        );
        assert_eq!(
            target.module.pre_init[0].data.len(),
            32,
            "RISC code fits the five-bit RAM"
        );
        assert_eq!(
            target.module.pre_init[1].data.len(),
            20,
            "data RAM includes the result word"
        );
    }

    const STORAGE: StorageId = StorageId(7);
    const LANE: LaneId = LaneId(3);

    fn mux_bank<'a, W>(cells: &'a mut [W], address_bits: usize) -> StorageBank<'a, [W]> {
        StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits,
            value: cells,
        }
    }

    fn circuit(params: u32, stmts: Vec<BIrStmt>, outputs: Vec<IRVarId>) -> BCircuit {
        BCircuit {
            params,
            stmts: stmts
                .into_iter()
                .map(|stmt| Node::new(stmt, (), None))
                .collect(),
            pre_init: vec![],
            outputs,
        }
    }

    #[test]
    fn executes_boolean_primitives() {
        let circuit = circuit(
            2,
            vec![
                BIrStmt::And(IRVarId(0), IRVarId(1)),
                BIrStmt::Or(IRVarId(0), IRVarId(1)),
                BIrStmt::Xor(IRVarId(2), IRVarId(3)),
                BIrStmt::Not(IRVarId(4)),
                BIrStmt::Zero,
                BIrStmt::One,
            ],
            vec![
                IRVarId(2),
                IRVarId(3),
                IRVarId(4),
                IRVarId(5),
                IRVarId(6),
                IRVarId(7),
            ],
        );
        let mut context = MuxTreeContext::new(());
        assert_eq!(
            execute(&mut context, &circuit, &[true, false], &mut []).unwrap(),
            vec![false, true, true, false, false, true]
        );
    }

    #[test]
    fn writes_and_reads_every_address() {
        let circuit = circuit(
            3,
            vec![
                BIrStmt::StorageWrite {
                    storage: STORAGE,
                    lane: LANE,
                    src: IRVarId(0),
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
            ],
            vec![IRVarId(4)],
        );
        for address in 0..4 {
            let mut cells = [false; 4];
            let mut banks = [mux_bank(&mut cells, 2)];
            let mut context = MuxTreeContext::new(());
            let outputs = execute(
                &mut context,
                &circuit,
                &[true, address & 1 != 0, address & 2 != 0],
                &mut banks,
            )
            .unwrap();
            assert_eq!(outputs, vec![true]);
            assert_eq!(cells, core::array::from_fn(|cell| cell == address));
        }
    }

    #[test]
    fn derives_and_installs_preinitialized_storage() {
        let mut circuit = circuit(
            2,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![IRVarId(0), IRVarId(1)],
            }],
            vec![IRVarId(2)],
        );
        circuit.pre_init = vec![BIrPreInitSegment {
            storage: STORAGE,
            lane: LANE,
            offset: 0,
            data: vec![false, true, true, false],
        }];
        assert_eq!(
            storage_requirements(&circuit),
            Ok(vec![StorageRequirement {
                storage: STORAGE,
                lane: LANE,
                address_bits: 2,
                cells: 4,
            }])
        );

        let mut cells = [false; 4];
        let mut banks = [mux_bank(&mut cells, 2)];
        let mut context = MuxTreeContext::new(());
        assert_eq!(
            execute(&mut context, &circuit, &[false, true], &mut banks).unwrap(),
            vec![true]
        );
        assert_eq!(cells, [false, true, true, false]);
    }

    #[test]
    fn sixty_four_bit_lane_reports_zero_dense_cells() {
        let addr: Vec<IRVarId> = (0..64).map(IRVarId).collect();
        let circuit = circuit(
            64,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr,
            }],
            vec![IRVarId(64)],
        );
        assert_eq!(
            storage_requirements(&circuit),
            Ok(vec![StorageRequirement {
                storage: STORAGE,
                lane: LANE,
                address_bits: 64,
                cells: 0,
            }])
        );
    }

    #[test]
    fn rejects_preinit_outside_addressable_memory() {
        let mut circuit = circuit(
            2,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![IRVarId(0), IRVarId(1)],
            }],
            vec![IRVarId(2)],
        );
        circuit.pre_init = vec![BIrPreInitSegment {
            storage: STORAGE,
            lane: LANE,
            offset: 3,
            data: vec![true, false],
        }];
        assert_eq!(
            storage_requirements(&circuit),
            Err(StorageLayoutError::PreInitOutOfBounds {
                storage: STORAGE,
                lane: LANE,
                offset: 3,
                data_len: 2,
                cells: 4,
            })
        );
    }

    #[test]
    fn initialized_execution_preserves_transition_storage() {
        let mut circuit = circuit(
            2,
            vec![
                BIrStmt::StorageWrite {
                    storage: STORAGE,
                    lane: LANE,
                    src: IRVarId(0),
                    addr: vec![IRVarId(1)],
                },
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(1)],
                },
            ],
            vec![IRVarId(3)],
        );
        circuit.pre_init = vec![BIrPreInitSegment {
            storage: STORAGE,
            lane: LANE,
            offset: 0,
            data: vec![false, false],
        }];
        let mut cells = [true; 2];
        let mut banks = [mux_bank(&mut cells, 1)];
        let mut context = MuxTreeContext::new(());
        initialize_storage(&mut context, &circuit, &mut banks).unwrap();
        assert_eq!(
            execute_initialized(&mut context, &circuit, &[true, true], &mut banks).unwrap(),
            vec![true]
        );
        assert_eq!(
            execute_initialized(&mut context, &circuit, &[false, false], &mut banks).unwrap(),
            vec![false]
        );
        assert_eq!(cells, [false, true]);
    }

    #[test]
    fn constants_prune_the_read_mux_tree() {
        let pruned = circuit(
            1,
            vec![
                BIrStmt::One,
                BIrStmt::Zero,
                BIrStmt::And(IRVarId(0), IRVarId(2)),
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(0), IRVarId(1), IRVarId(3)],
                },
            ],
            vec![IRVarId(4)],
        );
        let full = circuit(
            3,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![IRVarId(0), IRVarId(1), IRVarId(2)],
            }],
            vec![IRVarId(3)],
        );
        let mut pruned_cells = [false; 8];
        let mut full_cells = [false; 8];
        let mut pruned_banks = [mux_bank(&mut pruned_cells, 3)];
        let mut full_banks = [mux_bank(&mut full_cells, 3)];
        let mut pruned_context = MuxTreeContext::new(CountingContext::default());
        let mut full_context = MuxTreeContext::new(CountingContext::default());
        execute(&mut pruned_context, &pruned, &[false], &mut pruned_banks).unwrap();
        execute(
            &mut full_context,
            &full,
            &[false, false, false],
            &mut full_banks,
        )
        .unwrap();
        assert_eq!(pruned_context.inner().ands, 2);
        assert_eq!(pruned_context.inner().xors, 2);
        assert_eq!(full_context.inner().ands, 7);
        assert_eq!(full_context.inner().xors, 14);
    }

    #[test]
    fn constants_prune_the_write_demux_tree() {
        let pruned = circuit(
            2,
            vec![
                BIrStmt::One,
                BIrStmt::StorageWrite {
                    storage: STORAGE,
                    lane: LANE,
                    src: IRVarId(0),
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
            ],
            vec![],
        );
        let full = circuit(
            3,
            vec![BIrStmt::StorageWrite {
                storage: STORAGE,
                lane: LANE,
                src: IRVarId(0),
                addr: vec![IRVarId(1), IRVarId(2)],
            }],
            vec![],
        );
        let mut pruned_cells = [false; 4];
        let mut full_cells = [false; 4];
        let mut pruned_banks = [mux_bank(&mut pruned_cells, 2)];
        let mut full_banks = [mux_bank(&mut full_cells, 2)];
        let mut pruned_context = MuxTreeContext::new(CountingContext::default());
        let mut full_context = MuxTreeContext::new(CountingContext::default());
        execute(
            &mut pruned_context,
            &pruned,
            &[true, false],
            &mut pruned_banks,
        )
        .unwrap();
        execute(
            &mut full_context,
            &full,
            &[true, false, true],
            &mut full_banks,
        )
        .unwrap();
        assert_eq!(pruned_cells, [false, false, true, false]);
        assert_eq!(pruned_context.inner().ands, 4);
        assert_eq!(pruned_context.inner().xors, 5);
        assert_eq!(full_context.inner().ands, 10);
        assert_eq!(full_context.inner().xors, 10);
    }

    fn sparse_known_address(bits: usize, index: u64) -> Vec<StorageAddressBit<bool>> {
        (0..bits)
            .map(|bit| {
                let value = (index >> bit) & 1 == 1;
                StorageAddressBit {
                    wire: value,
                    known: Some(value),
                }
            })
            .collect()
    }

    #[test]
    fn sparse_bank_only_materializes_touched_cells_for_a_34_bit_lane() {
        let mut context = SparseMuxTreeContext::new(());
        let mut bank = SparseBank::<bool>::new();

        context
            .storage_write(&mut bank, &sparse_known_address(34, 5), true)
            .unwrap();
        context
            .storage_write(&mut bank, &sparse_known_address(34, 1 << 30), false)
            .unwrap();
        assert_eq!(bank.live_len(), 2, "only the two touched cells exist");

        assert!(context
            .storage_read(&mut bank, &sparse_known_address(34, 5))
            .unwrap());
        assert!(!context
            .storage_read(&mut bank, &sparse_known_address(34, 1 << 30))
            .unwrap());
        // An untouched address defaults to false and materializes on read.
        assert!(!context
            .storage_read(&mut bank, &sparse_known_address(34, 42))
            .unwrap());
        assert_eq!(bank.live_len(), 3);
    }

    #[test]
    fn sparse_known_address_access_never_touches_the_mux_tree() {
        let mut context = SparseMuxTreeContext::new(CountingContext::default());
        let mut bank = SparseBank::<bool>::new();

        context
            .storage_write(&mut bank, &sparse_known_address(34, 5), true)
            .unwrap();
        let value = context
            .storage_read(&mut bank, &sparse_known_address(34, 5))
            .unwrap();
        assert!(value);
        assert_eq!(context.inner().ands, 0);
        assert_eq!(context.inner().ors, 0);
        assert_eq!(context.inner().xors, 0);
    }

    #[test]
    fn symbolic_read_over_two_live_cells_uses_a_small_gate_count() {
        let mut context = SparseMuxTreeContext::new(CountingContext::default());
        let mut bank = SparseBank::<bool>::new();

        context
            .storage_write(&mut bank, &sparse_known_address(24, 3), true)
            .unwrap();
        context
            .storage_write(&mut bank, &sparse_known_address(24, 9), false)
            .unwrap();
        assert_eq!(bank.live_len(), 2);

        let unknown_address: Vec<StorageAddressBit<bool>> = (0..24)
            .map(|_| StorageAddressBit {
                wire: false,
                known: None,
            })
            .collect();
        context.storage_read(&mut bank, &unknown_address).unwrap();

        // Two live cells over 24 unknown bits: a handful of gates, nowhere
        // near a 16-million-way dense tree.
        assert!(context.inner().ands < 100, "ands = {}", context.inner().ands);
        assert!(context.inner().xors < 100, "xors = {}", context.inner().xors);
    }

    #[test]
    fn symbolic_write_succeeds_when_every_candidate_is_already_live() {
        let mut context = SparseMuxTreeContext::new(());
        let mut bank = SparseBank::<bool>::new();

        // Both 2-bit addresses consistent with bit1 == false are pre-touched.
        context
            .storage_write(&mut bank, &sparse_known_address(2, 0), false)
            .unwrap();
        context
            .storage_write(&mut bank, &sparse_known_address(2, 1), false)
            .unwrap();

        let symbolic_address = vec![
            StorageAddressBit {
                wire: true,
                known: None,
            },
            StorageAddressBit {
                wire: false,
                known: Some(false),
            },
        ];
        context
            .storage_write(&mut bank, &symbolic_address, true)
            .unwrap();

        assert!(!context
            .storage_read(&mut bank, &sparse_known_address(2, 0))
            .unwrap());
        assert!(context
            .storage_read(&mut bank, &sparse_known_address(2, 1))
            .unwrap());
    }

    #[test]
    fn symbolic_write_to_a_not_fully_live_candidate_set_fails_closed() {
        let mut context = SparseMuxTreeContext::new(());
        let mut bank = SparseBank::<bool>::new();

        // Only index 0 is live; index 1 (the other candidate for bit0
        // unknown, bit1 known false) has never been touched.
        context
            .storage_write(&mut bank, &sparse_known_address(2, 0), false)
            .unwrap();

        let symbolic_address = vec![
            StorageAddressBit {
                wire: true,
                known: None,
            },
            StorageAddressBit {
                wire: false,
                known: Some(false),
            },
        ];
        let error = context
            .storage_write(&mut bank, &symbolic_address, true)
            .unwrap_err();
        assert_eq!(error, SparseStorageError::UnboundedWrite);
    }

    #[test]
    fn sparse_bank_composes_with_execute() {
        let write_known = circuit(
            1,
            vec![
                BIrStmt::Zero,
                BIrStmt::StorageWrite {
                    storage: STORAGE,
                    lane: LANE,
                    src: IRVarId(0),
                    addr: vec![IRVarId(1)],
                },
            ],
            vec![],
        );
        let read_symbolic = circuit(
            1,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![IRVarId(0)],
            }],
            vec![IRVarId(1)],
        );
        let mut bank = SparseBank::<bool>::new();
        let mut context = SparseMuxTreeContext::new(());

        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut bank,
        }];
        execute(&mut context, &write_known, &[true], &mut banks).unwrap();
        assert_eq!(bank.live_len(), 1);

        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut bank,
        }];
        // The input param (false) matches the written index (0), exercising
        // the genuinely symbolic (unknown-bit) read path end to end.
        let outputs = execute(&mut context, &read_symbolic, &[false], &mut banks).unwrap();
        assert_eq!(outputs, vec![true]);
    }

    #[test]
    fn composes_with_the_recompile_recorder() {
        let circuit = circuit(
            2,
            vec![
                BIrStmt::Xor(IRVarId(0), IRVarId(1)),
                BIrStmt::Not(IRVarId(2)),
            ],
            vec![IRVarId(3)],
        );
        let mut recorder = MuxTreeContext::new(Recorder::new());
        let left = recorder.create(false).unwrap();
        let right = recorder.create(false).unwrap();
        let outputs = execute(&mut recorder, &circuit, &[left, right], &mut []).unwrap();
        let program = recorder.into_inner().finish(vec![left, right], outputs);
        assert_eq!(interpret(&program, &[true, false]), vec![false]);
        assert_eq!(interpret(&program, &[true, true]), vec![true]);
    }

    #[test]
    fn rejects_missing_and_misconfigured_storage() {
        let circuit = circuit(
            0,
            vec![BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![],
            }],
            vec![IRVarId(0)],
        );
        let mut context = MuxTreeContext::new(());
        assert_eq!(
            execute(&mut context, &circuit, &[], &mut []).unwrap_err(),
            ExecuteError::MissingStorageBank {
                storage: STORAGE,
                lane: LANE
            }
        );
        let mut cells = [false; 2];
        let mut banks = [mux_bank(&mut cells, 1)];
        assert_eq!(
            execute(&mut context, &circuit, &[], &mut banks).unwrap_err(),
            ExecuteError::StorageAddressWidth {
                storage: STORAGE,
                lane: LANE,
                address_bits: 0,
                declared_address_bits: 1,
            }
        );
    }

    #[derive(Default)]
    struct CountingContext {
        ands: usize,
        ors: usize,
        xors: usize,
    }

    impl HasError for CountingContext {
        type Error = Infallible;
    }

    impl ContextWithValue<bool> for CountingContext {
        type Wrapped = bool;
    }

    impl ContextWithCreate<bool> for CountingContext {
        fn create(&mut self, value: bool) -> Result<bool, Infallible> {
            Ok(value)
        }
    }

    impl ContextWithBitAnd<bool> for CountingContext {
        fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
            self.ands += 1;
            Ok(left & right)
        }

        fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
            *left = self.bitand(*left, right)?;
            Ok(())
        }
    }

    impl ContextWithBitOr<bool> for CountingContext {
        fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
            self.ors += 1;
            Ok(left | right)
        }

        fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
            *left = self.bitor(*left, right)?;
            Ok(())
        }
    }

    impl ContextWithBitXor<bool> for CountingContext {
        fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
            self.xors += 1;
            Ok(left ^ right)
        }

        fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
            *left = self.bitxor(*left, right)?;
            Ok(())
        }
    }

    struct DirectStorage {
        cells: [bool; 4],
        reads: usize,
        writes: usize,
    }

    impl ContextWithStorage<bool> for CountingContext {
        type Storage = DirectStorage;

        fn storage_read(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
        ) -> Result<bool, Self::Error> {
            storage.reads += 1;
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            Ok(storage.cells[index])
        }

        fn storage_write(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
            value: bool,
        ) -> Result<(), Self::Error> {
            storage.writes += 1;
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            storage.cells[index] = value;
            Ok(())
        }
    }

    #[derive(Default)]
    struct TestExternalRegistry {
        calls: Vec<(&'static str, usize, u64)>,
    }

    impl ExternalBitRegistry<CountingContext> for TestExternalRegistry {
        fn oracle_bit(
            &mut self,
            _context: &mut CountingContext,
            name: &str,
            args: &[bool],
            bit: usize,
            occurrence: u64,
        ) -> Result<bool, Infallible> {
            assert_eq!(name, "lookup");
            self.calls.push(("oracle", bit, occurrence));
            Ok(args[0] ^ (bit & 1 != 0))
        }

        fn rng_bit(
            &mut self,
            _context: &mut CountingContext,
            name: &str,
            bit: usize,
            occurrence: u64,
        ) -> Result<bool, Infallible> {
            assert_eq!(name, "nonce");
            self.calls.push(("rng", bit, occurrence));
            Ok((bit as u64 ^ occurrence) & 1 != 0)
        }

        fn action_store_bit(
            &mut self,
            _context: &mut CountingContext,
            name: &str,
            guard: bool,
            args: &[bool],
            fallback: bool,
            _storage: StorageId,
            _lane: LaneId,
            address: &[StorageAddressBit<bool>],
            bit: usize,
            occurrence: u64,
            cells: &mut DirectStorage,
        ) -> Result<(), Infallible> {
            assert_eq!(name, "commit");
            self.calls.push(("action", bit, occurrence));
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (place, value)| {
                    index | ((value.wire as usize) << place)
                });
            cells.cells[index] = if guard {
                args[bit % args.len()]
            } else {
                fallback
            };
            cells.writes += 1;
            Ok(())
        }
    }

    #[test]
    fn dispatches_external_bits_and_actions_store_the_guarded_value() {
        let circuit = circuit(
            5,
            vec![
                BIrStmt::OracleBit {
                    name: "lookup".into(),
                    args: vec![IRVarId(1)],
                    bit: 0,
                    occurrence: 11,
                },
                BIrStmt::RngBit {
                    name: "nonce".into(),
                    bit: 1,
                    occurrence: 12,
                },
                BIrStmt::ActionStoreBit {
                    name: "commit".into(),
                    guard: IRVarId(0),
                    args: vec![IRVarId(5), IRVarId(6)],
                    fallback: IRVarId(4),
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(2), IRVarId(3)],
                    bit: 0,
                    occurrence: 13,
                },
            ],
            vec![IRVarId(5), IRVarId(6)],
        );
        assert_eq!(
            storage_requirements(&circuit).unwrap(),
            vec![StorageRequirement {
                storage: STORAGE,
                lane: LANE,
                address_bits: 2,
                cells: 4,
            }]
        );

        let mut storage = DirectStorage {
            cells: [false; 4],
            reads: 0,
            writes: 0,
        };
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 2,
            value: &mut storage,
        }];
        let mut registry = TestExternalRegistry::default();
        let outputs = execute_with_externals(
            &mut CountingContext::default(),
            &circuit,
            &[false, true, true, false, true],
            &mut banks,
            &mut registry,
        )
        .unwrap();
        assert_eq!(outputs, [true, true]);
        assert_eq!(storage.cells, [false, true, false, false]);
        assert_eq!(storage.writes, 1);
        assert_eq!(
            registry.calls,
            vec![("oracle", 0, 11), ("rng", 1, 12), ("action", 0, 13)]
        );
    }

    #[test]
    fn external_bits_fail_closed_without_a_registry() {
        let circuit = circuit(
            0,
            vec![BIrStmt::OracleBit {
                name: "lookup".into(),
                args: vec![],
                bit: 0,
                occurrence: 0,
            }],
            vec![IRVarId(0)],
        );
        assert_eq!(
            execute(&mut CountingContext::default(), &circuit, &[], &mut []).unwrap_err(),
            ExecuteError::MissingExternalHandler {
                kind: "oracle",
                name: "lookup".into(),
            }
        );
    }

    #[test]
    fn delegates_storage_to_a_direct_context() {
        let circuit = circuit(
            3,
            vec![
                BIrStmt::StorageWrite {
                    storage: STORAGE,
                    lane: LANE,
                    src: IRVarId(0),
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
            ],
            vec![IRVarId(4)],
        );
        let mut storage = DirectStorage {
            cells: [false; 4],
            reads: 0,
            writes: 0,
        };
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 2,
            value: &mut storage,
        }];
        let outputs = execute(
            &mut CountingContext::default(),
            &circuit,
            &[true, true, false],
            &mut banks,
        )
        .unwrap();
        assert_eq!(outputs, [true]);
        assert_eq!(storage.cells, [false, true, false, false]);
        assert_eq!((storage.reads, storage.writes), (1, 1));
    }

    #[derive(Default)]
    struct TestChunkCodec(RefCell<Vec<BStmtChunk<()>>>);

    impl ChunkCodec<BStmtChunk<()>> for TestChunkCodec {
        type Error = Infallible;

        fn encode(&self, value: &BStmtChunk<()>, out: &mut Vec<u8>) -> Result<(), Self::Error> {
            let mut chunks = self.0.borrow_mut();
            let index = chunks.len() as u8;
            chunks.push(value.clone());
            out.push(index);
            Ok(())
        }

        fn decode(&self, bytes: &[u8]) -> Result<BStmtChunk<()>, Self::Error> {
            Ok(self.0.borrow()[bytes[0] as usize].clone())
        }
    }

    #[derive(Default)]
    struct TestScratch(BTreeMap<ScratchSlot, Vec<u8>>);

    impl ScratchStore for TestScratch {
        type Error = Infallible;

        fn put(&mut self, slot: ScratchSlot, bytes: &[u8]) -> Result<(), Self::Error> {
            self.0.insert(slot, bytes.to_vec());
            Ok(())
        }

        fn take(&mut self, slot: ScratchSlot) -> Result<Option<Vec<u8>>, Self::Error> {
            Ok(self.0.remove(&slot))
        }

        fn remove(&mut self, slot: ScratchSlot) -> Result<(), Self::Error> {
            self.0.remove(&slot);
            Ok(())
        }
    }

    struct BoolWireCodec;

    impl LazyWireCodec<bool> for BoolWireCodec {
        type Error = Infallible;

        fn encode(
            &self,
            wire: &bool,
            known: Option<bool>,
            out: &mut Vec<u8>,
        ) -> Result<(), Self::Error> {
            out.extend_from_slice(&[
                *wire as u8,
                known.map_or(0, |value| if value { 2 } else { 1 }),
            ]);
            Ok(())
        }

        fn decode(&self, bytes: &[u8]) -> Result<(bool, Option<bool>), Self::Error> {
            Ok((
                bytes[0] != 0,
                match bytes[1] {
                    1 => Some(false),
                    2 => Some(true),
                    _ => None,
                },
            ))
        }
    }

    #[test]
    fn chunked_runner_spills_wires_and_matches_eager_boolar() {
        let circuit = circuit(1, vec![BIrStmt::Not(IRVarId(0))], vec![IRVarId(1)]);
        let codec = TestChunkCodec::default();
        let mut source = MemorySource::default();
        let lazy = chunk_b_circuit(&circuit, 1, 3, &codec, &mut source).unwrap();
        let mut repository = Repository::new(
            source,
            CacheConfig {
                max_resident_bytes: 8,
                max_chunk_bytes: 8,
            },
        );
        let encrypted = EncryptedScratch::new(
            TestScratch::default(),
            [7; 32],
            [3; 16],
            ContentId::of(b"boolar-test"),
            1,
        )
        .unwrap();
        let mut scratch = AuthenticatedWireScratch::new(encrypted, BoolWireCodec);
        let mut banks: [StorageBank<'_, DirectStorage>; 0] = [];
        let output = execute_chunked::<_, (), _, _, _, _>(
            &mut CountingContext::default(),
            &lazy,
            &mut repository,
            &codec,
            &[true],
            &mut banks,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(output, [false]);
        assert_eq!(repository.cache().resident_bytes(), 1);
    }

    #[test]
    fn chunked_runner_dispatches_external_bits_and_storage_effects() {
        let circuit = circuit(
            5,
            vec![
                BIrStmt::OracleBit {
                    name: "lookup".into(),
                    args: vec![IRVarId(1)],
                    bit: 0,
                    occurrence: 11,
                },
                BIrStmt::RngBit {
                    name: "nonce".into(),
                    bit: 1,
                    occurrence: 12,
                },
                BIrStmt::ActionStoreBit {
                    name: "commit".into(),
                    guard: IRVarId(0),
                    args: vec![IRVarId(5), IRVarId(6)],
                    fallback: IRVarId(4),
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(2), IRVarId(3)],
                    bit: 0,
                    occurrence: 13,
                },
            ],
            vec![IRVarId(5), IRVarId(6)],
        );
        let codec = TestChunkCodec::default();
        let mut source = MemorySource::default();
        let lazy = chunk_b_circuit(&circuit, 1, 8, &codec, &mut source).unwrap();
        let mut repository = Repository::new(
            source,
            CacheConfig {
                max_resident_bytes: 64,
                max_chunk_bytes: 64,
            },
        );
        let encrypted = EncryptedScratch::new(
            TestScratch::default(),
            [7; 32],
            [3; 16],
            ContentId::of(b"boolar-external-test"),
            1,
        )
        .unwrap();
        let mut scratch = AuthenticatedWireScratch::new(encrypted, BoolWireCodec);
        let mut storage = DirectStorage {
            cells: [false; 4],
            reads: 0,
            writes: 0,
        };
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 2,
            value: &mut storage,
        }];
        let mut registry = TestExternalRegistry::default();
        let output = execute_chunked_with_externals::<_, (), _, _, _, _, _>(
            &mut CountingContext::default(),
            &lazy,
            &mut repository,
            &codec,
            &[false, true, true, false, true],
            &mut banks,
            &mut scratch,
            &mut registry,
        )
        .unwrap();
        assert_eq!(output, [true, true]);
        assert_eq!(storage.cells, [false, true, false, false]);
        assert_eq!(
            registry.calls,
            vec![("oracle", 0, 11), ("rng", 1, 12), ("action", 0, 13)]
        );
    }
}
