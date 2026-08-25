#![no_std]

//! Execute fused Boolar circuits through Cirrus Boolean contexts.
//!
//! The interpreter treats [`BCircuit`] parameters as unknown symbolic bits.
//! Facts derived from Boolar `Zero`/`One` statements are kept internally so
//! storage accesses only branch on address bits that are not statically known.

extern crate alloc;

use alloc::vec::Vec;
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithStorage,
    ContextWithValue, HasError, StorageAddressBit,
};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::StorageId;

type Wire<C> = <C as ContextWithValue<bool>>::Wrapped;

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

/// Caller-owned storage for one Boolar storage lane.
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
    /// The underlying Cirrus Boolean context rejected an operation.
    Context(E),
}

#[derive(Clone)]
struct Value<W> {
    wire: W,
    known: Option<bool>,
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

fn known_storage_address<W>(address: &[StorageAddressBit<W>]) -> Option<usize> {
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
    execute_impl(context, circuit, inputs, banks)
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
    execute_impl(context, circuit, inputs, banks)
}

fn execute_impl<C, P>(
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
            BIrStmt::OracleCall { .. }
            | BIrStmt::OracleBit { .. }
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
    let shift = u32::try_from(bits)
        .map_err(|_| StorageLayoutError::AddressWidthOverflow { bits })?;
    1usize
        .checked_shl(shift)
        .ok_or(StorageLayoutError::AddressWidthOverflow { bits })
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

fn apply_and<C>(
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

fn one<C>(
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

fn apply_not<C>(
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

fn apply_mux<C>(
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

fn candidates<W>(address: &[Value<W>], cells: usize) -> (Vec<usize>, Vec<usize>) {
    let unknown_bits: Vec<usize> = address
        .iter()
        .enumerate()
        .filter_map(|(bit, value)| value.known.is_none().then_some(bit))
        .collect();
    if unknown_bits.is_empty() {
        let mut index = 0usize;
        for (bit, value) in address.iter().enumerate() {
            if value.known == Some(true) {
                if bit >= usize::BITS as usize {
                    return (Vec::new(), unknown_bits);
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
            unknown_bits,
        );
    }
    let cells = (0..cells)
        .filter(|cell| {
            address.iter().enumerate().all(|(bit, value)| {
                if bit >= usize::BITS as usize {
                    return value.known != Some(true);
                }
                value
                    .known
                    .is_none_or(|expected| ((cell >> bit) & 1 != 0) == expected)
            })
        })
        .collect();
    (cells, unknown_bits)
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

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use cirrus_core::{
        ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, HasError,
    };
    use cirrus_recompile_core::{Recorder, interpret};
    use core::convert::Infallible;
    use volar_ir::boolar::BIrPreInitSegment;
    use volar_ir_common::Node;

    #[test]
    fn bounded_riscv_fixture_lowers_to_initialized_five_bit_memory() {
        use volar_riscv_test_programs::{parse_and_expand, wat_gen::test_program_wat};
        use volar_vaffle_target::{
            VaffleTarget, import_config::WaffleImportConfig, waffle_lower::lower_waffle_module,
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
}
