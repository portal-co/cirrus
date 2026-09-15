//! Sparse cell backing for the MUX-tree storage lowering.
//!
//! [`super::MuxTreeContext`]'s `StorageBank` is `2^address_bits` cells,
//! allocated eagerly. [`SparseMuxTreeContext`] is a sibling context whose
//! storage ([`SparseBank`]) materializes a cell only when a concrete address
//! actually reads or writes it, so a wide (e.g. 34-bit) lane costs only as
//! many cells as it has ever touched. Symbolic reads MUX over the *live* set
//! instead of the full address space; symbolic writes do the same but fail
//! closed ([`SparseStorageError::UnboundedWrite`]) when the address could
//! reach a cell that has never been materialized, since sparse storage can't
//! safely represent "some yet-unseen cell might now hold a new value."
//!
//! This is a second, specific [`ContextWithStorage`] implementation living
//! beside the dense one, not a generic replacement for it — see
//! `sparse-banks.md`.

use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithStorage,
    ContextWithValue, HasError, StorageAddressBit,
};

use crate::{
    BoolarContext, ExecuteError, Value, Wire, apply_and, apply_mux, apply_not,
    index_matches_known_bits, known_storage_address, one, unknown_bits,
};

/// Sparse, lazily-materialized backing for one [`SparseMuxTreeContext`] lane.
///
/// Cells are keyed by their concrete integer address and created only when a
/// known-address access first touches them. Un-materialized cells all read
/// as one shared, lazily-created default wire.
pub struct SparseBank<W> {
    cells: BTreeMap<usize, Value<W>>,
    default: Option<Value<W>>,
}

impl<W> Default for SparseBank<W> {
    fn default() -> Self {
        Self {
            cells: BTreeMap::new(),
            default: None,
        }
    }
}

impl<W> SparseBank<W> {
    /// An empty bank: no cell has been materialized yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of cells actually materialized so far.
    #[must_use]
    pub fn live_len(&self) -> usize {
        self.cells.len()
    }
}

/// Why [`SparseMuxTreeContext`] could not complete a storage access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SparseStorageError<E> {
    /// The caller-supplied Boolean context returned this error.
    Context(E),
    /// A symbolic write could reach a cell that has never been
    /// materialized. Sparse storage can't represent "an unseen cell might
    /// now hold a new value" without allocating densely, so it fails closed
    /// instead of writing an incomplete result.
    UnboundedWrite,
}

impl<E: fmt::Display> fmt::Display for SparseStorageError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(error) => error.fmt(formatter),
            Self::UnboundedWrite => formatter.write_str(
                "symbolic write could reach a cell sparse storage has never materialized",
            ),
        }
    }
}

impl<E: core::error::Error> core::error::Error for SparseStorageError<E> {}

fn map_execute_error<E>(error: ExecuteError<E>) -> SparseStorageError<E> {
    match error {
        ExecuteError::Context(error) => SparseStorageError::Context(error),
        _ => unreachable!("the sparse MUX tree only reports context errors"),
    }
}

fn default_value<C>(
    bank: &mut SparseBank<Wire<C>>,
    context: &mut C,
) -> Result<Value<Wire<C>>, SparseStorageError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    if let Some(value) = &bank.default {
        return Ok(value.clone());
    }
    let wire = context.create(false).map_err(SparseStorageError::Context)?;
    let value = Value {
        wire,
        known: Some(false),
    };
    bank.default = Some(value.clone());
    Ok(value)
}

fn known_read<C>(
    bank: &mut SparseBank<Wire<C>>,
    context: &mut C,
    index: usize,
) -> Result<Value<Wire<C>>, SparseStorageError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    if let Some(value) = bank.cells.get(&index) {
        return Ok(value.clone());
    }
    let value = default_value(bank, context)?;
    bank.cells.insert(index, value.clone());
    Ok(value)
}

/// AND, over every still-unknown bit of `address`, of "does this address bit
/// equal `index`'s bit here" — i.e. whether `address` equals the concrete
/// `index`, given the known bits already match (checked by the caller via
/// [`index_matches_known_bits`]).
fn equality_wire<C>(
    context: &mut C,
    address: &[Value<Wire<C>>],
    bits: &[usize],
    index: usize,
    canonical_one: &mut Option<Wire<C>>,
) -> Result<Value<Wire<C>>, SparseStorageError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let mut result = Value {
        wire: one(context, canonical_one).map_err(map_execute_error)?,
        known: Some(true),
    };
    for &bit in bits {
        let addr_bit = address[bit].clone();
        let term = if (index >> bit) & 1 != 0 {
            addr_bit
        } else {
            apply_not(context, addr_bit, canonical_one).map_err(map_execute_error)?
        };
        result = apply_and(context, result, term).map_err(map_execute_error)?;
    }
    Ok(result)
}

fn sparse_read_mux<C>(
    context: &mut C,
    bank: &mut SparseBank<Wire<C>>,
    address: &[Value<Wire<C>>],
) -> Result<Value<Wire<C>>, SparseStorageError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let bits = unknown_bits(address);
    let matches: Vec<(usize, Value<Wire<C>>)> = bank
        .cells
        .iter()
        .filter(|(index, _)| index_matches_known_bits(**index, address))
        .map(|(index, value)| (*index, value.clone()))
        .collect();
    let mut result = default_value(bank, context)?;
    let mut canonical_one = None;
    for (index, cell) in matches {
        let is_match = equality_wire(context, address, &bits, index, &mut canonical_one)?;
        result = apply_mux(context, is_match, result, cell).map_err(map_execute_error)?;
    }
    Ok(result)
}

fn sparse_write_mux<C>(
    context: &mut C,
    bank: &mut SparseBank<Wire<C>>,
    address: &[Value<Wire<C>>],
    source: Value<Wire<C>>,
) -> Result<(), SparseStorageError<C::Error>>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    let bits = unknown_bits(address);
    let needed = u32::try_from(bits.len())
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift));
    let candidates: Vec<usize> = bank
        .cells
        .keys()
        .copied()
        .filter(|index| index_matches_known_bits(*index, address))
        .collect();
    if Some(candidates.len()) != needed {
        return Err(SparseStorageError::UnboundedWrite);
    }
    let mut canonical_one = None;
    for index in candidates {
        let is_match = equality_wire(context, address, &bits, index, &mut canonical_one)?;
        let old = bank
            .cells
            .get(&index)
            .cloned()
            .expect("candidate index is a live key");
        let next = apply_mux(context, is_match, old, source.clone()).map_err(map_execute_error)?;
        bank.cells.insert(index, next);
    }
    Ok(())
}

/// A sparse-storage sibling of [`super::MuxTreeContext`].
///
/// Wrap a Boolean context in this adapter to back storage with
/// [`SparseBank`] instead of a dense `[Wire]` slice. It's a distinct type
/// rather than a generic parameter on `MuxTreeContext`: `Storage` is an
/// associated type, so one wrapper can't carry two different storage kinds
/// for the same inner context.
pub struct SparseMuxTreeContext<C> {
    inner: C,
}

impl<C> SparseMuxTreeContext<C> {
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

impl<C: HasError> HasError for SparseMuxTreeContext<C> {
    type Error = SparseStorageError<C::Error>;
}

impl<C, Val> ContextWithValue<Val> for SparseMuxTreeContext<C>
where
    C: ContextWithValue<Val>,
{
    type Wrapped = C::Wrapped;
}

impl<C> ContextWithCreate<bool> for SparseMuxTreeContext<C>
where
    C: ContextWithCreate<bool>,
{
    fn create(&mut self, value: bool) -> Result<Self::Wrapped, Self::Error> {
        self.inner
            .create(value)
            .map_err(SparseStorageError::Context)
    }
}

impl<C> ContextWithBitAnd<bool> for SparseMuxTreeContext<C>
where
    C: ContextWithBitAnd<bool>,
{
    fn bitand(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner
            .bitand(left, right)
            .map_err(SparseStorageError::Context)
    }

    fn bitand_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitand_assign(left, right)
            .map_err(SparseStorageError::Context)
    }
}

impl<C> ContextWithBitOr<bool> for SparseMuxTreeContext<C>
where
    C: ContextWithBitOr<bool>,
{
    fn bitor(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner
            .bitor(left, right)
            .map_err(SparseStorageError::Context)
    }

    fn bitor_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitor_assign(left, right)
            .map_err(SparseStorageError::Context)
    }
}

impl<C> ContextWithBitXor<bool> for SparseMuxTreeContext<C>
where
    C: ContextWithBitXor<bool>,
{
    fn bitxor(
        &mut self,
        left: Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<Self::Wrapped, Self::Error> {
        self.inner
            .bitxor(left, right)
            .map_err(SparseStorageError::Context)
    }

    fn bitxor_assign(
        &mut self,
        left: &mut Self::Wrapped,
        right: Self::Wrapped,
    ) -> Result<(), Self::Error> {
        self.inner
            .bitxor_assign(left, right)
            .map_err(SparseStorageError::Context)
    }
}

impl<C> ContextWithStorage<bool> for SparseMuxTreeContext<C>
where
    C: BoolarContext,
    Wire<C>: Clone,
{
    type Storage = SparseBank<Wire<C>>;

    fn storage_read(
        &mut self,
        bank: &mut Self::Storage,
        address: &[StorageAddressBit<Wire<C>>],
    ) -> Result<Wire<C>, Self::Error> {
        if let Some(index) = known_storage_address(address) {
            return known_read(bank, &mut self.inner, index).map(|value| value.wire);
        }
        let address: Vec<Value<Wire<C>>> = address
            .iter()
            .cloned()
            .map(|bit| Value {
                wire: bit.wire,
                known: bit.known,
            })
            .collect();
        sparse_read_mux(&mut self.inner, bank, &address).map(|value| value.wire)
    }

    fn storage_write(
        &mut self,
        bank: &mut Self::Storage,
        address: &[StorageAddressBit<Wire<C>>],
        value: Wire<C>,
    ) -> Result<(), Self::Error> {
        if let Some(index) = known_storage_address(address) {
            bank.cells.insert(
                index,
                Value {
                    wire: value,
                    known: None,
                },
            );
            return Ok(());
        }
        let address: Vec<Value<Wire<C>>> = address
            .iter()
            .cloned()
            .map(|bit| Value {
                wire: bit.wire,
                known: bit.known,
            })
            .collect();
        sparse_write_mux(
            &mut self.inner,
            bank,
            &address,
            Value {
                wire: value,
                known: None,
            },
        )
    }
}
