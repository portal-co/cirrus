#![no_std]

//! Execute fused Boolar circuits through Cirrus Boolean contexts.
//!
//! The interpreter treats [`BCircuit`] parameters as unknown symbolic bits.
//! Facts derived from Boolar `Zero`/`One` statements are kept internally so
//! storage accesses only branch on address bits that are not statically known.

extern crate alloc;

use alloc::vec::Vec;
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithValue,
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

/// Caller-owned single-bit storage for one Boolar storage lane.
///
/// The cell count must be a non-zero power of two. A storage statement whose
/// address has `N` bits uses a bank with exactly `2^N` cells.
pub struct StorageBank<'a, W> {
    /// Storage namespace selected by the Boolar statement.
    pub storage: StorageId,
    /// Type-disambiguating one-bit storage lane.
    pub lane: LaneId,
    /// The current wire for each cell, in little-endian address order.
    pub cells: &'a mut [W],
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
    /// A caller bank was empty or was not a power of two in length.
    InvalidStorageBankSize {
        storage: StorageId,
        lane: LaneId,
        cells: usize,
    },
    /// The address is too wide to describe a platform `usize` cell count.
    AddressWidthOverflow { bits: usize },
    /// A storage bank's capacity does not match the statement address width.
    StorageAddressWidth {
        storage: StorageId,
        lane: LaneId,
        address_bits: usize,
        cells: usize,
    },
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

/// Execute a fused Boolar circuit through `context`.
///
/// Parameters and caller-supplied storage cells are treated as unknown. The
/// interpreter tracks only values proven by Boolar constants and Boolean
/// operations on them, using those facts to shrink storage MUX/demux trees.
pub fn execute<C, P>(
    context: &mut C,
    circuit: &BCircuit<P>,
    inputs: &[Wire<C>],
    banks: &mut [StorageBank<'_, Wire<C>>],
) -> Result<Vec<Wire<C>>, ExecuteError<C::Error>>
where
    C: BoolarContext,
    P: Clone,
    Wire<C>: Clone,
{
    if inputs.len() != circuit.params as usize {
        return Err(ExecuteError::InputArity {
            expected: circuit.params as usize,
            found: inputs.len(),
        });
    }
    validate_banks(banks)?;

    let mut values = Vec::with_capacity(circuit.params as usize + circuit.stmts.len());
    values.extend(
        inputs
            .iter()
            .cloned()
            .map(|wire| Value { wire, known: None }),
    );
    let mut bank_facts: Vec<Vec<Option<bool>>> = banks
        .iter()
        .map(|bank| alloc::vec![None; bank.cells.len()])
        .collect();
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
                validate_address_width(*storage, *lane, address.len(), banks[bank].cells.len())?;
                read_storage(context, &banks[bank].cells, &bank_facts[bank], &address)?
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
                validate_address_width(*storage, *lane, address.len(), banks[bank].cells.len())?;
                write_storage(
                    context,
                    &mut banks[bank].cells,
                    &mut bank_facts[bank],
                    source,
                    &address,
                    &mut canonical_one,
                )?;
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

fn validate_banks<W, E>(banks: &[StorageBank<'_, W>]) -> Result<(), ExecuteError<E>> {
    for (index, bank) in banks.iter().enumerate() {
        if bank.cells.is_empty() || !bank.cells.len().is_power_of_two() {
            return Err(ExecuteError::InvalidStorageBankSize {
                storage: bank.storage,
                lane: bank.lane,
                cells: bank.cells.len(),
            });
        }
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

fn find_bank<W>(banks: &[StorageBank<'_, W>], storage: StorageId, lane: LaneId) -> Option<usize> {
    banks
        .iter()
        .position(|bank| bank.storage == storage && bank.lane == lane)
}

fn validate_address_width<E>(
    storage: StorageId,
    lane: LaneId,
    address_bits: usize,
    cells: usize,
) -> Result<(), ExecuteError<E>> {
    let expected = 1usize
        .checked_shl(address_bits as u32)
        .ok_or(ExecuteError::AddressWidthOverflow { bits: address_bits })?;
    if expected != cells {
        return Err(ExecuteError::StorageAddressWidth {
            storage,
            lane,
            address_bits,
            cells,
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
    let unknown_bits = address
        .iter()
        .enumerate()
        .filter_map(|(bit, value)| value.known.is_none().then_some(bit))
        .collect();
    let cells = (0..cells)
        .filter(|cell| {
            address.iter().enumerate().all(|(bit, value)| {
                value
                    .known
                    .is_none_or(|expected| ((cell >> bit) & 1 != 0) == expected)
            })
        })
        .collect();
    (cells, unknown_bits)
}

fn read_storage<C>(
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

fn write_storage<C>(
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
    use volar_ir_common::Node;

    const STORAGE: StorageId = StorageId(7);
    const LANE: LaneId = LaneId(3);

    fn circuit(params: u32, stmts: Vec<BIrStmt>, outputs: Vec<IRVarId>) -> BCircuit {
        BCircuit {
            params,
            stmts: stmts
                .into_iter()
                .map(|stmt| Node::new(stmt, (), None))
                .collect(),
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
        assert_eq!(
            execute(&mut (), &circuit, &[true, false], &mut []).unwrap(),
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
            let mut banks = [StorageBank {
                storage: STORAGE,
                lane: LANE,
                cells: &mut cells,
            }];
            let outputs = execute(
                &mut (),
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
        let mut pruned_banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            cells: &mut pruned_cells,
        }];
        let mut full_banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            cells: &mut full_cells,
        }];
        let mut pruned_context = CountingContext::default();
        let mut full_context = CountingContext::default();
        execute(&mut pruned_context, &pruned, &[false], &mut pruned_banks).unwrap();
        execute(
            &mut full_context,
            &full,
            &[false, false, false],
            &mut full_banks,
        )
        .unwrap();
        assert_eq!(pruned_context.ands, 2);
        assert_eq!(pruned_context.xors, 2);
        assert_eq!(full_context.ands, 7);
        assert_eq!(full_context.xors, 14);
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
        let mut pruned_banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            cells: &mut pruned_cells,
        }];
        let mut full_banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            cells: &mut full_cells,
        }];
        let mut pruned_context = CountingContext::default();
        let mut full_context = CountingContext::default();
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
        assert_eq!(pruned_context.ands, 4);
        assert_eq!(pruned_context.xors, 5);
        assert_eq!(full_context.ands, 10);
        assert_eq!(full_context.xors, 10);
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
        let mut recorder = Recorder::new();
        let left = recorder.create(false).unwrap();
        let right = recorder.create(false).unwrap();
        let outputs = execute(&mut recorder, &circuit, &[left, right], &mut []).unwrap();
        let program = recorder.finish(vec![left, right], outputs);
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
        assert_eq!(
            execute(&mut (), &circuit, &[], &mut []).unwrap_err(),
            ExecuteError::MissingStorageBank {
                storage: STORAGE,
                lane: LANE
            }
        );
        let mut cells = [false; 2];
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            cells: &mut cells,
        }];
        assert_eq!(
            execute(&mut (), &circuit, &[], &mut banks).unwrap_err(),
            ExecuteError::StorageAddressWidth {
                storage: STORAGE,
                lane: LANE,
                address_bits: 0,
                cells: 2,
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
}
