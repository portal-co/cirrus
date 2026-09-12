//! Mode-A authenticated Boolar execution traces.
//!
//! This module is deliberately an exhaustive, transparent audit protocol: it
//! commits one public Boolean value per circuit wire and opens *every* wire.
//! The verifier replays the supported pure Boolar gate relation against those
//! authenticated openings. It is a correctness reference for a future
//! chunked/folded proof, not a succinct proof and not zero knowledge.

use alloc::vec::Vec;
use core::fmt;

use sha3::{
    Sha3_256,
    digest::{Digest, FixedOutput},
};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::StorageId;

/// SHA3-256 output used by the Mode-A trace Merkle tree.
pub type TraceDigest = [u8; 32];

const LEAF_DOMAIN: u8 = 0x20;
const NODE_DOMAIN: u8 = 0x21;

/// A bottom-up authentication path for a single wire leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireAuthPath {
    /// Sibling hashes, starting at the leaf level.
    pub siblings: Vec<TraceDigest>,
}

/// An authenticated value for exactly one canonical wire index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireOpening {
    /// Canonical `WireId`, represented by its zero-based trace-leaf index.
    pub wire: usize,
    /// The public Boolean value assigned to this wire.
    pub value: bool,
    /// Merkle authentication path under [`BoolarTraceAudit::trace_root`].
    pub path: WireAuthPath,
}

/// A complete (and hence non-succinct) opening of a Boolar wire trace.
///
/// Openings must be present once, in canonical increasing wire order. The
/// first `circuit.params` openings are inputs; statement `i` writes wire
/// `circuit.params + i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoolarTraceAudit {
    /// Commitment to the complete canonical wire trace.
    pub trace_root: TraceDigest,
    /// Every wire opening. This is intentionally linear in the trace size.
    pub openings: Vec<WireOpening>,
    /// Full-trace RAM permutation witness for Boolar storage statements.
    ///
    /// It is intentionally public and linear-size in Mode A. `None` is valid
    /// only for a circuit with no storage pre-initialization or access.
    pub memory: Option<MemoryPermutationAudit>,
}

/// Whether one access observes or updates a bit-granular memory cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryAccessKind {
    /// Observe the value currently held by a cell.
    Read,
    /// Replace the value currently held by a cell.
    Write,
}

/// One fully public bit-granular RAM access.
///
/// `time` is the canonical execution position: all `pre_init` writes precede
/// statement accesses, and statement accesses follow `circuit.stmts` order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryAccess {
    /// Storage-bank identity from canonical Boolar IR.
    pub storage: StorageId,
    /// Bit-lane identity from canonical Boolar IR.
    pub lane: LaneId,
    /// Little-endian Boolean address bits, retaining arbitrary IR width.
    pub address: Vec<bool>,
    /// Canonical global access order.
    pub time: usize,
    /// Read or write operation.
    pub kind: MemoryAccessKind,
    /// Observed or written cell bit.
    pub value: bool,
}

/// A transparent RAM permutation witness.
///
/// `execution` is the IR-order access table. `address_sorted` contains exactly
/// the same records sorted by `(storage, lane, address, time)`. The verifier
/// checks their exact multiset equality and checks read/write consistency in
/// the sorted view. This is the full-trace analogue of a RAM permutation
/// argument; it becomes a succinct permutation proof only in a later IOP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPermutationAudit {
    /// Accesses in program order, derived from the circuit and wire openings.
    pub execution: Vec<MemoryAccess>,
    /// The same accesses sorted by memory location then execution time.
    pub address_sorted: Vec<MemoryAccess>,
}

/// Why a Mode-A trace cannot be made or accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceAuditError {
    /// The trace does not have one value for every input and statement wire.
    WrongWireCount { expected: usize, found: usize },
    /// This initial auditor intentionally supports only pure Boolean gates.
    UnsupportedStatement { wire: usize },
    /// An IR reference did not identify a prior input or statement wire.
    InvalidWireReference { wire: usize, referenced: u32 },
    /// The proof does not contain canonical, complete openings.
    NonCanonicalOpening { expected: usize, found: usize },
    /// An opening does not authenticate under the statement's trace root.
    InvalidAuthentication { wire: usize },
    /// A public input differs from the first canonical wire leaves.
    InputMismatch { wire: usize },
    /// A gate's committed output is inconsistent with committed inputs.
    GateMismatch { wire: usize },
    /// A designated output differs from the public output claim.
    OutputMismatch { output: usize },
    /// The supplied memory table does not match the circuit and wire trace.
    MemoryExecutionMismatch,
    /// The two public RAM tables are not a permutation of one another.
    MemoryNotPermutation,
    /// A memory read does not equal the most recent prior write (or zero).
    InvalidMemoryRead { time: usize },
    /// A tree cannot commit an empty trace.
    EmptyTrace,
}

impl fmt::Display for TraceAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongWireCount { expected, found } => {
                write!(f, "trace has {found} wires, expected {expected}")
            }
            Self::UnsupportedStatement { wire } => {
                write!(f, "wire {wire} is produced by unsupported Boolar statement")
            }
            Self::InvalidWireReference { wire, referenced } => {
                write!(f, "wire {wire} references invalid prior wire {referenced}")
            }
            Self::NonCanonicalOpening { expected, found } => {
                write!(f, "opening has wire index {found}, expected {expected}")
            }
            Self::InvalidAuthentication { wire } => {
                write!(f, "wire {wire} does not authenticate under trace root")
            }
            Self::InputMismatch { wire } => write!(f, "public input differs at wire {wire}"),
            Self::GateMismatch { wire } => write!(f, "gate relation fails at wire {wire}"),
            Self::OutputMismatch { output } => write!(f, "public output differs at index {output}"),
            Self::MemoryExecutionMismatch => {
                f.write_str("memory execution table does not match circuit trace")
            }
            Self::MemoryNotPermutation => {
                f.write_str("memory address table is not a permutation of execution table")
            }
            Self::InvalidMemoryRead { time } => write!(
                f,
                "memory read at time {time} does not observe latest value"
            ),
            Self::EmptyTrace => f.write_str("cannot commit an empty wire trace"),
        }
    }
}

impl core::error::Error for TraceAuditError {}

/// Build an exhaustive audit opening for `wire_values`.
///
/// This function only commits values. Call [`BoolarTraceAudit::verify`] to
/// establish that they evaluate `circuit`; separating these operations lets
/// tests construct malformed but well-authenticated traces.
pub fn commit_boolar_trace<P: Clone>(
    circuit: &BCircuit<P>,
    wire_values: &[bool],
) -> Result<BoolarTraceAudit, TraceAuditError> {
    let expected = wire_count(circuit);
    if wire_values.len() != expected {
        return Err(TraceAuditError::WrongWireCount {
            expected,
            found: wire_values.len(),
        });
    }
    let tree = TraceTree::commit(wire_values)?;
    let memory_execution = expected_memory_accesses(circuit, wire_values)?;
    Ok(BoolarTraceAudit {
        trace_root: tree.root(),
        openings: wire_values
            .iter()
            .copied()
            .enumerate()
            .map(|(wire, value)| WireOpening {
                wire,
                value,
                path: tree.open(wire),
            })
            .collect(),
        memory: (!memory_execution.is_empty()).then(|| {
            let mut address_sorted = memory_execution.clone();
            address_sorted.sort();
            MemoryPermutationAudit {
                execution: memory_execution,
                address_sorted,
            }
        }),
    })
}

impl BoolarTraceAudit {
    /// Authenticate and exhaustively replay a pure Boolar circuit.
    ///
    /// `public_inputs` and `claimed_outputs` are statement values, not merely
    /// metadata: they are checked against the canonical committed wire leaves.
    /// Storage reads and writes are checked through the full public RAM
    /// permutation witness. External and action statements remain unsupported
    /// until their own canonical trace ABI is specified.
    pub fn verify<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
        public_inputs: &[bool],
        claimed_outputs: &[bool],
    ) -> Result<(), TraceAuditError> {
        let expected = wire_count(circuit);
        if self.openings.len() != expected {
            return Err(TraceAuditError::WrongWireCount {
                expected,
                found: self.openings.len(),
            });
        }
        if public_inputs.len() != circuit.params as usize {
            return Err(TraceAuditError::WrongWireCount {
                expected: circuit.params as usize,
                found: public_inputs.len(),
            });
        }
        if claimed_outputs.len() != circuit.outputs.len() {
            return Err(TraceAuditError::WrongWireCount {
                expected: circuit.outputs.len(),
                found: claimed_outputs.len(),
            });
        }

        let mut values = Vec::with_capacity(expected);
        for (expected_wire, opening) in self.openings.iter().enumerate() {
            if opening.wire != expected_wire {
                return Err(TraceAuditError::NonCanonicalOpening {
                    expected: expected_wire,
                    found: opening.wire,
                });
            }
            if !verify_opening(
                &self.trace_root,
                expected,
                opening.wire,
                opening.value,
                &opening.path,
            ) {
                return Err(TraceAuditError::InvalidAuthentication {
                    wire: expected_wire,
                });
            }
            values.push(opening.value);
        }

        for (wire, input) in public_inputs.iter().copied().enumerate() {
            if values[wire] != input {
                return Err(TraceAuditError::InputMismatch { wire });
            }
        }

        for (statement_index, node) in circuit.stmts.iter().enumerate() {
            let wire = circuit.params as usize + statement_index;
            let required = match &node.kind {
                BIrStmt::Zero => false,
                BIrStmt::One => true,
                BIrStmt::And(a, b) => value_at(&values, wire, *a)? & value_at(&values, wire, *b)?,
                BIrStmt::Or(a, b) => value_at(&values, wire, *a)? | value_at(&values, wire, *b)?,
                BIrStmt::Xor(a, b) => value_at(&values, wire, *a)? ^ value_at(&values, wire, *b)?,
                BIrStmt::Not(a) => !value_at(&values, wire, *a)?,
                BIrStmt::StorageRead { .. } => values[wire],
                BIrStmt::StorageWrite { .. } => false,
                _ => return Err(TraceAuditError::UnsupportedStatement { wire }),
            };
            if values[wire] != required {
                return Err(TraceAuditError::GateMismatch { wire });
            }
        }

        self.verify_memory(circuit, &values)?;

        for (output, (wire, claimed)) in circuit.outputs.iter().zip(claimed_outputs).enumerate() {
            let value = value_at(&values, expected, *wire)?;
            if value != *claimed {
                return Err(TraceAuditError::OutputMismatch { output });
            }
        }
        Ok(())
    }

    fn verify_memory<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
        values: &[bool],
    ) -> Result<(), TraceAuditError> {
        let expected = expected_memory_accesses(circuit, values)?;
        match (&self.memory, expected.is_empty()) {
            (None, true) => return Ok(()),
            (None, false) => return Err(TraceAuditError::MemoryExecutionMismatch),
            (Some(_), true) => return Err(TraceAuditError::MemoryExecutionMismatch),
            (Some(memory), false) => {
                if memory.execution != expected {
                    return Err(TraceAuditError::MemoryExecutionMismatch);
                }
                let mut canonical = memory.execution.clone();
                canonical.sort();
                if canonical != memory.address_sorted {
                    return Err(TraceAuditError::MemoryNotPermutation);
                }
                let mut previous: Option<&MemoryAccess> = None;
                for access in &memory.address_sorted {
                    let same_cell = previous.is_some_and(|last| {
                        last.storage == access.storage
                            && last.lane == access.lane
                            && last.address == access.address
                    });
                    let expected_read = if same_cell {
                        previous.expect("same cell has previous access").value
                    } else {
                        false
                    };
                    if access.kind == MemoryAccessKind::Read && access.value != expected_read {
                        return Err(TraceAuditError::InvalidMemoryRead { time: access.time });
                    }
                    previous = Some(access);
                }
            }
        }
        Ok(())
    }
}

fn expected_memory_accesses<P: Clone>(
    circuit: &BCircuit<P>,
    values: &[bool],
) -> Result<Vec<MemoryAccess>, TraceAuditError> {
    let mut accesses = Vec::new();
    let mut time = 0usize;
    for segment in &circuit.pre_init {
        for (offset, value) in segment.data.iter().copied().enumerate() {
            accesses.push(MemoryAccess {
                storage: segment.storage,
                lane: segment.lane,
                address: increment_address(&segment.addr, offset),
                time,
                kind: MemoryAccessKind::Write,
                value,
            });
            time += 1;
        }
    }
    for (statement_index, node) in circuit.stmts.iter().enumerate() {
        let wire = circuit.params as usize + statement_index;
        match &node.kind {
            BIrStmt::StorageRead {
                storage,
                lane,
                addr,
            } => {
                accesses.push(MemoryAccess {
                    storage: *storage,
                    lane: *lane,
                    address: address_value(values, wire, addr)?,
                    time,
                    kind: MemoryAccessKind::Read,
                    value: values[wire],
                });
                time += 1;
            }
            BIrStmt::StorageWrite {
                storage,
                lane,
                src,
                addr,
            } => {
                accesses.push(MemoryAccess {
                    storage: *storage,
                    lane: *lane,
                    address: address_value(values, wire, addr)?,
                    time,
                    kind: MemoryAccessKind::Write,
                    value: value_at(values, wire, *src)?,
                });
                time += 1;
            }
            _ => {}
        }
    }
    Ok(accesses)
}

fn address_value(
    values: &[bool],
    current_wire: usize,
    address: &[IRVarId],
) -> Result<Vec<bool>, TraceAuditError> {
    address
        .iter()
        .map(|wire| value_at(values, current_wire, *wire))
        .collect()
}

fn increment_address(address: &[bool], mut offset: usize) -> Vec<bool> {
    let mut result = Vec::with_capacity(address.len());
    let mut carry = false;
    for &bit in address {
        let addend = offset & 1 != 0;
        result.push(bit ^ addend ^ carry);
        carry = (bit & addend) | (bit & carry) | (addend & carry);
        offset >>= 1;
    }
    // The Boolar interpreter's static-address helper also works at the
    // declared finite width: overflow intentionally wraps.
    result
}

fn wire_count<P: Clone>(circuit: &BCircuit<P>) -> usize {
    circuit.params as usize + circuit.stmts.len()
}

fn value_at(values: &[bool], current_wire: usize, wire: IRVarId) -> Result<bool, TraceAuditError> {
    let index = wire.0 as usize;
    if index >= current_wire {
        return Err(TraceAuditError::InvalidWireReference {
            wire: current_wire,
            referenced: wire.0,
        });
    }
    Ok(values[index])
}

#[derive(Clone, Debug)]
struct TraceTree {
    layers: Vec<Vec<TraceDigest>>,
    original_len: usize,
}

impl TraceTree {
    fn commit(values: &[bool]) -> Result<Self, TraceAuditError> {
        if values.is_empty() {
            return Err(TraceAuditError::EmptyTrace);
        }
        let original_len = values.len();
        let mut level: Vec<_> = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| hash_leaf(original_len, index, value))
            .collect();
        let padded = level.len().next_power_of_two();
        while level.len() < padded {
            // Explicit indexed padding avoids treating a copied final wire as
            // an additional committed wire.
            level.push(hash_padding(original_len, level.len()));
        }
        let mut layers = alloc::vec![level.clone()];
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks_exact(2) {
                next.push(hash_node(&pair[0], &pair[1]));
            }
            layers.push(next.clone());
            level = next;
        }
        Ok(Self {
            layers,
            original_len,
        })
    }

    fn root(&self) -> TraceDigest {
        self.layers.last().expect("nonempty tree")[0]
    }

    fn open(&self, wire: usize) -> WireAuthPath {
        debug_assert!(wire < self.original_len);
        let mut index = wire;
        let mut siblings = Vec::with_capacity(self.layers.len().saturating_sub(1));
        for layer in &self.layers[..self.layers.len() - 1] {
            siblings.push(layer[index ^ 1]);
            index /= 2;
        }
        WireAuthPath { siblings }
    }
}

fn hash_leaf(trace_len: usize, wire: usize, value: bool) -> TraceDigest {
    let mut hash = Sha3_256::new();
    hash.update([LEAF_DOMAIN]);
    hash.update((trace_len as u64).to_le_bytes());
    hash.update((wire as u64).to_le_bytes());
    hash.update([u8::from(value)]);
    hash.finalize_fixed().into()
}

fn hash_padding(trace_len: usize, padded_index: usize) -> TraceDigest {
    let mut hash = Sha3_256::new();
    hash.update([LEAF_DOMAIN]);
    hash.update((trace_len as u64).to_le_bytes());
    hash.update((padded_index as u64).to_le_bytes());
    hash.update([0xff]);
    hash.finalize_fixed().into()
}

fn hash_node(left: &TraceDigest, right: &TraceDigest) -> TraceDigest {
    let mut hash = Sha3_256::new();
    hash.update([NODE_DOMAIN]);
    hash.update(left);
    hash.update(right);
    hash.finalize_fixed().into()
}

fn verify_opening(
    root: &TraceDigest,
    trace_len: usize,
    wire: usize,
    value: bool,
    path: &WireAuthPath,
) -> bool {
    if trace_len == 0
        || wire >= trace_len
        || path.siblings.len() != trace_len.next_power_of_two().ilog2() as usize
    {
        return false;
    }
    let mut current = hash_leaf(trace_len, wire, value);
    let mut index = wire;
    for sibling in &path.siblings {
        current = if index & 1 == 0 {
            hash_node(&current, sibling)
        } else {
            hash_node(sibling, &current)
        };
        index /= 2;
    }
    current == *root
}
