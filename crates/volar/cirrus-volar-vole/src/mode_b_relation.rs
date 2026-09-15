//! Backend-neutral Mode-B relation descriptor.
//!
//! This is deliberately not a proof system. It describes the Boolean circuit
//! relation as R1CS-shaped rows whose coefficients remain signed integers until
//! a backend chooses a non-binary field.  Thus the same relation can be
//! lowered to KoalaBear/Spartan-WHIR (or another field) without changing the
//! Boolar semantics.

use alloc::{vec, vec::Vec};
use core::{cmp::Ordering, fmt};
use sha3::{Digest, Sha3_256};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::StorageId;

/// Stable relation version. Changes to wire/row semantics require a new one.
pub const MODE_B_RELATION_VERSION: u32 = 1;
/// KoalaBear base-field modulus used by Spartan-WHIR.
pub const KOALABEAR_MODULUS: u32 = 2_130_706_433;
/// The reference Spartan-WHIR target is KoalaBear's quintic extension.
pub const KOALABEAR_QUINTIC_DEGREE: usize = 5;
/// Full 32-bit address width supported by the initial RAM format.
pub const PRIME_RAM_ADDRESS_BITS: usize = 32;
/// Bounded storage-id width in the initial format.
pub const PRIME_RAM_STORAGE_BITS: usize = 16;
/// Bounded lane-id width in the initial format.
pub const PRIME_RAM_LANE_BITS: usize = 16;
/// Bounded execution-time width in the initial format.
pub const PRIME_RAM_TIME_BITS: usize = 32;
/// Domain-separated format identifier for the KoalaBear quintic RAM ABI.
pub const PRIME_RAM_FORMAT: &[u8] = b"koalabear-ext5-ram-v1";
/// A structural circuit identifier, independent of provenance annotations.
pub type CircuitId = [u8; 32];

/// A signed, field-independent linear combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearCombination {
    /// Constant coefficient.
    pub constant: i64,
    /// `(variable, coefficient)` terms. Variables are canonical and unique.
    pub terms: Vec<(usize, i64)>,
}
impl LinearCombination {
    /// A field-independent constant.
    pub fn constant(c: i64) -> Self {
        Self {
            constant: c,
            terms: Vec::new(),
        }
    }
    /// A single variable with coefficient one.
    pub fn var(v: usize) -> Self {
        Self {
            constant: 0,
            terms: vec![(v, 1)],
        }
    }
    /// Append one variable term.
    pub fn add(mut self, v: usize, c: i64) -> Self {
        self.terms.push((v, c));
        self
    }
    /// Append several variable terms.
    pub fn add_terms(mut self, terms: Vec<(usize, i64)>) -> Self {
        self.terms.extend(terms);
        self
    }
}

/// One R1CS equation `A(w) * B(w) = C(w)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csRow {
    /// Left linear combination.
    pub a: LinearCombination,
    /// Right linear combination.
    pub b: LinearCombination,
    /// Result linear combination.
    pub c: LinearCombination,
}

/// A public statement or challenge-slot binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicBinding {
    Input {
        index: usize,
        wire: usize,
    },
    Output {
        index: usize,
        wire: usize,
    },
    /// One base-field coefficient of the RAM permutation challenges. Indices
    /// `0..5` are `gamma` coefficients and `5..10` are `eta` coefficients.
    RamPermutationChallenge {
        index: usize,
        wire: usize,
    },
}

/// Generic sink used by the canonical Boolar statement scheduler.
///
/// Implementations decide whether a scheduled operation constrains actual
/// Boolean values, VOLE verifier correlations, or another profile-specific
/// representation. The scheduler owns wire order and fail-closed statement
/// dispatch; semantics own the emitted rows.
pub trait ConstraintSemantics {
    /// Emit constraints for one supported statement. `wire` is the canonical
    /// statement output index (`circuit.params + statement_index`).
    fn statement(
        &mut self,
        wire: usize,
        statement: &BIrStmt<IRVarId, StorageId>,
    ) -> Result<(), ModeBRelationError>;
}

/// The current actual-Boolean-value Mode-B lowering, exposed as scheduler
/// semantics. It is row-for-row compatible with [`ModeBRelation::from_boolar`].
#[derive(Clone, Debug)]
pub struct ActualBooleanSemantics {
    /// R1CS rows emitted so far.
    pub rows: Vec<R1csRow>,
    next_helper: usize,
}

impl ActualBooleanSemantics {
    /// Start a lowering for `wire_count` canonical primary wires.
    pub fn new(wire_count: usize) -> Self {
        let mut this = Self {
            rows: Vec::new(),
            next_helper: wire_count,
        };
        for wire in 0..wire_count {
            this.booleanity(wire);
        }
        this
    }

    fn booleanity(&mut self, wire: usize) {
        self.rows.push(R1csRow {
            a: LinearCombination::var(wire),
            b: LinearCombination::var(wire).add(wire, -1),
            c: LinearCombination::constant(0),
        });
    }

    fn prior(wire: usize, reference: IRVarId) -> Result<usize, ModeBRelationError> {
        let index = reference.0 as usize;
        if index >= wire {
            Err(ModeBRelationError::InvalidWireReference {
                wire,
                referenced: reference.0,
            })
        } else {
            Ok(index)
        }
    }

    fn linear_result(&mut self, a: LinearCombination, c: usize) {
        self.rows.push(R1csRow {
            a,
            b: LinearCombination::constant(1),
            c: LinearCombination::var(c),
        });
    }

    /// Number of primary wires plus allocated multiplication helpers.
    pub fn witness_count(&self) -> usize {
        self.next_helper
    }
}

impl ConstraintSemantics for ActualBooleanSemantics {
    fn statement(
        &mut self,
        wire: usize,
        statement: &BIrStmt<IRVarId, StorageId>,
    ) -> Result<(), ModeBRelationError> {
        match statement {
            BIrStmt::Zero => self.linear_result(LinearCombination::constant(0), wire),
            BIrStmt::One => self.linear_result(LinearCombination::constant(1), wire),
            BIrStmt::And(x, y) => {
                let x = Self::prior(wire, *x)?;
                let y = Self::prior(wire, *y)?;
                self.rows.push(R1csRow {
                    a: LinearCombination::var(x),
                    b: LinearCombination::var(y),
                    c: LinearCombination::var(wire),
                });
            }
            BIrStmt::Not(x) => {
                let x = Self::prior(wire, *x)?;
                self.linear_result(LinearCombination::constant(1).add(x, -1), wire);
            }
            BIrStmt::Xor(x, y) => {
                let x = Self::prior(wire, *x)?;
                let y = Self::prior(wire, *y)?;
                let helper = self.next_helper;
                self.next_helper += 1;
                self.rows.push(R1csRow {
                    a: LinearCombination::var(x),
                    b: LinearCombination::var(y),
                    c: LinearCombination::var(helper),
                });
                self.linear_result(LinearCombination::var(x).add(y, 1).add(helper, -2), wire);
            }
            BIrStmt::Or(x, y) => {
                let x = Self::prior(wire, *x)?;
                let y = Self::prior(wire, *y)?;
                let helper = self.next_helper;
                self.next_helper += 1;
                self.rows.push(R1csRow {
                    a: LinearCombination::var(x),
                    b: LinearCombination::var(y),
                    c: LinearCombination::var(helper),
                });
                self.linear_result(LinearCombination::var(x).add(y, 1).add(helper, -1), wire);
            }
            // The read value is constrained by the RAM relation.
            BIrStmt::StorageRead { .. } => {}
            // Boolar storage writes produce the mandated dummy zero bit.
            BIrStmt::StorageWrite { .. } => {
                self.linear_result(LinearCombination::constant(0), wire);
            }
            _ => return Err(ModeBRelationError::UnsupportedStatement { wire }),
        }
        Ok(())
    }
}

/// Run the canonical statement scheduler over a circuit and return the
/// semantics result. This is the shared schedule for actual values and later
/// VOLE verifier-state correlation semantics.
pub fn schedule_boolar_constraints<P, S>(
    circuit: &BCircuit<P>,
    mut semantics: S,
) -> Result<S, ModeBRelationError>
where
    P: Clone,
    S: ConstraintSemantics,
{
    for (statement_index, node) in circuit.stmts.iter().enumerate() {
        let wire = circuit.params as usize + statement_index;
        semantics.statement(wire, &node.kind)?;
    }
    Ok(semantics)
}

/// Versioned field configuration for the initial prime-field RAM format.
///
/// Record keys are encoded in the KoalaBear quintic extension, rather than a
/// single base-field element. This leaves room for a complete 32-bit address
/// plus bounded storage, lane, time, and operation fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeFieldRamConfig {
    pub base_modulus: u32,
    pub extension_degree: usize,
    pub storage_bits: usize,
    pub lane_bits: usize,
    pub address_bits: usize,
    pub time_bits: usize,
}
impl Default for PrimeFieldRamConfig {
    fn default() -> Self {
        Self {
            base_modulus: KOALABEAR_MODULUS,
            extension_degree: KOALABEAR_QUINTIC_DEGREE,
            storage_bits: PRIME_RAM_STORAGE_BITS,
            lane_bits: PRIME_RAM_LANE_BITS,
            address_bits: PRIME_RAM_ADDRESS_BITS,
            time_bits: PRIME_RAM_TIME_BITS,
        }
    }
}
impl PrimeFieldRamConfig {
    /// Validate the fixed initial format, including its 32-bit address claim.
    pub fn validate(&self) -> bool {
        self.base_modulus == KOALABEAR_MODULUS
            && self.extension_degree == KOALABEAR_QUINTIC_DEGREE
            && self.storage_bits == PRIME_RAM_STORAGE_BITS
            && self.lane_bits == PRIME_RAM_LANE_BITS
            && self.address_bits == PRIME_RAM_ADDRESS_BITS
            && self.time_bits == PRIME_RAM_TIME_BITS
    }
    /// Canonical bytes for binding the field/RAM ABI into the transcript.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PRIME_RAM_FORMAT);
        put_u32(&mut out, self.base_modulus);
        put_len(&mut out, self.extension_degree);
        put_len(&mut out, self.storage_bits);
        put_len(&mut out, self.lane_bits);
        put_len(&mut out, self.address_bits);
        put_len(&mut out, self.time_bits);
        out
    }
}

/// Storage relation metadata. The actual RAM argument is backend-selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageRelation {
    /// Number of canonical execution records, including pre-initialization.
    pub access_count: usize,
    /// Maximum address width in bits.
    pub max_address_bits: usize,
    /// Storage/lane domains occurring in the circuit.
    pub domains: Vec<(StorageId, LaneId)>,
}

/// Read/write operation in the backend-neutral RAM witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RamAccessKind {
    Read,
    Write,
}

/// A canonical RAM access record. Addresses remain bit vectors so a target
/// field cannot silently truncate an address during lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamAccess {
    pub storage: StorageId,
    pub lane: LaneId,
    pub address: Vec<bool>,
    pub time: usize,
    pub kind: RamAccessKind,
    pub value: bool,
}

/// Full witness for the RAM relation. A succinct backend replaces the explicit
/// permutation check with its selected, versioned RAM argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamWitness {
    pub execution: Vec<RamAccess>,
    pub address_sorted: Vec<RamAccess>,
}

/// A RAM record represented in the canonical KoalaBear-quintic polynomial
/// basis. Coefficients are always canonical base-field representatives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamRecord {
    /// `(K0, K1, K2, K3, K4)` for the packed record key.
    pub key: [u32; KOALABEAR_QUINTIC_DEGREE],
    /// The separately constrained Boolean value column.
    pub value: bool,
}

/// One materialized row of the sorted-table latest-value scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamScanRow {
    /// Canonical packed sorted-table record.
    pub record: PrimeRamRecord,
    /// Whether this row has the same `(storage, lane, address)` as its predecessor.
    pub same_cell: bool,
    /// `(!same_cell) & read`.
    pub first_read: bool,
    /// `same_cell & read`.
    pub later_read: bool,
    /// `(!same_cell) & write`.
    pub first_write: bool,
    /// `same_cell & write`.
    pub later_write: bool,
    /// Latest value before this row (zero for a new cell).
    pub prior_latest: bool,
    /// Latest value after this row.
    pub latest: bool,
}

/// Concrete prime-RAM witness layout emitted before backend-specific field
/// encoding. It materializes permutation inputs and every selector/value in
/// the sorted latest-value scan; it is not itself a Spartan-WHIR proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamMaterialization {
    /// Execution table after quintic key packing.
    pub execution: Vec<PrimeRamRecord>,
    /// Address/time sorted table plus explicit scan auxiliaries.
    pub sorted: Vec<PrimeRamScanRow>,
}

/// Variable locations for one bounded RAM record in the prime-field R1CS
/// layout. Bit arrays are little-endian; `key` is the quintic polynomial-basis
/// representation documented by `koalabear-ext5-ram-v1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamRecordLayout {
    pub storage: [usize; PRIME_RAM_STORAGE_BITS],
    pub lane: [usize; PRIME_RAM_LANE_BITS],
    pub address: [usize; PRIME_RAM_ADDRESS_BITS],
    pub time: [usize; PRIME_RAM_TIME_BITS],
    pub kind: usize,
    pub value: usize,
    pub key: [usize; KOALABEAR_QUINTIC_DEGREE],
}

/// Variable locations for one explicit sorted-table scan row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamScanLayout {
    pub record: PrimeRamRecordLayout,
    /// Per-bit cell equality flags, ordered storage/lane/address MSB first.
    pub cell_bit_equal: Vec<usize>,
    /// Prefix equality flags; entry zero is the fixed one prefix.
    pub cell_prefix_equal: Vec<usize>,
    /// One-hot first-differing-bit selectors for strict cell ordering.
    pub cell_first_difference: Vec<usize>,
    /// Time equality/prefix/first-difference gadget, gated by `same_cell`.
    pub time_bit_equal: Vec<usize>,
    pub time_prefix_equal: Vec<usize>,
    pub time_first_difference: Vec<usize>,
    pub same_cell: usize,
    pub first_read: usize,
    pub later_read: usize,
    pub first_write: usize,
    pub later_write: usize,
    pub prior_latest: usize,
    pub latest: usize,
}

/// Frozen variable layout and explicit static R1CS rows for the non-challenge
/// RAM constraints. The grand-product rows are emitted separately with public
/// challenge slots by [`PrimeRamR1cs::permutation_rows`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamR1cs {
    pub variable_count: usize,
    pub rows: Vec<R1csRow>,
    pub execution: Vec<PrimeRamRecordLayout>,
    pub sorted: Vec<PrimeRamScanLayout>,
}

/// Fiat--Shamir challenges represented in the canonical quintic basis. They
/// are transcript outputs, never prover-selected witness values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamPermutationChallenges {
    pub gamma: [i64; KOALABEAR_QUINTIC_DEGREE],
    pub eta: [i64; KOALABEAR_QUINTIC_DEGREE],
}

/// Challenge-slot variables and rows for the extension-field permutation.
/// The same static R1CS shape works for every transcript-derived challenge
/// value; challenge coefficients are public variables, not baked constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamPermutationR1cs {
    pub variable_count: usize,
    pub rows: Vec<R1csRow>,
    /// Public `gamma` challenge coefficients in the quintic basis.
    pub gamma: [usize; KOALABEAR_QUINTIC_DEGREE],
    /// Public `eta` challenge coefficients in the quintic basis.
    pub eta: [usize; KOALABEAR_QUINTIC_DEGREE],
    /// Per-record `gamma + K + eta * value` execution-table factors.
    pub compressed_execution: Vec<[usize; KOALABEAR_QUINTIC_DEGREE]>,
    /// Per-record `gamma + K + eta * value` sorted-table factors.
    pub compressed_sorted: Vec<[usize; KOALABEAR_QUINTIC_DEGREE]>,
    /// Extension inverses proving every sorted-table factor is nonzero.
    pub sorted_denominator_inverse: Vec<[usize; KOALABEAR_QUINTIC_DEGREE]>,
    /// `Z_0 .. Z_n`, each in the quintic polynomial basis.
    pub z: Vec<[usize; KOALABEAR_QUINTIC_DEGREE]>,
}

/// One frozen, backend-ready variable layout for the complete supported
/// Boolar relation. RAM variables begin at zero, followed by the primary
/// Boolar wires and gate helpers, followed by permutation-product variables.
/// All rows use this one coordinate system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnifiedR1cs {
    /// Structural circuit binding for the exported relation.
    pub circuit_id: CircuitId,
    /// Total number of variables addressed by the rows.
    pub variable_count: usize,
    /// Static Boolar, RAM, and challenge-bound permutation constraints.
    pub rows: Vec<R1csRow>,
    /// Public bindings rewritten into this unified coordinate system.
    pub public_bindings: Vec<PublicBinding>,
    /// First unified slot assigned to Boolar primary wires/helpers.
    pub primary_offset: usize,
    /// RAM layout, when the circuit has storage.
    pub ram: Option<PrimeRamR1cs>,
    /// Public-challenge-slot RAM permutation layout, when storage is present.
    pub permutation: Option<PrimeRamPermutationR1cs>,
}

/// A canonical sparse linear combination over the KoalaBear base field.
/// Constants and coefficients are representatives in `0..KOALABEAR_MODULUS`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KoalaBearLinearCombination {
    /// Canonical base-field constant coefficient.
    pub constant: u32,
    /// Strictly increasing variable indices with nonzero canonical coefficients.
    pub terms: Vec<(usize, u32)>,
}

/// One KoalaBear-base-field R1CS equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KoalaBearR1csRow {
    /// Left sparse linear combination.
    pub a: KoalaBearLinearCombination,
    /// Right sparse linear combination.
    pub b: KoalaBearLinearCombination,
    /// Result sparse linear combination.
    pub c: KoalaBearLinearCombination,
}

/// A complete assignment in the frozen [`UnifiedR1cs`] variable order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnifiedR1csWitness {
    /// Canonical KoalaBear values for every unified variable.
    pub values: Vec<u32>,
}

/// The unified relation lowered to canonical KoalaBear base-field matrix
/// coefficients. Extension arithmetic has already been expanded into these
/// base-field rows; this type performs no extension-field encoding itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KoalaBearR1cs {
    /// Structural circuit binding inherited from [`UnifiedR1cs`].
    pub circuit_id: CircuitId,
    /// The field modulus, fixed to [`KOALABEAR_MODULUS`].
    pub modulus: u32,
    /// Number of R1CS variables.
    pub variable_count: usize,
    /// Canonically lowered sparse rows.
    pub rows: Vec<KoalaBearR1csRow>,
    /// Unified public bindings.
    pub public_bindings: Vec<PublicBinding>,
}

/// Sparse matrix entry in the column layout consumed by `spartan-whir`:
/// `[ private witness | constant one | public inputs ]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpartanWhirMatrixEntry {
    /// Zero-based constraint row.
    pub row: usize,
    /// Zero-based matrix column.
    pub column: usize,
    /// Canonical KoalaBear coefficient.
    pub value: u32,
}

/// A dependency-free Spartan-WHIR R1CS-shape export. A std adapter can turn
/// this directly into `spartan_whir::{R1csShape, SparseMatrix}` without
/// changing ordering or coefficient representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanWhirR1csShape {
    /// Structural relation binding.
    pub circuit_id: CircuitId,
    /// Number of constraints.
    pub constraint_count: usize,
    /// Number of non-public witness columns.
    pub witness_count: usize,
    /// Number of externally supplied public inputs.
    pub public_input_count: usize,
    /// Unified variables in exact public-input order: outputs then inputs.
    pub public_wires: Vec<usize>,
    /// Sparse A matrix.
    pub a: Vec<SpartanWhirMatrixEntry>,
    /// Sparse B matrix.
    pub b: Vec<SpartanWhirMatrixEntry>,
    /// Sparse C matrix.
    pub c: Vec<SpartanWhirMatrixEntry>,
}

impl KoalaBearR1cs {
    /// Re-index the unified layout into Spartan-WHIR's `[W | 1 | X]` columns.
    /// Public values are ordered `claimed outputs || inputs || RAM challenge
    /// slots`; the first two groups follow the KoalaBear Circom convention,
    /// and challenge slots are the transcript-derived tail for storage RAM.
    pub fn export_spartan_whir_shape(&self) -> Result<SpartanWhirR1csShape, ModeBRelationError> {
        let mut public_wires = Vec::with_capacity(self.public_bindings.len());
        for output in self
            .public_bindings
            .iter()
            .filter_map(|binding| match *binding {
                PublicBinding::Output { wire, .. } => Some(wire),
                PublicBinding::Input { .. } | PublicBinding::RamPermutationChallenge { .. } => None,
            })
        {
            push_public_wire(&mut public_wires, output)?;
        }
        for input in
            self.public_bindings
                .iter()
                .filter_map(|binding| match *binding {
                    PublicBinding::Input { wire, .. } => Some(wire),
                    PublicBinding::Output { .. }
                    | PublicBinding::RamPermutationChallenge { .. } => None,
                })
        {
            push_public_wire(&mut public_wires, input)?;
        }
        for challenge in self
            .public_bindings
            .iter()
            .filter_map(|binding| match *binding {
                PublicBinding::RamPermutationChallenge { wire, .. } => Some(wire),
                PublicBinding::Input { .. } | PublicBinding::Output { .. } => None,
            })
        {
            push_public_wire(&mut public_wires, challenge)?;
        }
        let mut private_columns = vec![None; self.variable_count];
        let mut next_private = 0;
        for (wire, slot) in private_columns.iter_mut().enumerate() {
            if !public_wires.contains(&wire) {
                *slot = Some(next_private);
                next_private += 1;
            }
        }
        let public_input_count = public_wires.len();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for (row, constraint) in self.rows.iter().enumerate() {
            lower_spartan_lc(
                &constraint.a,
                row,
                next_private,
                &private_columns,
                &public_wires,
                &mut a,
            );
            lower_spartan_lc(
                &constraint.b,
                row,
                next_private,
                &private_columns,
                &public_wires,
                &mut b,
            );
            lower_spartan_lc(
                &constraint.c,
                row,
                next_private,
                &private_columns,
                &public_wires,
                &mut c,
            );
        }
        Ok(SpartanWhirR1csShape {
            circuit_id: self.circuit_id,
            constraint_count: self.rows.len(),
            witness_count: next_private,
            public_input_count,
            public_wires,
            a,
            b,
            c,
        })
    }
}

fn push_public_wire(public_wires: &mut Vec<usize>, wire: usize) -> Result<(), ModeBRelationError> {
    if public_wires.contains(&wire) {
        return Err(ModeBRelationError::DuplicatePublicWire { wire });
    }
    public_wires.push(wire);
    Ok(())
}

fn lower_spartan_lc(
    lc: &KoalaBearLinearCombination,
    row: usize,
    witness_count: usize,
    private_columns: &[Option<usize>],
    public_wires: &[usize],
    output: &mut Vec<SpartanWhirMatrixEntry>,
) {
    if lc.constant != 0 {
        output.push(SpartanWhirMatrixEntry {
            row,
            column: witness_count,
            value: lc.constant,
        });
    }
    for (wire, value) in &lc.terms {
        let column = private_columns[*wire].unwrap_or_else(|| {
            witness_count
                + 1
                + public_wires
                    .iter()
                    .position(|candidate| candidate == wire)
                    .expect("public wire")
        });
        output.push(SpartanWhirMatrixEntry {
            row,
            column,
            value: *value,
        });
    }
}

impl UnifiedR1cs {
    /// Check that a complete assignment satisfies every lowered base-field row.
    pub fn evaluate_koalabear_witness(
        &self,
        witness: &UnifiedR1csWitness,
    ) -> Result<(), ModeBRelationError> {
        if witness.values.len() != self.variable_count {
            return Err(ModeBRelationError::WrongUnifiedWitnessCount {
                expected: self.variable_count,
                found: witness.values.len(),
            });
        }
        let lowered = self.lower_koalabear()?;
        for (row, constraint) in lowered.rows.iter().enumerate() {
            if kb_lc_value(&constraint.a, &witness.values)
                * kb_lc_value(&constraint.b, &witness.values)
                % u64::from(KOALABEAR_MODULUS)
                != kb_lc_value(&constraint.c, &witness.values)
            {
                return Err(ModeBRelationError::UnsatisfiedUnifiedRow { row });
            }
        }
        Ok(())
    }

    /// Reduce signed relation coefficients modulo KoalaBear and canonicalize
    /// each sparse vector for a Spartan-WHIR matrix exporter.
    pub fn lower_koalabear(&self) -> Result<KoalaBearR1cs, ModeBRelationError> {
        Ok(KoalaBearR1cs {
            circuit_id: self.circuit_id,
            modulus: KOALABEAR_MODULUS,
            variable_count: self.variable_count,
            rows: self
                .rows
                .iter()
                .map(|row| {
                    Ok(KoalaBearR1csRow {
                        a: lower_linear_combination(&row.a, self.variable_count)?,
                        b: lower_linear_combination(&row.b, self.variable_count)?,
                        c: lower_linear_combination(&row.c, self.variable_count)?,
                    })
                })
                .collect::<Result<Vec<_>, ModeBRelationError>>()?,
            public_bindings: self.public_bindings.clone(),
        })
    }
}

impl PrimeRamR1cs {
    /// Emit the fixed-variable-order rows for `access_count` RAM records.
    ///
    /// The table contents, sort witness, and scan values are prover witness
    /// variables. The permutation layout adds public Fiat--Shamir challenge
    /// slots; a proving transcript assigns them after committing these columns.
    pub fn new(access_count: usize) -> Self {
        let mut next = 0;
        let mut rows = Vec::new();
        let mut execution = Vec::with_capacity(access_count);
        let mut sorted: Vec<PrimeRamScanLayout> = Vec::with_capacity(access_count);
        for time in 0..access_count {
            let record = allocate_record(&mut next, &mut rows);
            // Execution rows are canonical IR order, so time is not a prover
            // choice. The permutation transfers these unique times to Q.
            for (bit, variable) in record.time.iter().enumerate() {
                rows.push(equal_constant_row(*variable, ((time >> bit) & 1) as i64));
            }
            execution.push(record);
        }
        for index in 0..access_count {
            let record = allocate_record(&mut next, &mut rows);
            let same_cell = allocate_boolean(&mut next, &mut rows);
            let (
                cell_bit_equal,
                cell_prefix_equal,
                cell_first_difference,
                time_bit_equal,
                time_prefix_equal,
                time_first_difference,
            ) = if index == 0 {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            } else {
                let cell = allocate_cell_order_gadget(
                    &mut next,
                    &mut rows,
                    &sorted[index - 1].record,
                    &record,
                    same_cell,
                );
                let time = allocate_same_cell_time_order_gadget(
                    &mut next,
                    &mut rows,
                    &sorted[index - 1].record,
                    &record,
                    same_cell,
                );
                (cell.0, cell.1, cell.2, time.0, time.1, time.2)
            };
            let first_read = allocate_boolean(&mut next, &mut rows);
            let later_read = allocate_boolean(&mut next, &mut rows);
            let first_write = allocate_boolean(&mut next, &mut rows);
            let later_write = allocate_boolean(&mut next, &mut rows);
            let prior_latest = allocate_boolean(&mut next, &mut rows);
            let latest = allocate_boolean(&mut next, &mut rows);
            // Exact one-hot selector definitions. `kind=0` means read.
            rows.push(product_row(
                lc_one_minus(same_cell),
                lc_one_minus(record.kind),
                first_read,
            ));
            rows.push(product_row(
                LinearCombination::var(same_cell),
                lc_one_minus(record.kind),
                later_read,
            ));
            rows.push(product_row(
                lc_one_minus(same_cell),
                LinearCombination::var(record.kind),
                first_write,
            ));
            rows.push(product_row(
                LinearCombination::var(same_cell),
                LinearCombination::var(record.kind),
                later_write,
            ));
            rows.push(equal_one_row(&[
                first_read,
                later_read,
                first_write,
                later_write,
            ]));
            // The latest-value scan, including the zero default for first reads.
            rows.push(product_row(
                LinearCombination::var(first_read),
                LinearCombination::var(record.value),
                next,
            ));
            next += 1; // zero-constrained temporary for first-read value
            let zero = next - 1;
            rows.push(zero_row(zero));
            rows.push(product_row(
                LinearCombination::var(later_read),
                lc_difference(record.value, prior_latest),
                next,
            ));
            next += 1;
            rows.push(zero_row(next - 1));
            rows.push(product_row(
                lc_sum(&[first_read, later_read]),
                lc_difference(latest, record.value),
                next,
            ));
            next += 1;
            rows.push(zero_row(next - 1));
            rows.push(product_row(
                lc_sum(&[first_write, later_write]),
                lc_difference(latest, record.value),
                next,
            ));
            next += 1;
            rows.push(zero_row(next - 1));
            if index == 0 {
                rows.push(equal_constant_row(prior_latest, 0));
                rows.push(equal_constant_row(same_cell, 0));
            } else {
                rows.push(equal_variables_row(prior_latest, sorted[index - 1].latest));
            }
            sorted.push(PrimeRamScanLayout {
                record,
                cell_bit_equal,
                cell_prefix_equal,
                cell_first_difference,
                time_bit_equal,
                time_prefix_equal,
                time_first_difference,
                same_cell,
                first_read,
                later_read,
                first_write,
                later_write,
                prior_latest,
                latest,
            });
        }
        Self {
            variable_count: next,
            rows,
            execution,
            sorted,
        }
    }

    /// Build the RAM rows and bind each execution-table record to the actual
    /// Boolar circuit statement. `primary_wire_offset` is the first Boolar
    /// wire in the enclosing R1CS variable ordering; it prevents a backend
    /// from proving a detached RAM table.
    pub fn for_boolar<P: Clone>(
        circuit: &BCircuit<P>,
        primary_wire_offset: usize,
    ) -> Result<Self, ModeBRelationError> {
        let storage =
            storage_relation(circuit).expect("caller only uses RAM layout for storage circuits");
        validate_prime_ram_bounds(circuit, &storage, &PrimeFieldRamConfig::default())?;
        let mut layout = Self::new(storage.access_count);
        if primary_wire_offset < layout.variable_count {
            return Err(ModeBRelationError::RamVariableLayoutOverlap);
        }
        let mut index = 0;
        for segment in &circuit.pre_init {
            for (offset, value) in segment.data.iter().copied().enumerate() {
                bind_execution_record_constants(
                    &mut layout.rows,
                    &layout.execution[index],
                    segment.storage,
                    segment.lane,
                    &increment_address(&segment.addr, offset),
                    index,
                    RamAccessKind::Write,
                    value,
                );
                index += 1;
            }
        }
        for (statement, node) in circuit.stmts.iter().enumerate() {
            let wire = circuit.params as usize + statement;
            match &node.kind {
                BIrStmt::StorageRead {
                    storage,
                    lane,
                    addr,
                } => {
                    bind_execution_record_wires(
                        &mut layout.rows,
                        &layout.execution[index],
                        *storage,
                        *lane,
                        addr,
                        index,
                        RamAccessKind::Read,
                        primary_wire_offset,
                        wire,
                    )?;
                    index += 1;
                }
                BIrStmt::StorageWrite {
                    storage,
                    lane,
                    addr,
                    src,
                } => {
                    let source = value_wire(0, wire, *src)?;
                    bind_execution_record_wires(
                        &mut layout.rows,
                        &layout.execution[index],
                        *storage,
                        *lane,
                        addr,
                        index,
                        RamAccessKind::Write,
                        primary_wire_offset,
                        source,
                    )?;
                    index += 1;
                }
                _ => {}
            }
        }
        debug_assert_eq!(index, layout.execution.len());
        layout.variable_count = layout
            .variable_count
            .max(primary_wire_offset + circuit.params as usize + circuit.stmts.len());
        Ok(layout)
    }

    /// Append explicit quintic-extension grand-product rows with public
    /// challenge slots. This realizes
    /// `Z[i+1] * (gamma + R(Q[i])) = Z[i] * (gamma + R(E[i]))`, with fixed
    /// `Z[0] = Z[n] = 1`, without baking challenge values into the R1CS shape.
    /// A sound proving lifecycle must assign the slots only after committing
    /// the relevant RAM columns.
    pub fn permutation_rows(&self) -> PrimeRamPermutationR1cs {
        let mut next = self.variable_count;
        let mut rows = Vec::new();
        let gamma = core::array::from_fn(|_| {
            let variable = next;
            next += 1;
            variable
        });
        let eta = core::array::from_fn(|_| {
            let variable = next;
            next += 1;
            variable
        });
        let compressed_execution = self
            .execution
            .iter()
            .map(|record| allocate_compressed_record(&mut next, &mut rows, record, &gamma, &eta))
            .collect::<Vec<_>>();
        let compressed_sorted = self
            .sorted
            .iter()
            .map(|scan| {
                allocate_compressed_record(&mut next, &mut rows, &scan.record, &gamma, &eta)
            })
            .collect::<Vec<_>>();
        let sorted_denominator_inverse = compressed_sorted
            .iter()
            .map(|denominator| {
                let inverse = core::array::from_fn(|_| {
                    let variable = next;
                    next += 1;
                    variable
                });
                let product = extension_product_rows(&mut next, &mut rows, *denominator, inverse);
                constrain_extension_constant(&mut rows, product, [1, 0, 0, 0, 0]);
                inverse
            })
            .collect::<Vec<_>>();
        let mut z = Vec::with_capacity(self.execution.len() + 1);
        for _ in 0..=self.execution.len() {
            z.push(core::array::from_fn(|_| {
                let variable = next;
                next += 1;
                variable
            }));
        }
        constrain_extension_constant(&mut rows, z[0], [1, 0, 0, 0, 0]);
        constrain_extension_constant(&mut rows, *z.last().expect("nonempty Z"), [1, 0, 0, 0, 0]);
        for index in 0..self.execution.len() {
            let left = extension_product_rows(
                &mut next,
                &mut rows,
                z[index + 1],
                compressed_sorted[index],
            );
            let right =
                extension_product_rows(&mut next, &mut rows, z[index], compressed_execution[index]);
            for coordinate in 0..KOALABEAR_QUINTIC_DEGREE {
                rows.push(equal_variables_row(left[coordinate], right[coordinate]));
            }
        }
        PrimeRamPermutationR1cs {
            variable_count: next,
            rows,
            gamma,
            eta,
            compressed_execution,
            compressed_sorted,
            sorted_denominator_inverse,
            z,
        }
    }
}

fn bind_execution_record_constants(
    rows: &mut Vec<R1csRow>,
    record: &PrimeRamRecordLayout,
    storage: StorageId,
    lane: LaneId,
    address: &[bool],
    time: usize,
    kind: RamAccessKind,
    value: bool,
) {
    bind_constant_bits(rows, &record.storage, storage.0 as u64);
    bind_constant_bits(rows, &record.lane, lane.0 as u64);
    bind_address_constants(rows, &record.address, address);
    bind_constant_bits(rows, &record.time, time as u64);
    rows.push(equal_constant_row(
        record.kind,
        i64::from(kind == RamAccessKind::Write),
    ));
    rows.push(equal_constant_row(record.value, i64::from(value)));
}

fn bind_execution_record_wires(
    rows: &mut Vec<R1csRow>,
    record: &PrimeRamRecordLayout,
    storage: StorageId,
    lane: LaneId,
    address: &[IRVarId],
    time: usize,
    kind: RamAccessKind,
    primary_wire_offset: usize,
    value: usize,
) -> Result<(), ModeBRelationError> {
    bind_constant_bits(rows, &record.storage, storage.0 as u64);
    bind_constant_bits(rows, &record.lane, lane.0 as u64);
    for (destination, source) in record.address.iter().zip(address) {
        rows.push(equal_variables_row(
            *destination,
            value_wire(primary_wire_offset, usize::MAX, *source)?,
        ));
    }
    for destination in record.address.iter().skip(address.len()) {
        rows.push(equal_constant_row(*destination, 0));
    }
    bind_constant_bits(rows, &record.time, time as u64);
    rows.push(equal_constant_row(
        record.kind,
        i64::from(kind == RamAccessKind::Write),
    ));
    rows.push(equal_variables_row(
        record.value,
        primary_wire_offset + value,
    ));
    Ok(())
}

fn bind_constant_bits(rows: &mut Vec<R1csRow>, variables: &[usize], value: u64) {
    for (bit, variable) in variables.iter().enumerate() {
        rows.push(equal_constant_row(*variable, ((value >> bit) & 1) as i64));
    }
}

fn bind_address_constants(rows: &mut Vec<R1csRow>, variables: &[usize], address: &[bool]) {
    for (bit, variable) in variables.iter().enumerate() {
        rows.push(equal_constant_row(
            *variable,
            i64::from(address.get(bit).copied().unwrap_or(false)),
        ));
    }
}

fn value_wire(offset: usize, current: usize, value: IRVarId) -> Result<usize, ModeBRelationError> {
    let value = value.0 as usize;
    if value >= current {
        return Err(ModeBRelationError::InvalidWireReference {
            wire: current,
            referenced: value as u32,
        });
    }
    Ok(offset + value)
}

fn cell_bits_msb_first(record: &PrimeRamRecordLayout) -> Vec<usize> {
    let mut bits =
        Vec::with_capacity(PRIME_RAM_STORAGE_BITS + PRIME_RAM_LANE_BITS + PRIME_RAM_ADDRESS_BITS);
    bits.extend(record.storage.iter().rev().copied());
    bits.extend(record.lane.iter().rev().copied());
    bits.extend(record.address.iter().rev().copied());
    bits
}

/// Enforce equality bits, their AND-prefix, and strict lexicographic ordering
/// when `same_cell` is zero. For the first differing bit, predecessor=0 and
/// current=1; if all bits match, `same_cell=1`.
fn allocate_cell_order_gadget(
    next: &mut usize,
    rows: &mut Vec<R1csRow>,
    predecessor: &PrimeRamRecordLayout,
    current: &PrimeRamRecordLayout,
    same_cell: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let previous_bits = cell_bits_msb_first(predecessor);
    let current_bits = cell_bits_msb_first(current);
    let mut equal = Vec::with_capacity(previous_bits.len());
    let mut prefix = Vec::with_capacity(previous_bits.len() + 1);
    let mut first_difference = Vec::with_capacity(previous_bits.len());
    let one = *next;
    *next += 1;
    rows.push(equal_constant_row(one, 1));
    prefix.push(one);
    for (previous, current) in previous_bits.into_iter().zip(current_bits) {
        let product = *next;
        *next += 1;
        rows.push(product_row(
            LinearCombination::var(previous),
            LinearCombination::var(current),
            product,
        ));
        let equality = allocate_boolean(next, rows);
        // equality = 1 - previous - current + 2*previous*current.
        rows.push(R1csRow {
            a: LinearCombination::constant(0),
            b: LinearCombination::constant(1),
            c: LinearCombination {
                constant: -1,
                terms: vec![(previous, 1), (current, 1), (product, -2), (equality, 1)],
            },
        });
        equal.push(equality);
        let next_prefix = allocate_boolean(next, rows);
        rows.push(product_row(
            LinearCombination::var(*prefix.last().expect("prefix seed")),
            LinearCombination::var(equality),
            next_prefix,
        ));
        let first = allocate_boolean(next, rows);
        rows.push(product_row(
            LinearCombination::var(prefix[prefix.len() - 1]),
            lc_one_minus(equality),
            first,
        ));
        // A first difference is permitted only as 0 -> 1.
        let zero = *next;
        *next += 1;
        rows.push(product_row(
            LinearCombination::var(first),
            lc_one_minus(current),
            zero,
        ));
        rows.push(zero_row(zero));
        prefix.push(next_prefix);
        first_difference.push(first);
    }
    rows.push(equal_variables_row(
        same_cell,
        *prefix.last().expect("nonempty prefix"),
    ));
    let mut selector_sum = lc_sum(&first_difference);
    selector_sum = selector_sum.add(same_cell, 1);
    rows.push(R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: LinearCombination {
            constant: -1,
            terms: selector_sum.terms,
        },
    });
    (equal, prefix, first_difference)
}

/// Require increasing time only when the adjacent records are for one cell.
fn allocate_same_cell_time_order_gadget(
    next: &mut usize,
    rows: &mut Vec<R1csRow>,
    predecessor: &PrimeRamRecordLayout,
    current: &PrimeRamRecordLayout,
    same_cell: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut equal = Vec::with_capacity(PRIME_RAM_TIME_BITS);
    let mut prefix = vec![same_cell];
    let mut first_difference = Vec::with_capacity(PRIME_RAM_TIME_BITS);
    for (&previous, &current) in predecessor.time.iter().rev().zip(current.time.iter().rev()) {
        let product = *next;
        *next += 1;
        rows.push(product_row(
            LinearCombination::var(previous),
            LinearCombination::var(current),
            product,
        ));
        let equality = allocate_boolean(next, rows);
        rows.push(R1csRow {
            a: LinearCombination::constant(0),
            b: LinearCombination::constant(1),
            c: LinearCombination {
                constant: -1,
                terms: vec![(previous, 1), (current, 1), (product, -2), (equality, 1)],
            },
        });
        equal.push(equality);
        let next_prefix = allocate_boolean(next, rows);
        rows.push(product_row(
            LinearCombination::var(*prefix.last().expect("prefix seed")),
            LinearCombination::var(equality),
            next_prefix,
        ));
        let first = allocate_boolean(next, rows);
        rows.push(product_row(
            LinearCombination::var(*prefix.last().expect("prefix seed")),
            lc_one_minus(equality),
            first,
        ));
        let zero = *next;
        *next += 1;
        rows.push(product_row(
            LinearCombination::var(first),
            lc_one_minus(current),
            zero,
        ));
        rows.push(zero_row(zero));
        prefix.push(next_prefix);
        first_difference.push(first);
    }
    // A same-cell successor must have exactly one increasing first time bit.
    rows.push(R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: lc_sum(&first_difference).add(same_cell, -1),
    });
    (equal, prefix, first_difference)
}

/// Allocate `gamma + K + eta * value` for one RAM record. The challenge
/// coefficients are public variables, while `eta_value` is a private auxiliary
/// constrained as `eta_coordinate * value`.
fn allocate_compressed_record(
    next: &mut usize,
    rows: &mut Vec<R1csRow>,
    record: &PrimeRamRecordLayout,
    gamma: &[usize; KOALABEAR_QUINTIC_DEGREE],
    eta: &[usize; KOALABEAR_QUINTIC_DEGREE],
) -> [usize; KOALABEAR_QUINTIC_DEGREE] {
    core::array::from_fn(|coordinate| {
        let eta_value = *next;
        *next += 1;
        rows.push(product_row(
            LinearCombination::var(eta[coordinate]),
            LinearCombination::var(record.value),
            eta_value,
        ));
        let compressed = *next;
        *next += 1;
        rows.push(R1csRow {
            a: LinearCombination::constant(0),
            b: LinearCombination::constant(1),
            c: LinearCombination::var(compressed)
                .add(gamma[coordinate], -1)
                .add(record.key[coordinate], -1)
                .add(eta_value, -1),
        });
        compressed
    })
}

fn constrain_extension_constant(
    rows: &mut Vec<R1csRow>,
    value: [usize; KOALABEAR_QUINTIC_DEGREE],
    constant: [i64; KOALABEAR_QUINTIC_DEGREE],
) {
    for coordinate in 0..KOALABEAR_QUINTIC_DEGREE {
        rows.push(equal_constant_row(value[coordinate], constant[coordinate]));
    }
}

/// Multiply two quintic-basis values. Each base-field product is its own R1CS
/// row; the final coordinate rows apply `X^5 = 1 - X^2`.
fn extension_product_rows(
    next: &mut usize,
    rows: &mut Vec<R1csRow>,
    left: [usize; KOALABEAR_QUINTIC_DEGREE],
    right: [usize; KOALABEAR_QUINTIC_DEGREE],
) -> [usize; KOALABEAR_QUINTIC_DEGREE] {
    let mut products = [[0usize; KOALABEAR_QUINTIC_DEGREE]; KOALABEAR_QUINTIC_DEGREE];
    for i in 0..KOALABEAR_QUINTIC_DEGREE {
        for j in 0..KOALABEAR_QUINTIC_DEGREE {
            products[i][j] = *next;
            *next += 1;
            rows.push(product_row(
                LinearCombination::var(left[i]),
                LinearCombination::var(right[j]),
                products[i][j],
            ));
        }
    }
    let output = core::array::from_fn(|_| {
        let variable = *next;
        *next += 1;
        variable
    });
    for coordinate in 0..KOALABEAR_QUINTIC_DEGREE {
        let mut c = LinearCombination::var(output[coordinate]);
        for i in 0..KOALABEAR_QUINTIC_DEGREE {
            for j in 0..KOALABEAR_QUINTIC_DEGREE {
                let coefficient = extension_monomial_reduction(i + j)[coordinate];
                if coefficient != 0 {
                    c = c.add(products[i][j], -coefficient);
                }
            }
        }
        rows.push(R1csRow {
            a: LinearCombination::constant(0),
            b: LinearCombination::constant(1),
            c,
        });
    }
    output
}

fn extension_monomial_reduction(degree: usize) -> [i64; KOALABEAR_QUINTIC_DEGREE] {
    let mut polynomial = vec![0_i64; (degree + 1).max(KOALABEAR_QUINTIC_DEGREE)];
    polynomial[degree] = 1;
    for current in (KOALABEAR_QUINTIC_DEGREE..=degree).rev() {
        let coefficient = polynomial[current];
        if coefficient != 0 {
            // x^current = x^(current-5) - x^(current-3).
            polynomial[current - KOALABEAR_QUINTIC_DEGREE] += coefficient;
            polynomial[current - 3] -= coefficient;
        }
    }
    core::array::from_fn(|index| polynomial[index])
}

fn allocate_boolean(next: &mut usize, rows: &mut Vec<R1csRow>) -> usize {
    let variable = *next;
    *next += 1;
    rows.push(R1csRow {
        a: LinearCombination::var(variable),
        b: lc_one_minus(variable),
        c: LinearCombination::constant(0),
    });
    variable
}
fn allocate_record(next: &mut usize, rows: &mut Vec<R1csRow>) -> PrimeRamRecordLayout {
    let storage = core::array::from_fn(|_| allocate_boolean(next, rows));
    let lane = core::array::from_fn(|_| allocate_boolean(next, rows));
    let address = core::array::from_fn(|_| allocate_boolean(next, rows));
    let time = core::array::from_fn(|_| allocate_boolean(next, rows));
    let kind = allocate_boolean(next, rows);
    let value = allocate_boolean(next, rows);
    let key = core::array::from_fn(|_| {
        let v = *next;
        *next += 1;
        v
    });
    for (coordinate, terms) in prime_key_terms(&storage, &lane, &address, &time, kind)
        .into_iter()
        .enumerate()
    {
        rows.push(R1csRow {
            a: LinearCombination::constant(0),
            b: LinearCombination::constant(1),
            c: LinearCombination::var(key[coordinate])
                .add_terms(terms.into_iter().map(|(v, c)| (v, -c)).collect()),
        });
    }
    PrimeRamRecordLayout {
        storage,
        lane,
        address,
        time,
        kind,
        value,
        key,
    }
}
fn assign_unified_value(
    values: &mut [Option<u32>],
    slot: usize,
    value: u32,
) -> Result<(), ModeBRelationError> {
    let value = value % KOALABEAR_MODULUS;
    match values.get_mut(slot) {
        Some(existing @ None) => {
            *existing = Some(value);
            Ok(())
        }
        Some(Some(previous)) if *previous == value => Ok(()),
        Some(_) => Err(ModeBRelationError::InconsistentUnifiedWitness { variable: slot }),
        None => Err(ModeBRelationError::R1csVariableOutOfBounds {
            variable: slot,
            variable_count: values.len(),
        }),
    }
}

fn assign_ram_record(
    values: &mut [Option<u32>],
    layout: &PrimeRamRecordLayout,
    access: &RamAccess,
    record: &PrimeRamRecord,
) -> Result<(), ModeBRelationError> {
    for (i, slot) in layout.storage.iter().enumerate() {
        assign_unified_value(values, *slot, (access.storage.0 >> i) & 1)?;
    }
    for (i, slot) in layout.lane.iter().enumerate() {
        assign_unified_value(values, *slot, (access.lane.0 >> i) & 1)?;
    }
    for (i, slot) in layout.address.iter().enumerate() {
        assign_unified_value(values, *slot, u32::from(access.address[i]))?;
    }
    for (i, slot) in layout.time.iter().enumerate() {
        assign_unified_value(values, *slot, ((access.time >> i) & 1) as u32)?;
    }
    assign_unified_value(
        values,
        layout.kind,
        u32::from(access.kind == RamAccessKind::Write),
    )?;
    assign_unified_value(values, layout.value, u32::from(record.value))?;
    for (slot, value) in layout.key.iter().zip(record.key) {
        assign_unified_value(values, *slot, value)?;
    }
    Ok(())
}

fn assign_sort_gadget_values(
    values: &mut [Option<u32>],
    layouts: &[PrimeRamScanLayout],
    accesses: &[RamAccess],
) -> Result<(), ModeBRelationError> {
    for index in 1..layouts.len() {
        let previous = &accesses[index - 1];
        let current = &accesses[index];
        let layout = &layouts[index];
        let cell_previous = cell_bits_msb_first(&layouts[index - 1].record);
        let cell_current = cell_bits_msb_first(&layout.record);
        let mut prefix = true;
        for (((previous_slot, current_slot), equal_slot), (prefix_slot, first_slot)) in
            cell_previous
                .iter()
                .zip(cell_current)
                .zip(&layout.cell_bit_equal)
                .zip(
                    layout
                        .cell_prefix_equal
                        .iter()
                        .skip(1)
                        .zip(&layout.cell_first_difference),
                )
        {
            let previous_bit = values[*previous_slot] == Some(1);
            let current_bit = values[current_slot] == Some(1);
            let equal = previous_bit == current_bit;
            assign_unified_value(values, *equal_slot, u32::from(equal))?;
            assign_unified_value(values, *prefix_slot, u32::from(prefix && equal))?;
            assign_unified_value(values, *first_slot, u32::from(prefix && !equal))?;
            prefix &= equal;
        }
        if let Some(seed) = layout.cell_prefix_equal.first() {
            assign_unified_value(values, *seed, 1)?;
        }
        let mut time_prefix = layout.same_cell != usize::MAX
            && previous.storage == current.storage
            && previous.lane == current.lane
            && previous.address == current.address;
        for (((previous_slot, current_slot), equal_slot), (prefix_slot, first_slot)) in layouts
            [index - 1]
            .record
            .time
            .iter()
            .rev()
            .zip(layout.record.time.iter().rev())
            .zip(&layout.time_bit_equal)
            .zip(
                layout
                    .time_prefix_equal
                    .iter()
                    .skip(1)
                    .zip(&layout.time_first_difference),
            )
        {
            let equal = values[*previous_slot] == values[*current_slot];
            assign_unified_value(values, *equal_slot, u32::from(equal))?;
            assign_unified_value(values, *prefix_slot, u32::from(time_prefix && equal))?;
            assign_unified_value(values, *first_slot, u32::from(time_prefix && !equal))?;
            time_prefix &= equal;
        }
    }
    Ok(())
}

fn kb_lc_value(lc: &KoalaBearLinearCombination, values: &[u32]) -> u64 {
    lc.terms
        .iter()
        .fold(u64::from(lc.constant), |sum, (slot, coefficient)| {
            (sum + u64::from(*coefficient) * u64::from(values[*slot]))
                % u64::from(KOALABEAR_MODULUS)
        })
}

fn complete_unified_assignment(
    rows: &KoalaBearR1cs,
    values: &mut [Option<u32>],
) -> Result<(), ModeBRelationError> {
    let modulus = i64::from(KOALABEAR_MODULUS);
    loop {
        let mut progress = false;
        for row in &rows.rows {
            let a = known_kb_lc(&row.a, values, modulus);
            let b = known_kb_lc(&row.b, values, modulus);
            let c = known_kb_lc(&row.c, values, modulus);
            if let (Some(a), Some(b), None) = (a, b, c) {
                if let Some((slot, coefficient)) = single_unknown(&row.c, values) {
                    let known = known_kb_lc_without(&row.c, values, modulus);
                    let target = (a * b - known).rem_euclid(modulus);
                    assign_unified_value(
                        values,
                        slot,
                        (target * kb_inverse(coefficient, modulus)).rem_euclid(modulus) as u32,
                    )?;
                    progress = true;
                }
            }
        }
        if !progress {
            break;
        }
    }
    Ok(())
}

fn known_kb_lc(
    lc: &KoalaBearLinearCombination,
    values: &[Option<u32>],
    modulus: i64,
) -> Option<i64> {
    if lc.terms.iter().any(|(slot, _)| values[*slot].is_none()) {
        None
    } else {
        Some(known_kb_lc_without(lc, values, modulus))
    }
}
fn known_kb_lc_without(
    lc: &KoalaBearLinearCombination,
    values: &[Option<u32>],
    modulus: i64,
) -> i64 {
    lc.terms.iter().fold(
        i64::from(lc.constant),
        |sum, (slot, coefficient)| match values[*slot] {
            Some(value) => (sum + i64::from(*coefficient) * i64::from(value)) % modulus,
            None => sum,
        },
    )
}
fn single_unknown(lc: &KoalaBearLinearCombination, values: &[Option<u32>]) -> Option<(usize, i64)> {
    let mut unknown = lc.terms.iter().filter(|(slot, _)| values[*slot].is_none());
    let (slot, coefficient) = unknown.next()?;
    unknown
        .next()
        .is_none()
        .then_some((*slot, i64::from(*coefficient)))
}
fn kb_inverse(value: i64, modulus: i64) -> i64 {
    let (mut a, mut b, mut x0, mut x1) = (value.rem_euclid(modulus), modulus, 1_i64, 0_i64);
    while b != 0 {
        let q = a / b;
        (a, b, x0, x1) = (b, a - q * b, x1, x0 - q * x1);
    }
    x0.rem_euclid(modulus)
}

fn ram_permutation_products(
    materialized: &PrimeRamMaterialization,
    challenges: &PrimeRamPermutationChallenges,
) -> Result<Vec<[u32; KOALABEAR_QUINTIC_DEGREE]>, ModeBRelationError> {
    let mut z = vec![[0; KOALABEAR_QUINTIC_DEGREE]; materialized.execution.len() + 1];
    z[0][0] = 1;
    for index in 0..materialized.execution.len() {
        let execution = compressed_record_value(&materialized.execution[index], challenges);
        let sorted = compressed_record_value(&materialized.sorted[index].record, challenges);
        z[index + 1] = ext_mul(ext_mul(z[index], execution), ext_inverse(sorted)?);
    }
    if z.last() != Some(&[1, 0, 0, 0, 0]) {
        return Err(ModeBRelationError::RamPermutationEndpointMismatch);
    }
    Ok(z)
}
fn compressed_record_value(
    record: &PrimeRamRecord,
    challenges: &PrimeRamPermutationChallenges,
) -> [u32; KOALABEAR_QUINTIC_DEGREE] {
    core::array::from_fn(|i| {
        (challenges.gamma[i].rem_euclid(i64::from(KOALABEAR_MODULUS)) as u32
            + record.key[i]
            + if record.value {
                challenges.eta[i].rem_euclid(i64::from(KOALABEAR_MODULUS)) as u32
            } else {
                0
            })
            % KOALABEAR_MODULUS
    })
}
fn ext_mul(
    left: [u32; KOALABEAR_QUINTIC_DEGREE],
    right: [u32; KOALABEAR_QUINTIC_DEGREE],
) -> [u32; KOALABEAR_QUINTIC_DEGREE] {
    let mut out = [0_u64; KOALABEAR_QUINTIC_DEGREE];
    for i in 0..KOALABEAR_QUINTIC_DEGREE {
        for j in 0..KOALABEAR_QUINTIC_DEGREE {
            for (k, coefficient) in extension_monomial_reduction(i + j).iter().enumerate() {
                out[k] = (out[k]
                    + (i64::from(*coefficient) * i64::from(left[i]) * i64::from(right[j]))
                        .rem_euclid(i64::from(KOALABEAR_MODULUS)) as u64)
                    % u64::from(KOALABEAR_MODULUS);
            }
        }
    }
    out.map(|v| v as u32)
}
fn ext_inverse(
    value: [u32; KOALABEAR_QUINTIC_DEGREE],
) -> Result<[u32; KOALABEAR_QUINTIC_DEGREE], ModeBRelationError> {
    let mut matrix = [[0_i64; 6]; KOALABEAR_QUINTIC_DEGREE];
    for column in 0..KOALABEAR_QUINTIC_DEGREE {
        let basis = core::array::from_fn(|i| u32::from(i == column));
        let product = ext_mul(value, basis);
        for row in 0..KOALABEAR_QUINTIC_DEGREE {
            matrix[row][column] = i64::from(product[row]);
        }
    }
    for row in 0..KOALABEAR_QUINTIC_DEGREE {
        matrix[row][KOALABEAR_QUINTIC_DEGREE] = i64::from(row == 0);
    }
    let modulus = i64::from(KOALABEAR_MODULUS);
    for pivot in 0..KOALABEAR_QUINTIC_DEGREE {
        let swap = (pivot..KOALABEAR_QUINTIC_DEGREE)
            .find(|row| matrix[*row][pivot] != 0)
            .ok_or(ModeBRelationError::RamPermutationZeroDenominator)?;
        matrix.swap(pivot, swap);
        let inverse = kb_inverse(matrix[pivot][pivot], modulus);
        for entry in &mut matrix[pivot] {
            *entry = (*entry * inverse).rem_euclid(modulus);
        }
        for row in 0..KOALABEAR_QUINTIC_DEGREE {
            if row != pivot {
                let factor = matrix[row][pivot];
                for col in pivot..=KOALABEAR_QUINTIC_DEGREE {
                    matrix[row][col] =
                        (matrix[row][col] - factor * matrix[pivot][col]).rem_euclid(modulus);
                }
            }
        }
    }
    Ok(core::array::from_fn(|row| {
        matrix[row][KOALABEAR_QUINTIC_DEGREE] as u32
    }))
}

fn lower_linear_combination(
    value: &LinearCombination,
    variable_count: usize,
) -> Result<KoalaBearLinearCombination, ModeBRelationError> {
    let modulus = i64::from(KOALABEAR_MODULUS);
    let mut terms: Vec<(usize, i64)> = value
        .terms
        .iter()
        .map(|(variable, coefficient)| {
            if *variable >= variable_count {
                return Err(ModeBRelationError::R1csVariableOutOfBounds {
                    variable: *variable,
                    variable_count,
                });
            }
            Ok((*variable, coefficient.rem_euclid(modulus)))
        })
        .collect::<Result<_, _>>()?;
    terms.sort_unstable_by_key(|(variable, _)| *variable);
    let mut canonical: Vec<(usize, i64)> = Vec::with_capacity(terms.len());
    for (variable, coefficient) in terms {
        if let Some((previous, accumulated)) = canonical.last_mut()
            && *previous == variable
        {
            *accumulated = (*accumulated + coefficient).rem_euclid(modulus);
        } else {
            canonical.push((variable, coefficient));
        }
    }
    canonical.retain(|(_, coefficient)| *coefficient != 0);
    Ok(KoalaBearLinearCombination {
        constant: value.constant.rem_euclid(modulus) as u32,
        terms: canonical
            .into_iter()
            .map(|(variable, coefficient)| (variable, coefficient as u32))
            .collect(),
    })
}

fn shift_row(row: &R1csRow, offset: usize) -> R1csRow {
    R1csRow {
        a: shift_linear_combination(&row.a, offset),
        b: shift_linear_combination(&row.b, offset),
        c: shift_linear_combination(&row.c, offset),
    }
}

fn shift_linear_combination(value: &LinearCombination, offset: usize) -> LinearCombination {
    LinearCombination {
        constant: value.constant,
        terms: value
            .terms
            .iter()
            .map(|(variable, coefficient)| (offset + variable, *coefficient))
            .collect(),
    }
}

fn lc_one_minus(variable: usize) -> LinearCombination {
    LinearCombination::constant(1).add(variable, -1)
}
fn lc_difference(left: usize, right: usize) -> LinearCombination {
    LinearCombination::var(left).add(right, -1)
}
fn lc_sum(values: &[usize]) -> LinearCombination {
    values
        .iter()
        .fold(LinearCombination::constant(0), |lc, v| lc.add(*v, 1))
}
fn product_row(a: LinearCombination, b: LinearCombination, c: usize) -> R1csRow {
    R1csRow {
        a,
        b,
        c: LinearCombination::var(c),
    }
}
fn zero_row(variable: usize) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: LinearCombination::var(variable),
    }
}
fn equal_constant_row(variable: usize, constant: i64) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: LinearCombination {
            constant: -constant,
            terms: vec![(variable, 1)],
        },
    }
}
fn equal_variables_row(left: usize, right: usize) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: lc_difference(left, right),
    }
}
fn equal_one_row(values: &[usize]) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(0),
        b: LinearCombination::constant(1),
        c: LinearCombination {
            constant: -1,
            terms: values.iter().map(|v| (*v, 1)).collect(),
        },
    }
}
fn prime_key_terms(
    storage: &[usize; PRIME_RAM_STORAGE_BITS],
    lane: &[usize; PRIME_RAM_LANE_BITS],
    address: &[usize; PRIME_RAM_ADDRESS_BITS],
    time: &[usize; PRIME_RAM_TIME_BITS],
    kind: usize,
) -> [Vec<(usize, i64)>; KOALABEAR_QUINTIC_DEGREE] {
    let mut terms: [Vec<(usize, i64)>; KOALABEAR_QUINTIC_DEGREE] =
        core::array::from_fn(|_| Vec::new());
    for i in 0..16 {
        terms[0].push((storage[i], 1_i64 << i));
    }
    for i in 0..14 {
        terms[0].push((lane[i], 1_i64 << (16 + i)));
    }
    for i in 14..16 {
        terms[1].push((lane[i], 1_i64 << (i - 14)));
    }
    for i in 0..28 {
        terms[1].push((address[i], 1_i64 << (2 + i)));
    }
    for i in 28..32 {
        terms[2].push((address[i], 1_i64 << (i - 28)));
    }
    for i in 0..26 {
        terms[2].push((time[i], 1_i64 << (4 + i)));
    }
    for i in 26..32 {
        terms[3].push((time[i], 1_i64 << (i - 26)));
    }
    terms[3].push((kind, 1_i64 << 6));
    terms
}

impl Ord for RamAccess {
    fn cmp(&self, other: &Self) -> Ordering {
        self.storage
            .cmp(&other.storage)
            .then_with(|| self.lane.cmp(&other.lane))
            .then_with(|| self.address.iter().rev().cmp(other.address.iter().rev()))
            .then_with(|| self.time.cmp(&other.time))
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.value.cmp(&other.value))
    }
}
impl PartialOrd for RamAccess {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Canonical public statement for a relation invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeBPublicInstance {
    pub version: u32,
    pub circuit_id: CircuitId,
    pub public_inputs: Vec<bool>,
    pub claimed_outputs: Vec<bool>,
}

/// The executable, backend-neutral relation descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeBRelation {
    /// Relation ABI version.
    pub version: u32,
    /// Structural circuit binding.
    pub circuit_id: CircuitId,
    /// Number of primary Boolar wires.
    pub wire_count: usize,
    /// Total witness variables, including non-wire gate helpers.
    pub witness_count: usize,
    /// R1CS rows, with signed coefficients.
    pub rows: Vec<R1csRow>,
    /// Public input/output bindings.
    pub public_bindings: Vec<PublicBinding>,
    /// Storage metadata, if storage is present.
    pub storage: Option<StorageRelation>,
    /// Initial KoalaBear-quintic RAM ABI (present when storage is present).
    pub prime_ram: Option<PrimeFieldRamConfig>,
}

/// Errors while constructing or evaluating the relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModeBRelationError {
    /// Oracle, RNG, and action ABIs are intentionally not accepted yet.
    UnsupportedStatement { wire: usize },
    /// A statement points at a non-prior wire.
    InvalidWireReference { wire: usize, referenced: u32 },
    /// The supplied Boolean witness has the wrong size.
    WrongWitnessCount { expected: usize, found: usize },
    /// A public vector has the wrong size.
    WrongPublicCount { expected: usize, found: usize },
    /// An R1CS row does not evaluate to zero.
    UnsatisfiedRow { row: usize },
    /// A public bit binding disagrees with the witness.
    PublicMismatch { binding: usize },
    /// A complete unified field witness has the wrong number of values.
    WrongUnifiedWitnessCount { expected: usize, found: usize },
    /// A complete unified field witness leaves an internal variable unset.
    IncompleteUnifiedWitness,
    /// Two materialization paths assigned different values to one variable.
    InconsistentUnifiedWitness { variable: usize },
    /// A lowered unified R1CS row is unsatisfied.
    UnsatisfiedUnifiedRow { row: usize },
    /// A permutation denominator was zero in the quintic extension.
    RamPermutationZeroDenominator,
    /// The grand product did not end at the required one element.
    RamPermutationEndpointMismatch,
    /// A RAM witness is absent for a circuit that accesses storage.
    MissingRamWitness,
    /// A RAM witness was supplied for a circuit without storage.
    UnexpectedRamWitness,
    /// The execution RAM table is not derived from the circuit and wires.
    RamExecutionMismatch,
    /// The address table is not the canonical permutation of execution.
    RamNotPermutation,
    /// A RAM read did not observe the prior write or zero default.
    InvalidRamRead { time: usize },
    /// The relation/circuit pair is not the one from which the relation arose.
    CircuitIdMismatch,
    /// A materialized prime-RAM row does not match the canonical RAM witness.
    RamMaterializationMismatch,
    /// RAM auxiliaries and primary Boolar wires were assigned overlapping slots.
    RamVariableLayoutOverlap,
    /// A storage-bearing unified export requires transcript-derived RAM challenges.
    MissingRamPermutationChallenges,
    /// A row referenced a variable outside its declared unified layout.
    R1csVariableOutOfBounds {
        /// Referenced variable.
        variable: usize,
        /// Declared variable count.
        variable_count: usize,
    },
    /// A public binding names one unified wire more than once.
    DuplicatePublicWire {
        /// Duplicated unified wire index.
        wire: usize,
    },
    /// A storage-free unified export must not receive RAM challenges.
    UnexpectedRamPermutationChallenges,
    /// A circuit exceeds the fixed KoalaBear-quintic RAM ABI bounds.
    RamBoundsExceeded {
        /// Bounded RAM field that exceeded its configured width or count.
        field: &'static str,
        /// Inclusive maximum allowed by the fixed ABI.
        maximum: u64,
        /// Value requested by the circuit.
        found: u64,
    },
}
impl fmt::Display for ModeBRelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStatement { wire } => {
                write!(f, "unsupported statement at wire {wire}")
            }
            Self::InvalidWireReference { wire, referenced } => {
                write!(f, "wire {wire} references invalid wire {referenced}")
            }
            Self::WrongWitnessCount { expected, found } => {
                write!(f, "witness has {found} values, expected {expected}")
            }
            Self::WrongPublicCount { expected, found } => {
                write!(f, "public vector has {found} values, expected {expected}")
            }
            Self::UnsatisfiedRow { row } => write!(f, "relation row {row} is unsatisfied"),
            Self::PublicMismatch { binding } => {
                write!(f, "public binding {binding} is unsatisfied")
            }
            Self::WrongUnifiedWitnessCount { expected, found } => {
                write!(f, "unified witness has {found} values, expected {expected}")
            }
            Self::IncompleteUnifiedWitness => {
                f.write_str("unified witness did not assign every variable")
            }
            Self::InconsistentUnifiedWitness { variable } => write!(
                f,
                "unified witness assigns variable {variable} inconsistently"
            ),
            Self::UnsatisfiedUnifiedRow { row } => {
                write!(f, "unified R1CS row {row} is unsatisfied")
            }
            Self::RamPermutationZeroDenominator => {
                f.write_str("RAM permutation denominator is zero")
            }
            Self::RamPermutationEndpointMismatch => {
                f.write_str("RAM permutation grand-product endpoint is not one")
            }
            Self::MissingRamWitness => f.write_str("RAM witness is required"),
            Self::UnexpectedRamWitness => {
                f.write_str("RAM witness supplied for a storage-free circuit")
            }
            Self::RamExecutionMismatch => {
                f.write_str("RAM execution table does not match circuit wires")
            }
            Self::RamNotPermutation => {
                f.write_str("RAM address table is not the canonical permutation")
            }
            Self::InvalidRamRead { time } => write!(f, "RAM read at time {time} is invalid"),
            Self::CircuitIdMismatch => f.write_str("circuit does not match relation circuit_id"),
            Self::RamMaterializationMismatch => {
                f.write_str("prime-RAM materialization does not match canonical RAM witness")
            }
            Self::RamVariableLayoutOverlap => {
                f.write_str("prime-RAM and primary-wire R1CS slots overlap")
            }
            Self::MissingRamPermutationChallenges => {
                f.write_str("storage-bearing unified export requires RAM permutation challenges")
            }
            Self::R1csVariableOutOfBounds {
                variable,
                variable_count,
            } => write!(
                f,
                "R1CS variable {variable} is outside declared layout of {variable_count} variables"
            ),
            Self::DuplicatePublicWire { wire } => {
                write!(f, "unified wire {wire} appears in multiple public bindings")
            }
            Self::UnexpectedRamPermutationChallenges => {
                f.write_str("storage-free unified export received RAM permutation challenges")
            }
            Self::RamBoundsExceeded {
                field,
                maximum,
                found,
            } => write!(
                f,
                "RAM {field} value {found} exceeds configured maximum {maximum}"
            ),
        }
    }
}
impl core::error::Error for ModeBRelationError {}

impl ModeBRelation {
    /// Lower the supported Boolar subset to field-independent R1CS rows.
    pub fn from_boolar<P: Clone>(circuit: &BCircuit<P>) -> Result<Self, ModeBRelationError> {
        let wires = circuit.params as usize + circuit.stmts.len();
        // Booleanity is explicit, so a prime-field backend cannot accept
        // non-Boolean intermediate values satisfying only gate equations.
        let semantics = schedule_boolar_constraints(circuit, ActualBooleanSemantics::new(wires))?;
        let rows = semantics.rows;
        let helpers = semantics.next_helper;
        let mut public_bindings: Vec<_> = (0..circuit.params as usize)
            .map(|index| PublicBinding::Input { index, wire: index })
            .collect();
        public_bindings.extend(circuit.outputs.iter().enumerate().map(|(index, v)| {
            PublicBinding::Output {
                index,
                wire: v.0 as usize,
            }
        }));
        let storage = storage_relation(circuit);
        if let Some(storage) = &storage {
            validate_prime_ram_bounds(circuit, storage, &PrimeFieldRamConfig::default())?;
        }
        Ok(Self {
            version: MODE_B_RELATION_VERSION,
            circuit_id: structural_circuit_id(circuit),
            wire_count: wires,
            witness_count: helpers,
            rows,
            public_bindings,
            prime_ram: storage.as_ref().map(|_| PrimeFieldRamConfig::default()),
            storage,
        })
    }

    /// Export all supported Boolar, RAM, and permutation constraints in one
    /// static variable coordinate system. For storage circuits, permutation
    /// challenges are public slots in the R1CS; this exporter deliberately
    /// does not accept or bake in challenge values. A proving transcript must
    /// assign those slots only after the required RAM-column commitment.
    pub fn export_unified_r1cs<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
    ) -> Result<UnifiedR1cs, ModeBRelationError> {
        if self.circuit_id != structural_circuit_id(circuit) {
            return Err(ModeBRelationError::CircuitIdMismatch);
        }

        let mut rows;
        let (primary_offset, ram, permutation) = if self.storage.is_some() {
            let static_ram =
                PrimeRamR1cs::new(self.storage.as_ref().expect("checked").access_count);
            let primary_offset = static_ram.variable_count;
            let mut ram = PrimeRamR1cs::for_boolar(circuit, primary_offset)?;
            // `for_boolar` knows the primary wire span; helpers are allocated
            // by the Boolar lowering and immediately follow that span.
            ram.variable_count = primary_offset + self.witness_count;
            let permutation = ram.permutation_rows();
            rows = ram.rows.clone();
            rows.extend(permutation.rows.iter().cloned());
            (primary_offset, Some(ram), Some(permutation))
        } else {
            rows = Vec::new();
            (0, None, None)
        };
        rows.extend(self.rows.iter().map(|row| shift_row(row, primary_offset)));
        let mut public_bindings = self
            .public_bindings
            .iter()
            .map(|binding| match *binding {
                PublicBinding::Input { index, wire } => PublicBinding::Input {
                    index,
                    wire: primary_offset + wire,
                },
                PublicBinding::Output { index, wire } => PublicBinding::Output {
                    index,
                    wire: primary_offset + wire,
                },
                PublicBinding::RamPermutationChallenge { .. } => {
                    unreachable!("Mode-B relations do not own unified challenge slots")
                }
            })
            .collect::<Vec<_>>();
        if let Some(permutation) = &permutation {
            public_bindings.extend(permutation.gamma.iter().enumerate().map(|(index, wire)| {
                PublicBinding::RamPermutationChallenge { index, wire: *wire }
            }));
            public_bindings.extend(permutation.eta.iter().enumerate().map(|(index, wire)| {
                PublicBinding::RamPermutationChallenge {
                    index: KOALABEAR_QUINTIC_DEGREE + index,
                    wire: *wire,
                }
            }));
        }
        let variable_count = permutation
            .as_ref()
            .map_or(primary_offset + self.witness_count, |layout| {
                layout.variable_count
            });
        Ok(UnifiedR1cs {
            circuit_id: self.circuit_id,
            variable_count,
            rows,
            public_bindings,
            primary_offset,
            ram,
            permutation,
        })
    }

    /// Materialize every variable in the unified KoalaBear field layout.
    ///
    /// This is a differential-oracle witness builder. For a RAM circuit its
    /// challenges must be transcript outputs from a commitment to the RAM
    /// columns; callers must not let a prover select them.
    pub fn materialize_unified_witness<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
        primary_witness: &[bool],
        public_inputs: &[bool],
        claimed_outputs: &[bool],
        ram_witness: Option<&RamWitness>,
        challenges: Option<&PrimeRamPermutationChallenges>,
    ) -> Result<UnifiedR1csWitness, ModeBRelationError> {
        self.evaluate_bool_with_ram(
            circuit,
            primary_witness,
            public_inputs,
            claimed_outputs,
            ram_witness,
        )?;
        match (&self.storage, challenges) {
            (Some(_), None) => return Err(ModeBRelationError::MissingRamPermutationChallenges),
            (None, Some(_)) => return Err(ModeBRelationError::UnexpectedRamPermutationChallenges),
            _ => {}
        }
        let unified = self.export_unified_r1cs(circuit)?;
        let mut values = vec![None; unified.variable_count];
        for (wire, value) in primary_witness.iter().copied().enumerate() {
            assign_unified_value(&mut values, unified.primary_offset + wire, u32::from(value))?;
        }
        if let (Some(layout), Some(ram)) = (&unified.ram, ram_witness) {
            let materialized = PrimeRamMaterialization::from_ram_witness(ram)?;
            for ((record_layout, access), record) in layout
                .execution
                .iter()
                .zip(&ram.execution)
                .zip(&materialized.execution)
            {
                assign_ram_record(&mut values, record_layout, access, record)?;
            }
            for ((scan_layout, access), scan) in layout
                .sorted
                .iter()
                .zip(&ram.address_sorted)
                .zip(&materialized.sorted)
            {
                assign_ram_record(&mut values, &scan_layout.record, access, &scan.record)?;
                assign_unified_value(
                    &mut values,
                    scan_layout.same_cell,
                    u32::from(scan.same_cell),
                )?;
                assign_unified_value(
                    &mut values,
                    scan_layout.first_read,
                    u32::from(scan.first_read),
                )?;
                assign_unified_value(
                    &mut values,
                    scan_layout.later_read,
                    u32::from(scan.later_read),
                )?;
                assign_unified_value(
                    &mut values,
                    scan_layout.first_write,
                    u32::from(scan.first_write),
                )?;
                assign_unified_value(
                    &mut values,
                    scan_layout.later_write,
                    u32::from(scan.later_write),
                )?;
                assign_unified_value(
                    &mut values,
                    scan_layout.prior_latest,
                    u32::from(scan.prior_latest),
                )?;
                assign_unified_value(&mut values, scan_layout.latest, u32::from(scan.latest))?;
            }
            assign_sort_gadget_values(&mut values, &layout.sorted, &ram.address_sorted)?;
            let permutation = unified
                .permutation
                .as_ref()
                .expect("RAM has permutation layout");
            let challenges = challenges.expect("RAM challenges checked");
            for (slot, value) in permutation.gamma.iter().zip(challenges.gamma) {
                assign_unified_value(
                    &mut values,
                    *slot,
                    value.rem_euclid(i64::from(KOALABEAR_MODULUS)) as u32,
                )?;
            }
            for (slot, value) in permutation.eta.iter().zip(challenges.eta) {
                assign_unified_value(
                    &mut values,
                    *slot,
                    value.rem_euclid(i64::from(KOALABEAR_MODULUS)) as u32,
                )?;
            }
            for (slots, record) in permutation
                .sorted_denominator_inverse
                .iter()
                .zip(&materialized.sorted)
            {
                let denominator = compressed_record_value(&record.record, challenges);
                let inverse = ext_inverse(denominator)?;
                for (slot, coordinate) in slots.iter().zip(inverse) {
                    assign_unified_value(&mut values, *slot, coordinate)?;
                }
            }
            let z = ram_permutation_products(&materialized, challenges)?;
            for (slots, value) in permutation.z.iter().zip(z) {
                for (slot, coordinate) in slots.iter().zip(value) {
                    assign_unified_value(&mut values, *slot, coordinate)?;
                }
            }
        }
        complete_unified_assignment(&unified.lower_koalabear()?, &mut values)?;
        let values = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(ModeBRelationError::IncompleteUnifiedWitness)?;
        let witness = UnifiedR1csWitness { values };
        unified.evaluate_koalabear_witness(&witness)?;
        Ok(witness)
    }

    /// Evaluate the descriptor over a Boolean primary witness and outputs.
    /// Helpers are derived deterministically from the circuit rows; this gives
    /// the future prime-field lowering an executable differential oracle.
    pub fn evaluate_bool(
        &self,
        witness: &[bool],
        public_inputs: &[bool],
        claimed_outputs: &[bool],
    ) -> Result<(), ModeBRelationError> {
        if witness.len() != self.wire_count {
            return Err(ModeBRelationError::WrongWitnessCount {
                expected: self.wire_count,
                found: witness.len(),
            });
        }
        let input_count = self
            .public_bindings
            .iter()
            .filter(|b| matches!(b, PublicBinding::Input { .. }))
            .count();
        if public_inputs.len() != input_count {
            return Err(ModeBRelationError::WrongPublicCount {
                expected: input_count,
                found: public_inputs.len(),
            });
        }
        let output_count = self
            .public_bindings
            .iter()
            .filter(|b| matches!(b, PublicBinding::Output { .. }))
            .count();
        if claimed_outputs.len() != output_count {
            return Err(ModeBRelationError::WrongPublicCount {
                expected: output_count,
                found: claimed_outputs.len(),
            });
        }
        // This evaluator checks all primary rows and derives helper products
        // from the row's single-variable product pattern.
        let mut values = witness.to_vec();
        values.resize(self.witness_count, false);
        for (i, row) in self.rows.iter().enumerate() {
            if row.c.terms.len() == 1
                && row.c.terms[0].1 == 1
                && row.c.terms[0].0 >= self.wire_count
                && row.a.terms.len() == 1
                && row.b.terms.len() == 1
            {
                values[row.c.terms[0].0] = values[row.a.terms[0].0] & values[row.b.terms[0].0];
            }
            if eval(&row.a, &values) * eval(&row.b, &values) != eval(&row.c, &values) {
                return Err(ModeBRelationError::UnsatisfiedRow { row: i });
            }
        }
        for binding in &self.public_bindings {
            match *binding {
                PublicBinding::Input { index, wire } if values[wire] != public_inputs[index] => {
                    return Err(ModeBRelationError::PublicMismatch { binding: index });
                }
                PublicBinding::Output { index, wire } if values[wire] != claimed_outputs[index] => {
                    return Err(ModeBRelationError::PublicMismatch {
                        binding: input_count + index,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Evaluate the complete relation, including the transparent RAM
    /// semantics. This is the differential oracle for a future RAM lowering.
    pub fn evaluate_bool_with_ram<P: Clone>(
        &self,
        circuit: &BCircuit<P>,
        witness: &[bool],
        public_inputs: &[bool],
        claimed_outputs: &[bool],
        ram: Option<&RamWitness>,
    ) -> Result<(), ModeBRelationError> {
        if self.circuit_id != structural_circuit_id(circuit) {
            return Err(ModeBRelationError::CircuitIdMismatch);
        }
        self.evaluate_bool(witness, public_inputs, claimed_outputs)?;
        let expected = RamWitness::from_boolar(circuit, witness)?;
        match (self.storage.is_some(), ram) {
            (false, None) => Ok(()),
            (false, Some(_)) => Err(ModeBRelationError::UnexpectedRamWitness),
            (true, None) => Err(ModeBRelationError::MissingRamWitness),
            (true, Some(actual)) if actual.execution != expected.execution => {
                Err(ModeBRelationError::RamExecutionMismatch)
            }
            (true, Some(actual)) => verify_ram(actual),
        }
    }

    /// Canonical, length-prefixed descriptor bytes for a proof-backend
    /// exporter. This format is relation ABI v1, not a Spartan-WHIR proof.
    /// Every Rust count is encoded as a `u64`; implementations must reject an
    /// input that cannot fit that representation before constructing a proof.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"cirrus-mode-b-relation-v1");
        put_u32(&mut out, self.version);
        out.extend_from_slice(&self.circuit_id);
        put_len(&mut out, self.wire_count);
        put_len(&mut out, self.witness_count);
        put_len(&mut out, self.rows.len());
        for row in &self.rows {
            put_lc(&mut out, &row.a);
            put_lc(&mut out, &row.b);
            put_lc(&mut out, &row.c);
        }
        put_len(&mut out, self.public_bindings.len());
        for binding in &self.public_bindings {
            match binding {
                PublicBinding::Input { index, wire } => {
                    out.push(0);
                    put_len(&mut out, *index);
                    put_len(&mut out, *wire);
                }
                PublicBinding::Output { index, wire } => {
                    out.push(1);
                    put_len(&mut out, *index);
                    put_len(&mut out, *wire);
                }
                PublicBinding::RamPermutationChallenge { index, wire } => {
                    out.push(2);
                    put_len(&mut out, *index);
                    put_len(&mut out, *wire);
                }
            }
        }
        match &self.storage {
            None => out.push(0),
            Some(storage) => {
                out.push(1);
                put_len(&mut out, storage.access_count);
                put_len(&mut out, storage.max_address_bits);
                put_len(&mut out, storage.domains.len());
                for (id, lane) in &storage.domains {
                    put_u32(&mut out, id.0);
                    put_u32(&mut out, lane.0);
                }
            }
        }
        match &self.prime_ram {
            None => out.push(0),
            Some(config) => {
                out.push(1);
                out.extend_from_slice(&config.canonical_bytes());
            }
        }
        out
    }
}

impl ModeBPublicInstance {
    /// Build the statement tied to `relation`.
    pub fn new(
        relation: &ModeBRelation,
        public_inputs: Vec<bool>,
        claimed_outputs: Vec<bool>,
    ) -> Self {
        Self {
            version: relation.version,
            circuit_id: relation.circuit_id,
            public_inputs,
            claimed_outputs,
        }
    }
    /// Canonical, length-prefixed statement bytes. A field adapter must map
    /// these components injectively and bind the raw circuit-id bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"cirrus-mode-b-instance-v1");
        put_u32(&mut out, self.version);
        out.extend_from_slice(&self.circuit_id);
        put_bits(&mut out, &self.public_inputs);
        put_bits(&mut out, &self.claimed_outputs);
        out
    }
}

impl PrimeRamMaterialization {
    /// Materialize the fixed KoalaBear-quintic RAM ABI from a canonical RAM
    /// witness. `ram` must already satisfy the execution/permutation/read
    /// reference relation; this method independently checks the scan values.
    pub fn from_ram_witness(ram: &RamWitness) -> Result<Self, ModeBRelationError> {
        verify_ram(ram)?;
        let execution = ram
            .execution
            .iter()
            .map(pack_prime_ram_record)
            .collect::<Result<Vec<_>, _>>()?;
        let mut sorted = Vec::with_capacity(ram.address_sorted.len());
        let mut previous: Option<&RamAccess> = None;
        let mut latest = false;
        for access in &ram.address_sorted {
            let same_cell = previous.is_some_and(|prior| same_ram_cell(prior, access));
            let prior_latest = if same_cell { latest } else { false };
            let read = access.kind == RamAccessKind::Read;
            let first_read = !same_cell && read;
            let later_read = same_cell && read;
            let first_write = !same_cell && !read;
            let later_write = same_cell && !read;
            latest = if read { prior_latest } else { access.value };
            sorted.push(PrimeRamScanRow {
                record: pack_prime_ram_record(access)?,
                same_cell,
                first_read,
                later_read,
                first_write,
                later_write,
                prior_latest,
                latest,
            });
            previous = Some(access);
        }
        let materialized = Self { execution, sorted };
        materialized.validate_against(ram)?;
        Ok(materialized)
    }

    /// Validate the materialized permutation keys, selectors, and latest-value
    /// scan against the canonical RAM tables.
    pub fn validate_against(&self, ram: &RamWitness) -> Result<(), ModeBRelationError> {
        if self.execution.len() != ram.execution.len()
            || self.sorted.len() != ram.address_sorted.len()
        {
            return Err(ModeBRelationError::RamMaterializationMismatch);
        }
        for (record, access) in self.execution.iter().zip(&ram.execution) {
            if record != &pack_prime_ram_record(access)? {
                return Err(ModeBRelationError::RamMaterializationMismatch);
            }
        }
        let mut prior: Option<&RamAccess> = None;
        for (index, (row, access)) in self.sorted.iter().zip(&ram.address_sorted).enumerate() {
            let same = prior.is_some_and(|p| same_ram_cell(p, access));
            let read = access.kind == RamAccessKind::Read;
            let prior_latest = if same {
                self.sorted[index - 1].latest
            } else {
                false
            };
            let expected_latest = if read { prior_latest } else { access.value };
            if row.record != pack_prime_ram_record(access)?
                || row.same_cell != same
                || row.first_read != (!same && read)
                || row.later_read != (same && read)
                || row.first_write != (!same && !read)
                || row.later_write != (same && !read)
                || row.prior_latest != prior_latest
                || row.latest != expected_latest
                || (read && access.value != prior_latest)
            {
                return Err(ModeBRelationError::RamMaterializationMismatch);
            }
            prior = Some(access);
        }
        Ok(())
    }
}

impl RamWitness {
    /// Derive the execution and canonical sorted table from a Boolar witness.
    pub fn from_boolar<P: Clone>(
        circuit: &BCircuit<P>,
        values: &[bool],
    ) -> Result<Self, ModeBRelationError> {
        let mut execution = Vec::new();
        let mut time = 0;
        for segment in &circuit.pre_init {
            for (offset, value) in segment.data.iter().copied().enumerate() {
                execution.push(RamAccess {
                    storage: segment.storage,
                    lane: segment.lane,
                    address: normalize_address(increment_address(&segment.addr, offset)),
                    time,
                    kind: RamAccessKind::Write,
                    value,
                });
                time += 1;
            }
        }
        for (s, node) in circuit.stmts.iter().enumerate() {
            let wire = circuit.params as usize + s;
            match &node.kind {
                BIrStmt::StorageRead {
                    storage,
                    lane,
                    addr,
                } => {
                    execution.push(RamAccess {
                        storage: *storage,
                        lane: *lane,
                        address: normalize_address(address_value(values, wire, addr)?),
                        time,
                        kind: RamAccessKind::Read,
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
                    execution.push(RamAccess {
                        storage: *storage,
                        lane: *lane,
                        address: normalize_address(address_value(values, wire, addr)?),
                        time,
                        kind: RamAccessKind::Write,
                        value: value_at(values, wire, *src)?,
                    });
                    time += 1;
                }
                _ => {}
            }
        }
        let mut address_sorted = execution.clone();
        address_sorted.sort();
        Ok(Self {
            execution,
            address_sorted,
        })
    }
}

fn eval(l: &LinearCombination, values: &[bool]) -> i64 {
    l.constant
        + l.terms
            .iter()
            .map(|(v, c)| if values[*v] { *c } else { 0 })
            .sum::<i64>()
}

fn pack_prime_ram_record(access: &RamAccess) -> Result<PrimeRamRecord, ModeBRelationError> {
    if access.address.len() != PRIME_RAM_ADDRESS_BITS || access.time > u32::MAX as usize {
        return Err(ModeBRelationError::RamMaterializationMismatch);
    }
    let address = bits_to_u32(&access.address);
    let kind = u32::from(access.kind == RamAccessKind::Write);
    let lane = access.lane.0;
    let key = [
        access.storage.0 | ((lane & 0x3fff) << 16),
        ((lane >> 14) & 0x3) | ((address & 0x0fffffff) << 2),
        ((address >> 28) & 0xf) | ((access.time as u32 & 0x03ff_ffff) << 4),
        ((access.time as u32 >> 26) & 0x3f) | (kind << 6),
        0,
    ];
    Ok(PrimeRamRecord {
        key,
        value: access.value,
    })
}

fn bits_to_u32(bits: &[bool]) -> u32 {
    bits.iter()
        .take(32)
        .enumerate()
        .fold(0, |value, (i, bit)| value | (u32::from(*bit) << i))
}

fn same_ram_cell(a: &RamAccess, b: &RamAccess) -> bool {
    a.storage == b.storage && a.lane == b.lane && a.address == b.address
}

fn verify_ram(ram: &RamWitness) -> Result<(), ModeBRelationError> {
    let mut canonical = ram.execution.clone();
    canonical.sort();
    if canonical != ram.address_sorted {
        return Err(ModeBRelationError::RamNotPermutation);
    }
    let mut prior: Option<&RamAccess> = None;
    for access in &ram.address_sorted {
        let same = prior.is_some_and(|p| {
            p.storage == access.storage && p.lane == access.lane && p.address == access.address
        });
        let expected = if same {
            prior.expect("same cell has a predecessor").value
        } else {
            false
        };
        if access.kind == RamAccessKind::Read && access.value != expected {
            return Err(ModeBRelationError::InvalidRamRead { time: access.time });
        }
        prior = Some(access);
    }
    Ok(())
}

fn address_value(
    values: &[bool],
    current: usize,
    address: &[IRVarId],
) -> Result<Vec<bool>, ModeBRelationError> {
    address
        .iter()
        .map(|v| value_at(values, current, *v))
        .collect()
}
fn value_at(values: &[bool], current: usize, value: IRVarId) -> Result<bool, ModeBRelationError> {
    let i = value.0 as usize;
    if i >= current {
        Err(ModeBRelationError::InvalidWireReference {
            wire: current,
            referenced: value.0,
        })
    } else {
        Ok(values[i])
    }
}
fn normalize_address(mut address: Vec<bool>) -> Vec<bool> {
    address.resize(PRIME_RAM_ADDRESS_BITS, false);
    address
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
    result
}

fn validate_prime_ram_bounds<P: Clone>(
    circuit: &BCircuit<P>,
    storage: &StorageRelation,
    config: &PrimeFieldRamConfig,
) -> Result<(), ModeBRelationError> {
    debug_assert!(config.validate());
    let max_storage = (1_u64 << config.storage_bits) - 1;
    let max_lane = (1_u64 << config.lane_bits) - 1;
    let max_accesses = 1_u64 << config.time_bits;
    for (storage_id, lane_id) in &storage.domains {
        if u64::from(storage_id.0) > max_storage {
            return Err(ModeBRelationError::RamBoundsExceeded {
                field: "storage ID",
                maximum: max_storage,
                found: u64::from(storage_id.0),
            });
        }
        if u64::from(lane_id.0) > max_lane {
            return Err(ModeBRelationError::RamBoundsExceeded {
                field: "lane ID",
                maximum: max_lane,
                found: u64::from(lane_id.0),
            });
        }
    }
    if storage.max_address_bits > config.address_bits {
        return Err(ModeBRelationError::RamBoundsExceeded {
            field: "address width",
            maximum: config.address_bits as u64,
            found: storage.max_address_bits as u64,
        });
    }
    if (storage.access_count as u64) > max_accesses {
        return Err(ModeBRelationError::RamBoundsExceeded {
            field: "access count",
            maximum: max_accesses,
            found: storage.access_count as u64,
        });
    }
    // `storage_relation` is computed from exactly these circuit fields. Keep
    // this argument to make the validation boundary explicit and future-proof.
    let _ = circuit;
    Ok(())
}

fn storage_relation<P: Clone>(circuit: &BCircuit<P>) -> Option<StorageRelation> {
    let mut access_count = 0;
    let mut max_address_bits = 0;
    let mut domains = Vec::new();
    let mut observe = |storage: StorageId, lane: LaneId, width: usize| {
        access_count += 1;
        max_address_bits = max_address_bits.max(width);
        if !domains.contains(&(storage, lane)) {
            domains.push((storage, lane));
        }
    };
    for segment in &circuit.pre_init {
        for _ in &segment.data {
            observe(segment.storage, segment.lane, segment.addr.len());
        }
    }
    for node in &circuit.stmts {
        match &node.kind {
            BIrStmt::StorageRead {
                storage,
                lane,
                addr,
            } => observe(*storage, *lane, addr.len()),
            BIrStmt::StorageWrite {
                storage,
                lane,
                addr,
                ..
            } => observe(*storage, *lane, addr.len()),
            _ => {}
        }
    }
    (!domains.is_empty()).then_some(StorageRelation {
        access_count,
        max_address_bits,
        domains,
    })
}

fn structural_circuit_id<P: Clone>(circuit: &BCircuit<P>) -> CircuitId {
    let mut h = Sha3_256::new();
    h.update(b"cirrus-mode-b-circuit-v1");
    h.update(circuit.params.to_le_bytes());
    h.update((circuit.pre_init.len() as u64).to_le_bytes());
    for segment in &circuit.pre_init {
        h.update(segment.storage.0.to_le_bytes());
        h.update(segment.lane.0.to_le_bytes());
        hash_bits(&mut h, &segment.addr);
        hash_bits(&mut h, &segment.data);
    }
    h.update((circuit.stmts.len() as u64).to_le_bytes());
    for node in &circuit.stmts {
        hash_statement(&mut h, &node.kind);
    }
    h.update((circuit.outputs.len() as u64).to_le_bytes());
    for output in &circuit.outputs {
        h.update(output.0.to_le_bytes());
    }
    h.finalize().into()
}
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_len(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_le_bytes());
}
fn put_bits(out: &mut Vec<u8>, bits: &[bool]) {
    put_len(out, bits.len());
    for bit in bits {
        out.push(u8::from(*bit));
    }
}
fn put_lc(out: &mut Vec<u8>, lc: &LinearCombination) {
    out.extend_from_slice(&lc.constant.to_le_bytes());
    put_len(out, lc.terms.len());
    for (v, c) in &lc.terms {
        put_len(out, *v);
        out.extend_from_slice(&c.to_le_bytes());
    }
}

fn hash_u32s(h: &mut Sha3_256, values: &[IRVarId]) {
    h.update((values.len() as u64).to_le_bytes());
    for value in values {
        h.update(value.0.to_le_bytes());
    }
}
fn hash_bits(h: &mut Sha3_256, values: &[bool]) {
    h.update((values.len() as u64).to_le_bytes());
    for value in values {
        h.update([u8::from(*value)]);
    }
}
fn hash_statement(h: &mut Sha3_256, s: &BIrStmt<IRVarId, StorageId>) {
    match s {
        BIrStmt::Zero => {
            h.update([0]);
        }
        BIrStmt::One => {
            h.update([1]);
        }
        BIrStmt::And(a, b) => {
            h.update([2]);
            h.update(a.0.to_le_bytes());
            h.update(b.0.to_le_bytes());
        }
        BIrStmt::Or(a, b) => {
            h.update([3]);
            h.update(a.0.to_le_bytes());
            h.update(b.0.to_le_bytes());
        }
        BIrStmt::Xor(a, b) => {
            h.update([4]);
            h.update(a.0.to_le_bytes());
            h.update(b.0.to_le_bytes());
        }
        BIrStmt::Not(a) => {
            h.update([5]);
            h.update(a.0.to_le_bytes());
        }
        BIrStmt::StorageRead {
            storage,
            lane,
            addr,
        } => {
            h.update([6]);
            h.update(storage.0.to_le_bytes());
            h.update(lane.0.to_le_bytes());
            hash_u32s(h, addr);
        }
        BIrStmt::StorageWrite {
            storage,
            lane,
            src,
            addr,
        } => {
            h.update([7]);
            h.update(storage.0.to_le_bytes());
            h.update(lane.0.to_le_bytes());
            h.update(src.0.to_le_bytes());
            hash_u32s(h, addr);
        }
        // Unsupported variants never produce a relation, but their tag is
        // included so this routine cannot accidentally identify one as a
        // supported circuit during error reporting or future extension.
        _ => {
            h.update([255]);
        }
    }
}
