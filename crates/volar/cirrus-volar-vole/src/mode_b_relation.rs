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
pub const KOALABEAR_MODULUS: u32 = 2_013_265_921;
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
    fn constant(c: i64) -> Self {
        Self {
            constant: c,
            terms: Vec::new(),
        }
    }
    fn var(v: usize) -> Self {
        Self {
            constant: 0,
            terms: vec![(v, 1)],
        }
    }
    fn add(mut self, v: usize, c: i64) -> Self {
        self.terms.push((v, c));
        self
    }
    fn add_terms(mut self, terms: Vec<(usize, i64)>) -> Self {
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

/// A public bit binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicBinding {
    Input { index: usize, wire: usize },
    Output { index: usize, wire: usize },
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
/// RAM constraints. The grand-product rows are challenge-bound and must be
/// appended by the Spartan-WHIR adapter after transcript challenges exist.
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

/// Challenge-bound variables and rows for the extension-field permutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeRamPermutationR1cs {
    pub variable_count: usize,
    pub rows: Vec<R1csRow>,
    /// `Z_0 .. Z_n`, each in the quintic polynomial basis.
    pub z: Vec<[usize; KOALABEAR_QUINTIC_DEGREE]>,
}

impl PrimeRamR1cs {
    /// Emit the fixed-variable-order rows for `access_count` RAM records.
    ///
    /// The table contents, sort witness, and scan values are prover witness
    /// variables. The adapter adds Fiat--Shamir-dependent extension-field
    /// grand-product rows after committing these columns.
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

    /// Append explicit quintic-extension grand-product rows for transcript
    /// challenges. This realizes `Z[i+1] * (gamma + R(Q[i])) = Z[i] *
    /// (gamma + R(E[i]))`, with fixed `Z[0] = Z[n] = 1`.
    pub fn permutation_rows(
        &self,
        challenges: &PrimeRamPermutationChallenges,
    ) -> PrimeRamPermutationR1cs {
        let mut next = self.variable_count;
        let mut rows = Vec::new();
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
            let sorted = compressed_record_lcs(&self.sorted[index].record, challenges);
            let execution = compressed_record_lcs(&self.execution[index], challenges);
            let left = extension_product_rows(&mut next, &mut rows, z[index + 1], sorted);
            let right = extension_product_rows(&mut next, &mut rows, z[index], execution);
            for coordinate in 0..KOALABEAR_QUINTIC_DEGREE {
                rows.push(equal_variables_row(left[coordinate], right[coordinate]));
            }
        }
        PrimeRamPermutationR1cs {
            variable_count: next,
            rows,
            z,
        }
    }
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
            a: LinearCombination::constant(1),
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
        a: LinearCombination::constant(1),
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
            a: LinearCombination::constant(1),
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
        a: LinearCombination::constant(1),
        b: LinearCombination::constant(1),
        c: lc_sum(&first_difference).add(same_cell, -1),
    });
    (equal, prefix, first_difference)
}

fn compressed_record_lcs(
    record: &PrimeRamRecordLayout,
    challenges: &PrimeRamPermutationChallenges,
) -> [LinearCombination; KOALABEAR_QUINTIC_DEGREE] {
    core::array::from_fn(|coordinate| LinearCombination {
        constant: challenges.gamma[coordinate],
        terms: vec![
            (record.key[coordinate], 1),
            (record.value, challenges.eta[coordinate]),
        ],
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
    right: [LinearCombination; KOALABEAR_QUINTIC_DEGREE],
) -> [usize; KOALABEAR_QUINTIC_DEGREE] {
    let mut products = [[0usize; KOALABEAR_QUINTIC_DEGREE]; KOALABEAR_QUINTIC_DEGREE];
    for i in 0..KOALABEAR_QUINTIC_DEGREE {
        for j in 0..KOALABEAR_QUINTIC_DEGREE {
            products[i][j] = *next;
            *next += 1;
            rows.push(product_row(
                LinearCombination::var(left[i]),
                right[j].clone(),
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
            a: LinearCombination::constant(1),
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
            a: LinearCombination::constant(1),
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
        a: LinearCombination::constant(1),
        b: LinearCombination::constant(1),
        c: LinearCombination::var(variable),
    }
}
fn equal_constant_row(variable: usize, constant: i64) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(1),
        b: LinearCombination::constant(1),
        c: LinearCombination {
            constant: -constant,
            terms: vec![(variable, 1)],
        },
    }
}
fn equal_variables_row(left: usize, right: usize) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(1),
        b: LinearCombination::constant(1),
        c: lc_difference(left, right),
    }
}
fn equal_one_row(values: &[usize]) -> R1csRow {
    R1csRow {
        a: LinearCombination::constant(1),
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
        let mut rows = Vec::new();
        // Booleanity is explicit, so a prime-field backend cannot accept
        // non-Boolean intermediate values satisfying only gate equations.
        for w in 0..wires {
            rows.push(R1csRow {
                a: LinearCombination::var(w),
                b: LinearCombination::var(w).add(w, -1),
                c: LinearCombination::constant(0),
            });
        }
        let mut helpers = wires;
        for (s, node) in circuit.stmts.iter().enumerate() {
            let w = circuit.params as usize + s;
            let prior = |v: IRVarId| -> Result<usize, ModeBRelationError> {
                let i = v.0 as usize;
                if i >= w {
                    Err(ModeBRelationError::InvalidWireReference {
                        wire: w,
                        referenced: v.0,
                    })
                } else {
                    Ok(i)
                }
            };
            let (a, b, c) = match &node.kind {
                BIrStmt::Zero => (
                    LinearCombination::constant(0),
                    LinearCombination::constant(1),
                    LinearCombination::var(w),
                ),
                BIrStmt::One => (
                    LinearCombination::constant(1),
                    LinearCombination::constant(1),
                    LinearCombination::var(w),
                ),
                BIrStmt::And(x, y) => (
                    LinearCombination::var(prior(*x)?),
                    LinearCombination::var(prior(*y)?),
                    LinearCombination::var(w),
                ),
                BIrStmt::Not(x) => (
                    LinearCombination::constant(1).add(prior(*x)?, -1),
                    LinearCombination::constant(1),
                    LinearCombination::var(w),
                ),
                BIrStmt::Xor(x, y) => {
                    let h = helpers;
                    helpers += 1;
                    rows.push(R1csRow {
                        a: LinearCombination::var(prior(*x)?),
                        b: LinearCombination::var(prior(*y)?),
                        c: LinearCombination::var(h),
                    });
                    (
                        LinearCombination::var(prior(*x)?)
                            .add(prior(*y)?, 1)
                            .add(h, -2),
                        LinearCombination::constant(1),
                        LinearCombination::var(w),
                    )
                }
                BIrStmt::Or(x, y) => {
                    let h = helpers;
                    helpers += 1;
                    rows.push(R1csRow {
                        a: LinearCombination::var(prior(*x)?),
                        b: LinearCombination::var(prior(*y)?),
                        c: LinearCombination::var(h),
                    });
                    (
                        LinearCombination::var(prior(*x)?)
                            .add(prior(*y)?, 1)
                            .add(h, -1),
                        LinearCombination::constant(1),
                        LinearCombination::var(w),
                    )
                }
                // The read value is constrained by the RAM relation below.
                BIrStmt::StorageRead { .. } => continue,
                // Boolar storage writes produce the mandated dummy zero bit.
                BIrStmt::StorageWrite { .. } => (
                    LinearCombination::constant(0),
                    LinearCombination::constant(1),
                    LinearCombination::var(w),
                ),
                _ => return Err(ModeBRelationError::UnsupportedStatement { wire: w }),
            };
            rows.push(R1csRow { a, b, c });
        }
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
