//! Generic, capability-gated compact variant bytecode (`CRBV`).
//!
//! A `CRBV` payload carries one precomputed schedule for a pure Boolean
//! region of a base [`CompactProgram`](crate::CompactProgram) (`CRBC`),
//! produced outside this crate by the single scheduling authority (Volar's
//! BinFHE V2 weaver). This module only serializes, validates, and interprets
//! the schedule as data; it never discovers cones, reorders records, or
//! recalculates failure budgets. See
//! `crates/recompile/fhe-v2-variant-bytecode-integration-plan.md`.
//!
//! # Format
//!
//! All integers are canonical shortest unsigned LEB128 (`u32` range). The
//! layout is fixed-order and self-delimiting; a decoder rejects trailing
//! bytes, unknown opcodes, and noncanonical integers.
//!
//! ```text
//! magic "CRBV"
//! variant_kind:uLEB          (1 = BinFHE V2 bootstrap plan)
//! variant_version:uLEB       (1)
//! profile:uLEB               (0=Toy, 1=ToyNoisy, 2=Std128, 3=Custom)
//! source_digest[32]          SHA-256 binding to the base CRBC executable
//! plan_digest[32]            SHA-256 over every byte after this field
//! plan_hash:u64le            producer's diagnostic hash (NOT a binding)
//! k_max:uLEB                 maximum LUT arity
//! wire_imports:  count:uLEB, base-slot:uLEB..      (positional: arena id i)
//! cell_imports:  count:uLEB, binding:uLEB..        (positional: arena id i)
//! wire_exports:  count:uLEB, (value:uLEB, base-slot:uLEB)..
//! cell_exports:  count:uLEB, (value:uLEB, binding:uLEB)..
//! lut_section:    len:uLEB, bytes
//! layer_section:  len:uLEB, bytes
//! ```
//!
//! Import lists are plain slot/binding lists because input arena ids are
//! dense (`0..count`), so position is the arena id. Export lists are
//! `(value, destination)` pairs: a Boolean region may legitimately export
//! one arena value twice, while driving two different values into one
//! destination is a conflict, so duplicates are rejected per destination.
//!
//! # Arenas, records, and layers
//!
//! Three typed arenas: `Wire` (Boolean wires), `Rgsw` (circuit-bootstrap
//! results), and `Cell` (RLWE content cells). Value ids are implicit
//! append-only positions within each arena. The layer section is a sequence
//! of `LAYER_BEGIN record_count .. LAYER_END`; empty layers are rejected.
//! Operands must reference values produced in **earlier** layers (strict
//! layer independence), so the layer barrier stays meaningful for any later
//! parallel backend. `LUT` inputs are LSB-first and its input count must
//! equal the table arity; tables are packed LSB-first with zero padding.
//!
//! # Admission and execution
//!
//! 1. [`VariantProgram::validate`] performs all structural checks and
//!    resource-cap enforcement without allocating or executing.
//! 2. [`VariantProgram::check_source`] verifies the whole-entry binding
//!    against a validated base executable: source digest, v1 empty cell
//!    arenas, imports drawn from base input slots, and export destinations
//!    covering exactly the base output slots.
//! 3. [`VariantProgram::execute`] checks the operation set's declared
//!    kind/version, calls [`VariantOperationSet::admit`], then runs records
//!    in canonical order. Once execution starts, any operation-set failure
//!    aborts; there is no partial fallback to base execution.
//!
//! The v1 binding replaces a whole base entry region; partial-region
//! replacement is a later phase and needs its own liveness proof format.

use alloc::vec::Vec;

use sha2::Digest as _;

use crate::{CompactProgram, DecodeError, Reader, write_u32};

const MAGIC: &[u8; 4] = b"CRBV";

/// The only variant version this implementation reads or writes.
pub const VARIANT_VERSION_V1: u32 = 1;
/// `variant_kind` naming a Volar BinFHE V2 bootstrap-plan schedule.
pub const KIND_BINFHE_V2: u32 = 1;

const OP_LAYER_BEGIN: u8 = 0;
const OP_LAYER_END: u8 = 1;
const OP_CONST: u8 = 2;
const OP_NOT: u8 = 3;
const OP_LUT: u8 = 4;
const OP_CIRCUIT_BOOT: u8 = 5;
const OP_RGSW_MUX: u8 = 6;

const SOURCE_DIGEST_DOMAIN: &[u8] = b"CRBV-SRC-1";
const PLAN_DIGEST_DOMAIN: &[u8] = b"CRBV-PLAN-1";

/// The profile a schedule was built for (same tag mapping as Volar `VBP1`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantProfile {
    /// Exact, noiseless correctness fixture (test-only).
    Toy,
    /// Seeded noisy small-scale profile (test-only).
    ToyNoisy,
    /// Production-shaped profile; rejected until Volar's V2 parameter gate
    /// and a Cirrus deployment review both admit it.
    Std128,
    /// Caller-defined profile; never admitted by the baseline operation sets.
    Custom,
}

impl VariantProfile {
    const fn tag(self) -> u32 {
        match self {
            Self::Toy => 0,
            Self::ToyNoisy => 1,
            Self::Std128 => 2,
            Self::Custom => 3,
        }
    }

    const fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(Self::Toy),
            1 => Some(Self::ToyNoisy),
            2 => Some(Self::Std128),
            3 => Some(Self::Custom),
            _ => None,
        }
    }

    /// Whether the producer classifies this profile as test-only.
    pub const fn is_test_profile(self) -> bool {
        matches!(self, Self::Toy | Self::ToyNoisy)
    }
}

/// Caller resource caps enforced during validation, before any execution.
///
/// Targets should derive these from their actual byte/RAM budgets; export
/// deduplication and source-binding checks are quadratic in list length, so
/// `max_io` also bounds validation work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantLimits {
    /// Maximum total CRBV byte length.
    pub max_section_bytes: usize,
    /// Maximum number of layers.
    pub max_layers: u32,
    /// Maximum total records across all layers.
    pub max_records: u32,
    /// Maximum summed LUT table bits.
    pub max_lut_bits: u32,
    /// Maximum admitted `k_max` (LUT arity cap).
    pub max_k_max: u32,
    /// Maximum final length of any one arena, imports included.
    pub max_arena_values: u32,
    /// Maximum length of any one import or export list.
    pub max_io: u32,
}

impl VariantLimits {
    /// Generous host-side caps for compilation and tests.
    pub const HOST: Self = Self {
        max_section_bytes: 1 << 20,
        max_layers: 1 << 12,
        max_records: 1 << 16,
        max_lut_bits: 1 << 20,
        max_k_max: 16,
        max_arena_values: 1 << 16,
        max_io: 1 << 12,
    };

    /// Format-bounds-only caps for the writer's encode-time self-check.
    const UNBOUNDED: Self = Self {
        max_section_bytes: usize::MAX,
        max_layers: u32::MAX,
        max_records: u32::MAX,
        max_lut_bits: u32::MAX,
        max_k_max: u32::MAX,
        max_arena_values: u32::MAX,
        max_io: u32::MAX,
    };
}

/// Why CRBV bytes or a CRBV-to-base binding were rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantError {
    /// The byte stream has no `CRBV` magic.
    BadMagic,
    /// The variant version is not implemented by this decoder.
    UnsupportedVersion,
    /// The profile tag is unknown.
    UnsupportedProfile,
    /// A length, digest, or record extends past the provided byte slice.
    Truncated,
    /// A LEB128 integer is overlong, noncanonical, or overflows `u32`.
    InvalidInteger,
    /// A length-delimited section is malformed or trailing bytes remain.
    InvalidSection,
    /// The recomputed plan digest does not match the header.
    PlanDigestMismatch,
    /// A LUT table violates shape, packing, padding, or `k_max` rules.
    InvalidLut,
    /// A layer is empty, nested, or missing its `LAYER_END`.
    InvalidLayer,
    /// A record has an unknown opcode or malformed operands.
    InvalidRecord,
    /// An operand references a value outside its arena, produced by the
    /// same layer, or not yet produced.
    BadReference,
    /// An export names a value that does not exist.
    BadExport,
    /// Two exports drive different values into one destination.
    DuplicateExport,
    /// A caller resource cap was exceeded.
    LimitExceeded,
    /// The recomputed source digest does not match the supplied base
    /// executable.
    SourceMismatch,
    /// A v1 whole-entry variant only binds a pure base region: the base
    /// executable declares storage banks or static initialization.
    EffectfulBase,
    /// A v1 whole-entry binding must not import or export RLWE cells.
    UnsupportedCellBinding,
    /// A whole-entry binding does not reproduce the base input/output slot
    /// contract.
    SlotBindingMismatch,
}

/// Why a [`VariantSpec`] could not be encoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// The spec requests a variant version this writer does not emit.
    UnsupportedVersion,
    /// A count or length exceeds the canonical `u32` fields.
    TooLarge,
    /// The spec violates CRBV structural rules (found by re-decoding the
    /// emitted bytes).
    InvalidSpec(VariantError),
}

/// Why variant execution stopped. Any operation-set failure is terminal:
/// the interpreter never falls back to base execution part-way through.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantExecError<E> {
    /// The operation set's declared kind/version does not match the payload.
    CapabilityMismatch,
    /// The operation set rejected the profile/`k_max` during admission,
    /// before any value was created.
    Admission(E),
    /// An operation failed; execution aborted at that record.
    Operation(E),
}

fn read_error(error: DecodeError) -> VariantError {
    match error {
        DecodeError::Truncated => VariantError::Truncated,
        DecodeError::InvalidInteger => VariantError::InvalidInteger,
        _ => VariantError::InvalidSection,
    }
}

fn push_uleb_stack(bytes: &mut [u8; 5], mut value: u32) -> usize {
    let mut len = 0;
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes[len] = byte;
        len += 1;
        if value == 0 {
            return len;
        }
    }
}

/// Compute the cryptographic binding between a CRBV payload and its base
/// CRBC executable.
///
/// The digest covers the canonical base bytes (format version, slots,
/// input/output tables, bank and initialization declarations, and the whole
/// entry region) plus the selected variant kind/version, so a format
/// revision, a different region, or a different binding list cannot
/// accidentally select the same schedule.
pub fn compute_source_digest(kind: u32, version: u32, crbc: &[u8]) -> [u8; 32] {
    let mut hasher = sha2::Sha256::new();
    hasher.update(SOURCE_DIGEST_DOMAIN);
    hasher.update((crbc.len() as u64).to_le_bytes());
    hasher.update(crbc);
    let mut leb = [0u8; 5];
    let kind_len = push_uleb_stack(&mut leb, kind);
    hasher.update(&leb[..kind_len]);
    let version_len = push_uleb_stack(&mut leb, version);
    hasher.update(&leb[..version_len]);
    hasher.finalize().into()
}

fn compute_plan_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = sha2::Sha256::new();
    hasher.update(PLAN_DIGEST_DOMAIN);
    hasher.update(payload);
    hasher.finalize().into()
}

/// One owned host-side record for the CRBV writer. Output ids are implicit:
/// each record appends exactly one value to its result arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariantRecord {
    /// Cleartext constant wire.
    Const {
        /// The Boolean value.
        value: bool,
    },
    /// Exact NOT of an earlier-layer wire.
    Not {
        /// Input wire id.
        input: u32,
    },
    /// Multi-input LUT read; `inputs` are LSB-first.
    Lut {
        /// Input wire ids, LSB-first; length must equal the table arity.
        inputs: Vec<u32>,
        /// Index into [`VariantSpec::luts`].
        table: u32,
    },
    /// Circuit bootstrap: Boolean wire to RGSW value.
    CircuitBootstrap {
        /// Input wire id.
        input: u32,
    },
    /// Oblivious select between two cells: `sel ? then_cell : else_cell`.
    RgswMux {
        /// Selector RGSW id.
        sel: u32,
        /// Cell selected when the selector value is true.
        then_cell: u32,
        /// Cell selected when the selector value is false.
        else_cell: u32,
    },
}

/// One exported value: the arena value id and the destination binding
/// (a base CRBC slot for wires; a caller-declared binding for cells).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantExport {
    /// Exported arena value id.
    pub value: u32,
    /// Destination binding (base slot for wires).
    pub slot: u32,
}

/// Owned host-side description of one CRBV payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantSpec {
    /// Variant kind; only [`KIND_BINFHE_V2`] has defined semantics today.
    pub kind: u32,
    /// Must be [`VARIANT_VERSION_V1`].
    pub version: u32,
    /// Profile the schedule was built for.
    pub profile: VariantProfile,
    /// [`compute_source_digest`] over the base CRBC executable.
    pub source_digest: [u8; 32],
    /// Producer's diagnostic plan hash (carried, never verified).
    pub plan_hash: u64,
    /// Maximum LUT arity admitted by this schedule.
    pub k_max: u32,
    /// Base slots feeding the positional wire-input arena.
    pub wire_imports: Vec<u32>,
    /// Caller-declared bindings feeding the positional cell-input arena.
    pub cell_imports: Vec<u32>,
    /// Ordered wire exports.
    pub wire_exports: Vec<VariantExport>,
    /// Ordered cell exports.
    pub cell_exports: Vec<VariantExport>,
    /// Logical LUT tables in source order; each length is `1 << arity`.
    pub luts: Vec<Vec<bool>>,
    /// Topologically ordered layers of independent records.
    pub layers: Vec<Vec<VariantRecord>>,
}

/// Canonically encode one CRBV payload.
///
/// The writer computes the plan digest over the emitted payload and then
/// re-decodes its own output as a self-check; structural violations are
/// reported as [`EncodeError::InvalidSpec`].
pub fn encode_variant(spec: &VariantSpec) -> Result<Vec<u8>, EncodeError> {
    if spec.version != VARIANT_VERSION_V1 {
        return Err(EncodeError::UnsupportedVersion);
    }
    let u32_len = |len: usize| u32::try_from(len).map_err(|_| EncodeError::TooLarge);

    let mut payload = Vec::new();
    payload.extend_from_slice(&spec.plan_hash.to_le_bytes());
    write_u32(&mut payload, spec.k_max);
    write_u32(&mut payload, u32_len(spec.wire_imports.len())?);
    for slot in &spec.wire_imports {
        write_u32(&mut payload, *slot);
    }
    write_u32(&mut payload, u32_len(spec.cell_imports.len())?);
    for slot in &spec.cell_imports {
        write_u32(&mut payload, *slot);
    }
    write_u32(&mut payload, u32_len(spec.wire_exports.len())?);
    for export in &spec.wire_exports {
        write_u32(&mut payload, export.value);
        write_u32(&mut payload, export.slot);
    }
    write_u32(&mut payload, u32_len(spec.cell_exports.len())?);
    for export in &spec.cell_exports {
        write_u32(&mut payload, export.value);
        write_u32(&mut payload, export.slot);
    }

    let mut luts = Vec::new();
    write_u32(&mut luts, u32_len(spec.luts.len())?);
    for entries in &spec.luts {
        if entries.is_empty() || !entries.len().is_power_of_two() {
            return Err(EncodeError::InvalidSpec(VariantError::InvalidLut));
        }
        let arity = entries.len().trailing_zeros();
        write_u32(&mut luts, arity);
        write_u32(&mut luts, u32_len(entries.len())?);
        for chunk in entries.chunks(8) {
            let mut packed = 0u8;
            for (bit, entry) in chunk.iter().enumerate() {
                packed |= u8::from(*entry) << bit;
            }
            luts.push(packed);
        }
    }

    let mut layers = Vec::new();
    for layer in &spec.layers {
        layers.push(OP_LAYER_BEGIN);
        write_u32(&mut layers, u32_len(layer.len())?);
        for record in layer {
            match record {
                VariantRecord::Const { value } => {
                    layers.push(OP_CONST);
                    layers.push(u8::from(*value));
                }
                VariantRecord::Not { input } => {
                    layers.push(OP_NOT);
                    write_u32(&mut layers, *input);
                }
                VariantRecord::Lut { inputs, table } => {
                    layers.push(OP_LUT);
                    write_u32(&mut layers, u32_len(inputs.len())?);
                    for input in inputs {
                        write_u32(&mut layers, *input);
                    }
                    write_u32(&mut layers, *table);
                }
                VariantRecord::CircuitBootstrap { input } => {
                    layers.push(OP_CIRCUIT_BOOT);
                    write_u32(&mut layers, *input);
                }
                VariantRecord::RgswMux { sel, then_cell, else_cell } => {
                    layers.push(OP_RGSW_MUX);
                    write_u32(&mut layers, *sel);
                    write_u32(&mut layers, *then_cell);
                    write_u32(&mut layers, *else_cell);
                }
            }
        }
        layers.push(OP_LAYER_END);
    }

    write_u32(&mut payload, u32_len(luts.len())?);
    payload.extend_from_slice(&luts);
    write_u32(&mut payload, u32_len(layers.len())?);
    payload.extend_from_slice(&layers);

    let plan_digest = compute_plan_digest(&payload);
    let mut bytes = Vec::with_capacity(4 + 3 + 1 + 64 + payload.len());
    bytes.extend_from_slice(MAGIC);
    write_u32(&mut bytes, spec.kind);
    write_u32(&mut bytes, spec.version);
    write_u32(&mut bytes, spec.profile.tag());
    bytes.extend_from_slice(&spec.source_digest);
    bytes.extend_from_slice(&plan_digest);
    bytes.extend_from_slice(&payload);

    VariantProgram::validate(&bytes, &VariantLimits::UNBOUNDED)
        .map_err(EncodeError::InvalidSpec)?;
    Ok(bytes)
}

/// One validated import/export slot list (borrowed, canonical).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IdList<'a> {
    bytes: &'a [u8],
    count: u32,
}

impl<'a> IdList<'a> {
    fn ids(self) -> impl Iterator<Item = u32> + 'a {
        let mut reader = Reader::new(self.bytes);
        core::iter::from_fn(move || reader.u32().ok())
    }
}

fn read_id_list<'a>(
    reader: &mut Reader<'a>,
    bytes: &'a [u8],
    limits: &VariantLimits,
) -> Result<IdList<'a>, VariantError> {
    let count = reader.u32().map_err(read_error)?;
    if count > limits.max_io {
        return Err(VariantError::LimitExceeded);
    }
    let start = reader.at;
    for _ in 0..count {
        reader.u32().map_err(read_error)?;
    }
    Ok(IdList { bytes: &bytes[start..reader.at], count })
}

fn read_export_list<'a>(
    reader: &mut Reader<'a>,
    bytes: &'a [u8],
    limits: &VariantLimits,
) -> Result<IdList<'a>, VariantError> {
    let count = reader.u32().map_err(read_error)?;
    if count > limits.max_io {
        return Err(VariantError::LimitExceeded);
    }
    let start = reader.at;
    for _ in 0..count {
        reader.u32().map_err(read_error)?;
        reader.u32().map_err(read_error)?;
    }
    Ok(IdList { bytes: &bytes[start..reader.at], count })
}

/// Find one validated LUT table by id, returning `(arity, packed entries)`.
fn lut_table(section: &[u8], index: u32) -> (u32, &[u8]) {
    let mut reader = Reader::new(section);
    let count = reader.u32().expect("validated LUT section");
    assert!(index < count, "validated LUT id");
    for current in 0..count {
        let arity = reader.u32().expect("validated LUT arity");
        let bit_count = reader.u32().expect("validated LUT bit count");
        let packed = reader
            .take((bit_count as usize).div_ceil(8))
            .expect("validated LUT table");
        if current == index {
            return (arity, packed);
        }
    }
    unreachable!("validated LUT id")
}

fn table_bit(packed: &[u8], index: u32) -> bool {
    packed[(index / 8) as usize] >> (index % 8) & 1 != 0
}

fn table_is_constant(bit_count: u32, packed: &[u8]) -> bool {
    let first = table_bit(packed, 0);
    (1..bit_count).all(|index| table_bit(packed, index) == first)
}

/// A validated, borrowed CRBV payload.
///
/// Constructed only by [`VariantProgram::validate`], so every accessor and
/// the execution loop may assume the structural rules hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantProgram<'a> {
    bytes: &'a [u8],
    kind: u32,
    version: u32,
    profile: VariantProfile,
    source_digest: [u8; 32],
    plan_digest: [u8; 32],
    plan_hash: u64,
    k_max: u32,
    wire_imports: IdList<'a>,
    cell_imports: IdList<'a>,
    wire_exports: IdList<'a>,
    cell_exports: IdList<'a>,
    luts: &'a [u8],
    layers: &'a [u8],
    lut_count: u32,
    layer_count: u32,
    record_count: u32,
    bootstrap_count: u64,
    wire_len: u32,
    rgsw_len: u32,
    cell_len: u32,
    max_table_bits: u32,
}

impl<'a> VariantProgram<'a> {
    /// Validate one CRBV byte stream under `limits` without allocating or
    /// executing any operation.
    pub fn validate(bytes: &'a [u8], limits: &VariantLimits) -> Result<Self, VariantError> {
        if bytes.len() > limits.max_section_bytes {
            return Err(VariantError::LimitExceeded);
        }
        let mut reader = Reader::new(bytes);
        if reader.take(4).map_err(read_error)? != MAGIC {
            return Err(VariantError::BadMagic);
        }
        let kind = reader.u32().map_err(read_error)?;
        let version = reader.u32().map_err(read_error)?;
        if version != VARIANT_VERSION_V1 {
            return Err(VariantError::UnsupportedVersion);
        }
        let profile_tag = reader.u32().map_err(read_error)?;
        let profile =
            VariantProfile::from_tag(profile_tag).ok_or(VariantError::UnsupportedProfile)?;
        let source_digest: [u8; 32] = reader
            .take(32)
            .map_err(read_error)?
            .try_into()
            .expect("32 requested bytes");
        let plan_digest: [u8; 32] = reader
            .take(32)
            .map_err(read_error)?
            .try_into()
            .expect("32 requested bytes");
        if compute_plan_digest(&bytes[reader.at..]) != plan_digest {
            return Err(VariantError::PlanDigestMismatch);
        }
        let plan_hash = u64::from_le_bytes(
            reader
                .take(8)
                .map_err(read_error)?
                .try_into()
                .expect("8 requested bytes"),
        );
        let k_max = reader.u32().map_err(read_error)?;
        if k_max > limits.max_k_max {
            return Err(VariantError::LimitExceeded);
        }
        let wire_imports = read_id_list(&mut reader, bytes, limits)?;
        let cell_imports = read_id_list(&mut reader, bytes, limits)?;
        let wire_exports = read_export_list(&mut reader, bytes, limits)?;
        let cell_exports = read_export_list(&mut reader, bytes, limits)?;
        let luts_len = reader.u32().map_err(read_error)? as usize;
        let luts = reader.take(luts_len).map_err(read_error)?;
        let layers_len = reader.u32().map_err(read_error)? as usize;
        let layers = reader.take(layers_len).map_err(read_error)?;
        if !reader.is_empty() {
            return Err(VariantError::InvalidSection);
        }

        let (lut_count, max_table_bits) = validate_luts(luts, k_max, limits)?;
        let (layer_count, record_count, bootstrap_count, wire_len, rgsw_len, cell_len) =
            validate_layers(
                layers,
                luts,
                lut_count,
                wire_imports.count,
                cell_imports.count,
                limits,
            )?;
        validate_exports(&wire_exports, wire_len)?;
        validate_exports(&cell_exports, cell_len)?;

        Ok(Self {
            bytes,
            kind,
            version,
            profile,
            source_digest,
            plan_digest,
            plan_hash,
            k_max,
            wire_imports,
            cell_imports,
            wire_exports,
            cell_exports,
            luts,
            layers,
            lut_count,
            layer_count,
            record_count,
            bootstrap_count,
            wire_len,
            rgsw_len,
            cell_len,
            max_table_bits,
        })
    }

    /// The validated canonical payload bytes.
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The payload's variant kind.
    pub const fn kind(&self) -> u32 {
        self.kind
    }

    /// The payload's variant version.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// The profile the schedule was built for.
    pub const fn profile(&self) -> VariantProfile {
        self.profile
    }

    /// The declared source-binding digest.
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }

    /// The verified payload digest, for digest-pinned selection policies.
    pub const fn plan_digest(&self) -> [u8; 32] {
        self.plan_digest
    }

    /// The producer's diagnostic plan hash. Never a deployment binding.
    pub const fn plan_hash(&self) -> u64 {
        self.plan_hash
    }

    /// The schedule's maximum LUT arity.
    pub const fn k_max(&self) -> u32 {
        self.k_max
    }

    /// Number of LUT tables.
    pub const fn lut_count(&self) -> u32 {
        self.lut_count
    }

    /// Number of layers.
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Total records across all layers.
    pub const fn record_count(&self) -> u32 {
        self.record_count
    }

    /// Scheduled bootstrap operations: non-constant LUT reads plus circuit
    /// bootstraps. Lets a host recompute the producer's failure-budget
    /// relation for test evidence.
    pub const fn bootstrap_count(&self) -> u64 {
        self.bootstrap_count
    }

    /// Required wire-arena capacity (imports plus produced values).
    pub const fn wire_capacity(&self) -> u32 {
        self.wire_len
    }

    /// Required RGSW-arena capacity.
    pub const fn rgsw_capacity(&self) -> u32 {
        self.rgsw_len
    }

    /// Required cell-arena capacity (imports plus produced values).
    pub const fn cell_capacity(&self) -> u32 {
        self.cell_len
    }

    /// Bits of the largest single LUT table; the `lut_entries` scratch must
    /// be at least this wide.
    pub const fn max_table_bits(&self) -> u32 {
        self.max_table_bits
    }

    /// Base slots feeding the positional wire-input arena.
    pub fn wire_imports(&self) -> impl Iterator<Item = u32> + '_ {
        self.wire_imports.ids()
    }

    /// Caller-declared bindings feeding the positional cell-input arena.
    pub fn cell_imports(&self) -> impl Iterator<Item = u32> + '_ {
        self.cell_imports.ids()
    }

    /// Ordered wire exports as `(value, destination)` pairs.
    pub fn wire_exports(&self) -> impl Iterator<Item = VariantExport> + '_ {
        export_pairs(self.wire_exports)
    }

    /// Ordered cell exports as `(value, destination)` pairs.
    pub fn cell_exports(&self) -> impl Iterator<Item = VariantExport> + '_ {
        export_pairs(self.cell_exports)
    }

    /// Verify the v1 whole-entry binding against a validated base CRBC
    /// executable.
    ///
    /// This recomputes the source digest and enforces the v1 contract: the
    /// base is a pure region (no storage banks or static initialization, so
    /// no effect can hide inside a byte-identical executable), there are no
    /// cell imports or exports (base `Program` storage is Boolean-wire, not
    /// the V2 RLWE-cell model), every wire import names a base input slot,
    /// and the wire export destinations reproduce the base output slot table
    /// exactly (as a multiset, so duplicate base outputs bind the same
    /// value).
    pub fn check_source(&self, base: &CompactProgram<'_>) -> Result<(), VariantError> {
        if compute_source_digest(self.kind, self.version, base.bytes()) != self.source_digest {
            return Err(VariantError::SourceMismatch);
        }
        if !base.banks.is_empty() || !base.init.is_empty() {
            return Err(VariantError::EffectfulBase);
        }
        if self.cell_imports.count != 0 || self.cell_exports.count != 0 {
            return Err(VariantError::UnsupportedCellBinding);
        }
        for import in self.wire_imports.ids() {
            let mut found = false;
            let mut inputs = Reader::new(base.inputs);
            while !inputs.is_empty() {
                if inputs.u32().expect("validated base input table") == import {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(VariantError::SlotBindingMismatch);
            }
        }
        // Multiset equality between export destinations and the base output
        // table: every base output is driven exactly as often as declared.
        let mut export_count = 0u32;
        for export in export_pairs(self.wire_exports) {
            export_count += 1;
            let slot = export.slot;
            let in_exports = export_pairs(self.wire_exports)
                .filter(|export| export.slot == slot)
                .count();
            let mut in_base = 0usize;
            let mut outputs = Reader::new(base.outputs);
            while !outputs.is_empty() {
                if outputs.u32().expect("validated base output table") == slot {
                    in_base += 1;
                }
            }
            if in_exports != in_base {
                return Err(VariantError::SlotBindingMismatch);
            }
        }
        let mut base_output_count = 0u32;
        let mut outputs = Reader::new(base.outputs);
        while !outputs.is_empty() {
            outputs.u32().expect("validated base output table");
            base_output_count += 1;
        }
        if export_count != base_output_count {
            return Err(VariantError::SlotBindingMismatch);
        }
        Ok(())
    }

    /// Execute the payload through one caller-supplied operation set.
    ///
    /// The operation set's declared kind/version must match the payload and
    /// its [`VariantOperationSet::admit`] must accept the profile and
    /// `k_max`; both are checked before any value is created. Records run in
    /// canonical layer order; the first operation failure aborts execution.
    ///
    /// `wire_inputs`/`cell_inputs` are consumed positionally (import arena
    /// order). Buffers must be at least the validated capacities. After
    /// success, exported values are read back from the arenas at the ids
    /// yielded by [`Self::wire_exports`]/[`Self::cell_exports`].
    ///
    /// This entry point enforces capability binding but not source binding:
/// admission against a base executable is [`Self::check_source`], which a
    /// deployment performs when selecting the variant (plan §5.2).
    pub fn execute<Ops: VariantOperationSet>(
        &self,
        ops: &mut Ops,
        wire_inputs: &[Ops::Wire],
        cell_inputs: &[Ops::Cell],
        buffers: &mut VariantBuffers<'_, Ops::Wire, Ops::Rgsw, Ops::Cell>,
    ) -> Result<(), VariantExecError<Ops::Error>> {
        if Ops::KIND != self.kind || Ops::VERSION != self.version {
            return Err(VariantExecError::CapabilityMismatch);
        }
        ops.admit(self.profile, self.k_max)
            .map_err(VariantExecError::Admission)?;
        assert_eq!(
            wire_inputs.len(),
            self.wire_imports.count as usize,
            "wire import count must match validated payload"
        );
        assert_eq!(
            cell_inputs.len(),
            self.cell_imports.count as usize,
            "cell import count must match validated payload"
        );
        assert!(
            buffers.wires.len() >= self.wire_len as usize,
            "wire arena capacity must match validated payload"
        );
        assert!(
            buffers.rgsws.len() >= self.rgsw_len as usize,
            "RGSW arena capacity must match validated payload"
        );
        assert!(
            buffers.cells.len() >= self.cell_len as usize,
            "cell arena capacity must match validated payload"
        );
        assert!(
            buffers.lut_entries.len() >= self.max_table_bits as usize,
            "LUT entry scratch capacity must match validated payload"
        );
        let wires: &mut [Option<Ops::Wire>] = &mut *buffers.wires;
        let rgsws: &mut [Option<Ops::Rgsw>] = &mut *buffers.rgsws;
        let cells: &mut [Option<Ops::Cell>] = &mut *buffers.cells;
        let lut_entries: &mut [bool] = &mut *buffers.lut_entries;
        for slot in wires[..self.wire_len as usize].iter_mut() {
            *slot = None;
        }
        for slot in rgsws[..self.rgsw_len as usize].iter_mut() {
            *slot = None;
        }
        for slot in cells[..self.cell_len as usize].iter_mut() {
            *slot = None;
        }
        for (slot, input) in wires.iter_mut().zip(wire_inputs.iter()) {
            *slot = Some(input.clone());
        }
        for (slot, input) in cells.iter_mut().zip(cell_inputs.iter()) {
            *slot = Some(input.clone());
        }

        let mut wires_len = self.wire_imports.count;
        let mut rgsws_len = 0u32;
        let mut cells_len = self.cell_imports.count;
        let mut reader = Reader::new(self.layers);
        while !reader.is_empty() {
            let begin = reader.byte().expect("validated layer");
            debug_assert_eq!(begin, OP_LAYER_BEGIN);
            let count = reader.u32().expect("validated layer");
            for _ in 0..count {
                match reader.byte().expect("validated record") {
                    OP_CONST => {
                        let value = reader.byte().expect("validated record") != 0;
                        wires[wires_len as usize] =
                            Some(ops.constant(value).map_err(VariantExecError::Operation)?);
                        wires_len += 1;
                    }
                    OP_NOT => {
                        let input = reader.u32().expect("validated record") as usize;
                        let value = ops
                            .not(wires[input].as_ref().expect("validated operand"))
                            .map_err(VariantExecError::Operation)?;
                        wires[wires_len as usize] = Some(value);
                        wires_len += 1;
                    }
                    OP_LUT => {
                        let arity = reader.u32().expect("validated record");
                        let ids_start = reader.at;
                        for _ in 0..arity {
                            reader.u32().expect("validated record");
                        }
                        let ids = &self.layers[ids_start..reader.at];
                        let table = reader.u32().expect("validated record");
                        let (table_arity, packed) = lut_table(self.luts, table);
                        debug_assert_eq!(arity, table_arity);
                        let bit_count = 1usize << table_arity;
                        for (index, entry) in lut_entries[..bit_count].iter_mut().enumerate() {
                            *entry = table_bit(packed, index as u32);
                        }
                        let wires_ref: &[Option<Ops::Wire>] = wires;
                        let mut id_reader = Reader::new(ids);
                        let mut inputs = core::iter::from_fn(move || {
                            let id = id_reader.u32().ok()? as usize;
                            wires_ref[id].as_ref()
                        });
                        let value = ops
                            .lut(&mut inputs, &lut_entries[..bit_count], self.k_max)
                            .map_err(VariantExecError::Operation)?;
                        wires[wires_len as usize] = Some(value);
                        wires_len += 1;
                    }
                    OP_CIRCUIT_BOOT => {
                        let input = reader.u32().expect("validated record") as usize;
                        let value = ops
                            .circuit_bootstrap(wires[input].as_ref().expect("validated operand"))
                            .map_err(VariantExecError::Operation)?;
                        rgsws[rgsws_len as usize] = Some(value);
                        rgsws_len += 1;
                    }
                    OP_RGSW_MUX => {
                        let sel = reader.u32().expect("validated record") as usize;
                        let then_cell = reader.u32().expect("validated record") as usize;
                        let else_cell = reader.u32().expect("validated record") as usize;
                        let value = ops
                            .rgsw_mux(
                                rgsws[sel].as_ref().expect("validated operand"),
                                cells[then_cell].as_ref().expect("validated operand"),
                                cells[else_cell].as_ref().expect("validated operand"),
                            )
                            .map_err(VariantExecError::Operation)?;
                        cells[cells_len as usize] = Some(value);
                        cells_len += 1;
                    }
                    _ => unreachable!("validated record opcode"),
                }
            }
            let end = reader.byte().expect("validated layer");
            debug_assert_eq!(end, OP_LAYER_END);
        }
        debug_assert_eq!(wires_len, self.wire_len);
        debug_assert_eq!(rgsws_len, self.rgsw_len);
        debug_assert_eq!(cells_len, self.cell_len);
        Ok(())
    }
}

fn export_pairs(list: IdList<'_>) -> impl Iterator<Item = VariantExport> + '_ {
    let mut reader = Reader::new(list.bytes);
    core::iter::from_fn(move || {
        let value = reader.u32().ok()?;
        let slot = reader.u32().ok()?;
        Some(VariantExport { value, slot })
    })
}

fn validate_luts(
    section: &[u8],
    k_max: u32,
    limits: &VariantLimits,
) -> Result<(u32, u32), VariantError> {
    let mut reader = Reader::new(section);
    let count = reader.u32().map_err(read_error)?;
    let mut total_bits = 0u32;
    let mut max_bits = 0u32;
    for _ in 0..count {
        let arity = reader.u32().map_err(read_error)?;
        if arity > k_max {
            return Err(VariantError::InvalidLut);
        }
        let bit_count = reader.u32().map_err(read_error)?;
        let expected = 1u32.checked_shl(arity).ok_or(VariantError::InvalidLut)?;
        if bit_count != expected {
            return Err(VariantError::InvalidLut);
        }
        let packed = reader
            .take((bit_count as usize).div_ceil(8))
            .map_err(read_error)?;
        if bit_count % 8 != 0
            && packed.last().expect("non-empty LUT table") >> (bit_count % 8) != 0
        {
            return Err(VariantError::InvalidLut);
        }
        total_bits = total_bits
            .checked_add(bit_count)
            .ok_or(VariantError::LimitExceeded)?;
        if total_bits > limits.max_lut_bits {
            return Err(VariantError::LimitExceeded);
        }
        max_bits = max_bits.max(bit_count);
    }
    if !reader.is_empty() {
        return Err(VariantError::InvalidSection);
    }
    Ok((count, max_bits))
}

#[allow(clippy::type_complexity)]
fn validate_layers(
    section: &[u8],
    luts: &[u8],
    lut_count: u32,
    wire_imports: u32,
    cell_imports: u32,
    limits: &VariantLimits,
) -> Result<(u32, u32, u64, u32, u32, u32), VariantError> {
    let mut reader = Reader::new(section);
    let mut wires = wire_imports;
    let mut rgsws = 0u32;
    let mut cells = cell_imports;
    let mut layer_count = 0u32;
    let mut record_count = 0u32;
    let mut bootstrap_count = 0u64;
    let push = |len: &mut u32| -> Result<(), VariantError> {
        *len = len.checked_add(1).ok_or(VariantError::LimitExceeded)?;
        if *len > limits.max_arena_values {
            return Err(VariantError::LimitExceeded);
        }
        Ok(())
    };
    while !reader.is_empty() {
        if reader.byte().map_err(read_error)? != OP_LAYER_BEGIN {
            return Err(VariantError::InvalidLayer);
        }
        layer_count += 1;
        if layer_count > limits.max_layers {
            return Err(VariantError::LimitExceeded);
        }
        let count = reader.u32().map_err(read_error)?;
        if count == 0 {
            return Err(VariantError::InvalidLayer);
        }
        let (layer_wires, layer_rgsws, layer_cells) = (wires, rgsws, cells);
        for _ in 0..count {
            let opcode = reader.byte().map_err(read_error)?;
            record_count += 1;
            if record_count > limits.max_records {
                return Err(VariantError::LimitExceeded);
            }
            match opcode {
                OP_CONST => {
                    if reader.byte().map_err(read_error)? > 1 {
                        return Err(VariantError::InvalidRecord);
                    }
                    push(&mut wires)?;
                }
                OP_NOT => {
                    if reader.u32().map_err(read_error)? >= layer_wires {
                        return Err(VariantError::BadReference);
                    }
                    push(&mut wires)?;
                }
                OP_LUT => {
                    let input_count = reader.u32().map_err(read_error)?;
                    for _ in 0..input_count {
                        if reader.u32().map_err(read_error)? >= layer_wires {
                            return Err(VariantError::BadReference);
                        }
                    }
                    let table = reader.u32().map_err(read_error)?;
                    if table >= lut_count {
                        return Err(VariantError::BadReference);
                    }
                    let (arity, packed) = lut_table(luts, table);
                    if input_count != arity {
                        return Err(VariantError::InvalidRecord);
                    }
                    if !table_is_constant(1 << arity, packed) {
                        bootstrap_count += 1;
                    }
                    push(&mut wires)?;
                }
                OP_CIRCUIT_BOOT => {
                    if reader.u32().map_err(read_error)? >= layer_wires {
                        return Err(VariantError::BadReference);
                    }
                    bootstrap_count += 1;
                    push(&mut rgsws)?;
                }
                OP_RGSW_MUX => {
                    let sel = reader.u32().map_err(read_error)?;
                    let then_cell = reader.u32().map_err(read_error)?;
                    let else_cell = reader.u32().map_err(read_error)?;
                    if sel >= layer_rgsws || then_cell >= layer_cells || else_cell >= layer_cells
                    {
                        return Err(VariantError::BadReference);
                    }
                    push(&mut cells)?;
                }
                _ => return Err(VariantError::InvalidRecord),
            }
        }
        if reader.byte().map_err(read_error)? != OP_LAYER_END {
            return Err(VariantError::InvalidLayer);
        }
    }
    Ok((layer_count, record_count, bootstrap_count, wires, rgsws, cells))
}

fn validate_exports(list: &IdList<'_>, arena_len: u32) -> Result<(), VariantError> {
    for (index, export) in export_pairs(*list).enumerate() {
        if export.value >= arena_len {
            return Err(VariantError::BadExport);
        }
        for (other_index, other) in export_pairs(*list).enumerate() {
            if other_index > index && other.slot == export.slot && other.value != export.value {
                return Err(VariantError::DuplicateExport);
            }
        }
    }
    Ok(())
}

/// Caller-owned bounded storage for variant execution. Every slice must be
/// at least the validated capacity reported by [`VariantProgram`]; the
/// target path never allocates from untrusted counts.
pub struct VariantBuffers<'a, Wire, Rgsw, Cell> {
    /// Wire arena, sized to [`VariantProgram::wire_capacity`].
    pub wires: &'a mut [Option<Wire>],
    /// RGSW arena, sized to [`VariantProgram::rgsw_capacity`].
    pub rgsws: &'a mut [Option<Rgsw>],
    /// Cell arena, sized to [`VariantProgram::cell_capacity`].
    pub cells: &'a mut [Option<Cell>],
    /// LUT entry scratch, sized to [`VariantProgram::max_table_bits`].
    pub lut_entries: &'a mut [bool],
}

/// One caller-supplied set of variant semantics.
///
/// The interpreter owns arena id binding, validation, exports, and layer
/// barriers; the operation set owns only the primitive implementations.
/// Calls occur exactly in decoded record order: this trait is a semantic
/// seam, not permission to change scheduling.
pub trait VariantOperationSet {
    /// Clear or encrypted Boolean wire value.
    type Wire: Clone;
    /// Circuit-bootstrap result value.
    type Rgsw;
    /// RLWE content cell value.
    type Cell: Clone;
    /// Admission or operation failure.
    type Error;

    /// The single variant kind this set implements.
    const KIND: u32;
    /// The single variant version this set implements.
    const VERSION: u32;

    /// Admit one payload's profile and `k_max` before any value is created.
    fn admit(&mut self, profile: VariantProfile, k_max: u32) -> Result<(), Self::Error>;
    /// Materialize a constant wire (trivial encryption for FHE sets).
    fn constant(&mut self, value: bool) -> Result<Self::Wire, Self::Error>;
    /// Exact NOT of one wire.
    fn not(&mut self, input: &Self::Wire) -> Result<Self::Wire, Self::Error>;
    /// Multi-input LUT read. `inputs` yields the record's input wires in
    /// LSB-first order and `entries` is the address-ordered table; the
    /// circuit-wide arity cap `k_max` fixes the wire encoding for FHE sets.
    fn lut<'w>(
        &mut self,
        inputs: &mut dyn Iterator<Item = &'w Self::Wire>,
        entries: &[bool],
        k_max: u32,
    ) -> Result<Self::Wire, Self::Error>
    where
        Self::Wire: 'w;
    /// Circuit bootstrap: Boolean wire to RGSW value.
    fn circuit_bootstrap(&mut self, input: &Self::Wire) -> Result<Self::Rgsw, Self::Error>;
    /// Oblivious select: `selector ? then_cell : else_cell`.
    fn rgsw_mux(
        &mut self,
        selector: &Self::Rgsw,
        then_cell: &Self::Cell,
        else_cell: &Self::Cell,
    ) -> Result<Self::Cell, Self::Error>;
}

/// Why the clear baseline operation set rejected a payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClearError {
    /// The clear set executes test profiles only.
    UnsupportedProfile,
    /// A LUT address fell outside its table (impossible for validated
    /// payloads; defensive).
    MalformedLut,
}

/// The baseline clear Boolean operation set: a deterministic schedule
/// oracle with `Wire = Rgsw = Cell = bool`.
///
/// Clear semantics are explicit and match Volar's independent
/// [`BootstrapPlan::execute_clear`] oracle: a circuit bootstrap is the
/// identity on the Boolean value, and an RGSW mux selects its `then` cell
/// exactly when the selector value is true. Test profiles only.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClearVariantOps;

/// The clear operation set never fails an operation, only admission.
pub type ClearExecError = VariantExecError<ClearError>;

impl VariantOperationSet for ClearVariantOps {
    type Wire = bool;
    type Rgsw = bool;
    type Cell = bool;
    type Error = ClearError;

    const KIND: u32 = KIND_BINFHE_V2;
    const VERSION: u32 = VARIANT_VERSION_V1;

    fn admit(&mut self, profile: VariantProfile, _k_max: u32) -> Result<(), Self::Error> {
        if profile.is_test_profile() {
            Ok(())
        } else {
            Err(ClearError::UnsupportedProfile)
        }
    }

    fn constant(&mut self, value: bool) -> Result<bool, Self::Error> {
        Ok(value)
    }

    fn not(&mut self, input: &bool) -> Result<bool, Self::Error> {
        Ok(!*input)
    }

    fn lut<'w>(
        &mut self,
        inputs: &mut dyn Iterator<Item = &'w bool>,
        entries: &[bool],
        _k_max: u32,
    ) -> Result<bool, Self::Error>
    where
        bool: 'w,
    {
        let mut address = 0usize;
        for (bit, input) in inputs.enumerate() {
            if *input {
                address |= 1 << bit;
            }
        }
        entries.get(address).copied().ok_or(ClearError::MalformedLut)
    }

    fn circuit_bootstrap(&mut self, input: &bool) -> Result<bool, Self::Error> {
        Ok(*input)
    }

    fn rgsw_mux(&mut self, selector: &bool, then_cell: &bool, else_cell: &bool) -> Result<bool, Self::Error> {
        Ok(if *selector { *then_cell } else { *else_cell })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transpile;
    use alloc::vec;
    use cirrus_recompile_core::{Idx, Op, Program};

    const LIMITS: VariantLimits = VariantLimits::HOST;

    fn leb(bytes: &mut Vec<u8>, value: u32) {
        write_u32(bytes, value);
    }

    /// Assemble a CRBV payload from raw sections, computing both digests.
    fn assemble(
        imports: &[u32],
        cell_imports: &[u32],
        exports: &[(u32, u32)],
        cell_exports: &[(u32, u32)],
        lut_section: &[u8],
        layer_section: &[u8],
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x5aa5_u64.to_le_bytes());
        leb(&mut payload, 4); // k_max
        leb(&mut payload, imports.len() as u32);
        for slot in imports {
            leb(&mut payload, *slot);
        }
        leb(&mut payload, cell_imports.len() as u32);
        for slot in cell_imports {
            leb(&mut payload, *slot);
        }
        leb(&mut payload, exports.len() as u32);
        for (value, slot) in exports {
            leb(&mut payload, *value);
            leb(&mut payload, *slot);
        }
        leb(&mut payload, cell_exports.len() as u32);
        for (value, slot) in cell_exports {
            leb(&mut payload, *value);
            leb(&mut payload, *slot);
        }
        leb(&mut payload, lut_section.len() as u32);
        payload.extend_from_slice(lut_section);
        leb(&mut payload, layer_section.len() as u32);
        payload.extend_from_slice(layer_section);

        let plan_digest = compute_plan_digest(&payload);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        leb(&mut bytes, KIND_BINFHE_V2);
        leb(&mut bytes, VARIANT_VERSION_V1);
        leb(&mut bytes, 0); // Toy
        bytes.extend_from_slice(&[0xcc; 32]);
        bytes.extend_from_slice(&plan_digest);
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn empty_luts() -> Vec<u8> {
        vec![0]
    }

    fn pack(entries: &[bool]) -> Vec<u8> {
        let mut luts = vec![1]; // one table
        let arity = entries.len().trailing_zeros();
        leb(&mut luts, arity);
        leb(&mut luts, entries.len() as u32);
        for chunk in entries.chunks(8) {
            let mut byte = 0u8;
            for (bit, entry) in chunk.iter().enumerate() {
                byte |= u8::from(*entry) << bit;
            }
            luts.push(byte);
        }
        luts
    }

    /// A canonical multi-arena sample: two wire imports, two cell imports,
    /// three layers exercising every opcode.
    fn sample_spec(source_digest: [u8; 32]) -> VariantSpec {
        VariantSpec {
            kind: KIND_BINFHE_V2,
            version: VARIANT_VERSION_V1,
            profile: VariantProfile::Toy,
            source_digest,
            plan_hash: 0x5aa5,
            k_max: 4,
            wire_imports: vec![0, 1],
            cell_imports: vec![7, 8],
            wire_exports: vec![
                VariantExport { value: 4, slot: 2 },
                VariantExport { value: 5, slot: 3 },
            ],
            cell_exports: vec![VariantExport { value: 2, slot: 9 }],
            luts: vec![
                vec![false, true, true, false],              // XOR
                vec![true, true],                            // constant-1 table
                vec![false, false, false, true, false, true, true, true], // majority
            ],
            layers: vec![
                vec![
                    VariantRecord::Lut { inputs: vec![0, 1], table: 0 },   // w2 = w0 ^ w1
                    VariantRecord::Const { value: true },                  // w3
                ],
                vec![
                    VariantRecord::Not { input: 2 },                       // w4 = !w2
                    VariantRecord::Lut { inputs: vec![0,1,2], table: 2 }, // w5 = maj(w0,w1,w2)
                ],
                vec![
                    VariantRecord::CircuitBootstrap { input: 4 },          // r0 = w4
                    VariantRecord::Lut { inputs: vec![3], table: 1 },      // w6 = const-1
                ],
                vec![
                    VariantRecord::RgswMux { sel: 0, then_cell: 0, else_cell: 1 }, // c2
                ],
            ],
        }
    }

    fn sample_bytes() -> Vec<u8> {
        encode_variant(&sample_spec([0xcc; 32])).unwrap()
    }

    fn clear_run(
        program: &VariantProgram<'_>,
        wires: &[bool],
        cells: &[bool],
    ) -> (Vec<Option<bool>>, Vec<Option<bool>>, Vec<Option<bool>>) {
        let mut wire_arena = vec![None; program.wire_capacity() as usize];
        let mut rgsw_arena = vec![None; program.rgsw_capacity() as usize];
        let mut cell_arena = vec![None; program.cell_capacity() as usize];
        let mut entries = vec![false; program.max_table_bits() as usize];
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        program
            .execute(&mut ClearVariantOps, wires, cells, &mut buffers)
            .unwrap();
        (wire_arena, rgsw_arena, cell_arena)
    }

    #[test]
    fn canonical_round_trip_reports_metadata() {
        let bytes = sample_bytes();
        assert_eq!(bytes, sample_bytes(), "encoding must be deterministic");
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        assert_eq!(program.kind(), KIND_BINFHE_V2);
        assert_eq!(program.version(), VARIANT_VERSION_V1);
        assert_eq!(program.profile(), VariantProfile::Toy);
        assert_eq!(program.plan_hash(), 0x5aa5);
        assert_eq!(program.k_max(), 4);
        assert_eq!(program.lut_count(), 3);
        assert_eq!(program.layer_count(), 4);
        assert_eq!(program.record_count(), 7);
        // XOR LUT + majority LUT + circuit bootstrap; the constant-1 table
        // and the arity-1 read of it are free.
        assert_eq!(program.bootstrap_count(), 3);
        assert_eq!(program.wire_capacity(), 7);
        assert_eq!(program.rgsw_capacity(), 1);
        assert_eq!(program.cell_capacity(), 3);
        assert_eq!(program.max_table_bits(), 8);
        assert_eq!(program.wire_imports().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(program.cell_imports().collect::<Vec<_>>(), vec![7, 8]);
        assert_eq!(
            program.wire_exports().map(|e| (e.value, e.slot)).collect::<Vec<_>>(),
            vec![(4, 2), (5, 3)]
        );
        assert_eq!(
            program.cell_exports().map(|e| (e.value, e.slot)).collect::<Vec<_>>(),
            vec![(2, 9)]
        );
        // The header is magic (4) + kind LEB (1) + version LEB (1) +
        // profile LEB (1) + source digest (32) + plan digest (32).
        assert_eq!(program.plan_digest(), compute_plan_digest(&bytes[71..]));
    }

    #[test]
    fn rejects_envelope_mutation() {
        let bytes = sample_bytes();
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::BadMagic));
        let mut bad = bytes.clone();
        bad[5] = 9; // variant_version
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::UnsupportedVersion));
        let mut bad = bytes.clone();
        bad[4] = 0x81; // overlong kind LEB
        bad.insert(5, 0x00);
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::InvalidInteger));
        let mut bad = bytes.clone();
        bad[6] = 4; // unknown profile tag
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::UnsupportedProfile));
        let mut bad = bytes.clone();
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::PlanDigestMismatch));
        let mut bad = bytes.clone();
        bad.push(0);
        assert_eq!(VariantProgram::validate(&bad, &LIMITS), Err(VariantError::PlanDigestMismatch));
    }

    #[test]
    fn rejects_every_truncation() {
        let bytes = sample_bytes();
        for prefix in 0..bytes.len() {
            assert!(
                VariantProgram::validate(&bytes[..prefix], &LIMITS).is_err(),
                "prefix {prefix} must not validate"
            );
        }
    }

    #[test]
    fn rejects_malformed_layers() {
        let luts = pack(&[false, true, true, false]);
        // Empty layer.
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &[OP_LAYER_BEGIN, 0, OP_LAYER_END]);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLayer));
        // Missing LAYER_END.
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &[OP_LAYER_BEGIN, 1, OP_CONST, 1]);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::Truncated));
        // Nested LAYER_BEGIN inside a record position.
        let bytes = assemble(
            &[0], &[], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 1, OP_LAYER_BEGIN, 0, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidRecord));
        // Stray bytes before the first LAYER_BEGIN.
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &[OP_CONST, 1, OP_LAYER_END]);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLayer));
        // Record count shorter than the actual records.
        let bytes = assemble(
            &[0], &[], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 1, OP_CONST, 1, OP_CONST, 0, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLayer));
    }

    #[test]
    fn rejects_bad_arena_references() {
        let luts = pack(&[false, true, true, false]);
        // Future wire id.
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &[OP_LAYER_BEGIN, 1, OP_NOT, 5, OP_LAYER_END]);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::BadReference));
        // Same-layer dependency.
        let bytes = assemble(
            &[0], &[], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 2, OP_CONST, 1, OP_NOT, 1, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::BadReference));
        // Cross-arena: RGSW mux with no RGSW values.
        let bytes = assemble(
            &[0], &[0, 1], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 1, OP_RGSW_MUX, 0, 0, 1, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::BadReference));
        // LUT id out of range.
        let bytes = assemble(
            &[0], &[], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 1, OP_LUT, 2, 0, 0, 9, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::BadReference));
        // LUT input count mismatch with table arity.
        let bytes = assemble(
            &[0], &[], &[(0, 1)], &[], &luts,
            &[OP_LAYER_BEGIN, 1, OP_LUT, 1, 0, 0, OP_LAYER_END],
        );
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidRecord));
        // Export value id out of range.
        let bytes = assemble(&[0], &[], &[(9, 1)], &[], &luts, &[OP_LAYER_BEGIN, 1, OP_CONST, 1, OP_LAYER_END]);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::BadExport));
    }

    #[test]
    fn rejects_conflicting_duplicate_exports_only() {
        let luts = empty_luts();
        let layers = [OP_LAYER_BEGIN, 1, OP_CONST, 1, OP_LAYER_END];
        // Same value driven into one destination twice is a no-op, accepted.
        let bytes = assemble(&[0], &[], &[(0, 3), (0, 3)], &[], &luts, &layers);
        assert!(VariantProgram::validate(&bytes, &LIMITS).is_ok());
        // Two different values into one destination conflict.
        let bytes = assemble(&[0], &[], &[(0, 3), (1, 3)], &[], &luts, &layers);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::DuplicateExport));
    }

    #[test]
    fn rejects_malformed_luts() {
        let layers = [OP_LAYER_BEGIN, 1, OP_CONST, 1, OP_LAYER_END];
        // bit_count != 1 << arity.
        let mut luts = vec![1];
        leb(&mut luts, 2);
        leb(&mut luts, 5);
        luts.push(0);
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &layers);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLut));
        // Non-zero padding bits in a 2-entry table.
        let mut luts = vec![1];
        leb(&mut luts, 1);
        leb(&mut luts, 2);
        luts.push(0b1000_0001);
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &layers);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLut));
        // Arity above k_max.
        let mut luts = vec![1];
        leb(&mut luts, 5);
        leb(&mut luts, 32);
        luts.extend_from_slice(&[0; 4]);
        let bytes = assemble(&[0], &[], &[(0, 1)], &[], &luts, &layers);
        assert_eq!(VariantProgram::validate(&bytes, &LIMITS), Err(VariantError::InvalidLut));
    }

    #[test]
    fn enforces_resource_limits() {
        let bytes = sample_bytes();
        let tight = |patch: &mut dyn FnMut(&mut VariantLimits)| {
            let mut limits = VariantLimits::HOST;
            patch(&mut limits);
            VariantProgram::validate(&bytes, &limits).map(drop)
        };
        assert_eq!(tight(&mut |l| l.max_section_bytes = 8), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_layers = 3), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_records = 6), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_lut_bits = 13), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_k_max = 3), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_arena_values = 2), Err(VariantError::LimitExceeded));
        assert_eq!(tight(&mut |l| l.max_io = 1), Err(VariantError::LimitExceeded));
        assert!(tight(&mut |_| {}).is_ok());
    }

    /// Base: four declared inputs, one XOR gate, outputs `[4, 2]`.
    fn base_crbc() -> Vec<u8> {
        let program = Program {
            ops: vec![
                Op::Create(false),
                Op::Create(false),
                Op::Create(false),
                Op::Create(false),
                Op::BitXor(Idx(0), Idx(1)),
            ],
            inputs: vec![Idx(0), Idx(1), Idx(2), Idx(3)],
            outputs: vec![Idx(4), Idx(2)],
            ..Default::default()
        };
        transpile(&program).unwrap()
    }

    /// Same base with one extra slot, producing different CRBC bytes.
    fn other_base_crbc() -> Vec<u8> {
        let program = Program {
            ops: vec![
                Op::Create(false),
                Op::Create(false),
                Op::Create(false),
                Op::Create(false),
                Op::Create(false),
                Op::BitXor(Idx(0), Idx(1)),
            ],
            inputs: vec![Idx(0), Idx(1), Idx(2), Idx(3)],
            outputs: vec![Idx(5), Idx(2)],
            ..Default::default()
        };
        transpile(&program).unwrap()
    }

    fn source_digest(crbc: &[u8]) -> [u8; 32] {
        compute_source_digest(KIND_BINFHE_V2, VARIANT_VERSION_V1, crbc)
    }

    /// Whole-entry binding spec for `base_crbc`: imports base input slots 0
    /// and 1; exports drive base outputs 4 and 2.
    fn base_spec(digest: [u8; 32]) -> VariantSpec {
        VariantSpec {
            kind: KIND_BINFHE_V2,
            version: VARIANT_VERSION_V1,
            profile: VariantProfile::Toy,
            source_digest: digest,
            plan_hash: 0x5aa5,
            k_max: 4,
            wire_imports: vec![0,1],
            cell_imports: Vec::new(),
            wire_exports: vec![
                VariantExport { value: 3, slot: 4 },
                VariantExport { value: 2, slot: 2 },
            ],
            cell_exports: Vec::new(),
            luts: vec![vec![false,true,true,false]],
            layers: vec![
                vec![VariantRecord::Lut { inputs: vec![0,1], table: 0 }],
                vec![VariantRecord::Not { input: 2 }],
            ],
        }
    }

    #[test]
    fn source_binding_accepts_whole_entry_and_rejects_mismatches() {
        let base_bytes = base_crbc();
        let base = CompactProgram::validate(&base_bytes).unwrap();
        let bytes = encode_variant(&base_spec(source_digest(&base_bytes))).unwrap();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        assert_eq!(program.check_source(&base), Ok(()));

        // A different base executable breaks the digest.
        let other_bytes = other_base_crbc();
        let other = CompactProgram::validate(&other_bytes).unwrap();
        assert_eq!(program.check_source(&other), Err(VariantError::SourceMismatch));

        // Import naming a non-input base slot.
        let mut spec = base_spec(source_digest(&base_bytes));
        spec.wire_imports = vec![0,5];
        spec.layers = vec![
            vec![VariantRecord::Lut { inputs: vec![0,1], table: 0 }],
            vec![VariantRecord::Not { input: 2 }],
        ];
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        assert_eq!(program.check_source(&base), Err(VariantError::SlotBindingMismatch));

        // Export destinations that do not reproduce the base outputs.
        let mut spec = base_spec(source_digest(&base_bytes));
        spec.wire_exports = vec![VariantExport { value: 3, slot: 4 }];
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        assert_eq!(program.check_source(&base), Err(VariantError::SlotBindingMismatch));

        // v1 rejects cell bindings at the base boundary.
        let mut spec = base_spec(source_digest(&base_bytes));
        spec.cell_imports = vec![7,8];
        spec.cell_exports = vec![VariantExport { value: 2, slot: 9 }];
        spec.layers.push(vec![VariantRecord::CircuitBootstrap { input: 3 }]);
        spec.layers.push(vec![VariantRecord::RgswMux { sel: 0, then_cell: 0, else_cell: 1 }]);
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        assert_eq!(program.check_source(&base), Err(VariantError::UnsupportedCellBinding));
    }

    #[test]
    fn clear_execution_matches_truth_tables() {
        let bytes = sample_bytes();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        for w0 in [false, true] {
            for w1 in [false, true] {
                for c0 in [false, true] {
                    for c1 in [false, true] {
                        let (wires, rgsws, cells) = clear_run(&program, &[w0, w1], &[c0, c1]);
                        let xor = w0 ^ w1;
                        let majority = (w0 && w1) || (w0 && xor) || (w1 && xor);
                        assert_eq!(wires[2], Some(xor));
                        assert_eq!(wires[3], Some(true));
                        assert_eq!(wires[4], Some(!xor));
                        assert_eq!(wires[5], Some(majority));
                        assert_eq!(wires[6], Some(true)); // constant table read
                        assert_eq!(rgsws[0], Some(!xor), "circuit bootstrap is the identity");
                        assert_eq!(cells[2], Some(if !xor { c0 } else { c1 }));
                    }
                }
            }
        }
    }

    /// Instrumented operation set recording the call sequence.
    struct RecordingOps {
        log: Vec<&'static str>,
        fail_on_lut: Option<usize>,
        luts_seen: usize,
    }

    impl VariantOperationSet for RecordingOps {
        type Wire = bool;
        type Rgsw = bool;
        type Cell = bool;
        type Error = ClearError;

        const KIND: u32 = KIND_BINFHE_V2;
        const VERSION: u32 = VARIANT_VERSION_V1;

        fn admit(&mut self, profile: VariantProfile, _k_max: u32) -> Result<(), Self::Error> {
            ClearVariantOps.admit(profile, _k_max)
        }
        fn constant(&mut self, value: bool) -> Result<bool, Self::Error> {
            self.log.push("const");
            Ok(value)
        }
        fn not(&mut self, input: &bool) -> Result<bool, Self::Error> {
            self.log.push("not");
            Ok(!*input)
        }
        fn lut<'w>(
            &mut self,
            inputs: &mut dyn Iterator<Item = &'w bool>,
            entries: &[bool],
            k_max: u32,
        ) -> Result<bool, Self::Error>
        where
            bool: 'w,
        {
            self.luts_seen += 1;
            if self.fail_on_lut == Some(self.luts_seen) {
                self.log.push("lut:fail");
                return Err(ClearError::MalformedLut);
            }
            self.log.push("lut");
            ClearVariantOps.lut(inputs, entries, k_max)
        }
        fn circuit_bootstrap(&mut self, input: &bool) -> Result<bool, Self::Error> {
            self.log.push("cb");
            Ok(*input)
        }
        fn rgsw_mux(&mut self, selector: &bool, then_cell: &bool, else_cell: &bool) -> Result<bool, Self::Error> {
            self.log.push("mux");
            ClearVariantOps.rgsw_mux(selector, then_cell, else_cell)
        }
    }

    fn recording_run(
        program: &VariantProgram<'_>,
        ops: &mut RecordingOps,
    ) -> Result<(), VariantExecError<ClearError>> {
        let mut wire_arena = vec![None; program.wire_capacity() as usize];
        let mut rgsw_arena = vec![None; program.rgsw_capacity() as usize];
        let mut cell_arena = vec![None; program.cell_capacity() as usize];
        let mut entries = vec![false; program.max_table_bits() as usize];
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        program.execute(ops, &[true, false], &[false, true], &mut buffers)
    }

    #[test]
    fn execution_follows_canonical_layer_order() {
        let bytes = sample_bytes();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        let mut ops = RecordingOps { log: Vec::new(), fail_on_lut: None, luts_seen: 0 };
        recording_run(&program, &mut ops).unwrap();
        assert_eq!(
            ops.log,
            vec!["lut", "const", "not", "lut", "cb", "lut", "mux"],
            "records run in canonical layer order with no early later-layer reads"
        );
    }

    #[test]
    fn operation_failure_aborts_without_fallback() {
        let bytes = sample_bytes();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        let mut ops = RecordingOps { log: Vec::new(), fail_on_lut: Some(2), luts_seen: 0 };
        let result = recording_run(&program, &mut ops);
        assert_eq!(
            result,
            Err(VariantExecError::Operation(ClearError::MalformedLut)),
            "the failing record surfaces its error"
        );
        assert_eq!(
            ops.log,
            vec!["lut", "const", "not", "lut:fail"],
            "no later record runs and nothing falls back"
        );
    }

    #[test]
    fn capability_and_admission_are_checked_first() {
        struct WrongKind;
        impl VariantOperationSet for WrongKind {
            type Wire = bool;
            type Rgsw = bool;
            type Cell = bool;
            type Error = ClearError;
            const KIND: u32 = 99;
            const VERSION: u32 = VARIANT_VERSION_V1;
            fn admit(&mut self, _: VariantProfile, _: u32) -> Result<(), ClearError> { panic!("must not admit") }
            fn constant(&mut self, _: bool) -> Result<bool, ClearError> { panic!("must not run") }
            fn not(&mut self, _: &bool) -> Result<bool, ClearError> { panic!("must not run") }
            fn lut<'w>(&mut self, _: &mut dyn Iterator<Item = &'w bool>, _: &[bool], _: u32) -> Result<bool, ClearError> where bool: 'w { panic!("must not run") }
            fn circuit_bootstrap(&mut self, _: &bool) -> Result<bool, ClearError> { panic!("must not run") }
            fn rgsw_mux(&mut self, _: &bool, _: &bool, _: &bool) -> Result<bool, ClearError> { panic!("must not run") }
        }
        let bytes = sample_bytes();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        let mut wire_arena = vec![None; 8];
        let mut rgsw_arena = vec![None; 1];
        let mut cell_arena = vec![None; 3];
        let mut entries = vec![false; 8];
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        assert_eq!(
            program.execute(&mut WrongKind, &[true, false], &[false, true], &mut buffers),
            Err(VariantExecError::CapabilityMismatch)
        );

        // The clear set rejects non-test profiles during admission.
        let mut spec = sample_spec([0xcc; 32]);
        spec.profile = VariantProfile::Std128;
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &LIMITS).unwrap();
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        assert_eq!(
            program.execute(&mut ClearVariantOps, &[true, false], &[false, true], &mut buffers),
            Err(VariantExecError::Admission(ClearError::UnsupportedProfile))
        );
    }

    #[test]
    fn encoder_rejects_invalid_specs() {
        let mut spec = sample_spec([0xcc; 32]);
        spec.version = 2;
        assert_eq!(encode_variant(&spec), Err(EncodeError::UnsupportedVersion));
        let mut spec = sample_spec([0xcc; 32]);
        spec.luts.push(vec![true, false, true]); // not a power of two
        assert_eq!(
            encode_variant(&spec),
            Err(EncodeError::InvalidSpec(VariantError::InvalidLut))
        );
        let mut spec = sample_spec([0xcc; 32]);
        spec.layers[0].push(VariantRecord::Not { input: 2 }); // same-layer dep
        assert_eq!(
            encode_variant(&spec),
            Err(EncodeError::InvalidSpec(VariantError::BadReference))
        );
    }
}
