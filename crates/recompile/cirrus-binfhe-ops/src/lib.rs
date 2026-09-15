// @pinnedness: unpinned
// @stability: very-unstable
//! Test-only BinFHE V2 operation sets for CRBV compact variants.
//!
//! [`BinFheToyOps`] implements the
//! [`VariantOperationSet`](cirrus_recompile_bytecode::variant::VariantOperationSet)
//! seam by delegating every primitive to Volar's existing V2 public API
//! (`binfhe_lut_read_dyn`, circuit bootstrap, RGSW CMUX, trivial encryption)
//! at the exact `toy` profile. No cryptographic equations are copied, and no
//! scheduling happens here: the CRBV interpreter drives records in canonical
//! order. This crate admits [`VariantProfile::Toy`] only; `ToyNoisy` needs a
//! seeded-noise/budget-observation review. A `Std128` operation set may land
//! separately (V2 is barely not paper-pinned), but production selection stays
//! fail-closed until Volar's §9 evidence (automatable estimator run + failure
//! recomputation) is logged and production review passes: a much more powerful
//! model with grants from the owner, or a cryptographer directly.
//!
//! This is a host/test adapter: ciphertext values are owned by the arenas
//! and cloned on use, exactly like the V2 reference executor `execute_plan`.

#![warn(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use volar_spec::binfhe::circuit_bs::{CircuitBootstrappingKey, circuit_bootstrap};
use volar_spec::binfhe::keys::BootstrappingKey;
use volar_spec::binfhe::lwe::{LweCiphertext, binfhe_not, binfhe_trivial, wire_delta};
use volar_spec::binfhe::params::toy;
use volar_spec::binfhe::pbs::binfhe_lut_read_dyn;
use volar_spec::binfhe::rgsw::{RgswCiphertext, cmux};
use volar_spec::binfhe::rlwe::RlweCiphertext;

use cirrus_recompile_bytecode::variant::{VariantOperationSet, VariantProfile};

/// V2 Toy ciphertext types at the profile's const-generic shape.
pub type ToyWire = LweCiphertext<{ toy::N_LWE }>;
/// V2 Toy RGSW value (circuit-bootstrap result).
pub type ToyRgsw = RgswCiphertext<{ toy::BIG_N }, { toy::BS_ELL }>;
/// V2 Toy RLWE content cell.
pub type ToyCell = RlweCiphertext<{ toy::BIG_N }>;
/// V2 Toy bootstrapping key.
pub type ToyBk = BootstrappingKey<{ toy::N_LWE }, { toy::BIG_N }, { toy::BS_ELL }, { toy::KS_ELL }>;
/// V2 Toy circuit-bootstrapping key (contains the bootstrapping key).
pub type ToyCbk = CircuitBootstrappingKey<
    { toy::N_LWE },
    { toy::BIG_N },
    { toy::BS_ELL },
    { toy::KS_ELL },
    { toy::PRIV_ELL },
>;

/// Why [`BinFheToyOps`] rejected or failed an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToyOpsError {
    /// Only the exact noiseless Toy profile is admitted.
    UnsupportedProfile,
    /// The operation set's fixed `k_max` does not match the payload.
    KMaxMismatch,
    /// A LUT input count is inconsistent with the schedule's arity (a
    /// validated payload cannot trigger this; defensive).
    ArityMismatch,
    /// The V2 primitive rejected the table shape (impossible for validated
    /// payloads; defensive).
    MalformedLut,
}

/// The BinFHE V2 Toy operation set: real V2 reference operations over
/// caller-owned keys.
///
/// `k_max` is fixed at construction because it fixes the wire encoding
/// `Delta = q / 2^(k_max + 1)`; a payload built for any other arity cap is
/// rejected at admission, before any value is created.
pub struct BinFheToyOps<'a> {
    /// Circuit-bootstrapping key; also carries the bootstrapping key used by
    /// LUT reads.
    pub cbk: &'a ToyCbk,
    /// Circuit-wide maximum LUT arity this key set executes against.
    pub k_max: u32,
}

impl VariantOperationSet for BinFheToyOps<'_> {
    type Wire = ToyWire;
    type Rgsw = ToyRgsw;
    type Cell = ToyCell;
    type Error = ToyOpsError;

    const KIND: u32 = cirrus_recompile_bytecode::variant::KIND_BINFHE_V2;
    const VERSION: u32 = cirrus_recompile_bytecode::variant::VARIANT_VERSION_V1;

    fn admit(&mut self, profile: VariantProfile, k_max: u32) -> Result<(), Self::Error> {
        if profile != VariantProfile::Toy {
            return Err(ToyOpsError::UnsupportedProfile);
        }
        if k_max != self.k_max {
            return Err(ToyOpsError::KMaxMismatch);
        }
        Ok(())
    }

    fn constant(&mut self, value: bool) -> Result<Self::Wire, Self::Error> {
        Ok(binfhe_trivial::<{ toy::N_LWE }, { toy::LOG_Q_LWE }>(
            value,
            wire_delta::<{ toy::LOG_Q_LWE }>(self.k_max as usize),
        ))
    }

    fn not(&mut self, input: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        Ok(binfhe_not::<{ toy::N_LWE }, { toy::LOG_Q_LWE }>(
            input,
            wire_delta::<{ toy::LOG_Q_LWE }>(self.k_max as usize),
        ))
    }

    fn lut<'w>(
        &mut self,
        inputs: &mut dyn Iterator<Item = &'w Self::Wire>,
        entries: &[bool],
        k_max: u32,
    ) -> Result<Self::Wire, Self::Error>
    where
        Self::Wire: 'w,
    {
        if k_max != self.k_max {
            return Err(ToyOpsError::KMaxMismatch);
        }
        if !entries.len().is_power_of_two() {
            return Err(ToyOpsError::MalformedLut);
        }
        let arity = entries.len().trailing_zeros() as usize;
        let cts: Vec<ToyWire> = inputs.cloned().collect();
        if cts.len() != arity {
            return Err(ToyOpsError::ArityMismatch);
        }
        Ok(binfhe_lut_read_dyn::<
            { toy::N_LWE },
            { toy::BIG_N },
            { toy::LOG_Q },
            { toy::LOG_Q_LWE },
            { toy::LOG_MOD_KS },
            { toy::BS_ELL },
            { toy::BS_BASE_LOG },
            { toy::KS_ELL },
            { toy::KS_BASE_LOG },
        >(&cts, entries, k_max as usize, &self.cbk.bk))
    }

    fn circuit_bootstrap(&mut self, input: &Self::Wire) -> Result<Self::Rgsw, Self::Error> {
        Ok(circuit_bootstrap::<
            { toy::N_LWE },
            { toy::BIG_N },
            { toy::LOG_Q },
            { toy::LOG_Q_LWE },
            { toy::BS_ELL },
            { toy::BS_BASE_LOG },
            { toy::KS_ELL },
            { toy::PRIV_ELL },
            { toy::PRIV_BASE_LOG },
        >(input, self.cbk, self.k_max as usize))
    }

    fn rgsw_mux(
        &mut self,
        selector: &Self::Rgsw,
        then_cell: &Self::Cell,
        else_cell: &Self::Cell,
    ) -> Result<Self::Cell, Self::Error> {
        Ok(cmux::<{ toy::BIG_N }, { toy::LOG_Q }, { toy::BS_ELL }, { toy::BS_BASE_LOG }>(
            selector, then_cell, else_cell,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use cirrus_recompile_bytecode::variant::{
        ClearVariantOps, VariantBuffers, VariantExecError, VariantExport, VariantLimits,
        VariantProgram, VariantRecord, VariantSpec, encode_variant,
    };
    use volar_spec::SpecRng;
    use volar_spec::binfhe::circuit_bs::gen_circuit_bootstrapping_key;
    use volar_spec::binfhe::lwe::{gen_lwe_secret_key, lwe_decrypt, lwe_encrypt, wire_delta};
    use volar_spec::binfhe::plan::{BootstrapPlan, FailureBudget, LutSpec, PlanOp, execute_plan};
    use volar_spec::binfhe::rlwe::{RlweSecretKey, gen_rlwe_secret_key, rlwe_phase};

    /// Deterministic splitmix64 RNG (same construction as Volar's fixtures).
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
    }

    impl SpecRng for TestRng {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            z as u32
        }
    }

    const K_MAX: u32 = 4;

    struct ToyKeys {
        lwe: volar_spec::binfhe::lwe::LweSecretKey<{ toy::N_LWE }>,
        rlwe: RlweSecretKey<{ toy::BIG_N }>,
        cbk: ToyCbk,
    }

    fn toy_keys(seed: u64) -> ToyKeys {
        let mut rng = TestRng::new(seed);
        let lwe = gen_lwe_secret_key(&mut rng);
        let rlwe = gen_rlwe_secret_key(&mut rng);
        let cbk = gen_circuit_bootstrapping_key::<
            { toy::N_LWE },
            { toy::BIG_N },
            { toy::LOG_Q },
            { toy::LOG_Q_LWE },
            { toy::LOG_MOD_KS },
            { toy::BS_ELL },
            { toy::BS_BASE_LOG },
            { toy::KS_ELL },
            { toy::KS_BASE_LOG },
            { toy::PRIV_ELL },
            { toy::PRIV_BASE_LOG },
            { toy::CBD_ETA },
            _,
        >(&lwe, &rlwe, &mut rng);
        ToyKeys { lwe, rlwe, cbk }
    }

    fn encrypt_wire(keys: &ToyKeys, bit: bool, seed: u64) -> ToyWire {
        let mut rng = TestRng::new(seed);
        lwe_encrypt::<{ toy::N_LWE }, { toy::LOG_Q_LWE }, 0, _>(
            bit,
            wire_delta::<{ toy::LOG_Q_LWE }>(K_MAX as usize),
            &keys.lwe,
            &mut rng,
        )
    }

    fn decrypt_wire(keys: &ToyKeys, ct: &ToyWire) -> bool {
        lwe_decrypt::<{ toy::N_LWE }, { toy::LOG_Q_LWE }>(
            ct,
            &keys.lwe,
            wire_delta::<{ toy::LOG_Q_LWE }>(K_MAX as usize),
        )
    }

    /// Encrypt a Boolean cell as an RLWE scalar message at `Delta' = Q / 8`.
    fn encrypt_cell(keys: &ToyKeys, bit: bool, seed: u64) -> ToyCell {
        let mut rng = TestRng::new(seed);
        let msg = if bit { 1u32 << (toy::LOG_Q - 3) } else { 0 };
        volar_spec::binfhe::rlwe::rlwe_encrypt_scalar::<{ toy::BIG_N }, { toy::LOG_Q }, 0, _>(
            msg, &keys.rlwe, &mut rng,
        )
    }

    fn cell_bit(keys: &ToyKeys, ct: &ToyCell) -> bool {
        let phase = rlwe_phase::<{ toy::BIG_N }, { toy::LOG_Q }>(ct, &keys.rlwe);
        phase[0] >> (toy::LOG_Q - 3) & 1 != 0
    }

    /// One LUT cone: `w2 = XOR(w0, w1)` as a single fused table read.
    fn xor_spec() -> VariantSpec {
        VariantSpec {
            kind: cirrus_recompile_bytecode::variant::KIND_BINFHE_V2,
            version: cirrus_recompile_bytecode::variant::VARIANT_VERSION_V1,
            profile: VariantProfile::Toy,
            source_digest: [0; 32],
            plan_hash: 0,
            k_max: K_MAX,
            wire_imports: vec![0, 1],
            cell_imports: Vec::new(),
            wire_exports: vec![VariantExport { value: 2, slot: 2 }],
            cell_exports: Vec::new(),
            luts: vec![vec![false, true, true, false]],
            layers: vec![vec![VariantRecord::Lut { inputs: vec![0, 1], table: 0 }]],
        }
    }

    /// Transcode a validated CRBV spec back into a `BootstrapPlan`, the
    /// direction the host adapter (Phase A) performs. Implicit output ids
    /// are recovered by replaying the append-only arenas.
    fn replay_plan(spec: &VariantSpec) -> BootstrapPlan {
        let mut wires = spec.wire_imports.len() as u32;
        let mut rgsws = 0u32;
        let mut cells = spec.cell_imports.len() as u32;
        let mut layers = Vec::new();
        let mut budget = FailureBudget { per_bootstrap_log2: 0, total_log2: 0 };
        let mut bootstraps = 0u64;
        for layer in &spec.layers {
            let mut ops = Vec::new();
            for record in layer {
                match record {
                    VariantRecord::Const { value } => {
                        ops.push(PlanOp::Const { out: wires, value: *value });
                        wires += 1;
                    }
                    VariantRecord::Not { input } => {
                        ops.push(PlanOp::Not { input: *input, out: wires });
                        wires += 1;
                    }
                    VariantRecord::Lut { inputs, table } => {
                        ops.push(PlanOp::Lut {
                            inputs: volar_spec::binfhe::plan::LutInputs::from_slice(inputs),
                            table: *table,
                            out: wires,
                        });
                        wires += 1;
                        let entries = &spec.luts[*table as usize];
                        if entries.iter().any(|e| *e != entries[0]) {
                            bootstraps += 1;
                        }
                    }
                    VariantRecord::CircuitBootstrap { input } => {
                        ops.push(PlanOp::CircuitBootstrap { input: *input, out: rgsws });
                        rgsws += 1;
                        bootstraps += 1;
                    }
                    VariantRecord::RgswMux { sel, then_cell, else_cell } => {
                        ops.push(PlanOp::RgswMux {
                            sel: *sel,
                            then_cell: *then_cell,
                            else_cell: *else_cell,
                            out: cells,
                        });
                        cells += 1;
                    }
                }
            }
            layers.push(ops);
        }
        if bootstraps > 0 {
            // Smallest consistent budget (mirrors Volar's validate rule).
            let log2_count = 64 - bootstraps.leading_zeros();
            budget.per_bootstrap_log2 = 30;
            budget.total_log2 = 30 + log2_count;
        }
        BootstrapPlan {
            profile: volar_spec::binfhe::plan::ProfileId::Toy,
            k_max: spec.k_max,
            luts: spec
                .luts
                .iter()
                .map(|entries| LutSpec { entries: entries.clone() })
                .collect(),
            layers,
            num_inputs: spec.wire_imports.len() as u32,
            num_cells: spec.cell_imports.len() as u32,
            outputs: spec.wire_exports.iter().map(|export| export.value).collect(),
            cell_outputs: spec.cell_exports.iter().map(|export| export.value).collect(),
            budget,
        }
    }

    fn run_crbv(
        program: &VariantProgram<'_>,
        keys: &ToyKeys,
        wires: &[ToyWire],
        cells: &[ToyCell],
    ) -> (Vec<Option<ToyWire>>, Vec<Option<ToyCell>>) {
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
        let mut ops = BinFheToyOps { cbk: &keys.cbk, k_max: K_MAX };
        program.execute(&mut ops, wires, cells, &mut buffers).unwrap();
        (wire_arena, cell_arena)
    }

    fn run_reference(
        plan: &BootstrapPlan,
        keys: &ToyKeys,
        wires: &[ToyWire],
        cells: &[ToyCell],
    ) -> (Vec<ToyWire>, Vec<ToyCell>) {
        execute_plan::<
            { toy::N_LWE },
            { toy::BIG_N },
            { toy::LOG_Q },
            { toy::LOG_Q_LWE },
            { toy::LOG_MOD_KS },
            { toy::BS_ELL },
            { toy::BS_BASE_LOG },
            { toy::KS_ELL },
            { toy::KS_BASE_LOG },
            { toy::PRIV_ELL },
            { toy::PRIV_BASE_LOG },
        >(plan, wires, cells, &keys.cbk.bk, &keys.cbk)
    }

    fn run_clear(
        program: &VariantProgram<'_>,
        wires: &[bool],
        cells: &[bool],
    ) -> (Vec<Option<bool>>, Vec<Option<bool>>) {
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
        program.execute(&mut ClearVariantOps, wires, cells, &mut buffers).unwrap();
        (wire_arena, cell_arena)
    }

    /// The three-way differential harness: for one spec, compare
    /// `BootstrapPlan::execute_plan`, CRBV+`BinFheToyOps`, and both clear
    /// evaluators over every input combination.
    fn three_way(spec: &VariantSpec, wires: u32, cells: u32) {
        let plan = replay_plan(spec);
        plan.validate().unwrap();
        let bytes = encode_variant(spec).unwrap();
        let program = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        assert_eq!(program.bootstrap_count(), plan.bootstrap_op_count());
        let keys = toy_keys(0x50A1);
        for wire_bits in 0..(1u32 << wires) {
            for cell_bits in 0..(1u32 << cells) {
                let clear_inputs: Vec<bool> = (0..wires).map(|i| wire_bits >> i & 1 != 0).collect();
                let clear_cells: Vec<bool> = (0..cells).map(|i| cell_bits >> i & 1 != 0).collect();
                // Clear oracles must agree exactly.
                let (plan_wires, plan_cells) = plan.execute_clear(&clear_inputs, &clear_cells);
                let (crbv_wires, crbv_cells) = run_clear(&program, &clear_inputs, &clear_cells);
                for (index, export) in spec.wire_exports.iter().enumerate() {
                    assert_eq!(
                        crbv_wires[export.value as usize],
                        Some(plan_wires[export.value as usize]),
                        "clear wire export {index} for {wire_bits:04b}/{cell_bits:02b}"
                    );
                }
                for export in &spec.cell_exports {
                    assert_eq!(
                        crbv_cells[export.value as usize],
                        Some(plan_cells[export.value as usize]),
                        "clear cell export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                }
                // Encrypted executors must agree with the clear truth table.
                let enc_wires: Vec<ToyWire> = clear_inputs
                    .iter()
                    .enumerate()
                    .map(|(i, bit)| encrypt_wire(&keys, *bit, 5000 + i as u64))
                    .collect();
                let enc_cells: Vec<ToyCell> = clear_cells
                    .iter()
                    .enumerate()
                    .map(|(i, bit)| encrypt_cell(&keys, *bit, 7000 + i as u64))
                    .collect();
                let (ref_wires, ref_cells) = run_reference(&plan, &keys, &enc_wires, &enc_cells);
                let (crbv_wires, crbv_cells) = run_crbv(&program, &keys, &enc_wires, &enc_cells);
                for export in &spec.wire_exports {
                    let expected = plan_wires[export.value as usize];
                    assert_eq!(
                        decrypt_wire(&keys, &ref_wires[export.value as usize]),
                        expected,
                        "reference wire export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                    assert_eq!(
                        decrypt_wire(
                            &keys,
                            crbv_wires[export.value as usize].as_ref().expect("executed wire"),
                        ),
                        expected,
                        "CRBV wire export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                }
                for export in &spec.cell_exports {
                    let expected = plan_cells[export.value as usize];
                    assert_eq!(
                        cell_bit(&keys, &ref_cells[export.value as usize]),
                        expected,
                        "reference cell export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                    assert_eq!(
                        cell_bit(
                            &keys,
                            crbv_cells[export.value as usize].as_ref().expect("executed cell"),
                        ),
                        expected,
                        "CRBV cell export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                }
            }
        }
    }

    #[test]
    fn one_lut_cone_agrees_with_reference_executor() {
        three_way(&xor_spec(), 2, 0);
    }

    #[test]
    fn multi_layer_cone_agrees_with_reference_executor() {
        // w2 = AND(w0, w1); w3 = XOR(w2, w2') via a second layer; also a
        // free NOT and a constant read exercising non-bootstrap records.
        let spec = VariantSpec {
            wire_exports: vec![
                VariantExport { value: 5, slot: 5 },
                VariantExport { value: 4, slot: 4 },
            ],
            luts: vec![
                vec![false, false, false, true], // AND
                vec![false, true, true, false], // XOR
            ],
            layers: vec![
                vec![
                    VariantRecord::Lut { inputs: vec![0, 1], table: 0 }, // w2 = w0 & w1
                    VariantRecord::Const { value: true },                // w3
                ],
                vec![
                    VariantRecord::Not { input: 2 },                        // w4 = !w2
                    VariantRecord::Lut { inputs: vec![2, 1], table: 1 },   // w5 = w2 ^ w1
                ],
            ],
            ..xor_spec()
        };
        three_way(&spec, 2, 0);
    }

    #[test]
    fn circuit_bootstrap_and_rgsw_mux_agree_with_reference_executor() {
        // r0 = cb(w0); c2 = r0 ? c0 : c1; w2 = NOT(w0) for a wire export.
        let spec = VariantSpec {
            cell_imports: vec![7, 8],
            wire_exports: vec![VariantExport { value: 2, slot: 2 }],
            cell_exports: vec![VariantExport { value: 2, slot: 9 }],
            luts: vec![vec![false, true]], // identity table
            layers: vec![
                vec![
                    VariantRecord::CircuitBootstrap { input: 0 }, // r0 = w0
                    VariantRecord::Not { input: 0 },              // w2 = !w0
                ],
                vec![VariantRecord::RgswMux { sel: 0, then_cell: 0, else_cell: 1 }], // c2
            ],
            ..xor_spec()
        };
        three_way(&spec, 2, 2);
    }

    #[test]
    fn rejects_non_toy_profiles_and_k_max_mismatch() {
        let keys = toy_keys(0x50A1);
        let mut spec = xor_spec();
        spec.profile = VariantProfile::ToyNoisy;
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        let mut wire_arena = vec![None; 4];
        let mut rgsw_arena = vec![None; 1];
        let mut cell_arena = vec![None; 1];
        let mut entries = vec![false; 8];
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        let mut ops = BinFheToyOps { cbk: &keys.cbk, k_max: K_MAX };
        let inputs = [encrypt_wire(&keys, true, 1), encrypt_wire(&keys, false, 2)];
        assert_eq!(
            program.execute(&mut ops, &inputs, &[], &mut buffers),
            Err(VariantExecError::Admission(ToyOpsError::UnsupportedProfile))
        );
        // A payload built for a different arity cap must not execute.
        let mut spec = xor_spec();
        spec.k_max = 2;
        spec.luts = vec![vec![false, true, true, false]];
        let bytes = encode_variant(&spec).unwrap();
        let program = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        let mut buffers = VariantBuffers {
            wires: &mut wire_arena,
            rgsws: &mut rgsw_arena,
            cells: &mut cell_arena,
            lut_entries: &mut entries,
        };
        let mut ops = BinFheToyOps { cbk: &keys.cbk, k_max: K_MAX };
        assert_eq!(
            program.execute(&mut ops, &inputs, &[], &mut buffers),
            Err(VariantExecError::Admission(ToyOpsError::KMaxMismatch))
        );
    }
}
