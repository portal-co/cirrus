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
use volar_spec::binfhe::params::{std128, toy};
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

/// V2 Std128 ciphertext types at the profile's const-generic shape.
pub type Std128Wire = LweCiphertext<{ std128::N_LWE }>;
/// V2 Std128 RGSW value.
pub type Std128Rgsw = RgswCiphertext<{ std128::BIG_N }, { std128::BS_ELL }>;
/// V2 Std128 RLWE content cell.
pub type Std128Cell = RlweCiphertext<{ std128::BIG_N }>;
/// V2 Std128 bootstrapping key.
pub type Std128Bk =
    BootstrappingKey<{ std128::N_LWE }, { std128::BIG_N }, { std128::BS_ELL }, { std128::KS_ELL }>;
/// V2 Std128 circuit-bootstrapping key.
pub type Std128Cbk = CircuitBootstrappingKey<
    { std128::N_LWE },
    { std128::BIG_N },
    { std128::BS_ELL },
    { std128::KS_ELL },
    { std128::PRIV_ELL },
>;

/// The BinFHE V2 Std128 operation set: real V2 operations over the
/// OpenFHE-transcribed `std128` profile.
///
/// **Fail-closed for production.** V2 is barely not paper-pinned, so this
/// path executes today, but every run is an UNAPPROVED transcript until
/// Volar's §9 evidence (automatable lattice-estimator run plus failure
/// recomputation) is logged and the production review gate passes: review by
/// a much more powerful model with grants from the owner, or a cryptographer
/// directly. Do not select this operation set for production payloads until
/// then.
pub struct BinFheStd128Ops<'a> {
    /// Circuit-bootstrapping key; also carries the bootstrapping key used by
    /// LUT reads.
    pub cbk: &'a Std128Cbk,
    /// Circuit-wide maximum LUT arity this key set executes against.
    pub k_max: u32,
}

impl VariantOperationSet for BinFheStd128Ops<'_> {
    type Wire = Std128Wire;
    type Rgsw = Std128Rgsw;
    type Cell = Std128Cell;
    type Error = ToyOpsError;

    const KIND: u32 = cirrus_recompile_bytecode::variant::KIND_BINFHE_V2;
    const VERSION: u32 = cirrus_recompile_bytecode::variant::VARIANT_VERSION_V1;

    fn admit(&mut self, profile: VariantProfile, k_max: u32) -> Result<(), Self::Error> {
        if profile != VariantProfile::Std128 {
            return Err(ToyOpsError::UnsupportedProfile);
        }
        if k_max != self.k_max {
            return Err(ToyOpsError::KMaxMismatch);
        }
        Ok(())
    }

    fn constant(&mut self, value: bool) -> Result<Self::Wire, Self::Error> {
        Ok(binfhe_trivial::<{ std128::N_LWE }, { std128::LOG_Q_LWE }>(
            value,
            wire_delta::<{ std128::LOG_Q_LWE }>(self.k_max as usize),
        ))
    }

    fn not(&mut self, input: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        Ok(binfhe_not::<{ std128::N_LWE }, { std128::LOG_Q_LWE }>(
            input,
            wire_delta::<{ std128::LOG_Q_LWE }>(self.k_max as usize),
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
        let cts: Vec<Std128Wire> = inputs.cloned().collect();
        if cts.len() != arity {
            return Err(ToyOpsError::ArityMismatch);
        }
        Ok(binfhe_lut_read_dyn::<
            { std128::N_LWE },
            { std128::BIG_N },
            { std128::LOG_Q },
            { std128::LOG_Q_LWE },
            { std128::LOG_MOD_KS },
            { std128::BS_ELL },
            { std128::BS_BASE_LOG },
            { std128::KS_ELL },
            { std128::KS_BASE_LOG },
        >(&cts, entries, k_max as usize, &self.cbk.bk))
    }

    fn circuit_bootstrap(&mut self, input: &Self::Wire) -> Result<Self::Rgsw, Self::Error> {
        Ok(circuit_bootstrap::<
            { std128::N_LWE },
            { std128::BIG_N },
            { std128::LOG_Q },
            { std128::LOG_Q_LWE },
            { std128::BS_ELL },
            { std128::BS_BASE_LOG },
            { std128::KS_ELL },
            { std128::PRIV_ELL },
            { std128::PRIV_BASE_LOG },
        >(input, self.cbk, self.k_max as usize))
    }

    fn rgsw_mux(
        &mut self,
        selector: &Self::Rgsw,
        then_cell: &Self::Cell,
        else_cell: &Self::Cell,
    ) -> Result<Self::Cell, Self::Error> {
        Ok(cmux::<{ std128::BIG_N }, { std128::LOG_Q }, { std128::BS_ELL }, { std128::BS_BASE_LOG }>(
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

    /// One BinFHE V2 profile under differential test: keys, operation set,
    /// and the reference executor, all at the profile's const-generic shape.
    trait Plan: Copy {
        /// Key material (secret keys stay caller-owned).
        type Keys: PlanKeys;
        /// Clear/encrypted wire value.
        type Wire: Clone;
        /// RGSW value.
        type Rgsw;
        /// RLWE cell value.
        type Cell: Clone;
        /// The operation set executed through the CRBV interpreter.
        type Ops<'a>: VariantOperationSet<Wire = Self::Wire, Rgsw = Self::Rgsw, Cell = Self::Cell, Error = ToyOpsError>;

        /// Build the operation set over the caller's keys.
        fn ops<'a>(&self, keys: &'a Self::Keys) -> Self::Ops<'a>;
        /// Encrypt one Boolean wire.
        fn encrypt_wire(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Wire;
        /// Decrypt one wire to its clear value.
        fn decrypt_wire(&self, keys: &Self::Keys, ct: &Self::Wire) -> bool;
        /// Encrypt one Boolean cell.
        fn encrypt_cell(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Cell;
        /// Decode one cell to its clear value.
        fn cell_bit(&self, keys: &Self::Keys, ct: &Self::Cell) -> bool;
        /// Run the payload through CRBV plus this profile's operation set.
        fn execute_crbv(
            &self,
            program: &VariantProgram<'_>,
            keys: &Self::Keys,
            wires: &[Self::Wire],
            cells: &[Self::Cell],
        ) -> (Vec<Option<Self::Wire>>, Vec<Option<Self::Cell>>) {
            let mut wire_arena: Vec<Option<Self::Wire>> =
                (0..program.wire_capacity()).map(|_| None).collect();
            let mut rgsw_arena: Vec<Option<Self::Rgsw>> =
                (0..program.rgsw_capacity()).map(|_| None).collect();
            let mut cell_arena: Vec<Option<Self::Cell>> =
                (0..program.cell_capacity()).map(|_| None).collect();
            let mut entries = vec![false; program.max_table_bits() as usize];
            let mut buffers = VariantBuffers {
                wires: &mut wire_arena,
                rgsws: &mut rgsw_arena,
                cells: &mut cell_arena,
                lut_entries: &mut entries,
            };
            let mut ops = self.ops(keys);
            program.execute(&mut ops, wires, cells, &mut buffers).unwrap();
            (wire_arena, cell_arena)
        }
        /// Run Volar's reference executor for the same schedule.
        fn execute_reference(
            &self,
            plan: &BootstrapPlan,
            keys: &Self::Keys,
            wires: &[Self::Wire],
            cells: &[Self::Cell],
        ) -> (Vec<Self::Wire>, Vec<Self::Cell>);
    }

    /// Deterministic key construction for one profile.
    trait PlanKeys {
        /// Generate keys from a fixed seed.
        fn new(seed: u64) -> Self;
    }

    /// The exact noiseless Toy profile (fast; exhaustive differentials).
    #[derive(Clone, Copy)]
    struct ToyPlan;

    /// The OpenFHE-transcribed `std128` profile. Production selection stays
    /// fail-closed until Volar §9 evidence plus the production review gate;
    /// this path is exercised here because V2 is barely not paper-pinned.
    #[derive(Clone, Copy)]
    struct Std128Plan;

    fn gen_keys<P: Plan>(seed: u64) -> P::Keys {
        P::Keys::new(seed)
    }

    struct ToyKeys {
        lwe: volar_spec::binfhe::lwe::LweSecretKey<{ toy::N_LWE }>,
        rlwe: RlweSecretKey<{ toy::BIG_N }>,
        cbk: ToyCbk,
    }

    impl PlanKeys for ToyKeys {
        fn new(seed: u64) -> Self {
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
    }

    fn toy_keys(seed: u64) -> ToyKeys {
        gen_keys::<ToyPlan>(seed)
    }

    fn encrypt_wire(keys: &ToyKeys, bit: bool, seed: u64) -> ToyWire {
        ToyPlan.encrypt_wire(keys, bit, seed)
    }

    fn decrypt_wire(keys: &ToyKeys, ct: &ToyWire) -> bool {
        ToyPlan.decrypt_wire(keys, ct)
    }

    impl Plan for ToyPlan {
        type Keys = ToyKeys;
        type Wire = ToyWire;
        type Rgsw = ToyRgsw;
        type Cell = ToyCell;
        type Ops<'a> = BinFheToyOps<'a>;

        fn ops<'a>(&self, keys: &'a Self::Keys) -> Self::Ops<'a> {
            BinFheToyOps { cbk: &keys.cbk, k_max: K_MAX }
        }

        fn encrypt_wire(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Wire {
            let mut rng = TestRng::new(seed);
            lwe_encrypt::<{ toy::N_LWE }, { toy::LOG_Q_LWE }, 0, _>(
                bit,
                wire_delta::<{ toy::LOG_Q_LWE }>(K_MAX as usize),
                &keys.lwe,
                &mut rng,
            )
        }

        fn decrypt_wire(&self, keys: &Self::Keys, ct: &Self::Wire) -> bool {
            lwe_decrypt::<{ toy::N_LWE }, { toy::LOG_Q_LWE }>(
                ct,
                &keys.lwe,
                wire_delta::<{ toy::LOG_Q_LWE }>(K_MAX as usize),
            )
        }

        fn encrypt_cell(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Cell {
            let mut rng = TestRng::new(seed);
            let msg = if bit { 1u32 << (toy::LOG_Q - 3) } else { 0 };
            volar_spec::binfhe::rlwe::rlwe_encrypt_scalar::<{ toy::BIG_N }, { toy::LOG_Q }, 0, _>(
                msg, &keys.rlwe, &mut rng,
            )
        }

        fn cell_bit(&self, keys: &Self::Keys, ct: &Self::Cell) -> bool {
            rlwe_phase::<{ toy::BIG_N }, { toy::LOG_Q }>(ct, &keys.rlwe)[0] >> (toy::LOG_Q - 3) & 1
                != 0
        }

        fn execute_reference(
            &self,
            plan: &BootstrapPlan,
            keys: &Self::Keys,
            wires: &[Self::Wire],
            cells: &[Self::Cell],
        ) -> (Vec<Self::Wire>, Vec<Self::Cell>) {
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
    }

    struct Std128Keys {
        lwe: volar_spec::binfhe::lwe::LweSecretKey<{ std128::N_LWE }>,
        rlwe: RlweSecretKey<{ std128::BIG_N }>,
        cbk: Std128Cbk,
    }

    impl PlanKeys for Std128Keys {
        fn new(seed: u64) -> Self {
            let mut rng = TestRng::new(seed);
            let lwe = gen_lwe_secret_key(&mut rng);
            let rlwe = gen_rlwe_secret_key(&mut rng);
            let cbk = gen_circuit_bootstrapping_key::<
                { std128::N_LWE },
                { std128::BIG_N },
                { std128::LOG_Q },
                { std128::LOG_Q_LWE },
                { std128::LOG_MOD_KS },
                { std128::BS_ELL },
                { std128::BS_BASE_LOG },
                { std128::KS_ELL },
                { std128::KS_BASE_LOG },
                { std128::PRIV_ELL },
                { std128::PRIV_BASE_LOG },
                { std128::CBD_ETA },
                _,
            >(&lwe, &rlwe, &mut rng);
            Std128Keys { lwe, rlwe, cbk }
        }
    }

    impl Plan for Std128Plan {
        type Keys = Std128Keys;
        type Wire = Std128Wire;
        type Rgsw = Std128Rgsw;
        type Cell = Std128Cell;
        type Ops<'a> = BinFheStd128Ops<'a>;

        fn ops<'a>(&self, keys: &'a Self::Keys) -> Self::Ops<'a> {
            BinFheStd128Ops { cbk: &keys.cbk, k_max: K_MAX }
        }

        fn encrypt_wire(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Wire {
            let mut rng = TestRng::new(seed);
            lwe_encrypt::<{ std128::N_LWE }, { std128::LOG_Q_LWE }, { std128::CBD_ETA }, _>(
                bit,
                wire_delta::<{ std128::LOG_Q_LWE }>(K_MAX as usize),
                &keys.lwe,
                &mut rng,
            )
        }

        fn decrypt_wire(&self, keys: &Self::Keys, ct: &Self::Wire) -> bool {
            lwe_decrypt::<{ std128::N_LWE }, { std128::LOG_Q_LWE }>(
                ct,
                &keys.lwe,
                wire_delta::<{ std128::LOG_Q_LWE }>(K_MAX as usize),
            )
        }

        fn encrypt_cell(&self, keys: &Self::Keys, bit: bool, seed: u64) -> Self::Cell {
            let mut rng = TestRng::new(seed);
            let msg = if bit { 1u32 << (std128::LOG_Q - 3) } else { 0 };
            volar_spec::binfhe::rlwe::rlwe_encrypt_scalar::<
                { std128::BIG_N },
                { std128::LOG_Q },
                { std128::CBD_ETA },
                _,
            >(msg, &keys.rlwe, &mut rng)
        }

        fn cell_bit(&self, keys: &Self::Keys, ct: &Self::Cell) -> bool {
            rlwe_phase::<{ std128::BIG_N }, { std128::LOG_Q }>(ct, &keys.rlwe)[0]
                >> (std128::LOG_Q - 3)
                & 1
                != 0
        }

        fn execute_reference(
            &self,
            plan: &BootstrapPlan,
            keys: &Self::Keys,
            wires: &[Self::Wire],
            cells: &[Self::Cell],
        ) -> (Vec<Self::Wire>, Vec<Self::Cell>) {
            execute_plan::<
                { std128::N_LWE },
                { std128::BIG_N },
                { std128::LOG_Q },
                { std128::LOG_Q_LWE },
                { std128::LOG_MOD_KS },
                { std128::BS_ELL },
                { std128::BS_BASE_LOG },
                { std128::KS_ELL },
                { std128::KS_BASE_LOG },
                { std128::PRIV_ELL },
                { std128::PRIV_BASE_LOG },
            >(plan, wires, cells, &keys.cbk.bk, &keys.cbk)
        }
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

    /// Transcode a CRBV spec into a `BootstrapPlan`, mirroring the host
    /// adapter direction (VBP1 decode -> CRBV spec). Implicit output ids
    /// are recovered by replaying the append-only arenas.
    pub fn replay_plan(spec: &VariantSpec) -> BootstrapPlan {
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

    /// The three-way differential harness: for one spec, compare Volar's
    /// reference `execute_plan`, CRBV+the profile's operation set, and the
    /// clear oracles over every input combination.
    fn three_way<P: Plan>(spec: &VariantSpec, plan: P, wires: u32, cells: u32) {
        let keys = P::Keys::new(0x50A1);
        three_way_with(spec, plan, &keys, wires, cells);
    }

    fn three_way_with<P: Plan>(spec: &VariantSpec, plan: P, keys: &P::Keys, wires: u32, cells: u32) {
        let replayed = replay_plan(spec);
        replayed.validate().unwrap();
        let bytes = encode_variant(spec).unwrap();
        let program = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        assert_eq!(program.bootstrap_count(), replayed.bootstrap_op_count());
        for wire_bits in 0..(1u32 << wires) {
            for cell_bits in 0..(1u32 << cells) {
                let clear_inputs: Vec<bool> = (0..wires).map(|i| wire_bits >> i & 1 != 0).collect();
                let clear_cells: Vec<bool> = (0..cells).map(|i| cell_bits >> i & 1 != 0).collect();
                // Clear oracles must agree exactly (test profiles only: the
                // clear set admits no production-shaped profile).
                let (plan_wires, plan_cells) = replayed.execute_clear(&clear_inputs, &clear_cells);
                if spec.profile.is_test_profile() {
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
                }
                // Encrypted executors must agree with the clear truth table.
                let enc_wires: Vec<P::Wire> = clear_inputs
                    .iter()
                    .enumerate()
                    .map(|(i, bit)| plan.encrypt_wire(keys, *bit, 5000 + i as u64))
                    .collect();
                let enc_cells: Vec<P::Cell> = clear_cells
                    .iter()
                    .enumerate()
                    .map(|(i, bit)| plan.encrypt_cell(keys, *bit, 7000 + i as u64))
                    .collect();
                let (ref_wires, ref_cells) = plan.execute_reference(&replayed, keys, &enc_wires, &enc_cells);
                let (crbv_wires, crbv_cells) = plan.execute_crbv(&program, keys, &enc_wires, &enc_cells);
                for export in &spec.wire_exports {
                    let expected = plan_wires[export.value as usize];
                    assert_eq!(
                        plan.decrypt_wire(keys, &ref_wires[export.value as usize]),
                        expected,
                        "reference wire export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                    assert_eq!(
                        plan.decrypt_wire(
                            keys,
                            crbv_wires[export.value as usize].as_ref().expect("executed wire"),
                        ),
                        expected,
                        "CRBV wire export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                }
                for export in &spec.cell_exports {
                    let expected = plan_cells[export.value as usize];
                    assert_eq!(
                        plan.cell_bit(keys, &ref_cells[export.value as usize]),
                        expected,
                        "reference cell export for {wire_bits:04b}/{cell_bits:02b}"
                    );
                    assert_eq!(
                        plan.cell_bit(
                            keys,
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
        three_way(&xor_spec(), ToyPlan, 2, 0);
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
        three_way(&spec, ToyPlan, 2, 0);
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
        three_way(&spec, ToyPlan, 2, 2);
    }

    // ---- Std128 profile (Volar §9-unapproved; exercised because V2 is
    // barely not paper-pinned). Heavy const-generic keygen: run with
    // `--ignored` in release. ----

    #[test]
    #[ignore = "Std128 keygen is heavy; run with --ignored in release"]
    fn std128_one_lut_cone_agrees_with_reference_executor() {
        let mut spec = xor_spec();
        spec.profile = VariantProfile::Std128;
        three_way(&spec, Std128Plan, 2, 0);
    }

    // Blocked upstream: `PrivateKeySwitchingKey<BIG_N, PRIV_ELL>` is a
    // stack array of BIG_N*PRIV_ELL RLWE ciphertexts (~72 MB at Std128), so
    // `gen_circuit_bootstrapping_key` overflows an 8 MB stack before this
    // test can run. Volar's Vec-elimination work is moving that key
    // material to heap; re-enable once a heap/borrowed CBK exists.
    #[test]
    #[ignore = "Std128 CBK keygen overflows the stack; see comment"]
    fn std128_circuit_bootstrap_and_rgsw_mux_agree_with_reference_executor() {
        let mut spec = VariantSpec {
            profile: VariantProfile::Std128,
            cell_imports: vec![7,8],
            wire_exports: vec![VariantExport { value: 2, slot: 2 }],
            cell_exports: vec![VariantExport { value: 2, slot: 9 }],
            luts: vec![vec![false,true]], // identity table
            layers: vec![
                vec![
                    VariantRecord::CircuitBootstrap { input: 0 }, // r0 = w0
                    VariantRecord::Not { input: 0 },              // w2 = !w0
                ],
                vec![VariantRecord::RgswMux { sel: 0, then_cell: 0, else_cell: 1 }], // c2
            ],
            ..xor_spec()
        };
        spec.profile = VariantProfile::Std128;
        three_way(&spec, Std128Plan, 2, 2);
    }

    #[test]
    #[ignore = "Std128 keygen is heavy; run with --ignored in release"]
    fn std128_rejects_toy_profile_at_admission() {
        // CBK keygen is exercised even though admission must fail: the cost
        // is the point (fail-closed does not mean fail-cheap-to-reach).
        let keys = gen_keys::<Std128Plan>(0x57D128);
        let spec = xor_spec(); // Toy profile payload
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
        let mut ops = BinFheStd128Ops { cbk: &keys.cbk, k_max: K_MAX };
        let inputs = [
            Std128Plan.encrypt_wire(&keys, true, 1),
            Std128Plan.encrypt_wire(&keys, false, 2),
        ];
        assert_eq!(
            program.execute(&mut ops, &inputs, &[], &mut buffers),
            Err(VariantExecError::Admission(ToyOpsError::UnsupportedProfile))
        );
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

    // ---- Phase D: base-CRBC replacement binding -------------------------

    use cirrus_recompile_bytecode::{CompactProgram, transpile};
    use cirrus_recompile_bytecode::variant::compute_source_digest;
    use volar_ir::boolar::{BIrStmt, LaneId};
    use volar_ir::circuit::BCircuit;
    use volar_ir::ir::IRVarId;
    use volar_ir_common::{Node, StorageId};

    /// Build the CRBV spec for a whole base CRBC entry region from its
    /// validated plan and the base executable's slot identity.
    ///
    /// This mirrors the host compilation flow (plan §5.1): the base
    /// `Program`'s declared input slots bind the plan's wire inputs in
    /// order, the plan's outputs bind the base output slots in order, and
    /// the source digest covers the canonical base bytes.
    pub fn crbv_spec_for_base(
        plan: &BootstrapPlan,
        crbc: &[u8],
        base_inputs: &[u32],
        base_outputs: &[u32],
    ) -> VariantSpec {
        let mut spec = VariantSpec {
            kind: cirrus_recompile_bytecode::variant::KIND_BINFHE_V2,
            version: cirrus_recompile_bytecode::variant::VARIANT_VERSION_V1,
            profile: match plan.profile {
                volar_spec::binfhe::plan::ProfileId::Toy => VariantProfile::Toy,
                volar_spec::binfhe::plan::ProfileId::ToyNoisy => VariantProfile::ToyNoisy,
                volar_spec::binfhe::plan::ProfileId::Std128 => VariantProfile::Std128,
                volar_spec::binfhe::plan::ProfileId::Custom => VariantProfile::Custom,
            },
            source_digest: compute_source_digest(
                cirrus_recompile_bytecode::variant::KIND_BINFHE_V2,
                cirrus_recompile_bytecode::variant::VARIANT_VERSION_V1,
                crbc,
            ),
            plan_hash: plan.plan_hash(),
            k_max: plan.k_max,
            wire_imports: base_inputs.to_vec(),
            cell_imports: Vec::new(),
            wire_exports: Vec::new(),
            cell_exports: Vec::new(),
            luts: plan.luts.iter().map(|lut| lut.entries.clone()).collect(),
            layers: Vec::new(),
        };
        // CRBV output ids are implicit append-only positions, matching
        // `BootstrapPlan`'s arenas one for one; copy the records through.
        for layer in &plan.layers {
            let mut records = Vec::new();
            for op in layer {
                match op {
                    PlanOp::Const { value, .. } => {
                        records.push(VariantRecord::Const { value: *value });
                    }
                    PlanOp::Not { input, .. } => {
                        records.push(VariantRecord::Not { input: *input });
                    }
                    PlanOp::Lut { inputs, table, .. } => {
                        records.push(VariantRecord::Lut {
                            inputs: inputs.as_ref().to_vec(),
                            table: *table,
                        });
                    }
                    PlanOp::CircuitBootstrap { input, .. } => {
                        records.push(VariantRecord::CircuitBootstrap { input: *input });
                    }
                    PlanOp::RgswMux { sel, then_cell, else_cell, .. } => {
                        records.push(VariantRecord::RgswMux {
                            sel: *sel,
                            then_cell: *then_cell,
                            else_cell: *else_cell,
                        });
                    }
                }
            }
            spec.layers.push(records);
        }
        assert_eq!(plan.outputs.len(), base_outputs.len(), "export binding width");
        spec.wire_exports = plan
            .outputs
            .iter()
            .zip(base_outputs.iter())
            .map(|(value, slot)| VariantExport { value: *value, slot: *slot })
            .collect();
        spec
    }

    /// Boolar `(a XOR b) AND (a OR b)` == `a XOR b`, as a `BCircuit`.
    fn xor_and_or_bcircuit() -> BCircuit {
        BCircuit {
            params: 2,
            stmts: vec![
                Node::new(BIrStmt::Xor(IRVarId(0), IRVarId(1)), (), None),
                Node::new(BIrStmt::Or(IRVarId(0), IRVarId(1)), (), None),
                Node::new(BIrStmt::And(IRVarId(2), IRVarId(3)), (), None),
            ],
            pre_init: vec![],
            outputs: vec![IRVarId(4)],
        }
    }

    /// The schedule the weaver produces for `xor_and_or_birblocks`: one
    /// fused 2-input LUT (`a XOR b`) in one layer.
    fn xor_fused_plan() -> BootstrapPlan {
        let plan = BootstrapPlan {
            profile: volar_spec::binfhe::plan::ProfileId::Toy,
            k_max: K_MAX,
            luts: vec![LutSpec { entries: vec![false, true, true, false] }],
            layers: vec![vec![PlanOp::Lut {
                inputs: volar_spec::binfhe::plan::LutInputs::from_slice(&[0, 1]),
                table: 0,
                out: 2,
            }]],
            num_inputs: 2,
            num_cells: 0,
            outputs: vec![2],
            cell_outputs: Vec::new(),
            budget: FailureBudget { per_bootstrap_log2: 30, total_log2: 31 },
        };
        plan.validate().unwrap();
        plan
    }

    #[test]
    fn whole_entry_variant_replaces_base_and_matches_clear_truth() {
        // Host flow: Boolar -> base Program -> CRBC; weaver plan -> CRBV.
        let program = cirrus_volar_boolar::lower_boolar_program(&xor_and_or_bcircuit()).unwrap();
        let crbc = transpile(&program).unwrap();
        let base = CompactProgram::validate(&crbc).unwrap();
        let plan = xor_fused_plan();
        let spec = crbv_spec_for_base(&plan, &crbc, &[0, 1], &[4]);
        let bytes = encode_variant(&spec).unwrap();
        let variant = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        assert_eq!(variant.check_source(&base), Ok(()), "source binding admits the base");

        // Base-only execution vs clear variant vs encrypted variant.
        let keys = toy_keys(0xD0D0);
        for a in [false, true] {
            for b in [false, true] {
                // Base Boolean execution (no storage); inputs are the two
                // declared input wires.
                let mut scratch = [None, None, None, None, None];
                let base_out = base.execute(&mut (), &mut scratch, &[a, b]).unwrap();
                // Clear variant.
                let (crbv_wires, _) = run_clear(&variant, &[a, b], &[]);
                // Encrypted variant.
                let enc: Vec<ToyWire> = [a, b]
                    .iter()
                    .enumerate()
                    .map(|(i, bit)| encrypt_wire(&keys, *bit, 9000 + i as u64))
                    .collect();
                let (enc_wires, _) = ToyPlan.execute_crbv(&variant, &keys, &enc, &[]);
                let expected = a ^ b;
                assert_eq!(base_out, vec![expected], "base execution for ({a},{b})");
                assert_eq!(crbv_wires[2], Some(expected), "clear variant for ({a},{b})");
                assert_eq!(
                    decrypt_wire(&keys, enc_wires[2].as_ref().expect("executed")),
                    expected,
                    "encrypted variant for ({a},{b})"
                );
            }
        }
        // Evidence: the fused variant is smaller than the gate-by-gate base
        // entry and needs exactly one bootstrap.
        assert_eq!(variant.bootstrap_count(), 1);
        assert_eq!(variant.record_count(), 1);
    }

    #[test]
    fn vbp1_crbv_round_trip_is_consistent() {
        // The host adapter transcode direction: plan -> VBP1 -> decode ->
        // CRBV spec; both serialized views must decode to the same plan.
        let plan = xor_fused_plan();
        let vbp1 = volar_spec::binfhe::plan_codec::encode_plan(&plan).unwrap();
        let decoded = volar_spec::binfhe::plan_codec::decode_plan(&vbp1).unwrap();
        assert_eq!(decoded.plan_hash(), plan.plan_hash());

        // CRBV view of the same schedule must carry the same plan hash and
        // replay to the same BootstrapPlan structure.
        let spec = crbv_spec_for_base(&decoded, b"CRBC", &[0, 1], &[2]);
        let replayed = replay_plan(&spec);
        replayed.validate().unwrap();
        assert_eq!(replayed.plan_hash(), plan.plan_hash());
        assert_eq!(spec.plan_hash, plan.plan_hash());
    }

    #[test]
    fn effect_and_storage_boundaries_are_rejected() {
        // A circuit with a storage pre-init is not a pure region.
        let mut with_preinit = xor_and_or_bcircuit();
        with_preinit.pre_init.push(volar_ir::boolar::BIrPreInitSegment {
            storage: StorageId(0),
            lane: LaneId(0),
            addr: vec![],
            data: vec![true],
        });
        let program = cirrus_volar_boolar::lower_boolar_program(&with_preinit).unwrap();
        assert!(
            !program.storage_init.is_empty(),
            "the base program keeps the effect: a plan must not cover it"
        );
        let crbc = transpile(&program).unwrap();
        CompactProgram::validate(&crbc).unwrap();

        // A variant built for the pure circuit must not bind the effected base.
        let plan = xor_fused_plan();
        let spec = crbv_spec_for_base(&plan, &crbc, &[0, 1], &[4]);
        let bytes = encode_variant(&spec).unwrap();
        let variant = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        // A width-0 bank/init happens to be digest-identical, so the digest
        // alone cannot see this effect; v1 binding rejects any base that
        // declares storage at all (fail-closed purity).
        assert_eq!(
            variant.check_source(&CompactProgram::validate(&crbc).unwrap()),
            Err(cirrus_recompile_bytecode::variant::VariantError::EffectfulBase)
        );

        // A storage-reading statement lowers to a storage op; any plan
        // spanning it is rejected at the weaver (HasPreInitialization /
        // UnsupportedStmt). Cirrus-side, a pure variant never covers a base
        // whose entry contains storage records, because the entry bytes are
        // digest-bound.
        let mut with_read = xor_and_or_bcircuit();
        with_read.stmts.push(Node::new(
            BIrStmt::StorageRead { storage: StorageId(0), lane: LaneId(0), addr: vec![] },
            (),
            None,
        ));
        with_read.outputs = vec![IRVarId(5)];
        let program = cirrus_volar_boolar::lower_boolar_program(&with_read).unwrap();
        assert!(!program.storage_ops.is_empty());
        let crbc = transpile(&program).unwrap();
        let base = CompactProgram::validate(&crbc).unwrap();
        let spec = crbv_spec_for_base(&plan, &crbc, &[0, 1], &[5]);
        let bytes = encode_variant(&spec).unwrap();
        let variant = VariantProgram::validate(&bytes, &VariantLimits::HOST).unwrap();
        assert_eq!(
            variant.check_source(&base),
            Err(cirrus_recompile_bytecode::variant::VariantError::EffectfulBase),
            "a base with storage records is not a pure variant region"
        );
    }

    /// Phase E workload record: base vs fused-variant footprint for the
    /// `(a^b)&(a|b)` circuit, printed for the measurement log.
    #[test]
    fn phase_e_workload_measurements() {
        let program = cirrus_volar_boolar::lower_boolar_program(&xor_and_or_bcircuit()).unwrap();
        let crbc = transpile(&program).unwrap();
        let base = CompactProgram::validate(&crbc).unwrap();
        let plan = xor_fused_plan();
        let spec = crbv_spec_for_base(&plan, &crbc, &[0,1], &[4]);
        let crbv = encode_variant(&spec).unwrap();
        let variant = VariantProgram::validate(&crbv, &VariantLimits::HOST).unwrap();
        // Base: 5 slots, 2 declared inputs, 1 output, 5 entry records
        // (2 input constants + xor + or + and) before OP_END.
        // Variant: one fused 2-input LUT, one bootstrap, no gate-by-gate ops.
        let base_ops = 5u32;
        let record = [
            ("base_crbc_bytes", crbc.len() as u64),
            ("crbv_bytes", crbv.len() as u64),
            ("base_slots", base.slots() as u64),
            ("base_entry_ops", base_ops as u64),
            ("variant_records", variant.record_count() as u64),
            ("variant_layers", variant.layer_count() as u64),
            ("variant_luts", variant.lut_count() as u64),
            ("variant_bootstraps", variant.bootstrap_count()),
            ("variant_wire_arena", variant.wire_capacity() as u64),
            ("variant_rgsw_arena", variant.rgsw_capacity() as u64),
            ("variant_cell_arena", variant.cell_capacity() as u64),
            ("variant_lut_scratch_bits", variant.max_table_bits() as u64),
        ];
        for (name, value) in record {
            eprintln!("phase_e {name} {value}");
        }
        // Fused schedule replaces 3 Boolean gates with one LUT read and one
        // bootstrap; the CRBV payload is larger than this tiny base because
        // of its 64-byte digest envelope, and wins as cone count grows.
        assert_eq!(variant.record_count(), 1);
        assert_eq!(variant.bootstrap_count(), 1);
        assert!(crbv.len() > 64, "digest envelope dominates tiny payloads");
    }

    #[test]
    fn base_only_policy_executes_unmodified_base() {
        // BaseOnly: no variant selected; the base runs exactly as before.
        let program = cirrus_volar_boolar::lower_boolar_program(&xor_and_or_bcircuit()).unwrap();
        let crbc = transpile(&program).unwrap();
        let base = CompactProgram::validate(&crbc).unwrap();
        for a in [false, true] {
            for b in [false, true] {
                let mut scratch = [None, None, None, None, None];
                assert_eq!(
                    base.execute(&mut (), &mut scratch, &[a, b]).unwrap(),
                    vec![a ^ b]
                );
            }
        }
    }
}
