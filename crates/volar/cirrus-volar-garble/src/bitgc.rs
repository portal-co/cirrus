//! BitGC garbled-circuit `cirrus_core::Context` backend built on
//! `volar_spec::bitgc`'s per-gate primitives.
//!
//! Mirrors [`crate::VolarGarbleBackend`]/[`crate::VolarEvalBackend`]'s
//! streaming/pull shape (one record per charged gate, XOR free for the
//! fast variant), but delegates the offline expansion and gate arithmetic
//! to `volar_spec::bitgc`. This is the **server-class optional substitute**
//! for the half-gate baseline: the ERT/MCU profiles reject it (see
//! `docs/bitgc-variants-plan.md` §3 and the TinyLabels deployment split).
//!
//! # Structure
//!
//! - Garbler side: [`BitGcGarbleBackend`] consumes the offline expansion
//!   stream ([`volar_spec::bitgc::seed::stream_expansion`]) and the
//!   circuit's gate topology to emit one [`Stitch`] record per charged
//!   gate to a `Pusher`, in circuit order.
//! - Evaluator side: [`BitGcEvalBackend`] consumes ordered [`Stitch`]
//!   records plus the online-level expanded material and evaluates.
//!
//! # Pinning status
//!
//! The gate equations are the reference adapter behind `GateEquation`
//! (BITGC-LEDGER-04). Do not use for protected data; see
//! `volar_spec::bitgc`'s module docs.

extern crate alloc;

use alloc::vec::Vec;
use core::{fmt, marker::PhantomData};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithValue,
    HasError, Pusher,
};
use volar_spec::bitgc::Variant;
use volar_spec::bitgc::common::{
    Circuit, ExpandedGate, GateEquation, GlobalDelta, Stitch, WireState,
};
use volar_spec::bitgc::seed::{ExpansionStep, OfflineEvaluator, OfflineGarbler};
use volar_spec::bitgc::swhe::{Ciphertext, Context as SwheContext, KeySwitchKeySet, SecretKey};

/// The garbler-side BitGC context: streams one [`Stitch`] record per
/// charged gate to [`Self::queue`], in circuit order.
///
/// The garbling pass is driven explicitly: the caller feeds the offline
/// expansion stream and the circuit gates through [`Self::garble_gate`]
/// (or runs [`Self::garble_circuit`] when the whole circuit is known up
/// front), mirroring how `VolarGarbleBackend`'s table sink is driven by
/// the interpreter.
pub struct BitGcGarbleBackend<'a, 'b, E: GateEquation> {
    /// Ordered streaming destination for one [`Stitch`] per charged gate.
    pub queue: &'a mut (dyn Pusher<Stitch> + 'b),
    /// The offline garbler state (secret key, KSKs, expansion plan).
    pub garbler: &'a OfflineGarbler,
    /// The global offset Δ (garbler-chosen, LSB 1).
    pub delta: GlobalDelta,
    /// The gate equation (variant semantics).
    pub equation: E,
    /// Plaintext modulus (from the SWHE profile).
    pub plaintext_modulus: u64,
    marker: PhantomData<&'b ()>,
}

impl<'a, 'b, E: GateEquation> BitGcGarbleBackend<'a, 'b, E> {
    /// Wrap a record sink with the garbler's offline state and Δ.
    pub fn new(
        queue: &'a mut (dyn Pusher<Stitch> + 'b),
        garbler: &'a OfflineGarbler,
        delta: GlobalDelta,
        equation: E,
        plaintext_modulus: u64,
    ) -> Self {
        Self {
            queue,
            garbler,
            delta,
            equation,
            plaintext_modulus,
            marker: PhantomData,
        }
    }

    /// Garble one gate from its expansion step and inputs, pushing the
    /// stitching record if the gate is charged. Returns the output wire's
    /// garbler-side state.
    pub fn garble_gate(
        &mut self,
        index: usize,
        is_and: bool,
        a: &WireState,
        b: &WireState,
        step: &ExpansionStep,
    ) -> WireState {
        let out = step.wire_state;
        if is_and {
            let stitch = self.equation.garble_and(index, a, b, &out);
            self.queue.push(stitch);
            // final π_out = candidate ⊕ correction
            WireState {
                a0: out.a0,
                pi: out.pi ^ stitch.correction(),
            }
        } else {
            let out = if self.equation.xor_is_charged() {
                out
            } else {
                // Free-XOR structural rule.
                WireState {
                    a0: (a.a0 + b.a0) % self.plaintext_modulus,
                    pi: a.pi ^ b.pi,
                }
            };
            if let Some(stitch) = self.equation.garble_xor(self.plaintext_modulus, a, b, &out) {
                self.queue.push(stitch);
            }
            out
        }
    }

    /// Garble a whole circuit from a pre-collected expansion stream,
    /// pushing records in circuit order. Convenience for non-streaming
    /// use (tests, small circuits); streaming integrations drive
    /// [`Self::garble_gate`] directly.
    pub fn garble_circuit(
        &mut self,
        circuit: &Circuit,
        states: &[WireState],
    ) -> Result<(), BitGcError> {
        if states.len() != circuit.wires() {
            return Err(BitGcError::WireCount);
        }
        let mut wires = states.to_vec();
        for (index, gate) in circuit.gates.iter().enumerate() {
            let out_wire = circuit.inputs + index;
            let (ai, bi) = gate.operands();
            let (a, b) = (wires[ai], wires[bi]);
            let out = wires[out_wire];
            wires[out_wire] = match gate {
                volar_spec::bitgc::common::Gate::And { .. } => {
                    let stitch = self.equation.garble_and(index, &a, &b, &out);
                    self.queue.push(stitch);
                    WireState {
                        a0: out.a0,
                        pi: out.pi ^ stitch.correction(),
                    }
                }
                volar_spec::bitgc::common::Gate::Xor { .. } => {
                    let out = if self.equation.xor_is_charged() {
                        out
                    } else {
                        WireState {
                            a0: (a.a0 + b.a0) % self.plaintext_modulus,
                            pi: a.pi ^ b.pi,
                        }
                    };
                    if let Some(stitch) =
                        self.equation
                            .garble_xor(self.plaintext_modulus, &a, &b, &out)
                    {
                        self.queue.push(stitch);
                    }
                    out
                }
            };
        }
        Ok(())
    }
}

/// An error while replaying a BitGC stitching stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitGcError {
    /// A record was required after the stream ended.
    Exhausted,
    /// The expansion/wire-state count does not match the circuit.
    WireCount,
    /// The circuit topology is invalid.
    Topology,
    /// A gate evaluation failed at the SWHE layer.
    Crypto,
}

impl fmt::Display for BitGcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("stitching record stream is exhausted"),
            Self::WireCount => formatter.write_str("wire-state count does not match circuit"),
            Self::Topology => formatter.write_str("invalid circuit topology"),
            Self::Crypto => formatter.write_str("SWHE evaluation error"),
        }
    }
}

impl core::error::Error for BitGcError {}

/// The evaluator-side BitGC context: pulls one [`Stitch`] per charged
/// gate from [`Self::records`], plus the online-level expanded material.
pub struct BitGcEvalBackend<E: GateEquation, I>
where
    I: Iterator<Item = Stitch>,
{
    /// The ordered source of stitching records.
    pub records: I,
    /// The online-level expanded material, one per wire.
    pub expanded: OfflineEvaluator,
    /// The SWHE context and evaluation key material.
    pub ctx: SwheContext,
    /// Key-switching keys for the online gate phase.
    pub ksks: KeySwitchKeySet,
    /// The global offset Δ.
    pub delta: GlobalDelta,
    /// The gate equation (variant semantics).
    pub equation: E,
    /// The output-opening key (garbler-side secret, for the reference
    /// adapter's semi-honest output opening; a session binds this to a
    /// transcript in a real deployment).
    pub output_key: SecretKey,
}

/// Which variant's `cirrus_core` Boolean context this is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitGcProfile {
    /// BitGC-small: 1 bit per AND, 1 bit per XOR.
    Small,
    /// BitGC-fast: 5 bits per AND, free XOR.
    Fast,
}

impl BitGcProfile {
    /// The `volar_spec::bitgc` variant.
    pub const fn variant(self) -> Variant {
        match self {
            Self::Small => Variant::Small,
            Self::Fast => Variant::Fast,
        }
    }
}

impl<E: GateEquation, I> HasError for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    type Error = BitGcError;
}

impl<E: GateEquation, I> ContextWithValue<bool> for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    type Wrapped = Ciphertext;
}

impl<E: GateEquation, I> ContextWithCreate<bool> for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    fn create(&mut self, _val: bool) -> Result<Ciphertext, BitGcError> {
        // The evaluator's wire labels arrive as encrypted input labels
        // from the garbler; `create` cannot mint one locally. Callers
        // drive evaluation through `evaluate_circuit` with the garbler's
        // input labels instead.
        Err(BitGcError::Topology)
    }
}

impl<E: GateEquation, I> ContextWithBitXor<bool> for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    fn bitxor(&mut self, a: Ciphertext, b: Ciphertext) -> Result<Ciphertext, BitGcError> {
        // Free-XOR: payload add at the active-label level.
        self.ctx.add(&a, &b).map_err(|_| BitGcError::Crypto)
    }

    fn bitxor_assign(&mut self, a: &mut Ciphertext, b: Ciphertext) -> Result<(), BitGcError> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}

impl<E: GateEquation, I> ContextWithBitAnd<bool> for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    fn bitand(&mut self, _a: Ciphertext, _b: Ciphertext) -> Result<Ciphertext, BitGcError> {
        // AND requires the expanded material per gate (indices, topology),
        // which the trait's free-floating form lacks. Use
        // `evaluate_circuit`, which drives gates in circuit order.
        Err(BitGcError::Topology)
    }

    fn bitand_assign(&mut self, _a: &mut Ciphertext, _b: Ciphertext) -> Result<(), BitGcError> {
        Err(BitGcError::Topology)
    }
}

impl<E: GateEquation, I> ContextWithBitOr<bool> for BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    fn bitor(&mut self, a: Ciphertext, b: Ciphertext) -> Result<Ciphertext, BitGcError> {
        let either = self.bitxor(a.clone(), b.clone())?;
        let both = self.bitand(a, b)?;
        self.bitxor(either, both)
    }

    fn bitor_assign(&mut self, a: &mut Ciphertext, b: Ciphertext) -> Result<(), BitGcError> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}

impl<E: GateEquation, I> BitGcEvalBackend<E, I>
where
    I: Iterator<Item = Stitch>,
{
    /// Evaluate a whole circuit from the garbler's input labels and the
    /// record stream, returning the decrypted output bits.
    pub fn evaluate_circuit(
        &mut self,
        circuit: &Circuit,
        input_labels: &[Ciphertext],
    ) -> Result<Vec<bool>, BitGcError> {
        if input_labels.len() != circuit.inputs {
            return Err(BitGcError::WireCount);
        }
        let top = self.expanded.wires[0].ct_a0.level();
        let mut wires: Vec<Ciphertext> = Vec::with_capacity(circuit.wires());
        for label in input_labels {
            wires.push(
                self.ctx
                    .switch_to(label, top)
                    .map_err(|_| BitGcError::Crypto)?,
            );
        }
        for (index, gate) in circuit.gates.iter().enumerate() {
            let out_wire = circuit.inputs + index;
            let (ai, bi) = gate.operands();
            let expanded = ExpandedGate {
                a: self.expanded.wires[ai].clone(),
                b: self.expanded.wires[bi].clone(),
                out: self.expanded.wires[out_wire].clone(),
            };
            let la = &wires[ai];
            let lb = &wires[bi];
            let out = match gate {
                volar_spec::bitgc::common::Gate::And { .. } => {
                    let stitch = self.records.next().ok_or(BitGcError::Exhausted)?;
                    self.equation
                        .eval_and(
                            &self.ctx,
                            la,
                            lb,
                            &expanded,
                            &self.delta,
                            &self.ksks,
                            &stitch,
                        )
                        .map_err(|_| BitGcError::Crypto)?
                }
                volar_spec::bitgc::common::Gate::Xor { .. } => {
                    let stitch = if self.equation.xor_is_charged() {
                        Some(self.records.next().ok_or(BitGcError::Exhausted)?)
                    } else {
                        None
                    };
                    self.equation
                        .eval_xor(
                            &self.ctx,
                            la,
                            lb,
                            &expanded,
                            &self.delta,
                            &self.ksks,
                            stitch.as_ref(),
                        )
                        .map_err(|_| BitGcError::Crypto)?
                }
            };
            wires.push(out);
        }

        // Open the outputs (reference semi-honest opening).
        let p = self.ctx.parameters().plaintext_modulus;
        let inv_delta = self.delta.inverse(p).ok_or(BitGcError::Crypto)?;
        let mut outputs = Vec::with_capacity(circuit.outputs.len());
        for &wire in &circuit.outputs {
            let expanded = &self.expanded.wires[wire];
            let label = &wires[wire];
            let ct_a0 = self
                .ctx
                .switch_to(&expanded.ct_a0, label.level())
                .map_err(|_| BitGcError::Crypto)?;
            let diff = self
                .ctx
                .sub(label, &ct_a0)
                .map_err(|_| BitGcError::Crypto)?;
            let payload = self
                .ctx
                .mul_plain(&diff, &self.ctx.constant_slots(inv_delta))
                .map_err(|_| BitGcError::Crypto)?;
            let payload_bits = self
                .ctx
                .decrypt(&self.output_key, &payload)
                .map_err(|_| BitGcError::Crypto)?;
            let ct_pi = self
                .ctx
                .switch_to(&expanded.ct_pi, payload.level())
                .map_err(|_| BitGcError::Crypto)?;
            let pi_bits = self
                .ctx
                .decrypt(&self.output_key, &ct_pi)
                .map_err(|_| BitGcError::Crypto)?;
            let masked = payload_bits[expanded.slot_a0] % 2;
            let pi = pi_bits[expanded.slot_pi] % 2;
            outputs.push((masked ^ pi) == 1);
        }
        Ok(outputs)
    }
}
