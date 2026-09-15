//! VOLE verifier-state R1CS semantics over the native KoalaBear field.
//!
//! This module implements the initial **public verifier-state, native-field**
//! profile from `trace-to-no-std-spartan-whir-plan.md`. It does not support
//! existing binary-field `GF(2^k)` VOLE traces: characteristic-two field
//! arithmetic cannot be embedded into KoalaBear. A later profile must instead
//! prove bit/byte-decomposed binary-field operations explicitly.
//!
//! The relation is statement-bound: canonical wire order, gate topology, lane
//! count/order, input/setup correlations, and output claims are part of the
//! fixed variable layout. It assumes an ideal/native VOLE setup transcript
//! for `V_w`; proving that setup is outside this slice.

use alloc::{vec, vec::Vec};

use volar_ir::{boolar::BIrStmt, ir::IRVarId};
use volar_ir_common::StorageId;

use crate::{
    CircuitId, ConstraintSemantics, KOALABEAR_MODULUS, LinearCombination, ModeBRelationError,
    R1csRow, UnifiedR1csWitness,
};

/// Versioned profile ID for the initial VOLE verifier-state relation.
pub const VOLE_VERIFIER_RELATION_PROFILE: &[u8] = b"cirrus-vole-verifier-kb-v1";
/// Number of KoalaBear lanes in the initial public verifier-state profile.
pub const VOLE_VERIFIER_LANES: usize = 4;

/// Full public verifier-state witness for the native-field profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoleVerifierTrace {
    /// Verifier offsets, one per lane. This initial profile is public.
    pub delta: [u32; VOLE_VERIFIER_LANES],
    /// Actual bit for every canonical input and statement wire, lifted to
    /// canonical `0/1` KoalaBear representatives on every lane.
    pub x: Vec<[u32; VOLE_VERIFIER_LANES]>,
    /// Setup mask/share component for every canonical wire.
    pub v: Vec<[u32; VOLE_VERIFIER_LANES]>,
    /// Prover-sent hats for AND statements in canonical statement order.
    pub hats: Vec<[u32; VOLE_VERIFIER_LANES]>,
}

impl VoleVerifierTrace {
    /// Validate profile lengths and canonical scalar representatives.
    pub fn validate(&self, wire_count: usize, hat_count: usize) -> Result<(), ModeBRelationError> {
        if self.x.len() != wire_count || self.v.len() != wire_count {
            return Err(ModeBRelationError::WrongWitnessCount {
                expected: wire_count,
                found: self.x.len().max(self.v.len()),
            });
        }
        if self.hats.len() != hat_count {
            return Err(ModeBRelationError::WrongWitnessCount {
                expected: hat_count,
                found: self.hats.len(),
            });
        }
        let canonical = |values: &[u32]| values.iter().all(|value| *value < KOALABEAR_MODULUS);
        if !canonical(&self.delta)
            || !self.x.iter().all(|value| canonical(value))
            || !self.v.iter().all(|value| canonical(value))
            || !self.hats.iter().all(|value| canonical(value))
        {
            return Err(ModeBRelationError::RamMaterializationMismatch);
        }
        Ok(())
    }

    /// Compute every verifier Q share from the setup correlation.
    pub fn q_values(&self) -> Vec<[u32; VOLE_VERIFIER_LANES]> {
        self.x
            .iter()
            .zip(&self.v)
            .map(|(x, v)| {
                core::array::from_fn(|lane| {
                    (v[lane] + x[lane] * self.delta[lane]) % KOALABEAR_MODULUS
                })
            })
            .collect()
    }
}

/// Frozen variable layout for the native-field verifier-state relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoleVerifierRelationLayout {
    /// Total variables addressed by `rows`.
    pub variable_count: usize,
    /// `delta[lane]` variables.
    pub delta: [usize; VOLE_VERIFIER_LANES],
    /// Per-wire actual lifted bits, `x[wire][lane]`.
    pub x: Vec<[usize; VOLE_VERIFIER_LANES]>,
    /// Per-wire setup mask/share variables, `v[wire][lane]`.
    pub v: Vec<[usize; VOLE_VERIFIER_LANES]>,
    /// Per-wire verifier shares, `q[wire][lane]`.
    pub q: Vec<[usize; VOLE_VERIFIER_LANES]>,
    /// Per-AND prover hat variables, in canonical statement order.
    pub hats: Vec<[usize; VOLE_VERIFIER_LANES]>,
}

/// Native-field verifier-state correlation semantics.
///
/// The emitted rows bind:
///
/// ```text
/// x_w[l] in {0,1}
/// Q_w[l] = V_w[l] + x_w[l] * Delta[l]
/// Zero: Q_w[l] = 0
/// One:  Q_w[l] = Delta[l]
/// Not:  Q_c[l] = Delta[l] - Q_a[l]
/// Xor:  Q_c[l] = Q_a[l] + Q_b[l]
/// And:  hat[l] = V_a[l] * V_b[l]
/// ```
///
/// The AND row uses the prover-step formula `hat = V_a * V_b`. Together with
/// the setup correlations this implies the verifier check
/// `Q_a * Q_b + hat = Q_c * Delta` for an honest setup/output transcript. The
/// setup and output-opening claims remain explicit parts of the outer
/// statement; this slice does not prove the VOLE COT itself.
#[derive(Clone, Debug)]
pub struct VoleVerifierSemantics {
    /// Structural circuit binding.
    pub circuit_id: CircuitId,
    /// R1CS rows emitted so far.
    pub rows: Vec<R1csRow>,
    /// Frozen variable layout.
    pub layout: VoleVerifierRelationLayout,
}

impl VoleVerifierSemantics {
    /// Allocate the full relation layout for `wire_count` canonical wires.
    pub fn new(circuit_id: CircuitId, wire_count: usize) -> Self {
        let mut next = 0;
        let mut allocate_lane_array = || {
            core::array::from_fn(|_| {
                let variable = next;
                next += 1;
                variable
            })
        };
        let delta = allocate_lane_array();
        let x = (0..wire_count)
            .map(|_| allocate_lane_array())
            .collect::<Vec<_>>();
        let v = (0..wire_count)
            .map(|_| allocate_lane_array())
            .collect::<Vec<_>>();
        let q = (0..wire_count)
            .map(|_| allocate_lane_array())
            .collect::<Vec<_>>();
        let layout = VoleVerifierRelationLayout {
            variable_count: next,
            delta,
            x,
            v,
            q,
            hats: Vec::new(),
        };
        let mut this = Self {
            circuit_id,
            rows: Vec::new(),
            layout,
        };
        for wire in 0..wire_count {
            this.emit_setup_rows(wire);
        }
        this
    }

    fn allocate_hat(&mut self) -> [usize; VOLE_VERIFIER_LANES] {
        let start = self.layout.variable_count;
        self.layout.variable_count += VOLE_VERIFIER_LANES;
        let hat = core::array::from_fn(|lane| start + lane);
        self.layout.hats.push(hat);
        hat
    }

    fn emit_setup_rows(&mut self, wire: usize) {
        for lane in 0..VOLE_VERIFIER_LANES {
            let x = self.layout.x[wire][lane];
            let v = self.layout.v[wire][lane];
            let q = self.layout.q[wire][lane];
            let delta = self.layout.delta[lane];
            self.rows.push(R1csRow {
                a: LinearCombination::var(x),
                b: lc_one_minus(x),
                c: LinearCombination::constant(0),
            });
            // x*Delta = Q - V.
            self.rows.push(R1csRow {
                a: LinearCombination::var(x),
                b: LinearCombination::var(delta),
                c: LinearCombination::var(q).add(v, -1),
            });
        }
    }

    fn linear_gate(&mut self, wire: usize, terms: impl Fn(usize) -> LinearCombination) {
        for lane in 0..VOLE_VERIFIER_LANES {
            self.rows.push(R1csRow {
                a: LinearCombination::constant(0),
                b: LinearCombination::constant(1),
                c: terms(lane).add(self.layout.q[wire][lane], -1),
            });
        }
    }

    fn prior(&self, wire: usize, reference: IRVarId) -> Result<usize, ModeBRelationError> {
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

    /// Materialize the complete KoalaBear assignment for a validated trace.
    pub fn materialize_witness(
        &self,
        trace: &VoleVerifierTrace,
    ) -> Result<UnifiedR1csWitness, ModeBRelationError> {
        trace.validate(self.layout.x.len(), self.layout.hats.len())?;
        let q = trace.q_values();
        let mut values = vec![0_u32; self.layout.variable_count];
        for lane in 0..VOLE_VERIFIER_LANES {
            values[self.layout.delta[lane]] = trace.delta[lane];
        }
        for wire in 0..trace.x.len() {
            for lane in 0..VOLE_VERIFIER_LANES {
                values[self.layout.x[wire][lane]] = trace.x[wire][lane];
                values[self.layout.v[wire][lane]] = trace.v[wire][lane];
                values[self.layout.q[wire][lane]] = q[wire][lane];
            }
        }
        for (hat, value) in self.layout.hats.iter().zip(&trace.hats) {
            for lane in 0..VOLE_VERIFIER_LANES {
                values[hat[lane]] = value[lane];
            }
        }
        let witness = UnifiedR1csWitness { values };
        evaluate_signed_rows(&self.rows, &witness)?;
        Ok(witness)
    }
}

impl ConstraintSemantics for VoleVerifierSemantics {
    fn statement(
        &mut self,
        wire: usize,
        statement: &BIrStmt<IRVarId, StorageId>,
    ) -> Result<(), ModeBRelationError> {
        match statement {
            BIrStmt::Zero => self.linear_gate(wire, |_| LinearCombination::constant(0)),
            BIrStmt::One => {
                let delta = self.layout.delta;
                self.linear_gate(wire, |lane| LinearCombination::var(delta[lane]))
            }
            BIrStmt::Not(x) => {
                let x = self.prior(wire, *x)?;
                let delta = self.layout.delta;
                let q_x = self.layout.q[x];
                self.linear_gate(wire, |lane| {
                    LinearCombination::var(delta[lane]).add(q_x[lane], -1)
                });
            }
            BIrStmt::Xor(x, y) => {
                let x = self.prior(wire, *x)?;
                let y = self.prior(wire, *y)?;
                let q_x = self.layout.q[x];
                let q_y = self.layout.q[y];
                self.linear_gate(wire, |lane| {
                    LinearCombination::var(q_x[lane]).add(q_y[lane], 1)
                });
            }
            BIrStmt::And(x, y) => {
                let x = self.prior(wire, *x)?;
                let y = self.prior(wire, *y)?;
                let hat = self.allocate_hat();
                for lane in 0..VOLE_VERIFIER_LANES {
                    self.rows.push(R1csRow {
                        a: LinearCombination::var(self.layout.v[x][lane]),
                        b: LinearCombination::var(self.layout.v[y][lane]),
                        c: LinearCombination::var(hat[lane]),
                    });
                }
            }
            // Or must be decomposed by the source semantics; storage/external
            // statements have no decided VOLE ABI and remain fail-closed.
            _ => return Err(ModeBRelationError::UnsupportedStatement { wire }),
        }
        Ok(())
    }
}

fn evaluate_signed_rows(
    rows: &[R1csRow],
    witness: &UnifiedR1csWitness,
) -> Result<(), ModeBRelationError> {
    let value = |variable: usize| -> Result<u64, ModeBRelationError> {
        witness
            .values
            .get(variable)
            .map(|value| u64::from(*value))
            .ok_or(ModeBRelationError::R1csVariableOutOfBounds {
                variable,
                variable_count: witness.values.len(),
            })
    };
    let lc = |linear: &LinearCombination| -> Result<u64, ModeBRelationError> {
        let mut result = linear.constant.rem_euclid(i64::from(KOALABEAR_MODULUS)) as u64;
        for (variable, coefficient) in &linear.terms {
            result = (result
                + (*coefficient).rem_euclid(i64::from(KOALABEAR_MODULUS)) as u64
                    * value(*variable)?)
                % u64::from(KOALABEAR_MODULUS);
        }
        Ok(result)
    };
    for (row, constraint) in rows.iter().enumerate() {
        if lc(&constraint.a)? * lc(&constraint.b)? % u64::from(KOALABEAR_MODULUS)
            != lc(&constraint.c)?
        {
            return Err(ModeBRelationError::UnsatisfiedUnifiedRow { row });
        }
    }
    Ok(())
}

fn lc_one_minus(variable: usize) -> LinearCombination {
    LinearCombination::constant(1).add(variable, -1)
}
