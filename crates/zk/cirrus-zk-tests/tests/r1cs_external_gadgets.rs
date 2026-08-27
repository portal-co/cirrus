//! Constrained external Boolar callbacks use caller-supplied gadgets and
//! preserve direct action storage/fallback semantics.

use ark_bn254::Fr;
use ark_r1cs_std::{eq::EqGadget, prelude::*};
use ark_relations::gr1cs::{ConstraintSystem, ConstraintSystemRef, SynthesisError};
use cirrus_r1cs_backend::{
    R1csBackend,
    circuit::{ExternalPrimitiveGadgets, R1csExternalBitRegistry},
};
use cirrus_volar_boolar::{StorageBank, execute_with_externals};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::{Node, StorageId};

const STORAGE: StorageId = StorageId(3);
const LANE: LaneId = LaneId(1);

struct BitGadgets;

impl ExternalPrimitiveGadgets<Fr> for BitGadgets {
    fn external_binding(&self) -> Vec<Fr> {
        vec![Fr::from(1u64)]
    }

    fn oracle_bit(
        &self,
        _cs: ConstraintSystemRef<Fr>,
        name: &str,
        args: &[Boolean<Fr>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<Fr>, SynthesisError> {
        assert_eq!((name, bit, occurrence), ("lookup", 0, 4));
        Ok(args[0].clone())
    }

    fn rng_bit(
        &self,
        _cs: ConstraintSystemRef<Fr>,
        name: &str,
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<Fr>, SynthesisError> {
        assert_eq!((name, bit, occurrence), ("nonce", 1, 5));
        Ok(Boolean::constant(true))
    }

    fn action_bit(
        &self,
        _cs: ConstraintSystemRef<Fr>,
        name: &str,
        args: &[Boolean<Fr>],
        bit: usize,
        occurrence: u64,
    ) -> Result<Boolean<Fr>, SynthesisError> {
        assert_eq!((name, bit, occurrence), ("commit", 0, 6));
        Ok(args[0].clone())
    }
}

fn circuit() -> BCircuit {
    BCircuit {
        params: 4,
        stmts: vec![
            Node::new(
                BIrStmt::OracleBit {
                    name: "lookup".into(),
                    args: vec![IRVarId(1)],
                    bit: 0,
                    occurrence: 4,
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::RngBit {
                    name: "nonce".into(),
                    bit: 1,
                    occurrence: 5,
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::ActionStoreBit {
                    name: "commit".into(),
                    guard: IRVarId(0),
                    args: vec![IRVarId(4), IRVarId(5)],
                    fallback: IRVarId(3),
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(2)],
                    bit: 0,
                    occurrence: 6,
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr: vec![IRVarId(2)],
                },
                (),
                None,
            ),
        ],
        pre_init: vec![],
        outputs: vec![IRVarId(7)],
    }
}

fn prove_execution(guard: bool, fallback: bool, expected: bool) {
    let cs = ConstraintSystem::<Fr>::new_ref();
    let mut backend = R1csBackend::new(cs.clone());
    let inputs = [guard, true, true, fallback]
        .into_iter()
        .map(|bit| Boolean::new_witness(cs.clone(), || Ok(bit)))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut storage = [
        Boolean::new_witness(cs.clone(), || Ok(false)).unwrap(),
        Boolean::new_witness(cs.clone(), || Ok(false)).unwrap(),
    ];
    let mut banks = [StorageBank {
        storage: STORAGE,
        lane: LANE,
        address_bits: 1,
        value: &mut storage[..],
    }];
    let gadgets = BitGadgets;
    let mut registry = R1csExternalBitRegistry::new(&gadgets);
    let output =
        execute_with_externals(&mut backend, &circuit(), &inputs, &mut banks, &mut registry)
            .unwrap();
    output[0]
        .enforce_equal(&Boolean::constant(expected))
        .unwrap();
    assert!(cs.is_satisfied().unwrap());
}

#[test]
fn external_r1cs_gadgets_dispatch_and_action_storage_selects_guard_or_fallback() {
    prove_execution(true, false, true);
    prove_execution(false, false, false);
}
