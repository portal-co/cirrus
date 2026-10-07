//! Locked VOLE handlers: shared `&` contexts, circuit-granularity threads,
//! and Boolar banks over `&LockedStorage`.

use std::sync::Mutex;

use cipher::consts::U1;
use cirrus_core::{
    ContextWithBitAnd, ContextWithCreate, ContextWithCreateByRef, Pusher, SharedContext,
    StorageAddressBit,
};
use cirrus_recompile_core::Recorder;
use cirrus_volar_boolar::{StorageBank, execute as execute_boolar};
use cirrus_volar_vole::{
    LockedVoleProverContext, LockedVoleProverStorage, LockedVoleProverStorageContext,
    LockedVoleVerifierContext, LockedVoleVerifierStorage, LockedVoleVerifierStorageContext,
    MutexPuller, MutexPusher, StorageReadWitness, StorageWriteWitness, VoleStorageConfig,
};
use hybrid_array::Array;
use volar_ir::boolar::{BIrStmt, LaneId};
use volar_ir::circuit::BCircuit;
use volar_ir::ir::IRVarId;
use volar_ir_common::{Node, StorageId};
use volar_spec::{
    field::Galois128,
    vole::{Delta, Q, Vope},
};

struct VecPusher<T>(Vec<T>);
impl<T> Default for VecPusher<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}
impl<T> Pusher<T> for VecPusher<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

fn bit_to_t(b: bool) -> Galois128 {
    Galois128(b as u128)
}

fn delta() -> Delta<U1, Galois128> {
    Delta {
        delta: Array::from_fn(|_| Galois128(23)),
    }
}

fn and_program(left: bool, right: bool) -> cirrus_recompile_core::Program {
    let mut recorder = Recorder::new();
    let a = recorder.create(left).unwrap();
    let b = recorder.create(right).unwrap();
    let out = recorder.bitand(a, b).unwrap();
    recorder.finish(vec![], vec![out])
}

fn assert_share(
    prover: &Vope<U1, Galois128, U1>,
    verifier: &Q<U1, Galois128>,
    delta: &Delta<U1, Galois128>,
) {
    assert!(
        prover.clone() * delta.clone() == *verifier,
        "prover/verifier share mismatch"
    );
}

fn storage_config() -> VoleStorageConfig<Galois128> {
    VoleStorageConfig {
        address_coefficients: vec![Galois128(7)],
        r0: Galois128(11),
        r1: Galois128(13),
        r2: Galois128(17),
        r3: Galois128(19),
        timestamp_to_t: |timestamp| Galois128(timestamp as u128),
    }
}

fn verifier_bit(value: bool, delta: Galois128) -> Q<U1, Galois128> {
    Q {
        q: Array::from_fn(|_| if value { delta } else { Galois128(0) }),
    }
}

const STORAGE: StorageId = StorageId(0);
const LANE: LaneId = LaneId(0);

fn write_then_read_circuit() -> BCircuit {
    BCircuit {
        params: 1,
        stmts: [
            BIrStmt::One,
            BIrStmt::StorageWrite {
                storage: STORAGE,
                lane: LANE,
                src: IRVarId(0),
                addr: vec![IRVarId(1)],
            },
            BIrStmt::StorageRead {
                storage: STORAGE,
                lane: LANE,
                addr: vec![IRVarId(1)],
            },
        ]
        .into_iter()
        .map(|stmt| Node::new(stmt, (), None))
        .collect(),
        pre_init: vec![],
        outputs: vec![IRVarId(3)],
    }
}

#[test]
fn sequential_shared_reference_runs_two_programs() {
    let program_true = and_program(true, true);
    let program_false = and_program(true, false);
    let hats = MutexPusher::new(VecPusher::default());
    let prover = LockedVoleProverContext {
        hats: &hats,
        bit_to_t,
    };
    let prover_true = cirrus_recompile_rt::execute(&mut &prover, &program_true, &[]).unwrap();
    let prover_false = cirrus_recompile_rt::execute(&mut &prover, &program_false, &[]).unwrap();
    drop(prover);
    let hats = hats.into_inner();

    let puller = MutexPuller::new(hats.0.into_iter());
    let verifier = LockedVoleVerifierContext {
        delta: delta(),
        hats: &puller,
    };
    let verifier_true = cirrus_recompile_rt::execute(&mut &verifier, &program_true, &[]).unwrap();
    let verifier_false = cirrus_recompile_rt::execute(&mut &verifier, &program_false, &[]).unwrap();
    let d = delta();
    assert_share(&prover_true[0], &verifier_true[0], &d);
    assert_share(&prover_false[0], &verifier_false[0], &d);
    assert_eq!(prover_true[0].u[0][0], bit_to_t(true));
    assert_eq!(prover_false[0].u[0][0], bit_to_t(false));
}

#[test]
fn threaded_circuit_granularity_preserves_recorded_order() {
    let programs = [and_program(true, true), and_program(true, false)];
    let hats = MutexPusher::new(VecPusher::default());
    let prover = LockedVoleProverContext {
        hats: &hats,
        bit_to_t,
    };
    let order = Mutex::new(Vec::new());
    let circuit_lock = Mutex::new(());
    let prover_outputs = Mutex::new([None, None]);
    std::thread::scope(|scope| {
        for index in 0..2 {
            let prover = &prover;
            let programs = &programs;
            let order = &order;
            let circuit_lock = &circuit_lock;
            let prover_outputs = &prover_outputs;
            scope.spawn(move || {
                let _guard = circuit_lock.lock().unwrap();
                order.lock().unwrap().push(index);
                let output =
                    cirrus_recompile_rt::execute(&mut &*prover, &programs[index], &[]).unwrap();
                prover_outputs.lock().unwrap()[index] = Some(output);
            });
        }
    });
    drop(prover);
    let hats = hats.into_inner();
    let order = order.into_inner().unwrap();
    let prover_outputs = prover_outputs.into_inner().unwrap();
    assert_eq!(order.len(), 2);

    let puller = MutexPuller::new(hats.0.into_iter());
    let verifier = LockedVoleVerifierContext {
        delta: delta(),
        hats: &puller,
    };
    let d = delta();
    for index in order {
        let verifier_output =
            cirrus_recompile_rt::execute(&mut &verifier, &programs[index], &[]).unwrap();
        let prover_output = prover_outputs[index].as_ref().unwrap();
        assert_share(&prover_output[0], &verifier_output[0], &d);
    }
}

#[test]
fn boolar_shared_context_uses_locked_storage_banks() {
    let circuit = write_then_read_circuit();
    let hats = MutexPusher::new(VecPusher::default());
    let ctx = LockedVoleProverStorageContext {
        inner: LockedVoleProverContext {
            hats: &hats,
            bit_to_t,
        },
    };
    let prover_zero = ctx.create_by_ref(false).unwrap();
    let prover_one = ctx.create_by_ref(true).unwrap();
    let storage = LockedVoleProverStorage::new(
        prover_zero.clone(),
        prover_one.clone(),
        storage_config(),
        [
            StorageReadWitness {
                value: prover_one.clone(),
                predecessor_timestamp: 1,
            },
            StorageReadWitness {
                value: prover_zero.clone(),
                predecessor_timestamp: 3,
            },
        ],
        [
            StorageWriteWitness {
                overwritten: prover_zero.clone(),
                predecessor_timestamp: 0,
            },
            StorageWriteWitness {
                overwritten: prover_one.clone(),
                predecessor_timestamp: 2,
            },
        ],
    );
    let address = [StorageAddressBit {
        wire: prover_one.clone(),
        known: Some(true),
    }];
    storage
        .initialize(&address, prover_zero.clone(), 0)
        .unwrap();

    let mut shared = SharedContext(&ctx);
    {
        let mut slot: &LockedVoleProverStorage<U1, Galois128> = &storage;
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut slot,
        }];
        let outputs =
            execute_boolar(&mut shared, &circuit, &[prover_one.clone()], &mut banks).unwrap();
        assert_eq!(outputs[0].u[0][0], bit_to_t(true));
    }
    {
        let mut slot: &LockedVoleProverStorage<U1, Galois128> = &storage;
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut slot,
        }];
        let outputs =
            execute_boolar(&mut shared, &circuit, &[prover_zero.clone()], &mut banks).unwrap();
        assert_eq!(outputs[0].u[0][0], bit_to_t(false));
    }
    storage.drain(&address, prover_zero.clone(), 4).unwrap();
    let opening = storage.finish().unwrap();
    drop(shared);
    drop(ctx);
    let hats = hats.into_inner();

    let delta_value = Galois128(23);
    let puller = MutexPuller::new(hats.0.into_iter());
    let verifier_ctx = LockedVoleVerifierStorageContext {
        inner: LockedVoleVerifierContext {
            delta: delta(),
            hats: &puller,
        },
    };
    let verifier_zero = verifier_bit(false, delta_value);
    let verifier_one = verifier_bit(true, delta_value);
    let verifier_storage = LockedVoleVerifierStorage::new(
        verifier_zero.clone(),
        verifier_one.clone(),
        storage_config(),
        [
            StorageReadWitness {
                value: verifier_one.clone(),
                predecessor_timestamp: 1,
            },
            StorageReadWitness {
                value: verifier_zero.clone(),
                predecessor_timestamp: 3,
            },
        ],
        [
            StorageWriteWitness {
                overwritten: verifier_zero.clone(),
                predecessor_timestamp: 0,
            },
            StorageWriteWitness {
                overwritten: verifier_one.clone(),
                predecessor_timestamp: 2,
            },
        ],
    );
    let verifier_address = [StorageAddressBit {
        wire: verifier_one.clone(),
        known: Some(true),
    }];
    verifier_storage
        .initialize(&verifier_address, verifier_zero.clone(), 0)
        .unwrap();
    let mut verifier_shared = SharedContext(&verifier_ctx);
    {
        let mut slot: &LockedVoleVerifierStorage<U1, Galois128> = &verifier_storage;
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut slot,
        }];
        execute_boolar(
            &mut verifier_shared,
            &circuit,
            &[verifier_one.clone()],
            &mut banks,
        )
        .unwrap();
    }
    {
        let mut slot: &LockedVoleVerifierStorage<U1, Galois128> = &verifier_storage;
        let mut banks = [StorageBank {
            storage: STORAGE,
            lane: LANE,
            address_bits: 1,
            value: &mut slot,
        }];
        execute_boolar(
            &mut verifier_shared,
            &circuit,
            &[verifier_zero.clone()],
            &mut banks,
        )
        .unwrap();
    }
    verifier_storage
        .drain(&verifier_address, verifier_zero, 4)
        .unwrap();
    verifier_storage.finish(&opening).unwrap();
}
