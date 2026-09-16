extern crate std;

use core::{array, convert::Infallible};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};
use rv_asm::{Imm, Inst, Reg, Xlen};
use std::vec::Vec;

use crate::{
    CallAction, CallEvent, DefaultHandler, EcallOutcome, ErtError, Handler, RawMemory,
    RvDefaultHandler, RvHandler, ert64_emit,
};

/// A test handler that forwards everything to an inner handler and routes
/// call boundaries to a closure. Generic over the inner context so the same
/// wrapper serves the native `bool` backend and the recording backend.
struct Hooked<C, F, H> {
    inner: DefaultHandler<C, F>,
    hook: H,
}

impl<C: HasError, F, H> HasError for Hooked<C, F, H> {
    type Error = C::Error;
}

impl<C: ContextWithValue<bool>, F, H> ContextWithValue<bool> for Hooked<C, F, H> {
    type Wrapped = C::Wrapped;
}

impl<C: ContextWithBitAnd<bool>, F, H> ContextWithBitAnd<bool> for Hooked<C, F, H> {
    fn bitand(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitand(a, b)
    }

    fn bitand_assign(
        &mut self,
        a: &mut C::Wrapped,
        b: C::Wrapped,
    ) -> Result<(), C::Error> {
        self.inner.bitand_assign(a, b)
    }
}

impl<C: ContextWithBitOr<bool>, F, H> ContextWithBitOr<bool> for Hooked<C, F, H> {
    fn bitor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitor(a, b)
    }

    fn bitor_assign(
        &mut self,
        a: &mut C::Wrapped,
        b: C::Wrapped,
    ) -> Result<(), C::Error> {
        self.inner.bitor_assign(a, b)
    }
}

impl<C: ContextWithBitXor<bool>, F, H> ContextWithBitXor<bool> for Hooked<C, F, H> {
    fn bitxor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.inner.bitxor(a, b)
    }

    fn bitxor_assign(
        &mut self,
        a: &mut C::Wrapped,
        b: C::Wrapped,
    ) -> Result<(), C::Error> {
        self.inner.bitxor_assign(a, b)
    }
}

impl<C, F, H> ContextWithStorage<bool> for Hooked<C, F, H>
where
    C: ContextWithStorage<bool>,
{
    type Storage = C::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<C::Wrapped>],
    ) -> Result<C::Wrapped, C::Error> {
        self.inner.storage_read(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<C::Wrapped>],
        value: C::Wrapped,
    ) -> Result<(), C::Error> {
        self.inner.storage_write(storage, address, value)
    }
}

impl<C, F, H, const BITS: usize> Handler<bool, BITS> for Hooked<C, F, H>
where
    C: crate::ContextWithRvOps<bool>,
    F: FnMut(&mut C, &[[C::Wrapped; BITS]]) -> Result<[u8; 32], C::Error>,
    C::Wrapped: Clone,
    C::Error: core::error::Error,
{
    fn ecall(
        &mut self,
        regs: &mut [[C::Wrapped; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &C::Wrapped,
        one: &C::Wrapped,
    ) -> Result<EcallOutcome, C::Error> {
        self.inner.ecall(regs, reg_consts, offsets, zero, one)
    }
}

impl<C, F, H, const BITS: usize> RvHandler<bool, BITS> for Hooked<C, F, H>
where
    C: crate::ContextWithRvOps<bool>,
    F: FnMut(&mut C, &[[C::Wrapped; BITS]]) -> Result<[u8; 32], C::Error>,
    C::Wrapped: Clone,
    C::Error: core::error::Error,
    H: FnMut(
        CallEvent,
        &mut [[C::Wrapped; BITS]],
        &mut [Option<u64>],
        &mut [Option<i64>],
        &C::Wrapped,
        &C::Wrapped,
    ) -> Result<CallAction, C::Error>,
{
    fn call_hook(
        &mut self,
        event: CallEvent,
        regs: &mut [[C::Wrapped; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &C::Wrapped,
        one: &C::Wrapped,
    ) -> Result<CallAction, C::Error> {
        (self.hook)(event, regs, reg_consts, offsets, zero, one)
    }
}

fn word<const BITS: usize>(value: u64) -> [bool; BITS] {
    array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn value<const BITS: usize>(word: &[bool; BITS]) -> u64 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u64) << bit))
}

fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|instruction| instruction.encode_normal(Xlen::Rv64).to_le_bytes())
        .collect()
}

fn no_hash(_: &mut (), _: &[[bool; 64]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn run_hooked<H>(
    mem: &[u8],
    hook: H,
    regs: &mut [[bool; 64]; 32],
    constants: &mut [Option<u64>; 32],
) -> Result<(), ErtError<Infallible>>
where
    H: FnMut(
        CallEvent,
        &mut [[bool; 64]],
        &mut [Option<u64>],
        &mut [Option<i64>],
        &bool,
        &bool,
    ) -> Result<CallAction, Infallible>,
{
    let mut handler = Hooked {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
        hook,
    };
    let mut vstack = [false; 2048];
    let storage_bits = vstack.len();
    let mut rstack = [0u64; 16];
    ert64_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(mem),
        &mut rstack,
        0,
        regs,
        constants,
        false,
        true,
    )
}

fn exit_setup(regs: &mut [[bool; 64]; 32], constants: &mut [Option<u64>; 32]) {
    regs[Reg::A0.0 as usize] = word(0xffff_ffff);
    constants[Reg::A0.0 as usize] = Some(0xffff_ffff);
}

#[test]
fn observer_counts_calls_and_returns() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    let mem = program([
        Inst::Jal {
            offset: Imm::new_i32(8),
            dest: Reg::RA,
        },
        Inst::Ecall,
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let mut events = Vec::new();
    let result = run_hooked(
        &mem,
        |event, _, _, _, _, _| {
            events.push(event);
            Ok(CallAction::Proceed)
        },
        &mut regs,
        &mut constants,
    );
    assert!(result.is_ok());
    assert_eq!(
        events,
        [
            CallEvent::Jal {
                caller_pc: 0,
                target: 8,
                link: Reg::RA,
            },
            CallEvent::Return {
                from_pc: 8,
                target: 4,
            },
        ]
    );
}

#[test]
fn return_now_replaces_the_callee() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    let mem = program([
        // Marker set before the call.
        Inst::Addi {
            imm: Imm::new_i32(99),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Jal {
            offset: Imm::new_i32(12),
            dest: Reg::RA,
        },
        // The hook rewrote a1; the exit selector is installed after the call
        // because the hook is allowed to clobber a0.
        Inst::Addi {
            imm: Imm::new_i32(-1),
            dest: Reg::A0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        // Callee at 16: would set t0 = 7 and return.
        Inst::Addi {
            imm: Imm::new_i32(7),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let result = run_hooked(
        &mem,
        |event, regs, reg_consts, _, zero, one| {
            if let CallEvent::Jal { target: 16, .. } = event {
                // Replace the callee: a1 = 42.
                reg_consts[Reg::A1.0 as usize] = Some(42);
                regs[Reg::A1.0 as usize] =
                    array::from_fn(|bit| if (42u64 >> bit) & 1 == 0 { *zero } else { *one });
                return Ok(CallAction::ReturnNow);
            }
            Ok(CallAction::Proceed)
        },
        &mut regs,
        &mut constants,
    );
    assert!(result.is_ok());
    // The callee never ran (t0 keeps its marker from before the call), and
    // the hook's result registers hold.
    assert_eq!(value(&regs[Reg::T0.0 as usize]), 99);
    assert_eq!(value(&regs[Reg::A1.0 as usize]), 42);
}

#[test]
fn divert_redirects_a_call_to_a_stub() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    let mem = program([
        Inst::Jal {
            offset: Imm::new_i32(8),
            dest: Reg::RA,
        },
        Inst::Ecall,
        // Original callee at 8: sets t0 = 1.
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
        // Diversion stub at 16: sets t1 = 7.
        Inst::Addi {
            imm: Imm::new_i32(7),
            dest: Reg::T1,
            src1: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);
    let result = run_hooked(
        &mem,
        |event, _, _, _, _, _| {
            if let CallEvent::Jal { target: 8, .. } = event {
                return Ok(CallAction::Divert(16));
            }
            Ok(CallAction::Proceed)
        },
        &mut regs,
        &mut constants,
    );
    assert!(result.is_ok());
    assert_eq!(value(&regs[Reg::T0.0 as usize]), 0);
    assert_eq!(value(&regs[Reg::T1.0 as usize]), 7);
}

#[test]
fn unresolved_jalr_fails_closed_by_default_and_diverts_on_request() {
    // t0 holds a symbolic target (no concrete metadata), so the jalr is
    // unresolved.
    let program_image = program([
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::T0,
            dest: Reg::RA,
        },
        Inst::Ecall,
        // Diversion target at 8: mark t1, return.
        Inst::Addi {
            imm: Imm::new_i32(5),
            dest: Reg::T1,
            src1: Reg::ZERO,
        },
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
    ]);

    // Default: historical fail-closed.
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    regs[Reg::T0.0 as usize] = word(8); // symbolic: no concrete metadata
    let result = run_hooked(
        &program_image,
        |_, _, _, _, _, _| Ok(CallAction::Proceed),
        &mut regs,
        &mut constants,
    );
    assert!(matches!(result, Err(ErtError::Unexpected)));

    // Hooked: resolve to the stub at 8.
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    regs[Reg::T0.0 as usize] = word(8);
    let result = run_hooked(
        &program_image,
        |event, _, _, _, _, _| {
            if let CallEvent::UnresolvedJalr { .. } = event {
                return Ok(CallAction::Divert(8));
            }
            Ok(CallAction::Proceed)
        },
        &mut regs,
        &mut constants,
    );
    assert!(result.is_ok());
    assert_eq!(value(&regs[Reg::T1.0 as usize]), 5);
}

#[test]
fn divert_on_return_overrides_the_landing() {
    let mut regs = [[false; 64]; 32];
    let mut constants = [None; 32];
    exit_setup(&mut regs, &mut constants);
    let mem = program([
        Inst::Jal {
            offset: Imm::new_i32(12),
            dest: Reg::RA,
        },
        Inst::Addi {
            imm: Imm::new_i32(1),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
        // Callee at 12: return.
        Inst::Jalr {
            offset: Imm::ZERO,
            base: Reg::RA,
            dest: Reg::ZERO,
        },
        // Diverted landing at 16: set t0 = 2, then exit.
        Inst::Addi {
            imm: Imm::new_i32(2),
            dest: Reg::T0,
            src1: Reg::ZERO,
        },
        Inst::Ecall,
    ]);
    let result = run_hooked(
        &mem,
        |event, _, _, _, _, _| {
            if let CallEvent::Return { .. } = event {
                return Ok(CallAction::Divert(16));
            }
            Ok(CallAction::Proceed)
        },
        &mut regs,
        &mut constants,
    );
    assert!(result.is_ok());
    assert_eq!(value(&regs[Reg::T0.0 as usize]), 2);
}

#[cfg(feature = "call-hooks")]
mod registry {
    use super::*;
    use crate::hooks::{CallRegistry, CallReplacement, apply_replacement};

    #[test]
    fn registry_replaces_registered_targets_only() {
        let mut registry = CallRegistry::new();
        registry.insert(16, CallReplacement::ReturnConstants { a0: 7, a1: 9 });

        let mut regs = [[false; 64]; 32];
        let mut constants = [None; 32];
        exit_setup(&mut regs, &mut constants);
        let mem = program([
            // Marker set before the call.
            Inst::Addi {
                imm: Imm::new_i32(99),
                dest: Reg::T0,
                src1: Reg::ZERO,
            },
            Inst::Jal {
                offset: Imm::new_i32(12),
                dest: Reg::RA,
            },
            // The registry clobbers a0/a1; reinstall the exit selector.
            Inst::Addi {
                imm: Imm::new_i32(-1),
                dest: Reg::A0,
                src1: Reg::ZERO,
            },
            Inst::Ecall,
            // Callee at 16: would set t0 = 5.
            Inst::Addi {
                imm: Imm::new_i32(5),
                dest: Reg::T0,
                src1: Reg::ZERO,
            },
            Inst::Jalr {
                offset: Imm::ZERO,
                base: Reg::RA,
                dest: Reg::ZERO,
            },
        ]);
        let result = run_hooked(
            &mem,
            |event, regs, reg_consts, offsets, zero, one| {
                if let Some(replacement) = registry.lookup(&event) {
                    return Ok(apply_replacement(
                        replacement, regs, reg_consts, offsets, zero, one,
                    ));
                }
                Ok(CallAction::Proceed)
            },
            &mut regs,
            &mut constants,
        );
        assert!(result.is_ok());
        // The callee never ran; the registry wrote the results, and the
        // program reinstalled the exit selector over a0.
        assert_eq!(value(&regs[Reg::T0.0 as usize]), 99);
        assert_eq!(value(&regs[Reg::A1.0 as usize]), 9);
    }
}

#[cfg(feature = "prepared-recording")]
mod recorder {
    use cirrus_recompile_core::{Idx, Op, Recorder};
    use cirrus_volar_boolar::MuxTreeContext;

    use super::*;

    /// A handler whose hook emits a real gate (a0 = a1 ^ a2) through the
    /// recording context, so the prepared artifact contains the hook's work.
    struct GateEmittingHandler {
        context: MuxTreeContext<Recorder>,
    }

    impl HasError for GateEmittingHandler {
        type Error = Infallible;
    }

    impl ContextWithValue<bool> for GateEmittingHandler {
        type Wrapped = Idx;
    }

    impl ContextWithBitAnd<bool> for GateEmittingHandler {
        fn bitand(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
            self.context.bitand(a, b)
        }

        fn bitand_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
            self.context.bitand_assign(a, b)
        }
    }

    impl ContextWithBitOr<bool> for GateEmittingHandler {
        fn bitor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
            self.context.bitor(a, b)
        }

        fn bitor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
            self.context.bitor_assign(a, b)
        }
    }

    impl ContextWithBitXor<bool> for GateEmittingHandler {
        fn bitxor(&mut self, a: Idx, b: Idx) -> Result<Idx, Infallible> {
            self.context.bitxor(a, b)
        }

        fn bitxor_assign(&mut self, a: &mut Idx, b: Idx) -> Result<(), Infallible> {
            self.context.bitxor_assign(a, b)
        }
    }

    impl ContextWithStorage<bool> for GateEmittingHandler {
        type Storage = [Idx];

        fn storage_read(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<Idx>],
        ) -> Result<Idx, Infallible> {
            self.context.storage_read(storage, address)
        }

        fn storage_write(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<Idx>],
            value: Idx,
        ) -> Result<(), Infallible> {
            self.context.storage_write(storage, address, value)
        }
    }

    impl Handler<bool, 64> for GateEmittingHandler {
        fn ecall(
            &mut self,
            _: &mut [[Idx; 64]],
            _: &mut [Option<u64>],
            _: &mut [Option<i64>],
            _: &Idx,
            _: &Idx,
        ) -> Result<EcallOutcome, Infallible> {
            Ok(EcallOutcome::Exit)
        }
    }

    impl RvHandler<bool, 64> for GateEmittingHandler {
        fn call_hook(
            &mut self,
            event: CallEvent,
            regs: &mut [[Idx; 64]],
            reg_consts: &mut [Option<u64>],
            offsets: &mut [Option<i64>],
            zero: &Idx,
            one: &Idx,
        ) -> Result<CallAction, Infallible> {
            if let CallEvent::Jal { .. } = event {
                // a0 = a1 ^ a2, one gate per bit, through the recorder.
                for bit in 0..64 {
                    regs[Reg::A0.0 as usize][bit] = self.context.bitxor(
                        regs[Reg::A1.0 as usize][bit],
                        regs[Reg::A2.0 as usize][bit],
                    )?;
                }
                reg_consts[Reg::A0.0 as usize] = None;
                offsets[Reg::A0.0 as usize] = None;
                let _ = (zero, one);
                return Ok(CallAction::ReturnNow);
            }
            Ok(CallAction::Proceed)
        }
    }

    /// A hook emission is just gates on the handler's context: the recorder
    /// records them like any other instruction's.
    #[test]
    fn hook_gate_emissions_are_recorded() {
        let mut handler = GateEmittingHandler {
            context: MuxTreeContext::new(Recorder::new()),
        };
        let mem = program([
            Inst::Jal {
                offset: Imm::new_i32(12),
                dest: Reg::RA,
            },
            Inst::Ecall,
            Inst::Addi {
                imm: Imm::new_i32(1),
                dest: Reg::T0,
                src1: Reg::ZERO,
            },
            // Callee at 12: never runs.
            Inst::Addi {
                imm: Imm::new_i32(5),
                dest: Reg::A0,
                src1: Reg::ZERO,
            },
            Inst::Jalr {
                offset: Imm::ZERO,
                base: Reg::RA,
                dest: Reg::ZERO,
            },
        ]);
        let mut regs = [[Idx(0); 64]; 32];
        let mut constants = [None; 32];
        let mut vstack = [Idx(0); 2048];
        let storage_bits = vstack.len();
        let mut rstack = [0u64; 16];
        // Wire inputs: a1 = 3, a2 = 5 as recorded constants.
        let zero = handler.context.create(false).unwrap();
        let one = handler.context.create(true).unwrap();
        for cell in &mut vstack {
            *cell = zero;
        }
        for regs_row in &mut regs {
            *regs_row = [zero; 64];
        }
        use cirrus_core::ContextWithCreate;
        for bit in 0..64 {
            regs[Reg::A1.0 as usize][bit] = if (3u64 >> bit) & 1 == 0 { zero } else { one };
            regs[Reg::A2.0 as usize][bit] = if (5u64 >> bit) & 1 == 0 { zero } else { one };
        }
        constants[Reg::A1.0 as usize] = Some(3);
        constants[Reg::A2.0 as usize] = Some(5);

        ert64_emit(
            &mut handler,
            &mut vstack,
            storage_bits,
            RawMemory::from(&mem[..]),
            &mut rstack,
            0,
            &mut regs,
            &mut constants,
            zero,
            one,
        )
        .map_err(|_: ErtError<Infallible>| ())
        .expect("hooked program runs");

        // The hook's 64 XOR gates must be in the recorded trace.
        let recorder = handler.context.into_inner();
        let program = recorder.finish(std::vec![], std::vec![]);
        let xor_gates = program
            .ops
            .iter()
            .filter(|op| matches!(op, Op::BitXor(..)))
            .count();
        assert_eq!(xor_gates, 64, "one recorded XOR gate per result bit");
        // The callee's ADDI ran no gates (constant path), so nothing else
        // was recorded.
        assert_eq!(program.ops.len(), 2 + 64);
    }
}
