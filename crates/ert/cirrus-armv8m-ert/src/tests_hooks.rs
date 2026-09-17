extern crate std;

use core::convert::Infallible;
use std::vec::Vec;

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};
use cirrus_ert_core::{EcallOutcome, Handler};

use crate::{
    ArmCallAction, ArmCallEvent, ArmHandler, ErtError, RawMemory, SecurityAttribute, SecurityState,
    ert_emit,
};

struct TestHandler {
    action: ArmCallAction,
    events: Vec<ArmCallEvent>,
}

impl HasError for TestHandler {
    type Error = Infallible;
}

impl ContextWithValue<bool> for TestHandler {
    type Wrapped = bool;
}

impl ContextWithBitAnd<bool> for TestHandler {
    fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left & right)
    }

    fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left &= right;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for TestHandler {
    fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left | right)
    }

    fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left |= right;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for TestHandler {
    fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left ^ right)
    }

    fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left ^= right;
        Ok(())
    }
}

impl ContextWithStorage<bool> for TestHandler {
    type Storage = [bool];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Infallible> {
        let index = address.iter().enumerate().fold(0, |index, (bit, part)| {
            index | ((part.wire as usize) << bit)
        });
        Ok(storage[index])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Infallible> {
        let index = address.iter().enumerate().fold(0, |index, (bit, part)| {
            index | ((part.wire as usize) << bit)
        });
        storage[index] = value;
        Ok(())
    }
}

impl Handler<bool> for TestHandler {
    fn ecall(
        &mut self,
        _regs: &mut [[bool; 32]],
        constants: &mut [Option<u64>],
        _offsets: &mut [Option<i64>],
        _zero: &bool,
        _one: &bool,
    ) -> Result<EcallOutcome, Infallible> {
        match constants[0] {
            Some(u64::MAX) | Some(0xffff_ffff) => Ok(EcallOutcome::Exit),
            _ => Ok(EcallOutcome::Unexpected),
        }
    }
}

impl ArmHandler<bool> for TestHandler {
    fn svc_permitted(&mut self, _state: SecurityState) -> bool {
        true
    }

    fn security_attribute(&mut self, _address: u32) -> SecurityAttribute {
        SecurityAttribute::Secure
    }

    fn call_hook(
        &mut self,
        event: ArmCallEvent,
        _regs: &mut [[bool; 32]],
        _constants: &mut [Option<u32>],
        _offsets: &mut [Option<i32>],
        _zero: &bool,
        _one: &bool,
    ) -> Result<ArmCallAction, Infallible> {
        self.events.push(event);
        Ok(self.action)
    }
}

fn image(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn run(
    handler: &mut TestHandler,
    code: &[u16],
    constants: &mut [Option<u32>; 16],
) -> Result<(), ErtError<Infallible>> {
    let image = image(code);
    let mut regs = [[false; 32]; 16];
    let mut rstack = [0; 2];
    let mut storage = [false; 64];
    let storage_bits = storage.len();
    ert_emit(
        handler,
        &mut storage,
        storage_bits,
        RawMemory::from(image.as_slice()),
        &mut rstack,
        1,
        &mut regs,
        constants,
        false,
        true,
    )
}

#[test]
fn direct_call_return_now_skips_the_callee_and_does_not_push_a_frame() {
    // `BL 8; movs r0, #0; subs r0, #1; svc #0; udf`. The hook makes the
    // invalid target unreachable and continuation must resume at byte 4.
    let mut handler = TestHandler {
        action: ArmCallAction::ReturnNow,
        events: Vec::new(),
    };
    let mut constants = [None; 16];
    assert!(
        run(
            &mut handler,
            &[0xf000, 0xf802, 0x2000, 0x3801, 0xdf00, 0xbe00],
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(
        handler.events,
        [ArmCallEvent::DirectCall {
            caller_pc: 0,
            target: 8,
            return_pc: 4,
        }]
    );
}

#[test]
fn unresolved_register_call_is_offered_to_the_hook_before_failing_closed() {
    // `BLX r3; movs r0, #0; subs r0, #1; svc #0` with symbolic r3.
    let mut handler = TestHandler {
        action: ArmCallAction::ReturnNow,
        events: Vec::new(),
    };
    let mut constants = [None; 16];
    assert!(
        run(
            &mut handler,
            &[0x4798, 0x2000, 0x3801, 0xdf00],
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(
        handler.events,
        [ArmCallEvent::RegisterCall {
            caller_pc: 0,
            register: 3,
            target: None,
            return_pc: 2,
        }]
    );
}

#[test]
fn invalid_diverted_register_call_fails_closed() {
    let mut handler = TestHandler {
        action: ArmCallAction::Divert(3),
        events: Vec::new(),
    };
    let mut constants = [None; 16];
    assert!(matches!(
        run(&mut handler, &[0x4798], &mut constants),
        Err(ErtError::Unexpected)
    ));
}
