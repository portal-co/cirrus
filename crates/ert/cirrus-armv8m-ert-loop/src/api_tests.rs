extern crate std;

use core::{array, convert::Infallible};

use cirrus_armv8m_ert::{
    ArmHandler, EcallOutcome, FLAG_Z, Flag, FlagWire, Handler, RawMemory, SecurityAttribute,
    SecurityState,
};
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithStorage, ContextWithValue,
    HasError, StorageAddressBit,
};

use crate::{CandidateTable, initial_snapshot, step};

struct PlainHandler;

impl HasError for PlainHandler {
    type Error = Infallible;
}

impl ContextWithValue<bool> for PlainHandler {
    type Wrapped = bool;
}

impl ContextWithBitAnd<bool> for PlainHandler {
    fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left & right)
    }

    fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left &= right;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for PlainHandler {
    fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left | right)
    }

    fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left |= right;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for PlainHandler {
    fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Infallible> {
        Ok(left ^ right)
    }

    fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Infallible> {
        *left ^= right;
        Ok(())
    }
}

impl ContextWithStorage<bool> for PlainHandler {
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

impl Handler<bool> for PlainHandler {
    fn ecall(
        &mut self,
        _regs: &mut [[bool; 32]],
        constants: &mut [Option<u64>],
        _offsets: &mut [Option<i64>],
        _zero: &bool,
        _one: &bool,
    ) -> Result<EcallOutcome, Infallible> {
        if matches!(constants[0], Some(u64::MAX) | Some(0xffff_ffff)) {
            Ok(EcallOutcome::Exit)
        } else {
            Ok(EcallOutcome::Unexpected)
        }
    }
}

impl ArmHandler<bool> for PlainHandler {
    fn svc_permitted(&mut self, _state: SecurityState) -> bool {
        true
    }

    fn security_attribute(&mut self, _address: u32) -> SecurityAttribute {
        SecurityAttribute::Secure
    }
}

fn image(words: &[u16]) -> std::vec::Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

#[test]
fn symbolic_thumb_branch_advances_through_shared_candidate_generations() {
    // `bne 4; movs r0,#0; subs r0,#1; svc #0`.  The Z flag is symbolic,
    // so the first generation yields candidates 4 (taken) and 2
    // (fall-through). Both bodies then reach the standard exit SVC.
    let code = image(&[0xd100, 0x2000, 0x3801, 0xdf00]);
    let mut handler = PlainHandler;
    let mut storage = [false; 128];
    let mut rstack = [0u32; 4];
    let storage_bits = storage.len();
    let mut snapshot = initial_snapshot::<_, _, _, 4>(
        &mut handler,
        &mut storage,
        storage_bits,
        RawMemory::from(code.as_slice()),
        &mut rstack,
        1,
        false,
        true,
    )
    .unwrap_or_else(|_| panic!("Thumb loop step must succeed"));
    snapshot.constants[0] = Some(0);
    snapshot.regs[0] = word(0);
    snapshot.flags[FLAG_Z] = Flag {
        wire: FlagWire::Direct(false),
        value: None,
    };

    let mut backing = [0u32; 4];
    let mut candidates = CandidateTable::new(&mut backing, 0).unwrap();
    let first = step(
        &mut handler,
        &mut storage,
        storage_bits,
        RawMemory::from(code.as_slice()),
        &mut rstack,
        &mut candidates,
        snapshot,
        word(0),
        false,
        false,
        true,
    )
    .unwrap_or_else(|_| panic!("Thumb loop step must succeed"));
    assert_eq!(candidates.current(), &[4, 2]);
    assert!(!first.exited);

    let second = step(
        &mut handler,
        &mut storage,
        storage_bits,
        RawMemory::from(code.as_slice()),
        &mut rstack,
        &mut candidates,
        first.snapshot,
        first.virtual_ip,
        first.done,
        false,
        true,
    )
    .unwrap_or_else(|_| panic!("second Thumb loop step failed"));
    assert!(candidates.is_empty());
    assert!(second.exited);
    assert!(second.done);
}
