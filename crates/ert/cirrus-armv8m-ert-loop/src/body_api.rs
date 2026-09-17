use crate::execute_body;
use cirrus_armv8m_ert::{ArmHandler, Machine, StorageRuntime};

use crate::{ThumbBoundary, ThumbSnapshot, ThumbSnapshotError};

/// Execute one Thumb body from a caller-owned snapshot and return the updated
/// snapshot alongside its symbolic control-flow boundary.
///
/// This is the adapter's first complete state-transfer API. It keeps storage,
/// the Boolean handler, return-stack backing, and all symbolic wires caller
/// owned; no allocator or hidden machine state is introduced.
#[allow(clippy::too_many_arguments)]
pub fn execute_snapshot<H, W, E, const FRAMES: usize>(
    handler: &mut H,
    storage: &mut H::Storage,
    storage_bits: usize,
    mem: cirrus_armv8m_ert::RawMemory<'_>,
    rstack: &mut [u32],
    snapshot: ThumbSnapshot<W, FRAMES>,
    zero: W,
    one: W,
) -> Result<(ThumbBoundary<W>, ThumbSnapshot<W, FRAMES>), cirrus_armv8m_ert::ErtError<E>>
where
    H: ArmHandler<bool, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
{
    if snapshot.rsp > rstack.len() {
        return Err(cirrus_armv8m_ert::ErtError::Unexpected);
    }
    let mut regs = snapshot.regs;
    let mut constants = snapshot.constants;
    let mut runtime = StorageRuntime::new(handler, storage, zero.clone(), one.clone());
    let mut machine = Machine::new(
        &mut runtime,
        mem,
        rstack,
        storage_bits,
        snapshot.pc,
        &mut regs,
        &mut constants,
        zero,
        one,
        snapshot.sp,
    );
    machine.offsets = snapshot.offsets;
    machine.flags = snapshot.flags;
    machine.stack_top = snapshot.stack_top;
    machine.rsp = snapshot.rsp;
    machine.itstate = snapshot.itstate;
    machine.security_state = snapshot.security_state;
    machine.rstack[..snapshot.rsp].copy_from_slice(&snapshot.rstack[..snapshot.rsp]);

    let boundary = execute_body(&mut machine)?;
    let updated =
        ThumbSnapshot::capture(&machine).map_err(|ThumbSnapshotError::ReturnStackCapacity| {
            cirrus_armv8m_ert::ErtError::Unexpected
        })?;
    Ok((boundary, updated))
}
