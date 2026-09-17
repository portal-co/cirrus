use cirrus_armv8m_ert::{ArmHandler, ErtError, RawMemory};
use cirrus_ert_loop_core::{CandidateDriver, DriveError};

use crate::{ThumbBoundary, ThumbSnapshot, execute_snapshot, fold_registers, merge_agreement};

/// Fixed-capacity Thumb candidate body driver.
///
/// This is the ISA adapter boundary for the shared scheduler. It owns the
/// current symbolic snapshot and folds each body's wire-backed registers,
/// while concrete SP/rstack/ITSTATE/TrustZone state is checked for agreement.
pub struct ThumbGenerationDriver<'a, H, W, E, const FRAMES: usize>
where
    H: ArmHandler<bool, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
{
    handler: &'a mut H,
    storage: &'a mut H::Storage,
    storage_bits: usize,
    mem: RawMemory<'a>,
    rstack: &'a mut [u32],
    zero: W,
    one: W,
    base: ThumbSnapshot<W, FRAMES>,
    accumulator: ThumbSnapshot<W, FRAMES>,
    agreement: Option<crate::ThumbAgreement<FRAMES>>,
    any_exit: bool,
}

impl<'a, H, W, E, const FRAMES: usize> ThumbGenerationDriver<'a, H, W, E, FRAMES>
where
    H: ArmHandler<bool, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
{
    /// Create a driver from the common state at the beginning of a generation.
    pub fn new(
        handler: &'a mut H,
        storage: &'a mut H::Storage,
        storage_bits: usize,
        mem: RawMemory<'a>,
        rstack: &'a mut [u32],
        base: ThumbSnapshot<W, FRAMES>,
        zero: W,
        one: W,
    ) -> Self {
        Self {
            handler,
            storage,
            storage_bits,
            mem,
            rstack,
            zero,
            one,
            accumulator: base.clone(),
            base,
            agreement: None,
            any_exit: false,
        }
    }

    /// Finish the generation and return its folded snapshot.
    pub fn finish(self) -> Result<ThumbSnapshot<W, FRAMES>, ErtError<E>> {
        if self.any_exit {
            Ok(self.accumulator)
        } else {
            Ok(self.accumulator)
        }
    }
}

impl<'a, H, W, E, const FRAMES: usize> CandidateDriver<u32>
    for ThumbGenerationDriver<'a, H, W, E, FRAMES>
where
    H: ArmHandler<bool, Wrapped = W, Error = E> + ?Sized,
    W: Clone,
    E: core::error::Error,
{
    type Error = ErtError<E>;

    fn execute(
        &mut self,
        candidate: u32,
        successors: &mut dyn FnMut(u32) -> Result<(), DriveError<Self::Error>>,
    ) -> Result<(), DriveError<Self::Error>> {
        let mut body = self.base.clone();
        body.pc = candidate;
        let (boundary, updated) = execute_snapshot(
            self.handler,
            self.storage,
            self.storage_bits,
            self.mem,
            self.rstack,
            body,
            self.zero.clone(),
            self.one.clone(),
        )
        .map_err(DriveError::Driver)?;
        merge_agreement(&mut self.agreement, updated.agreement())
            .map_err(|_| DriveError::Driver(ErtError::Unexpected))?;
        let active = self.one.clone();
        fold_registers(self.handler, active, &updated, &mut self.accumulator)
            .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
        match boundary {
            ThumbBoundary::Branch { .. } => {
                for successor in boundary.successors().expect("branch has successors") {
                    successors(u32::from(successor))?;
                }
            }
            ThumbBoundary::Exit => self.any_exit = true,
        }
        Ok(())
    }
}
