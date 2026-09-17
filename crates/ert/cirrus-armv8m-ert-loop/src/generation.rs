use cirrus_armv8m_ert::{ArmHandler, ErtError, RawMemory};
use cirrus_ert_core::{ComparePredicate, compare_word};
use cirrus_ert_loop_core::{
    CandidateDriver, CandidateTable, DriveError, drive_generation, predicated_value,
};

use crate::{
    ThumbBoundary, ThumbSnapshot, execute_snapshot, fold_flags, fold_registers, merge_agreement,
};

/// Fixed-capacity Thumb candidate body driver.
///
/// This is the ISA adapter boundary for the shared scheduler. It owns the
/// current symbolic snapshot and folds each body's wire-backed registers,
/// while concrete SP/rstack/ITSTATE/TrustZone state is checked for agreement.
pub struct ThumbGenerationDriver<'a, H, W, E, const FRAMES: usize>
where
    H: ArmHandler<bool, Wrapped = W, Error = E>,
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
    vip: [W; 32],
    next_vip: [W; 32],
    done: W,
}

impl<'a, H, W, E, const FRAMES: usize> ThumbGenerationDriver<'a, H, W, E, FRAMES>
where
    H: ArmHandler<bool, Wrapped = W, Error = E>,
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
        vip: [W; 32],
    ) -> Self {
        Self {
            handler,
            storage,
            storage_bits,
            mem,
            rstack,
            zero: zero.clone(),
            one,
            accumulator: base.clone(),
            base,
            agreement: None,
            any_exit: false,
            vip: vip.clone(),
            next_vip: vip,
            done: zero.clone(),
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

/// Execute one shared candidate generation with the Thumb adapter.
///
/// Candidate iteration, successor deduplication, and fixed-capacity overflow
/// handling are delegated to `cirrus-ert-loop-core`; the returned snapshot is
/// the adapter's folded state after every body has completed.
pub fn run_generation<'a, H, W, E, const FRAMES: usize>(
    table: &mut CandidateTable<'_, u32>,
    driver: ThumbGenerationDriver<'a, H, W, E, FRAMES>,
) -> Result<ThumbSnapshot<W, FRAMES>, ErtError<E>>
where
    H: ArmHandler<bool, Wrapped = W, Error = E>,
    W: Clone,
    E: core::error::Error,
{
    let mut driver = driver;
    drive_generation(table, &mut driver).map_err(|error| match error {
        DriveError::Driver(error) => error,
        DriveError::Table(_) => ErtError::Unexpected,
    })?;
    driver.finish()
}

impl<'a, H, W, E, const FRAMES: usize> CandidateDriver<u32>
    for ThumbGenerationDriver<'a, H, W, E, FRAMES>
where
    H: ArmHandler<bool, Wrapped = W, Error = E>,
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
        let candidate_word = core::array::from_fn(|bit| {
            if (u64::from(candidate) >> bit) & 1 == 0 {
                self.zero.clone()
            } else {
                self.one.clone()
            }
        });
        let active = compare_word(
            self.handler,
            &self.vip,
            &candidate_word,
            ComparePredicate::Eq,
            &self.one,
        )
        .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
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
        fold_registers(
            self.handler,
            active.clone(),
            &updated,
            &mut self.accumulator,
        )
        .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
        fold_flags(
            self.handler,
            active.clone(),
            &updated,
            &mut self.accumulator,
        )
        .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
        let body_done = match boundary {
            ThumbBoundary::Branch {
                taken,
                fallthrough,
                condition,
            } => {
                let taken_word: [W; 32] = core::array::from_fn(|bit| {
                    if (u64::from(taken) >> bit) & 1 == 0 {
                        self.zero.clone()
                    } else {
                        self.one.clone()
                    }
                });
                let fallthrough_word: [W; 32] = core::array::from_fn(|bit| {
                    if (u64::from(fallthrough) >> bit) & 1 == 0 {
                        self.zero.clone()
                    } else {
                        self.one.clone()
                    }
                });
                for bit in 0..32 {
                    let body_bit = predicated_value(
                        self.handler,
                        condition.clone(),
                        taken_word[bit].clone(),
                        fallthrough_word[bit].clone(),
                    )
                    .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
                    self.next_vip[bit] = predicated_value(
                        self.handler,
                        active.clone(),
                        body_bit,
                        self.next_vip[bit].clone(),
                    )
                    .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
                }
                successors(taken)?;
                successors(fallthrough)?;
                self.zero.clone()
            }
            ThumbBoundary::Exit => {
                self.any_exit = true;
                self.one.clone()
            }
        };
        self.done = predicated_value(self.handler, active, body_done, self.done.clone())
            .map_err(|error| DriveError::Driver(ErtError::Emitted(error)))?;
        Ok(())
    }
}
