#![no_std]
#![warn(missing_docs)]

//! AArch64 adapter for the architecture-neutral ERT loop scheduler.
//!
//! The adapter keeps A64 decoding and state ownership in `cirrus-aarch64-ert`;
//! this crate provides boundary snapshots, concrete agreement, and predicated
//! folding for the shared candidate scheduler.

use cirrus_aarch64_ert::State;
use cirrus_ert_loop_core::predicated_value;

/// A symbolic A64 loop-boundary snapshot.
#[derive(Clone)]
pub struct Aarch64Snapshot<W> {
    /// Full facade state, including GPRs, SP, NZCV, and done.
    pub state: State<W>,
    /// Concrete fetch address at the boundary.
    pub pc: u64,
}

/// Capture a boundary snapshot.
pub fn capture<W: Clone>(state: &State<W>, pc: u64) -> Aarch64Snapshot<W> {
    Aarch64Snapshot {
        state: state.clone(),
        pc,
    }
}

/// Restore a boundary snapshot into `state`.
pub fn restore<W: Clone>(snapshot: &Aarch64Snapshot<W>, state: &mut State<W>) {
    state.clone_from(&snapshot.state);
}

/// Fold an executed candidate body into an accumulated state under `active`.
///
/// Concrete metadata survives only when both sides agree; disagreement makes
/// that fact unknown rather than choosing a host-side value.
pub fn fold_snapshot<C, W>(
    context: &mut C,
    active: &W,
    candidate: &Aarch64Snapshot<W>,
    accumulated: &mut Aarch64Snapshot<W>,
    zero: &W,
) -> Result<(), Aarch64FoldError<C::Error>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: Clone,
{
    if accumulated.state.sp != candidate.state.sp {
        return Err(Aarch64FoldError::ConcreteStateDivergence);
    }
    for register in 0..31 {
        fold_word(
            context,
            active,
            &candidate.state.regs[register],
            &mut accumulated.state.regs[register],
            zero,
        )?;
        if accumulated.state.constants[register] != candidate.state.constants[register] {
            accumulated.state.constants[register] = None;
        }
    }
    fold_word(
        context,
        active,
        &candidate.state.sp_word,
        &mut accumulated.state.sp_word,
        zero,
    )?;
    for flag in 0..4 {
        accumulated.state.nzcv[flag] = predicated_value(
            context,
            active.clone(),
            candidate.state.nzcv[flag].clone(),
            accumulated.state.nzcv[flag].clone(),
        )
        .map_err(Aarch64FoldError::Context)?;
        if accumulated.state.nzcv_constants[flag] != candidate.state.nzcv_constants[flag] {
            accumulated.state.nzcv_constants[flag] = None;
        }
    }
    accumulated.state.done = predicated_value(
        context,
        active.clone(),
        candidate.state.done.clone(),
        accumulated.state.done.clone(),
    )
    .map_err(Aarch64FoldError::Context)?;
    Ok(())
}

fn fold_word<C, W>(
    context: &mut C,
    active: &W,
    candidate: &[W; 64],
    accumulated: &mut [W; 64],
    _zero: &W,
) -> Result<(), Aarch64FoldError<C::Error>>
where
    C: cirrus_core::ContextWithValue<bool, Wrapped = W>
        + cirrus_core::ContextWithBitAnd<bool, Wrapped = W>
        + cirrus_core::ContextWithBitOr<bool, Wrapped = W>
        + cirrus_core::ContextWithBitXor<bool, Wrapped = W>,
    W: Clone,
{
    for (old, new) in accumulated.iter_mut().zip(candidate) {
        *old = predicated_value(context, active.clone(), new.clone(), old.clone())
            .map_err(Aarch64FoldError::Context)?;
    }
    Ok(())
}

/// A fold failed, either because a context gate failed or concrete A64 state
/// diverged across candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Aarch64FoldError<E> {
    /// The Boolean context rejected a folding gate.
    Context(E),
    /// Concrete SP diverged between live candidate bodies.
    ConcreteStateDivergence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_merges_symbolic_bits_and_forgets_divergent_metadata() {
        let mut accumulated = capture(
            &cirrus_aarch64_ert::initial_state_with_arguments(false, &true, 64, &[Some(1)])
                .unwrap(),
            0,
        );
        let mut candidate = accumulated.clone();
        candidate.state.regs[0][0] = true;
        candidate.state.constants[0] = Some(3);
        accumulated.state.constants[0] = Some(1);
        fold_snapshot(&mut (), &true, &candidate, &mut accumulated, &false).unwrap();
        assert_eq!(accumulated.state.regs[0][0], true);
        assert_eq!(accumulated.state.constants[0], None);
        let mut inactive = accumulated.clone();
        inactive.state.constants[1] = Some(7);
        fold_snapshot(&mut (), &false, &inactive, &mut accumulated, &false).unwrap();
        assert_eq!(accumulated.state.regs[0][0], true);
        assert_eq!(accumulated.state.constants[0], None);
    }

    #[test]
    fn folding_rejects_divergent_sp() {
        let mut accumulated = capture(
            &cirrus_aarch64_ert::initial_state_with_arguments(false, &true, 64, &[]).unwrap(),
            0,
        );
        let mut candidate = accumulated.clone();
        candidate.state.sp = Some(128);
        assert_eq!(
            fold_snapshot(&mut (), &true, &candidate, &mut accumulated, &false),
            Err(Aarch64FoldError::ConcreteStateDivergence)
        );
    }
}
