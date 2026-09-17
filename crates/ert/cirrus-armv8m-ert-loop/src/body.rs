use cirrus_armv8m_ert::Machine;

/// The symbolic control-flow boundary reached by one Thumb body.
#[derive(Clone)]
pub enum ThumbBoundary<W> {
    /// A conditional branch whose condition is represented by `condition`.
    Branch {
        /// The branch condition, true for the taken target.
        condition: W,
        /// Normalized even taken target.
        taken: u32,
        /// Normalized even fall-through target.
        fallthrough: u32,
    },
    /// The guest exited through `SVC #0`.
    Exit,
}

impl<W> ThumbBoundary<W> {
    /// Return the concrete successor fetch addresses in architectural order.
    ///
    /// The condition wire is intentionally not inspected here; the caller
    /// feeds it into its state/virtual-IP mux while retaining this stable
    /// taken-then-fallthrough ordering.
    pub fn successors(&self) -> Option<[u32; 2]> {
        match self {
            Self::Branch {
                taken, fallthrough, ..
            } => Some([*taken, *fallthrough]),
            Self::Exit => None,
        }
    }
}

///
/// Concrete branches, calls, returns, and IT-predicated instructions continue
/// inline exactly through [`Machine::execute_decoded`]. A conditional `B` or
/// `CBZ`/`CBNZ` whose condition cannot be decided from the machine's concrete
/// metadata closes the body and returns a symbolic boundary. The machine's
/// `pc` remains at the boundary instruction; callers choose the next virtual
/// candidate and snapshot/fold policy.
pub fn execute_body<W, E>(
    machine: &mut Machine<'_, W, E>,
) -> Result<ThumbBoundary<W>, cirrus_armv8m_ert::ErtError<E>>
where
    W: Clone,
    E: core::error::Error,
{
    loop {
        let decoded = machine.decode()?;
        if let Some((target, Some(condition))) = decoded.operation.branch_info() {
            if machine.condition_value(condition)?.is_none() {
                let condition = machine.condition_wire(condition)?;
                return Ok(ThumbBoundary::Branch {
                    condition,
                    taken: target,
                    fallthrough: machine.pc.wrapping_add(decoded.len),
                });
            }
        }
        if let Some((register, nonzero, target)) = decoded.operation.compare_branch_info() {
            if machine.constants[register as usize].is_none() {
                let zero = machine.zero.clone();
                let zero_word = core::array::from_fn(|_| zero.clone());
                let condition = cirrus_ert_core::compare_word(
                    &mut *machine.t,
                    &machine.regs[register as usize],
                    &zero_word,
                    if nonzero {
                        cirrus_ert_core::ComparePredicate::Ne
                    } else {
                        cirrus_ert_core::ComparePredicate::Eq
                    },
                    &machine.one,
                )
                .map_err(cirrus_armv8m_ert::ErtError::Emitted)?;
                let _ = zero;
                return Ok(ThumbBoundary::Branch {
                    condition,
                    taken: target,
                    fallthrough: machine.pc.wrapping_add(decoded.len),
                });
            }
        }
        match machine.execute_decoded(decoded)? {
            cirrus_armv8m_ert::Flow::Next(next) => machine.pc = next,
            cirrus_armv8m_ert::Flow::Exit => return Ok(ThumbBoundary::Exit),
        }
    }
}
