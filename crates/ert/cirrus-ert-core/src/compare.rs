//! A symbolic 32-bit word comparison, materialized as a single Boolean wire.
//!
//! No facade needed this before: branches always required both operands to
//! already be concrete (see each facade's `branch`/`condition` handling), so
//! there was never a reason to turn a comparison into circuit output. The
//! early-exit-loop deoptimization changes that — it needs to fold a
//! secret-dependent comparison into an accumulator via [`select_word`]
//! instead of branching on it.

use core::mem::MaybeUninit;

use crate::{BitOp, ContextWithErtOps, bitwise_word, invert_word};

/// The six RV32/Armv8-M branch/compare conditions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparePredicate {
    /// `left == right`.
    Eq,
    /// `left != right`.
    Ne,
    /// `left >= right`, unsigned.
    GeU,
    /// `left < right`, unsigned.
    LtU,
    /// `left >= right`, signed.
    GeS,
    /// `left < right`, signed.
    LtS,
}

/// Evaluate `predicate` between two symbolic words, returning a single
/// Boolean wire (`one` if the predicate holds, `zero` otherwise).
///
/// Built entirely from [`ContextWithErtOps`]'s AND/OR/XOR gates, matching
/// every other primitive in this crate — no mux capability is required.
pub fn compare_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; 32],
    right: &[W; 32],
    predicate: ComparePredicate,
    one: &W,
) -> Result<W, E> {
    match predicate {
        ComparePredicate::Eq | ComparePredicate::Ne => {
            let xor = bitwise_word(t, left, right, BitOp::Xor)?;
            let differs = or_reduce(t, &xor)?;
            match predicate {
                ComparePredicate::Ne => Ok(differs),
                _ => t.bitxor(differs, one.clone()),
            }
        }
        ComparePredicate::GeU | ComparePredicate::LtU => {
            let (_, carry_out) = subtract_with_carry_out(t, left, right, one)?;
            match predicate {
                ComparePredicate::GeU => Ok(carry_out),
                _ => t.bitxor(carry_out, one.clone()),
            }
        }
        ComparePredicate::GeS | ComparePredicate::LtS => {
            let (diff, _) = subtract_with_carry_out(t, left, right, one)?;
            let sign_l = left[31].clone();
            let sign_r = right[31].clone();
            let sign_d = diff[31].clone();
            let operands_differ = t.bitxor(sign_l.clone(), sign_r)?;
            let result_differs_from_left = t.bitxor(sign_l, sign_d.clone())?;
            let overflow = t.bitand(operands_differ, result_differs_from_left)?;
            let signed_lt = t.bitxor(sign_d, overflow)?;
            match predicate {
                ComparePredicate::LtS => Ok(signed_lt),
                _ => t.bitxor(signed_lt, one.clone()),
            }
        }
    }
}

/// `left - right` via two's-complement addition, plus the top-bit carry-out
/// (unsigned "no borrow occurred" flag) that [`crate::add_bits_with`] doesn't
/// expose. Mirrors each facade's own subtract handler, just also keeping the
/// carry rather than discarding it.
fn subtract_with_carry_out<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; 32],
    right: &[W; 32],
    one: &W,
) -> Result<([W; 32], W), E> {
    let inverted_right = invert_word(t, right, one.clone())?;
    let mut carry = one.clone();
    let mut output: [MaybeUninit<W>; 32] = [const { MaybeUninit::uninit() }; 32];
    for i in 0..32 {
        let without_carry = t.bitxor(left[i].clone(), inverted_right[i].clone())?;
        output[i] = MaybeUninit::new(t.bitxor(without_carry, carry.clone())?);
        let a = t.bitand(left[i].clone(), inverted_right[i].clone())?;
        let b = t.bitand(left[i].clone(), carry.clone())?;
        let c = t.bitand(inverted_right[i].clone(), carry.clone())?;
        let remaining_pairs = t.bitor(b, c)?;
        carry = t.bitor(a, remaining_pairs)?;
    }
    // SAFETY: every element was initialized by the loop above.
    Ok((output.map(|value| unsafe { value.assume_init() }), carry))
}

fn or_reduce<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    bits: &[W; 32],
) -> Result<W, E> {
    let mut acc = bits[0].clone();
    for bit in &bits[1..] {
        acc = t.bitor(acc, bit.clone())?;
    }
    Ok(acc)
}
