//! A symbolic 32-bit word comparison, materialized as a single Boolean wire.
//!
//! Control-flow decisions still require concrete values, but RV32 `slt*`,
//! Arm status flags, and the early-exit-loop deoptimization all need a
//! comparison as circuit data. This module supplies that result without
//! weakening either facade's concrete-only branch discipline.

use crate::{add_bits_with_carry_out, bitwise_word, invert_word, BitOp, ContextWithErtOps};

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

/// Derive the signed-overflow flag for `left + right = result`.
pub fn add_overflow<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: W,
    right: W,
    result: W,
) -> Result<W, E> {
    let left_differs = t.bitxor(left, result.clone())?;
    let right_differs = t.bitxor(right, result)?;
    t.bitand(left_differs, right_differs)
}

/// Derive the signed-overflow flag for `left - right = result`.
pub fn subtract_overflow<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: W,
    right: W,
    result: W,
) -> Result<W, E> {
    let operands_differ = t.bitxor(left.clone(), right)?;
    let result_differs = t.bitxor(left, result)?;
    t.bitand(operands_differ, result_differs)
}

/// Evaluate an Arm NZCV condition from symbolic flag wires.
///
/// Returns `None` for the reserved condition encoding 15. Condition 14 is
/// the unconditional `AL` form and returns `one` without emitting a gate.
pub fn arm_condition<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    n: W,
    z: W,
    c: W,
    v: W,
    condition: u8,
    one: &W,
) -> Result<Option<W>, E> {
    let result = match condition {
        0 => z,
        1 => invert_boolean(t, z, one)?,
        2 => c,
        3 => invert_boolean(t, c, one)?,
        4 => n,
        5 => invert_boolean(t, n, one)?,
        6 => v,
        7 => invert_boolean(t, v, one)?,
        8 => {
            let not_z = invert_boolean(t, z, one)?;
            t.bitand(c, not_z)?
        }
        9 => {
            let not_c = invert_boolean(t, c, one)?;
            t.bitor(not_c, z)?
        }
        10 => {
            let differs = t.bitxor(n, v)?;
            invert_boolean(t, differs, one)?
        }
        11 => t.bitxor(n, v)?,
        12 => {
            let not_z = invert_boolean(t, z, one)?;
            let differs = t.bitxor(n, v)?;
            let equal = invert_boolean(t, differs, one)?;
            t.bitand(not_z, equal)?
        }
        13 => {
            let differs = t.bitxor(n, v)?;
            t.bitor(z, differs)?
        }
        14 => one.clone(),
        _ => return Ok(None),
    };
    Ok(Some(result))
}

fn invert_boolean<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    wire: W,
    one: &W,
) -> Result<W, E> {
    t.bitxor(wire, one.clone())
}

/// Evaluate an Arm NZCV condition using optional concrete flag facts.
///
/// The outer `None` denotes the reserved condition encoding 15. For a valid
/// condition, the inner value is `None` until every flag it observes is known.
pub fn arm_condition_value(
    n: Option<bool>,
    z: Option<bool>,
    c: Option<bool>,
    v: Option<bool>,
    condition: u8,
) -> Option<Option<bool>> {
    let value = match condition {
        0 => z,
        1 => z.map(|value| !value),
        2 => c,
        3 => c.map(|value| !value),
        4 => n,
        5 => n.map(|value| !value),
        6 => v,
        7 => v.map(|value| !value),
        8 => c.zip(z).map(|(c, z)| c && !z),
        9 => c.zip(z).map(|(c, z)| !c || z),
        10 => n.zip(v).map(|(n, v)| n == v),
        11 => n.zip(v).map(|(n, v)| n != v),
        12 => z.zip(n.zip(v)).map(|(z, (n, v))| !z && n == v),
        13 => z.zip(n.zip(v)).map(|(z, (n, v))| z || n != v),
        14 => Some(true),
        _ => return None,
    };
    Some(value)
}

/// Evaluate `predicate` between two symbolic words, returning a single
/// Boolean wire (`one` if the predicate holds, `zero` otherwise).
///
/// Built entirely from [`ContextWithErtOps`]'s AND/OR/XOR gates, matching
/// every other primitive in this crate — no mux capability is required.
pub fn compare_word<W: Clone, E, const N: usize>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; N],
    right: &[W; N],
    predicate: ComparePredicate,
    one: &W,
) -> Result<W, E> {
    match predicate {
        ComparePredicate::Eq | ComparePredicate::Ne => {
            let xor = bitwise_word(t, left, right, BitOp::Xor)?;
            let equal = zero_word(t, &xor, one)?;
            match predicate {
                ComparePredicate::Eq => Ok(equal),
                ComparePredicate::Ne => t.bitxor(equal, one.clone()),
                _ => unreachable!("the outer match limits equality predicates"),
            }
        }
        ComparePredicate::GeU | ComparePredicate::LtU => {
            let (_, carry_out) = subtract_word_with_carry_out(t, left, right, one)?;
            match predicate {
                ComparePredicate::GeU => Ok(carry_out),
                _ => t.bitxor(carry_out, one.clone()),
            }
        }
        ComparePredicate::GeS | ComparePredicate::LtS => {
            let (diff, _) = subtract_word_with_carry_out(t, left, right, one)?;
            let sign_l = left[N - 1].clone();
            let sign_r = right[N - 1].clone();
            let sign_d = diff[N - 1].clone();
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
pub fn subtract_word_with_carry_out<W: Clone, E, const N: usize>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; N],
    right: &[W; N],
    one: &W,
) -> Result<([W; N], W), E> {
    let inverted_right = invert_word(t, right, one.clone())?;
    add_bits_with_carry_out(t, left, &inverted_right, one.clone())
}

/// Return a Boolean wire that is one exactly when `word` is zero.
pub fn zero_word<W: Clone, E, const N: usize>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    word: &[W; N],
    one: &W,
) -> Result<W, E> {
    let mut acc = word[0].clone();
    for bit in &word[1..] {
        acc = t.bitor(acc, bit.clone())?;
    }
    t.bitxor(acc, one.clone())
}
