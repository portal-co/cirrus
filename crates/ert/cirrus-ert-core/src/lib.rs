#![no_std]
#![warn(missing_docs)]

//! Shared implementation support for the symbolic execution facades.
//!
//! This crate deliberately exposes only the plumbing needed by architecture
//! facades. It is not a stable, standalone interpreter API. Both facades use
//! the same little-endian symbolic word representation: bit zero is at index
//! zero, and an optional `u32` travels alongside each word when it is known.

use core::{array, marker::PhantomData, mem::MaybeUninit};

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor};
use volar_circuit_exec_core::{SelectEmitter, select as emit_select};

mod compare;
#[cfg(test)]
mod tests;

pub use compare::{ComparePredicate, compare_word};

/// The Boolean operations needed by the shared symbolic-word machinery.
pub trait ContextWithErtOps<Val>:
    ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>
{
}

impl<Val, T: ContextWithBitAnd<Val> + ContextWithBitOr<Val> + ContextWithBitXor<Val>>
    ContextWithErtOps<Val> for T
{
}

/// Adapts an ERT Boolean context to the shared select-emission seam.
///
/// ERT receives its zero/one wires from the caller ABI, so it intentionally
/// implements only the AND/XOR interface needed for selection rather than
/// fabricating a constant constructor.
pub struct ErtSelectEmitter<'a, T: ?Sized> {
    context: &'a mut T,
}

impl<'a, T: ?Sized> ErtSelectEmitter<'a, T> {
    /// Borrow one ERT context for shared Boolean select emission.
    pub fn new(context: &'a mut T) -> Self {
        Self { context }
    }
}

impl<T, W: Clone, E> SelectEmitter for ErtSelectEmitter<'_, T>
where
    T: ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized,
{
    type Wire = W;
    type Error = E;

    fn and(&mut self, left: W, right: W) -> Result<W, E> {
        self.context.bitand(left, right)
    }

    fn xor(&mut self, left: W, right: W) -> Result<W, E> {
        self.context.bitxor(left, right)
    }
}

/// A handler for a facade's environment-call instruction (RV32 `ECALL`,
/// Armv8-M `SVC #0`), and any others a caller adds.
///
/// `regs`/`reg_consts`/`offsets` are slices over the full register file at
/// the call — sliced rather than fixed-size-array-typed so this one trait
/// shape serves both RV32's 32 registers and Armv8-M's 16 registers. An
/// implementation is expected to index only known ABI-fixed positions (e.g.
/// "register 0" and "the eight registers following it"); nothing about a
/// well-behaved `ecall` implementation needs the total register count.
/// `zero`/`one` are the caller's symbolic Boolean constants. Returning `Err`
/// aborts execution with a caller-emitted error; return
/// [`EcallOutcome::Unexpected`] instead to reject just this particular call
/// (e.g. an unrecognized register-zero value) without fabricating one.
///
/// The caller-balanced-stack requirement for a successful exit is enforced
/// by each facade's own interpreter, not by the handler.
pub trait Handler<Val>: ContextWithErtOps<Val> {
    /// Handle an environment call.
    fn ecall(
        &mut self,
        regs: &mut [[Self::Wrapped; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &Self::Wrapped,
        one: &Self::Wrapped,
    ) -> Result<EcallOutcome, Self::Error>;

    /// Options for the opt-in secret-dependent early-exit loop
    /// deoptimization (see [`EarlyExitLoopOptions`]).
    ///
    /// Defaults to disabled, which preserves today's behavior exactly: a
    /// facade's `branch`/`condition` handling hard-errors the moment it
    /// meets a symbolic condition it cannot resolve concretely.
    fn early_exit_loop_options(&self) -> EarlyExitLoopOptions {
        EarlyExitLoopOptions::default()
    }
}

/// Options controlling the opt-in "deoptimize secret-dependent early-exit
/// loops" recognizer: instead of hard-erroring the moment a facade meets a
/// branch it cannot resolve concretely, it may recognize a narrow,
/// provably-safe idiom (a concrete-bounded loop with exactly one
/// secret-dependent early exit that only ever writes a constant/loop-invariant
/// value) and replay it as an always-runs-every-iteration loop with the
/// exit folded in via [`select_word`], instead of failing.
///
/// Anything outside that narrow idiom keeps hard-erroring exactly as before
/// — this is purely an additive relaxation, never a silent miscompile risk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EarlyExitLoopOptions {
    /// Attempt the recognizer instead of hard-erroring on a symbolic branch.
    pub enabled: bool,
    /// Bound on how many instructions the recognizer will statically scan
    /// while classifying a candidate loop body, before giving up and falling
    /// through to the ordinary hard error.
    pub max_lookahead_instructions: u16,
}

impl Default for EarlyExitLoopOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_lookahead_instructions: 256,
        }
    }
}

/// The effect of a handled environment call on control flow.
pub enum EcallOutcome {
    /// Continue execution at the next instruction.
    Continue,
    /// Exit the program, once the interpreter confirms the stack is balanced.
    Exit,
    /// This specific call (e.g. an unrecognized register-zero value) is invalid.
    Unexpected,
}

/// A read-only byte mapping whose base is guest address zero.
///
/// A bounded mapping is normally made with [`RawMemory::from_slice`]. The
/// unbounded constructor is for bare-metal images where guest virtual address
/// zero intentionally has no valid host slice representation.
#[derive(Clone, Copy)]
pub struct RawMemory<'a> {
    base: *const u8,
    len: Option<usize>,
    detect: Option<(u32, u32)>,
    marker: PhantomData<&'a [u8]>,
}

impl<'a> RawMemory<'a> {
    /// Borrow `memory` as a bounded mapping rooted at guest address zero.
    pub fn from_slice(memory: &'a [u8]) -> Self {
        Self {
            base: memory.as_ptr(),
            len: Some(memory.len()),
            detect: None,
            marker: PhantomData,
        }
    }

    /// Overlay a 4-byte little-endian word read at `address` with `value`, in
    /// place of the underlying backing bytes. This lets guest code that reads
    /// its real (e.g. zero-initialized) value when run natively detect
    /// whether it is running under this interpreter instead.
    pub fn with_ert_detect(mut self, address: u32, value: u32) -> Self {
        self.detect = Some((address, value));
        self
    }

    /// Read a fixed number of bytes, rejecting an overflowing or out-of-range
    /// address before any pointer is dereferenced.
    #[doc(hidden)]
    pub fn read<const N: usize>(&self, address: u32) -> Option<[u8; N]> {
        debug_assert!(N > 0);
        address.checked_add(N.checked_sub(1)? as u32)?;
        if N == 4 {
            if let Some((detect_address, value)) = self.detect {
                if address == detect_address {
                    let bytes = value.to_le_bytes();
                    return Some(array::from_fn(|i| bytes[i]));
                }
            }
        }
        if let Some(len) = self.len {
            let start = address as usize;
            if start.checked_add(N)? > len {
                return None;
            }
        }
        Some(array::from_fn(|offset| {
            // SAFETY: bounded ranges were checked above. For an unbounded
            // mapping, this is precisely the contract of `RawMemory::new`.
            unsafe { self.base.wrapping_add(address as usize + offset).read() }
        }))
    }
}

impl<'a> From<&'a [u8]> for RawMemory<'a> {
    fn from(memory: &'a [u8]) -> Self {
        Self::from_slice(memory)
    }
}

impl RawMemory<'static> {
    /// Create an unbounded or explicitly bounded raw guest-memory mapping.
    ///
    /// `base` represents guest address zero and may be null. With `Some(len)`,
    /// every byte in `base..base + len` must be readable. With `None`, every
    /// fetched instruction byte and every concrete-load byte reached by the
    /// guest must be readable for the execution's duration.
    ///
    /// # Safety
    ///
    /// The caller must uphold the corresponding readability guarantee. A
    /// supplied bound supplies interpreter checks; it cannot validate `base`.
    pub const unsafe fn new(base: *const u8, len: Option<usize>) -> Self {
        Self {
            base,
            len,
            detect: None,
            marker: PhantomData,
        }
    }
}

/// The three bitwise operations used by canonical word handlers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitOp {
    /// Bitwise conjunction.
    And,
    /// Bitwise disjunction.
    Or,
    /// Bitwise exclusive-or.
    Xor,
}

/// The supported 32-bit shift directions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Shift {
    /// Shift towards more-significant bits.
    Left,
    /// Shift right and insert zeroes.
    LogicalRight,
    /// Shift right and replicate the sign bit.
    ArithmeticRight,
    /// Rotate right, including the bits displaced at the low end.
    RotateRight,
}

/// The four RV32/Arm scalar product views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Product {
    /// Low 32 bits of the product.
    Low,
    /// High 32 bits of a signed product.
    HighSigned,
    /// High 32 bits of signed-left, unsigned-right product.
    HighSignedUnsigned,
    /// High 32 bits of an unsigned product.
    HighUnsigned,
}

/// Form a symbolic constant word.
pub fn constant_word<W: Clone>(zero: &W, one: &W, value: u32) -> [W; 32] {
    array::from_fn(|bit| {
        if (value >> bit) & 1 == 0 {
            zero.clone()
        } else {
            one.clone()
        }
    })
}

/// Add fixed-width little-endian words with an initial carry through a
/// caller-supplied Boolean gate constructor.
///
/// This is the architecture-neutral core of the symbolic adder.  ERT routes
/// it through a [`ContextWithErtOps`] implementation below, while frontends
/// that retain concrete-bit metadata can use it with a simplifying gate
/// constructor and still share the exact carry circuit.
pub fn add_bits_with<W: Clone, E, const N: usize>(
    v: &[W; N],
    w: &[W; N],
    mut carry: W,
    mut gate: impl FnMut(BitOp, W, W) -> Result<W, E>,
) -> Result<[W; N], E> {
    let mut output: [MaybeUninit<W>; N] = [const { MaybeUninit::uninit() }; N];
    for i in 0..N {
        let without_carry = gate(BitOp::Xor, v[i].clone(), w[i].clone())?;
        output[i] = MaybeUninit::new(gate(BitOp::Xor, without_carry, carry.clone())?);
        let a = gate(BitOp::And, v[i].clone(), w[i].clone())?;
        let b = gate(BitOp::And, v[i].clone(), carry.clone())?;
        let c = gate(BitOp::And, w[i].clone(), carry.clone())?;
        let remaining_pairs = gate(BitOp::Or, b, c)?;
        carry = gate(BitOp::Or, a, remaining_pairs)?;
    }
    Ok(output.map(|value| unsafe { value.assume_init() }))
}

/// Add fixed-width little-endian words with an initial carry.
pub fn add_bits<W: Clone, E, const N: usize>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    v: &[W; N],
    w: &[W; N],
    carry: W,
) -> Result<[W; N], E> {
    add_bits_with(v, w, carry, |operation, left, right| match operation {
        BitOp::And => t.bitand(left, right),
        BitOp::Or => t.bitor(left, right),
        BitOp::Xor => t.bitxor(left, right),
    })
}

/// Add two 32-bit words.
pub fn add_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; 32],
    right: &[W; 32],
    zero: W,
) -> Result<[W; 32], E> {
    add_bits(t, left, right, zero)
}

/// Select `then_word` when `condition` is set, else select `else_word`.
///
/// The implementation is the circuit identity `else ^ (condition & (then ^
/// else))`, so no mux capability is required from the Boolean context.
pub fn select_word<W: Clone, E, const N: usize>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    condition: W,
    then_word: &[W; N],
    else_word: &[W; N],
) -> Result<[W; N], E> {
    let mut emitter = ErtSelectEmitter::new(t);
    try_array(|bit| {
        emit_select(
            &mut emitter,
            condition.clone(),
            then_word[bit].clone(),
            else_word[bit].clone(),
        )
    })
}

/// Apply a bitwise operation to every bit of two 32-bit words.
pub fn bitwise_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; 32],
    right: &[W; 32],
    operation: BitOp,
) -> Result<[W; 32], E> {
    try_array(|bit| match operation {
        BitOp::And => t.bitand(left[bit].clone(), right[bit].clone()),
        BitOp::Or => t.bitor(left[bit].clone(), right[bit].clone()),
        BitOp::Xor => t.bitxor(left[bit].clone(), right[bit].clone()),
    })
}

/// AND/OR/XOR a host-known `constant` against a symbolic word, hardcoding
/// every bit the constant alone determines with zero `t.bitand`/`t.bitor`
/// calls for AND/OR. XOR is accepted for call-site uniformity only — it
/// still calls `t.bitxor` once per set bit, identical in cost to
/// `bitwise_word`, since XOR has no constant-side shortcut to exploit.
pub fn partial_bitwise_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    constant: u32,
    symbolic: &[W; 32],
    zero: &W,
    one: &W,
    operation: BitOp,
) -> Result<[W; 32], E> {
    try_array(|bit| {
        let set = (constant >> bit) & 1 != 0;
        Ok(match operation {
            BitOp::And if set => symbolic[bit].clone(),
            BitOp::And => zero.clone(),
            BitOp::Or if set => one.clone(),
            BitOp::Or => symbolic[bit].clone(),
            BitOp::Xor if set => t.bitxor(symbolic[bit].clone(), one.clone())?,
            BitOp::Xor => symbolic[bit].clone(),
        })
    })
}

/// `constant & !symbolic`, for BitClear's concrete-left-operand case (not
/// expressible via [`partial_bitwise_word`], since it needs a per-bit
/// inverted copy rather than a direct copy).
pub fn partial_and_not_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    constant: u32,
    symbolic: &[W; 32],
    zero: &W,
    one: &W,
) -> Result<[W; 32], E> {
    try_array(|bit| {
        if (constant >> bit) & 1 == 0 {
            Ok(zero.clone())
        } else {
            t.bitxor(symbolic[bit].clone(), one.clone())
        }
    })
}

/// Invert every bit of a word using the supplied Boolean one wire.
pub fn invert_word<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    word: &[W; 32],
    one: W,
) -> Result<[W; 32], E> {
    try_array(|bit| t.bitxor(word[bit].clone(), one.clone()))
}

/// Shift or rotate a snapshot by a host-known amount.
pub fn fixed_shift<W: Clone>(source: &[W; 32], amount: u32, direction: Shift, zero: &W) -> [W; 32] {
    let amount = amount.min(32);
    let fill = match direction {
        Shift::ArithmeticRight => source[31].clone(),
        Shift::Left | Shift::LogicalRight | Shift::RotateRight => zero.clone(),
    };
    array::from_fn(|destination_bit| match direction {
        Shift::Left if destination_bit >= amount as usize => {
            source[destination_bit - amount as usize].clone()
        }
        Shift::LogicalRight | Shift::ArithmeticRight
            if destination_bit + (amount as usize) < 32 =>
        {
            source[destination_bit + amount as usize].clone()
        }
        Shift::RotateRight => source[(destination_bit + (amount as usize & 31)) & 31].clone(),
        _ => fill.clone(),
    })
}

/// Compute a five-stage RV32-style symbolic barrel shift.
pub fn rv32_runtime_shift<W: Clone, E>(
    t: &mut (impl ContextWithErtOps<bool, Wrapped = W, Error = E> + ?Sized),
    source: &[W; 32],
    amount: &[W; 32],
    direction: Shift,
    zero: &W,
) -> Result<[W; 32], E> {
    let mut output = source.clone();
    for stage in 0..5 {
        let candidate = fixed_shift(&output, 1 << stage, direction, zero);
        output = select_word(t, amount[stage].clone(), &candidate, &output)?;
    }
    Ok(output)
}

/// Compute a host-known 32-bit product view.
pub fn concrete_product(product: Product, left: u32, right: u32) -> u32 {
    match product {
        Product::Low => left.wrapping_mul(right),
        Product::HighSigned => (((left as i32 as i64) * (right as i32 as i64)) >> 32) as u32,
        Product::HighSignedUnsigned => {
            (((left as i32 as i64) * (right as u64 as i64)) >> 32) as u32
        }
        Product::HighUnsigned => ((left as u64 * right as u64) >> 32) as u32,
    }
}

fn try_array<T, E, const N: usize>(mut f: impl FnMut(usize) -> Result<T, E>) -> Result<[T; N], E> {
    let mut output: [MaybeUninit<T>; N] = [const { MaybeUninit::uninit() }; N];
    for index in 0..N {
        match f(index) {
            Ok(value) => output[index].write(value),
            Err(error) => {
                for value in output.iter_mut().take(index) {
                    // SAFETY: the loop has initialized exactly these elements.
                    unsafe { value.assume_init_drop() };
                }
                return Err(error);
            }
        };
    }
    // SAFETY: every element was initialized by the loop above.
    Ok(unsafe { (&output as *const _ as *const [T; N]).read() })
}
