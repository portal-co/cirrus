#![no_std]
#![warn(missing_docs)]

//! Symbolically execute a deliberately small Armv8-M Thumb-2 subset,
//! including virtual Secure/Non-secure TrustZone-M interworking.
//!
//! The interpreter represents every architectural word as 32 little-endian
//! Boolean wires and keeps an optional concrete word beside it. Concrete data
//! resolves branches, IT conditions, addresses outside the symbolic stack, and
//! the small host-call ABI; symbolic data is emitted through
//! [`ContextWithArmv8mOps`]. The data-flow circuits, stack model, raw-memory
//! handling, selectors, shifts, and multipliers are shared with `cirrus-ert`.
//! For constrained streaming-garbled deployments, this Thumb facade is the
//! primary measurement target: the locked SHA-256 workload currently emits
//! substantially fewer non-free gates than its RV32IM counterpart. RV32IM
//! remains a supported compatibility target rather than a deprecated facade.
//!
//! This facade is for one Thumb-only Armv8-M Mainline/Cortex-M33 thread. It
//! does not emulate exceptions, MPU state, floating point, DSP/MVE, atomics,
//! or semihosting. The entry address must have bit zero set; the interpreter
//! clears that bit only for fetching Thumb instructions.
//!
//! Supported compiler-oriented forms include scalar moves and constants,
//! arithmetic and logical instructions, immediate and register shifts/rotates,
//! scalar and long multiplication, stack and concrete loads/stores, Thumb
//! branches/calls/returns, IT blocks, `SVC #0`, and the Secure/Non-secure
//! interworking instructions `SG`, `BXNS`, and `BLXNS`. ARM uses its architectural
//! register-shift count rules rather than RV32's low-five-bit rule. Symbolic
//! shifts and multiplications synthesize selection circuits; a known shift
//! count or multiplicand takes the smaller constant path.
//!
//! Control decisions, non-stack symbolic addresses, unsupported encodings,
//! Arm-state targets, and malformed Thumb images are rejected. `SVC #0` with
//! concrete `r0 = 0` calls the eight-word hash callback with `r1` through `r8`;
//! `r0 = u32::MAX` exits once `sp` is restored. [`ert_func`] applies the
//! AAPCS32 word ABI: `r0` through `r3`, then a full-descending stack aligned to
//! eight bytes at the public interface.
//!
//! The interpreter starts in the Secure state and tracks Secure/Non-secure
//! transitions through `SG`, `BXNS`, and `BLXNS`, gated against a
//! caller-supplied [`SecurityAttribute`] classifier reached through
//! [`ArmHandler::security_attribute`] — mirroring real SAU/IDAU address
//! attribution, but as a purely virtual, host-tracked concept with no real
//! hardware backing. A host can pass a real attribution map through
//! unchanged, narrow it into a stricter sandboxing policy, or synthesize an
//! entirely virtual one for an emulated or user-mode desktop context.
//! [`ArmHandler::svc_permitted`] additionally gates `SVC #0` on the current
//! [`SecurityState`]. [`ArmDefaultHandler`] tunnels any [`Handler`] through
//! as an [`ArmHandler`] with caller-supplied policy closures for both.
//!
//! # References
//!
//! Thumb instruction widths, PC-relative forms, APSR/IT behavior, calls, and
//! `SVC` follow the supplied Armv8-M Architecture Reference Manual
//! ([DDI0553B](https://documentation-service.arm.com/static/66b9cafa32f35b31ceb30a11)).
//! The public register and stack convention follows
//! [AAPCS32](https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst).

use core::{array, error::Error, ops::Range};

use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithValue, HasError};
use cirrus_ert_core::{
    BitOp, Product, Shift, add_bits, bitwise_word, concrete_product, constant_word, fixed_shift,
    invert_word, partial_and_not_word, partial_bitwise_word, select_word,
};

pub use cirrus_ert_core::{EcallOutcome, Handler, RawMemory};

#[cfg(test)]
mod tests;

const REG_COUNT: usize = 16;
const SP: u8 = 13;
const LR: u8 = 14;
const PC: u8 = 15;
const ABI_REGS: [u8; 4] = [0, 1, 2, 3];

/// Boolean operations required by the Armv8-M facade.
pub trait ContextWithArmv8mOps<Val>: cirrus_ert_core::ContextWithErtOps<Val> {}

impl<Val, T: cirrus_ert_core::ContextWithErtOps<Val> + ?Sized> ContextWithArmv8mOps<Val> for T {}

/// A Thumb image decoding failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// The instruction bytes are not an encoding in the supported Thumb subset.
    Unsupported(u32),
    /// A 32-bit Thumb instruction was truncated by the mapped memory.
    Truncated,
    /// An architectural field is reserved or malformed for the selected form.
    Malformed(u32),
}

/// An error while decoding or symbolically executing an Armv8-M program.
pub enum ErtError<E> {
    /// The caller-provided Boolean context or hash callback returned this error.
    Emitted(E),
    /// The Thumb image could not be decoded.
    Decode(DecodeError),
    /// The program is outside the supported well-behaved subset.
    Unexpected,
}

/// A [`Handler`] that reproduces the historical `SVC #0` convention: concrete
/// `r0 = 0` calls the `hash` callback on the eight words `r1` through `r8`,
/// and concrete `r0 = u32::MAX` exits.
pub struct DefaultHandler<C, F> {
    /// The Boolean context bit operations, and the hash callback's own
    /// concrete-type access, are both delegated to this context.
    pub context: C,
    /// The hash callback invoked for the hash `SVC #0`. Its first argument is
    /// the same context passed via `context`, letting a caller's closure use
    /// inherent/concrete methods beyond the three bit-op trait methods while
    /// computing a hash.
    pub hash: F,
}

impl<C: HasError, F> HasError for DefaultHandler<C, F> {
    type Error = C::Error;
}

impl<C: ContextWithValue<bool>, F> ContextWithValue<bool> for DefaultHandler<C, F> {
    type Wrapped = C::Wrapped;
}

impl<C: ContextWithBitAnd<bool>, F> ContextWithBitAnd<bool> for DefaultHandler<C, F> {
    fn bitand(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitand_assign(a, b)
    }
}

impl<C: ContextWithBitOr<bool>, F> ContextWithBitOr<bool> for DefaultHandler<C, F> {
    fn bitor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitor_assign(a, b)
    }
}

impl<C: ContextWithBitXor<bool>, F> ContextWithBitXor<bool> for DefaultHandler<C, F> {
    fn bitxor(&mut self, a: C::Wrapped, b: C::Wrapped) -> Result<C::Wrapped, C::Error> {
        self.context.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut C::Wrapped, b: C::Wrapped) -> Result<(), C::Error> {
        self.context.bitxor_assign(a, b)
    }
}

impl<C, F, W: Clone, E: Error> Handler<bool> for DefaultHandler<C, F>
where
    C: ContextWithArmv8mOps<bool, Wrapped = W, Error = E>,
    F: FnMut(&mut C, &[[W; 32]]) -> Result<[u8; 32], E>,
{
    fn ecall(
        &mut self,
        regs: &mut [[W; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, E> {
        match reg_consts[0] {
            Some(0) => {
                let output = (self.hash)(&mut self.context, &regs[1..9])?;
                for (index, bytes) in output.chunks_exact(4).enumerate() {
                    let register = index + 1;
                    let value = u32::from_le_bytes(array::from_fn(|byte| bytes[byte]));
                    offsets[register] = None;
                    reg_consts[register] = Some(value);
                    regs[register] = constant_word(zero, one, value);
                }
                Ok(EcallOutcome::Continue)
            }
            Some(u32::MAX) => Ok(EcallOutcome::Exit),
            _ => Ok(EcallOutcome::Unexpected),
        }
    }
}

/// A concrete, host-only virtual security state — mirrors real Armv8-M
/// TrustZone-M security states, but is a purely virtual/host-tracked concept
/// with no real hardware backing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityState {
    /// The guest's current virtual state is Non-secure.
    NonSecure,
    /// The guest's current virtual state is Secure.
    Secure,
}

/// The virtual security attribution of an address — mirrors real SAU/IDAU
/// region classification. `NonSecureCallable` behaves as `Secure` for
/// fetch-permission purposes (only reachable from Non-secure state by
/// landing exactly on `SG`); the distinction exists for a caller's own
/// attribution policy to make (e.g. only NSC regions contain `SG` gateways).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityAttribute {
    /// The address is Non-secure.
    NonSecure,
    /// The address is Secure and Non-secure-callable (an `SG` gateway may
    /// live here).
    NonSecureCallable,
    /// The address is Secure and not Non-secure-callable.
    Secure,
}

/// An Arm-specific [`Handler`] extension gating `SVC #0` and Secure/Non-secure
/// state transitions on the interpreter's tracked virtual security state
/// (see [`SecurityState`]) and a caller-supplied address attribution.
pub trait ArmHandler<Val>: Handler<Val> {
    /// Whether an `SVC #0` reached while the CPU is in `state` may proceed
    /// to [`Handler::ecall`]. Called by the interpreter before dispatch;
    /// returning `false` rejects the call as if it were unrecognized.
    fn svc_permitted(&mut self, state: SecurityState) -> bool;

    /// The virtual security attribution of `address` (see
    /// [`SecurityAttribute`]). Called by the interpreter on every fetch
    /// while the CPU is Non-secure, and by `SG`. A caller can pass a real
    /// host attribution map through unchanged, narrow it to a stricter
    /// policy, or supply an entirely synthetic map when there is no real
    /// hardware backing (e.g. a desktop/user-mode test).
    fn security_attribute(&mut self, address: u32) -> SecurityAttribute;
}

/// Tunnels any [`Handler`] through as an [`ArmHandler`], adding both policy
/// closures — like [`DefaultHandler`]'s `hash` receiving `context` — as
/// closures that receive the wrapped `inner` handler directly.
pub struct ArmDefaultHandler<H, G, A> {
    /// The wrapped handler.
    pub inner: H,
    /// The `SVC #0` gate policy. Receives `inner` and the current
    /// [`SecurityState`]; `|_inner, _state| true` permits every call.
    pub svc_permitted: G,
    /// The address attribution policy. Receives `inner` and the address
    /// being classified.
    pub security_attribute: A,
}

impl<H: HasError, G, A> HasError for ArmDefaultHandler<H, G, A> {
    type Error = H::Error;
}

impl<H: ContextWithValue<bool>, G, A> ContextWithValue<bool> for ArmDefaultHandler<H, G, A> {
    type Wrapped = H::Wrapped;
}

impl<H: ContextWithBitAnd<bool>, G, A> ContextWithBitAnd<bool> for ArmDefaultHandler<H, G, A> {
    fn bitand(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitand(a, b)
    }
    fn bitand_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitand_assign(a, b)
    }
}

impl<H: ContextWithBitOr<bool>, G, A> ContextWithBitOr<bool> for ArmDefaultHandler<H, G, A> {
    fn bitor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitor(a, b)
    }
    fn bitor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitor_assign(a, b)
    }
}

impl<H: ContextWithBitXor<bool>, G, A> ContextWithBitXor<bool> for ArmDefaultHandler<H, G, A> {
    fn bitxor(&mut self, a: H::Wrapped, b: H::Wrapped) -> Result<H::Wrapped, H::Error> {
        self.inner.bitxor(a, b)
    }
    fn bitxor_assign(&mut self, a: &mut H::Wrapped, b: H::Wrapped) -> Result<(), H::Error> {
        self.inner.bitxor_assign(a, b)
    }
}

impl<H: Handler<bool>, G, A> Handler<bool> for ArmDefaultHandler<H, G, A> {
    fn ecall(
        &mut self,
        regs: &mut [[H::Wrapped; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &H::Wrapped,
        one: &H::Wrapped,
    ) -> Result<EcallOutcome, H::Error> {
        self.inner.ecall(regs, reg_consts, offsets, zero, one)
    }
}

impl<H: Handler<bool>, G: FnMut(&mut H, SecurityState) -> bool, A: FnMut(&mut H, u32) -> SecurityAttribute>
    ArmHandler<bool> for ArmDefaultHandler<H, G, A>
{
    fn svc_permitted(&mut self, state: SecurityState) -> bool {
        (self.svc_permitted)(&mut self.inner, state)
    }

    fn security_attribute(&mut self, address: u32) -> SecurityAttribute {
        (self.security_attribute)(&mut self.inner, address)
    }
}

/// Add two little-endian symbolic 32-bit words with an initial carry bit.
///
/// `zero` and `one` are retained to match the RISC-V compatibility helper.
pub fn simple_add<W: Clone, E: Error>(
    t: &mut (dyn ContextWithArmv8mOps<bool, Wrapped = W, Error = E> + '_),
    v: &[W; 32],
    w: &[W; 32],
    carry: W,
    _zero: W,
    _one: W,
) -> Result<[W; 32], E> {
    add_bits(t, v, w, carry)
}

/// Invoke an Arm Thumb function through the AAPCS32 word ABI.
///
/// The first four arguments and results occupy `r0` through `r3`; remaining
/// words use the caller-provided symbolic stack. `vstack` must provide a
/// byte-length divisible by eight so that the public interface's `sp` stays
/// eight-byte aligned. `pc` must be an odd Thumb function pointer.
#[allow(clippy::too_many_arguments)]
pub fn ert_func<W: Clone, E: Error, const N: usize, const M: usize>(
    t: &mut (dyn ArmHandler<bool, Wrapped = W, Error = E> + '_),
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 16],
    reg_consts: &mut [Option<u32>; 16],
    zero: W,
    one: W,
    args: [([W; 32], Option<u32>); N],
) -> Result<[([W; 32], Option<u32>); M], ErtError<E>> {
    let stack_pointer = abi_stack_pointer(vstack, N.max(M)).ok_or(ErtError::Unexpected)?;
    if stack_pointer & 7 != 0 || pc & 1 == 0 {
        return Err(ErtError::Unexpected);
    }
    write_abi_args(regs, reg_consts, vstack, stack_pointer, args);
    Machine::new(
        t,
        mem,
        rstack,
        vstack,
        pc & !1,
        regs,
        reg_consts,
        zero,
        one,
        stack_pointer,
    )
    .run()?;
    Ok(read_abi_results(regs, reg_consts, vstack, stack_pointer))
}

/// Execute Thumb instructions until the supported exit `SVC #0`.
///
/// `pc` is an odd Thumb entry pointer. The interpreter initializes `sp` to the
/// end of `vstack` and requires that initial byte address to be eight-byte
/// aligned.
#[allow(clippy::too_many_arguments)]
pub fn ert_emit<W: Clone, E: Error>(
    t: &mut (dyn ArmHandler<bool, Wrapped = W, Error = E> + '_),
    mem: RawMemory<'_>,
    rstack: &mut [u32],
    vstack: &mut [W],
    pc: u32,
    regs: &mut [[W; 32]; 16],
    reg_consts: &mut [Option<u32>; 16],
    zero: W,
    one: W,
) -> Result<(), ErtError<E>> {
    let stack_pointer = u32::try_from(vstack.len() / 8).map_err(|_| ErtError::Unexpected)?;
    if stack_pointer & 7 != 0 || pc & 1 == 0 {
        return Err(ErtError::Unexpected);
    }
    Machine::new(
        t,
        mem,
        rstack,
        vstack,
        pc & !1,
        regs,
        reg_consts,
        zero,
        one,
        stack_pointer,
    )
    .run()
}

fn abi_stack_pointer<W>(vstack: &[W], values: usize) -> Option<u32> {
    let extra = values.saturating_sub(ABI_REGS.len());
    let bytes = u32::try_from(vstack.len() / 8).ok()?;
    let reservation = u32::try_from(extra.checked_mul(4)?).ok()?;
    let reservation = (reservation + 7) & !7;
    bytes.checked_sub(reservation)
}

fn write_abi_args<W: Clone, const N: usize>(
    regs: &mut [[W; 32]; REG_COUNT],
    reg_consts: &mut [Option<u32>; REG_COUNT],
    vstack: &mut [W],
    sp: u32,
    args: [([W; 32], Option<u32>); N],
) {
    for (index, (value, constant)) in args.into_iter().enumerate() {
        if let Some(&register) = ABI_REGS.get(index) {
            regs[register as usize] = value;
            reg_consts[register as usize] = constant;
        } else {
            let start = sp as usize * 8 + 32 * (index - ABI_REGS.len());
            vstack[start..start + 32].clone_from_slice(&value);
        }
    }
}

fn read_abi_results<W: Clone, const M: usize>(
    regs: &[[W; 32]; REG_COUNT],
    reg_consts: &[Option<u32>; REG_COUNT],
    vstack: &[W],
    sp: u32,
) -> [([W; 32], Option<u32>); M] {
    array::from_fn(|index| {
        if let Some(&register) = ABI_REGS.get(index) {
            (
                regs[register as usize].clone(),
                reg_consts[register as usize],
            )
        } else {
            let start = sp as usize * 8 + 32 * (index - ABI_REGS.len());
            (array::from_fn(|bit| vstack[start + bit].clone()), None)
        }
    })
}

#[derive(Clone, Copy)]
enum Flow {
    Next(u32),
    Exit,
}

#[derive(Clone, Copy)]
enum LoadKind {
    ByteSigned,
    HalfSigned,
    ByteUnsigned,
    HalfUnsigned,
    Word,
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    AddCarry,
    Sub,
    SubCarry,
    ReverseSub,
}

#[derive(Clone, Copy)]
enum Operand {
    Register(u8),
    Immediate(u32),
    Shifted {
        register: u8,
        shift: Shift,
        amount: ShiftAmount,
    },
}

#[derive(Clone, Copy)]
enum ShiftAmount {
    Immediate(u32),
    Register(u8),
}

#[derive(Clone, Copy)]
enum Op {
    Nop,
    It {
        condition: u8,
        mask: u8,
    },
    Move {
        dest: u8,
        source: Operand,
        set_flags: bool,
    },
    MoveTop {
        dest: u8,
        immediate: u16,
    },
    MoveNot {
        dest: u8,
        source: Operand,
        set_flags: bool,
    },
    Arithmetic {
        kind: Arithmetic,
        dest: u8,
        left: Operand,
        right: Operand,
        set_flags: bool,
    },
    Bitwise {
        kind: BitOp,
        dest: u8,
        left: Operand,
        right: Operand,
        set_flags: bool,
    },
    BitClear {
        dest: u8,
        left: Operand,
        right: Operand,
        set_flags: bool,
    },
    Compare {
        left: Operand,
        right: Operand,
        add: bool,
    },
    Test {
        left: Operand,
        right: Operand,
    },
    Shift {
        dest: u8,
        source: u8,
        direction: Shift,
        amount: ShiftAmount,
        set_flags: bool,
    },
    Multiply {
        dest: u8,
        left: u8,
        right: u8,
        add: Option<u8>,
        subtract: bool,
    },
    LongMultiply {
        dest_low: u8,
        dest_high: u8,
        left: u8,
        right: u8,
        signed: bool,
    },
    Load {
        dest: u8,
        base: u8,
        offset: i32,
        kind: LoadKind,
    },
    Store {
        source: u8,
        base: u8,
        offset: i32,
        width: usize,
    },
    StoreDouble {
        first: u8,
        second: u8,
        base: u8,
        offset: i32,
    },
    LoadIndexed {
        dest: u8,
        base: u8,
        index: u8,
        shift: u8,
        kind: LoadKind,
    },
    StoreIndexed {
        source: u8,
        base: u8,
        index: u8,
        shift: u8,
        width: usize,
    },
    Push {
        list: u16,
    },
    Pop {
        list: u16,
    },
    LoadMultiple {
        base: u8,
        list: u16,
        write_back: bool,
    },
    StoreMultiple {
        base: u8,
        list: u16,
        write_back: bool,
    },
    Branch {
        target: u32,
        condition: Option<u8>,
    },
    CompareBranch {
        register: u8,
        nonzero: bool,
        target: u32,
    },
    Call {
        target: u32,
    },
    CallRegister {
        register: u8,
    },
    BranchRegister {
        register: u8,
    },
    /// `SG`: the Secure Gateway instruction.
    SecureGateway,
    /// `BXNS`/`BLXNS`: branch (and, if `link`, link) exchange Non-secure.
    BranchExchangeNonSecure {
        register: u8,
        link: bool,
    },
    Svc(u8),
}

struct Decoded {
    operation: Op,
    len: u32,
}

struct Machine<'a, W, E> {
    t: &'a mut (dyn ArmHandler<bool, Wrapped = W, Error = E> + 'a),
    mem: RawMemory<'a>,
    rstack: &'a mut [u32],
    vstack: &'a mut [W],
    pc: u32,
    regs: &'a mut [[W; 32]; REG_COUNT],
    constants: &'a mut [Option<u32>; REG_COUNT],
    zero: W,
    one: W,
    sp: u32,
    stack_top: u32,
    rsp: usize,
    offsets: [Option<i32>; REG_COUNT],
    flags: [Option<bool>; 4],
    itstate: u8,
    security_state: SecurityState,
}

impl<'a, W: Clone, E: Error> Machine<'a, W, E> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        t: &'a mut (dyn ArmHandler<bool, Wrapped = W, Error = E> + 'a),
        mem: RawMemory<'a>,
        rstack: &'a mut [u32],
        vstack: &'a mut [W],
        pc: u32,
        regs: &'a mut [[W; 32]; REG_COUNT],
        constants: &'a mut [Option<u32>; REG_COUNT],
        zero: W,
        one: W,
        sp: u32,
    ) -> Self {
        let mut machine = Self {
            t,
            mem,
            rstack,
            vstack,
            pc,
            regs,
            constants,
            zero,
            one,
            sp,
            stack_top: sp,
            rsp: 0,
            offsets: [None; REG_COUNT],
            flags: [Some(false); 4],
            itstate: 0,
            security_state: SecurityState::Secure,
        };
        machine.offsets[SP as usize] = Some(0);
        machine.constants[SP as usize] = None;
        machine.regs[SP as usize] = machine.word_from_constant(sp);
        machine
    }

    fn run(mut self) -> Result<(), ErtError<E>> {
        loop {
            let decoded = self.decode()?;
            let is_it = matches!(decoded.operation, Op::It { .. });
            if is_it && self.itstate != 0 {
                return Err(ErtError::Unexpected);
            }
            let execute = if is_it {
                true
            } else {
                self.condition_passed()?
            };
            let flow = if execute {
                self.execute(decoded.operation, decoded.len)?
            } else {
                Flow::Next(self.pc.wrapping_add(decoded.len))
            };
            if !is_it && self.itstate != 0 {
                self.advance_it();
            }
            match flow {
                Flow::Next(next) => self.pc = next,
                Flow::Exit => return Ok(()),
            }
        }
    }

    fn decode(&self) -> Result<Decoded, ErtError<E>> {
        let first = u16::from_le_bytes(self.mem.read::<2>(self.pc).ok_or(ErtError::Unexpected)?);
        let wide = first & 0xe000 == 0xe000 && first & 0x1800 != 0;
        if wide {
            let second = u16::from_le_bytes(
                self.mem
                    .read::<2>(self.pc.wrapping_add(2))
                    .ok_or(ErtError::Decode(DecodeError::Malformed(self.pc)))?,
            );
            decode32(self.pc, first, second).map_err(ErtError::Decode)
        } else {
            decode16(self.pc, first).map_err(ErtError::Decode)
        }
    }

    fn condition_passed(&self) -> Result<bool, ErtError<E>> {
        if self.itstate == 0 {
            return Ok(true);
        }
        self.condition((self.itstate >> 4) & 15)
    }

    fn condition(&self, condition: u8) -> Result<bool, ErtError<E>> {
        let [n, z, c, v] = self.flags;
        let required = |flag: Option<bool>| flag.ok_or(ErtError::Unexpected);
        match condition {
            0 => Ok(required(z)?),
            1 => Ok(!required(z)?),
            2 => Ok(required(c)?),
            3 => Ok(!required(c)?),
            4 => Ok(required(n)?),
            5 => Ok(!required(n)?),
            6 => Ok(required(v)?),
            7 => Ok(!required(v)?),
            8 => Ok(required(c)? && !required(z)?),
            9 => Ok(!required(c)? || required(z)?),
            10 => Ok(required(n)? == required(v)?),
            11 => Ok(required(n)? != required(v)?),
            12 => Ok(!required(z)? && required(n)? == required(v)?),
            13 => Ok(required(z)? || required(n)? != required(v)?),
            14 => Ok(true),
            _ => Err(ErtError::Unexpected),
        }
    }

    fn advance_it(&mut self) {
        if self.itstate & 7 == 0 {
            self.itstate = 0;
        } else {
            self.itstate = (self.itstate & 0xe0) | ((self.itstate << 1) & 0x1f);
        }
    }

    fn execute(&mut self, operation: Op, len: u32) -> Result<Flow, ErtError<E>> {
        if self.security_state == SecurityState::NonSecure
            && !matches!(operation, Op::SecureGateway)
            && self.t.security_attribute(self.pc) != SecurityAttribute::NonSecure
        {
            return Err(ErtError::Unexpected);
        }
        match operation {
            Op::Nop => self.next(len),
            Op::It { condition, mask } => {
                if condition >= 14 || mask == 0 {
                    return Err(ErtError::Decode(DecodeError::Malformed(
                        ((condition as u32) << 4) | mask as u32,
                    )));
                }
                self.itstate = (condition << 4) | mask;
                self.next(len)
            }
            Op::Move {
                dest,
                source,
                set_flags,
            } => {
                let (word, value) = self.operand(source)?;
                self.write(dest, word, value);
                if let Operand::Register(source) = source {
                    self.offsets[dest as usize] = self.offsets[source as usize];
                }
                if set_flags {
                    self.set_nz(value);
                }
                self.next(len)
            }
            Op::MoveTop { dest, immediate } => {
                let mut word = self.regs[dest as usize].clone();
                for bit in 16..32 {
                    word[bit] = if (immediate >> (bit - 16)) & 1 == 0 {
                        self.zero.clone()
                    } else {
                        self.one.clone()
                    };
                }
                let value = self.constants[dest as usize]
                    .map(|value| (value & 0xffff) | ((immediate as u32) << 16));
                self.write(dest, word, value);
                self.next(len)
            }
            Op::MoveNot {
                dest,
                source,
                set_flags,
            } => {
                let (word, value) = self.operand(source)?;
                let word =
                    invert_word(self.t, &word, self.one.clone()).map_err(ErtError::Emitted)?;
                let value = value.map(|value| !value);
                self.write(dest, word, value);
                if set_flags {
                    self.set_nz(value);
                }
                self.next(len)
            }
            Op::Arithmetic {
                kind,
                dest,
                left,
                right,
                set_flags,
            } => self.arithmetic(kind, dest, left, right, set_flags, len),
            Op::Bitwise {
                kind,
                dest,
                left,
                right,
                set_flags,
            } => {
                let (left_word, left_value) = self.operand(left)?;
                let (right_word, right_value) = self.operand(right)?;
                let (word, value) = if let (Some(l), Some(r)) = (left_value, right_value) {
                    let value = fold(kind, l, r);
                    (self.word_from_constant(value), Some(value))
                } else if let Some((constant, symbolic)) = match (left_value, right_value) {
                    (Some(l), None) => Some((l, &right_word)),
                    (None, Some(r)) => Some((r, &left_word)),
                    _ => None,
                } {
                    if degenerate(kind, constant) {
                        let value = degenerate_value(kind);
                        (self.word_from_constant(value), Some(value))
                    } else {
                        (
                            partial_bitwise_word(
                                self.t, constant, symbolic, &self.zero, &self.one, kind,
                            )
                            .map_err(ErtError::Emitted)?,
                            None,
                        )
                    }
                } else {
                    (
                        bitwise_word(self.t, &left_word, &right_word, kind)
                            .map_err(ErtError::Emitted)?,
                        None,
                    )
                };
                self.write(dest, word, value);
                if set_flags {
                    self.set_nz(value);
                }
                self.next(len)
            }
            Op::BitClear {
                dest,
                left,
                right,
                set_flags,
            } => {
                let (left_word, left_value) = self.operand(left)?;
                let (right_word, right_value) = self.operand(right)?;
                let (word, value) = if let (Some(l), Some(r)) = (left_value, right_value) {
                    let value = l & !r;
                    (self.word_from_constant(value), Some(value))
                } else if let Some(right) = right_value {
                    if right == u32::MAX {
                        (self.word_from_constant(0), Some(0))
                    } else {
                        (
                            partial_bitwise_word(
                                self.t,
                                !right,
                                &left_word,
                                &self.zero,
                                &self.one,
                                BitOp::And,
                            )
                            .map_err(ErtError::Emitted)?,
                            None,
                        )
                    }
                } else if let Some(left) = left_value {
                    if left == 0 {
                        (self.word_from_constant(0), Some(0))
                    } else {
                        (
                            partial_and_not_word(self.t, left, &right_word, &self.zero, &self.one)
                                .map_err(ErtError::Emitted)?,
                            None,
                        )
                    }
                } else {
                    let inverted = invert_word(self.t, &right_word, self.one.clone())
                        .map_err(ErtError::Emitted)?;
                    (
                        bitwise_word(self.t, &left_word, &inverted, BitOp::And)
                            .map_err(ErtError::Emitted)?,
                        None,
                    )
                };
                self.write(dest, word, value);
                if set_flags {
                    self.set_nz(value);
                }
                self.next(len)
            }
            Op::Compare { left, right, add } => {
                let (_, left_value) = self.operand(left)?;
                let (_, right_value) = self.operand(right)?;
                if add {
                    self.set_add_flags(left_value, right_value, false);
                } else {
                    self.set_sub_flags(left_value, right_value, true);
                }
                self.next(len)
            }
            Op::Test { left, right } => {
                let (_, left_value) = self.operand(left)?;
                let (_, right_value) = self.operand(right)?;
                self.set_nz(
                    left_value
                        .zip(right_value)
                        .map(|(left, right)| left & right),
                );
                self.next(len)
            }
            Op::Shift {
                dest,
                source,
                direction,
                amount,
                set_flags,
            } => {
                let (word, value) = self.shift_word(source, direction, amount)?;
                self.write(dest, word, value);
                if set_flags {
                    self.set_nz(value);
                }
                self.next(len)
            }
            Op::Multiply {
                dest,
                left,
                right,
                add,
                subtract,
            } => self.multiply(dest, left, right, add, subtract, len),
            Op::LongMultiply {
                dest_low,
                dest_high,
                left,
                right,
                signed,
            } => self.long_multiply(dest_low, dest_high, left, right, signed, len),
            Op::Load {
                dest,
                base,
                offset,
                kind,
            } => self.load(dest, base, offset, kind, len),
            Op::Store {
                source,
                base,
                offset,
                width,
            } => self.store(source, base, offset, width, len),
            Op::StoreDouble {
                first,
                second,
                base,
                offset,
            } => {
                self.store(first, base, offset, 32, len)?;
                self.store(second, base, offset.wrapping_add(4), 32, len)
            }
            Op::LoadIndexed {
                dest,
                base,
                index,
                shift,
                kind,
            } => {
                let offset = self.indexed_offset(index, shift)?;
                self.load(dest, base, offset, kind, len)
            }
            Op::StoreIndexed {
                source,
                base,
                index,
                shift,
                width,
            } => {
                let offset = self.indexed_offset(index, shift)?;
                self.store(source, base, offset, width, len)
            }
            Op::Push { list } => self.push(list, len),
            Op::Pop { list } => self.pop(list, len),
            Op::LoadMultiple {
                base,
                list,
                write_back,
            } => self.load_multiple(base, list, write_back, len),
            Op::StoreMultiple {
                base,
                list,
                write_back,
            } => self.store_multiple(base, list, write_back, len),
            Op::Branch { target, condition } => {
                let taken = condition.map_or(Ok(true), |condition| self.condition(condition))?;
                Ok(Flow::Next(if taken {
                    target
                } else {
                    self.pc.wrapping_add(len)
                }))
            }
            Op::CompareBranch {
                register,
                nonzero,
                target,
            } => {
                let value = self.constants[register as usize].ok_or(ErtError::Unexpected)?;
                Ok(Flow::Next(if (value != 0) == nonzero {
                    target
                } else {
                    self.pc + len
                }))
            }
            Op::Call { target } => self.call(target, len),
            Op::CallRegister { register } => {
                let target = self.constants[register as usize].ok_or(ErtError::Unexpected)?;
                if target & 1 == 0 {
                    return Err(ErtError::Unexpected);
                }
                self.call(target & !1, len)
            }
            Op::BranchRegister { register } => self.branch_register(register),
            Op::SecureGateway => self.secure_gateway(len),
            Op::BranchExchangeNonSecure { register, link } => {
                self.branch_exchange_non_secure(register, link, len)
            }
            Op::Svc(0) => {
                if !self.t.svc_permitted(self.security_state) {
                    return Err(ErtError::Unexpected);
                }
                match self.t.ecall(
                    &mut self.regs[..],
                    &mut self.constants[..],
                    &mut self.offsets[..],
                    &self.zero,
                    &self.one,
                ) {
                    Ok(EcallOutcome::Continue) => self.next(len),
                    Ok(EcallOutcome::Exit) if self.sp == self.stack_top => Ok(Flow::Exit),
                    Ok(EcallOutcome::Exit) | Ok(EcallOutcome::Unexpected) => {
                        Err(ErtError::Unexpected)
                    }
                    Err(e) => Err(ErtError::Emitted(e)),
                }
            }
            Op::Svc(_) => Err(ErtError::Unexpected),
        }
    }

    fn next(&self, len: u32) -> Result<Flow, ErtError<E>> {
        Ok(Flow::Next(self.pc.wrapping_add(len)))
    }

    fn word_from_constant(&self, value: u32) -> [W; 32] {
        constant_word(&self.zero, &self.one, value)
    }

    fn write(&mut self, register: u8, word: [W; 32], value: Option<u32>) {
        self.offsets[register as usize] = None;
        self.regs[register as usize] = word;
        self.constants[register as usize] = value;
    }

    fn write_constant(&mut self, register: u8, value: u32) {
        self.write(register, self.word_from_constant(value), Some(value));
    }

    fn set_nz(&mut self, value: Option<u32>) {
        self.flags[0] = value.map(|value| value >> 31 != 0);
        self.flags[1] = value.map(|value| value == 0);
    }

    fn set_add_flags(&mut self, left: Option<u32>, right: Option<u32>, carry: bool) {
        let value = left
            .zip(right)
            .map(|(left, right)| left.wrapping_add(right).wrapping_add(carry as u32));
        self.set_nz(value);
        self.flags[2] = left
            .zip(right)
            .map(|(left, right)| (left as u64 + right as u64 + carry as u64) >> 32 != 0);
        self.flags[3] = left.zip(right).map(|(left, right)| {
            let result = left.wrapping_add(right).wrapping_add(carry as u32);
            ((left ^ result) & (right ^ result) & 0x8000_0000) != 0
        });
    }

    fn set_sub_flags(&mut self, left: Option<u32>, right: Option<u32>, carry: bool) {
        let borrow = !carry;
        let value = left
            .zip(right)
            .map(|(left, right)| left.wrapping_sub(right).wrapping_sub(borrow as u32));
        self.set_nz(value);
        self.flags[2] = left
            .zip(right)
            .map(|(left, right)| left >= right.wrapping_add(borrow as u32));
        self.flags[3] = left.zip(right).map(|(left, right)| {
            let result = left.wrapping_sub(right).wrapping_sub(borrow as u32);
            ((left ^ right) & (left ^ result) & 0x8000_0000) != 0
        });
    }

    fn operand(&mut self, operand: Operand) -> Result<([W; 32], Option<u32>), ErtError<E>> {
        match operand {
            Operand::Register(register) => Ok((
                self.regs[register as usize].clone(),
                self.constants[register as usize],
            )),
            Operand::Immediate(value) => Ok((self.word_from_constant(value), Some(value))),
            Operand::Shifted {
                register,
                shift,
                amount,
            } => self.shift_word(register, shift, amount),
        }
    }

    fn shift_word(
        &mut self,
        register: u8,
        direction: Shift,
        amount: ShiftAmount,
    ) -> Result<([W; 32], Option<u32>), ErtError<E>> {
        let source = self.regs[register as usize].clone();
        let source_value = self.constants[register as usize];
        match amount {
            ShiftAmount::Immediate(amount) => Ok((
                fixed_shift(&source, amount, direction, &self.zero),
                source_value.map(|value| concrete_shift(value, amount, direction)),
            )),
            ShiftAmount::Register(register) => {
                let amount_word = self.regs[register as usize].clone();
                match self.constants[register as usize] {
                    Some(amount) => Ok((
                        fixed_shift(
                            &source,
                            arm_shift_count(amount, direction),
                            direction,
                            &self.zero,
                        ),
                        source_value.map(|value| concrete_shift(value, amount, direction)),
                    )),
                    None => Ok((
                        arm_runtime_shift(self.t, &source, &amount_word, direction, &self.zero)
                            .map_err(ErtError::Emitted)?,
                        None,
                    )),
                }
            }
        }
    }

    fn arithmetic(
        &mut self,
        kind: Arithmetic,
        dest: u8,
        left: Operand,
        right: Operand,
        set_flags: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let (left_word, left_value) = self.operand(left)?;
        let (right_word, right_value) = self.operand(right)?;
        let carry = match kind {
            Arithmetic::Add | Arithmetic::Sub | Arithmetic::ReverseSub => self.zero.clone(),
            Arithmetic::AddCarry | Arithmetic::SubCarry => match self.flags[2] {
                Some(true) => self.one.clone(),
                Some(false) => self.zero.clone(),
                None => return Err(ErtError::Unexpected),
            },
        };
        let (left_word, right_word, left_value, right_value, subtraction) = match kind {
            Arithmetic::Add | Arithmetic::AddCarry => {
                (left_word, right_word, left_value, right_value, false)
            }
            Arithmetic::Sub | Arithmetic::SubCarry => {
                let inverted = invert_word(self.t, &right_word, self.one.clone())
                    .map_err(ErtError::Emitted)?;
                (left_word, inverted, left_value, right_value, true)
            }
            Arithmetic::ReverseSub => {
                let inverted =
                    invert_word(self.t, &left_word, self.one.clone()).map_err(ErtError::Emitted)?;
                (right_word, inverted, right_value, left_value, true)
            }
        };
        let value = match kind {
            Arithmetic::Add => left_value
                .zip(right_value)
                .map(|(left, right)| left.wrapping_add(right)),
            Arithmetic::AddCarry => left_value
                .zip(right_value)
                .zip(self.flags[2])
                .map(|((left, right), carry)| left.wrapping_add(right).wrapping_add(carry as u32)),
            Arithmetic::Sub => left_value
                .zip(right_value)
                .map(|(left, right)| left.wrapping_sub(right)),
            Arithmetic::SubCarry => {
                left_value
                    .zip(right_value)
                    .zip(self.flags[2])
                    .map(|((left, right), carry)| {
                        left.wrapping_sub(right).wrapping_sub((!carry) as u32)
                    })
            }
            Arithmetic::ReverseSub => left_value
                .zip(right_value)
                .map(|(left, right)| right.wrapping_sub(left)),
        };
        let word = if let Some(value) = value {
            self.word_from_constant(value)
        } else {
            let carry = if subtraction && matches!(kind, Arithmetic::Sub | Arithmetic::ReverseSub)
            {
                self.one.clone()
            } else {
                carry
            };
            add_bits(self.t, &left_word, &right_word, carry).map_err(ErtError::Emitted)?
        };
        if dest == SP {
            let new_sp = match (kind, left, right) {
                (Arithmetic::Add, Operand::Register(SP), Operand::Immediate(amount)) => {
                    self.sp.wrapping_add(amount)
                }
                (Arithmetic::Sub, Operand::Register(SP), Operand::Immediate(amount)) => {
                    self.sp.wrapping_sub(amount)
                }
                _ => return Err(ErtError::Unexpected),
            };
            self.adjust_sp(new_sp);
        } else {
            self.write(dest, word, value);
            self.offsets[dest as usize] = match (kind, left, right) {
                (Arithmetic::Add, Operand::Register(base), Operand::Immediate(offset)) => {
                    self.offsets[base as usize].map(|base| base.wrapping_add(offset as i32))
                }
                (Arithmetic::Sub, Operand::Register(base), Operand::Immediate(offset)) => {
                    self.offsets[base as usize].map(|base| base.wrapping_sub(offset as i32))
                }
                _ => None,
            };
        }
        if set_flags {
            match kind {
                Arithmetic::Add => self.set_add_flags(left_value, right_value, false),
                Arithmetic::AddCarry => {
                    self.set_add_flags(left_value, right_value, self.flags[2].unwrap_or(false))
                }
                Arithmetic::Sub => self.set_sub_flags(left_value, right_value, true),
                Arithmetic::SubCarry => {
                    self.set_sub_flags(left_value, right_value, self.flags[2].unwrap_or(false))
                }
                Arithmetic::ReverseSub => self.set_sub_flags(right_value, left_value, true),
            }
        }
        self.next(len)
    }

    fn adjust_sp(&mut self, new_sp: u32) {
        let old_sp = self.sp;
        let delta = new_sp.wrapping_sub(old_sp) as i32;
        self.sp = new_sp;
        for offset in self.offsets.iter_mut().flatten() {
            *offset = offset.wrapping_sub(delta);
        }
        self.offsets[SP as usize] = Some(0);
        self.constants[SP as usize] = None;
        self.regs[SP as usize] = self.word_from_constant(new_sp);
    }

    fn multiply(
        &mut self,
        dest: u8,
        left: u8,
        right: u8,
        add: Option<u8>,
        subtract: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let left_word = self.regs[left as usize].clone();
        let right_word = self.regs[right as usize].clone();
        let left_value = self.constants[left as usize];
        let right_value = self.constants[right as usize];
        let product_value = left_value
            .zip(right_value)
            .map(|(left, right)| left.wrapping_mul(right));
        let (add_word, add_value) = match add {
            Some(add) => (
                Some(self.regs[add as usize].clone()),
                self.constants[add as usize],
            ),
            None => (None, None),
        };
        let value = match add {
            Some(_) if subtract => add_value
                .zip(product_value)
                .map(|(add, product)| add.wrapping_sub(product)),
            Some(_) => product_value
                .zip(add_value)
                .map(|(product, add)| product.wrapping_add(add)),
            None => product_value,
        };
        let word = if let Some(value) = value {
            self.word_from_constant(value)
        } else {
            let mut word = multiply_word(
                self.t,
                &left_word,
                &right_word,
                left_value,
                right_value,
                Product::Low,
                &self.zero,
                &self.one,
            )
            .map_err(ErtError::Emitted)?;
            if let Some(add_word) = add_word {
                word = if subtract {
                    let inverted =
                        invert_word(self.t, &word, self.one.clone()).map_err(ErtError::Emitted)?;
                    add_bits(self.t, &add_word, &inverted, self.one.clone())
                        .map_err(ErtError::Emitted)?
                } else {
                    add_bits(self.t, &word, &add_word, self.zero.clone())
                        .map_err(ErtError::Emitted)?
                };
            }
            word
        };
        self.write(dest, word, value);
        self.next(len)
    }

    fn long_multiply(
        &mut self,
        dest_low: u8,
        dest_high: u8,
        left: u8,
        right: u8,
        signed: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let left_word = self.regs[left as usize].clone();
        let right_word = self.regs[right as usize].clone();
        let left_value = self.constants[left as usize];
        let right_value = self.constants[right as usize];
        let product = if signed {
            Product::HighSigned
        } else {
            Product::HighUnsigned
        };
        let high = multiply_word(
            self.t,
            &left_word,
            &right_word,
            left_value,
            right_value,
            product,
            &self.zero,
            &self.one,
        )
        .map_err(ErtError::Emitted)?;
        let low = multiply_word(
            self.t,
            &left_word,
            &right_word,
            left_value,
            right_value,
            Product::Low,
            &self.zero,
            &self.one,
        )
        .map_err(ErtError::Emitted)?;
        self.write(
            dest_low,
            low,
            left_value
                .zip(right_value)
                .map(|(left, right)| left.wrapping_mul(right)),
        );
        self.write(
            dest_high,
            high,
            left_value
                .zip(right_value)
                .map(|(left, right)| concrete_product(product, left, right)),
        );
        self.next(len)
    }

    fn stack_offset(&self, base: u8, offset: i32) -> Result<i32, ErtError<E>> {
        match self.offsets[base as usize] {
            Some(base) => Ok(base.wrapping_add(offset)),
            None => Err(ErtError::Unexpected),
        }
    }

    fn indexed_offset(&self, index: u8, shift: u8) -> Result<i32, ErtError<E>> {
        let value = self.constants[index as usize].ok_or(ErtError::Unexpected)?;
        Ok(value.wrapping_shl(shift as u32) as i32)
    }

    fn stack_bits(&self, offset: i32, width: usize) -> Result<Range<usize>, ErtError<E>> {
        let address = self.sp.wrapping_add_signed(offset) as usize;
        let start = address.checked_mul(8).ok_or(ErtError::Unexpected)?;
        let end = start.checked_add(width).ok_or(ErtError::Unexpected)?;
        if end > self.vstack.len() {
            return Err(ErtError::Unexpected);
        }
        Ok(start..end)
    }

    fn load(
        &mut self,
        dest: u8,
        base: u8,
        offset: i32,
        kind: LoadKind,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        if base == PC {
            let address = (self.pc.wrapping_add(4) & !3).wrapping_add_signed(offset);
            return self.load_concrete(dest, address, kind, len);
        }
        match self.offsets[base as usize] {
            Some(base_offset) => {
                let width = load_width(kind);
                let signed = matches!(kind, LoadKind::ByteSigned | LoadKind::HalfSigned);
                let range = self.stack_bits(base_offset.wrapping_add(offset), width)?;
                let word = array::from_fn(|bit| {
                    if bit < width {
                        self.vstack[range.start + bit].clone()
                    } else if signed {
                        self.vstack[range.start + width - 1].clone()
                    } else {
                        self.zero.clone()
                    }
                });
                self.write(dest, word, None);
            }
            None => {
                let address = self.constants[base as usize]
                    .ok_or(ErtError::Unexpected)?
                    .wrapping_add_signed(offset);
                return self.load_concrete(dest, address, kind, len);
            }
        }
        self.next(len)
    }

    fn load_concrete(
        &mut self,
        dest: u8,
        address: u32,
        kind: LoadKind,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let value = match kind {
            LoadKind::ByteSigned => {
                i8::from_le_bytes(self.mem.read::<1>(address).ok_or(ErtError::Unexpected)?) as i32
                    as u32
            }
            LoadKind::HalfSigned => {
                i16::from_le_bytes(self.mem.read::<2>(address).ok_or(ErtError::Unexpected)?) as i32
                    as u32
            }
            LoadKind::ByteUnsigned => {
                u8::from_le_bytes(self.mem.read::<1>(address).ok_or(ErtError::Unexpected)?) as u32
            }
            LoadKind::HalfUnsigned => {
                u16::from_le_bytes(self.mem.read::<2>(address).ok_or(ErtError::Unexpected)?) as u32
            }
            LoadKind::Word => {
                u32::from_le_bytes(self.mem.read::<4>(address).ok_or(ErtError::Unexpected)?)
            }
        };
        self.write_constant(dest, value);
        self.next(len)
    }

    fn store(
        &mut self,
        source: u8,
        base: u8,
        offset: i32,
        width: usize,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let offset = self.stack_offset(base, offset)?;
        let range = self.stack_bits(offset, width)?;
        for bit in 0..width {
            self.vstack[range.start + bit] = self.regs[source as usize][bit].clone();
        }
        self.next(len)
    }

    fn push(&mut self, list: u16, len: u32) -> Result<Flow, ErtError<E>> {
        let count = list.count_ones() as u32;
        let start = self.sp.checked_sub(count * 4).ok_or(ErtError::Unexpected)?;
        self.adjust_sp(start);
        let mut slot = 0;
        for register in 0..REG_COUNT {
            if list & (1 << register) != 0 {
                let range = self.stack_bits((slot * 4) as i32, 32)?;
                self.vstack[range].clone_from_slice(&self.regs[register].clone());
                slot += 1;
            }
        }
        self.next(len)
    }

    fn pop(&mut self, list: u16, len: u32) -> Result<Flow, ErtError<E>> {
        let count = list.count_ones() as u32;
        let mut slot = 0;
        let mut returns = false;
        for register in 0..REG_COUNT {
            if list & (1 << register) != 0 {
                let range = self.stack_bits((slot * 4) as i32, 32)?;
                let word = array::from_fn(|bit| self.vstack[range.start + bit].clone());
                if register == PC as usize {
                    returns = true;
                } else {
                    self.write(register as u8, word, None);
                }
                slot += 1;
            }
        }
        self.adjust_sp(self.sp.wrapping_add(count * 4));
        if returns {
            self.return_from_call()
        } else {
            self.next(len)
        }
    }

    fn load_multiple(
        &mut self,
        base: u8,
        list: u16,
        write_back: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let base_offset = self.stack_offset(base, 0)?;
        let mut slot = 0;
        for register in 0..REG_COUNT {
            if list & (1 << register) != 0 {
                let range = self.stack_bits(base_offset.wrapping_add((slot * 4) as i32), 32)?;
                self.write(
                    register as u8,
                    array::from_fn(|bit| self.vstack[range.start + bit].clone()),
                    None,
                );
                slot += 1;
            }
        }
        if write_back {
            if base == SP {
                self.adjust_sp(self.sp.wrapping_add(slot * 4));
            } else {
                return Err(ErtError::Unexpected);
            }
        }
        self.next(len)
    }

    fn store_multiple(
        &mut self,
        base: u8,
        list: u16,
        write_back: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        let base_offset = self.stack_offset(base, 0)?;
        let mut slot = 0;
        for register in 0..REG_COUNT {
            if list & (1 << register) != 0 {
                let range = self.stack_bits(base_offset.wrapping_add((slot * 4) as i32), 32)?;
                self.vstack[range].clone_from_slice(&self.regs[register].clone());
                slot += 1;
            }
        }
        if write_back {
            if base == SP {
                self.adjust_sp(self.sp.wrapping_add(slot * 4));
            } else {
                return Err(ErtError::Unexpected);
            }
        }
        self.next(len)
    }

    fn call(&mut self, target: u32, len: u32) -> Result<Flow, ErtError<E>> {
        let return_pc = self.pc.wrapping_add(len);
        *self.rstack.get_mut(self.rsp).ok_or(ErtError::Unexpected)? = return_pc;
        self.rsp += 1;
        self.write_constant(LR, return_pc | 1);
        Ok(Flow::Next(target))
    }

    fn branch_register(&mut self, register: u8) -> Result<Flow, ErtError<E>> {
        if register == LR {
            return self.return_from_call();
        }
        let target = self.constants[register as usize].ok_or(ErtError::Unexpected)?;
        if target & 1 == 0 {
            return Err(ErtError::Unexpected);
        }
        Ok(Flow::Next(target & !1))
    }

    fn return_from_call(&mut self) -> Result<Flow, ErtError<E>> {
        self.rsp = self.rsp.checked_sub(1).ok_or(ErtError::Unexpected)?;
        Ok(Flow::Next(self.rstack[self.rsp]))
    }

    fn secure_gateway(&mut self, len: u32) -> Result<Flow, ErtError<E>> {
        if self.t.security_attribute(self.pc) != SecurityAttribute::NonSecure
            && self.security_state == SecurityState::NonSecure
        {
            self.security_state = SecurityState::Secure;
            if let Some(lr) = self.constants[LR as usize] {
                self.write_constant(LR, lr & !1);
            }
        }
        self.next(len)
    }

    fn branch_exchange_non_secure(
        &mut self,
        register: u8,
        link: bool,
        len: u32,
    ) -> Result<Flow, ErtError<E>> {
        if self.security_state == SecurityState::NonSecure {
            return Err(ErtError::Unexpected);
        }
        let target = self.constants[register as usize].ok_or(ErtError::Unexpected)?;
        if target & 1 == 0 {
            self.security_state = SecurityState::NonSecure;
        }
        if link {
            let return_pc = self.pc.wrapping_add(len);
            *self.rstack.get_mut(self.rsp).ok_or(ErtError::Unexpected)? = return_pc;
            self.rsp += 1;
            self.write_constant(LR, return_pc | 1);
        }
        Ok(Flow::Next(target & !1))
    }
}

fn fold(operation: BitOp, a: u32, b: u32) -> u32 {
    match operation {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Xor => a ^ b,
    }
}

fn degenerate(operation: BitOp, constant: u32) -> bool {
    match operation {
        BitOp::And => constant == 0,
        BitOp::Or => constant == u32::MAX,
        BitOp::Xor => false,
    }
}

fn degenerate_value(operation: BitOp) -> u32 {
    match operation {
        BitOp::And => 0,
        BitOp::Or => u32::MAX,
        BitOp::Xor => unreachable!("Xor is never degenerate"),
    }
}

fn load_width(kind: LoadKind) -> usize {
    match kind {
        LoadKind::ByteSigned | LoadKind::ByteUnsigned => 8,
        LoadKind::HalfSigned | LoadKind::HalfUnsigned => 16,
        LoadKind::Word => 32,
    }
}

fn arm_shift_count(amount: u32, direction: Shift) -> u32 {
    match direction {
        Shift::RotateRight => amount & 31,
        _ => (amount & 0xff).min(32),
    }
}

fn concrete_shift(value: u32, amount: u32, direction: Shift) -> u32 {
    let amount = arm_shift_count(amount, direction);
    match direction {
        Shift::Left => {
            if amount == 32 {
                0
            } else {
                value << amount
            }
        }
        Shift::LogicalRight => {
            if amount == 32 {
                0
            } else {
                value >> amount
            }
        }
        Shift::ArithmeticRight => {
            if amount == 32 {
                ((value as i32) >> 31) as u32
            } else {
                (value as i32 >> amount) as u32
            }
        }
        Shift::RotateRight => value.rotate_right(amount),
    }
}

fn arm_runtime_shift<W: Clone, E>(
    t: &mut (impl ContextWithArmv8mOps<bool, Wrapped = W, Error = E> + ?Sized),
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
    if !matches!(direction, Shift::RotateRight) {
        let mut saturated = zero.clone();
        for bit in 5..8 {
            saturated = t.bitor(saturated, amount[bit].clone())?;
        }
        let fill = match direction {
            Shift::ArithmeticRight => array::from_fn(|_| source[31].clone()),
            _ => array::from_fn(|_| zero.clone()),
        };
        output = select_word(t, saturated, &fill, &output)?;
    }
    Ok(output)
}

fn multiply_word<W: Clone, E>(
    t: &mut (impl ContextWithArmv8mOps<bool, Wrapped = W, Error = E> + ?Sized),
    left: &[W; 32],
    right: &[W; 32],
    left_constant: Option<u32>,
    right_constant: Option<u32>,
    product: Product,
    zero: &W,
    one: &W,
) -> Result<[W; 32], E> {
    if let (Some(left), Some(right)) = (left_constant, right_constant) {
        return Ok(constant_word(
            zero,
            one,
            concrete_product(product, left, right),
        ));
    }
    if left_constant == Some(0) || right_constant == Some(0) {
        return Ok(array::from_fn(|_| zero.clone()));
    }
    let wide = !matches!(product, Product::Low);
    let mut accumulator: [W; 64] = array::from_fn(|_| zero.clone());
    let multiplicand: [W; 64] = array::from_fn(|bit| {
        if bit < 32 {
            left[bit].clone()
        } else {
            zero.clone()
        }
    });
    let mut addend = multiplicand;
    let multiplier = right_constant.or(left_constant);
    let symbolic_multiplier = if right_constant.is_some() {
        left
    } else {
        right
    };
    for bit in 0..32 {
        let sum = add_bits(t, &accumulator, &addend, zero.clone())?;
        accumulator = match multiplier {
            Some(value) if value & (1 << bit) != 0 => sum,
            Some(_) => accumulator,
            None => select_word(t, symbolic_multiplier[bit].clone(), &sum, &accumulator)?,
        };
        addend = array::from_fn(|index| {
            if index == 0 {
                zero.clone()
            } else {
                addend[index - 1].clone()
            }
        });
    }
    let mut result = if wide {
        array::from_fn(|bit| accumulator[32 + bit].clone())
    } else {
        array::from_fn(|bit| accumulator[bit].clone())
    };
    if wide && matches!(product, Product::HighSigned | Product::HighSignedUnsigned) {
        result = subtract_if_negative(t, &result, left, right, left_constant, zero, one)?;
    }
    if wide && matches!(product, Product::HighSigned) {
        result = subtract_if_negative(t, &result, right, left, right_constant, zero, one)?;
    }
    Ok(result)
}

fn subtract_if_negative<W: Clone, E>(
    t: &mut (impl ContextWithArmv8mOps<bool, Wrapped = W, Error = E> + ?Sized),
    value: &[W; 32],
    signed: &[W; 32],
    subtrahend: &[W; 32],
    signed_constant: Option<u32>,
    _zero: &W,
    one: &W,
) -> Result<[W; 32], E> {
    let inverted = invert_word(t, subtrahend, one.clone())?;
    let difference = add_bits(t, value, &inverted, one.clone())?;
    match signed_constant {
        Some(constant) if constant >> 31 == 0 => Ok(value.clone()),
        Some(_) => Ok(difference),
        None => select_word(t, signed[31].clone(), &difference, value),
    }
}

fn decode16(pc: u32, instruction: u16) -> Result<Decoded, DecodeError> {
    let decoded = |operation| Ok::<_, DecodeError>(Decoded { operation, len: 2 });
    let bits = instruction as u32;
    if instruction == 0xbf00 {
        return decoded(Op::Nop);
    }
    if instruction & 0xff00 == 0xbf00 {
        let condition = ((instruction >> 4) & 15) as u8;
        let mask = (instruction & 15) as u8;
        return decoded(Op::It { condition, mask });
    }
    if instruction & 0xf800 == 0x0000 {
        return decoded(Op::Shift {
            dest: (instruction & 7) as u8,
            source: ((instruction >> 3) & 7) as u8,
            direction: Shift::Left,
            amount: ShiftAmount::Immediate(((instruction >> 6) & 31) as u32),
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x0800 {
        return decoded(Op::Shift {
            dest: (instruction & 7) as u8,
            source: ((instruction >> 3) & 7) as u8,
            direction: Shift::LogicalRight,
            amount: ShiftAmount::Immediate(nonzero_shift((instruction >> 6) & 31)),
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x1000 {
        return decoded(Op::Shift {
            dest: (instruction & 7) as u8,
            source: ((instruction >> 3) & 7) as u8,
            direction: Shift::ArithmeticRight,
            amount: ShiftAmount::Immediate(nonzero_shift((instruction >> 6) & 31)),
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x1800 {
        let subtract = instruction & 0x0200 != 0;
        let immediate = instruction & 0x0400 != 0;
        let right = if immediate {
            Operand::Immediate(((instruction >> 6) & 7) as u32)
        } else {
            Operand::Register(((instruction >> 6) & 7) as u8)
        };
        return decoded(Op::Arithmetic {
            kind: if subtract {
                Arithmetic::Sub
            } else {
                Arithmetic::Add
            },
            dest: (instruction & 7) as u8,
            left: Operand::Register(((instruction >> 3) & 7) as u8),
            right,
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x2000 {
        return decoded(Op::Move {
            dest: ((instruction >> 8) & 7) as u8,
            source: Operand::Immediate((instruction & 255) as u32),
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x2800 {
        return decoded(Op::Compare {
            left: Operand::Register(((instruction >> 8) & 7) as u8),
            right: Operand::Immediate((instruction & 255) as u32),
            add: false,
        });
    }
    if instruction & 0xf800 == 0x3000 {
        return decoded(Op::Arithmetic {
            kind: Arithmetic::Add,
            dest: ((instruction >> 8) & 7) as u8,
            left: Operand::Register(((instruction >> 8) & 7) as u8),
            right: Operand::Immediate((instruction & 255) as u32),
            set_flags: true,
        });
    }
    if instruction & 0xf800 == 0x3800 {
        return decoded(Op::Arithmetic {
            kind: Arithmetic::Sub,
            dest: ((instruction >> 8) & 7) as u8,
            left: Operand::Register(((instruction >> 8) & 7) as u8),
            right: Operand::Immediate((instruction & 255) as u32),
            set_flags: true,
        });
    }
    if instruction & 0xfc00 == 0x4000 {
        return decode16_data(instruction);
    }
    if instruction & 0xfc00 == 0x4400 {
        return decode16_special(instruction);
    }
    if instruction & 0xf800 == 0x4800 {
        return decoded(Op::Load {
            dest: ((instruction >> 8) & 7) as u8,
            base: PC,
            offset: ((instruction & 255) << 2) as i32,
            kind: LoadKind::Word,
        });
    }
    if instruction & 0xf200 == 0x5000 {
        return decode16_register_load_store(instruction);
    }
    if instruction & 0xe000 == 0x6000 {
        return decode16_immediate_load_store(instruction);
    }
    if instruction & 0xf000 == 0x8000 {
        let load = instruction & 0x0800 != 0;
        return decoded(if load {
            Op::Load {
                dest: (instruction & 7) as u8,
                base: ((instruction >> 3) & 7) as u8,
                offset: (((instruction >> 6) & 31) * 2) as i32,
                kind: LoadKind::HalfUnsigned,
            }
        } else {
            Op::Store {
                source: (instruction & 7) as u8,
                base: ((instruction >> 3) & 7) as u8,
                offset: (((instruction >> 6) & 31) * 2) as i32,
                width: 16,
            }
        });
    }
    if instruction & 0xf000 == 0x9000 {
        let load = instruction & 0x0800 != 0;
        return decoded(if load {
            Op::Load {
                dest: ((instruction >> 8) & 7) as u8,
                base: SP,
                offset: ((instruction & 255) * 4) as i32,
                kind: LoadKind::Word,
            }
        } else {
            Op::Store {
                source: ((instruction >> 8) & 7) as u8,
                base: SP,
                offset: ((instruction & 255) * 4) as i32,
                width: 32,
            }
        });
    }
    if instruction & 0xf800 == 0xa000 {
        return decoded(Op::Move {
            dest: ((instruction >> 8) & 7) as u8,
            source: Operand::Immediate(
                (pc.wrapping_add(4) & !3).wrapping_add(((instruction & 255) << 2) as u32),
            ),
            set_flags: false,
        });
    }
    if instruction & 0xf800 == 0xa800 {
        return decoded(Op::Arithmetic {
            kind: Arithmetic::Add,
            dest: ((instruction >> 8) & 7) as u8,
            left: Operand::Register(SP),
            right: Operand::Immediate(((instruction & 255) << 2) as u32),
            set_flags: false,
        });
    }
    if instruction & 0xff00 == 0xb000 {
        return decoded(Op::Arithmetic {
            kind: if instruction & 0x0080 == 0 {
                Arithmetic::Add
            } else {
                Arithmetic::Sub
            },
            dest: SP,
            left: Operand::Register(SP),
            right: Operand::Immediate(((instruction & 0x7f) << 2) as u32),
            set_flags: false,
        });
    }
    if instruction & 0xf500 == 0xb100 {
        let nonzero = instruction & 0x0800 != 0;
        let immediate = (((instruction >> 9) & 1) << 6) | (((instruction >> 3) & 31) << 1);
        return decoded(Op::CompareBranch {
            register: (instruction & 7) as u8,
            nonzero,
            target: pc.wrapping_add(4).wrapping_add(immediate as u32),
        });
    }
    if instruction & 0xfe00 == 0xb400 {
        let list = (instruction & 255)
            | if instruction & 0x0100 != 0 {
                1 << LR
            } else {
                0
            };
        return decoded(Op::Push { list });
    }
    if instruction & 0xfe00 == 0xbc00 {
        let list = (instruction & 255)
            | if instruction & 0x0100 != 0 {
                1 << PC
            } else {
                0
            };
        return decoded(Op::Pop { list });
    }
    if instruction & 0xf000 == 0xc000 {
        let list = instruction & 255;
        return decoded(if instruction & 0x0800 != 0 {
            Op::LoadMultiple {
                base: ((instruction >> 8) & 7) as u8,
                list,
                write_back: true,
            }
        } else {
            Op::StoreMultiple {
                base: ((instruction >> 8) & 7) as u8,
                list,
                write_back: true,
            }
        });
    }
    if instruction & 0xf000 == 0xd000 {
        let condition = ((instruction >> 8) & 15) as u8;
        if condition == 15 {
            return decoded(Op::Svc((instruction & 255) as u8));
        }
        if condition == 14 {
            return Err(DecodeError::Unsupported(bits));
        }
        return decoded(Op::Branch {
            target: pc
                .wrapping_add(4)
                .wrapping_add(sign_extend(((instruction & 255) as u32) << 1, 9)),
            condition: Some(condition),
        });
    }
    if instruction & 0xf800 == 0xe000 {
        return decoded(Op::Branch {
            target: pc
                .wrapping_add(4)
                .wrapping_add(sign_extend(((instruction & 0x7ff) as u32) << 1, 12)),
            condition: None,
        });
    }
    Err(DecodeError::Unsupported(bits))
}

fn decode16_data(instruction: u16) -> Result<Decoded, DecodeError> {
    let dest = (instruction & 7) as u8;
    let source = ((instruction >> 3) & 7) as u8;
    let op = (instruction >> 6) & 15;
    let operation = match op {
        0 => Op::Bitwise {
            kind: BitOp::And,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        1 => Op::Bitwise {
            kind: BitOp::Xor,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        2 => Op::Shift {
            dest,
            source: dest,
            direction: Shift::Left,
            amount: ShiftAmount::Register(source),
            set_flags: true,
        },
        3 => Op::Shift {
            dest,
            source: dest,
            direction: Shift::LogicalRight,
            amount: ShiftAmount::Register(source),
            set_flags: true,
        },
        4 => Op::Shift {
            dest,
            source: dest,
            direction: Shift::ArithmeticRight,
            amount: ShiftAmount::Register(source),
            set_flags: true,
        },
        5 => Op::Arithmetic {
            kind: Arithmetic::AddCarry,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        6 => Op::Arithmetic {
            kind: Arithmetic::SubCarry,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        7 => Op::Shift {
            dest,
            source: dest,
            direction: Shift::RotateRight,
            amount: ShiftAmount::Register(source),
            set_flags: true,
        },
        8 => Op::Test {
            left: Operand::Register(dest),
            right: Operand::Register(source),
        },
        9 => Op::Arithmetic {
            kind: Arithmetic::ReverseSub,
            dest,
            left: Operand::Register(source),
            right: Operand::Immediate(0),
            set_flags: true,
        },
        10 => Op::Compare {
            left: Operand::Register(dest),
            right: Operand::Register(source),
            add: false,
        },
        11 => Op::Compare {
            left: Operand::Register(dest),
            right: Operand::Register(source),
            add: true,
        },
        12 => Op::Bitwise {
            kind: BitOp::Or,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        13 => Op::Multiply {
            dest,
            left: dest,
            right: source,
            add: None,
            subtract: false,
        },
        14 => Op::BitClear {
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: true,
        },
        15 => Op::MoveNot {
            dest,
            source: Operand::Register(source),
            set_flags: true,
        },
        _ => unreachable!(),
    };
    Ok(Decoded { operation, len: 2 })
}

fn decode16_special(instruction: u16) -> Result<Decoded, DecodeError> {
    let operation = (instruction >> 8) & 3;
    let dest = ((instruction & 7) | ((instruction >> 4) & 8)) as u8;
    let source = ((instruction >> 3) & 15) as u8;
    let operation = match operation {
        0 => Op::Arithmetic {
            kind: Arithmetic::Add,
            dest,
            left: Operand::Register(dest),
            right: Operand::Register(source),
            set_flags: false,
        },
        1 => Op::Compare {
            left: Operand::Register(dest),
            right: Operand::Register(source),
            add: false,
        },
        2 => Op::Move {
            dest,
            source: Operand::Register(source),
            set_flags: false,
        },
        3 => {
            let non_secure = instruction & 0x0004 != 0;
            match (instruction & 0x0080 == 0, non_secure) {
                (true, false) => Op::BranchRegister { register: source },
                (true, true) => Op::BranchExchangeNonSecure {
                    register: source,
                    link: false,
                },
                (false, false) => Op::CallRegister { register: source },
                (false, true) => Op::BranchExchangeNonSecure {
                    register: source,
                    link: true,
                },
            }
        }
        _ => unreachable!(),
    };
    Ok(Decoded { operation, len: 2 })
}

fn decode16_register_load_store(instruction: u16) -> Result<Decoded, DecodeError> {
    let op = (instruction >> 9) & 7;
    let offset_register = ((instruction >> 6) & 7) as u8;
    let base = ((instruction >> 3) & 7) as u8;
    let register = (instruction & 7) as u8;
    let operation = match op {
        0 => Op::StoreIndexed {
            source: register,
            base,
            index: offset_register,
            shift: 0,
            width: 32,
        },
        1 => Op::StoreIndexed {
            source: register,
            base,
            index: offset_register,
            shift: 0,
            width: 16,
        },
        2 => Op::StoreIndexed {
            source: register,
            base,
            index: offset_register,
            shift: 0,
            width: 8,
        },
        3 => Op::LoadIndexed {
            dest: register,
            base,
            index: offset_register,
            shift: 0,
            kind: LoadKind::ByteSigned,
        },
        4 => Op::LoadIndexed {
            dest: register,
            base,
            index: offset_register,
            shift: 0,
            kind: LoadKind::Word,
        },
        5 => Op::LoadIndexed {
            dest: register,
            base,
            index: offset_register,
            shift: 0,
            kind: LoadKind::HalfUnsigned,
        },
        6 => Op::LoadIndexed {
            dest: register,
            base,
            index: offset_register,
            shift: 0,
            kind: LoadKind::ByteUnsigned,
        },
        7 => Op::LoadIndexed {
            dest: register,
            base,
            index: offset_register,
            shift: 0,
            kind: LoadKind::HalfSigned,
        },
        _ => unreachable!(),
    };
    Ok(Decoded { operation, len: 2 })
}

fn decode16_immediate_load_store(instruction: u16) -> Result<Decoded, DecodeError> {
    let byte = instruction & 0xf000 == 0x7000;
    let load = instruction & 0x0800 != 0;
    let scale = if byte { 1 } else { 4 };
    let operation = if load {
        Op::Load {
            dest: (instruction & 7) as u8,
            base: ((instruction >> 3) & 7) as u8,
            offset: (((instruction >> 6) & 31) * scale) as i32,
            kind: if byte {
                LoadKind::ByteUnsigned
            } else {
                LoadKind::Word
            },
        }
    } else {
        Op::Store {
            source: (instruction & 7) as u8,
            base: ((instruction >> 3) & 7) as u8,
            offset: (((instruction >> 6) & 31) * scale) as i32,
            width: if byte { 8 } else { 32 },
        }
    };
    Ok(Decoded { operation, len: 2 })
}

fn decode32(pc: u32, first: u16, second: u16) -> Result<Decoded, DecodeError> {
    let full = ((first as u32) << 16) | second as u32;
    let decoded = |operation| Ok::<_, DecodeError>(Decoded { operation, len: 4 });
    let dest = ((second >> 8) & 15) as u8;
    let data_register = ((second >> 12) & 15) as u8;
    let source = (second & 15) as u8;

    // SG: the Secure Gateway instruction, a fixed 32-bit encoding with both
    // halfwords identical.
    if first == 0xe97f && second == 0xe97f {
        return decoded(Op::SecureGateway);
    }

    // Load/store multiple and the wide PUSH/POP forms used by compiler
    // prologues. The shared stack handlers retain their usual static model.
    if first == 0xe92d {
        return decoded(Op::Push { list: second });
    }
    if first == 0xe8bd {
        return decoded(Op::Pop { list: second });
    }
    if first & 0xfff0 == 0xe880 {
        return decoded(Op::StoreMultiple {
            base: (first & 15) as u8,
            list: second,
            write_back: first & 0x20 != 0,
        });
    }
    if first & 0xfff0 == 0xe890 {
        return decoded(Op::LoadMultiple {
            base: (first & 15) as u8,
            list: second,
            write_back: first & 0x20 != 0,
        });
    }
    if first & 0xfff0 == 0xe9c0 {
        return decoded(Op::StoreDouble {
            first: ((second >> 12) & 15) as u8,
            second: ((second >> 8) & 15) as u8,
            base: (first & 15) as u8,
            offset: ((second & 255) as i32) * 4,
        });
    }

    // T32 data processing (shifted register). The imm3:imm2 field is shared
    // by MOV/shift and the AND/ORR/EOR/BIC families.
    if matches!(first & 0xffe0, 0xea00 | 0xea20 | 0xea40 | 0xea80) {
        let amount = (((second >> 12) & 7) << 2 | ((second >> 6) & 3)) as u32;
        let shift = match (second >> 4) & 3 {
            0 => Shift::Left,
            1 => Shift::LogicalRight,
            2 => Shift::ArithmeticRight,
            _ => Shift::RotateRight,
        };
        let left = Operand::Register((first & 15) as u8);
        let right = Operand::Shifted {
            register: source,
            shift,
            amount: ShiftAmount::Immediate(amount),
        };
        return match first & 0xfff0 {
            0xea00 => decoded(Op::Bitwise {
                kind: BitOp::And,
                dest,
                left,
                right,
                set_flags: false,
            }),
            0xea20 => decoded(Op::BitClear {
                dest,
                left,
                right,
                set_flags: false,
            }),
            0xea40 if first & 15 == 15 => decoded(Op::Shift {
                dest,
                source,
                direction: shift,
                amount: ShiftAmount::Immediate(amount),
                set_flags: false,
            }),
            0xea40 => decoded(Op::Bitwise {
                kind: BitOp::Or,
                dest,
                left,
                right,
                set_flags: false,
            }),
            0xea80 => decoded(Op::Bitwise {
                kind: BitOp::Xor,
                dest,
                left,
                right,
                set_flags: false,
            }),
            _ => Err(DecodeError::Unsupported(full)),
        };
    }
    if first & 0xffe0 == 0xeb00 || first & 0xffe0 == 0xeba0 {
        let amount = (((second >> 12) & 7) << 2 | ((second >> 6) & 3)) as u32;
        let shift = match (second >> 4) & 3 {
            0 => Shift::Left,
            1 => Shift::LogicalRight,
            2 => Shift::ArithmeticRight,
            _ => Shift::RotateRight,
        };
        return decoded(Op::Arithmetic {
            kind: if first & 0xffe0 == 0xeb00 {
                Arithmetic::Add
            } else {
                Arithmetic::Sub
            },
            dest,
            left: Operand::Register((first & 15) as u8),
            right: Operand::Shifted {
                register: source,
                shift,
                amount: ShiftAmount::Immediate(amount),
            },
            set_flags: false,
        });
    }

    // Single word loads/stores: positive immediate-offset and register-indexed
    // forms. Indexed stack references require a concrete index, preserving the
    // same address discipline as the RISC-V facade.
    if first & 0xfff0 == 0xf8d0 {
        return decoded(Op::Load {
            dest: data_register,
            base: (first & 15) as u8,
            offset: (second & 0x0fff) as i32,
            kind: LoadKind::Word,
        });
    }
    if first & 0xfff0 == 0xf8c0 {
        return decoded(Op::Store {
            source: data_register,
            base: (first & 15) as u8,
            offset: (second & 0x0fff) as i32,
            width: 32,
        });
    }
    if first & 0xfff0 == 0xf850 {
        return decoded(Op::LoadIndexed {
            dest: ((second >> 12) & 15) as u8,
            base: (first & 15) as u8,
            index: source,
            shift: ((second >> 4) & 3) as u8,
            kind: LoadKind::Word,
        });
    }
    if first & 0xfff0 == 0xf840 {
        return decoded(Op::StoreIndexed {
            source: ((second >> 12) & 15) as u8,
            base: (first & 15) as u8,
            index: source,
            shift: ((second >> 4) & 3) as u8,
            width: 32,
        });
    }

    if first & 0xfff0 == 0xfb00 {
        let accumulate = ((second >> 12) & 15) as u8;
        return decoded(Op::Multiply {
            dest,
            left: (first & 15) as u8,
            right: source,
            add: (accumulate != PC).then_some(accumulate),
            subtract: second & 0x10 != 0,
        });
    }
    if first & 0xfff0 == 0xfba0 || first & 0xfff0 == 0xfb80 {
        return decoded(Op::LongMultiply {
            dest_low: ((second >> 12) & 15) as u8,
            dest_high: ((second >> 8) & 15) as u8,
            left: (first & 15) as u8,
            right: source,
            signed: first & 0xfff0 == 0xfb80,
        });
    }

    // Thumb modified-immediate logical/arithmetic instructions. ThumbExpandImm
    // is used for both the compact and rotated immediate forms.
    if first & 0xf800 == 0xf000
        && first & 0xfbf0 != 0xf240
        && first & 0xfbf0 != 0xf2c0
        && first & 0xfbf0 != 0xf200
        && !(first & 0xf800 == 0xf000 && second & 0xd000 == 0xd000)
    {
        let immediate = thumb_expand_imm(first, second);
        let left = Operand::Register((first & 15) as u8);
        let opcode = first & 0xfbe0;
        if first & 0xfbff == 0xf04f {
            return decoded(Op::Move {
                dest,
                source: Operand::Immediate(immediate),
                set_flags: false,
            });
        }
        let operation = match opcode {
            0xf000 => Op::Bitwise {
                kind: BitOp::And,
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            0xf020 => Op::BitClear {
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            0xf040 => Op::Bitwise {
                kind: BitOp::Or,
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            0xf080 => Op::Bitwise {
                kind: BitOp::Xor,
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            0xf100 => Op::Arithmetic {
                kind: Arithmetic::Add,
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            0xf1a0 if dest == PC => Op::Compare {
                left,
                right: Operand::Immediate(immediate),
                add: false,
            },
            0xf1a0 => Op::Arithmetic {
                kind: Arithmetic::Sub,
                dest,
                left,
                right: Operand::Immediate(immediate),
                set_flags: false,
            },
            _ => return Err(DecodeError::Unsupported(full)),
        };
        return decoded(operation);
    }
    // MOVW and MOVT, T32 encoding A1.
    if first & 0xfbf0 == 0xf240 || first & 0xfbf0 == 0xf2c0 {
        let immediate = (((first as u32 >> 10) & 1) << 11)
            | (((first as u32) & 15) << 12)
            | (((second as u32 >> 12) & 7) << 8)
            | (second as u32 & 255);
        let dest = ((second >> 8) & 15) as u8;
        if first & 0xfbf0 == 0xf240 {
            return decoded(Op::Move {
                dest,
                source: Operand::Immediate(immediate),
                set_flags: false,
            });
        }
        return decoded(Op::MoveTop {
            dest,
            immediate: immediate as u16,
        });
    }
    // ADDW uses a plain 12-bit immediate rather than ThumbExpandImm.
    if first & 0xfbf0 == 0xf200 {
        let immediate = (((first as u32 >> 10) & 1) << 11)
            | (((first as u32) & 15) << 12)
            | (((second as u32 >> 12) & 7) << 8)
            | (second as u32 & 255);
        return decoded(Op::Arithmetic {
            kind: Arithmetic::Add,
            dest,
            left: Operand::Register((first & 15) as u8),
            right: Operand::Immediate(immediate),
            set_flags: false,
        });
    }
    // BL / B.W share the split immediate construction. BL has bit 12 set in
    // the second halfword; B.W has it clear.
    if first & 0xf800 == 0xf000 && second & 0xd000 == 0xd000 {
        let s = ((first >> 10) & 1) as u32;
        let j1 = ((second >> 13) & 1) as u32;
        let j2 = ((second >> 11) & 1) as u32;
        let i1 = !(j1 ^ s) & 1;
        let i2 = !(j2 ^ s) & 1;
        let immediate = sign_extend(
            (s << 24)
                | (i1 << 23)
                | (i2 << 22)
                | (((first as u32) & 0x03ff) << 12)
                | (((second as u32) & 0x07ff) << 1),
            25,
        );
        let target = pc.wrapping_add(4).wrapping_add(immediate);
        return if second & 0x1000 != 0 {
            decoded(Op::Call { target })
        } else {
            decoded(Op::Branch {
                target,
                condition: None,
            })
        };
    }
    Err(DecodeError::Unsupported(full))
}

fn thumb_expand_imm(first: u16, second: u16) -> u32 {
    let imm12 = (((first as u32 >> 10) & 1) << 11)
        | (((second as u32 >> 12) & 7) << 8)
        | (second as u32 & 255);
    match imm12 >> 10 {
        0 => match (imm12 >> 8) & 3 {
            0 => imm12 & 255,
            1 => (imm12 & 255) * 0x0001_0001,
            2 => (imm12 & 255) * 0x0100_0100,
            _ => (imm12 & 255) * 0x0101_0101,
        },
        _ => (0x80 | (imm12 & 0x7f)).rotate_right((imm12 >> 7) & 31),
    }
}

fn nonzero_shift(value: u16) -> u32 {
    if value == 0 { 32 } else { value as u32 }
}

fn sign_extend(value: u32, bits: u32) -> u32 {
    ((value << (32 - bits)) as i32 >> (32 - bits)) as u32
}
