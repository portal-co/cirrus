use core::{array, error::Error, mem::MaybeUninit, ops::Range};

pub(crate) use cirrus_ert_core::add_bits;
use rv_asm::{Imm, Inst, Reg, Xlen};

use crate::{EcallOutcome, ErtError, RawMemory, handlers};

/// Object-safe bridge which keeps the machine monomorphic while its caller
/// supplies arbitrary `ContextWithStorage` implementations.
#[doc(hidden)]
pub trait Runtime<W, const BITS: usize>:
    cirrus_ert_core::ContextWithErtOps<bool, Wrapped = W>
{
    fn ecall(
        &mut self,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, Self::Error>;

    fn call_hook(
        &mut self,
        event: crate::CallEvent,
        regs: &mut [[W; BITS]],
        reg_consts: &mut [Option<u64>],
        offsets: &mut [Option<i64>],
        zero: &W,
        one: &W,
    ) -> Result<crate::CallAction, Self::Error>;

    fn storage_read_bit(&mut self, bit: usize) -> Result<W, Self::Error>;

    fn storage_write_bit(&mut self, bit: usize, value: W) -> Result<(), Self::Error>;

    #[cfg(feature = "early-exit-loops")]
    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions;
}

#[doc(hidden)]
pub const ABI_REGS: [Reg; 8] = [
    Reg::A0,
    Reg::A1,
    Reg::A2,
    Reg::A3,
    Reg::A4,
    Reg::A5,
    Reg::A6,
    Reg::A7,
];

/// The all-ones mask of a `BITS`-wide concrete word.
pub(crate) const fn word_mask<const BITS: usize>() -> u64 {
    if BITS == 64 { u64::MAX } else { (1u64 << BITS) - 1 }
}

/// Sign-extend the low `BITS` of `value` into an `i64`.
pub(crate) fn sext<const BITS: usize>(value: u64) -> i64 {
    ((value << (64 - BITS)) as i64) >> (64 - BITS)
}

/// The concrete return-stack word type. RV32 callers keep their historical
/// `&mut [u32]` stacks; RV64 uses `u64`. This keeps the machine free of
/// width-conversion buffers (no allocation anywhere in the interpreter).
#[doc(hidden)]
pub trait RstackWord: Copy {
    fn from_u64(value: u64) -> Self;
    fn into_u64(self) -> u64;
}

impl RstackWord for u32 {
    fn from_u64(value: u64) -> Self {
        value as u32
    }

    fn into_u64(self) -> u64 {
        u64::from(self)
    }
}

impl RstackWord for u64 {
    fn from_u64(value: u64) -> Self {
        value
    }

    fn into_u64(self) -> u64 {
        self
    }
}

#[doc(hidden)]
pub struct Machine<'a, W, E, const BITS: usize, R: RstackWord> {
    #[doc(hidden)]
    pub t: &'a mut (dyn Runtime<W, BITS, Error = E> + 'a),
    #[doc(hidden)]
    pub mem: RawMemory<'a>,
    #[doc(hidden)]
    pub rstack: &'a mut [R],
    #[doc(hidden)]
    pub storage_bits: usize,
    #[doc(hidden)]
    pub pc: u64,
    /// Byte length of the instruction most recently decoded at `pc`
    /// (4 for normal encodings, 2 for compressed). Set by [`Machine::run`]
    /// before every handler invocation.
    #[doc(hidden)]
    pub inst_len: u64,
    #[doc(hidden)]
    pub regs: &'a mut [[W; BITS]; 32],
    #[doc(hidden)]
    pub reg_consts: &'a mut [Option<u64>; 32],
    #[doc(hidden)]
    pub zero: W,
    #[doc(hidden)]
    pub one: W,
    #[doc(hidden)]
    pub sp: u64,
    #[doc(hidden)]
    pub stack_top: u64,
    #[doc(hidden)]
    pub rsp: u64,
    #[doc(hidden)]
    pub offs: [Option<i64>; 32],
    #[cfg(feature = "early-exit-loops")]
    pub(crate) loop_sites: [Option<crate::early_exit::RecognizedSite>; 8],
}

pub(crate) enum LoadAddress {
    Stack(u64),
    Concrete(u64),
}

impl<'a, W: Clone, E: Error, const BITS: usize, R: RstackWord> Machine<'a, W, E, BITS, R> {
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn new(
        t: &'a mut (dyn Runtime<W, BITS, Error = E> + 'a),
        mem: RawMemory<'a>,
        rstack: &'a mut [R],
        storage_bits: usize,
        pc: u64,
        regs: &'a mut [[W; BITS]; 32],
        reg_consts: &'a mut [Option<u64>; 32],
        zero: W,
        one: W,
        sp: u64,
    ) -> Self {
        Self {
            t,
            mem,
            rstack,
            storage_bits,
            pc,
            regs,
            reg_consts,
            zero,
            one,
            sp,
            stack_top: sp,
            rsp: 0,
            inst_len: 4,
            offs: [const { None }; 32],
            #[cfg(feature = "early-exit-loops")]
            loop_sites: [const { None }; 8],
        }
    }

    pub(crate) fn run(mut self) -> Result<(), ErtError<E>> {
        loop {
            // Compressed instructions permit two-byte alignment; an odd PC
            // is never valid for fetch and fails closed.
            if self.pc & 1 != 0 {
                return Err(ErtError::Unexpected);
            }
            let (instruction, len) = self.decode()?;
            self.inst_len = len;
            self.reset_fixed_registers();
            match handlers::execute(&mut self, instruction)? {
                handlers::Flow::Next(pc) => self.pc = pc,
                handlers::Flow::Exit => return Ok(()),
            }
        }
    }

    #[doc(hidden)]
    pub fn decode(&self) -> Result<(Inst, u64), ErtError<E>> {
        let xlen = if BITS == 64 { Xlen::Rv64 } else { Xlen::Rv32 };
        let half = u16::from_le_bytes(self.mem.read64(self.pc).ok_or(ErtError::Unexpected)?);
        if Inst::first_byte_is_compressed(half as u8) {
            return Inst::decode_compressed(half, xlen)
                .map(|instruction| (instruction, 2))
                .map_err(ErtError::Decode);
        }
        let word = u32::from_le_bytes(self.mem.read64(self.pc).ok_or(ErtError::Unexpected)?);
        rv_asm::Inst::decode(word, xlen)
            .map(|(instruction, _)| (instruction, 4))
            .map_err(ErtError::Decode)
    }

    #[doc(hidden)]
    pub fn reset_fixed_registers(&mut self) {
        for register_bit in self.regs[Reg::ZERO.0 as usize].iter_mut() {
            *register_bit = self.zero.clone();
        }
        self.reg_consts[Reg::ZERO.0 as usize] = Some(0);
        self.offs[Reg::ZERO.0 as usize] = None;

        self.reg_consts[Reg::SP.0 as usize] = None;
        self.offs[Reg::SP.0 as usize] = Some(0);
        for (i, register_bit) in self.regs[Reg::SP.0 as usize].iter_mut().enumerate() {
            *register_bit = if (self.sp >> i) & 1 == 0 {
                self.zero.clone()
            } else {
                self.one.clone()
            };
        }
    }

    #[doc(hidden)]
    pub fn word_from_constant(&self, value: u64) -> [W; BITS] {
        let value = value & word_mask::<BITS>();
        array::from_fn(|i| {
            if (value >> i) & 1 == 0 {
                self.zero.clone()
            } else {
                self.one.clone()
            }
        })
    }

    #[doc(hidden)]
    pub fn write_constant(&mut self, dest: Reg, value: u64) {
        self.offs[dest.0 as usize] = None;
        self.reg_consts[dest.0 as usize] = Some(value & word_mask::<BITS>());
        self.regs[dest.0 as usize] = self.word_from_constant(value);
    }

    pub(crate) fn stack_offset(&self, base: Reg, offset: Imm) -> Result<u64, ErtError<E>> {
        let relative = match base {
            x if x == Reg::SP => i64::from(offset.as_i32()),
            x if self.offs[x.0 as usize].is_some() => self.offs[base.0 as usize]
                .unwrap()
                .wrapping_add(i64::from(offset.as_i32())),
            _ => return Err(ErtError::Unexpected),
        };
        Ok(self.sp.wrapping_add_signed(relative))
    }

    pub(crate) fn stack_bits(
        &self,
        address: u64,
        width: usize,
    ) -> Result<Range<usize>, ErtError<E>> {
        let start = usize::try_from(address)
            .ok()
            .and_then(|address| address.checked_mul(8))
            .ok_or(ErtError::Unexpected)?;
        let end = start.checked_add(width).ok_or(ErtError::Unexpected)?;
        if end > self.storage_bits {
            return Err(ErtError::Unexpected);
        }
        Ok(start..end)
    }

    pub(crate) fn read_stack_bit(&mut self, bit: usize) -> Result<W, ErtError<E>> {
        if bit >= self.storage_bits {
            return Err(ErtError::Unexpected);
        }
        self.t.storage_read_bit(bit).map_err(ErtError::Emitted)
    }

    pub(crate) fn write_stack_bit(&mut self, bit: usize, value: W) -> Result<(), ErtError<E>> {
        if bit >= self.storage_bits {
            return Err(ErtError::Unexpected);
        }
        self.t
            .storage_write_bit(bit, value)
            .map_err(ErtError::Emitted)
    }

    pub(crate) fn load_address(&self, base: Reg, offset: Imm) -> Result<LoadAddress, ErtError<E>> {
        match base {
            x if x == Reg::SP || self.offs[x.0 as usize].is_some() => {
                Ok(LoadAddress::Stack(self.stack_offset(base, offset)?))
            }
            x if self.reg_consts[x.0 as usize].is_some() => Ok(LoadAddress::Concrete(
                self.reg_consts[x.0 as usize].unwrap(),
            )),
            _ => Err(ErtError::Unexpected),
        }
    }
}

#[doc(hidden)]
pub fn write_abi_args<W, E: Error, const N: usize, const BITS: usize>(
    t: &mut (dyn Runtime<W, BITS, Error = E> + '_),
    regs: &mut [[W; BITS]; 32],
    reg_consts: &mut [Option<u64>; 32],
    storage_bits: usize,
    sp: u64,
    args: [([W; BITS], Option<u64>); N],
) -> Result<(), E> {
    for (i, (value, constant)) in args.into_iter().enumerate() {
        match ABI_REGS.get(i).copied() {
            Some(register) => {
                regs[register.0 as usize] = value;
                reg_consts[register.0 as usize] = constant;
            }
            None => {
                let stack_start = sp as usize * 8 + BITS * (i - ABI_REGS.len());
                debug_assert!(stack_start + BITS <= storage_bits);
                for (bit, value) in value.into_iter().enumerate() {
                    t.storage_write_bit(stack_start + bit, value)?;
                }
            }
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn read_abi_results<W: Clone, E: Error, const M: usize, const BITS: usize>(
    t: &mut (dyn Runtime<W, BITS, Error = E> + '_),
    regs: &[[W; BITS]; 32],
    reg_consts: &[Option<u64>; 32],
    storage_bits: usize,
    sp: u64,
) -> Result<[([W; BITS], Option<u64>); M], E> {
    let mut results: [MaybeUninit<([W; BITS], Option<u64>)>; M] =
        [const { MaybeUninit::uninit() }; M];
    for (i, result) in results.iter_mut().enumerate() {
        result.write(match ABI_REGS.get(i).copied() {
            Some(register) => (
                regs[register.0 as usize].clone(),
                reg_consts[register.0 as usize],
            ),
            None => {
                let stack_start = sp as usize * 8 + BITS * (i - ABI_REGS.len());
                debug_assert!(stack_start + BITS <= storage_bits);
                let mut word: [MaybeUninit<W>; BITS] = [const { MaybeUninit::uninit() }; BITS];
                for (bit, slot) in word.iter_mut().enumerate() {
                    slot.write(t.storage_read_bit(stack_start + bit)?);
                }
                let word = word.map(|bit| unsafe { bit.assume_init() });
                (word, None)
            }
        });
    }
    Ok(results.map(|result| unsafe { result.assume_init() }))
}
