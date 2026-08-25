use core::{array, error::Error, mem::MaybeUninit, ops::Range};

pub(crate) use cirrus_ert_core::add_bits;
use rv_asm::{Imm, Inst, Reg, Xlen};

use crate::{EcallOutcome, ErtError, RawMemory, handlers};

/// Object-safe bridge which keeps the machine monomorphic while its caller
/// supplies arbitrary `ContextWithStorage` implementations.
pub(crate) trait Runtime<W>: cirrus_ert_core::ContextWithErtOps<bool, Wrapped = W> {
    fn ecall(
        &mut self,
        regs: &mut [[W; 32]],
        reg_consts: &mut [Option<u32>],
        offsets: &mut [Option<i32>],
        zero: &W,
        one: &W,
    ) -> Result<EcallOutcome, Self::Error>;

    fn storage_read_bit(&mut self, bit: usize) -> Result<W, Self::Error>;

    fn storage_write_bit(&mut self, bit: usize, value: W) -> Result<(), Self::Error>;

    #[cfg(feature = "early-exit-loops")]
    fn early_exit_loop_options(&self) -> cirrus_ert_core::EarlyExitLoopOptions;
}

pub(crate) const ABI_REGS: [Reg; 8] = [
    Reg::A0,
    Reg::A1,
    Reg::A2,
    Reg::A3,
    Reg::A4,
    Reg::A5,
    Reg::A6,
    Reg::A7,
];

pub(crate) struct Machine<'a, W, E> {
    pub(crate) t: &'a mut (dyn Runtime<W, Error = E> + 'a),
    pub(crate) mem: RawMemory<'a>,
    pub(crate) rstack: &'a mut [u32],
    pub(crate) storage_bits: usize,
    pub(crate) pc: u32,
    pub(crate) regs: &'a mut [[W; 32]; 32],
    pub(crate) reg_consts: &'a mut [Option<u32>; 32],
    pub(crate) zero: W,
    pub(crate) one: W,
    pub(crate) sp: u32,
    pub(crate) stack_top: u32,
    pub(crate) rsp: u32,
    pub(crate) offs: [Option<i32>; 32],
    #[cfg(feature = "early-exit-loops")]
    pub(crate) loop_sites: [Option<crate::early_exit::RecognizedSite>; 8],
}

pub(crate) enum LoadAddress {
    Stack(Imm),
    Concrete(u32),
}

impl<'a, W: Clone, E: Error> Machine<'a, W, E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        t: &'a mut (dyn Runtime<W, Error = E> + 'a),
        mem: RawMemory<'a>,
        rstack: &'a mut [u32],
        storage_bits: usize,
        pc: u32,
        regs: &'a mut [[W; 32]; 32],
        reg_consts: &'a mut [Option<u32>; 32],
        zero: W,
        one: W,
        sp: u32,
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
            offs: [const { None }; 32],
            #[cfg(feature = "early-exit-loops")]
            loop_sites: [const { None }; 8],
        }
    }

    pub(crate) fn run(mut self) -> Result<(), ErtError<E>> {
        loop {
            self.pc &= !3;
            let instruction = self.decode()?;
            self.reset_fixed_registers();
            match handlers::execute(&mut self, instruction)? {
                handlers::Flow::Next(pc) => self.pc = pc,
                handlers::Flow::Exit => return Ok(()),
            }
        }
    }

    fn decode(&self) -> Result<Inst, ErtError<E>> {
        rv_asm::Inst::decode(
            u32::from_le_bytes(self.mem.read(self.pc).ok_or(ErtError::Unexpected)?),
            Xlen::Rv32,
        )
        .map(|(instruction, _)| instruction)
        .map_err(ErtError::Decode)
    }

    fn reset_fixed_registers(&mut self) {
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

    pub(crate) fn word_from_constant(&self, value: u32) -> [W; 32] {
        array::from_fn(|i| {
            if (value >> i) & 1 == 0 {
                self.zero.clone()
            } else {
                self.one.clone()
            }
        })
    }

    pub(crate) fn write_constant(&mut self, dest: Reg, value: u32) {
        self.offs[dest.0 as usize] = None;
        self.reg_consts[dest.0 as usize] = Some(value);
        self.regs[dest.0 as usize] = self.word_from_constant(value);
    }

    pub(crate) fn stack_offset(&self, base: Reg, offset: Imm) -> Result<Imm, ErtError<E>> {
        let relative = match base {
            x if x == Reg::SP => offset.as_i32(),
            x if self.offs[x.0 as usize].is_some() => self.offs[base.0 as usize]
                .unwrap()
                .wrapping_add(offset.as_i32()),
            _ => return Err(ErtError::Unexpected),
        };
        Ok(Imm::new_i32(self.sp.wrapping_add_signed(relative) as i32))
    }

    pub(crate) fn stack_bits(
        &self,
        address: Imm,
        width: usize,
    ) -> Result<Range<usize>, ErtError<E>> {
        let start = (address.as_u32() as usize)
            .checked_mul(8)
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

pub(crate) fn write_abi_args<W, E: Error, const N: usize>(
    t: &mut (dyn Runtime<W, Error = E> + '_),
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    storage_bits: usize,
    sp: u32,
    args: [([W; 32], Option<u32>); N],
) -> Result<(), E> {
    for (i, (value, constant)) in args.into_iter().enumerate() {
        match ABI_REGS.get(i).copied() {
            Some(register) => {
                regs[register.0 as usize] = value;
                reg_consts[register.0 as usize] = constant;
            }
            None => {
                let stack_start = sp as usize * 8 + 32 * (i - ABI_REGS.len());
                debug_assert!(stack_start + 32 <= storage_bits);
                for (bit, value) in value.into_iter().enumerate() {
                    t.storage_write_bit(stack_start + bit, value)?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn read_abi_results<W: Clone, E: Error, const M: usize>(
    t: &mut (dyn Runtime<W, Error = E> + '_),
    regs: &[[W; 32]; 32],
    reg_consts: &[Option<u32>; 32],
    storage_bits: usize,
    sp: u32,
) -> Result<[([W; 32], Option<u32>); M], E> {
    let mut results: [MaybeUninit<([W; 32], Option<u32>)>; M] =
        [const { MaybeUninit::uninit() }; M];
    for (i, result) in results.iter_mut().enumerate() {
        result.write(match ABI_REGS.get(i).copied() {
            Some(register) => (
                regs[register.0 as usize].clone(),
                reg_consts[register.0 as usize],
            ),
            None => {
                let stack_start = sp as usize * 8 + 32 * (i - ABI_REGS.len());
                debug_assert!(stack_start + 32 <= storage_bits);
                let mut word: [MaybeUninit<W>; 32] = [const { MaybeUninit::uninit() }; 32];
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
