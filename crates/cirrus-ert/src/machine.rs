use core::{array, error::Error, mem::MaybeUninit, ops::Range};

use rv_asm::{Imm, Inst, Reg, Xlen};

use crate::{ContextWithRvOps, ErtError, RawMemory, handlers};

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
    pub(crate) t: &'a mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + 'a),
    pub(crate) hash: &'a mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + 'a),
    pub(crate) mem: RawMemory,
    pub(crate) rstack: &'a mut [u32],
    pub(crate) vstack: &'a mut [W],
    pub(crate) pc: u32,
    pub(crate) regs: &'a mut [[W; 32]; 32],
    pub(crate) reg_consts: &'a mut [Option<u32>; 32],
    pub(crate) zero: W,
    pub(crate) one: W,
    pub(crate) sp: u32,
    pub(crate) stack_top: u32,
    pub(crate) rsp: u32,
    pub(crate) offs: [Option<i32>; 32],
}

pub(crate) enum LoadAddress {
    Stack(Imm),
    Concrete(u32),
}

pub(crate) fn add_bits<W: Clone, E: Error, const N: usize>(
    t: &mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + '_),
    v: &[W; N],
    w: &[W; N],
    mut carry: W,
) -> Result<[W; N], E> {
    let mut output: [MaybeUninit<W>; N] = [const { MaybeUninit::uninit() }; N];
    for i in 0..N {
        let sum_without_carry = t.bitxor(v[i].clone(), w[i].clone())?;
        output[i] = MaybeUninit::new(t.bitxor(sum_without_carry, carry.clone())?);
        let inputs = [v[i].clone(), w[i].clone(), carry.clone()];
        let mut pairs: [MaybeUninit<W>; 3] = [const { MaybeUninit::uninit() }; 3];
        for i in 0..3 {
            pairs[i] = MaybeUninit::new(
                t.bitand(inputs[(i + 2) % 3].clone(), inputs[(i + 1) % 3].clone())?,
            );
        }
        let [a, b, c] = pairs.map(|value| unsafe { value.assume_init() });
        let carry_without_a = t.bitor(c, b)?;
        carry = t.bitor(a, carry_without_a)?;
    }
    Ok(output.map(|value| unsafe { value.assume_init() }))
}

impl<'a, W: Clone, E: Error> Machine<'a, W, E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        t: &'a mut (dyn ContextWithRvOps<bool, Wrapped = W, Error = E> + 'a),
        hash: &'a mut (dyn FnMut(&[[W; 32]]) -> Result<[u8; 32], E> + 'a),
        mem: RawMemory,
        rstack: &'a mut [u32],
        vstack: &'a mut [W],
        pc: u32,
        regs: &'a mut [[W; 32]; 32],
        reg_consts: &'a mut [Option<u32>; 32],
        zero: W,
        one: W,
        sp: u32,
    ) -> Self {
        Self {
            t,
            hash,
            mem,
            rstack,
            vstack,
            pc,
            regs,
            reg_consts,
            zero,
            one,
            sp,
            stack_top: sp,
            rsp: 0,
            offs: [const { None }; 32],
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
        if end > self.vstack.len() {
            return Err(ErtError::Unexpected);
        }
        Ok(start..end)
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

pub(crate) fn write_abi_args<W, const N: usize>(
    regs: &mut [[W; 32]; 32],
    reg_consts: &mut [Option<u32>; 32],
    vstack: &mut [W],
    sp: u32,
    args: [([W; 32], Option<u32>); N],
) {
    for (i, (value, constant)) in args.into_iter().enumerate() {
        match ABI_REGS.get(i).copied() {
            Some(register) => {
                regs[register.0 as usize] = value;
                reg_consts[register.0 as usize] = constant;
            }
            None => {
                let stack_start = sp as usize * 8 + 32 * (i - ABI_REGS.len());
                let stack_slot = &mut vstack[stack_start..stack_start + 32];
                for (bit, value) in value.into_iter().enumerate() {
                    stack_slot[bit] = value;
                }
            }
        }
    }
}

pub(crate) fn read_abi_results<W: Clone, const M: usize>(
    regs: &[[W; 32]; 32],
    reg_consts: &[Option<u32>; 32],
    vstack: &[W],
    sp: u32,
) -> [([W; 32], Option<u32>); M] {
    array::from_fn(|i| match ABI_REGS.get(i).copied() {
        Some(register) => (
            regs[register.0 as usize].clone(),
            reg_consts[register.0 as usize],
        ),
        None => {
            let stack_start = sp as usize * 8 + 32 * (i - ABI_REGS.len());
            let stack_slot = &vstack[stack_start..stack_start + 32];
            (array::from_fn(|bit| stack_slot[bit].clone()), None)
        }
    })
}
