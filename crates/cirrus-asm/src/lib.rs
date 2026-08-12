#![no_std]

use core::{convert::Infallible, error::Error};

use cirrus_core::{ContextWithValue, HasError};
use portal_pc_asm_common::types::{mem::MemorySize, reg::Reg};
use portal_solutions_asm_aarch64::AArch64Arch;
use portal_solutions_asm_x86_64::X64Arch;
pub trait AsmValue {}
pub struct Asm<'a, Context, T: ?Sized> {
    pub arch: &'a mut T,
    pub context: &'a mut Context,
    pub used_regs: [u32; 32],
    pub offset: u32,
}
pub trait InternalContext: HasError {
    fn save_regs(&mut self, reg1: Reg, reg2: Reg) -> Result<(), Self::Error>;
}
impl<'a, 'b, Context, E: Error> HasError
    for Asm<'a, Context, dyn portal_solutions_asm_aarch64::out::WriterCore<Context, Error = E> + 'b>
{
    type Error = E;
}
impl<'a, 'b, Context, E: Error> HasError
    for Asm<'a, Context, dyn portal_solutions_asm_x86_64::out::WriterCore<Context, Error = E> + 'b>
{
    type Error = E;
}
impl<'a, 'b, Context, E: Error> InternalContext
    for Asm<'a, Context, dyn portal_solutions_asm_aarch64::out::WriterCore<Context, Error = E> + 'b>
{
    fn save_regs(&mut self, reg1: Reg, reg2: Reg) -> Result<(), Self::Error> {
        self.offset += 16;
        for (Reg(i),u) in [(reg1,self.offset - 8),(reg2,self.offset)]{
            self.used_regs[i as usize] = u;
        }
        self.arch.stp(
            self.context,
            AArch64Arch::default(),
            &reg1,
            &reg2,
            &portal_solutions_asm_aarch64::out::arg::MemArgKind::Mem {
                base: &Reg(31),
                offset: None,
                disp: -16,
                size: MemorySize::_128,
                reg_class: portal_solutions_asm_aarch64::RegisterClass::Gpr,
                mode: portal_solutions_asm_aarch64::out::arg::AddressingMode::PreIndex,
            },
        )
    }
}
impl<'a, 'b, Context, E: Error> InternalContext
    for Asm<'a, Context, dyn portal_solutions_asm_x86_64::out::WriterCore<Context, Error = E> + 'b>
{
    fn save_regs(&mut self, reg1: Reg, reg2: Reg) -> Result<(), Self::Error> {
        self.offset += 16;
        for (Reg(i),u) in [(reg1,self.offset - 8),(reg2,self.offset)]{
            self.used_regs[i as usize] = u;
            self.arch.push(self.context, X64Arch::default(), &Reg(i))?;
        }
        Ok(())
    }
}
impl<'a, Context, T: AsmValue, W: ?Sized> ContextWithValue<T> for Asm<'a, Context, W>
where
    Asm<'a, Context, W>: InternalContext,
{
    type Wrapped = Reg;
}
