#![no_std]

use core::convert::Infallible;

use cirrus_core::ContextWithValue;
use portal_pc_asm_common::types::reg::Reg;
pub trait AsmValue{

}
pub struct Asm<'a,Context,T: ?Sized>{
    pub arch: &'a mut T,
    pub context: &'a mut Context,
    pub used_regs: [bool; 32],
}
impl<'a,'b, Context,T: AsmValue> ContextWithValue<T> for Asm<'a,Context,dyn portal_solutions_asm_aarch64::out::WriterCore<Context, Error = Infallible> + 'b>{
    type Wrapped = Reg;
}