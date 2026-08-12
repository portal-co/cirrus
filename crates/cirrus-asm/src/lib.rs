#![no_std]

use core::{convert::Infallible, error::Error};

use cirrus_core::{ContextWithValue, HasError};
use portal_pc_asm_common::types::reg::Reg;
pub trait AsmValue {}
pub struct Asm<'a, Context, T: ?Sized> {
    pub arch: &'a mut T,
    pub context: &'a mut Context,
    pub used_regs: [bool; 32],
}
pub trait InternalContext: HasError {}
impl<'a, 'b, Context,E: Error> HasError
    for Asm<
        'a,
        Context,
        dyn portal_solutions_asm_aarch64::out::WriterCore<Context, Error = E> + 'b,
    >
{
    type Error = E;
}
impl<'a, 'b, Context,E: Error> InternalContext
    for Asm<
        'a,
        Context,
        dyn portal_solutions_asm_aarch64::out::WriterCore<Context, Error = E> + 'b,
    >
{
}
impl<'a, Context, T: AsmValue, W: ?Sized> ContextWithValue<T> for Asm<'a, Context, W>
where
    Asm<'a, Context, W>: InternalContext,
{
    type Wrapped = Reg;
}
