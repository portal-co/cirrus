#![no_std]

use core::ops::{Add, Mul};

use cirrus_core::{ContextWithAdd, ContextWithSub, ContextWithValue};
use digest::{Digest, array::Array};
pub trait HashVal<Val> {
    fn hash<D: Digest>(&mut self, digest: &mut D, val: Val) -> Array<u8, D::OutputSize>;
}
pub trait HashWrapped<Val>: ContextWithValue<Val> + HashVal<Self::Wrapped> {}
impl<Val, T: ContextWithValue<Val> + HashVal<Self::Wrapped>> HashWrapped<Val> for T {}
pub trait CreateBeaverTriple<Val> {
    fn beaver(&mut self) -> [Val; 3];
}
pub trait Open<Val> {
    type Opening;
    fn open(&mut self, val: Val) -> Self::Opening;
    fn close(&mut self, opening: Self::Opening) -> Val;
}
pub trait BeaverOpening<B: BeaverMul<Val> + ?Sized, Val>:
    Sized
    + Add<Self, Output = Self>
    + Mul<B::Wrapped, Output = B::Wrapped>
    + Mul<Self, Output = Self>
    + Clone
{
}
impl<
    B: BeaverMul<Val> + ?Sized,
    Val,
    T: Add<Self, Output = Self>
        + Mul<B::Wrapped, Output = B::Wrapped>
        + Mul<Self, Output = Self>
        + Clone,
> BeaverOpening<B, Val> for T
{
}
pub trait BeaverMul<Val>:
    ContextWithValue<Val, Wrapped: Clone> + ContextWithSub<Val> + ContextWithAdd<Val>
{
    fn beaver_mul<O: Open<Self::Wrapped, Opening: BeaverOpening<Self, Val>>>(
        &mut self,
        opening: &mut O,
        am: Self::Wrapped,
        bm: Self::Wrapped,
        beaver: [Self::Wrapped; 3],
    ) -> Result<Self::Wrapped,Self::Error> {
        let [a, b, c] = beaver;
        let d = opening.open(self.sub(am, a.clone())?);
        let e = opening.open(self.sub(bm, b.clone())?);
        let mut v = opening.close(d.clone() * e.clone());
        self.add_assign(&mut v, d * b)?;
        self.add_assign(&mut v, e * a)?;
        self.add_assign(&mut v, c)?;
        return Ok(v);
    }
}
impl<
    Val,
    T: ?Sized + ContextWithValue<Val, Wrapped: Clone> + ContextWithSub<Val> + ContextWithAdd<Val>,
> BeaverMul<Val> for T
{
}
