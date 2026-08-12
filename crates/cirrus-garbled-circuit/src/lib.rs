#![no_std]

use core::{array, convert::Infallible};

use cirrus_core::{Bit, ContextWithAdd, ContextWithMul, ContextWithSub, ContextWithValue, HasError};
use digest::{Digest, array::Array};
pub trait Pusher<T>{
    fn push(&mut self, x: T);
}
pub struct GC<'a,'b,D: Digest>{
    pub queue: &'a mut (dyn Pusher<[Array<u8, D::OutputSize>; 4]> + 'b),
    pub seed: Array<u8,D::OutputSize>,
    pub delta: Array<u8,D::OutputSize>,
}
impl<D: Digest> HasError for GC<'_, '_, D> {
    type Error = Infallible;
}
impl<D: Digest> ContextWithValue<Bit> for GC<'_, '_, D> {
    type Wrapped = Array<u8, D::OutputSize>;
}
impl<D: Digest> ContextWithAdd<Bit> for GC<'_, '_, D> {
    fn add(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        Ok(Array::from_fn(|i| a[i] ^ b[i]))
    }

    fn add_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<(), <Self as cirrus_core::HasError>::Error> {
        for (a, b) in a.iter_mut().zip(b) {
            *a ^= b
        }
        Ok(())
    }
}
impl<D: Digest> ContextWithSub<Bit> for GC<'_, '_, D> {
    fn sub(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        Ok(Array::from_fn(|i| a[i] ^ b[i]))
    }

    fn sub_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<(), <Self as cirrus_core::HasError>::Error> {
        for (a, b) in a.iter_mut().zip(b) {
            *a ^= b
        }
        Ok(())
    }
}
impl<D: Digest> ContextWithMul<Bit> for GC<'_, '_, D>{
    fn mul(&mut self,a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped) -> Result<<Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,<Self as cirrus_core::HasError>::Error> {
        self.seed = D::digest(&self.seed);
        let mut new = self.seed.clone();
        new[0] |= 0x01;
        self.queue.push(array::from_fn(|i|{
            let a = ((i & 1) == 1) ^ (self.delta[0] & 0x01 == 1);
            let b = ((i & 2) == 2) ^ (self.delta[0] & 0x01 == 1);
            let r = a & b;
            let mut x = new.clone();
            if r{
                for (a,b) in x.iter_mut().zip(self.delta.clone()){
                    *a ^= b
                }
            }
            return x;
        }));
        return Ok(new);
    }

    fn mul_assign(&mut self,a: &mut <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped) -> Result<(),<Self as cirrus_core::HasError>::Error> {
        *a = self.mul(a.clone(), b)?;
        Ok(())
    }
}