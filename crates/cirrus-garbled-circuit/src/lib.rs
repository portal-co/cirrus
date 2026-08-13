#![no_std]

use core::{array, convert::Infallible};

use cirrus_core::{Bit, ContextWithAdd, ContextWithMul, ContextWithSub, ContextWithValue, HasError, Pusher};
use digest::{Digest, array::Array};

pub struct GC<'a,'b,D: Digest,const N: usize>{
    pub queue: &'a mut (dyn Pusher<[[u8;N]; 4]> + 'b),
    pub seed: Array<u8,D::OutputSize>,
    pub delta: Array<u8,D::OutputSize>,
}
impl<D: Digest, const N: usize> HasError for GC<'_, '_, D,N> {
    type Error = Infallible;
}
impl<D: Digest, const N: usize> ContextWithValue<Bit> for GC<'_, '_, D,N> {
    type Wrapped = [u8; N];
}
impl<D: Digest, const N: usize> ContextWithAdd<Bit> for GC<'_, '_, D,N> {
    fn add(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        Ok(array::from_fn(|i| a[i] ^ b[i]))
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
impl<D: Digest, const N: usize> ContextWithSub<Bit> for GC<'_, '_, D,N> {
    fn sub(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
    ) -> Result<
        <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,
        <Self as cirrus_core::HasError>::Error,
    > {
        Ok(array::from_fn(|i| a[i] ^ b[i]))
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
impl<D: Digest, const N: usize> ContextWithMul<Bit> for GC<'_, '_, D,N>{
    fn mul(&mut self,a: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,b: <Self as cirrus_core::ContextWithValue<Bit>>::Wrapped) -> Result<<Self as cirrus_core::ContextWithValue<Bit>>::Wrapped,<Self as cirrus_core::HasError>::Error> {
        self.seed = D::digest(&self.seed);
        let new = self.seed.clone();
        let new: [u8; N] = array::from_fn(|i|new[i]);
        self.queue.push(array::from_fn(|i|{
            let a = ((i & 1) == 1) ^ (a[0] & 0x01 == 1) ^ (self.delta[0] & 0x01 == 1);
            let b = ((i & 2) == 2) ^ (b[0] & 0x01 == 1) ^ (self.delta[0] & 0x01 == 1);
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