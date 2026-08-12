#![no_std]

use cirrus_core::{ContextWithAdd, ContextWithSub, ContextWithValue};
pub struct Vole<Wrapped> {
    pub wrapped: Wrapped,
}
impl<Wrapped: ContextWithValue<Val>, Val> ContextWithValue<Val> for Vole<Wrapped> {
    type Wrapped = (Wrapped::Wrapped, Wrapped::Wrapped);
}
impl<Wrapped: ContextWithAdd<Val>, Val> ContextWithAdd<Val> for Vole<Wrapped> {
    fn add(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
    ) -> <Self as cirrus_core::ContextWithValue<Val>>::Wrapped {
        let (a0, a1) = a;
        let (b0, b1) = b;
        (self.wrapped.add(a0, b0), self.wrapped.add(a1, b1))
    }

    fn add_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
    ) {
        let (a0,a1)=a;
        let (b0,b1) = b;
        self.wrapped.add_assign(a0, b0);
        self.wrapped.add_assign(a1, b1);
    }
}
impl<Wrapped: ContextWithSub<Val>, Val> ContextWithSub<Val> for Vole<Wrapped> {
    fn sub(
        &mut self,
        a: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
    ) -> <Self as cirrus_core::ContextWithValue<Val>>::Wrapped {
        let (a0, a1) = a;
        let (b0, b1) = b;
        (self.wrapped.sub(a0, b0), self.wrapped.sub(a1, b1))
    }

    fn sub_assign(
        &mut self,
        a: &mut <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
        b: <Self as cirrus_core::ContextWithValue<Val>>::Wrapped,
    ) {
        let (a0,a1)=a;
        let (b0,b1) = b;
        self.wrapped.sub_assign(a0, b0);
        self.wrapped.sub_assign(a1, b1);
    }
}
