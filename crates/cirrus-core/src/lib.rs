#![no_std]
use core::{convert::Infallible, error::Error};

pub use paste::paste;
pub trait HasError{
    type Error: Error;
}
impl HasError for (){
    type Error = Infallible;
}
pub trait ContextWithValue<Val>: HasError {
    type Wrapped;
}
impl<Val> ContextWithValue<Val> for () {
    type Wrapped = Val;
}
#[macro_export]
macro_rules! context_with_binop {
    ($name:ident, $method:ident, $orig:ident) => {
        $crate::paste!{
            pub trait $name<Val>: $crate::ContextWithValue<Val>{
                fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error>;
                fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error>;
            }
            const _: ()={
                impl<Val: ::core::ops::$orig<Val, Output = Val> + ::core::ops::[<$orig Assign>]<Val>> $name<Val> for (){
                    fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error>{
                        Ok(::core::ops::$orig::$method(a,b))
                    }
                    fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error>{
                        ::core::ops::[<$orig Assign>]::[<$method _assign>](a,b);
                        Ok(())
                    }
                }
            };
        }
    };
}
context_with_binop!(ContextWithAdd, add, Add);
context_with_binop!(ContextWithSub, sub, Sub);
context_with_binop!(ContextWithMul, mul, Mul);
context_with_binop!(ContextWithDiv, div, Div);
context_with_binop!(ContextWithBitAnd, bitand, BitAnd);
context_with_binop!(ContextWithBitOr, bitor, BitOr);
context_with_binop!(ContextWithBitXor, bitxor, BitXor);
pub trait ContextWithMux<Val>: ContextWithValue<bool> + ContextWithValue<Val> {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped,Self::Error>;
}
impl<Val> ContextWithMux<Val> for () {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped,Self::Error> {
        Ok(if cond { then } else { r#else })
    }
}
