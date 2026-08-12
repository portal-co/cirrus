#![no_std]
pub use paste::paste;
pub trait CirrusContextWithValue<Val> {
    type Wrapped;
}
#[macro_export]
macro_rules! context_with_binop {
    ($name:ident, $method:ident) => {
        $crate::paste!{
            pub trait $name<Val>: $crate::CirrusContextWithValue<Val>{
                fn $method(&mut self, a: <Self as $crate::CirrusContextWithValue<Val>>::Wrapped, b: <Self as $crate::CirrusContextWithValue<Val>>::Wrapped) -> <Self as $crate::CirrusContextWithValue<Val>>::Wrapped;
                fn [<$method _assign>](&mut self, a: &mut <Self as $crate::CirrusContextWithValue<Val>>::Wrapped, b: <Self as $crate::CirrusContextWithValue<Val>>::Wrapped);
            }
        }
    };
}
context_with_binop!(ContextWithAdd, add);
context_with_binop!(ContextWithSub, sub);
context_with_binop!(ContextWithMul, mul);
context_with_binop!(ContextWithDiv, div);
