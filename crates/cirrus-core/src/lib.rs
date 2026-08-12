#![no_std]
pub use paste::paste;
pub trait ContextWithValue<Val> {
    type Wrapped;
}
#[macro_export]
macro_rules! context_with_binop {
    ($name:ident, $method:ident) => {
        $crate::paste!{
            pub trait $name<Val>: $crate::ContextWithValue<Val>{
                fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> <Self as $crate::ContextWithValue<Val>>::Wrapped;
                fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped);
            }
        }
    };
}
context_with_binop!(ContextWithAdd, add);
context_with_binop!(ContextWithSub, sub);
context_with_binop!(ContextWithMul, mul);
context_with_binop!(ContextWithDiv, div);
pub trait ContextWithMux<Val>: ContextWithValue<bool> + ContextWithValue<Val> {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> <Self as ContextWithValue<Val>>::Wrapped;
}
