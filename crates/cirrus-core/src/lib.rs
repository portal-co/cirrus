#![no_std]
use core::{
    convert::Infallible,
    error::Error,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

pub use paste::paste;
pub trait Pusher<T> {
    fn push(&mut self, x: T);
}
pub struct Bit(pub bool);
impl Add for Bit {
    type Output = Bit;

    fn add(self, rhs: Self) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}
impl AddAssign for Bit {
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl Sub for Bit {
    type Output = Bit;

    fn sub(self, rhs: Self) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}
impl SubAssign for Bit {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl Mul for Bit {
    type Output = Bit;

    fn mul(self, rhs: Self) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}
impl MulAssign for Bit {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
pub trait HasError {
    type Error: Error;
}
impl HasError for () {
    type Error = Infallible;
}
pub trait ContextWithValue<Val>: HasError {
    type Wrapped;
}

/// A Boolean wire used as one little-endian bit of a symbolic storage address.
///
/// `known` is deliberately only a fact about the wire, rather than a second
/// representation of it.  Contexts which can exploit public address bits (for
/// example a dense MUX-tree implementation) may use it to avoid unnecessary
/// work, while authenticated storage backends can consume `wire` directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageAddressBit<W> {
    pub wire: W,
    pub known: Option<bool>,
}

/// A context which can access caller-owned symbolic storage.
///
/// Storage is intentionally an associated type rather than state owned by the
/// context.  This lets one execution context work with several independent
/// storage namespaces, and lets capable contexts replace a MUX-tree lowering
/// with a native authenticated implementation.  Addresses are provided least
/// significant bit first, matching Volar IR's Boolean storage lanes.
pub trait ContextWithStorage<Val>: ContextWithValue<bool> + ContextWithValue<Val> {
    type Storage: ?Sized;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error>;

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error>;
}
pub trait ContextWithCreate<Val>: ContextWithValue<Val> {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error>;
}
impl<Val> ContextWithValue<Val> for () {
    type Wrapped = Val;
}
impl<Val> ContextWithCreate<Val> for () {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        Ok(val)
    }
}

/// Plaintext dense storage for the native Boolean test host.
///
/// Symbolic addresses are concrete `bool`s in this host, so they resolve to
/// an ordinary little-endian slice index. Bounds are a caller/layout error
/// and therefore panic just like ordinary slice indexing.
impl ContextWithStorage<bool> for () {
    type Storage = [bool];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Self::Error> {
        Ok(storage[concrete_storage_index(address)])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Self::Error> {
        storage[concrete_storage_index(address)] = value;
        Ok(())
    }
}

fn concrete_storage_index(address: &[StorageAddressBit<bool>]) -> usize {
    address
        .iter()
        .enumerate()
        .fold(0usize, |index, (bit, address_bit)| {
            if address_bit.wire {
                index
                    | (1usize
                        .checked_shl(bit as u32)
                        .expect("plaintext storage address exceeds usize width"))
            } else {
                index
            }
        })
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
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error>;
}
impl<Val> ContextWithMux<Val> for () {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        Ok(if cond { then } else { r#else })
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;

    struct TypedStorageContext;

    impl HasError for TypedStorageContext {
        type Error = Infallible;
    }

    impl ContextWithValue<bool> for TypedStorageContext {
        type Wrapped = bool;
    }

    impl ContextWithValue<u16> for TypedStorageContext {
        type Wrapped = u16;
    }

    impl ContextWithStorage<u16> for TypedStorageContext {
        type Storage = [u16];

        fn storage_read(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
        ) -> Result<u16, Self::Error> {
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            Ok(storage[index])
        }

        fn storage_write(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
            value: u16,
        ) -> Result<(), Self::Error> {
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            storage[index] = value;
            Ok(())
        }
    }

    #[test]
    fn storage_is_generic_and_external_to_its_context() {
        let mut context = TypedStorageContext;
        let mut storage = [0u16; 4];
        let address = [
            StorageAddressBit {
                wire: true,
                known: Some(true),
            },
            StorageAddressBit {
                wire: false,
                known: Some(false),
            },
        ];
        context
            .storage_write(&mut storage, &address, 0xbeef)
            .unwrap();
        assert_eq!(context.storage_read(&mut storage, &address), Ok(0xbeef));
        assert_eq!(storage, [0, 0xbeef, 0, 0]);
    }
}
