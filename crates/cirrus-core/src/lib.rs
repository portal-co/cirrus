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

/// Shared-reference variant of [`Pusher`].
///
/// Types that can accept values without exclusive access implement this
/// subtrait; [`Pusher`] is then implemented for `&Self`.
pub trait PusherByRef<T>: Pusher<T> {
    fn push_by_ref(&self, x: T);
}

impl<T, P: Pusher<T> + ?Sized> Pusher<T> for &mut P {
    fn push(&mut self, x: T) {
        (**self).push(x);
    }
}

impl<T, P: PusherByRef<T> + ?Sized> PusherByRef<T> for &mut P {
    fn push_by_ref(&self, x: T) {
        (**self).push_by_ref(x);
    }
}

impl<T, P: PusherByRef<T> + ?Sized> Pusher<T> for &P {
    fn push(&mut self, x: T) {
        (**self).push_by_ref(x);
    }
}

impl<T, P: PusherByRef<T> + ?Sized> PusherByRef<T> for &P {
    fn push_by_ref(&self, x: T) {
        (**self).push_by_ref(x);
    }
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
impl<C: HasError + ?Sized> HasError for &C {
    type Error = C::Error;
}
impl<C: HasError + ?Sized> HasError for &mut C {
    type Error = C::Error;
}
pub trait ContextWithValue<Val>: HasError {
    type Wrapped;
}

impl<Val, C: ContextWithValue<Val> + ?Sized> ContextWithValue<Val> for &C {
    type Wrapped = C::Wrapped;
}
impl<Val, C: ContextWithValue<Val> + ?Sized> ContextWithValue<Val> for &mut C {
    type Wrapped = C::Wrapped;
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

/// Shared-reference variant of [`ContextWithStorage`].
///
/// Storage methods take `&Self::Storage` so an interior-mutable (locked)
/// store can be shared across compatible circuits. [`SharedContext`] remaps
/// the associated storage type to `&'a C::Storage` so existing `&mut Storage`
/// interpreter signatures can hold a unique binding to that shared reference.
pub trait ContextWithStorageByRef<Val>: ContextWithStorage<Val> {
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error>;

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error>;
}

impl<Val, C: ContextWithStorage<Val> + ?Sized> ContextWithStorage<Val> for &mut C {
    type Storage = C::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).storage_read(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        (**self).storage_write(storage, address, value)
    }
}

impl<Val, C: ContextWithStorageByRef<Val> + ?Sized> ContextWithStorageByRef<Val> for &mut C {
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).storage_read_by_ref(storage, address)
    }

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        (**self).storage_write_by_ref(storage, address, value)
    }
}

impl<Val, C: ContextWithStorageByRef<Val> + ?Sized> ContextWithStorage<Val> for &C {
    type Storage = C::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).storage_read_by_ref(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        (**self).storage_write_by_ref(storage, address, value)
    }
}

impl<Val, C: ContextWithStorageByRef<Val> + ?Sized> ContextWithStorageByRef<Val> for &C {
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).storage_read_by_ref(storage, address)
    }

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        (**self).storage_write_by_ref(storage, address, value)
    }
}

pub trait ContextWithCreate<Val>: ContextWithValue<Val> {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error>;
}

/// Shared-reference variant of [`ContextWithCreate`].
pub trait ContextWithCreateByRef<Val>: ContextWithCreate<Val> {
    fn create_by_ref(&self, val: Val) -> Result<Self::Wrapped, Self::Error>;
}

impl<Val, C: ContextWithCreate<Val> + ?Sized> ContextWithCreate<Val> for &mut C {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        (**self).create(val)
    }
}

impl<Val, C: ContextWithCreateByRef<Val> + ?Sized> ContextWithCreateByRef<Val> for &mut C {
    fn create_by_ref(&self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        (**self).create_by_ref(val)
    }
}

impl<Val, C: ContextWithCreateByRef<Val> + ?Sized> ContextWithCreate<Val> for &C {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        (**self).create_by_ref(val)
    }
}

impl<Val, C: ContextWithCreateByRef<Val> + ?Sized> ContextWithCreateByRef<Val> for &C {
    fn create_by_ref(&self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        (**self).create_by_ref(val)
    }
}

impl<Val> ContextWithValue<Val> for () {
    type Wrapped = Val;
}
impl<Val> ContextWithCreate<Val> for () {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        Ok(val)
    }
}
impl<Val> ContextWithCreateByRef<Val> for () {
    fn create_by_ref(&self, val: Val) -> Result<Self::Wrapped, Self::Error> {
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

/// Interpreter adapter that remaps [`ContextWithStorage::Storage`] to a
/// shared reference.
///
/// Gate operations forward through the inner context's `*ByRef` traits.
/// Storage methods take `&mut &'a C::Storage`, so each caller can own a
/// unique binding to the same locked store:
///
/// ```ignore
/// let mut slot: &LockedStorage = &shared;
/// execute(&mut SharedContext(&ctx), ..., &mut [StorageBank { value: &mut slot, .. }]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SharedContext<'a, C: ?Sized>(pub &'a C);

impl<C: HasError + ?Sized> HasError for SharedContext<'_, C> {
    type Error = C::Error;
}

impl<Val, C: ContextWithValue<Val> + ?Sized> ContextWithValue<Val> for SharedContext<'_, C> {
    type Wrapped = C::Wrapped;
}

impl<Val, C: ContextWithCreateByRef<Val> + ?Sized> ContextWithCreate<Val> for SharedContext<'_, C> {
    fn create(&mut self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        self.0.create_by_ref(val)
    }
}

impl<Val, C: ContextWithCreateByRef<Val> + ?Sized> ContextWithCreateByRef<Val>
    for SharedContext<'_, C>
{
    fn create_by_ref(&self, val: Val) -> Result<Self::Wrapped, Self::Error> {
        self.0.create_by_ref(val)
    }
}

impl<'a, Val, C> ContextWithStorage<Val> for SharedContext<'a, C>
where
    C: ContextWithStorageByRef<Val> + ?Sized,
    C::Storage: 'a,
{
    type Storage = &'a C::Storage;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        self.0.storage_read_by_ref(storage, address)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        self.0.storage_write_by_ref(storage, address, value)
    }
}

impl<'a, Val, C> ContextWithStorageByRef<Val> for SharedContext<'a, C>
where
    C: ContextWithStorageByRef<Val> + ?Sized,
    C::Storage: 'a,
{
    fn storage_read_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        self.0.storage_read_by_ref(storage, address)
    }

    fn storage_write_by_ref(
        &self,
        storage: &Self::Storage,
        address: &[StorageAddressBit<<Self as ContextWithValue<bool>>::Wrapped>],
        value: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<(), Self::Error> {
        self.0.storage_write_by_ref(storage, address, value)
    }
}

#[macro_export]
macro_rules! context_with_binop {
    ($name:ident, $method:ident, $orig:ident) => {
        $crate::paste!{
            pub trait $name<Val>: $crate::ContextWithValue<Val>{
                fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error>;
                fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error>;
            }
            pub trait [<$name ByRef>]<Val>: $name<Val> {
                fn [<$method _by_ref>](&self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error>;
                fn [<$method _assign_by_ref>](&self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error>;
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
                impl<Val: ::core::ops::$orig<Val, Output = Val> + ::core::ops::[<$orig Assign>]<Val>> [<$name ByRef>]<Val> for (){
                    fn [<$method _by_ref>](&self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error>{
                        Ok(::core::ops::$orig::$method(a,b))
                    }
                    fn [<$method _assign_by_ref>](&self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error>{
                        ::core::ops::[<$orig Assign>]::[<$method _assign>](a,b);
                        Ok(())
                    }
                }
                impl<Val, C: $name<Val> + ?Sized> $name<Val> for &mut C {
                    fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        (**self).$method(a, b)
                    }
                    fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        (**self).[<$method _assign>](a, b)
                    }
                }
                impl<Val, C: [<$name ByRef>]<Val> + ?Sized> [<$name ByRef>]<Val> for &mut C {
                    fn [<$method _by_ref>](&self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        (**self).[<$method _by_ref>](a, b)
                    }
                    fn [<$method _assign_by_ref>](&self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        (**self).[<$method _assign_by_ref>](a, b)
                    }
                }
                impl<Val, C: [<$name ByRef>]<Val> + ?Sized> $name<Val> for &C {
                    fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        (**self).[<$method _by_ref>](a, b)
                    }
                    fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        (**self).[<$method _assign_by_ref>](a, b)
                    }
                }
                impl<Val, C: [<$name ByRef>]<Val> + ?Sized> [<$name ByRef>]<Val> for &C {
                    fn [<$method _by_ref>](&self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        (**self).[<$method _by_ref>](a, b)
                    }
                    fn [<$method _assign_by_ref>](&self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        (**self).[<$method _assign_by_ref>](a, b)
                    }
                }
                impl<Val, C: [<$name ByRef>]<Val> + ?Sized> $name<Val> for $crate::SharedContext<'_, C> {
                    fn $method(&mut self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        self.0.[<$method _by_ref>](a, b)
                    }
                    fn [<$method _assign>](&mut self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        self.0.[<$method _assign_by_ref>](a, b)
                    }
                }
                impl<Val, C: [<$name ByRef>]<Val> + ?Sized> [<$name ByRef>]<Val> for $crate::SharedContext<'_, C> {
                    fn [<$method _by_ref>](&self, a: <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<<Self as $crate::ContextWithValue<Val>>::Wrapped,<Self as $crate::HasError>::Error> {
                        self.0.[<$method _by_ref>](a, b)
                    }
                    fn [<$method _assign_by_ref>](&self, a: &mut <Self as $crate::ContextWithValue<Val>>::Wrapped, b: <Self as $crate::ContextWithValue<Val>>::Wrapped) -> Result<(),<Self as $crate::HasError>::Error> {
                        self.0.[<$method _assign_by_ref>](a, b)
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

/// Shared-reference variant of [`ContextWithMux`].
pub trait ContextWithMuxByRef<Val>: ContextWithMux<Val> {
    fn mux_by_ref(
        &self,
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

impl<Val> ContextWithMuxByRef<Val> for () {
    fn mux_by_ref(
        &self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        Ok(if cond { then } else { r#else })
    }
}

impl<Val, C: ContextWithMux<Val> + ?Sized> ContextWithMux<Val> for &mut C {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).mux(cond, then, r#else)
    }
}

impl<Val, C: ContextWithMuxByRef<Val> + ?Sized> ContextWithMuxByRef<Val> for &mut C {
    fn mux_by_ref(
        &self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).mux_by_ref(cond, then, r#else)
    }
}

impl<Val, C: ContextWithMuxByRef<Val> + ?Sized> ContextWithMux<Val> for &C {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).mux_by_ref(cond, then, r#else)
    }
}

impl<Val, C: ContextWithMuxByRef<Val> + ?Sized> ContextWithMuxByRef<Val> for &C {
    fn mux_by_ref(
        &self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        (**self).mux_by_ref(cond, then, r#else)
    }
}

impl<Val, C: ContextWithMuxByRef<Val> + ?Sized> ContextWithMux<Val> for SharedContext<'_, C> {
    fn mux(
        &mut self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        self.0.mux_by_ref(cond, then, r#else)
    }
}

impl<Val, C: ContextWithMuxByRef<Val> + ?Sized> ContextWithMuxByRef<Val> for SharedContext<'_, C> {
    fn mux_by_ref(
        &self,
        cond: <Self as ContextWithValue<bool>>::Wrapped,
        then: <Self as ContextWithValue<Val>>::Wrapped,
        r#else: <Self as ContextWithValue<Val>>::Wrapped,
    ) -> Result<<Self as ContextWithValue<Val>>::Wrapped, Self::Error> {
        self.0.mux_by_ref(cond, then, r#else)
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

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

    struct CellStorageContext;

    impl HasError for CellStorageContext {
        type Error = Infallible;
    }

    impl ContextWithValue<bool> for CellStorageContext {
        type Wrapped = bool;
    }

    impl ContextWithValue<u16> for CellStorageContext {
        type Wrapped = u16;
    }

    impl ContextWithCreate<bool> for CellStorageContext {
        fn create(&mut self, val: bool) -> Result<bool, Self::Error> {
            Ok(val)
        }
    }

    impl ContextWithCreateByRef<bool> for CellStorageContext {
        fn create_by_ref(&self, val: bool) -> Result<bool, Self::Error> {
            Ok(val)
        }
    }

    impl ContextWithStorage<u16> for CellStorageContext {
        type Storage = [Cell<u16>];

        fn storage_read(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
        ) -> Result<u16, Self::Error> {
            self.storage_read_by_ref(storage, address)
        }

        fn storage_write(
            &mut self,
            storage: &mut Self::Storage,
            address: &[StorageAddressBit<bool>],
            value: u16,
        ) -> Result<(), Self::Error> {
            self.storage_write_by_ref(storage, address, value)
        }
    }

    impl ContextWithStorageByRef<u16> for CellStorageContext {
        fn storage_read_by_ref(
            &self,
            storage: &Self::Storage,
            address: &[StorageAddressBit<bool>],
        ) -> Result<u16, Self::Error> {
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            Ok(storage[index].get())
        }

        fn storage_write_by_ref(
            &self,
            storage: &Self::Storage,
            address: &[StorageAddressBit<bool>],
            value: u16,
        ) -> Result<(), Self::Error> {
            let index = address
                .iter()
                .enumerate()
                .fold(0usize, |index, (bit, value)| {
                    index | ((value.wire as usize) << bit)
                });
            storage[index].set(value);
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

    #[test]
    fn unit_context_shared_reference_create_xor_storage() {
        let context = ();
        let mut shared = &context;
        assert_eq!(shared.create(true), Ok(true));
        assert_eq!(shared.bitxor(true, false), Ok(true));
        // Dense `[bool]` cells need exclusive `&mut [bool]`, available through
        // the `&mut ()` blanket rather than `ContextWithStorageByRef`.
        let mut host = ();
        let mut storage = [false, false, false, false];
        let address = [StorageAddressBit {
            wire: true,
            known: Some(true),
        }];
        host.storage_write(&mut storage, &address, true).unwrap();
        assert_eq!(host.storage_read(&mut storage, &address), Ok(true));
        assert_eq!(storage, [false, true, false, false]);
    }

    #[test]
    fn shared_context_remaps_storage_to_a_reference() {
        let context = CellStorageContext;
        let storage = [Cell::new(0), Cell::new(0), Cell::new(0), Cell::new(0)];
        let mut shared = SharedContext(&context);
        let mut slot: &[Cell<u16>] = &storage;
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
        shared.storage_write(&mut slot, &address, 0xbeef).unwrap();
        assert_eq!(shared.storage_read(&mut slot, &address), Ok(0xbeef));
        assert_eq!(shared.create(true), Ok(true));
        assert_eq!(storage[1].get(), 0xbeef);
        fn assert_storage_is_ref<'a, C>(_: &SharedContext<'a, C>)
        where
            SharedContext<'a, C>: ContextWithStorage<u16, Storage = &'a C::Storage>,
            C: ContextWithStorageByRef<u16>,
        {
        }
        assert_storage_is_ref(&shared);
    }
}
