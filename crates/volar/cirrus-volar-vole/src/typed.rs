//! Typed Volar values for the VOLE contexts.
//!
//! These adapters execute [`cirrus_recompile_core::TypedProgram`] operations
//! directly against the normal VOLE gate stream.  Values are represented as
//! canonical little-endian bundles of authenticated bits, but an add, multiply
//! or shift is dispatched as one typed operation by the typed-program runner;
//! it is never re-recorded through Boolean `Program`/`PreparedProgram`.

use alloc::vec::Vec;
use core::ops::{Add, Mul};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithValue,
};
use cirrus_recompile_core::{
    TypeId, TypeTable, TypedBinaryOp, TypedCompareOp, TypedContext, VolarType,
};
use hybrid_array::ArraySize;
use volar_spec::{
    field::Invert,
    vole::{VoleArray, Vope, Q},
};

use crate::{VoleProverContext, VoleVerifierContext};

/// Canonical little-endian authenticated bit bundle for one typed Volar
/// value.  Its length is exactly `TypeTable::layout(ty).bits`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedVoleValue<W> {
    /// Least-significant bit first, recursively in tuple field and vector
    /// lane order.
    pub bits: Vec<W>,
}

type Wire<C> = <C as ContextWithValue<bool>>::Wrapped;

fn layout_bits(types: &TypeTable, ty: TypeId) -> usize {
    types
        .layout(ty)
        .expect("typed program validates its type table before VOLE execution")
        .bits
}

fn scalar_segments(types: &TypeTable, ty: TypeId, segments: &mut Vec<usize>) {
    match types
        .get(ty)
        .expect("typed program validates its type references before VOLE execution")
    {
        VolarType::Vector { lanes, element } => {
            for _ in 0..*lanes {
                scalar_segments(types, *element, segments);
            }
        }
        VolarType::Tuple(fields) => {
            for field in fields {
                scalar_segments(types, *field, segments);
            }
        }
        _ => segments.push(layout_bits(types, ty)),
    }
}

fn typed_constant<C>(
    context: &mut C,
    types: &TypeTable,
    ty: TypeId,
    words: &[u64],
) -> Result<TypedVoleValue<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>,
    Wire<C>: Clone,
{
    let bits = (0..layout_bits(types, ty))
        .map(|index| {
            let word = words.get(index / 64).copied().unwrap_or(0);
            context.create(((word >> (index % 64)) & 1) != 0)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TypedVoleValue { bits })
}

fn zero_bits<C>(context: &mut C, width: usize) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>,
{
    (0..width).map(|_| context.create(false)).collect()
}

fn one_bits<C>(context: &mut C, width: usize) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>,
{
    let mut bits = zero_bits(context, width)?;
    if let Some(bit) = bits.first_mut() {
        *bit = context.create(true)?;
    }
    Ok(bits)
}

fn xor_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(a, b)| context.bitxor(a, b))
        .collect()
}

fn and_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithBitAnd<bool>,
    Wire<C>: Clone,
{
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(a, b)| context.bitand(a, b))
        .collect()
}

fn or_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithBitOr<bool>,
    Wire<C>: Clone,
{
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(a, b)| context.bitor(a, b))
        .collect()
}

fn not_bits<C>(context: &mut C, a: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool> + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let mut result = Vec::with_capacity(a.len());
    for bit in a.iter().cloned() {
        let one = context.create(true)?;
        result.push(context.bitxor(bit, one)?);
    }
    Ok(result)
}

fn add_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let mut carry = context.create(false)?;
    let mut result = Vec::with_capacity(a.len());
    for (a, b) in a.iter().cloned().zip(b.iter().cloned()) {
        let axb = context.bitxor(a.clone(), b.clone())?;
        result.push(context.bitxor(axb.clone(), carry.clone())?);
        let ab = context.bitand(a, b)?;
        let carry_axb = context.bitand(carry, axb)?;
        carry = context.bitor(ab, carry_axb)?;
    }
    Ok(result)
}

fn sub_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let inverted = not_bits(context, b)?;
    let one = one_bits(context, a.len())?;
    let negated = add_bits(context, &inverted, &one)?;
    add_bits(context, a, &negated)
}

fn mux_bits<C>(
    context: &mut C,
    condition: Wire<C>,
    then: &[Wire<C>],
    r#else: &[Wire<C>],
) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithMux<bool>,
    Wire<C>: Clone,
{
    then.iter()
        .cloned()
        .zip(r#else.iter().cloned())
        .map(|(then, r#else)| context.mux(condition.clone(), then, r#else))
        .collect()
}

fn shl_const<C>(
    context: &mut C,
    a: &[Wire<C>],
    amount: usize,
    arithmetic: bool,
) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>,
    Wire<C>: Clone,
{
    let width = a.len();
    (0..width)
        .map(|index| {
            if amount <= index {
                Ok(a[index - amount].clone())
            } else if arithmetic && !a.is_empty() {
                Ok(a[width - 1].clone())
            } else {
                context.create(false)
            }
        })
        .collect()
}

fn lshr_const<C>(
    context: &mut C,
    a: &[Wire<C>],
    amount: usize,
    arithmetic: bool,
) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>,
    Wire<C>: Clone,
{
    let width = a.len();
    (0..width)
        .map(|index| {
            if index
                .checked_add(amount)
                .is_some_and(|source| source < width)
            {
                Ok(a[index + amount].clone())
            } else if arithmetic && !a.is_empty() {
                Ok(a[width - 1].clone())
            } else {
                context.create(false)
            }
        })
        .collect()
}

fn variable_shift_bits<C>(
    context: &mut C,
    a: &[Wire<C>],
    amount: &[Wire<C>],
    left: bool,
    arithmetic: bool,
) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool> + ContextWithMux<bool>,
    Wire<C>: Clone,
{
    let mut current = a.to_vec();
    for (shift_bit, index) in amount.iter().cloned().zip(0usize..) {
        let shift = 1usize.checked_shl(index as u32).unwrap_or(usize::MAX);
        let shifted = if left {
            shl_const(context, &current, shift, false)?
        } else {
            lshr_const(context, &current, shift, arithmetic)?
        };
        current = mux_bits(context, shift_bit, &shifted, &current)?;
    }
    Ok(current)
}

fn mul_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let width = a.len();
    let mut result = zero_bits(context, width)?;
    for (shift, control) in b.iter().cloned().enumerate() {
        let shifted = shl_const(context, a, shift, false)?;
        let partial = shifted
            .into_iter()
            .map(|bit| context.bitand(control.clone(), bit))
            .collect::<Result<Vec<_>, _>>()?;
        result = add_bits(context, &result, &partial)?;
    }
    Ok(result)
}

fn eq_bit<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Wire<C>, C::Error>
where
    C: ContextWithCreate<bool> + ContextWithBitAnd<bool> + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let mut equal = context.create(true)?;
    for (a, b) in a.iter().cloned().zip(b.iter().cloned()) {
        let different = context.bitxor(a, b)?;
        let one = context.create(true)?;
        let same = context.bitxor(different, one)?;
        equal = context.bitand(equal, same)?;
    }
    Ok(equal)
}

fn ult_bit<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Wire<C>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let mut less = context.create(false)?;
    let mut equal = context.create(true)?;
    for (a, b) in a.iter().cloned().zip(b.iter().cloned()).rev() {
        let one = context.create(true)?;
        let not_a = context.bitxor(a.clone(), one)?;
        let a_less_b = context.bitand(not_a, b.clone())?;
        let equal_less = context.bitand(equal.clone(), a_less_b)?;
        less = context.bitor(less, equal_less)?;
        let different = context.bitxor(a, b)?;
        let one = context.create(true)?;
        let same = context.bitxor(different, one)?;
        equal = context.bitand(equal, same)?;
    }
    Ok(less)
}

fn udiv_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>
        + ContextWithMux<bool>,
    Wire<C>: Clone,
{
    let width = a.len();
    let mut quotient = zero_bits(context, width)?;
    let mut remainder = zero_bits(context, width)?;
    for index in (0..width).rev() {
        remainder = shl_const(context, &remainder, 1, false)?;
        if !remainder.is_empty() {
            remainder[0] = a[index].clone();
        }
        let less = ult_bit(context, &remainder, b)?;
        let one = context.create(true)?;
        let enough = context.bitxor(less, one)?;
        let subtracted = sub_bits(context, &remainder, b)?;
        remainder = mux_bits(context, enough.clone(), &subtracted, &remainder)?;
        quotient[index] = enough;
    }
    Ok(quotient)
}

fn sdiv_bits<C>(context: &mut C, a: &[Wire<C>], b: &[Wire<C>]) -> Result<Vec<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>
        + ContextWithMux<bool>,
    Wire<C>: Clone,
{
    if a.is_empty() {
        return Ok(Vec::new());
    }
    let sign_a = a[a.len() - 1].clone();
    let sign_b = b[b.len() - 1].clone();
    let zero = zero_bits(context, a.len())?;
    let neg_a = sub_bits(context, &zero, a)?;
    let neg_b = sub_bits(context, &zero, b)?;
    let abs_a = mux_bits(context, sign_a.clone(), &neg_a, a)?;
    let abs_b = mux_bits(context, sign_b.clone(), &neg_b, b)?;
    let quotient = udiv_bits(context, &abs_a, &abs_b)?;
    let negate = context.bitxor(sign_a, sign_b)?;
    let negated = sub_bits(context, &zero, &quotient)?;
    mux_bits(context, negate, &negated, &quotient)
}

fn binary_value<C>(
    context: &mut C,
    types: &TypeTable,
    ty: TypeId,
    op: TypedBinaryOp,
    a: TypedVoleValue<Wire<C>>,
    b: TypedVoleValue<Wire<C>>,
) -> Result<TypedVoleValue<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>
        + ContextWithMux<bool>,
    Wire<C>: Clone,
{
    let mut widths = Vec::new();
    scalar_segments(types, ty, &mut widths);
    let mut bits = Vec::with_capacity(a.bits.len());
    let mut offset = 0usize;
    for width in widths {
        let left = &a.bits[offset..offset + width];
        let right = &b.bits[offset..offset + width];
        let part = match op {
            TypedBinaryOp::Add => add_bits(context, left, right)?,
            TypedBinaryOp::Sub => sub_bits(context, left, right)?,
            TypedBinaryOp::Mul => mul_bits(context, left, right)?,
            TypedBinaryOp::Udiv => udiv_bits(context, left, right)?,
            TypedBinaryOp::Sdiv => sdiv_bits(context, left, right)?,
            TypedBinaryOp::And => and_bits(context, left, right)?,
            TypedBinaryOp::Or => or_bits(context, left, right)?,
            TypedBinaryOp::Xor => xor_bits(context, left, right)?,
            TypedBinaryOp::Shl => variable_shift_bits(context, left, right, true, false)?,
            TypedBinaryOp::Lshr => variable_shift_bits(context, left, right, false, false)?,
            TypedBinaryOp::Ashr => variable_shift_bits(context, left, right, false, true)?,
        };
        bits.extend(part);
        offset += width;
    }
    Ok(TypedVoleValue { bits })
}

fn compare_value<C>(
    context: &mut C,
    _types: &TypeTable,
    op: TypedCompareOp,
    a: TypedVoleValue<Wire<C>>,
    b: TypedVoleValue<Wire<C>>,
) -> Result<TypedVoleValue<Wire<C>>, C::Error>
where
    C: ContextWithCreate<bool>
        + ContextWithBitAnd<bool>
        + ContextWithBitOr<bool>
        + ContextWithBitXor<bool>,
    Wire<C>: Clone,
{
    let equal = eq_bit(context, &a.bits, &b.bits)?;
    let unsigned_less = ult_bit(context, &a.bits, &b.bits)?;
    let unsigned_greater = ult_bit(context, &b.bits, &a.bits)?;
    let signed_less = if a.bits.is_empty() {
        context.create(false)?
    } else {
        let mut left = a.bits.clone();
        let mut right = b.bits.clone();
        let last = left.len() - 1;
        let one = context.create(true)?;
        left[last] = context.bitxor(left[last].clone(), one)?;
        let one = context.create(true)?;
        right[last] = context.bitxor(right[last].clone(), one)?;
        ult_bit(context, &left, &right)?
    };
    let signed_greater = if a.bits.is_empty() {
        context.create(false)?
    } else {
        let mut left = a.bits.clone();
        let mut right = b.bits.clone();
        let last = left.len() - 1;
        let one = context.create(true)?;
        left[last] = context.bitxor(left[last].clone(), one)?;
        let one = context.create(true)?;
        right[last] = context.bitxor(right[last].clone(), one)?;
        ult_bit(context, &right, &left)?
    };
    let result = match op {
        TypedCompareOp::Eq => equal,
        TypedCompareOp::Ne => {
            let one = context.create(true)?;
            context.bitxor(equal, one)?
        }
        TypedCompareOp::Ult => unsigned_less,
        TypedCompareOp::Ule => context.bitor(unsigned_less, equal)?,
        TypedCompareOp::Ugt => unsigned_greater,
        TypedCompareOp::Uge => context.bitor(unsigned_greater, equal)?,
        TypedCompareOp::Slt => signed_less,
        TypedCompareOp::Sle => context.bitor(signed_less, equal)?,
        TypedCompareOp::Sgt => signed_greater,
        TypedCompareOp::Sge => context.bitor(signed_greater, equal)?,
    };
    Ok(TypedVoleValue {
        bits: alloc::vec![result],
    })
}

fn repack_value<W: Clone>(
    types: &TypeTable,
    ty: TypeId,
    bits: &[(TypedVoleValue<W>, u8)],
) -> TypedVoleValue<W> {
    assert_eq!(
        bits.len(),
        layout_bits(types, ty),
        "typed program validation guarantees a complete bit repack"
    );
    TypedVoleValue {
        bits: bits
            .iter()
            .map(|(value, bit)| value.bits[usize::from(*bit)].clone())
            .collect(),
    }
}

impl<N, T> TypedContext for VoleProverContext<'_, '_, N, T>
where
    N: VoleArray<T>,
    T: Clone + Add<Output = T> + Mul<Output = T> + Default,
{
    type Value = TypedVoleValue<Vope<N, T, cipher::consts::U1>>;

    fn constant(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        words: &[u64],
    ) -> Result<Self::Value, Self::Error> {
        typed_constant(self, types, ty, words)
    }

    fn binary(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        op: TypedBinaryOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        binary_value(self, types, ty, op, a, b)
    }

    fn not(
        &mut self,
        _types: &TypeTable,
        _ty: TypeId,
        value: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(TypedVoleValue {
            bits: not_bits(self, &value.bits)?,
        })
    }

    fn bit_repack(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        bits: &[(Self::Value, u8)],
    ) -> Result<Self::Value, Self::Error> {
        Ok(repack_value(types, ty, bits))
    }

    fn compare(
        &mut self,
        types: &TypeTable,
        _ty: TypeId,
        op: TypedCompareOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        compare_value(self, types, op, a, b)
    }

    fn mux(
        &mut self,
        _types: &TypeTable,
        _ty: TypeId,
        cond: Self::Value,
        then: Self::Value,
        r#else: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(TypedVoleValue {
            bits: mux_bits(self, cond.bits[0].clone(), &then.bits, &r#else.bits)?,
        })
    }
}

impl<N, T, I> TypedContext for VoleVerifierContext<N, T, I>
where
    N: ArraySize,
    T: Clone + Add<Output = T> + Mul<Output = T> + Invert + Default,
    I: Iterator<Item = hybrid_array::Array<T, N>>,
{
    type Value = TypedVoleValue<Q<N, T>>;

    fn constant(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        words: &[u64],
    ) -> Result<Self::Value, Self::Error> {
        typed_constant(self, types, ty, words)
    }

    fn binary(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        op: TypedBinaryOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        binary_value(self, types, ty, op, a, b)
    }

    fn not(
        &mut self,
        _types: &TypeTable,
        _ty: TypeId,
        value: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(TypedVoleValue {
            bits: not_bits(self, &value.bits)?,
        })
    }

    fn bit_repack(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        bits: &[(Self::Value, u8)],
    ) -> Result<Self::Value, Self::Error> {
        Ok(repack_value(types, ty, bits))
    }

    fn compare(
        &mut self,
        types: &TypeTable,
        _ty: TypeId,
        op: TypedCompareOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        compare_value(self, types, op, a, b)
    }

    fn mux(
        &mut self,
        _types: &TypeTable,
        _ty: TypeId,
        cond: Self::Value,
        then: Self::Value,
        r#else: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(TypedVoleValue {
            bits: mux_bits(self, cond.bits[0].clone(), &then.bits, &r#else.bits)?,
        })
    }
}
