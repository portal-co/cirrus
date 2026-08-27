//! Conversion from Volar IR's type table to Cirrus's portable mirror.

use alloc::vec::Vec;
use cirrus_recompile_core::{TypeId, TypeTable, VolarType};
use volar_ir::ir::{IRType, IRTypeId, IRTypes};
use volar_ir_common::Type;

/// Cirrus's mirror table together with the source-ID translation vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VolarTypeMap {
    /// Portable Cirrus type table.
    pub types: TypeTable,
    /// `ids[source.0]` is the equivalent ID in [`types`](Self::types).
    pub ids: Vec<TypeId>,
}

/// Why a Volar IR type table cannot be used by a GF(2) Cirrus context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VolarTypeMapError {
    /// A type reference was outside the Volar source table.
    InvalidReference(IRTypeId),
    /// A source type directly or indirectly contained itself.
    RecursiveType(IRTypeId),
    /// Z3 values do not have a GF(2) Cirrus representation.
    Z3(IRTypeId),
    /// Volar block/function types are control-flow descriptors, not values a
    /// typed Cirrus trace can contain.
    ControlType(IRTypeId),
    /// A source vector has no lanes.
    EmptyVector(IRTypeId),
}

/// Mirror every representable Volar IR type into a Cirrus [`TypeTable`].
///
/// Packed `_128`/`_256`, AES8, Galois64, lane vectors, and tuples retain
/// their exact identity.  Z3 and control-flow-only types are rejected at this
/// boundary, before any VOLE/garbling backend sees them.
pub fn lower_volar_types(source: &IRTypes) -> Result<VolarTypeMap, VolarTypeMapError> {
    let mut types = TypeTable::new();
    let mut ids = alloc::vec![None; source.0.len()];
    let mut visiting = alloc::vec![false; source.0.len()];
    for index in 0..source.0.len() {
        lower_type(
            IRTypeId(index as u32),
            source,
            &mut types,
            &mut ids,
            &mut visiting,
        )?;
    }
    Ok(VolarTypeMap {
        types,
        ids: ids
            .into_iter()
            .map(|id| id.expect("every source type was lowered"))
            .collect(),
    })
}

fn lower_type(
    id: IRTypeId,
    source: &IRTypes,
    types: &mut TypeTable,
    ids: &mut [Option<TypeId>],
    visiting: &mut [bool],
) -> Result<TypeId, VolarTypeMapError> {
    let index = id.0 as usize;
    if index >= source.0.len() {
        return Err(VolarTypeMapError::InvalidReference(id));
    }
    if let Some(mapped) = ids[index] {
        return Ok(mapped);
    }
    if visiting[index] {
        return Err(VolarTypeMapError::RecursiveType(id));
    }
    visiting[index] = true;
    let mapped = match &source.0[index] {
        IRType::Primitive(Type::Bit) => types.bit(),
        IRType::Primitive(Type::_8) => types.intern(VolarType::U8),
        IRType::Primitive(Type::_16) => types.intern(VolarType::U16),
        IRType::Primitive(Type::_32) => types.intern(VolarType::U32),
        IRType::Primitive(Type::_64) => types.intern(VolarType::U64),
        IRType::Primitive(Type::_128) => types.intern(VolarType::U128),
        IRType::Primitive(Type::_256) => types.intern(VolarType::U256),
        IRType::Primitive(Type::AES8) => types.intern(VolarType::Aes8),
        IRType::Primitive(Type::Galois64) => types.intern(VolarType::Galois64),
        IRType::Primitive(Type::Z3) => return Err(VolarTypeMapError::Z3(id)),
        IRType::Vec(lanes, element) => {
            if *lanes == 0 {
                return Err(VolarTypeMapError::EmptyVector(id));
            }
            let element = lower_type(*element, source, types, ids, visiting)?;
            types.intern(VolarType::Vector {
                lanes: *lanes,
                element,
            })
        }
        IRType::Tuple(fields) => {
            let fields = fields
                .iter()
                .map(|field| lower_type(*field, source, types, ids, visiting))
                .collect::<Result<Vec<_>, _>>()?;
            types.intern(VolarType::Tuple(fields))
        }
        IRType::Block { .. } | IRType::Func { .. } => {
            return Err(VolarTypeMapError::ControlType(id));
        }
        _ => return Err(VolarTypeMapError::ControlType(id)),
    };
    visiting[index] = false;
    ids[index] = Some(mapped);
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrors_wide_vectors_and_tuples_without_a_volar_dependency_leak() {
        let mut source = IRTypes::new();
        let wide = source.primitive(Type::_256);
        let word = source.primitive(Type::_32);
        let vector = source.push(IRType::Vec(4, word));
        let tuple = source.push(IRType::Tuple(alloc::vec![wide, vector]));

        let mapped = lower_volar_types(&source).unwrap();
        assert_eq!(
            mapped
                .types
                .layout(mapped.ids[wide.0 as usize])
                .unwrap()
                .bits,
            256
        );
        assert_eq!(
            mapped
                .types
                .layout(mapped.ids[vector.0 as usize])
                .unwrap()
                .bits,
            128
        );
        assert_eq!(
            mapped
                .types
                .layout(mapped.ids[tuple.0 as usize])
                .unwrap()
                .bits,
            384
        );
    }

    #[test]
    fn rejects_z3_before_typed_execution() {
        let mut source = IRTypes::new();
        let z3 = source.primitive(Type::Z3);
        assert_eq!(lower_volar_types(&source), Err(VolarTypeMapError::Z3(z3)));
    }
}
