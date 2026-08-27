//! Direct lowering from one straight-line Volar circuit to typed Cirrus IR.
//!
//! Boolar remains the bit-level interchange for existing circuit consumers.
//! This adapter is the parallel entry point for a Volar circuit whose values
//! should keep their native type identity until a typed host (for example the
//! VOLE implementation) chooses how to represent them.

use alloc::{string::String, vec::Vec};
use cirrus_recompile_core::{
    ExternalKind, Idx, TypeId, TypedActionTarget, TypedBinaryOp, TypedExternalOp, TypedOp,
    TypedProgram, TypedProgramError,
};
use volar_ir::ir::{IRBlockTargetId, IRBlocks, IRStmt, IRTerminator, IRTypeId, IRTypes, IRVarId};
use volar_ir_common::Constant;

use crate::{VolarTypeMapError, lower_volar_types};

/// Why a Volar circuit could not be represented by [`TypedProgram`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypedLowerError {
    /// The source type table contains a value Cirrus cannot represent.
    Types(VolarTypeMapError),
    /// Typed Cirrus programs are presently straight-line, but the source was
    /// not one block ending in a direct return.
    NotStraightLineCircuit,
    /// A statement referenced a source SSA value that was not available.
    MissingValue(IRVarId),
    /// A source aggregate was used where a projected value was required.
    AggregateValue(IRVarId),
    /// An aggregate projection was malformed.
    InvalidProjection(IRVarId),
    /// The source statement has no typed Cirrus equivalent yet.
    UnsupportedStatement,
    /// The constructed typed program violated its structural contracts.
    Program(TypedProgramError),
}

#[derive(Clone, Debug)]
enum ValueRef {
    Value(Idx),
    Oracle {
        name: String,
        args: Vec<Idx>,
        output_tys: Vec<TypeId>,
        occurrence: u64,
    },
    Action {
        name: String,
        guard: Idx,
        args: Vec<Idx>,
        fallbacks: Vec<Idx>,
        output_tys: Vec<TypeId>,
        occurrence: u64,
    },
}

struct Builder {
    program: TypedProgram,
    source_types: Vec<TypeId>,
    next_occurrence: u64,
}

impl Builder {
    fn source_ty(&self, ty: IRTypeId) -> Result<TypeId, TypedLowerError> {
        self.source_types
            .get(ty.0 as usize)
            .copied()
            .ok_or(TypedLowerError::UnsupportedStatement)
    }

    fn words(&self, ty: TypeId, constant: Constant) -> Result<Vec<u64>, TypedLowerError> {
        let words = self
            .program
            .types
            .layout(ty)
            .map_err(|_| TypedLowerError::UnsupportedStatement)?
            .words;
        let raw = [
            constant.lo as u64,
            (constant.lo >> 64) as u64,
            constant.hi as u64,
            (constant.hi >> 64) as u64,
        ];
        Ok((0..words)
            .map(|index| raw.get(index).copied().unwrap_or(0))
            .collect())
    }

    fn zero(&mut self, ty: TypeId) -> Result<Idx, TypedLowerError> {
        self.constant(ty, Constant { hi: 0, lo: 0 })
    }

    fn constant(&mut self, ty: TypeId, constant: Constant) -> Result<Idx, TypedLowerError> {
        let words = self.words(ty, constant)?;
        Ok(self.push(ty, TypedOp::Constant(words)))
    }

    fn push(&mut self, ty: TypeId, op: TypedOp) -> Idx {
        let slot = Idx(self.program.ops.len() as u32);
        self.program.ops.push(op);
        self.program.slot_tys.push(ty);
        slot
    }

    fn value(values: &[ValueRef], var: IRVarId) -> Result<Idx, TypedLowerError> {
        match values.get(var.0 as usize) {
            Some(ValueRef::Value(value)) => Ok(*value),
            Some(_) => Err(TypedLowerError::AggregateValue(var)),
            None => Err(TypedLowerError::MissingValue(var)),
        }
    }

    fn values(values: &[ValueRef], vars: &[IRVarId]) -> Result<Vec<Idx>, TypedLowerError> {
        vars.iter().map(|var| Self::value(values, *var)).collect()
    }

    fn lower_poly(
        &mut self,
        values: &[ValueRef],
        ty: TypeId,
        coeffs: &alloc::collections::BTreeMap<Vec<IRVarId>, u8>,
        constant: Constant,
    ) -> Result<Idx, TypedLowerError> {
        let is_bit = self.program.types.is_bit(ty);
        let mut acc = self.constant(ty, constant)?;
        for (monomial, coefficient) in coeffs {
            if coefficient & 1 == 0 {
                continue;
            }
            let product = if is_bit {
                let mut inputs = Self::values(values, monomial)?;
                let mut product = match inputs.pop() {
                    Some(product) => product,
                    None => self.constant(ty, Constant { hi: 0, lo: 1 })?,
                };
                for input in inputs {
                    product = self.push(
                        ty,
                        TypedOp::Binary {
                            op: TypedBinaryOp::And,
                            a: product,
                            b: input,
                        },
                    );
                }
                product
            } else {
                let mut dominant = None;
                let mut selectors = Vec::new();
                for var in monomial {
                    let value = Self::value(values, *var)?;
                    let value_ty = self.program.slot_tys[value.get()];
                    if self.program.types.is_bit(value_ty) {
                        selectors.push(value);
                    } else if value_ty == ty && dominant.replace(value).is_none() {
                        // The one non-Bit factor is the selected wide value.
                    } else {
                        return Err(TypedLowerError::UnsupportedStatement);
                    }
                }
                let Some(mut product) = dominant else {
                    // Volar's non-Bit Poly contract permits at most one wide
                    // factor, but bit-only monomials have no typed wide value
                    // to contribute without an explicit promotion operation.
                    return Err(TypedLowerError::UnsupportedStatement);
                };
                for selector in selectors {
                    let zero = self.zero(ty)?;
                    product = self.push(
                        ty,
                        TypedOp::Mux {
                            cond: selector,
                            then: product,
                            r#else: zero,
                        },
                    );
                }
                product
            };
            acc = self.push(
                ty,
                TypedOp::Binary {
                    op: TypedBinaryOp::Xor,
                    a: acc,
                    b: product,
                },
            );
        }
        Ok(acc)
    }

    fn lower_rotate(
        &mut self,
        source: Idx,
        ty: TypeId,
        amount: usize,
        right: bool,
    ) -> Result<Idx, TypedLowerError> {
        let width = self
            .program
            .types
            .layout(ty)
            .map_err(|_| TypedLowerError::UnsupportedStatement)?
            .bits;
        if width == 0 {
            return Err(TypedLowerError::UnsupportedStatement);
        }
        let amount = amount % width;
        if amount == 0 {
            return Ok(self.push(ty, TypedOp::Copy(source)));
        }
        let left = if right { width - amount } else { amount };
        let right_amount = width - left;
        let left_shift = self.constant(
            ty,
            Constant {
                hi: 0,
                lo: left as u128,
            },
        )?;
        let right_shift = self.constant(
            ty,
            Constant {
                hi: 0,
                lo: right_amount as u128,
            },
        )?;
        let shifted_left = self.push(
            ty,
            TypedOp::Binary {
                op: TypedBinaryOp::Shl,
                a: source,
                b: left_shift,
            },
        );
        let shifted_right = self.push(
            ty,
            TypedOp::Binary {
                op: TypedBinaryOp::Lshr,
                a: source,
                b: right_shift,
            },
        );
        Ok(self.push(
            ty,
            TypedOp::Binary {
                op: TypedBinaryOp::Or,
                a: shifted_left,
                b: shifted_right,
            },
        ))
    }

    fn external(
        &mut self,
        kind: ExternalKind,
        name: String,
        args: Vec<Idx>,
        result_ty: TypeId,
        result_index: usize,
        occurrence: u64,
    ) -> u32 {
        let id = self.program.externals.len() as u32;
        self.program.externals.push(TypedExternalOp {
            kind,
            name,
            args,
            result_tys: alloc::vec![result_ty],
            result_index,
            occurrence,
        });
        id
    }
}

/// Lower a straight-line, return-terminated Volar circuit to [`TypedProgram`].
///
/// The result keeps the original value type in every slot.  External calls
/// remain declarations in `program.externals`: oracle/RNG values and legacy
/// action outputs use [`TypedOp::External`], while [`IRStmt::ActionStore`]
/// becomes an effect-only [`TypedOp::ActionStore`] with direct storage targets.
pub fn lower_volar_circuit<P: Clone>(
    blocks: &IRBlocks<P>,
    source_types: &IRTypes,
) -> Result<TypedProgram, TypedLowerError> {
    if blocks.blocks.len() != 1 {
        return Err(TypedLowerError::NotStraightLineCircuit);
    }
    let block = &blocks.blocks[0];
    let return_vars = match &block.terminator {
        IRTerminator::Jmp { target } if matches!(target.dest, IRBlockTargetId::Return) => {
            &target.args
        }
        _ => return Err(TypedLowerError::NotStraightLineCircuit),
    };
    let map = lower_volar_types(source_types).map_err(TypedLowerError::Types)?;
    let mut builder = Builder {
        program: TypedProgram {
            types: map.types,
            ops: Vec::new(),
            slot_tys: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            externals: Vec::new(),
        },
        source_types: map.ids,
        next_occurrence: 0,
    };

    // Inputs occupy normal constant slots, which the typed executor replaces
    // with caller values before evaluating any instruction.
    let mut values = Vec::with_capacity(block.params.len() + block.stmts.len());
    for ty in &block.params {
        let ty = builder.source_ty(*ty)?;
        let input = builder.zero(ty)?;
        builder.program.inputs.push(input);
        values.push(ValueRef::Value(input));
    }

    for node in &block.stmts {
        let value = match &node.kind {
            IRStmt::Const(constant, ty) => {
                ValueRef::Value(builder.constant(builder.source_ty(*ty)?, *constant)?)
            }
            IRStmt::Transmute {
                src,
                src_ty: _,
                dst_ty,
            } => {
                let source = Builder::value(&values, *src)?;
                let destination_ty = builder.source_ty(*dst_ty)?;
                ValueRef::Value(builder.push(destination_ty, TypedOp::Copy(source)))
            }
            IRStmt::Poly {
                ty,
                coeffs,
                constant,
            } => ValueRef::Value(builder.lower_poly(
                &values,
                builder.source_ty(*ty)?,
                coeffs,
                *constant,
            )?),
            IRStmt::Rol { src, ty, n } => ValueRef::Value(builder.lower_rotate(
                Builder::value(&values, *src)?,
                builder.source_ty(*ty)?,
                *n,
                false,
            )?),
            IRStmt::Ror { src, ty, n } => ValueRef::Value(builder.lower_rotate(
                Builder::value(&values, *src)?,
                builder.source_ty(*ty)?,
                *n,
                true,
            )?),
            IRStmt::OracleCall {
                name,
                args,
                output_tys,
                result_ty: _,
            } => {
                let occurrence = builder.next_occurrence;
                builder.next_occurrence += 1;
                ValueRef::Oracle {
                    name: name.clone(),
                    args: Builder::values(&values, args)?,
                    output_tys: output_tys
                        .iter()
                        .map(|ty| builder.source_ty(*ty))
                        .collect::<Result<_, _>>()?,
                    occurrence,
                }
            }
            IRStmt::OracleOutput { call, idx, ty } => {
                let ValueRef::Oracle {
                    name,
                    args,
                    output_tys,
                    occurrence,
                } = values
                    .get(call.0 as usize)
                    .ok_or(TypedLowerError::MissingValue(*call))?
                else {
                    return Err(TypedLowerError::InvalidProjection(*call));
                };
                let result_ty = builder.source_ty(*ty)?;
                if output_tys.get(*idx) != Some(&result_ty) {
                    return Err(TypedLowerError::InvalidProjection(*call));
                }
                let external = builder.external(
                    ExternalKind::Oracle,
                    name.clone(),
                    args.clone(),
                    result_ty,
                    *idx,
                    *occurrence,
                );
                ValueRef::Value(builder.push(result_ty, TypedOp::External(external)))
            }
            IRStmt::ActionCall {
                name,
                guard,
                args,
                fallbacks,
                output_tys,
                result_ty: _,
            } => {
                let occurrence = builder.next_occurrence;
                builder.next_occurrence += 1;
                ValueRef::Action {
                    name: name.clone(),
                    guard: Builder::value(&values, *guard)?,
                    args: Builder::values(&values, args)?,
                    fallbacks: Builder::values(&values, fallbacks)?,
                    output_tys: output_tys
                        .iter()
                        .map(|ty| builder.source_ty(*ty))
                        .collect::<Result<_, _>>()?,
                    occurrence,
                }
            }
            IRStmt::ActionOutput { call, idx, ty } => {
                let ValueRef::Action {
                    name,
                    guard,
                    args,
                    fallbacks,
                    output_tys,
                    occurrence,
                } = values
                    .get(call.0 as usize)
                    .ok_or(TypedLowerError::MissingValue(*call))?
                else {
                    return Err(TypedLowerError::InvalidProjection(*call));
                };
                let result_ty = builder.source_ty(*ty)?;
                let fallback = *fallbacks
                    .get(*idx)
                    .ok_or(TypedLowerError::InvalidProjection(*call))?;
                if output_tys.get(*idx) != Some(&result_ty) {
                    return Err(TypedLowerError::InvalidProjection(*call));
                }
                let external = builder.external(
                    ExternalKind::Action,
                    name.clone(),
                    args.clone(),
                    result_ty,
                    *idx,
                    *occurrence,
                );
                let result = builder.push(result_ty, TypedOp::External(external));
                ValueRef::Value(builder.push(
                    result_ty,
                    TypedOp::Mux {
                        cond: *guard,
                        then: result,
                        r#else: fallback,
                    },
                ))
            }
            IRStmt::ActionStore {
                name,
                guard,
                args,
                fallbacks,
                output_tys,
                targets,
            } => {
                let result_tys = output_tys
                    .iter()
                    .map(|ty| builder.source_ty(*ty))
                    .collect::<Result<Vec<_>, _>>()?;
                if result_tys.len() != targets.len() || result_tys.len() != fallbacks.len() {
                    return Err(TypedLowerError::UnsupportedStatement);
                }
                let occurrence = builder.next_occurrence;
                builder.next_occurrence += 1;
                let external = builder.program.externals.len() as u32;
                builder.program.externals.push(TypedExternalOp {
                    kind: ExternalKind::Action,
                    name: name.clone(),
                    args: Builder::values(&values, args)?,
                    result_tys,
                    result_index: 0,
                    occurrence,
                });
                let targets = targets
                    .iter()
                    .map(|target| {
                        Ok(TypedActionTarget {
                            storage: target.storage.0 as u64,
                            address: Builder::value(&values, target.addr)?,
                        })
                    })
                    .collect::<Result<Vec<_>, TypedLowerError>>()?;
                let effect = builder.push(
                    builder.program.types.bit(),
                    TypedOp::ActionStore {
                        external,
                        guard: Builder::value(&values, *guard)?,
                        fallbacks: Builder::values(&values, fallbacks)?,
                        targets,
                    },
                );
                ValueRef::Value(effect)
            }
            IRStmt::Rng { name, ty } => {
                let ty = builder.source_ty(*ty)?;
                let occurrence = builder.next_occurrence;
                builder.next_occurrence += 1;
                let external = builder.external(
                    ExternalKind::Rng,
                    name.clone(),
                    Vec::new(),
                    ty,
                    0,
                    occurrence,
                );
                ValueRef::Value(builder.push(ty, TypedOp::External(external)))
            }
            IRStmt::StorageRead { .. }
            | IRStmt::StorageWrite { .. }
            | IRStmt::Merge { .. }
            | IRStmt::Splat { .. }
            | IRStmt::Shuffle { .. } => return Err(TypedLowerError::UnsupportedStatement),
            _ => return Err(TypedLowerError::UnsupportedStatement),
        };
        values.push(value);
    }

    builder.program.outputs = Builder::values(&values, return_vars)?;
    builder
        .program
        .validate()
        .map_err(TypedLowerError::Program)?;
    Ok(builder.program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volar_ir::ir::{IRBlock, IRBranchTarget, IRType};
    use volar_ir_common::{ActionTarget, StorageId, Type};

    #[test]
    fn lowers_typed_externals_without_erasing_wide_values() {
        let mut types = IRTypes::new();
        let bit = types.bit();
        let wide = types.primitive(Type::_256);
        let address = types.primitive(Type::_64);
        let pair = types.push(IRType::Tuple(alloc::vec![wide]));
        let mut block = IRBlock {
            params: alloc::vec![bit, wide, wide, address],
            stmts: Vec::new(),
            terminator: IRTerminator::Jmp {
                target: IRBranchTarget::new(IRBlockTargetId::Return, alloc::vec![IRVarId(6)]),
            },
        };
        block.push_stmt(
            IRStmt::Rng {
                name: "nonce".into(),
                ty: wide,
            },
            (),
        );
        block.push_stmt(
            IRStmt::OracleCall {
                name: "lookup".into(),
                args: alloc::vec![IRVarId(4)],
                output_tys: alloc::vec![wide],
                result_ty: pair,
            },
            (),
        );
        block.push_stmt(
            IRStmt::OracleOutput {
                call: IRVarId(5),
                idx: 0,
                ty: wide,
            },
            (),
        );
        block.push_stmt(
            IRStmt::ActionStore {
                name: "commit".into(),
                guard: IRVarId(0),
                args: alloc::vec![IRVarId(1)],
                fallbacks: alloc::vec![IRVarId(2)],
                output_tys: alloc::vec![wide],
                targets: alloc::vec![ActionTarget {
                    storage: StorageId(3),
                    addr: IRVarId(3),
                }],
            },
            (),
        );

        let program = lower_volar_circuit(&IRBlocks::new(alloc::vec![block]), &types).unwrap();
        assert_eq!(program.inputs.len(), 4);
        assert_eq!(program.outputs.len(), 1);
        assert_eq!(program.externals.len(), 3);
        assert!(matches!(program.externals[0].kind, ExternalKind::Rng));
        assert!(matches!(program.externals[1].kind, ExternalKind::Oracle));
        assert!(matches!(program.externals[2].kind, ExternalKind::Action));
        assert!(matches!(
            program.ops.last(),
            Some(TypedOp::ActionStore { .. })
        ));
        assert_eq!(program.types.layout(program.slot_tys[4]).unwrap().bits, 256);
    }
}
