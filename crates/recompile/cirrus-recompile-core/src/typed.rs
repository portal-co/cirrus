//! Typed Cirrus program IR and its pluggable execution seam.
//!
//! The historical [`crate::Program`] is intentionally Boolean-only.  These
//! types retain Volar's GF(2) type information without tying Cirrus to the
//! Volar crates, and provide a direct typed-context execution path.  A host
//! may implement the context with native machine integers, authenticated VOLE
//! values, garbled values, or a lowering to Boolean wires.

use alloc::{string::String, vec::Vec};

use crate::{ExternalKind, Idx, TypeError, TypeId, TypeTable};
use cirrus_core::HasError;

/// A binary operation over two values of the same Volar type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypedBinaryOp {
    /// Modular addition.
    Add,
    /// Modular subtraction.
    Sub,
    /// Modular multiplication.
    Mul,
    /// Unsigned division.
    Udiv,
    /// Signed two's-complement division.
    Sdiv,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Shift left by the right operand.
    Shl,
    /// Logical shift right by the right operand.
    Lshr,
    /// Arithmetic shift right by the right operand.
    Ashr,
}

/// A comparison operation.  Its result always has the table's bit type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypedCompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
}

/// One typed external declaration, resolved by an execution host.
///
/// Oracle/RNG declarations, and legacy action-result declarations, have
/// exactly one [`result_tys`](Self::result_tys) item and are referenced
/// through [`TypedOp::External`].  A direct action may have multiple results
/// and is referenced through [`TypedOp::ActionStore`], which gives the host
/// every direct storage destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedExternalOp {
    /// External primitive class.
    pub kind: ExternalKind,
    /// Logical source name.
    pub name: String,
    /// Argument slot IDs in declared order.
    pub args: Vec<Idx>,
    /// Result types in declared order.
    pub result_tys: Vec<TypeId>,
    /// Declared result position.  Boolean-derived external traces use this
    /// to retain the source bit index; typed sources normally use zero.
    pub result_index: usize,
    /// Deterministic occurrence token for replay and RNG freshness.
    pub occurrence: u64,
}

/// One direct storage target for an [`TypedOp::ActionStore`] result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypedActionTarget {
    /// Logical Volar storage namespace.
    pub storage: u64,
    /// Slot holding the destination element address.
    pub address: Idx,
}

/// One typed straight-line instruction.
///
/// Like [`crate::Op`], an instruction's position is its output slot.  An
/// `ActionStore` is effect-only; it reserves a Boolean placeholder slot so
/// that source lowering can retain stable SSA numbering, and validation
/// rejects later uses of that placeholder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypedOp {
    /// Materialize a public constant represented as little-endian `u64`
    /// words, padded to its declared type's layout.
    Constant(Vec<u64>),
    /// Reinterpret an unchanged canonical bit layout as another type.
    ///
    /// This is the typed-program counterpart of Volar IR's `Transmute`.
    /// It performs no context operation; validation requires equal layouts.
    Copy(Idx),
    /// Assemble a value from individually selected source bits, in canonical
    /// little-endian output order.
    ///
    /// This retains Volar IR's `Merge`, `Splat`, and `Shuffle` semantics in
    /// the typed program rather than forcing an early conversion to Boolean
    /// `Program` slots.
    BitRepack(Vec<(Idx, u8)>),
    /// Apply one same-type binary operation.
    Binary {
        /// Operation selector.
        op: TypedBinaryOp,
        /// First input slot.
        a: Idx,
        /// Second input slot.
        b: Idx,
    },
    /// Invert every bit in the source value.
    Not(Idx),
    /// Compare two same-type inputs, producing one bit.
    Compare {
        /// Comparison selector.
        op: TypedCompareOp,
        /// First input slot.
        a: Idx,
        /// Second input slot.
        b: Idx,
    },
    /// Select a same-type value using a Boolean condition.
    Mux {
        /// Boolean condition slot.
        cond: Idx,
        /// Selected when `cond` is true.
        then: Idx,
        /// Selected when `cond` is false.
        r#else: Idx,
    },
    /// Resolve a one-result oracle or RNG declaration.
    External(u32),
    /// Invoke a multi-result action and direct every output to storage.
    ActionStore {
        /// Index into [`TypedProgram::externals`].
        external: u32,
        /// Boolean guard slot.
        guard: Idx,
        /// Fallback value per action result.
        fallbacks: Vec<Idx>,
        /// Storage destination per action result.
        targets: Vec<TypedActionTarget>,
    },
}

/// A typed, straight-line Cirrus program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedProgram {
    /// Cirrus-owned mirror of the Volar type table.
    pub types: TypeTable,
    /// Instruction trace; its position is each result's [`Idx`].
    pub ops: Vec<TypedOp>,
    /// Exact type of each slot in [`ops`](Self::ops).
    pub slot_tys: Vec<TypeId>,
    /// Input slots in caller-visible order.
    pub inputs: Vec<Idx>,
    /// Output slots in caller-visible order.
    pub outputs: Vec<Idx>,
    /// Pluggable external primitive declarations.
    pub externals: Vec<TypedExternalOp>,
}

/// Structural validation failure for [`TypedProgram`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypedProgramError {
    /// The mirrored Volar type table is malformed.
    Type(TypeError),
    /// `slot_tys` did not have exactly one entry per instruction.
    SlotTypeCount,
    /// A slot reference does not name an earlier program slot.
    InvalidSlot(Idx),
    /// A slot's declared type does not satisfy an operation's contract.
    TypeMismatch,
    /// A constant had the wrong number of canonical `u64` words.
    ConstantWords,
    /// An external ID was absent or incompatible with the instruction.
    InvalidExternal,
    /// An `ActionStore` effect placeholder was consumed as a data value.
    EffectValueUsed,
    /// Boolean storage operations have no typed-program counterpart yet.
    StorageUnsupported,
}

impl TypedProgram {
    /// Convert the original Boolean trace into a typed trace whose every
    /// slot has the canonical `Bit` type.  External occurrence and output-bit
    /// metadata are retained exactly, so callers can switch a host from the
    /// Boolean executor to the typed dispatcher without recompiling.
    pub fn from_boolean(program: &crate::Program) -> Result<Self, TypedProgramError> {
        let types = TypeTable::new();
        let bit = types.bit();
        let externals = program
            .externals
            .iter()
            .map(|external| TypedExternalOp {
                kind: external.kind,
                name: external.name.clone(),
                args: external.args.clone(),
                result_tys: alloc::vec![bit],
                result_index: external.bit,
                occurrence: external.occurrence,
            })
            .collect::<Vec<_>>();
        let mut ops = Vec::with_capacity(program.ops.len());
        for op in &program.ops {
            ops.push(match *op {
                crate::Op::Create(value) => TypedOp::Constant(alloc::vec![value as u64]),
                crate::Op::BitAnd(a, b) => TypedOp::Binary {
                    op: TypedBinaryOp::And,
                    a,
                    b,
                },
                crate::Op::BitOr(a, b) => TypedOp::Binary {
                    op: TypedBinaryOp::Or,
                    a,
                    b,
                },
                crate::Op::BitXor(a, b) => TypedOp::Binary {
                    op: TypedBinaryOp::Xor,
                    a,
                    b,
                },
                crate::Op::Mux { cond, then, r#else } => TypedOp::Mux { cond, then, r#else },
                crate::Op::External(external) => {
                    if external as usize >= externals.len() {
                        return Err(TypedProgramError::InvalidExternal);
                    }
                    TypedOp::External(external)
                }
                crate::Op::Storage(_) => return Err(TypedProgramError::StorageUnsupported),
            });
        }
        let typed = Self {
            types,
            slot_tys: alloc::vec![bit; ops.len()],
            ops,
            inputs: program.inputs.clone(),
            outputs: program.outputs.clone(),
            externals,
        };
        typed.validate()?;
        Ok(typed)
    }

    /// Validate all type, slot-order, and external contracts.
    pub fn validate(&self) -> Result<(), TypedProgramError> {
        self.types.validate().map_err(TypedProgramError::Type)?;
        if self.ops.len() != self.slot_tys.len() {
            return Err(TypedProgramError::SlotTypeCount);
        }
        for (index, op) in self.ops.iter().enumerate() {
            let out = Idx(index as u32);
            let out_ty = self.slot_tys[index];
            self.validate_op(out, out_ty, op)?;
        }
        for slot in self.inputs.iter().chain(&self.outputs) {
            self.slot_ty(*slot)?;
        }
        Ok(())
    }

    /// Return a slot's declared type after validating its index.
    pub fn slot_ty(&self, slot: Idx) -> Result<TypeId, TypedProgramError> {
        self.slot_tys
            .get(slot.get())
            .copied()
            .ok_or(TypedProgramError::InvalidSlot(slot))
    }

    /// Create a prepared artifact retaining the exact type table and slot
    /// identities.  Typed loop reabstraction intentionally starts as a
    /// separate pass: Boolean [`crate::PreparedProgram`] loop templates are
    /// not safe to reuse for heterogeneous values.
    pub fn prepare(&self) -> Result<TypedPreparedProgram, TypedProgramError> {
        self.validate()?;
        Ok(TypedPreparedProgram {
            program: self.clone(),
        })
    }

    fn validate_op(&self, out: Idx, out_ty: TypeId, op: &TypedOp) -> Result<(), TypedProgramError> {
        let type_of = |slot| self.earlier_slot_ty(slot, out);
        match op {
            TypedOp::Constant(words) => {
                let layout = self.types.layout(out_ty).map_err(TypedProgramError::Type)?;
                if words.len() != layout.words {
                    return Err(TypedProgramError::ConstantWords);
                }
            }
            TypedOp::Copy(value) => {
                let input_ty = type_of(*value)?;
                if self
                    .types
                    .layout(input_ty)
                    .map_err(TypedProgramError::Type)?
                    != self.types.layout(out_ty).map_err(TypedProgramError::Type)?
                {
                    return Err(TypedProgramError::TypeMismatch);
                }
            }
            TypedOp::BitRepack(bits) => {
                let layout = self.types.layout(out_ty).map_err(TypedProgramError::Type)?;
                if bits.len() != layout.bits {
                    return Err(TypedProgramError::TypeMismatch);
                }
                for (value, bit) in bits {
                    let input = type_of(*value)?;
                    if usize::from(*bit)
                        >= self
                            .types
                            .layout(input)
                            .map_err(TypedProgramError::Type)?
                            .bits
                    {
                        return Err(TypedProgramError::TypeMismatch);
                    }
                }
            }
            TypedOp::Binary { a, b, .. } => {
                if type_of(*a)? != out_ty || type_of(*b)? != out_ty {
                    return Err(TypedProgramError::TypeMismatch);
                }
            }
            TypedOp::Not(value) => {
                if type_of(*value)? != out_ty {
                    return Err(TypedProgramError::TypeMismatch);
                }
            }
            TypedOp::Compare { a, b, .. } => {
                if !self.types.is_bit(out_ty) || type_of(*a)? != type_of(*b)? {
                    return Err(TypedProgramError::TypeMismatch);
                }
            }
            TypedOp::Mux { cond, then, r#else } => {
                if !self.types.is_bit(type_of(*cond)?)
                    || type_of(*then)? != out_ty
                    || type_of(*r#else)? != out_ty
                {
                    return Err(TypedProgramError::TypeMismatch);
                }
            }
            TypedOp::External(external) => {
                let external = self
                    .externals
                    .get(*external as usize)
                    .ok_or(TypedProgramError::InvalidExternal)?;
                if external.result_tys.as_slice() != [out_ty] {
                    return Err(TypedProgramError::InvalidExternal);
                }
                for arg in &external.args {
                    type_of(*arg)?;
                }
            }
            TypedOp::ActionStore {
                external,
                guard,
                fallbacks,
                targets,
            } => {
                let external = self
                    .externals
                    .get(*external as usize)
                    .ok_or(TypedProgramError::InvalidExternal)?;
                if external.kind != ExternalKind::Action
                    || !self.types.is_bit(out_ty)
                    || !self.types.is_bit(type_of(*guard)?)
                    || external.result_tys.len() != fallbacks.len()
                    || external.result_tys.len() != targets.len()
                {
                    return Err(TypedProgramError::InvalidExternal);
                }
                for arg in &external.args {
                    type_of(*arg)?;
                }
                for ((fallback, expected), target) in
                    fallbacks.iter().zip(&external.result_tys).zip(targets)
                {
                    if type_of(*fallback)? != *expected {
                        return Err(TypedProgramError::TypeMismatch);
                    }
                    type_of(target.address)?;
                }
            }
        }
        Ok(())
    }

    fn earlier_slot_ty(&self, slot: Idx, out: Idx) -> Result<TypeId, TypedProgramError> {
        if slot >= out {
            return Err(TypedProgramError::InvalidSlot(slot));
        }
        if matches!(self.ops[slot.get()], TypedOp::ActionStore { .. }) {
            return Err(TypedProgramError::EffectValueUsed);
        }
        self.slot_ty(slot)
    }
}

/// A prepared typed program.
///
/// The initial representation is intentionally identity-prepared, but still
/// owns a fully validated immutable program.  This gives typed backends the
/// same `Program`/`PreparedProgram` hand-off point as Boolean backends before
/// heterogeneous loop compression is introduced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedPreparedProgram {
    /// Validated direct typed program.
    pub program: TypedProgram,
}

impl TypedPreparedProgram {
    /// Revalidate the retained program.
    pub fn validate(&self) -> Result<(), TypedProgramError> {
        self.program.validate()
    }

    /// Execute the prepared typed artifact through the same direct host seam
    /// as [`TypedProgram`].  Preparation currently preserves heterogeneous
    /// slot identities, so no type metadata is discarded on this path.
    pub fn execute<C, E>(
        &self,
        context: &mut C,
        inputs: &[C::Value],
        externals: &mut E,
    ) -> Result<Vec<C::Value>, TypedExecutionError<C::Error>>
    where
        C: TypedContext,
        E: TypedExternalRegistry<C>,
    {
        execute_typed(context, &self.program, inputs, externals)
    }
}

/// A direct typed execution backend.
///
/// Operations receive the explicit [`TypeTable`] and result type, so an
/// implementation can use native integers/vectors, a typed VOLE gadget, or
/// an internal Boolean lowering without losing source type identity.
pub trait TypedContext: HasError {
    /// Backend-specific representation of one typed Cirrus value.
    type Value: Clone;

    /// Materialize a little-endian public constant.
    fn constant(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        words: &[u64],
    ) -> Result<Self::Value, Self::Error>;

    /// Apply a same-type binary operation.
    fn binary(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        op: TypedBinaryOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Invert a typed value bitwise.
    fn not(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        value: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Assemble one typed value from source bits in canonical output order.
    fn bit_repack(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        bits: &[(Self::Value, u8)],
    ) -> Result<Self::Value, Self::Error>;

    /// Compare two same-type values, returning the table's bit type.
    fn compare(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        op: TypedCompareOp,
        a: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Select between two values of `ty` using a Boolean condition.
    fn mux(
        &mut self,
        types: &TypeTable,
        ty: TypeId,
        cond: Self::Value,
        then: Self::Value,
        r#else: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

/// Host dispatch for typed external primitives.
///
/// This is intentionally a separate pluggable interface: ZK-oriented hosts
/// may supply a gadget or commitment-aware implementation instead of having
/// external calls rejected by the generic executor.
pub trait TypedExternalRegistry<C: TypedContext> {
    /// Resolve one typed pure-oracle output.
    fn oracle(
        &mut self,
        context: &mut C,
        types: &TypeTable,
        name: &str,
        args: &[C::Value],
        result_ty: TypeId,
        result_index: usize,
        occurrence: u64,
    ) -> Result<C::Value, C::Error>;

    /// Resolve one typed fresh RNG output.
    fn rng(
        &mut self,
        context: &mut C,
        types: &TypeTable,
        name: &str,
        result_ty: TypeId,
        result_index: usize,
        occurrence: u64,
    ) -> Result<C::Value, C::Error>;

    /// Resolve one legacy action result.  New Volar lowering uses
    /// [`action_store`](Self::action_store) instead; this method exists so
    /// typed conversion remains compatible with recorded Boolean traces.
    fn action(
        &mut self,
        context: &mut C,
        types: &TypeTable,
        name: &str,
        args: &[C::Value],
        result_ty: TypeId,
        result_index: usize,
        occurrence: u64,
    ) -> Result<C::Value, C::Error>;

    /// Execute one direct typed action storage effect.
    #[allow(clippy::too_many_arguments)]
    fn action_store(
        &mut self,
        context: &mut C,
        types: &TypeTable,
        name: &str,
        guard: C::Value,
        args: &[C::Value],
        fallbacks: &[C::Value],
        result_tys: &[TypeId],
        targets: &[TypedActionTarget],
        addresses: &[C::Value],
        occurrence: u64,
    ) -> Result<(), C::Error>;
}

/// Execute a typed program through a direct typed context and external host.
pub fn execute_typed<C, E>(
    context: &mut C,
    program: &TypedProgram,
    inputs: &[C::Value],
    externals: &mut E,
) -> Result<Vec<C::Value>, TypedExecutionError<C::Error>>
where
    C: TypedContext,
    E: TypedExternalRegistry<C>,
{
    program.validate().map_err(TypedExecutionError::Program)?;
    if inputs.len() != program.inputs.len() {
        return Err(TypedExecutionError::InputCount {
            expected: program.inputs.len(),
            found: inputs.len(),
        });
    }
    let mut values = (0..program.ops.len()).map(|_| None).collect::<Vec<_>>();
    let mut is_input = alloc::vec![false; program.ops.len()];
    for (slot, input) in program.inputs.iter().zip(inputs) {
        values[slot.get()] = Some(input.clone());
        is_input[slot.get()] = true;
    }

    for (index, op) in program.ops.iter().enumerate() {
        if is_input[index] {
            continue;
        }
        if let TypedOp::Copy(input) = op {
            values[index] = Some(read_value(&values, *input)?);
            continue;
        }
        let out_ty = program.slot_tys[index];
        let value = match op {
            TypedOp::Constant(words) => context.constant(&program.types, out_ty, words),
            TypedOp::Binary { op, a, b } => context.binary(
                &program.types,
                out_ty,
                *op,
                read_value(&values, *a)?,
                read_value(&values, *b)?,
            ),
            TypedOp::Not(input) => {
                context.not(&program.types, out_ty, read_value(&values, *input)?)
            }
            TypedOp::BitRepack(bits) => context.bit_repack(
                &program.types,
                out_ty,
                &bits
                    .iter()
                    .map(|(value, bit)| Ok((read_value(&values, *value)?, *bit)))
                    .collect::<Result<Vec<_>, TypedExecutionError<C::Error>>>()?,
            ),
            TypedOp::Compare { op, a, b } => context.compare(
                &program.types,
                program.slot_ty(*a).map_err(TypedExecutionError::Program)?,
                *op,
                read_value(&values, *a)?,
                read_value(&values, *b)?,
            ),
            TypedOp::Mux { cond, then, r#else } => context.mux(
                &program.types,
                out_ty,
                read_value(&values, *cond)?,
                read_value(&values, *then)?,
                read_value(&values, *r#else)?,
            ),
            TypedOp::External(external) => {
                let external = program.externals.get(*external as usize).ok_or(
                    TypedExecutionError::Program(TypedProgramError::InvalidExternal),
                )?;
                let args = read_values(&values, &external.args)?;
                match external.kind {
                    ExternalKind::Oracle => externals.oracle(
                        context,
                        &program.types,
                        &external.name,
                        &args,
                        out_ty,
                        external.result_index,
                        external.occurrence,
                    ),
                    ExternalKind::Rng => externals.rng(
                        context,
                        &program.types,
                        &external.name,
                        out_ty,
                        external.result_index,
                        external.occurrence,
                    ),
                    ExternalKind::Action => externals.action(
                        context,
                        &program.types,
                        &external.name,
                        &args,
                        out_ty,
                        external.result_index,
                        external.occurrence,
                    ),
                }
            }
            TypedOp::ActionStore {
                external,
                guard,
                fallbacks,
                targets,
            } => {
                let external = program.externals.get(*external as usize).ok_or(
                    TypedExecutionError::Program(TypedProgramError::InvalidExternal),
                )?;
                let args = read_values(&values, &external.args)?;
                let fallbacks = read_values(&values, fallbacks)?;
                let addresses = targets
                    .iter()
                    .map(|target| read_value(&values, target.address))
                    .collect::<Result<Vec<_>, _>>()?;
                externals
                    .action_store(
                        context,
                        &program.types,
                        &external.name,
                        read_value(&values, *guard)?,
                        &args,
                        &fallbacks,
                        &external.result_tys,
                        targets,
                        &addresses,
                        external.occurrence,
                    )
                    .map_err(TypedExecutionError::Context)?;
                // Effect-only operations do not manufacture a value.  They
                // retain their unused Boolean placeholder for stable slots.
                context.constant(&program.types, out_ty, &[0])
            }
            TypedOp::Copy(_) => {
                unreachable!("copy instructions are handled before context dispatch")
            }
        }
        .map_err(TypedExecutionError::Context)?;
        values[index] = Some(value);
    }

    program
        .outputs
        .iter()
        .map(|slot| read_value(&values, *slot))
        .collect()
}

fn read_value<Value: Clone, Error>(
    values: &[Option<Value>],
    slot: Idx,
) -> Result<Value, TypedExecutionError<Error>> {
    values
        .get(slot.get())
        .and_then(Clone::clone)
        .ok_or(TypedExecutionError::UninitializedSlot(slot))
}

fn read_values<Value: Clone, Error>(
    values: &[Option<Value>],
    slots: &[Idx],
) -> Result<Vec<Value>, TypedExecutionError<Error>> {
    slots.iter().map(|slot| read_value(values, *slot)).collect()
}

/// Failure while executing a [`TypedProgram`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypedExecutionError<ContextError> {
    /// The program was structurally invalid.
    Program(TypedProgramError),
    /// Caller supplied the wrong number of typed inputs.
    InputCount {
        /// Program input count.
        expected: usize,
        /// Caller input count.
        found: usize,
    },
    /// An operation read a slot which had not been materialized.
    UninitializedSlot(Idx),
    /// The typed context or external host failed.
    Context(ContextError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    struct U64Context;

    impl HasError for U64Context {
        type Error = Infallible;
    }

    impl TypedContext for U64Context {
        type Value = u64;

        fn constant(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            words: &[u64],
        ) -> Result<Self::Value, Self::Error> {
            Ok(words[0])
        }

        fn binary(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            op: TypedBinaryOp,
            a: Self::Value,
            b: Self::Value,
        ) -> Result<Self::Value, Self::Error> {
            Ok(match op {
                TypedBinaryOp::Add => a.wrapping_add(b),
                TypedBinaryOp::Sub => a.wrapping_sub(b),
                TypedBinaryOp::Mul => a.wrapping_mul(b),
                TypedBinaryOp::Udiv | TypedBinaryOp::Sdiv => a / b,
                TypedBinaryOp::And => a & b,
                TypedBinaryOp::Or => a | b,
                TypedBinaryOp::Xor => a ^ b,
                TypedBinaryOp::Shl => a << b,
                TypedBinaryOp::Lshr | TypedBinaryOp::Ashr => a >> b,
            })
        }

        fn not(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            value: Self::Value,
        ) -> Result<Self::Value, Self::Error> {
            Ok(!value)
        }

        fn bit_repack(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            bits: &[(Self::Value, u8)],
        ) -> Result<Self::Value, Self::Error> {
            Ok(bits
                .iter()
                .enumerate()
                .take(64)
                .fold(0u64, |value, (index, (source, bit))| {
                    value | (((source >> *bit) & 1) << index)
                }))
        }

        fn compare(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            op: TypedCompareOp,
            a: Self::Value,
            b: Self::Value,
        ) -> Result<Self::Value, Self::Error> {
            Ok((match op {
                TypedCompareOp::Eq => a == b,
                TypedCompareOp::Ne => a != b,
                TypedCompareOp::Ult => a < b,
                TypedCompareOp::Ule => a <= b,
                TypedCompareOp::Ugt => a > b,
                TypedCompareOp::Uge => a >= b,
                TypedCompareOp::Slt => (a as i64) < (b as i64),
                TypedCompareOp::Sle => (a as i64) <= (b as i64),
                TypedCompareOp::Sgt => (a as i64) > (b as i64),
                TypedCompareOp::Sge => (a as i64) >= (b as i64),
            }) as u64)
        }

        fn mux(
            &mut self,
            _types: &TypeTable,
            _ty: TypeId,
            cond: Self::Value,
            then: Self::Value,
            r#else: Self::Value,
        ) -> Result<Self::Value, Self::Error> {
            Ok(if cond == 0 { r#else } else { then })
        }
    }

    struct NoExternals;

    impl TypedExternalRegistry<U64Context> for NoExternals {
        fn oracle(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            _name: &str,
            _args: &[u64],
            _result_ty: TypeId,
            _result_index: usize,
            _occurrence: u64,
        ) -> Result<u64, Infallible> {
            unreachable!()
        }

        fn rng(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            _name: &str,
            _result_ty: TypeId,
            _result_index: usize,
            _occurrence: u64,
        ) -> Result<u64, Infallible> {
            unreachable!()
        }

        fn action_store(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            _name: &str,
            _guard: u64,
            _args: &[u64],
            _fallbacks: &[u64],
            _result_tys: &[TypeId],
            _targets: &[TypedActionTarget],
            _addresses: &[u64],
            _occurrence: u64,
        ) -> Result<(), Infallible> {
            unreachable!()
        }

        fn action(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            _name: &str,
            _args: &[u64],
            _result_ty: TypeId,
            _result_index: usize,
            _occurrence: u64,
        ) -> Result<u64, Infallible> {
            unreachable!()
        }
    }

    #[derive(Default)]
    struct RecordingExternals {
        calls: Vec<String>,
    }

    impl TypedExternalRegistry<U64Context> for RecordingExternals {
        fn oracle(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            name: &str,
            args: &[u64],
            _result_ty: TypeId,
            _result_index: usize,
            occurrence: u64,
        ) -> Result<u64, Infallible> {
            self.calls
                .push(alloc::format!("oracle:{name}:{occurrence}"));
            Ok(args[0] + 1)
        }

        fn rng(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            name: &str,
            _result_ty: TypeId,
            _result_index: usize,
            occurrence: u64,
        ) -> Result<u64, Infallible> {
            self.calls.push(alloc::format!("rng:{name}:{occurrence}"));
            Ok(0)
        }

        fn action_store(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            name: &str,
            guard: u64,
            args: &[u64],
            fallbacks: &[u64],
            _result_tys: &[TypeId],
            targets: &[TypedActionTarget],
            addresses: &[u64],
            occurrence: u64,
        ) -> Result<(), Infallible> {
            self.calls.push(alloc::format!(
                "action:{name}:{occurrence}:{guard}:{}:{}:{}:{}",
                args[0],
                fallbacks[0],
                targets[0].storage,
                addresses[0]
            ));
            Ok(())
        }

        fn action(
            &mut self,
            _context: &mut U64Context,
            _types: &TypeTable,
            name: &str,
            _args: &[u64],
            _result_ty: TypeId,
            _result_index: usize,
            occurrence: u64,
        ) -> Result<u64, Infallible> {
            self.calls
                .push(alloc::format!("action-result:{name}:{occurrence}"));
            Ok(0)
        }
    }

    #[test]
    fn typed_program_keeps_u64_operations_out_of_boolean_slots() {
        let mut types = TypeTable::new();
        let word = types.push(crate::VolarType::U64);
        let program = TypedProgram {
            types,
            ops: alloc::vec![
                TypedOp::Constant(alloc::vec![2]),
                TypedOp::Constant(alloc::vec![40]),
                TypedOp::Binary {
                    op: TypedBinaryOp::Add,
                    a: Idx(0),
                    b: Idx(1),
                },
            ],
            slot_tys: alloc::vec![word, word, word],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(2)],
            externals: Vec::new(),
        };
        let mut context = U64Context;
        assert_eq!(
            execute_typed(&mut context, &program, &[], &mut NoExternals),
            Ok(alloc::vec![42])
        );
        assert_eq!(program.prepare().unwrap().program.slot_tys[2], word);
    }

    #[test]
    fn typed_preparation_retains_transmutes_without_a_context_operation() {
        let mut types = TypeTable::new();
        let byte = types.push(crate::VolarType::U8);
        let aes = types.push(crate::VolarType::Aes8);
        let program = TypedProgram {
            types,
            ops: alloc::vec![TypedOp::Constant(alloc::vec![0x5a]), TypedOp::Copy(Idx(0)),],
            slot_tys: alloc::vec![byte, aes],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(1)],
            externals: Vec::new(),
        };
        let prepared = program.prepare().unwrap();
        let mut context = U64Context;
        assert_eq!(
            prepared.execute(&mut context, &[], &mut NoExternals),
            Ok(alloc::vec![0x5a])
        );
    }

    #[test]
    fn typed_bit_repack_keeps_layout_manipulation_out_of_boolean_programs() {
        let mut types = TypeTable::new();
        let byte = types.push(crate::VolarType::U8);
        let program = TypedProgram {
            types,
            ops: alloc::vec![
                TypedOp::Constant(alloc::vec![0b1001_0110]),
                TypedOp::BitRepack((0..8).rev().map(|bit| (Idx(0), bit)).collect(),),
            ],
            slot_tys: alloc::vec![byte, byte],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(1)],
            externals: Vec::new(),
        };
        let mut context = U64Context;
        assert_eq!(
            execute_typed(&mut context, &program, &[], &mut NoExternals),
            Ok(alloc::vec![0b0110_1001])
        );
    }

    #[test]
    fn typed_externals_and_direct_action_store_are_host_pluggable() {
        let mut types = TypeTable::new();
        let word = types.push(crate::VolarType::U64);
        let bit = types.bit();
        let program = TypedProgram {
            types,
            ops: alloc::vec![
                TypedOp::Constant(alloc::vec![7]),
                TypedOp::External(0),
                TypedOp::Constant(alloc::vec![1]),
                TypedOp::Constant(alloc::vec![0]),
                TypedOp::Constant(alloc::vec![3]),
                TypedOp::ActionStore {
                    external: 1,
                    guard: Idx(2),
                    fallbacks: alloc::vec![Idx(3)],
                    targets: alloc::vec![TypedActionTarget {
                        storage: 9,
                        address: Idx(4),
                    }],
                },
            ],
            slot_tys: alloc::vec![word, word, bit, word, word, bit],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(1)],
            externals: alloc::vec![
                TypedExternalOp {
                    kind: ExternalKind::Oracle,
                    name: "lookup".into(),
                    args: alloc::vec![Idx(0)],
                    result_tys: alloc::vec![word],
                    result_index: 0,
                    occurrence: 4,
                },
                TypedExternalOp {
                    kind: ExternalKind::Action,
                    name: "commit".into(),
                    args: alloc::vec![Idx(1)],
                    result_tys: alloc::vec![word],
                    result_index: 0,
                    occurrence: 8,
                },
            ],
        };
        let mut context = U64Context;
        let mut externals = RecordingExternals::default();
        assert_eq!(
            execute_typed(&mut context, &program, &[], &mut externals),
            Ok(alloc::vec![8])
        );
        assert_eq!(
            externals.calls,
            alloc::vec!["oracle:lookup:4", "action:commit:8:1:8:0:9:3"]
        );
    }

    #[test]
    fn boolean_program_conversion_retains_bit_types_and_execution() {
        let boolean = crate::Program {
            ops: alloc::vec![
                crate::Op::Create(true),
                crate::Op::Create(false),
                crate::Op::BitOr(Idx(0), Idx(1)),
            ],
            inputs: Vec::new(),
            outputs: alloc::vec![Idx(2)],
            externals: Vec::new(),
            storage_ops: Vec::new(),
            storage_banks: Vec::new(),
            storage_init: Vec::new(),
        };
        let typed = boolean.typed().unwrap();
        assert!(typed.slot_tys.iter().all(|ty| typed.types.is_bit(*ty)));
        let mut context = U64Context;
        assert_eq!(
            execute_typed(&mut context, &typed, &[], &mut NoExternals),
            Ok(alloc::vec![1])
        );
    }
}
