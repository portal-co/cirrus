//! Cirrus-owned mirror of the GF(2) Volar type language.
//!
//! This module deliberately does not depend on `volar-ir-common`: a recorded
//! Cirrus program is a portable backend artifact and must remain usable by
//! hosts which only link Cirrus.  The Volar adapter converts between the two
//! equivalent type tables at its boundary.

use alloc::vec::Vec;

/// Index into a [`TypeTable`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TypeId(pub u32);

impl TypeId {
    /// Return this ID as a slice index.
    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

/// The primitive and aggregate types accepted by typed Cirrus programs.
///
/// Every primitive has a GF(2) bit layout. `_128` and `_256` are packed
/// little-endian integers; vectors are lane-wise values; tuples are products
/// in field declaration order.  Z3 intentionally has no representation
/// because it cannot be lowered through the GF(2) backends.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum VolarType {
    /// One Boolean bit.
    Bit,
    /// Packed eight-bit integer.
    U8,
    /// Packed sixteen-bit integer.
    U16,
    /// Packed thirty-two-bit integer.
    U32,
    /// Packed sixty-four-bit integer.
    U64,
    /// Packed one-hundred-twenty-eight-bit integer.
    U128,
    /// Packed two-hundred-fifty-six-bit integer.
    U256,
    /// Eight-bit AES field element.
    Aes8,
    /// Sixty-four-bit Galois field element.
    Galois64,
    /// Fixed-size homogeneous lane-wise vector.
    Vector {
        /// Number of lanes; must be nonzero.
        lanes: usize,
        /// Element type table entry.
        element: TypeId,
    },
    /// Heterogeneous product in declaration order.
    Tuple(Vec<TypeId>),
}

/// A validated fixed-layout view of one [`VolarType`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypeLayout {
    /// Number of bits in the value's canonical little-endian flattening.
    pub bits: usize,
    /// Number of `u64` words in that flattening, rounded up.
    pub words: usize,
}

/// Why a [`TypeTable`] is not a valid GF(2) Volar type table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypeError {
    /// A type referenced an ID outside this table.
    InvalidReference(TypeId),
    /// A vector specified zero lanes.
    EmptyVector,
    /// A type directly or indirectly contained itself.
    RecursiveType(TypeId),
    /// Computing a flattened bit layout overflowed `usize`.
    LayoutOverflow(TypeId),
}

/// A portable interning table for [`VolarType`].
///
/// `Default` always contains [`VolarType::Bit`] at [`Self::bit`], which keeps
/// existing Boolean `Program`s type-compatible even if they predate explicit
/// per-slot metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeTable {
    types: Vec<VolarType>,
}

impl Default for TypeTable {
    fn default() -> Self {
        Self {
            types: alloc::vec![VolarType::Bit],
        }
    }
}

impl TypeTable {
    /// Construct a table containing only the canonical bit type.
    pub fn new() -> Self {
        Self::default()
    }

    /// Canonical [`VolarType::Bit`] entry.
    pub const fn bit(&self) -> TypeId {
        TypeId(0)
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether the table has no registered entries.
    ///
    /// This is always false for a table constructed through [`new`](Self::new).
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Look up one registered type.
    pub fn get(&self, id: TypeId) -> Option<&VolarType> {
        self.types.get(id.get())
    }

    /// Append a type without deduplicating it and return its stable ID.
    pub fn push(&mut self, ty: VolarType) -> TypeId {
        let id = TypeId(self.types.len() as u32);
        self.types.push(ty);
        id
    }

    /// Intern `ty`, returning an existing equal entry when present.
    pub fn intern(&mut self, ty: VolarType) -> TypeId {
        self.types
            .iter()
            .position(|existing| existing == &ty)
            .map(|index| TypeId(index as u32))
            .unwrap_or_else(|| self.push(ty))
    }

    /// Validate references, vector dimensions, acyclicity, and every
    /// flattened layout.
    pub fn validate(&self) -> Result<(), TypeError> {
        let mut state = alloc::vec![Visit::Unseen; self.types.len()];
        for index in 0..self.types.len() {
            self.layout_visit(TypeId(index as u32), &mut state)?;
        }
        Ok(())
    }

    /// Get the canonical flattened GF(2) layout of a type.
    pub fn layout(&self, id: TypeId) -> Result<TypeLayout, TypeError> {
        let mut state = alloc::vec![Visit::Unseen; self.types.len()];
        self.layout_visit(id, &mut state)
    }

    /// Whether `id` is the canonical Boolean type.
    pub fn is_bit(&self, id: TypeId) -> bool {
        matches!(self.get(id), Some(VolarType::Bit))
    }

    fn layout_visit(&self, id: TypeId, state: &mut [Visit]) -> Result<TypeLayout, TypeError> {
        let index = id.get();
        let Some(entry) = state.get(index).copied() else {
            return Err(TypeError::InvalidReference(id));
        };
        match entry {
            Visit::Done(layout) => return Ok(layout),
            Visit::Visiting => return Err(TypeError::RecursiveType(id)),
            Visit::Unseen => {}
        }
        state[index] = Visit::Visiting;
        let ty = self.get(id).ok_or(TypeError::InvalidReference(id))?;
        let bits = match ty {
            VolarType::Bit => 1,
            VolarType::U8 | VolarType::Aes8 => 8,
            VolarType::U16 => 16,
            VolarType::U32 => 32,
            VolarType::U64 | VolarType::Galois64 => 64,
            VolarType::U128 => 128,
            VolarType::U256 => 256,
            VolarType::Vector { lanes, element } => {
                if *lanes == 0 {
                    return Err(TypeError::EmptyVector);
                }
                self.layout_visit(*element, state)?
                    .bits
                    .checked_mul(*lanes)
                    .ok_or(TypeError::LayoutOverflow(id))?
            }
            VolarType::Tuple(fields) => {
                let mut bits = 0usize;
                for field in fields {
                    bits = bits
                        .checked_add(self.layout_visit(*field, state)?.bits)
                        .ok_or(TypeError::LayoutOverflow(id))?;
                }
                bits
            }
        };
        let words = bits.checked_add(63).ok_or(TypeError::LayoutOverflow(id))? / 64;
        let layout = TypeLayout { bits, words };
        state[index] = Visit::Done(layout);
        Ok(layout)
    }
}

#[derive(Clone, Copy)]
enum Visit {
    Unseen,
    Visiting,
    Done(TypeLayout),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_layouts_preserve_packed_and_lane_semantics() {
        let mut types = TypeTable::new();
        let wide = types.push(VolarType::U256);
        let word = types.push(VolarType::U32);
        let lanes = types.push(VolarType::Vector {
            lanes: 4,
            element: word,
        });
        let pair = types.push(VolarType::Tuple(alloc::vec![wide, lanes]));

        assert_eq!(
            types.layout(wide),
            Ok(TypeLayout {
                bits: 256,
                words: 4
            })
        );
        assert_eq!(
            types.layout(lanes),
            Ok(TypeLayout {
                bits: 128,
                words: 2
            })
        );
        assert_eq!(
            types.layout(pair),
            Ok(TypeLayout {
                bits: 384,
                words: 6
            })
        );
        assert_eq!(types.validate(), Ok(()));
    }

    #[test]
    fn invalid_vector_and_reference_are_rejected() {
        let mut types = TypeTable::new();
        let missing = types.push(VolarType::Vector {
            lanes: 1,
            element: TypeId(99),
        });
        assert_eq!(
            types.layout(missing),
            Err(TypeError::InvalidReference(TypeId(99)))
        );

        let empty = types.push(VolarType::Vector {
            lanes: 0,
            element: types.bit(),
        });
        assert_eq!(types.layout(empty), Err(TypeError::EmptyVector));
    }
}
