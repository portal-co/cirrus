#![no_std]
#![warn(missing_docs)]

//! An R1CS-synthesis [`cirrus_core::Context`] backend: replays a recorded
//! `cirrus_recompile_core::Program`/`PreparedProgram` (or any other
//! frontend built on the same `Context` trait bundle) as `ark-r1cs-std`
//! [`Boolean<F>`] gadgets against an `ark-relations` constraint system,
//! exactly like this workspace's plaintext, garbled-circuit, or recording
//! backends replay the same trace against their own wire representation.
//!
//! [`R1csBackend`] implements only the five traits
//! `cirrus_recompile_rt::execute`/`execute_prepared` require
//! (`ContextWithCreate`/`BitAnd`/`BitOr`/`BitXor`/`Mux<bool>`); it does not
//! interpret ERT or LLVM IR itself. [`circuit::ProgramCircuit`] is the
//! proof-system-agnostic glue that replays a `Program` through a fresh
//! `R1csBackend` for both a SNARK's setup pass (shape only) and its proving
//! pass (with a real witness) -- see `cirrus-groth16` for a concrete
//! Groth16 instantiation built on top of it.
//!
//! # Witness-input policy
//!
//! `Op::Create(bool)` only ever represents a compile-time-known constant
//! (`create` below maps it straight to [`Boolean::constant`], never a
//! witness allocation). Symbolic/witness data instead flows in through
//! `program.inputs`, which a caller allocates as `Boolean::new_witness` (or
//! `new_input`, for a value that should be public) *before* calling
//! `execute`/`execute_prepared` -- the same convention every other backend
//! in this workspace uses for its own encoded inputs.

extern crate alloc;

use ark_ff::PrimeField;
use ark_r1cs_std::prelude::*;
use ark_relations::gr1cs::{ConstraintSystemRef, SynthesisError};
use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithMux,
    ContextWithStorage, ContextWithValue, HasError, StorageAddressBit,
};

pub mod circuit;

/// R1CS-synthesis `Context`: every Boolean operation is replayed as an
/// `ark-r1cs-std` [`Boolean<F>`] gadget against [`Self::cs`].
pub struct R1csBackend<F: PrimeField> {
    /// The constraint system every operation allocates into / enforces
    /// constraints against.
    pub cs: ConstraintSystemRef<F>,
}

impl<F: PrimeField> R1csBackend<F> {
    /// Wrap a constraint system.
    pub fn new(cs: ConstraintSystemRef<F>) -> Self {
        Self { cs }
    }
}

impl<F: PrimeField> HasError for R1csBackend<F> {
    type Error = SynthesisError;
}

impl<F: PrimeField> ContextWithValue<bool> for R1csBackend<F> {
    type Wrapped = Boolean<F>;
}

impl<F: PrimeField> ContextWithCreate<bool> for R1csBackend<F> {
    fn create(&mut self, val: bool) -> Result<Boolean<F>, SynthesisError> {
        Ok(Boolean::constant(val))
    }
}

impl<F: PrimeField> ContextWithBitAnd<bool> for R1csBackend<F> {
    fn bitand(&mut self, a: Boolean<F>, b: Boolean<F>) -> Result<Boolean<F>, SynthesisError> {
        Ok(&a & &b)
    }

    fn bitand_assign(&mut self, a: &mut Boolean<F>, b: Boolean<F>) -> Result<(), SynthesisError> {
        *a = self.bitand(a.clone(), b)?;
        Ok(())
    }
}

impl<F: PrimeField> ContextWithBitOr<bool> for R1csBackend<F> {
    fn bitor(&mut self, a: Boolean<F>, b: Boolean<F>) -> Result<Boolean<F>, SynthesisError> {
        Ok(&a | &b)
    }

    fn bitor_assign(&mut self, a: &mut Boolean<F>, b: Boolean<F>) -> Result<(), SynthesisError> {
        *a = self.bitor(a.clone(), b)?;
        Ok(())
    }
}

impl<F: PrimeField> ContextWithBitXor<bool> for R1csBackend<F> {
    fn bitxor(&mut self, a: Boolean<F>, b: Boolean<F>) -> Result<Boolean<F>, SynthesisError> {
        Ok(&a ^ &b)
    }

    fn bitxor_assign(&mut self, a: &mut Boolean<F>, b: Boolean<F>) -> Result<(), SynthesisError> {
        *a = self.bitxor(a.clone(), b)?;
        Ok(())
    }
}

impl<F: PrimeField> ContextWithMux<bool> for R1csBackend<F> {
    fn mux(
        &mut self,
        cond: Boolean<F>,
        then: Boolean<F>,
        r#else: Boolean<F>,
    ) -> Result<Boolean<F>, SynthesisError> {
        cond.select(&then, &r#else)
    }
}

/// Dense symbolic storage for the Boolar R1CS host. The caller chooses the
/// lane capacity; reads and writes are constrained with a one-hot MUX/demux
/// tree and therefore work even for secret addresses.
impl<F: PrimeField> ContextWithStorage<bool> for R1csBackend<F> {
    type Storage = [Boolean<F>];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Boolean<F>>],
    ) -> Result<Boolean<F>, Self::Error> {
        let mut result = Boolean::constant(false);
        for (index, cell) in storage.iter().enumerate() {
            let selector = storage_selector(address, index)?;
            result = &result ^ &(&selector & cell);
        }
        Ok(result)
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<Boolean<F>>],
        value: Boolean<F>,
    ) -> Result<(), Self::Error> {
        for (index, cell) in storage.iter_mut().enumerate() {
            let selector = storage_selector(address, index)?;
            *cell = selector.select(&value, cell)?;
        }
        Ok(())
    }
}

fn storage_selector<F: PrimeField>(
    address: &[StorageAddressBit<Boolean<F>>],
    index: usize,
) -> Result<Boolean<F>, SynthesisError> {
    let mut selector = Boolean::constant(true);
    for (bit, address_bit) in address.iter().enumerate() {
        let expected = (index >> bit) & 1 != 0;
        let actual = if expected {
            address_bit.wire.clone()
        } else {
            !address_bit.wire.clone()
        };
        selector = &selector & &actual;
    }
    Ok(selector)
}
