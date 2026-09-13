//! Independent `no_std` Spartan-WHIR implementation for Cirrus trace proofs.
//!
//! This crate is intentionally `no_std + alloc`. It does not reuse Plonky3's
//! field implementation; field arithmetic below is implemented directly over
//! canonical integers and is differentially tested against the pinned upstream
//! Spartan-WHIR/Plonky3 stack as an oracle.
//!
//! Slices 0--2 exist at this point: crate policy, KoalaBear and
//! quintic-extension arithmetic, and the polynomial/R1CS substrate. Hash,
//! transcript, sumcheck, PCS, and proving APIs are later slices and are not
//! claimed here.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

mod algebra;
mod extension;
mod field;
mod poly;
mod r1cs;

pub use algebra::FieldElement;
pub use extension::{QUINTIC_DEGREE, QuinticExtension, QuinticExtensionError};
pub use field::{KOALABEAR_MODULUS, KoalaBear, KoalaBearError};
pub use poly::{EqPolynomial, MultilinearPoint, PolyError, evaluate_mle_table};
pub use r1cs::{R1csError, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix};
