//! Independent `no_std` Spartan-WHIR implementation for Cirrus trace proofs.
//!
//! This crate is intentionally `no_std + alloc`. It does not reuse Plonky3's
//! field implementation; field arithmetic below is implemented directly over
//! canonical integers and is differentially tested against the pinned upstream
//! Spartan-WHIR/Plonky3 stack as an oracle.
//!
//! Only Slice 0/1 exists at this point: crate policy plus KoalaBear and
//! quintic-extension arithmetic. Sumcheck, transcripts, PCS, and proving APIs
//! are later slices and are not claimed here.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

mod extension;
mod field;

pub use extension::{QUINTIC_DEGREE, QuinticExtension, QuinticExtensionError};
pub use field::{KOALABEAR_MODULUS, KoalaBear, KoalaBearError};
