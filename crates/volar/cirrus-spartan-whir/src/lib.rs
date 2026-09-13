//! Independent `no_std` Spartan-WHIR implementation for Cirrus trace proofs.
//!
//! This crate is intentionally `no_std + alloc`. It does not reuse Plonky3's
//! field implementation; field arithmetic below is implemented directly over
//! canonical integers and is differentially tested against the pinned upstream
//! Spartan-WHIR/Plonky3 stack as an oracle.
//!
//! Slices 0--4 exist at this point: crate policy, KoalaBear and
//! quintic-extension arithmetic, polynomial/R1CS substrate, the Poseidon
//! hash/Merkle/transcript profile, and Spartan sumcheck with the DirectSparse
//! R1CS evaluation reduction. The WHIR polynomial commitment and the
//! setup/prove/verify key API are later slices and are not claimed here.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

mod algebra;
mod error;
mod extension;
mod field;
mod hash;
mod merkle;
mod poly;
mod poseidon2;
mod poseidon2_constants;
mod r1cs;
mod spartan;
mod sumcheck;
mod transcript;

pub use algebra::FieldElement;
pub use error::SpartanError;
pub use extension::{QUINTIC_DEGREE, QuinticExtension, QuinticExtensionError};
pub use field::{KOALABEAR_MODULUS, KoalaBear, KoalaBearError};
pub use hash::{
    POSEIDON_CHALLENGER_RATE, POSEIDON_DIGEST_ELEMENTS, POSEIDON_FIELD_HASH_RATE, PoseidonDigest,
    poseidon_compress2, poseidon_hash_fixed,
};
pub use merkle::{MerkleError, PoseidonMerklePath, PoseidonMerkleTree};
pub use poly::{
    CubicRoundPoly, EqPolynomial, MultilinearPoint, PolyError, QuadraticRoundPoly,
    evaluate_mle_table,
};
pub use poseidon2::{
    POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS_16, POSEIDON2_PARTIAL_ROUNDS_24,
    POSEIDON2_WIDTH_16, POSEIDON2_WIDTH_24, Poseidon2KoalaBear16, Poseidon2KoalaBear24,
};
pub use r1cs::{R1csError, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix};
pub use spartan::{
    DirectSparseError, DirectSparsePcs, DirectSparseProof, R1csInstance, bind_row_vars_joint,
    build_z_full, eq_point_eval, evaluate_public_half, evaluate_with_tables, matrix_z_slice,
    observe_context, prove_direct_sparse, recover_witness_eval, verify_direct_sparse,
};
pub use sumcheck::{
    InnerSumcheckProof, OuterSumcheckProof, prove_inner, prove_outer, verify_inner, verify_outer,
};
pub use transcript::{PoseidonTranscript, QUINTIC_TRANSCRIPT_SAMPLES, TranscriptError};
