//! Independent `no_std` Spartan-WHIR implementation for Cirrus trace proofs.
//!
//! This crate is intentionally `no_std + alloc`. It does not reuse Plonky3's
//! field implementation; field arithmetic below is implemented directly over
//! canonical integers and is differentially tested against the pinned upstream
//! Spartan-WHIR/Plonky3 stack as an oracle.
//!
//! Slices 0--6 exist at this point: crate policy, KoalaBear and
//! quintic-extension arithmetic, polynomial/R1CS substrate, the Poseidon
//! hash/Merkle/transcript profile, Spartan sumcheck with the DirectSparse
//! R1CS evaluation reduction, the plain (no-ZK) WHIR polynomial
//! commitment with its DirectSparse adapter, and the setup/prove/verify key
//! API with composed security budgeting and a controlled post-commitment
//! challenge-slot schedule. Serialization and fixtures are a later slice
//! and are not claimed here.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

mod algebra;
mod dft;
mod error;
mod extension;
mod field;
mod hash;
mod keys;
mod merkle;
mod poly;
mod poseidon2;
mod poseidon2_constants;
mod r1cs;
pub mod security;
mod spartan;
mod sumcheck;
mod transcript;
pub mod whir;

pub use algebra::FieldElement;
pub use dft::{
    DftError, KOALABEAR_TWO_ADICITY, dft_base, dft_batch_base, dft_batch_ext, two_adic_generator,
};
pub use error::SpartanError;
pub use extension::{QUINTIC_DEGREE, QuinticExtension, QuinticExtensionError};
pub use field::{KOALABEAR_MODULUS, KoalaBear, KoalaBearError};
pub use hash::{
    POSEIDON_CHALLENGER_RATE, POSEIDON_DIGEST_ELEMENTS, POSEIDON_FIELD_HASH_RATE, PoseidonDigest,
    poseidon_compress2, poseidon_hash_fixed,
};
pub use keys::{
    ChallengeSchedule, NoChallengeSchedule, SPARTAN_NO_ZK_PROTOCOL_ID, SPARTAN_PROOF_VERSION,
    SpartanKeyError, SpartanProof, SpartanProvingKey, SpartanVerifyingKey, setup,
    spartan_domain_separator,
};
pub use merkle::{MerkleError, PoseidonMerklePath, PoseidonMerkleTree, PrunedMerklePaths};
pub use poly::{
    CubicRoundPoly, EqPolynomial, MultilinearPoint, PolyError, QuadraticRoundPoly,
    evaluate_mle_table,
};
pub use poseidon2::{
    POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS_16, POSEIDON2_PARTIAL_ROUNDS_24,
    POSEIDON2_WIDTH_16, POSEIDON2_WIDTH_24, Poseidon2KoalaBear16, Poseidon2KoalaBear24,
};
pub use r1cs::{R1csError, R1csShape, R1csWitness, SparseMatEntry, SparseMatrix};
pub use security::{
    ComponentSecurity, ComposedSecurityBudget, SecurityBoundComponent, SecurityBudgetError,
    derive_direct_component_security, spartan_algebraic_error_terms_no_zk,
};
pub use spartan::{
    ChallengeSlotSchedule, DirectSparseError, DirectSparsePcs, DirectSparseProof, R1csInstance,
    bind_row_vars_joint, build_z_full, eq_point_eval, evaluate_public_half, evaluate_with_tables,
    matrix_z_slice, observe_context, prove_direct_sparse, prove_direct_sparse_with_schedule,
    recover_witness_eval, verify_direct_sparse, verify_direct_sparse_with_schedule,
};
pub use sumcheck::{
    InnerSumcheckProof, OuterSumcheckProof, prove_inner, prove_outer, verify_inner, verify_outer,
};
pub use transcript::{PoseidonTranscript, QUINTIC_TRANSCRIPT_SAMPLES, TranscriptError};
