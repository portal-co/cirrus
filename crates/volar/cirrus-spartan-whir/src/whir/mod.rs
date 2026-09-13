//! Plain (no-ZK) WHIR polynomial commitment scheme.
//!
//! This module ports the pinned upstream `p3-whir` prover/verifier and the
//! `spartan-whir` `Plonky3WhirPcs` adapter to `no_std`: protocol-parameter
//! derivation ([`params`]), the Fiat-Shamir domain separator
//! ([`domain_separator`]), the internal quadratic sumcheck ([`sumcheck`]),
//! the commitment/opening protocol ([`pcs`]), and the Spartan DirectSparse
//! boundary ([`spartan`]).

pub mod domain_separator;
pub mod params;
pub mod pcs;
pub mod spartan;
pub mod sumcheck;
