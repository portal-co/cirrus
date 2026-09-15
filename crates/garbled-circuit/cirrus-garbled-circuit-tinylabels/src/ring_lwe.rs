//! Compatibility re-export of Volar's shared TinyLabels Ring-LWE core.
//!
//! New Cirrus code must import [`volar_spec::tinylabels::ring_lwe`] directly;
//! this module remains temporarily so existing experimental callers retain a
//! stable path while the shared-core migration lands.

pub use volar_spec::tinylabels::ring_lwe::*;
