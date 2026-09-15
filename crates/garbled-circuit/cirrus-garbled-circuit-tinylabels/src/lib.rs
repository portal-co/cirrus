#![no_std]
#![warn(missing_docs)]

//! Cirrus adapter for the shared TinyLabels construction.
//!
//! The construction-neutral implementation lives in
//! [`volar_spec::tinylabels`]. Cirrus deliberately re-exports it instead of
//! maintaining a Ring-LWE fork: its own future responsibility is to bind that
//! shared core to fixed interpreter input manifests and bounded coroutine
//! framing. It does not change Cirrus's default direct-label delivery, table
//! stream formats, label width, or embedded admission policy.
//!
//! See [`../TINYLABELS_CROSS_REPO_PLAN.md`](../TINYLABELS_CROSS_REPO_PLAN.md)
//! for the ownership split and deployment gates.

pub use volar_spec::tinylabels::*;
