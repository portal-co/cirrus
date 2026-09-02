# Sparse banks

Dense [`StorageBank`](src/lib.rs) is `&mut [Wire]` of `2^address_bits` cells. WASM/spill lowering emits 32-bit element pointers; Boolar appends value bit-index bits as the high end of each LSB-first `addr` (32+2 = 34 wires → `2^34` cells). That does not fit.

**Before this handoff:** [`trim_storage_element_bits`](src/address_trim.rs) drops high *element* bits; [`trim_storage_addr_width`](src/address_trim.rs) then caps the concatenated addr at [`WASM_BYTE_ADDRESS_BITS`](src/address_trim.rs) (24 → `2^24` cells / 16 MiB bool image). Same aliasing as bounded WASM RAM (`WaffleImportConfig::with_memory_address_bits(24)`).

## Done

[`SparseBank`](src/sparse.rs) + [`SparseMuxTreeContext`](src/sparse.rs) are a sibling `ContextWithStorage` implementation to `MuxTreeContext`/dense `StorageBank`: cells are keyed in a `BTreeMap` and materialize only on the first known-address read or write, so a 34-bit lane costs exactly as many cells as it has touched. `known_storage_address` is checked first, unchanged — constant addresses still skip the MUX tree entirely. A symbolic (unknown-bit) read MUXes only over the bank's current live keys plus one shared lazily-created default wire, so a symbolic 24-bit read over two live cells costs a handful of gates, not `2^24`. A symbolic write does the same demux restricted to live candidates, but only when *every* concrete address consistent with the known bits is already live (e.g. `if cond { *A } else { *B }` where both were touched earlier); otherwise it fails closed with `SparseStorageError::UnboundedWrite`, since sparse storage can't safely represent "some never-before-seen cell might now hold a new value." This is the MuxTree/sparse-cell path for unauthenticated dense/sparse banks — it does not apply to (and was not needed by) native authenticated backends; see below.

This is deliberately a second, specific implementation beside the dense one, not a generic replacement — see "Not this task".

## Turned out not to need code changes here

The original handoff also listed two more items, written before `site`'s `wasm-loop` moved off the MUX-tree path entirely (commit `f27e969`, prior to this handoff landing). Re-checking them against the current code:

- ~~`storage_requirements` grows a sparse mode that reports `address_bits` without `cells = 2^address_bits`~~ — not needed. `cells_for_address_bits` is a `checked_shl`; it only errors at `address_bits >= 64`, so `storage_requirements` already reports a correct 34-bit `address_bits` today. The `cells` field is dense-specific bookkeeping (`StorageBank`'s allocation size) — any non-dense consumer (this crate's own `SparseBank`, or `site`'s `TraceStore`/`VoleProverStorage`/`VoleVerifierStorage`) simply doesn't read it.
- ~~Site `wasm-loop` drops the trim once `wasm_looped_circuit_cirrus_prove_verify` passes on the untrimmed 34-bit guest~~ — also not blocked on anything here. `VoleProverStorage`/`VoleVerifierStorage` are witness-trace accumulators with no `2^address_bits` allocation; in `site/crates/proofs/src/wasm_loop.rs`, `StorageRequirement::cells` is read in exactly one place, a test assertion, never for allocation. Dropping `trim_storage_element_bits`/`trim_storage_addr_width` in `wasm_path_to_fused_circuit` and widening that test is a `site`-repo-only change, unrelated to this crate.

## Not this task

Identity `WebProofBackend::verify`. LLVM frontend. Changing Boolar's bit-index-as-high-bits layout. Making `MuxTreeContext` itself generic over dense/sparse storage (would need a second type parameter whose default references the first — not worth it for one alternate backing; `SparseMuxTreeContext` is a separate type instead). Symbolic addressing for `cirrus-volar-garble` (its `VolarGarbleBackend`/`VolarEvalBackend` index storage concretely today, so they aren't a MUX-tree consumer yet).

## Where

`StorageBank`, `storage_requirements`, `MuxTreeContext` `storage_read` / `storage_write` (dense); `SparseBank`, `SparseMuxTreeContext` (sparse, `src/sparse.rs`); `execute`. `execute_chunked` chunks *statements* via lazy-repo; it is not sparse cells.
