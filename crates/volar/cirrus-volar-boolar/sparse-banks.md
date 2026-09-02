# Sparse banks

Dense [`StorageBank`](src/lib.rs) is `&mut [Wire]` of `2^address_bits` cells. WASM/spill lowering emits 32-bit element pointers; Boolar appends value bit-index bits as the high end of each LSB-first `addr` (32+2 = 34 wires → `2^34` cells). That does not fit.

**Today:** [`trim_storage_element_bits`](src/address_trim.rs) drops high *element* bits; [`trim_storage_addr_width`](src/address_trim.rs) then caps the concatenated addr at [`WASM_BYTE_ADDRESS_BITS`](src/address_trim.rs) (24 → `2^24` cells / 16 MiB bool image). Same aliasing as bounded WASM RAM (`WaffleImportConfig::with_memory_address_bits(24)`).

**This handoff:** sparse cell backing so a 34-bit lane executes without trim.

## Do

1. Add a `Storage` (or `SparseBank`) that materializes a cell on first read/write, keyed by the integer address. **Done when** a 34-bit lane with two live cells allocates two cells, not `2^34`.
2. Keep the `known_storage_address` fast path. **Done when** constant addresses still skip the MUX tree.
3. Symbolic addresses MUX over the *live* set, or fail closed with a named error if that set is unbounded. **Done when** a symbolic 24-bit (`2^24` cell) read over two live cells does not build a 16-million-way tree. Site `wasm-loop` now interprets the trimmed 24-bit guest through native VOLE storage (`VoleProverStorage` / `VoleVerifierStorage`) rather than a MUX tree; this handoff is the MuxTree/sparse-cell path for unauthenticated dense/sparse banks.
4. `storage_requirements` grows a sparse mode that reports `address_bits` without `cells = 2^address_bits`. **Done when** `execute` accepts an untrimmed fused WASM guest.
5. Site `wasm-loop` (`site/crates/proofs/src/wasm_loop.rs`) drops the trim once `wasm_looped_circuit_cirrus_prove_verify` passes on the untrimmed 34-bit guest. **Done when** that test no longer calls `trim_storage_element_bits`.

## Not this task

Identity `WebProofBackend::verify`. LLVM frontend. Changing Boolar’s bit-index-as-high-bits layout.

## Where

`StorageBank`, `storage_requirements`, `MuxTreeContext` `storage_read` / `storage_write`, `execute`. `execute_chunked` chunks *statements* via lazy-repo; it is not sparse cells.
