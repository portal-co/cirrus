# ERT vs WASM/LLVM frontends

Cirrus can ingest a guest three ways. Memory cost, not syntax, decides which
foundation to keep.

## Event-as-ERT-call (foundation)

Compile a well-behaved RV32 (or Thumb) function once. For each event:

1. Write the event into the caller-owned **virtual stack** (Boolean storage
   bits `ert_func` already uses for ABI words past `a7`, byte-addressed by
   `sp`).
2. Run `ert_func` **once** on that image.
3. Read the output from a result register (`a0`/`a1` on RV32).
4. Repeat. The instruction image and stack capacity stay fixed; only the
   stacked event words change.

Host and embedded share this loop. `RawMemory::from_slice` is the bounded
host mapping; QEMU is a compatibility gate for the same firmware.

Cost is explicit: symbolic stack bits + 32×32 register wires + the ELF text
you mapped. Site `site-proofs-guest` sizes that stack in kilobits, not
megabytes.

Constraints: concrete control flow and stack-form addresses; no compressed
instructions; supported RV32/Thumb subset only. Less expressive than a
WASM linear-memory guest.

## WASM / LLVM → VAFFLE → Volar IR → Boolar → Cirrus

```
WASM or LLVM
  → VAFFLE (volar-ir / waffle)
  → Volar IR (SSA)
  → movfuscate or unroll
  → Boolar IR
  → serialize / fuse BCircuit
  → Cirrus execute (VOLE storage, Recorder, …)
```

`volar-ir-build` is the WASM builder (`from_wasm_with_config` → inline →
`lower_to_volar_ir` → `movfuscate` → `lower_to_boolar` → `fuse`).
`cirrus-llvm-frontend` records one LLVM function with concrete control flow
and concrete addresses into a `Program`. Site `wasm-loop` is the WASM slice
already wired to native VOLE storage.

WASM is the more expressive guest (data-dependent CFG via movfuscate, linear
memory). LLVM 22 + `cirrus-llvm-pass` / `volar-llvm-import-core` is the
annotated-kernel path (addresses and branches must stay concrete).

## Memory: why this repo did not switch the foundation

| Path | What blows up | Typical floor |
|------|----------------|---------------|
| ERT virtual stack | Caller-chosen bit capacity | Tens of KiB for a tiny `on_event` |
| WASM → Boolar | 32-bit pointers; Boolar appends bit-index high bits (e.g. 34) | Dense bank `2^n` cells; 24-bit trim is still a **16 MiB** image per lane if muxed. Native VOLE storage avoids the mux, not the 32-bit pointer semantics. |
| LLVM frontend | Concrete alloca/regions named in the request | Live cells only, but needs LLVM 22 and rejects symbolic addresses/branches |

A tiny constant WASM `pick` already runs under `wasm-loop` with VOLE
storage. That does **not** make WASM the event-loop foundation: the next
real handler that spills a `i32` pointer reintroduces the 32/34-bit bank.
LLVM would have an acceptable *cell* cost for this XOR kernel, and is not
acceptable as a default site dependency (LLVM 22 toolchain, concrete-CF
contract).

Keep ERT as the event loop. Reach for WASM/LLVM when the guest needs
linear memory or LLVM-shaped kernels **and** the address/image budget is
named and paid (trim, sparse banks, or proven-concrete pointers).
