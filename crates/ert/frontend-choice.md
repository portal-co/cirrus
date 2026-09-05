# Cirrus guest ingest paths

Cirrus does not pick a frontend. Each path below is a supported ingest;
integrators choose per platform and guest shape.

## ERT (RV32 / Thumb)

Compile a well-behaved RV32 or Thumb function. `ert_func` maps the image,
runs once, and returns result registers. Extra ABI words live on the
caller-owned **virtual stack** (byte-addressed by `sp`). Host and embedded
share this interpreter: `RawMemory::from_slice` on desktop/server, QEMU as a
compatibility gate, on-device firmware the same subset.

Cost is explicit: stack bits + 32×32 register wires + mapped text.
Constraints: concrete control flow and stack-form addresses; no compressed
instructions; documented opcode subset only. RV32 `slt`/`slti` forms may
materialize symbolic Boolean data. Thumb additionally supports symbolic
NZCVQ APSR transfers, carry-consuming `ADC`/`SBC`, and a one-instruction IT
value materializer; symbolic branches, general predication, calls, returns,
and memory effects remain rejected. Less expressive than a WASM linear-memory
guest. Fits microcontrollers; SHA-256 compression is the locked garbling
workload.

## WASM → VAFFLE → Volar IR → Boolar → Cirrus

```
WASM
  → VAFFLE (volar-ir / waffle)
  → Volar IR (SSA)
  → movfuscate or unroll
  → Boolar IR
  → serialize / fuse BCircuit
  → Cirrus execute (VOLE storage, Recorder, …)
```

`volar-ir-build` is `from_wasm_with_config` → inline → `lower_to_volar_ir` →
`movfuscate` → `lower_to_boolar` → `fuse`. Data-dependent CFG becomes a
self-loop; 32-bit pointers keep a linear-memory image (dense banks are
`2^n` cells; a 24-bit trim is still 16 MiB per lane if muxed). Native VOLE
storage avoids the mux, not the pointer width. Browser-native WASM is a
different venue: a 32-bit linear memory can be up to 4 GiB without becoming
a Boolar bank.

## LLVM → `cirrus-llvm-frontend` → `Program`

LLVM 22 + `cirrus-llvm-pass` / `volar-llvm-import-core`. One function,
concrete control flow, concrete addresses; the request names every symbolic
byte. Live cells only. Rejects symbolic branches. Same Boolean `Program` as
a recorded ERT trace. Needs an LLVM 22 toolchain (`CIRRUS_LLVM_CONFIG` or
`LLVM_SYS_221_PREFIX`). Concrete-count `alloca` already works here
(`max_alloca_bytes`).

## LLVM → VAFFLE → Volar IR → Boolar → Cirrus

```
LLVM IR
  → VAFFLE (volar-llvm-vaffle-import / Pipeline::from_llvm)
  → Volar IR (SSA)
  → movfuscate or unroll
  → Boolar IR
  → serialize / fuse BCircuit
  → Cirrus execute
```

Same shape strategy as WASM: calls preserved, then unroll (concrete CF) or
movfuscate (symbolic CF). Distinct from LLVM-direct above.
`volar-ir-build` feature `llvm`. Constant-size scalar `alloca` and `switch`
import are landed ([`llvm-alloca.md`](../../../volar-ir/docs/llvm-alloca.md));
entry-param unpacking + Boolar fuse too
([`llvm-stack-spill-boolar.md`](../../../volar-ir/docs/llvm-stack-spill-boolar.md)).
Array/struct `alloca` is the rustc `-O0` / SLH-DSA ingest blocker
([`llvm-array-alloca.md`](../../../volar-ir/docs/llvm-array-alloca.md)).
Fused STACK addresses are 64-bit; `storage_requirements` reports `cells = 0`
(dense `2^64` does not fit). Needs the same LLVM 22 toolchain.
