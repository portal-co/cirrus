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
Use the `volar-ir-build` `llvm` feature. Scalar and array-of-integer `alloca`,
typed-view GEPs, `switch`, and normal calls now preserve correct stack/frame
layout ([`llvm-alloca.md`](../../../volar-ir/docs/llvm-alloca.md),
[`llvm-array-alloca.md`](../../../volar-ir/docs/llvm-array-alloca.md),
[`llvm-cross-function-calls.md`](../../../volar-ir/docs/llvm-cross-function-calls.md)).
Symbolic single-index stack/global GEP, pointer-identity dispatch, and
nonvolatile symbolic `memcpy`/`memset` lower to circuits for movfuscation
([`llvm-dynamic-stack-gep.md`](../../../volar-ir/docs/llvm-dynamic-stack-gep.md),
[`llvm-dynamic-global-gep.md`](../../../volar-ir/docs/llvm-dynamic-global-gep.md),
[`llvm-ptr-runtime-dispatch.md`](../../../volar-ir/docs/llvm-ptr-runtime-dispatch.md),
[`llvm-memset-symbolic.md`](../../../volar-ir/docs/llvm-memset-symbolic.md)).
The lowering also carries the newer polynomial and fusion optimizations through
Boolar ([`llvm-boolar-poly.md`](../../../volar-ir/docs/llvm-boolar-poly.md),
[`llvm-fuse-unroll.md`](../../../volar-ir/docs/llvm-fuse-unroll.md)). Actual
limits remain `invoke`/unwind, structs and general aggregate values,
multi-index GEPs, symbolic `memmove`, heap allocation, and deliberately
unbounded expansion. Dead `landingpad` blocks are skipped; reachable exception
handling is not ([`llvm-landingpad.md`](../../../volar-ir/docs/llvm-landingpad.md)).
Fused STACK addresses are 64-bit; `storage_requirements` reports `cells = 0`
(dense `2^64` does not fit). Needs the same LLVM 22 toolchain.
