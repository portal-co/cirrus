# cirrus

New circuit compiler, for everywhere

## Symbolic execution runtimes

`cirrus` includes two small, circuit-oriented symbolic interpreters that share
their Boolean-word circuits, symbolic-stack model, concrete-value tracking,
raw-memory mapping, and host-call convention through the internal
`cirrus-ert-core` crate.

- [`cirrus-ert`](crates/ert/cirrus-ert/src/lib.rs) is the compatible RV32 RISC-V
  facade.
- [`cirrus-armv8m-ert`](crates/ert/cirrus-armv8m-ert/src/lib.rs) is the non-secure,
  Thumb-only Armv8-M Mainline/Cortex-M33 facade. Its ABI wrapper follows
  AAPCS32; it accepts an odd Thumb entry address and sixteen core registers.

Both interpret a deliberately well-behaved compiler-oriented subset, not an
entire machine: control flow, flags, and non-stack addresses must stay
concrete; stacks and return stacks are caller supplied; unsupported encodings
and violations report `Unexpected`. ARM additionally excludes exception and
interrupt entry, TrustZone transitions, MPU state, floating point, DSP/MVE,
atomics, and semihosting.

Each facade has a bare-metal QEMU SHA-256 compression compatibility gate. The
same no-std workload is compiled for RV32IM and `thumbv8m.main-none-eabi`; the
ARM image uses the `mps2-an505` Cortex-M33 board, with its workload linked at
`0x2000_0000` to exercise unbounded raw guest addresses. These are bare-metal
tests, not Linux VMs. QEMU's AN505 model boots from its remapped flash vector
at `0x1000_0000`, so the fixture places only its vector/reset stub there while
keeping the interpreted code and constants at the high RAM address.

## Streaming garbled circuits

[`cirrus-garbled-circuit`](crates/garbled-circuit/cirrus-garbled-circuit/src/lib.rs) is the
stable four-row-table baseline backend. It emits every non-free table to a
caller-supplied `Pusher` in circuit order; it deliberately does not own a
circuit buffer, network driver, allocator, or async runtime. An embedded
integrator must stream or durably hand off each table before accepting the
next—typically through a coroutine/event-loop adapter with a small bounded
frame buffer. A full table vector is a host-test convenience, never a viable
microcontroller integration.

The locked SHA-256 workload makes Thumb/`thumbv8m.main-none-eabi` the primary
microcontroller garbling target: it emits 164,288 four-row tables (10.5 MB) at
16-byte labels, compared with RV32IM's 429,216 (27.5 MB). Its measured
symbolic-stack write span is 22.5 KiB, compared with 20.0 KiB for RV32IM;
table traffic, not retained-table RAM, is the dominant difference. RV32IM
remains supported as a compatibility and regression target.

All measurements are generated traffic with a streaming sink. Device RAM must
also include symbolic registers, the active symbolic-stack span, the return
stack, garbler state, and the integrator's bounded transport buffer. The
repository keeps alternative garbling implementations isolated from this
baseline so that their wire formats and security/performance tradeoffs can
iterate independently.

## `cirrus-ert`

`cirrus-ert` is a symbolic interpreter for a deliberately well-behaved subset
of RV32 RISC-V. It executes register values as Boolean wires while retaining
concrete metadata for the control flow and addresses that must remain known.
It supports immediate and register shifts plus low and high RV32 multiplication.
An unknown shift amount emits five symbolic selection stages and an unknown
multiplicand emits fixed long-multiplication rounds; concrete metadata routes
both operations through smaller specialized circuits.

The same interpreter is the host path and the embedded path. `ert_func` runs
a mapped RV32 image on a desktop or server (caller-owned virtual stack,
instruction bytes in `RawMemory::from_slice`). Bare-metal QEMU is a
compatibility gate for the same firmware, not the only supported venue. Site
`site-proofs-guest` uses the host `ert_func` loop: one call per event, event
words in the virtual stack, output in a result register.

It is intended for circuit-oriented execution, not as a general RISC-V
emulator. Programs must use the documented supported instruction subset,
aligned non-compressed control flow, concrete branch decisions, conventional
calls and returns, and caller-provided stacks with sufficient capacity. The
[crate documentation](crates/ert/cirrus-ert/src/lib.rs) describes the supported
instructions, buffers, ECALLs, and public entry points. When to use ERT versus
WASM/LLVM → VAFFLE → Boolar: [frontend-choice.md](crates/ert/frontend-choice.md).

Instruction images are supplied through `RawMemory`. `RawMemory::from(&slice)`
safely creates a bounded mapping for a host buffer; its lifetime keeps the
buffer borrowed. `unsafe RawMemory::new(...)` is reserved for an unbounded
bare-metal mapping at native RV32 addresses, without claiming that address zero
is the start of a valid 4 GiB slice. Host tests that relocate a QEMU-linked
ELF (workload at `0x8000_0000`) use the unbounded constructor with a pointer
adjusted so guest addresses land in the mapped buffer.
