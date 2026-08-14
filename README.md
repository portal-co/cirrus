# cirrus

New circuit compiler, for everywhere

## `cirrus-ert`

`cirrus-ert` is a symbolic interpreter for a deliberately well-behaved subset
of RV32 RISC-V. It executes register values as Boolean wires while retaining
concrete metadata for the control flow and addresses that must remain known.
It supports immediate and register shifts plus low and high RV32 multiplication.
An unknown shift amount emits five symbolic selection stages and an unknown
multiplicand emits fixed long-multiplication rounds; concrete metadata routes
both operations through smaller specialized circuits.

It is intended for circuit-oriented execution, not as a general RISC-V
emulator. Programs must use the documented supported instruction subset,
aligned non-compressed control flow, concrete branch decisions, conventional
calls and returns, and caller-provided stacks with sufficient capacity. The
[crate documentation](crates/cirrus-ert/src/lib.rs) describes the supported
instructions, buffers, ECALLs, and public entry points.

Instruction images are supplied through `RawMemory`, an unsafe raw guest-memory
mapping rather than a slice. A bounded mapping is appropriate for a host buffer;
an unbounded mapping can address a bare-metal RV32 image at its native addresses
without claiming that address zero is the start of a valid 4 GiB slice.
