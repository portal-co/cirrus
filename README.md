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
crate documentation in `crates/cirrus-ert/src/lib.rs` describes the supported
instructions, buffers, ECALLs, and public entry points.
