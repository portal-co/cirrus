# Plan: space-efficient local-IR interpreter and bytecode

> **Status:** implementation plan and research record. The `CRBC` layout and
> API sketches below are proposed v1 design input, not a committed public ABI.
>
> **Decision:** extend `Program`/`PreparedProgram` with first-class Boolean
> storage operations and static storage initialization, then lower the resulting
> artifact (and later other local IRs) to a separate, validated compact
> executable before interpreting. Do **not** make existing Rust data structures
> or their in-memory layout a bytecode ABI, and do **not** replace `Program` as
> the portable, fixed-identity circuit interchange format.

## 1. Goal

Provide a `no_std` interpreter path whose retained executable bytes and code
footprint are materially smaller than an interpreted Rust `Program`, while
preserving the exact execution order and backend calls of the existing
executors. The target is a caller-owned byte slice, a caller-owned scratch
buffer, and an allocation-free hot path suitable for the embedded Cirrus
profile. A host may allocate while *transpiling* and validating; a deployed
interpreter must not need `alloc`.

The initial target is the Boolean recompile IR:

- `Program` is a flat SSA-like trace. Each `Op`'s position is its output
  `Idx`; today its values are `Create`, `BitAnd`, `BitOr`, `BitXor`, `Mux`, or
  `External`. This plan extends it with storage reads and effect-placeholder
  writes, plus a declared storage layout/static-initialization table.
- `PreparedProgram` already makes destinations explicit, preserves exact input
  and output order, and represents repeated straight-line regions as nested
  `Statement::Loop` templates with invocation rows and a slot table.
- `PreparedProgram::compact_slots()` can reuse physical scratch slots after
  loop discovery, while reserving a prefix for declared inputs.
- `cirrus-recompile-rt` is the semantic reference today: raw execution walks
  `Program::ops`; prepared execution resolves table slots then applies the
  same operation. Both use caller-visible input/output order and current
  external-dispatch semantics. It must gain the same caller-owned-storage
  execution modes before bytecode storage support is implemented.

The new executable is **not** an alternative symbolic IR or a general-purpose
VM. It has no dynamic branches, calls, heap, dynamic stack, self-modifying
code, or instruction-level type checks. The lowering has already fixed the
execution schedule. The interpreter only decodes compact records, resolves
slots, routes declared storage accesses to caller-owned banks, invokes the
pinned Boolean backend operation, and reports malformed input before backend
execution.

### Success criteria

1. `execute_compact` is semantically equivalent to `execute_prepared`/
   `execute_prepared_with_externals` for every valid lowered artifact,
   including nested loops, compacted slots, declared inputs, external
   occurrence metadata, storage reads/writes, and one-shot versus persistent
   static-storage initialization semantics.
2. The default embedded feature has no `alloc`, no recursion, no `std`, no
   dynamic dispatch required by the bytecode loop, and bounded validation
   work/memory. The caller provides `&mut [MaybeUninit<Wrapped>]` or the
   existing appropriate scratch abstraction.
3. The executable has an explicit version, bounded lengths/counts, canonical
   integer encodings, and an all-or-nothing admission phase. Truncation,
   overflow, noncanonical encodings, unknown opcodes, out-of-range slots,
   invalid loop nesting, and inconsistent section lengths fail closed before
   an operation is issued.
4. Interpreter machine code and executable bytes are both measured, checked
   into a reproducible benchmark report, and gated against agreed budgets on
   the primary `thumbv8m.main-none-eabi` target. Raw `Program`, prepared
   in-memory, and compact executable measurements must be reported separately.
5. Existing raw `Program`, R1CS/Groth16, Rust-source, LLVM, and assembly
   consumers retain their current semantics and APIs. Bytecode is opt-in.

### Non-goals for v1

- Persisting or serializing backend `Wrapped` values.
- General data-dependent control flow, bytecode calls, typed arithmetic,
  debugger metadata, bytecode patching, JITing, or a stable cross-language
  ABI. Boolean `ContextWithStorage<bool>` reads/writes and Boolar-compatible
  static initialization are explicitly **in** scope; arbitrary typed storage,
  allocation, and a bytecode-owned heap are not.
- Replacing source-aware `PreparedProgram` construction/reoptimization or
  assuming every program benefits from loop abstraction.
- Combining executable bytes with untrusted external names/action policy;
  bytecode records reference an application-supplied external manifest.

## 2. Constraints and seams in the current repository

| Existing component | Constraint for this plan | Intended relationship |
| --- | --- | --- |
| `cirrus-recompile-core::{Program, Op, Idx}` | Positional output identity is required by raw consumers and ZK witnesses. | Remains canonical input/interchange. Transpile from it only through a prepared scheduled form. |
| `PreparedProgram` | Has explicit outputs, nested loops, table-driven slot references, validation, and optional slot compaction. | Extend it with storage prepared ops and storage/static-init metadata; retain its existing table/scoping model and do not expose its Rust `Vec` layout. |
| `cirrus-recompile-rt` | Owns reference operation/external semantics, but its prepared executor allocates `Vec<Option<_>>`, validates at execution, and recursively walks loop statements. | First add raw/prepared storage reference executors over caller-owned banks; retain them as bytecode oracle. Compact runtime is a sibling, not a hidden alternate path. |
| `cirrus_core::ContextWithStorage<bool>` | The established storage seam takes caller-owned storage and an LSB-first `&[StorageAddressBit<Wrapped>]`, returning a read value or applying a write. Namespace/lane routing belongs above the trait. | `Program` storage ops encode only stable bank IDs and slot-valued address/value operands; interpreter resolves IDs through a caller-supplied bank table and constructs `StorageAddressBit` from scratch values plus known-bit facts. |
| pinned backend ABI | `create`, binary ops, and `mux` already define the backend boundary. | Invoke the same traits/functions in the same schedule; add storage calls through `ContextWithStorage`, never encode function pointers in bytecode. |
| `ExternalOp` | Names and arguments are owned `String`/`Vec`; calls carry kind, bit, and occurrence. | Move variable metadata to a manifest/table and reference it by compact ID; v1 embedded profile rejects unresolved or malformed entries before run. |
| Boolar `BIrStmt::{StorageRead, StorageWrite, ActionStoreBit}` | Boolar uses `(StorageId, LaneId)`, LSB-first address vectors, caller-owned banks, and a separate `pre_init` lifecycle. A storage write still occupies a statement position; the current executor produces a false placeholder for it. | Make this the semantic compatibility target: emitted Boolar storage statements lower losslessly into the extended Boolean `Program`, including a non-readable effect placeholder and static initialization. Do not serialize `BIrStmt` Rust layouts. |
| `TypedProgram` / `TypedPreparedProgram` | Typed preparation is identity-only today; `ActionStore` owns direct typed effects and is deliberately distinct from Boolean trace semantics. | Keep typed storage/action work separately versioned. The Boolean storage extension does not redefine typed operations. |
| Boolar chunked execution | It already has a streamed/chunked circuit path and caller-defined wire codec/scratch. | Reuse the same bank and initialization semantics; a later direct compact lowering may share `CRBC` mechanics only after differential tests establish equivalence. |

The compact artifact is produced **after** `Program::prepare` and, when the
memory budget calls for it, after `compact_slots`. That order is required:
loop recognition relies on the original positional regularity, and the
existing compaction pass explicitly documents that it must follow loop
abstraction.

### 2.1 Required Boolean-storage extension

Storage is a first-class extension of the Boolean `Program` family, not an
`ExternalOp`, an opaque host callback, or a pre-lowering to MUX gates. The
extension must model the subset already executed by Boolar:

```rust
pub struct ProgramStorageBank {
    pub storage: u64,       // logical namespace, not a host pointer
    pub lane: u64,          // one-bit lane within that namespace
    pub address_bits: u32,  // fixed LSB-first address width
}

pub struct StorageInitSegment {
    pub bank: u32,
    pub addr: Vec<bool>,    // LSB-first start address
    pub data: Vec<bool>,    // consecutive public bits
}

pub enum Op {
    // Existing value-producing variants...
    StorageRead { bank: u32, address: Vec<Idx> },
    StorageWrite { bank: u32, address: Vec<Idx>, value: Idx },
}
```

The exact names/types remain an implementation decision, but these semantics
are required:

- `Program::storage_banks` is a canonical, unique table keyed by logical
  `(storage, lane)`. `storage_init` is an ordered table of public-bit segments
  using that table. A program declares every bank it can access; the runtime
  receives the actual caller-owned bank objects separately.
- `StorageRead` is value-producing: its positional `Idx` holds the returned
  Boolean wire. `StorageWrite` is effect-only but still occupies its positional
  `Idx` so a raw trace retains stable SSA numbering. That slot is an **effect
  placeholder**, not `false`: `Program`/`PreparedProgram` validation rejects
  it as an input, output, operand, address bit, external argument, or any
  later data use.
- `address.len()` must exactly equal the referenced bank's `address_bits`; the
  address order is least-significant bit first. Each address and write value
  must name an earlier value-producing slot. Reads may occur in loops and
  writes retain program order; no alias analysis, reordering, or implicit
  initialization is allowed.
- `StorageInitSegment` matches Boolar `BCircuit::pre_init`: applying a segment
  writes `data[0]`, `data[1]`, … at successively incremented LSB-first
  addresses from `addr`, with checked width/overflow. Segment ordering is
  observable when segments overlap.
- Raw and prepared storage execution call
  `ContextWithStorage<bool>::storage_read` / `storage_write` with
  `StorageAddressBit { wire, known }`. A compact known-fact sidecar preserves
  the same constant facts used by Boolar rather than treating every address
  bit as unknown.

`PreparedOp` must carry its resolved `bank`, dynamically sized
`Vec<PreparedSlot>` address operands, and for writes its value slot. This makes
loop table fields work for every address/value position. It means
`PreparedOp`/`Statement` need not remain `Copy`; use explicit `Clone` and audit
all optimizer/executor assumptions rather than flattening a storage access or
silently losing table references. Loop reabstraction may combine storage
statements only when bank, address arity, operation kind, and all operand
classification/template constraints match.

`compact_slots` must understand effect placeholders: they reserve source slot
identity but no live backend value. It must neither assign them a value-bearing
physical scratch location nor permit a placeholder to hide an invalid later
use. Update raw/prepared reference executors to track that state explicitly
rather than manufacturing a `false` backend value for it; the compact
executable likewise omits a write destination entirely.

### 2.2 Boolar lowering boundary

Add a direct, validated Boolar-to-`Program` adapter for the supported Boolean
statement subset: `Zero`, `One`, `And`, `Or`, `Xor`, `Not` (lowered using the
existing Boolean semantics), `StorageRead`, `StorageWrite`, `OracleBit`,
`RngBit`, and `ActionStoreBit` where supported. It copies Boolar bank identity,
address order, `pre_init`, statement order, and effect placeholders exactly.

Do **not** obtain this representation by running `Recorder` through a dense
MUX-tree storage context: that expands a native storage operation into gates,
loses the storage-bank operation boundary, and defeats the intended bytecode
compactness. The adapter should live at the existing Boolar/recompile
integration seam (with an optional dependency if necessary), not make
`cirrus-recompile-core` depend on Volar types. It owns the lossless mapping
between Boolar's ID newtypes and the Program's primitive storage IDs.

Before accepting an `ActionStoreBit` mapping, specify whether it remains an
external host action or becomes a direct `StorageWrite`. The default is to
retain it as an external action because it includes action semantics beyond a
plain write; only direct `StorageRead`/`StorageWrite` and `pre_init` are part
of the base storage extension.

## 3. Prior art and conclusions

| Prior art | Primary-source fact | Decision for Cirrus |
| --- | --- | --- |
| Bell, *Threaded Code* (CACM, 1973) | A threaded program is a sequence of routine addresses; parameters can follow the threaded link. On PDP-11, the paper reports threaded code typically 10–20% shorter and 2–3% slower than its hard-code comparison, and notes that only used service routines need be retained. | Take the **small, closed primitive set** and shared operation bodies. Reject address-threaded bytecode: host pointer width/relocation makes bytes nonportable, function-pointer dispatch harms deterministic validation, and the target has a closed set of Boolean and storage op families. |
| Forth’s indirect-threaded implementation (1978 contemporary account) | Dictionary entries separate compile-time names/links from a code address and parameter field; the article reports a standalone compiler/assembler/editor/runtime around 6 KiB, and a stripped stand-alone form deletes symbolic identifiers. | Separate runtime-required executable records from host-only names/debugging. Do not carry `String` names in the embedded executable. |
| UCSD p-System | Its historical p-code architecture is a portable abstract-machine precedent. | Take the two-stage *portable IR → compact executable* boundary, not its broad instruction set or dynamic machine model. |
| JVM specification | One-byte opcodes with byte-aligned operands consciously favor compactness over naive decoding speed; implicit stack operands and specialized forms such as `_0` avoid operand bytes, while `wide` expands rare large local indices. Static verification is performed once before execution. | Use a small opcode byte space and validate once. Do **not** adopt a value stack: Cirrus must name arbitrary scratch values and preserve `PreparedProgram` slot semantics. Consider only evidence-backed short forms after profiling. |
| Lua 5.4 `lopcodes.h` | Uses fixed 32-bit instructions: 7-bit opcode plus several 8/17/25-bit operand layouts, with an `EXTRAARG` extension for rare large operands. | Keep this as the fixed-width control experiment. It may win on decode/code size for uniformly large slot IDs, but is not the default because Boolean traces often have small/delta-friendly IDs and bytecode footprint is a primary goal. |
| WebAssembly binary specification | Uses one-byte opcode plus immediate arguments, unsigned/signed LEB128 with width bounds, structured blocks delimited by `else`/`end`, and explicit section lengths. | Adopt bounded, canonical LEB-style integer decoding and structured loop delimiters/lengths. Unlike Wasm, require canonical shortest encodings to give each executable one byte representation. |
| WebAssembly Micro Runtime | Its loader prepares decoded WebAssembly into interpreter-friendly bytecode; its fast interpreter can dispatch with a switch or computed-goto tables and its internal prepared form can use host-width dispatch data. | Keep the portable artifact compact and pointer-free; optional host-only decode/preparation is permitted only as a separate cache, never the embedded executable format. Start with `match`/`switch`; benchmark computed goto only where Rust/target support makes it a measured win. |
| Dalvik/DEX | A register machine with fixed-size frames uses 16-bit code units and several narrow/wide operand layouts; DEX uses LEB128 for metadata. | Retain explicit destination-first slot semantics and fixed scratch width metadata. Reject a register encoding tied to 4/8/16-bit operand formats until trace distributions prove it smaller than varints. |

The historical lesson is not that one representation wins universally. A
compact interpreter succeeds when its *operation vocabulary and operands fit
the workload*, verification moves out of the hot loop, symbolic/debug
information is excluded from deployment, and a fixed representation is
compared against a variable one with real workloads.

## 4. Proposed architecture

Introduce a small new crate, tentatively `cirrus-recompile-bytecode`, rather
than adding format parsing to `cirrus-recompile-core` or making
`cirrus-recompile-rt` own a persistent format:

```text
Program + storage metadata ──prepare/reoptimize──> PreparedProgram
          │                                  └──optional compact_slots──────┐
          └──Boolar adapter──────────────────────────────────────────────────┤
                                                                            │
                     host-only transpile + validate + size admission ────────┤
                                                                            v
                            CompactProgram<'a> / canonical `CRBC` bytes
                                                                            │
                     no_std validate-once ───────────────────────────────────┤
                                                                            v
                           allocation-free compact interpreter
                                      │
                 Boolean context/pinned ABI + caller-owned storage banks
```

### 4.1 Public boundary

The owning host-side API should look conceptually like:

```rust
pub struct CompactOptions {
    pub slot_mode: SlotMode,             // Preserve | CompactAfterPrepare
    pub encoding: EncodingPolicy,        // Varint | Fixed32 control experiment
    pub loops: LoopPolicy,               // Preserve | Flatten
    pub storage: StorageMode,             // Reject | BooleanBanks
    pub max_bytes: usize,
}

pub fn transpile(
    program: &PreparedProgram,
    options: &CompactOptions,
) -> Result<Vec<u8>, TranspileError>;

pub fn validate(bytes: &[u8]) -> Result<CompactProgram<'_>, DecodeError>;

pub fn execute<Backend>(
    program: CompactProgram<'_>,
    backend: &mut Backend,
    scratch: &mut [core::mem::MaybeUninit<Backend::Wrapped>],
    facts: &mut [KnownBit],
    address: &mut [core::mem::MaybeUninit<StorageAddressBit<Backend::Wrapped>>],
    inputs: &[Backend::Wrapped],
    banks: &mut [RuntimeStorageBank<'_, Backend::Storage>],
) -> Result<CompactOutputs<'_, Backend::Wrapped>, ExecuteError<Backend::Error>>;

pub fn initialize_storage<Backend>(
    program: CompactProgram<'_>,
    backend: &mut Backend,
    address: &mut [core::mem::MaybeUninit<StorageAddressBit<Backend::Wrapped>>],
    banks: &mut [RuntimeStorageBank<'_, Backend::Storage>],
) -> Result<(), ExecuteError<Backend::Error>>;
```

Names are illustrative, not a committed API. `CompactProgram<'a>` borrows a
validated byte slice and exposes only checked metadata and bounded program
ranges. It is `Copy`/small where possible; it must not retain a `Vec`, borrow
an external registry, or use a pointer into an unbounded decoded object.

Do **not** make `validate` optional for bytes received from storage/network.
For a trusted static `include_bytes!` artifact, offer an explicit unsafe or
build-generated admission path only if it demonstrably removes code without
letting malformed data reach the interpreter. The safe default is always
validate once then execute many times.

### 4.2 Execution state

The interpreter owns only fixed-size scalar state:

- `pc`/end pointer for the active byte range;
- the caller’s scratch slice and `KnownBit` sidecar, each of length exactly
  header `slot_count`; facts are `unknown`, `false`, `true`, or `effect`, and
  are maintained by the same Boolean constant propagation contract as Boolar;
- no input-membership bitmap in the hot path: initialize the header’s input
  slots in caller order, and have the transpiler omit the source `Create`
  definition for every declared input slot;
- a bounded loop-frame stack supplied by the caller or derived from a header
  `max_loop_depth`; each frame stores body bounds, next row, end row, table
  base, and its enclosing scope;
- one caller-provided reusable address workspace sized for the validated
  maximum bank address width; and
- a caller-owned, uniquely keyed storage-bank slice, whose entries carry the
  stable `(storage, lane)` identity, declared LSB-first address width, and
  `&mut Backend::Storage`; and
- an optional external manifest supplied by the embedding application.

Avoid `Vec<Option<Wrapped>>` in the compact path. The transpiler proves that
each read is defined before use, so storage can be `&mut [MaybeUninit<W>]`.
Inputs are initialized before entering the instruction stream with `unknown`
facts; `CONST*` establishes known facts; Boolean operations propagate only facts
that are exact under their truth tables; storage/external reads produce
`unknown`; and writes produce `effect`. Every output is read only after the
decoded program returns successfully and must have a non-effect fact. A backend
operation failure stops execution and leaves scratch/facts unspecified, matching
the current fail-fast contract.

Storage banks are never encoded as pointers or owned by the artifact. The
artifact's storage declaration allows admission to reject duplicate/missing
banks and address-width mismatches before it calls `ContextWithStorage`. It
must support both dense and sparse/native authenticated contexts: dense cell
counts are a caller/layout policy, not a bytecode-derived allocation request.

The v1 decoder/interpreter is intentionally iterative. Nested prepared loops
must be executed with an explicit bounded frame stack; recursion is not
permitted in the embedded feature. `max_loop_depth` is encoded and checked
against a compile-time/runtime caller limit before execution.

### 4.3 Semantic invariants preserved by transpilation

The transpiler must prove and the decoder must re-check structural properties
without depending on Rust type layout:

1. `slot_count`, every declared input/output, every static operand, every
   table value, and every external argument is in range.
2. A static/read slot is defined before it is used in its dynamic schedule;
   each input slot is initialized exactly from the caller input order, its
   source `Create` definition is omitted, and it is never computed into. This
   remains true after `compact_slots` because it reserves its input prefix.
3. A table slot can refer only to an active lexical loop scope; the encoded
   loop body cannot recursively contain itself and its row/table lengths agree.
4. Every storage operation names a declared `(storage, lane)` bank, has exactly
   that bank's declared LSB-first address width, and reads only earlier defined
   address/value slots. A read defines a value; a write defines an implicit
   effect-placeholder slot that no later value-bearing operation, output, or
   storage address/value may consume. In compact execution that placeholder
   has no scratch location and is represented only in validation/fact state.
5. Static initialization is a separate ordered program prefix. One-shot
   execution applies it before the entry range; persistent execution requires
   the caller to invoke it once and never replays it. Initializer addresses
   fit their declared bank widths and initializer bits are public constants.
6. Each opcode has one exact operand grammar. All record lengths, section
   lengths, multiplication/addition for row/table bounds, and pointer
   advances use checked arithmetic.
7. Output order, external `(kind, manifest ID, bit, occurrence, argument
   order)`, and storage read/write order equal the source `PreparedProgram`.
   The interpreter does not reorder independent operations merely because
   their values are ready.
8. The format carries no host pointer, `usize`, Rust enum discriminant,
   `Vec` capacity, endianness-dependent native integer, or backend value.

## 5. `CRBC` v1 executable format

`CRBC` is a proposed **private, versioned executable format**, not yet a
cross-version promise. It is deliberately self-delimiting so it can be stored
in flash, framed in a transport, or hashed without surrounding metadata.

### 5.1 Envelope

All multi-byte numeric fields are canonical unsigned LEB128 (`u32` in v1)
unless stated otherwise. `slotref` is the one exception: it is a canonical
`u64` LEB because its low-bit tag packs a full `u32` `Idx` without artificially
reducing the source IR’s index range. Canonical means the shortest
representation only; reject continuation after the legal width, overflow,
truncation, and redundant high zero groups. This is stricter than WebAssembly,
which accepts some non-shortest encodings, because canonical bytes simplify
hashing, golden tests, and size accounting.

```text
magic[4]       = b"CRBC"
version:u8     = 1
flags:u8       = 0 in v1; unknown bits reject
header_len:uLEB
header[...]    = bounded header fields below
sections[...]  = ordered, length-delimited sections
```

The header contains:

```text
slot_count:uLEB
input_count:uLEB       input_slots[input_count]:uLEB
output_count:uLEB      output_slots[output_count]:uLEB
external_count:uLEB
storage_bank_count:uLEB
max_loop_depth:uLEB
init_len:uLEB          byte length of the static-initialization range
entry_len:uLEB         byte length of the root instruction range
```

The v1 section order is fixed, not tag-dispatched: length-delimited external
manifest reference table, storage-bank declaration table, and loop table blob,
then exactly `init_len` bytes of static initialization and exactly `entry_len`
bytes of root instruction stream. Decoder consumption must equal every declared
length exactly. Fixed ordering is smaller and simpler than a generic section
map; a v2 may add known, length-delimited optional sections only behind a new
version or negotiated feature bit.

The external table contains numeric logical IDs only. Its strings, capability
policy, and action implementation belong to the deployment manifest rather
than flash-resident executable bytes. The transpiler accepts a stable mapping
from existing external metadata to those IDs and refuses absent/ambiguous
entries.

The storage-bank declaration table is sorted by a canonical `(storage_id,
lane_id)` key and has one entry per bank:

```text
storage_id:uLEB
lane_id:uLEB
address_bits:uLEB
```

These IDs are format-level numeric identities assigned by the Boolean Program
lowering; they are not native pointers or an implicit dense-memory index. The
transpiler must map Boolar's `StorageId`/`LaneId` losslessly and reject a
collision. The decoder requires sorted unique entries, a bounded address width,
and an exact caller-bank match before either initializer or entry execution.
`init_len = 0` is canonical when there is no static initialization.

### 5.2 Static initialization stream

The `init_len` range consists only of one opcode in v1:

```text
0x0B INIT_BITS bank_id, address_bits, start_address_packed, bit_count, packed_bits
```

`start_address_packed` is a canonical unsigned integer whose low
`address_bits` bits are the LSB-first numeric address; it must fit that width.
`packed_bits` contains exactly `ceil(bit_count / 8)` bytes, least-significant
bit first within each byte, and unused high bits in its final byte are zero.
`bit_count = 0` is rejected. The interpreter expands this compact public
segment into the reusable `StorageAddressBit` workspace and calls
`storage_write` once per bit in encoded order, incrementing the address with
checked width arithmetic. It lazily creates and retains one backend `zero` and
one backend `one` value for the initialization invocation, cloning the selected
one for each write; it does **not** materialize a scratch slot per initialized
bit. This preserves Boolar `pre_init` segment order and overlap semantics.

No `STORAGE_READ`, dynamic slot reference, external operation, loop, or
arbitrary Boolean instruction is legal in the init range. Initializer records
are not output-producing Program operations; they form the separately invoked
initialization lifecycle.

### 5.3 Instruction stream

Reserve one-byte opcodes `0x00..0x0f`; reject all others in v1. The default
encoding uses **destination-last or destination-first only where it helps
streaming decode**; choose destination-first consistently in the final spec.
The table below uses destination first:

```text
0x00 END                 end current range; legal only at its exact bound
0x01 CONST0  out
0x02 CONST1  out
0x03 AND     out, a, b
0x04 OR      out, a, b
0x05 XOR     out, a, b
0x06 MUX     out, cond, then, else
0x07 EXTERNAL      out, external_id
0x08 LOOP          loop_id
0x09 STORAGE_READ  out, bank_id, address_len, address[slotref]
0x0A STORAGE_WRITE bank_id, address_len, address[slotref], value
```

`STORAGE_WRITE` has a positional source/PreparedProgram output identity, but
its compact encoding deliberately has **no** `out` operand: it defines an
implicit effect placeholder, matching Boolar's statement-position semantics,
which must never be consumed as a Boolean value. It materializes no backend
wire. The raw/prepared reference executor must track the same non-value state;
the compact interpreter must not emit an unnecessary `create(false)` solely
for this effect.

Every slot operand is a canonical `slotref`; `CONST0`/`CONST1` replace a
generic boolean immediate. `address_len` must exactly equal the selected
bank's declared `address_bits`, and its ordered operands are the LSB-first
address bits required by `ContextWithStorage<bool>`. No `NOP`, branch, stack
operation, or generic extension opcode exists in v1. This gives the common
Boolean primitive set direct, small decoding paths and avoids an opcode family
whose only purpose is future proofing.

`END` is a structural delimiter, not an independently scheduled operation.
The root stream may either have an implicit end at `entry_len` or an explicit
terminal `END`; select one during implementation and reject the other to avoid
two canonical representations. Recommended choice: explicit `END`, which
makes nested body validation uniform and follows Wasm’s structured delimiter
model.

### 5.4 Loop descriptor and table representation

A `LOOP loop_id` references a descriptor in the loop section:

```text
body_len:uLEB
fields_per_row:uLEB
invocation_count:uLEB
invocations[invocation_count] = { first_row:uLEB, iterations:uLEB }
row_count:uLEB
slot_table[row_count * fields_per_row]:uLEB
body[body_len] = nested instruction range ending in END
```

The lowering assigns `loop_id`s in preorder, then serializes each descriptor
once. The execution frame resolves an operand either as a static slot (`uLEB`)
or a table reference:

```text
slotref = static-slot | table-slot
static-slot = u64(slot << 1) | 0
table-slot  = u64(depth << 1) | 1, field:uLEB
```

The first `uLEB` packs the static/table tag into its low bit. This avoids
spending a full opcode-specific tag byte and mirrors `PreparedSlot` exactly:
`depth=0` denotes the innermost active loop. For operands used outside any
loop, the transpiler emits a static slot only. The decoder checks scope depth
and `field < fields_per_row`; the interpreter computes row offsets using
checked arithmetic already admitted by validation.

Do not delta-code the slot table in the first implementation. Absolute slots
match `PreparedProgram`, permit direct indexed scratch access, and are simple
to validate. Add a second, explicitly selected table codec only if corpus data
shows that `(previous + signed delta)` reduces total executable bytes after
including its decoder and validation code.

### 5.5 Candidate compact forms, deferred until measurement

The baseline is deliberately boring: one opcode byte plus canonical varints.
The transpiler records statistics, then may add *only measured* variants:

- `AND_S`, `OR_S`, `XOR_S` with one-byte slots when all three operands are
  `< 256`;
- a storage-op form that omits `address_len` only when the selected bank's
  declared width supplies it unambiguously; and run-length/static-address
  variants for initializer segments only if they beat `INIT_BITS` including
  decoder code;
- `OUT_NEXT` variants when output is exactly a statically known next slot;
- a binary operand relative-to-output encoding when its signed deltas are
  materially smaller across real traces;
- a fixed 32-bit word stream, Lua-style bit fields, as an alternate
  `encoding` selected at artifact creation;
- direct template table indexes if operand table fields are more common than
  static slots.

Each variant must have one decoder path, an exact canonical selection rule in
the transpiler, property tests that prove it is never larger than its baseline
when selected, and a size/code-size benchmark result. Do not add a generic
prefix/escape mechanism speculatively.

## 6. Implementation sequence

### Phase 0 — freeze baselines and budgets

1. Add `crates/recompile/cirrus-recompile-bytecode` to the workspace with
   `#![no_std]`; default features must not pull in `alloc`. Keep the host
   transpiler behind an `alloc` feature or in a companion crate if that yields
   a smaller embedded dependency graph.
2. Create a corpus generator from existing `Recorder`/ERT fixtures, direct
   `PreparedProgram` tests, and Boolar circuits. At minimum include: empty and
   constant-only programs; every value and storage `Op`; arbitrary
   input/output placement; external ops; one-shot and persistent static-bank
   initialization (including lazy reusable zero/one values); concrete and
   symbolic LSB-first addresses; overlapping init
   segments; multiple `(storage, lane)` banks; nested loops; zero-iteration
   loop invocations; merged loops; loop tables with parent-scope references;
   raw and `compact_slots` forms; the locked Thumb SHA-256 trace; and a
   representative RV32 trace.
3. Record baseline metrics for each corpus entry: raw Rust heap estimate
   (`Op`/`Vec` metadata reported only as diagnostic), `PreparedProgram`
   estimated and actual retained allocation, `compact_slots` scratch width,
   static generated Rust/LLVM/assembly size where available, and the eventual
   `CRBC` byte length. Never compare only `UNROLLED_OP_BYTES` (currently an
   optimizer heuristic) to serialized bytes.
4. Establish target-specific budgets before selecting encodings: `.text`,
   `.rodata`, `.data + .bss`, maximum validation stack/frame storage,
   scratch width, executable bytes, and execution cycles/operations. The
   primary gate is release `thumbv8m.main-none-eabi`; host results are
   supplementary.

### Phase 1 — semantic scheduled intermediate and reference oracle

1. First implement the §2.1 `Program`/`PreparedProgram` storage extension:
   storage-bank declarations, ordered static-init segments, `StorageRead`,
   `StorageWrite`, effect-placeholder validation, prepared loop/table support,
   known-bit propagation, and liveness/slot-compaction rules. Add the matching
   raw/prepared `ContextWithStorage<bool>` executors with explicit one-shot and
   persistent initialization entry points.
2. Expose a narrow read-only scheduled traversal or place an internal adapter
   beside the transpiler. It must enumerate validated value and storage
   statements, known facts, static initialization, and entry/loop structure
   without flattening it or changing scheduling semantics.
3. Factor tests, not behavior: use the new prepared-storage executor plus
   current external executors as the reference oracle. Do not refactor the
   current runtime solely to share a large generic evaluator if it grows the
   embedded binary.
4. Implement an owned host `LoweredProgram`/writer state that receives a
   `PreparedProgram`, validates it first, optionally calls `compact_slots`
   only when requested, maps external metadata to manifest IDs, assigns loop
   IDs preorder, and emits exactly one canonical baseline representation.
5. Return a structured `TranspileError` for source validation failure, missing
   manifest/bank mapping, unsupported policy, `u32` limit breach, or
   `max_bytes` admission failure. No silent flattening or fallback encoding.
6. Add the lossless Boolar adapter described in §2.2 and differential tests
   against `cirrus-volar-boolar::{execute, execute_initialized}` for the
   supported statement subset. Do not let the bytecode work begin until this
   semantic boundary is passing.

### Phase 2 — bounded decoder and format admission

1. Implement `Reader<'a>` over `&[u8]` with checked byte consumption and
   canonical `read_u32_leb`. Make all error states explicit: bad magic,
   version, flags, noncanonical/overflow/truncated integer, invalid section
   length/order, unknown opcode, invalid slot/reference, malformed loop,
   unsupported external, and resource limit.
2. Validate the full program before returning `CompactProgram<'a>`. Validation
   records only offsets/lengths/counts that the interpreter needs; it does not
   allocate a decoded instruction vector. If loop descriptors cannot be
   revisited from bytes without a dynamic map, use their deterministic preorder
   index and a validated descriptor-offset table supplied by the caller or
   store a small fixed-offset directory in the artifact. Choose the smaller
   measured option.
3. Enforce caller-provided limits (`max_bytes`, `max_slots`, `max_loops`,
   `max_loop_depth`, `max_table_entries`, `max_external_args`, `max_banks`,
   `max_address_bits`, and `max_init_bits`) before any multiplication or
   pointer advance. Derive all nested range ends from validated section bounds,
   never from a sentinel alone.
4. Add golden bytes for a hand-written minimal corpus. Golden tests assert
   byte-for-byte canonical output, not merely successful round trip.

### Phase 3 — allocation-free Boolean interpreter

1. Implement iterative `execute_compact` on a validated `CompactProgram`.
   Decode instruction operands at execution from validated bytes; do not repeat
   format-wide validation. Per-instruction bounds may remain debug assertions
   if the validation proof makes them redundant in release.
2. Use a fixed `LoopFrame` supplied as `&mut [LoopFrame]` or a const generic
   only after evaluating binary-size impact. Return `LoopStackTooSmall` before
   executing if validated `max_loop_depth` exceeds capacity.
3. Initialize inputs in source order, run `CONST*`/Boolean ops through the same
   `cirrus_core` traits as `cirrus-recompile-rt`, and collect outputs in header
   order. Lowering omits the declared-input `Create` records, so the hot loop
   has no input-membership branch. Avoid cloning by using backend operations
   appropriate to `Wrapped`; preserve existing clone bounds initially if
   changing them would be a semantic/API risk.
4. Implement `initialize_compact_storage` and persistent execution. For each
   static-init bit or storage operation, resolve the validated bank ID, fill
   the reusable LSB-first `StorageAddressBit` workspace from slot/fact pairs,
   and call `ContextWithStorage<bool>`. A storage write updates only the host
   bank and marks its positional slot `effect`; a read writes the returned wire
   into scratch and marks it `unknown`. Do not allocate an address `Vec`, make
   a dense-bank assumption, or synthesize a false wire for the placeholder.
5. Add `execute_compact_with_externals` with a compact manifest interface.
   Resolve an ID to the existing external semantic fields before execution;
   preserve current `ExternalRegistry` call ordering and arguments. An
   embedded profile without external support rejects nonzero external counts at
   validation.
6. Keep raw and prepared execution public and tested. The compact path is an
   additional backend choice selected by an integrator, never an implicit
   optimization that changes diagnostic behavior.

### Phase 4 — local-IR generalization

1. Split the container/reader primitives from Boolean semantics only after the
   Boolean format is stable under the above benchmarks. The shared layer may
   own canonical integers, bounded ranges, header/version handling, and a
   length-delimited structured-body walker.
2. Define one `CompactLowering` trait per IR family with an explicit semantic
   version, resource declaration, opcode namespace, manifest requirements,
   validation rules, and reference executor. Do not force Boolean `Idx` and
   heterogeneous typed values into one slot encoding.
3. For `TypedPreparedProgram`, first add an explicit typed structured schedule
   and type-table/constant representation; then design its own record shapes.
   Reuse only the container mechanics. Its existing identity-prepared form is
   not evidence that Boolean loop table encodings are valid for it.
4. Treat the direct Boolar-to-Program adapter as a supported input, not merely
   a future evaluation. It must preserve `StorageRead`/`StorageWrite`, bank
   IDs, LSB-first addresses, `pre_init`, operation order, and effect
   placeholders. Continue to use Boolar's own lazy chunk format for its
   streaming-specific path; do not serialize foreign `BIrStmt` Rust layouts.

### Phase 5 — optimize only with evidence

1. Compare baseline varints with fixed 32-bit Lua-style words, short-slot
   forms, output-relative operands, delta table codecs, compact storage/address
   forms, static-init packing variants, and optional host-only predecode
   caches. Record both executable-byte savings and interpreter `.text` delta
   on Thumb.
2. Retain an encoding only when it improves the agreed weighted deployment
   objective (flash includes interpreter plus representative artifact bytes;
   RAM includes validation/frame state) and does not violate the cycle budget.
3. Keep format version 1 frozen once any artifact is checked into firmware.
   Incompatible changes create version 2. A decoder may support multiple
   versions only if the flash cost is explicitly accepted; otherwise migration
   occurs at the host transpiler boundary.

## 7. Test and measurement plan

### Correctness

- **Differential property tests:** generate valid small `Program`s through
  `Recorder`, prepare/compact/lower/decode/execute with a plaintext context,
  and compare outputs to `cirrus_recompile_core::interpret` and
  `cirrus_recompile_rt::execute_prepared`. Repeat each Boolean input vector
  where tractable.
- **Structured tests:** use manually built `PreparedProgram`s to cover nested
  table-depth resolution, variable invocation counts, zero iterations,
  descriptor row boundaries, all value/storage `PreparedOp` variants, inputs
  located inside a source trace, outputs pinned across slot reuse, external
  argument references, table-driven storage addresses, and attempted reads of
  effect placeholders.
- **Storage tests:** compare raw, prepared, compact, and Boolar execution on
  fake banks that log every `(bank, LSB-first address, read/write, value)`
  event. Cover multiple namespaces/lanes, concrete and symbolic address facts,
  missing/duplicate/width-mismatched banks, static-init overlap/order,
  initialization once versus persistent execution, storage loops, and a sparse
  bank. Assert no compact path turns a storage write into `create(false)`,
  allocates a scratch slot per initialized bit, or replays `pre_init` during
  persistent execution.
- **External tests:** fake registry records exact kind/name/argument order/bit/
  occurrence. Assert compact and reference paths produce identical transcripts
  and reject absent manifest IDs before invoking the backend.
- **Malformed corpus:** mutate every header, varint, record boundary, opcode,
  slot, loop descriptor, table element, range, count, manifest reference,
  storage declaration, bank ID, address arity, initializer address/length, and
  effect-placeholder use.
  Require no panic, no out-of-bounds access (Miri/sanitizers on host), bounded
  rejection, and zero backend calls during validation failure.
- **Canonicality:** for every accepted byte sequence, transcode/decode/reencode
  and assert identical bytes. For every integer field, test all redundant,
  truncated, overflow, and forbidden-width alternatives are rejected.
- **Fuzzing:** libFuzzer/AFL-style `validate` target plus a separate
  `validate_then_execute` plaintext target with resource caps. Run only host
  fuzzing; it does not substitute for embedded testing.

### Embedded and artifact tests

- Compile the compact interpreter and a minimal static `CRBC` artifact for
  `thumbv8m.main-none-eabi` in release with the same controlled linker/panic
  settings as the existing bare-metal fixtures. Record section sizes with a
  checked-in script using the installed toolchain’s size utility.
- Add a bare-metal QEMU test only after the compact runner is integrated into a
  real no-std fixture. Use the existing Cortex-M33/`mps2-an505` approach,
  explicit generated QEMU arguments, software emulation, ephemeral artifacts,
  bounded timeout, and preserved failure logs. If QEMU/image prerequisites are
  unavailable, fail closed and report missing coverage; do not claim Linux or
  device coverage from the macOS host.
- Run the locked Thumb SHA-256 workload through both reference prepared and
  compact execution. Compare output and complete backend-operation counters/
  traffic, then report executable bytes, scratch/fact/address-workspace width,
  frame capacity, and cycle/operation count. This workload’s garbling-table
  traffic is distinct from interpreter bytecode and must remain a separate
  column.
- Add a storage-bearing Boolar fixture that runs through the direct adapter and
  compact interpreter against the same dense and sparse caller banks. Measure
  it separately: dense-bank capacity is an integrator allocation cost, whereas
  the compact executable must be insensitive to physical bank representation.

### Acceptance table

| Gate | Required result |
| --- | --- |
| Format safety | All malformed and fuzzed inputs reject without panic/backend calls before admission. |
| Semantics | Compact/reference/Boolar outputs, storage-event logs, initialization lifecycle, and external transcripts match across corpus and property tests. |
| Determinism | Same prepared input/options/manifest mapping produces identical bytes. |
| Footprint | Measured Thumb `.text`/RAM and representative value-only/storage-bearing artifact bytes satisfy budgets, with raw/prepared/compact comparison published. |
| Execution | Compact path stays within workload cycle/throughput budget; any regression is explicit and accepted. |
| Compatibility | Existing workspace tests pass unchanged except for intentional new coverage; raw consumers remain unaffected. |

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Varint decoding costs more code/cycles than it saves. | Keep a fixed-word control implementation and choose per deployment only from measured total flash/RAM/cycle results. |
| Loop tables dominate artifact bytes. | Measure first; compare absolute, signed-delta, and template-relative table codecs. Preserve simple absolute slots as baseline. |
| Validation doubles parsing cost or needs allocations. | Validate once into bounded offsets/counts, then execute many times; make cold validation code feature-separable only if safe. |
| Inputs/slot reuse cause uninitialized reads. | Re-check def-before-use, input immutability, and effect-placeholder non-use in transpiler and decoder; test compacted schedules heavily. |
| Storage identity/width or initialization lifecycle drifts from Boolar. | Declare canonical `(storage, lane, address_bits)` banks and ordered init segments; require matching caller banks at admission and differential event-log tests for one-shot/persistent execution. |
| Storage bytecode accidentally assumes dense memory or allocates per operation. | Route exclusively through caller-owned `ContextWithStorage` banks and one reusable bounded address workspace; test dense, sparse, and native-context implementations. |
| External names/metadata erase semantics. | Use an explicit versioned manifest mapping and transcript tests; reject unresolved IDs. |
| Format growth recreates a general VM. | Enforce v1 non-goals and require a reference executor, measured use case, and opcode/version review for each extension. |
| Rust code footprint grows through generic abstractions/error formatting. | Keep hot decoder/interpreter monomorphic where measurement supports it, use compact error enums, gate host-only writer/diagnostics, and inspect Thumb map/section output in CI. |
| Artifact is treated as a trusted proof/circuit identity. | Keep executable bytes distinct from canonical `Program`/circuit statement identity; any hash/proof binding specifies its own canonical source representation and version. |

## 9. Source notes

1. James R. Bell, “[Threaded Code](https://figforth.org.uk/library/Threaded.Code.p370-bell.pdf),” *Communications of the ACM* 16(6), 1973, pp. 370–372. Direct historical source for threaded links, parameter-after-link layout, and reported space/time comparisons.
2. John James, “[Threaded Code](https://www.forth.org/fd/FD-V01N2.pdf),” *FORTH Dimensions* 1(2), 1978, pp. 17–18. Contemporary indirect-threaded Forth description, dictionary separation, and 6 KiB claim; historical context rather than a modern performance guarantee.
3. UCSD p-System, *[Internal Architecture Software Library](https://bitsavers.org/pdf/ti/professional/p-system/2232400-0001_UCSD_p-System_Internal_Architecture_Software_Library_Apr1983.pdf)* (1983). Historical portable p-code architecture reference. Fetch access was unavailable during this research; verify exact instruction-format claims against the manual before citing them as normative.
4. Oracle, *[Java Virtual Machine Specification §2.11](https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-2.html#jvms-2.11)* and *[§6](https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-6.html)* (Java SE 21). One-byte opcode/unaligned compactness rationale, implicit operands, specialized forms, and link-time verification.
5. Lua 5.4, *[`lopcodes.h`](https://github.com/lua/lua/blob/v5.4.8/lopcodes.h)*. Direct definition of the 32-bit `iABC`/`iABx`/`iAsBx`/`iAx`/`isJ` layouts and `EXTRAARG` convention.
6. WebAssembly Community Group, *[Binary values](https://webassembly.github.io/spec/core/binary/values.html)* and *[Binary instructions](https://webassembly.github.io/spec/core/binary/instructions.html)*. Normative LEB128 bounds, opcode/immediate layout, and `end`/`else` structured control delimiters.
7. Bytecode Alliance, *[wasm-micro-runtime fast interpreter](https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/core/iwasm/interpreter/wasm_interp_fast.c)* and *[loader](https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/core/iwasm/interpreter/wasm_loader.c)*. Production source showing separate loading/preparation, bounded LEB decoding, and switch/computed-goto interpreter dispatch alternatives.
8. Android Open Source Project, *[DEX format](https://source.android.com/docs/core/runtime/dex-format)* and *[Dalvik bytecode](https://source.android.com/docs/core/runtime/dalvik-bytecode)*. Primary documentation for LEB128 metadata and a fixed-frame register VM with narrow/wide operand forms.
9. `cirrus-recompile-core/src/lib.rs`, `cirrus-recompile-rt/src/lib.rs`, `cirrus-core/src/lib.rs`, `cirrus-recompile-core/src/typed.rs`, and `cirrus-volar-boolar/src/lib.rs` in this repository. Authoritative current Program/runtime/storage seams and Boolar storage lifecycle.

## 10. Decisions required before coding

1. Set concrete Thumb flash/RAM/cycle budgets and identify the exact firmware
   harness that owns static artifacts and external manifests.
2. Confirm whether compact artifacts are host-transpiled only in v1 or whether
   an `alloc` transpiler is also required on a non-embedded target.
3. Choose the external-manifest identity: numeric deployment-local IDs in v1
   (recommended) or a separately specified stable digest/name scheme.
4. Decide whether v1 accepts only `PreparedProgram` or also transparently
   prepares a raw `Program`. Recommended: offer a host convenience wrapper,
   but make the actual lowering input a validated prepared program so the
   optimization boundary stays explicit.
5. Approve baseline varint-plus-absolute-table encoding as the first measured
   format, with no short forms or delta tables until the Phase 5 report.
6. Approve the Boolean Program storage contract in §2.1, especially the
   non-readable effect placeholder, canonical logical bank identity, known-bit
   sidecar, and `pre_init` one-shot/persistent lifecycle.
7. Decide the ownership location and feature boundary for the Boolar adapter,
   while keeping `cirrus-recompile-core` independent of Volar types.


