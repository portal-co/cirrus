# Plan: Volar BinFHE V2 scheduling through generic variant bytecode

> **Status:** integration design and implementation sequence. This does not
> validate BinFHE V2 parameters, make a cryptographic-security claim, or make
> either repository's WIP API stable.
>
> **Source snapshot:** Volar `727ee67be2cd572090cb4d098b60ef13b18da597`
> (`Add BinFheScheme weaver, cone-fusion plan builder, and plan codegen`). Its
> working tree was dirty when this plan was written; the commit, not the
> worktree, is the normative source snapshot.
>
> **Decision:** retain Volar as the sole producer of FHE fusion and scheduling.
> Transcode its validated `BootstrapPlan` plus explicit source bindings into a
> canonical, generic **variant bytecode** carried beside CRBC. The Cirrus
> interpreter selects a caller-supplied operation set for that variant; it does
> not embed FHE keys, ciphertexts, function pointers, or a second scheduler.

## 1. Goal

Allow a Cirrus compact executable to run a pure Boolean region using a
precomputed BinFHE V2 schedule rather than naively replaying every Boolean
operation. The selected schedule may replace a cone with a multi-input LUT/PBS,
reuse circuit-bootstrap outputs, and preserve the weaver's topological layer
boundaries. The same executable remains understandable by a baseline operation
set for deterministic tests, but it must never silently substitute a different
FHE plan or fall back after partially executing one.

This is a **variant** of a base computation, not a new source IR:

```text
Volar Boolar circuit
  ├─ normal Boolar → Program → PreparedProgram → CRBC base executable
  └─ BinFheScheme scheduling → BootstrapPlan
          └─ validated adapter + source-slot binding → CRBV variant payload

CRBC container + CRBV payload + caller-selected operation set
  ├─ baseline Boolean operation set (test oracle)
  ├─ BinFHE Toy operation set (real V2 reference operations)
  └─ future profile-specific operation sets, only after their own gates
```

The target behavior is that the plan is executable **as data** in three
independent ways:

1. Volar's existing `volar_spec::binfhe::plan::execute_plan` is the V2
   cryptographic reference executor for a `BootstrapPlan`.
2. A Cirrus `VariantOperationSet` baseline executes the generic CRBV records
   serially and deterministically over a clear Boolean model.
3. A profile-specific BinFHE operation set invokes the corresponding V2
   primitives (`binfhe_lut_read_dyn`, circuit bootstrap, and RGSW CMUX) in the
   schedule prescribed by CRBV.

The V2 baseline is not a claim that parallel operations run concurrently:
within each layer its first implementation executes records in canonical byte
order. A later parallel backend may run independent records concurrently only
when it preserves each layer's read-before-write boundary and returns the same
arena bindings and outputs.

## 2. Volar V2 facts that constrain the integration

| Volar source | Current fact | Integration consequence |
| --- | --- | --- |
| `volar_spec::binfhe::plan::BootstrapPlan` | Holds `ProfileId`, `k_max`, logical LUTs, topologically ordered `layers`, input/output arena sizes, and a failure budget. `validate()` checks table shape, arena/topological references, output IDs, and budget consistency. | Call `BootstrapPlan::validate()` in the Volar-side adapter before any CRBV output. CRBV repeats required structural checks; it never trusts a deserialized plan solely because it has a hash. |
| `PlanOp::{Const, Not, Lut, CircuitBootstrap, RgswMux}` | V2 has three value arenas: LWE Boolean wire IDs, RGSW IDs, and RLWE cell IDs. A `Lut` consumes LSB-first wire IDs and a logical LUT; circuit bootstrap produces RGSW; `RgswMux` writes a cell. | Generic variant bytecode needs typed arenas, not only CRBC's one Boolean scratch array. A one-type `ContextWithBitAnd` extension is insufficient. |
| `BootstrapPlan::layers` | Operations are topological and operations within a layer are independent. The weaver derives layers from wire dependencies. | Encode explicit `LAYER_END` delimiters or record layer lengths. The interpreter may not move a record into another layer or observe a later layer before the earlier layer completes. |
| `BootstrapPlan::plan_hash()` | Provides a deterministic FNV-1a identifier for test evidence. | Carry it as diagnostic provenance only. Bind deployment selection to a cryptographic digest of canonical CRBV bytes plus source binding, not a non-cryptographic 64-bit convenience hash. |
| `binfhe_lut_read_dyn` | Is the common runtime-table LUT primitive already used by V2's plan executor and generated code; callers validate LUT shape first. | The BinFHE operation set calls this existing primitive for `LUT`; it must not rebuild selector math, table shape logic, or PBS scheduling in Cirrus. |
| `fhe_binfhe::build_bootstrap_plan` | The weaver performs cone fusion and materialization; storage reads/writes are currently explicit unsupported barriers. | Do not duplicate cone discovery in Cirrus. V1 variant regions are pure Boolean only and cannot cross a storage/external/RNG/action/control boundary. |
| V2 plan/docs status | V2 is marked unpinned and very unstable. `Toy`/`ToyNoisy` are test profiles; `Std128` remains gated on parameter/failure evidence. | Start with a Toy operation set only. A Std128 descriptor is rejected until Volar's V2 gate explicitly admits it. |

### 2.1 Existing Cirrus state

Cirrus now has:

- a Boolean `Program`/`PreparedProgram` with storage-bank declarations,
  static-initialization segments, `Storage` operations, and effect-placeholder
  validation;
- caller-owned raw/prepared storage executors and a direct Boolar lowering;
- `cirrus-recompile-bytecode` CRBC framing, canonical bounded integers,
  base Boolean records, and storage records; and
- no generic multi-arena variant interpreter or schedule payload yet.

The variant feature must be an additive `cirrus-recompile-bytecode` module or a
sibling crate. It cannot make `cirrus-recompile-core` depend on `volar-spec`:
Volar identifiers and concrete BinFHE ciphertext types stay behind the Volar
adapter and caller-supplied operation set boundary.

## 3. Invariants and non-goals

### Required invariants

1. **One scheduler.** Only Volar produces `BootstrapPlan` fusion/layering. The
   Cirrus adapter serializes a checked plan; it does not rediscover cones,
   reorder plan ops, coalesce LUTs, or recalculate failure budgets.
2. **Exact source binding.** A CRBV payload includes the canonical source-region
   identity, ordered imported base slots, ordered exported base slots, profile,
   `k_max`, LUT data, schedule bytes, and binding digest. It is invalid on any
   CRBC base executable other than the one it was built for.
3. **No effect crossing.** A variant region has no storage read/write, external
   call, RNG, action, control boundary, or live internal source value escaping
   except its declared exports. Storage and action positions remain base-CRBC
   barriers until a separate FHE storage-plan model exists.
4. **Typed arena safety.** Every operand/result carries an arena tag (`Wire`,
   `Rgsw`, or `Cell`). Validation ensures append-only IDs, type-correct use,
   layer topology, LUT arity, and variant output binding before operation-set
   dispatch.
5. **Capability binding.** The supplied operation set declares exactly one
   variant kind/version/profile family and supported opcode set. The
   interpreter rejects a mismatch before it creates any output value.
6. **No partial fallback.** Once variant execution starts, any operation-set
   failure aborts execution. The caller may select base CRBC *before* running,
   but cannot fall back after a partly executed FHE plan.
7. **Canonical bytes.** CRBV uses canonical shortest unsigned LEB128,
   fixed section ordering, strict lengths, sorted/deduplicated tables, and one
   representation of each schedule. Decoder rejects unknown flags/opcodes and
   noncanonical integers.
8. **No secret serialization.** CRBV contains public schedule metadata and
   public logical LUT entries only. FHE ciphertext inputs, keys, bootstrapping
   keys, circuit-bootstrap keys, and runtime cells are caller-owned values.

### Non-goals

- Replacing `BootstrapPlan`, moving its scheduling algorithm into Cirrus, or
  making FHE V2 stable/security-approved.
- Treating `ProfileId::Std128` as an approval to select production parameters.
- General FHE storage, action calls, generic arbitrary opcodes, GPU scheduling,
  or parallel execution in the initial integration.
- Using raw Rust enum layout, `rkyv` bytes, `TypeId`, type names, or function
  pointers as the variant wire format.


## 4. Generic variant bytecode (`CRBV`)

CRBV is a length-delimited section inside a future CRBC container. It is
**generic** in the sense that the interpreter only knows arenas, records,
imports/exports, and capability identifiers; semantic meaning comes from the
operation set selected by the caller. It is not a generic dynamically typed VM.

### 4.1 Envelope and source binding

```text
variant_kind:uLEB       = 1 for BinFHE V2 bootstrap plans
variant_version:uLEB    = 1
profile:uLEB            = Toy | ToyNoisy | Std128 | Custom (Volar mapping)
source_digest[32]       = BLAKE3(CRBC base canonical region binding)
plan_digest[32]         = BLAKE3(canonical CRBV payload after this header)
plan_hash:u64le         = copied Volar diagnostic BootstrapPlan::plan_hash()
k_max:uLEB
wire_imports: slot-list
cell_imports: slot-list
wire_exports: slot-list
cell_exports: slot-list
lut_section: length-delimited
layer_section: length-delimited
```

`source_digest` covers the CRBC format version, source entry/statement region,
base bank/layout declarations relevant to the region, imported source slots,
exported source slots, and the selected variant kind/version. The host computes
it after base transpilation; the target recomputes it from its validated
`CompactProgram` view before admitting CRBV. Do not rely on a byte offset alone:
format revisions, a different region, or a different binding list must not
accidentally select the same FHE schedule.

The integration requires a cryptographic digest implementation approved for the
embedded footprint. Until BLAKE3 is accepted in the target profile, CRBV stays
host/test-only; FNV-1a remains diagnostic only.

### 4.2 Typed arena model

```text
Arena 0: Wire  — encrypted/clear Boolean values; inputs are `wire_imports`
Arena 1: Rgsw  — circuit-bootstrap results; no base-slot imports in v1
Arena 2: Cell  — RLWE content cells; inputs are `cell_imports`
```

Wire and cell imports map the ordered plan input arenas to base CRBC slots or
caller-declared storage-cell bindings. In the first Cirrus integration,
`cell_imports` must be empty because current `Program` storage is Boolean-wire
storage, not the V2 RLWE-cell model. Retain the field now so the generic format
does not need an incompatible redesign for `RgswMux` support.

A value ID is implicit append-only position within its arena, exactly as in
Volar `BootstrapPlan`. Validation initializes arena lengths from import counts;
each record's output ID must equal the current length of its destination arena.
This is both smaller than explicit IDs and detects reordering/corruption.

### 4.3 Records and layers

```text
LAYER_BEGIN record_count:uLEB
  CONST          out:implicit, value:u1
  NOT            input:wire-id, out:implicit-wire
  LUT            input_count:uLEB, inputs[wire-id], lut_id:uLEB, out:implicit-wire
  CIRCUIT_BOOT   input:wire-id, out:implicit-rgsw
  RGSW_MUX       selector:rgsw-id, then:cell-id, else:cell-id, out:implicit-cell
LAYER_END
```

Every `LAYER_BEGIN` has exactly `record_count` records and is followed by
`LAYER_END`; the redundant delimiter supports streaming diagnostics and makes
truncation fail closed. Empty layers are rejected. Layer order, record order,
and implicit output ID order are canonical. `LUT` inputs are LSB-first,
matching `BootstrapPlan` and `binfhe_lut_read_dyn`.

The LUT section stores each logical table once:

```text
lut_count:uLEB
for every LUT in source order:
  arity:uLEB
  bit_count:uLEB = 1 << arity
  packed_entries[ceil(bit_count / 8)]  // LSB-first
```

Validate non-empty power-of-two shape, exact `bit_count`, zero unused bits,
`arity <= k_max`, and input-count equality. A constant LUT is still explicit;
the operation set may execute it through the plan's normal trivial-encryption
semantics, but it must not alter arena IDs or delete the record.

### 4.4 Variant operation-set interface

The bytecode crate defines an object-safe-by-construction, generic trait such
as:

```rust
pub trait VariantOperationSet {
    type Wire: Clone;
    type Rgsw;
    type Cell;
    type Error;

    const KIND: u32;
    const VERSION: u32;
    fn admit(&mut self, profile: VariantProfile, k_max: u32) -> Result<(), Self::Error>;
    fn constant(&mut self, value: bool) -> Result<Self::Wire, Self::Error>;
    fn not(&mut self, input: Self::Wire) -> Result<Self::Wire, Self::Error>;
    fn lut(&mut self, inputs: &[Self::Wire], entries: &[bool], k_max: u32)
        -> Result<Self::Wire, Self::Error>;
    fn circuit_bootstrap(&mut self, input: Self::Wire) -> Result<Self::Rgsw, Self::Error>;
    fn rgsw_mux(&mut self, selector: Self::Rgsw, then: Self::Cell, r#else: Self::Cell)
        -> Result<Self::Cell, Self::Error>;
}
```

The actual API should use caller-provided bounded arena storage instead of
`Vec` in the target path. The trait is a semantic seam, not permission for the
operation set to change scheduling: its calls occur exactly in decoded record
order. The interpreter owns arena ID binding, validation, exports, and layer
barriers; the operation set owns only the primitive implementation.

Provide these baseline implementations:

| Set | Purpose | Allowed profile |
| --- | --- | --- |
| `ClearVariantOps` | Plain Boolean `Wire=bool`, clear `Rgsw=bool`, clear cell model; exhaustive/minimal schedule oracle. | Test profiles only. |
| `BinFheToyOps` | Calls V2 Toy primitives using Volar's `binfhe_lut_read_dyn`, circuit bootstrap, and CMUX. | `Toy` only. |
| `BinFheToyNoisyOps` | Same operation surface with seeded noisy profile and budget observation. | `ToyNoisy` only. |
| `BinFheStd128Ops` | Deferred. | Reject until Volar’s V2 M8/parameter gate and a Cirrus deployment review approve it. |

## 5. Scheduling and interpretation flow

### 5.1 Host compilation flow

1. Lower the source through Volar's normal Boolar path and create the base
   Cirrus `Program`/CRBC artifact. Preserve existing storage/external metadata.
2. Ask Volar `fhe_binfhe::build_bootstrap_plan` for eligible pure regions.
   The adapter must receive a region map from source Boolar variables to base
   Program slots; it cannot infer correspondence from numerical IDs.
3. Call `BootstrapPlan::validate()`, then independently validate all source
   bindings: imports correspond to region leaves, exports correspond to region
   outputs, no non-exported region value remains live, and no effect/control
   boundary is crossed.
4. Canonically serialize the plan into CRBV, recompute the adapter's LUT/layer
   counts and failure-budget relation, bind its source digest, and store it as
   an optional CRBC variant section.
5. Generate test evidence containing base digest, CRBV digest, Volar plan hash,
   profile, `k_max`, bootstrap count, layer count, LUT count, imports/exports,
   and failure budget. This is provenance, not a security certificate.

### 5.2 Target admission flow

1. Validate CRBC and all candidate CRBV sections without executing operations.
2. Compare CRBV `kind/version/profile` with the supplied operation set; call
   `admit`; recompute and compare the source digest; enforce caller resource
   caps for section bytes, layers, records, LUT bits, arena capacities, and
   maximum LUT arity.
3. Select exactly one variant before execution. Selection policy is caller
   owned and explicit: e.g. `PreferBinFheToy`, `RequireDigest`, or `BaseOnly`.
4. Allocate/borrow fixed arena storage according to validated maximum counts.
   The target never allocates `Vec`s from untrusted counts.

### 5.3 Target execution flow

```text
for layer in canonical CRBV order:
    decode exactly record_count records
    execute each record through VariantOperationSet
    assert all outputs append to the expected arena
    cross layer boundary only after every record returned success
copy declared Wire/Cell exports to their bound CRBC destinations
resume base bytecode after the region
```

For initial integration, a variant replaces an entire base entry region. This
avoids introducing arbitrary CRBC jump targets before loop/region replacement
is designed. Once stable, add `VARIANT_CALL` to CRBC with an explicit entry
range and post-dominator resume offset; it must be validated as a structured
call-like region, not a free jump.

A partial-region variant is allowed only after all of these are serialized and
validated: base statement range, base resume range, import/export slot lists,
no-crossing liveness proof, and storage/external barrier proof. The interpreter
must write exports only after the corresponding final layer succeeds; however,
FHE operation sets may have mutable internal state, so errors remain terminal
rather than rollback candidates.

## 6. Work plan and commit boundaries

### Phase A — contracts and adapters (Volar-first)

1. In Volar, add a canonical `BootstrapPlan` encoder/decoder or a dedicated
   adapter-facing view. It must be separate from `rkyv` and cover all fields
   consumed by CRBV. Add deterministic round-trip and hash/golden tests.
2. Add a `PlanRegionBinding` result from the weaver: source Boolar IDs,
   ordered imports/exports, source-region identity, profile, and explicit
   effect/control barrier result. Do not expose a plan without a binding.
3. Add a Volar baseline plan interpreter trait that can run `PlanOp` over
   clear booleans independently of the BinFHE cryptographic executor.
4. Commit in Volar separately, with its existing V2 marker/status rules.

### Phase B — CRBV format and clear interpreter (Cirrus)

1. Add a `variant` module to `cirrus-recompile-bytecode`; retain CRBC as base
   framing and make CRBV an optional, length-delimited section.
2. Implement canonical writer, borrowed validator, source-digest binding, typed
   arena validation, LUT packing, layer record validation, and resource limits.
3. Implement `ClearVariantOps` and differential tests against Volar's clear
   plan interpreter for constants, NOT, all LUT arities admitted by V2 Toy,
   circuit-bootstrap identity semantics, and RGSW mux.
4. Commit this format/runtime phase independently. It must compile `no_std`; a
   host-only writer may use `alloc`.

### Phase C — BinFHE Toy operation set

1. Add an optional Cirrus integration crate depending on the pinned Volar V2
   spec API. It implements `BinFheToyOps` by delegating to existing V2 public
   primitives; no cryptographic equations are copied.
2. Add adapter round-trip tests: `BootstrapPlan → CRBV → validate →
   BinFheToyOps`, compared with Volar `execute_plan` on fixed deterministic
   keys/randomness and clear expected outputs.
3. Add generated-weaver vs CRBV-vs-`execute_plan` three-way differential
   tests for a one-LUT cone, a multi-layer cone, and a circuit-bootstrap/RGSW
   mux plan.
4. Commit separately. Mark it test-only/unstable and reject non-Toy profiles.

### Phase D — base-CRBC replacement integration

1. Add whole-entry variant selection plus explicit base/variant execution
   policy. Keep normal CRBC execution as a supported fallback chosen *before*
   execution.
2. Add effect/storage/external barrier tests proving a plan cannot cover such
   statements. Add source-digest mismatch and capability/profile mismatch tests.
3. Only after these pass, design structured partial-region `VARIANT_CALL` and
   its liveness proof format.

### Phase E — measurements and profile gates

1. Record base CRBC bytes, CRBV bytes, interpreter `.text`, arena RAM,
   bootstrap count, layer count, and wall-clock/cycle cost for clear and Toy
   operation sets. Compare fused schedule versus base gate-by-gate execution.
2. Run the existing Thumb artifact-size pipeline for the generic variant parser
   and clear operation set only; the full BinFHE V2 operation set is not an MCU
   admission claim without an explicit resource review.
3. Admit `ToyNoisy` only after deterministic failure-budget transcript tests.
   Admit `Std128` only after Volar M8 evidence and a separate Cirrus review.

## 7. Test matrix

| Test | Independent oracle / expected result |
| --- | --- |
| CRBV canonical encode/decode | Canonical re-encoding byte equality; reject redundant LEBs, bad lengths, unknown tags, malformed LUT packing. |
| Arena/topology validation | Hand-built invalid plans: future IDs, wrong implicit output, cross-arena ID, duplicate export, layer dependency violation. |
| Source binding | Change CRBC base bytes, source slot order, region boundaries, or imports/exports; admission must reject before operations run. |
| Clear schedule semantics | Exhaustive Boolean inputs for Toy-sized plans; compare CRBV `ClearVariantOps` to Volar's independent clear plan evaluator and direct Boolean truth tables. |
| V2 Toy differential | Fixed deterministic corpus: Volar `execute_plan`, generated BinFhe weaver code, and CRBV `BinFheToyOps` agree on decrypted outputs. |
| Layer boundary | Instrumented operation set records calls; records occur in canonical layer/order and later layer reads cannot occur early. |
| Operation-set capability | Wrong kind/version/profile/opcode set rejects before primitive calls. |
| Effect barrier | Plans spanning storage, action, RNG, oracle, or CFG boundaries fail at Volar binding construction and Cirrus validation. |
| Failure path | Operation-set error stops immediately; no base fallback or later layer record runs. |
| Footprint | Reproducible release `.text/.rodata/.bss`, CRBV bytes, arena maxima, bootstrap count, and comparison to unfused base. |

## 8. Risks and decisions required

| Risk | Mitigation |
| --- | --- |
| Duplicated scheduling diverges from Volar. | Volar produces the only plan; Cirrus validates/executes serialized records only. |
| Plan hash is mistaken for cryptographic binding. | Include a cryptographic source/variant digest; label Volar's FNV hash diagnostic only. |
| Generic variant turns into an unbounded VM. | Fixed kind/version, three known arenas, closed opcode table, capability-gated operation set, bounded sections. |
| FHE profile selected without evidence. | `BinFheToyOps` only initially; reject Std128 until both repositories record explicit admission. |
| Variant crosses storage/effect boundary. | Region binding carries an effect barrier proof; v1 is pure Boolean and whole-entry only. |
| Parallel execution changes observable state. | Serial canonical baseline; parallel implementation must be a separately tested operation-set capability. |
| CRBV duplicates large LUT tables. | Deduplicate only in the Volar plan producer; serialize LUT source order and measure before introducing compression. |

Before implementation, decide:

1. Which cryptographic digest is acceptable for embedded CRBV source binding.
2. Whether initial whole-entry variants are sufficient, or whether a concrete
   first workload requires structured partial-region `VARIANT_CALL`.
3. The Volar API location for `PlanRegionBinding` and canonical plan encoding.
4. The initial baseline clear semantics for `CircuitBootstrap`/`RgswMux`; they
   must be explicit and independently testable, not implicit coercions.
5. Resource caps and deployment ownership for Toy operation-set artifacts.

## 9. Sources

- Volar `docs/fhe/binfhe-v2-implementation-plan.md`, §§2, 5, 7–8, 10;
  snapshot `727ee67`. Defines V2 status, one scheduler/data plan decision,
  `BootstrapPlan`, V2 milestones, and security gates.
- Volar `crates/spec/volar-spec/src/binfhe/plan.rs`; snapshot `727ee67`.
  Primary code source for `PlanOp`, typed arenas, validation, layers, hash, and
  `execute_plan` reference semantics.
- Volar `crates/compiler/volar-weaver/src/fhe_binfhe.rs`; snapshot `727ee67`.
  Primary source for fusion, scheduling, unsupported effect barriers, and
  generated-code consumption.
- Volar `crates/spec/volar-spec/src/binfhe/pbs.rs`; snapshot `727ee67`.
  Primary source for `binfhe_lut_read_dyn` shared runtime-table LUT semantics.
- Cirrus `crates/recompile/space-efficient-interpreter-plan.md` and
  `crates/recompile/cirrus-recompile-bytecode/src/lib.rs`. Existing base
  executable/storage constraints that CRBV extends.

