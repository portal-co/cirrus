# ERT Arm roadmap: Thumb hooks, shared loop driver, and AArch64

**Status: Phases 1–2 core implementation complete; Phase 3 AArch64 decoder/control-state foundation underway.**

## Delivery record

### Phase 1 — Armv8-M call hooks — **COMPLETE**

* `ArmCallEvent` / `ArmCallAction` and the defaulted
  `ArmHandler::call_hook` now cover direct calls, register calls (including
  unresolved targets), conventional returns, and `BXNS`/`BLXNS`.
* `ArmDefaultHandler` preserves the new hook tunnel. `ReturnNow` runs before
  the LR/private-rstack commit; return `ReturnNow` remains fail-closed; all
  diverted targets are revalidated in their native transfer path.
* The optional `call-hooks` feature adds the allocator-using
  `ArmCallRegistry` and register-only `NoOp`/`ReturnConstants` replacements.
  The synchronous hook remains available in the default no-alloc build.

### Phase 2 — shared driver substrate — **CORE IMPLEMENTATION COMPLETE**

* `cirrus-ert-loop-core` owns the no-alloc, caller-buffered candidate
  generation lifecycle (deduplication, capacity failure, tail compaction), the
  common predicated-value circuit `old ^ (active & (new ^ old))`, and a sealed
  callback-style generation driver. The callback owns ISA state/folding while
  the core commits successors only after every body has succeeded.
* `cirrus-ert-loop` consumes both primitives without behavior changes; its
  all-feature unit suite and RV64 SHA workload are regression gates.
* `cirrus-ert-loop` now uses that callback boundary for its multi-candidate
  RISC-V path. The adapter owns RISC-V snapshots/body execution/metadata and
  state folding; the core owns only stable successor collection and atomic
  candidate-table commit. All existing loop and RV64 SHA regression gates pass.
* `cirrus-armv8m-ert-loop` now defines the fixed-capacity Thumb boundary
  snapshot, register-fold primitive, and fail-closed concrete agreement
  contract. It retains all 16 register words/metadata, lazy NZCVQ flags,
  PC/SP, rstack frames, ITSTATE, and virtual TrustZone state. The Arm facade
  exposes only doc-hidden decoded execution/state seam types needed to
  populate it.
* `cirrus-armv8m-ert-loop` now has a caller-owned `execute_snapshot` body
  runner. It restores a complete `ThumbSnapshot`, executes through the Arm
  interpreter's decoded seam, and captures the updated state plus a symbolic
  `B<cond>`/`CBZ`/`CBNZ` boundary or exit. No allocator or hidden
  storage/rstack state is introduced.
* A public no-alloc `initial_snapshot`/`step` API now initializes the Thumb
  state and advances one shared-scheduler generation while exposing snapshot,
  virtual-IP, completion, and structural-exit state. The end-to-end plaintext
  gate drives a symbolic `BNE` through two candidate generations to `SVC #0`.
  It verifies candidate order, exit completion, and done-wire behavior.
* Phase 2's core adapter implementation is complete. Follow-up Phase 2
  validation expands this gate to the remaining Thumb branch encodings,
  concrete fixture equivalence, predicated stack writes, and recorder replay;
  those tests do not change the state/scheduler seam.

### Phase 3 — AArch64 facade — **IN PROGRESS**

* Added `cirrus-aarch64-ert`, a separate no-alloc facade with
  `disarm64 0.1.26` pinned without default features and enabled only as a
  decode classification guard.
* The initial audited raw-mask allowlist covers `B`/`BL`, `B.cond`,
  `CBZ`/`CBNZ`, `TBZ`/`TBNZ`, `BR`/`BLR`, canonical `RET X30`, and bare-metal
  `SVC #0`. It extracts PC-relative targets locally, rejects unlisted decoded
  forms, checks PC alignment, and rejects reserved condition/register forms.
* The first symbolic state has 31 separate 64-bit GPR words, concrete metadata,
  separate SP, and a done wire. `CBZ`/`CBNZ` emits a virtual-IP select; the
  x31/XZR distinction is pinned by tests. Arithmetic, NZCV tracking, memory,
  AAPCS64 entry/exit, hooks, and the AArch64 loop adapter remain subsequent
  Phase 3 work.

This is the Arm counterpart to [`ert-looped-rv64-plan.md`](ert-looped-rv64-plan.md).
It deliberately separates three deliverables while designing their shared seams
up front:

1. call/return interception for the existing Armv8-M Thumb facade;
2. an architecture-neutral loop *driver* with RISC-V and Thumb adapters; and
3. a new AArch64 integer facade, decoded with `disarm64`, then made available
   to the same loop driver.

The order matters. The hook contract must be stable before loop adapters call
through it; the loop driver must become architecture-neutral before AArch64 is
added, otherwise two more copies of candidate folding and predicated storage
will drift.

## Research record

### Repository facts

| Area | Current implementation fact | Consequence |
|---|---|---|
| Thumb facade | `cirrus-armv8m-ert/src/lib.rs` is one 3.6k-line module: private `Machine`, `Runtime`, `Flow`, `Op`, decoding, and handlers are colocated. It has 16 `[W; 32]` registers, `u32` PC/SP/rstack, five lazy `Flag<W>` values (NZCVQ), `itstate`, and a virtual `SecurityState`. | Do **not** copy the module into a loop crate. Extract a deliberately doc-hidden execution seam first, retaining the current public API. |
| Thumb transfers | `Op::{Branch, CompareBranch, Call, CallRegister, BranchRegister, BranchExchangeNonSecure, SecureGateway}` are dispatched in `Machine::execute`; `call`, `branch_register`, and `return_from_call` own rstack/LR handling. `BLXNS`/`BXNS` additionally change `SecurityState`; `SG` changes state only under the current attribution policy. | The loop state must include flags, IT state, virtual security state, and rstack agreement — registers alone are unsound. |
| Existing RISC-V looper | `cirrus-ert-loop` has a no-alloc live-candidate scheduler, virtual IP/done wires, predicated writes, fold logic, bounded indirect targets, and an optional `alloc` precompute map. Its adapter is currently coupled to RISC-V `Machine`, `Inst`, 32 registers, and `RvHandler`. | Preserve it as the compatibility crate, but move only the scheduler/storage-fold substrate to a new architecture-neutral crate. |
| Shared core | `cirrus-ert-core` already provides width-generic Boolean arithmetic, comparison, selection, shifts, `RawMemory`, `Handler`, and `EcallOutcome`. | It is the correct home for truly architecture-neutral data circuits and possibly an architecture-neutral transfer **action**, but not for Arm/RISC-V register names or decoded instruction enums. |
| A64 decoder | Installed source is `disarm64 0.1.26` at `$CARGO_HOME/registry/.../disarm64-0.1.26`. `src/lib.rs` is `#![no_std]`; `decoder::decode(u32)` is available with its `full` feature. `Opcode` exposes a mnemonic/operation classification but decoding itself does not calculate a PC-relative target; its own formatter documents that PC is not used for PC-relative addressing. | Depend on `disarm64` with `default-features = false, features = ["full"]`; use it as the decode-classification guard, then derive PC-relative targets and semantic fields from the original `u32` under ERT-owned, audited masks. Never treat a successfully decoded instruction as supported. |

### Primary sources

* Existing implementation: `crates/ert/cirrus-armv8m-ert/src/lib.rs`, especially
  `ArmHandler`, `StorageRuntime`, `Machine::{run,decode,execute,call,branch_register,
  return_from_call,branch_exchange_non_secure}`; and
  `crates/ert/cirrus-ert-loop/src/lib.rs`.
* Arm ABI: [AAPCS64, Arm ABI repository](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst), sections “General-purpose registers”, “The stack”, and “Subroutine calls”. It specifies 31 GPRs (`x0`–`x30`), `SP`, arguments/results in `x0`–`x7`, `x30` as LR, `BL`/`BLR` link behaviour, and `SP mod 16 = 0` at accesses/public interfaces.
* Arm architectural reference: [Arm A-profile Architecture Reference Manual, DDI0487](https://developer.arm.com/documentation/ddi0487/latest/) for A64 instruction semantics; retain an offline revision/section reference in the implementation comments for each hand-extracted field.
* Decoder: `disarm64` 0.1.26 local `src/lib.rs`, `src/decoder_full.rs`, and the upstream [disarm64 README](https://github.com/kromych/disarm64/blob/master/Readme.md). Its generated instruction table includes the required `BRANCH_IMM`, `BRANCH_REG`, `CONDBRANCH`, `COMPBRANCH`, `TESTBRANCH`, ALU, and load/store classes.

## Non-negotiable invariants

* Default builds stay `#![no_std]` and do not link `alloc`. Candidate arrays,
  rstack, register state, and storage remain caller owned.
* Unsupported encodings and non-modelled state fail closed with `Unexpected`.
  No “best effort” interpretation of an instruction that `disarm64` happens to
  recognize.
* A loop boundary may not discard architectural state. At a merge, all
  concrete-only state must agree; wire-backed state is selected under the
  candidate activity predicate. In particular, Arm NZCV(Q), ITSTATE, Thumb
  security state, LR/rstack depth and entries must be included in the rule.
* A hook may observe or replace a transfer but cannot smuggle an unchecked
  target, privilege/security-state transition, or host-only secret result into
  an emitted circuit. Diverted targets always pass the same ISA alignment,
  mapping, attribution, and rstack checks as native transfers.
* The public `ert_func`/`ert_emit` APIs remain source-compatible. New seams are
  `#[doc(hidden)]`; high-level loop APIs live in their adapters, not in a
  public “universal machine” trait.

---

## Phase 1 — Armv8-M call hooks

### 1.1 Public contract

Add this default method to `ArmHandler<Val>`:

```rust
fn call_hook(
    &mut self,
    event: ArmCallEvent,
    regs: &mut [[Self::Wrapped; 32]],
    constants: &mut [Option<u32>],
    offsets: &mut [Option<i32>],
    zero: &Self::Wrapped,
    one: &Self::Wrapped,
) -> Result<ArmCallAction, Self::Error> {
    Ok(ArmCallAction::Proceed)
}
```

`ArmCallAction` deliberately mirrors RISC-V semantically but uses a `u32`
Thumb target:

* `Proceed` — preserve the existing transfer exactly.
* `ReturnNow` — replace a call only; the hook has already written output
  registers. Continue after the call; do not push the private rstack or change
  LR. On a return it is an error.
* `Divert(u32)` — transfer to a concrete alternate target after the native
  form’s validation. It is not an escape hatch for Arm-state code.

Keep `ArmCallEvent` architecture-specific rather than forcing Thumb register
numbers into RISC-V `CallEvent`:

* `DirectCall { caller_pc, target, return_pc }` (`BL`);
* `RegisterCall { caller_pc, register, target: Option<u32>, return_pc }`
  (`BLX Rm`/the supported register-call form); `None` means unresolved and
  lets a hook resolve it;
* `Return { from_pc, target }` (`BX LR`, `POP {..., pc}`, and every supported
  return-to-LR path);
* `NonSecureBranch { caller_pc, register, target: Option<u32>, link,
  state_before }` for `BXNS`/`BLXNS`.

`Divert` is sufficient for the first cut; do not add arbitrary register/state
mutators. The event distinguishes `BLXNS` because its target low bit changes
virtual security state. The implementation must compute security transition
from the **diverted** target through the normal `branch_exchange_non_secure`
path, never trust host-provided state.

### 1.2 Hook placement and precise semantics

1. Add the default method and the event/action types adjacent to `ArmHandler`.
   `ArmDefaultHandler` delegates it to its wrapped handler. Existing custom
   `ArmHandler` implementations compile unchanged because the method defaults.
2. Split `Machine::call` into “ask hook”, “validate/action”, and
   “commit link/rstack” helpers. Ask before pushing rstack/LR. `ReturnNow`
   therefore has no cleanup obligation.
3. Route `Op::Call` and resolved `Op::CallRegister` through that helper.
   For an unresolved register target, emit `RegisterCall { target: None }`
   before the historical `Unexpected`; `Proceed` still fails closed.
4. Route `branch_register(LR)` and PC-restoring `Pop` through a single
   `return_from_call` hook site. `Divert` replaces the rstack landing only
   after pop/depth validation. A non-LR `BX` remains a branch, not a hookable
   return, unless it is explicitly added later.
5. Route `BranchExchangeNonSecure` through `NonSecureBranch`; validate
   Secure→Non-secure eligibility, Thumb bit and attribution after any divert.
   `ReturnNow` is legal only when `link == true`; otherwise reject it.
6. Do **not** hook `SG`: it is a security gateway, not a call boundary.
   Its attribute/state checks must remain unconditional and concrete.

### 1.3 Optional registry

Mirror RISC-V’s `call-hooks` feature in `cirrus-armv8m-ert`:
`ArmCallRegistry` maps direct target PCs to canned `NoOp` and
`ReturnConstants` replacements. It is `alloc`-only; synchronous callbacks do
not need the feature. Keep `memcpy`, `memset`, and TrustZone gateway
replacements out of this phase: the current hook argument lacks a storage
handle and a policy-safe memory replacement needs one.

### 1.4 Tests

* observer sees exact caller/target/return PCs for `BL`, register call, `BX
  LR`, and `POP {...,pc}`;
* `ReturnNow` leaves callee marker untouched and does not consume rstack;
* direct and return `Divert` honour Thumb low-bit and target mapping checks;
* unresolved register call fails by default and succeeds only when hooked;
* `BLXNS`/`BXNS` tests prove the policy/state transition is evaluated after a
  divert; a secure-state mismatch fails closed;
* registry touches only registered direct calls;
* recording test proves hook-written Boolean outputs are recorded, while a
  documented concrete-only replacement remains concrete.

---

## Phase 2 — one loop-driver core, with a Thumb adapter

### 2.1 Crate split (do this before adding Thumb looping)

Create `cirrus-ert-loop-core`:

* `#![no_std]`, depends only on `cirrus-core` and `cirrus-ert-core`;
* owns virtual-IP equality/selection, done accumulation, candidate-table
  deduplication/overflow checks, deterministic `run(max_steps)`, and the
  predicated storage formula `old ^ (active & (new ^ old))`;
* has no RISC-V or Arm instruction enum, register name, decoder, ABI, or
  security policy.

Keep `cirrus-ert-loop` as the RISC-V adapter and preserve its public entry
points. Add `cirrus-armv8m-ert-loop` as the Thumb adapter. Both depend on the
core. This is a **shared driver**, not a forced shared ISA `Machine`.

Use a sealed, crate-private adapter trait in the core (or callback bundle), not
an expansive public trait:

```text
execute_body(state snapshot, normalized candidate PC, activity?)
  -> { state-after, continuation, concrete-agreement-state }
fold_state(active, body-after, accumulated-after)
append_successors(continuation, candidate table)
```

The driver needs only wire width, virtual-IP word construction/equality, clone
and fold operations supplied by the adapter. It must never inspect a register
index itself. The adapter retains the short-lived borrow of its ISA runtime,
which avoids exposing either interpreter’s private machine lifetime in the
public API.

### 2.2 Refactor the existing RISC-V adapter first

Before touching Thumb, port the current RISC-V looper unchanged in behaviour:

* retain `looped_ert_func`, `looped_ert64_func`, `IndirectTargets`, and
  precompute semantics;
* establish a byte-for-byte/plaintext equivalence test before/after the split
  for a concrete call loop, symbolic trip-count loop, predicated stack write,
  indirect target set, recorder replay, and real RV64 SHA image;
* keep the RISC-V tests as the core contract tests, so a future Arm change
  cannot silently alter candidate ordering or write predication.

### 2.3 Thumb execution seam and state model

Promote only the minimum existing Arm internals to `#[doc(hidden)]`:
`Runtime`, `StorageRuntime`-equivalent construction, `Machine`, decoded
operation/length, `Flow`, and body-level execution helpers. Do not expose the
hand decoder’s complete `Op` as a stable API.

A Thumb adapter snapshot must include:

* 16 32-bit register words plus `Option<u32>` constants and `Option<i32>`
  stack offsets;
* `sp`, `stack_top`, rstack depth and a fixed bound of in-flight rstack
  entries, as the RISC-V looper does;
* NZCVQ `Flag<W>` values, not merely their concrete values;
* `itstate` and `SecurityState`;
* normalized fetch PC (even `u32`) and virtual IP wires.

At a multi-candidate fold, select each register and each materialized/lazy flag
under `active`. Constants and offsets survive only when every continued body
agrees. `sp`, rstack depth/in-flight entries, `itstate`, and security state
must be equal across continued bodies or the adapter returns `Unexpected`.
This rejects secret-dependent TrustZone/IT control flow rather than creating an
unsound host-only selection. Exited paths contribute only to `done`.

### 2.4 Thumb boundaries and continuations

Intercept before ordinary `Machine::execute` only when existing concrete
metadata cannot decide a control transfer:

* `B<cond>`: derive the wire with the existing `condition_wire`, then select
  `{taken target, pc + decoded length}`.
* `CBZ`/`CBNZ`: compare the register to zero with the shared word comparator.
* A symbolic one-instruction IT value materializer remains inside the body:
  it is already data selection, not a control boundary. Any symbolic control
  transfer while ITSTATE is nonzero fails closed in the first cut.
* A symbolic `BLX`/`BX` can use a caller-declared bounded target table only
  after the Arm call-hook phase. Each declared value must be an odd Thumb
  pointer and pass the state/attribution policy. `BX LR` stays concrete-rstack
  only.
* `SVC #0` exit forms `done`; all other exceptions/SVC immediates fail closed.

Candidate PCs are stored normalized (even) in a `u64` table for the shared
core; link values retain Thumb bit zero set. The adapter validates that every
candidate fits `u32`, is halfword-aligned, maps in `RawMemory`, and is a Thumb
instruction boundary before execution.

### 2.5 Tests and gates

* Unit tests for symbolic `B<cond>` sourced from lazy NZCV, CBZ/CBNZ, branch
  targets after 16/32-bit instructions, and all condition-code truth tables.
* Differential suite: Thumb single-pass equals looped execution for every
  existing concrete-control-flow fixture, including APSR and one-instruction
  IT value tests.
* Loop-only tests: secret trip count; two candidates with writes to the same
  stack word; candidate overflow; different SP/LR/IT/security state fails
  closed; declared/non-declared indirect BX; a hooked call at a boundary.
* Recorder replay checks register outputs, NZCV-derived branch result, done,
  and storage output. `precompute` live/cache equivalence follows after the
  RISC-V split is stable.
* Add an Arm host SHA image loop gate first. Add a QEMU `mps2-an505` default
  feature gate only after it executes the loop adapter in the guest; retain the
  existing semihosting `SYS_EXIT_EXTENDED` protocol in
  `cirrus-armv8m-ert-selftest/src/main.rs`.

---

## Phase 3 — AArch64 facade using `disarm64`

### 3.1 New crate and ABI

Create `crates/ert/cirrus-aarch64-ert`, not a mode inside the Thumb crate.
A-profile A64 has different instruction encoding, register file, ABI, security
model, and exception model; sharing `cirrus-ert-core` and the loop driver is
enough.

* 31 GPR words (`x0`–`x30`) as `[W; 64]`; represent `SP` separately and treat
  encoding register 31 as `SP` only in forms where the ISA says so, otherwise
  as `XZR`. This distinction is mandatory.
* Maintain concrete `Option<u64>` and `Option<i64>` metadata; PC/SP/rstack are
  `u64`; LR is `x30`; the concrete rstack is caller supplied `[u64]`.
* `ert64_func` uses AAPCS64: `x0..x7`, then eight-byte caller stack slots;
  reserve enough space for max(arguments, results), subtract it from the top,
  and require 16-byte alignment. Results use `x0..x7`, then the same reserved
  stack window. Do not model `x8` indirect-result convention in v1; reject any
  ABI mode needing it.
* Define an A64 `SVC #0` guest ABI only if the guest runtime uses it: selector
  in `x0`, 32-byte hash input/output in `x1..x4`, and all-ones `x0` exits.
  Extend `cirrus-ert-rt` behind `target_arch = "aarch64"` and document it
  beside the RISC-V assembly. Avoid colliding with OS syscall semantics: the
  facade runs bare-metal/mapped images, not Linux userspace syscalls.

### 3.2 Decoder boundary

Add:

```toml
disarm64 = { version = "0.1.26", default-features = false, features = ["full"] }
```

Pin the exact tested minor version in `Cargo.lock`; do not generate or vendor
its multi-megabyte decoder table. `decode(raw)` must return `Some` before
ERT’s narrow decoder considers an instruction. Then match an explicit allowlist
of `Opcode` operation/mnemonic class **and** ERT-owned bit masks, extracting
fields from `raw`. Implement sign extension and PC-relative target arithmetic
in ERT, never in display formatting. Every unlisted decoder class returns
`DecodeError::Unsupported(raw)`.

Keep an instruction-to-source table in the crate docs/tests: Arm manual
encoding name, `disarm64` operation class, raw mask/value, fields, and the
unit/integration test that proves it. This is the audit record that prevents a
decoder upgrade from broadening the accepted ISA accidentally.

### 3.3 Initial supported A64 subset

Scope it from compiler output, validated by `llvm-mc`/`llvm-objdump` fixtures
and the SHA selftest, not from the entire `disarm64` table:

1. **Data/flags:** `ADD/SUB/ADDS/SUBS` immediate and shifted-register forms,
   `ADC/SBC` if emitted, logical shifted-register forms, `MOVZ/MOVK`,
   `CSEL` family needed by compiler lowering, `CMP/CMN/TST` aliases, and
   NZCV lazy materialization. `Wn` forms operate on 32 low bits and zero
   extend into `Xn`; `Xn` forms use 64 bits.
2. **Shifts/multiply:** immediate/register `LSL/LSR/ASR`, `ROR` where emitted,
   `MUL/MADD/MSUB`, and `UMULH/SMULH` only if required. Reuse the existing
   64-bit core circuits; do not add divide, crypto extensions, SIMD/FP, SVE,
   SME, PAC, MTE, LSE atomics, or system-register instructions.
3. **Addressing/memory:** `ADR`, `ADRP`, `LDR literal`, unsigned-immediate and
   unscaled stack/concrete loads/stores, plus `STP/LDP` pre/post-indexed frame
   forms. Support `LDR/STR` 8/16/32/64-bit integer widths with correct
   extension; add register-offset addressing only after actual fixture output
   demands it. Symbolic addresses remain stack-relative only.
4. **Control:** `B`, `BL`, `B.cond`, `CBZ/CBNZ`, `TBZ/TBNZ`, `BR`, `BLR`, and
   conventional `RET x30`. Direct targets are `pc + sign_extend(imm << 2)`;
   instruction length is always four bytes. Indirect `BR/BLR` follows the same
   concrete-or-declared bounded-target policy as other loop adapters.
5. **Termination:** `SVC #0` as above. `BRK`, `HLT`, `ERET`, exceptions,
   syscalls other than the declared `SVC #0`, and EL/system transitions fail
   closed.

### 3.4 Calls, hooks, and loop adapter

Introduce `Aarch64Handler` with an `Aarch64CallEvent`/action contract parallel
to Arm Thumb. Prefer moving only the architecture-neutral action semantics
(`Proceed`, `ReturnNow`, `Divert(u64)`) to `cirrus-ert-core`, re-exporting the
existing RISC-V name so its API remains stable; events remain per architecture.
A64 hooks cover `BL`, `BLR`, `RET x30`, and unresolved `BLR` before failure.
`RET Rn` where `Rn != x30` remains unsupported initially.

Add `cirrus-aarch64-ert-loop` as the third adapter to
`cirrus-ert-loop-core`. A64’s candidate state is simpler than Thumb’s (GPRs,
NZCV, SP/rstack), but must still distinguish `SP` from XZR and retain A64
stack alignment. Share the generic predicated-storage and candidate scheduler;
do not make the Thumb adapter depend on `disarm64`.

### 3.5 A64 validation matrix

* Pure decode tests: exact encodings generated by LLVM assembler, rejected
  near-miss/reserved forms, PC-relative sign boundaries, `Wn` vs `Xn`,
  XZR/SP register-31 split, and every load/store extension width.
* Plaintext semantics against hand-computed vectors, then native AArch64 host
  calls on this Apple Silicon machine for the safe integer fixture.
* Single-pass vs looped equivalence for concrete control flow; secret
  `B.cond`, `CBZ`, and `TBZ` loops; bounded `BLR`; candidate/stack divergence
  failures; recorder replay.
* New `cirrus-ert-aarch64-selftest` targeting `aarch64-unknown-none` (install
  it explicitly). First make the host ELF mapping gate deterministic. Only
  then add an explicit, timeout-bounded `qemu-system-aarch64 -machine virt`
  runner with a generated boot/linker layout and a documented semihosting or
  UART exit protocol. Do not substitute macOS host execution for Linux/QEMU
  coverage; preserve QEMU logs/artifacts on failure.

## Delivery sequence and commits

1. `[AI] Add Arm call interception to the ERT` — **complete** (implemented
   as focused commits for the hook contract, target validation, registry, and
   tests); no loop refactor.
2. `[AI] Extract architecture-neutral ERT loop driver` — **in progress**:
   allocation-free candidate scheduling and predicated writes now live in
   `cirrus-ert-loop-core`; complete callback-driven scheduling remains.
3. `[AI] Add Thumb adapter to the ERT loop driver` — promoted internal seam,
   full Thumb state folding, hooks-at-boundaries, host gate.
4. `[AI] Add AArch64 ERT facade with disarm64 decoding` — **in progress**:
   the allocation-free facade has an ERT-owned, `disarm64`-guarded decoder;
   symbolic state now executes direct/compare/test/register control flow,
   canonical `RET x30`, `ADR`/`ADRP`, `MOVZ`/`MOVK`, immediate and shifted
   register `ADD`/`SUB` (including `ADDS`/`SUBS`, `CMP`/`CMN` aliases, NZCV),
   logical shifted-register forms including `TST`, `CSEL`, and low-word
   `MADD`/`MSUB`/`MUL`. Concrete memory reads now cover PC-relative literal
   loads and unsigned-immediate or signed-unscaled byte/half/word/double
   loads with correct zero/sign extension, plus bounded concrete stores through
   the new mutable `RawMemory` seam, plus concrete LDP/STP pre/post/signed-
   offset frame forms with SP writeback; stack-relative symbolic storage is
   still rejected. `SVC #0` now accepts only the all-ones bare-metal exit
   selector; `step_with_hash` now implements the selector-0 32-byte digest
   ABI through x1..x4. AAPCS64 entry/result helpers now seed/read x0..x7 and
   eight-byte caller storage slots, and `cirrus-ert-rt` now has the
   target-gated AArch64 `SVC #0` hash/exit ABI, and an
   `aarch64-unknown-none` selftest image now links for the QEMU `virt`
   layout. The host ELF fixture now maps the built image, decodes its
   supported boot path, and reaches the declared exit. A timeout-bounded QEMU
   fixture is present but ignored because `virt` remains resident after the
   bare-metal SVC; it needs a UART/semihosting exit protocol before becoming a
   gate. High-half `SMULH`/`UMULH` now have audited symbolic/concrete
   semantics. Stack-relative unsigned-immediate loads/stores now move caller-
   owned symbolic bits through `ContextWithStorage`; remaining work is broader
   fixture coverage and a deterministic guest exit protocol.
5. `[AI] Add AArch64 looped ERT adapter` — **in progress**: the new
   `cirrus-aarch64-ert-loop` crate has fixed-capacity boundary snapshots,
   predicated GPR/SP/NZCV/done folding with concrete-SP divergence rejection,
   and shared-scheduler candidate body execution. Virtual-IP dispatch gates
   body effects and next-IP selection, exits predicate done, and the public
   no-alloc `initial_step`/`step` API carries loop state across generations.
   Remaining work is broader symbolic control-flow bodies, recording/
   precompute and equivalence tests.
6. `[AI] Add AArch64 bare-metal QEMU compatibility gate` — only once the
   boot/exit protocol is deterministic.

## Open risks and decisions to resolve before coding

* **`disarm64` API stability/size:** it is a generated broad decoder. Keep it
  optional only if code size proves material for embedded A64; the initial
  facade is not a Cortex-M target. Pin it and retain ERT masks as the support
  authority.
* **Thumb security state across symbolic control:** choosing fail-closed
  agreement is intentional. Symbolically selecting virtual TrustZone state
  would require a wire-backed policy model and changes the threat model.
* **Thumb flags:** lazy `FlagWire` is semantically useful. Materializing all
  flags at every fold would alter circuit shape; implement a `select_flag`
  that preserves deferred forms where possible, with materialization only as
  the conservative fallback.
* **A64 compiler drift:** derive the initial allowlist from checked-in LLVM
  disassembly fixtures and reject new codegen forms until reviewed. Do not use
  a Rust release upgrade as an implicit ISA expansion.
* **Prepared artifacts:** retain current loop recorder replay semantics first.
  `PreparedLoop`/repeat-until-done requires a dynamic invocation kind and is
  a separate design decision, not a prerequisite for hooks or the A64 facade.
