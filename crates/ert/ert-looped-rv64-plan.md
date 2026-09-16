# ERT: looped-circuit emulator, RV64 extension, call interception

**Status: Phases A/B complete; Phase C live executor and precompute cache implemented.** Remaining Phase C work is prepared-artifact APIs and a bare-metal QEMU integration gate. Research basis: `cirrus` (ERT crates),
`../volar-ir` (movfuscate / `lower_to_circuit_ir` / fuse), `../volar` (spec
conventions), `../rv-utils` (`rv-asm` RV64 decoder support, local checkout
ahead of the pinned rev), `../site` (step-circuit host glue), and
`../speet` (specialized recompilation philosophy, read-only reference).

Companion reading:

- [`frontend-choice.md`](frontend-choice.md) — ingest-path framing; the ERT
  section of this doc will need updating when phases land.
- [`../volar-ir/docs/llvm-fuse-unroll.md`](../../volar-ir/docs/llvm-fuse-unroll.md) —
  the "circuit-shape boundary" precedent: movfuscated self-loops must become
  **step circuits executed repeatedly**, never unconditional unrolls. The
  ERT looped circuit is the ISA-level analogue of that rule.
- [`../cirrus/crates/volar/cirrus-volar-vole/iop-statement-binding-corrections.md`](../cirrus/crates/volar/cirrus-volar-vole/iop-statement-binding-corrections.md) —
  chunk/state relation framing for IVC; informs how the looped ERT's
  register-file state should eventually be bound as `S_i → S_(i+1)`.

---

## 0. Problem statement

Today's ERT (`cirrus-ert`, RV32; `cirrus-armv8m-ert`, Thumb-2) is a
**single-pass symbolic interpreter**: concrete control flow is mandatory;
the first symbolic conditional branch that isn't an early-exit-loop
candidate hard-errors with `ErtError::Unexpected`. Loops only work because
their trip counts are concrete, so the *emitted circuit is as long as the
whole execution* (SHA-256 compress: 164k Thumb / 429k RV32 tables).

Three extensions are wanted, in this order of dependency:

1. **ERT → RV64** (word-generic core + RV64 facade + guest-side RV64 RT).
2. **Call interception** (host callbacks at `jal`/`jalr` call sites and at
   returns, behind a feature flag for the `alloc`-needing parts).
3. **A looped-circuit emulator on top**: the *same instruction semantics*,
   but the emitted circuit is one **step region per branch target**
   (specialized to the input image — fallthroughs stay fallthrough, all
   control flow that is statically resolvable is resolved live, while
   exploring paths), plus a small dispatch tail over a **virtual IP**.
   The executor runs the step circuit repeatedly, each iteration
   switching on the symbolic virtual IP to select the region whose body
   runs, and folding the next virtual IP computed by each region's
   backward/conditional branch. Exploration happens **live** — the
   emulator decodes from `RawMemory` and emits into the caller's
   `Context` with no allocation and no precomputed program; an `alloc`
   feature optionally adds up-front precomputation as a cache.

The user-facing result: a guest RV64/RV32 program with **secret-dependent
control flow** (bounded or unbounded trip counts, secret loop exits)
produces a circuit whose size is `O(program size)`, not `O(execution
length)` — the ERT counterpart of `movfuscate` + `lower_to_circuit_ir(1,
WithTerminationFlag)` + `execute_initialized`-in-a-loop that already exists
for the WASM/LLVM ingest paths.

## 1. Current-state summary (what we're building on)

### 1.1 What `cirrus-ert` already is

- `cirrus-ert-core` (`crates/ert/cirrus-ert-core/src/lib.rs`): the shared,
  architecture-neutral symbolic-word machinery — `add_bits_with_carry` /
  `add_word` / `select_word` / `partial_bitwise_word` / `fixed_shift` /
  `rv32_runtime_shift` / `arm_runtime_shift_with_carry` / `concrete_product`,
  the `Handler`/`EcallOutcome` ecall seam, `RawMemory` (bounded
  `from_slice` + unbounded bare-metal `new`, plus `with_ert_detect`
  overlay), and `EarlyExitLoopOptions`. **Already const-generic over word
  width in the adder** (`add_bits_with_carry<W, E, const N: usize>`), but
  most public helpers are pinned to `[W; 32]`.
- `cirrus-ert` (`crates/ert/cirrus-ert/src/{lib,machine,handlers}.rs`):
  RV32IM facade. `Machine` carries `regs: &mut [[W; 32]; 32]`,
  `reg_consts: [Option<u32>; 32]`, `offs: [Option<i32>; 32]` (stack
  pointer offset tracking), a concrete `rstack: &mut [u32]` return stack
  (enforces the supported direct-call/return convention), and
  `sp/stack_top`. `handlers::execute` matches every supported `Inst`;
  concrete fast paths everywhere; symbolic branches → `Unexpected` unless
  the `early-exit-loops` recognizer (`early_exit.rs`, a bounded forward
  disassembly — the in-repo precedent for "static scan over the image")
  fires.
- `cirrus-armv8m-ert`: Thumb-2 facade proving the "shared core, per-arch
  facade" pattern works, including a per-arch handler extension trait
  (`ArmHandler` with `fn early_exit_loop_options`; RV32's is the reserved
  no-op `RvHandler`/`RvDefaultHandler` tunnel).
- `cirrus-ert-rt`: guest-side runtime (`hash` ecall, `exit_with!`, XOF,
  `is_ert()` detect flag). Its RISC-V ecall path is currently
  `riscv32`-only (`#[cfg(target_arch = "riscv32")]` in `hash`); the
  `exit_with!` macro is already `riscv32|riscv64`.
- Entry points: `ert_emit` (run to exit ecall) and `ert_func` (RISC-V ABI
  wrapper: `a0..a7`, then caller-owned symbolic stack words;
  `PreparedRecorder` variants `ert_emit_prepared`/`ert_func_prepared`
  behind the `prepared-recording` feature, which is documented as "keeps
  allocator-requiring prepared artifact support out of bare-metal ERT
  users unless they opt in" — the precedent for the `alloc` feature flag).

### 1.2 Storage model

Symbolic memory is *only* the virtual stack: `StorageRuntime` couples the
handler's `ContextWithStorage<bool>` to caller-owned storage, addressed by
`sp`-relative stack-form addresses (concrete `LoadAddress::Concrete(u32)`
loads/stores hit `RawMemory` instead). Everything symbolic goes through
`ContextWithStorage::{storage_read, storage_write}` — today that's
`MuxTreeContext` (dense MUX/demux trees) or the `PreparedRecorder` (which
records `Op::Storage` into `PreparedProgram::storage_banks`). The
bare-metal selftest (`cirrus-ert-selftest`) also directly exercises a
Boolar `BCircuit` storage bank through `execute`, confirming the
native-storage route from a no_std binary.

### 1.3 Sibling "looped" precedent (the shape contract)

`../site/crates/proofs/src/{wasm_loop,llvm_loop,step_circuit}.rs` +
`../volar-ir/docs/llvm-fuse-unroll.md` establish the supported pattern:

1. Frontend lowers control flow to a **movfuscated self-loop** (one block,
   carried state, PC as data).
2. `lower_to_circuit_ir(1, WithTerminationFlag)` produces **one copy** of
   the step body with outputs `[done] ++ state ++ return`.
3. Host initializes storage once (`initialize_storage`), then loops
   `execute_initialized`, feeding `state` back as the next iteration's
   params until `done` (or a step budget), latching `return` on first
   done.
4. Tests prove `fuse(N, Unconditional)` (full unroll) ≡ N repeated steps
   of the one-step circuit — and HKDF-scale workloads are **fail-closed
   against unrolling** (26 GB / 9 min observed). Unrolling is explicitly
   the wrong default.

The ERT looped circuit adopts this exact contract at the ISA level, with
one deliberate difference: the movfuscate path's step circuit is a
*generic* interpreter-like block (dispatch over all blocks via
`emit_is_block` accumulators — see
`volar-ir/crates/ir/volar-ir-passes/src/dispatch_accumulator.rs`). The
ERT's step circuit is **specialized to the input image**: the body of each
branch target is emitted straight-line, fallthrough chains are flattened,
concrete control flow is resolved at circuit-construction time, and only
the residual (symbolic) dispatch remains as a virtual-IP switch. This is
the speet philosophy (specialized recompilation, `../speet`) transplanted
to circuit emission: an aggressively poor-man's `movfuscate` that knows
the guest code statically.

### 1.4 What `rv-asm` already gives us

The sibling `../rv-utils` checkout (the source of the `rv-asm` dependency)
already decodes/encodes **all** of RV64I: `AddiW/AddW/SubW`, `SlliW/SrliW/
SraiW/SllW/SrlW/SraW`, `Lwu`, `Ld`, `Sd`, 6-bit shift immediates, and
`Xlen::Rv64` plumbing (`Inst::decode(.., Xlen::Rv64)`). **However**: the
pinned rev in `cirrus/Cargo.lock` is `7fd8458` ("[AI?] Checkpoint"), and
the local checkout is at `dc485ea`, two commits ahead (`49f8251` RV64C
`C.ADDW/C.SUBW` compressed decode, `dc485ea` `C.ANDI` sign-extension).
The base RV64I decode coverage we need is already in the pinned rev, so
no dependency bump is strictly required for the ERT work — but the local
checkout fixes should be kept in mind if compressed decode ever enters
scope (the ERT rejects compressed instructions on purpose, so it should
not).

### 1.5 Constraints this plan must respect

- **Bare-metal no_std is a first-class target.** The Thumb facade is the
  primary microcontroller garbling target; everything added must keep
  `no_std` + no-`alloc` working with default features — the looped
  emulator included (its default mode is live exploration over a
  caller-sized, fixed-capacity region table). Anything needing `alloc`
  (the optional precompute pass, the call-interception registry) goes
  behind a feature flag, following the existing `prepared-recording`
  precedent.
- **Fail-closed.** Unsupported encodings/behaviors return
  `ErtError::Unexpected`, never silently miscompile (existing rule;
  `deloopify.rs` in `cirrus-llvm-pass` states the same policy).
- **Feature-flag cost discipline.** Per the `early-exit-loops` Cargo.toml
  comment: opt-in machinery must not cost unaffected callers.
- **QEMU gates.** New bare-metal coverage follows
  `crates/cirrus-coroutine/tests/bare_metal.rs` (QEMU runner for
  `riscv64gc-unknown-none-elf` already exists there) and the two
  `cirrus-*-selftest` crates; macOS-host Linux coverage uses Homebrew QEMU
  per repo instructions, software emulation only.

---

## 2. Design

### Phase A — RV64 extension (prerequisite)

**Goal:** `cirrus-ert` executes a well-behaved RV64IM subset with the same
semantics, keeping RV32 as a first-class configuration. No new guest
guarantees: concrete control flow, stack-form symbolic addresses,
documented opcode subset, aligned non-compressed instructions.

#### A.1 Word-width generalization in `cirrus-ert-core`

The adder (`add_bits_with_carry`) and `select_word` are already
`const N: usize`. Generalize the rest behind one parameter:

- Introduce a sealed **`WordWidth` trait** (`rv32`/`rv64` marker types) in
  `cirrus-ert-core`, associated consts: `BITS: usize` (32/64),
  `SHIFT_STAGES: usize` (5/6), `MASK: u64`.
- Width-polymorphic helpers become const-generic: `constant_word<N>`,
  `partial_bitwise_word<N>`, `invert_word<N>`, `fixed_shift<N>`,
  `rv32_runtime_shift` → **`runtime_shift<N>`** (loop bound becomes
  `SHIFT_STAGES`; RV64's `shamt[5]` simply adds a 32-place stage — the
  existing 5-stage structure generalizes verbatim).
- `compare_word` (`compare.rs`) gains a width parameter; predicates stay
  the same (`Eq/Ne/GeU/LtU/GeS/LtS`).
- `Product` gains the RV64 doubleword views. RV64 `MUL` = low 64;
  `MULH*` = high 64 of the 128-bit product — the existing
  zero-extend-to-2N + long-multiplication + `correct_high_product`
  structure in `handlers.rs` scales directly (`[W; 64]` → `[W; 128]`).
- **Keep the `[W; 32]` helpers as thin aliases** so `cirrus-armv8m-ert`
  (permanently 32-bit) compiles unchanged.

#### A.2 `cirrus-ert` facade changes

`machine.rs` and `handlers.rs` become width-generic over
`X: RvXlen` (a small crate-internal trait mirroring `rv_asm::Xlen`):

- Register file `[[W; X::BITS]; 32]`, `reg_consts: [Option<u64>; 32]`,
  `offs: [Option<i64>; 32]`, `pc: u64`, `sp/stack_top/rsp: u64`,
  `rstack: &mut [u64]`, `RawMemory` reads stay byte-based (addresses widen
  to `u64`; the existing `u32` API gets a `u64` sibling rather than a
  breaking change — `RawMemory::read` is `#[doc(hidden)]`, so this is
  internal).
- Public API: keep `ert_func`/`ert_emit` (RV32) and add
  **`ert64_func`/`ert64_emit`** (or a generic `ert_func_xlen::<Rv64>`;
  decided at implementation time — default to *separate names* to match
  the crate's explicit style and avoid inference churn at call sites).
- New instruction handlers, all reuse of existing circuits:
  - `AddiW/AddW/SubW`: 32-bit add path + sign-extend of bit 31 (a
    64-bit `select` is *not* needed — sign extension is wire fan-out).
  - `SlliW/SrliW/SraiW`, `SllW/SrlW/SraW`: the 32-bit shift circuits with
    result sign-extension; 5 stages, amount from `rs2[4:0]`.
  - 64-bit shifts: 6-stage barrel shifter.
  - `Lwu/Ld/Sd`: the existing load/store paths with width 32 (zero-ext) /
    64; `Lw` on RV64 **sign-extends** (per `rv-asm` docs) — must be
    handled in the stack-load path (sign-extend from bit 31) and the
    concrete path.
  - `Ecall`: unchanged (the `a0`-discriminated `Handler` contract widens
    its slices; see A.3).
- ABI: `ert64_func` uses the same a0–a7 then symbolic-stack convention,
  with 8-byte stack slots and 16-byte alignment per the RV64 psABI.

#### A.3 `Handler` trait widening

`cirrus_ert_core::Handler::ecall` currently takes
`regs: &mut [[Self::Wrapped; 32]]`, `reg_consts: &mut [Option<u32>]`,
`offsets: &mut [Option<i32>]`. Options:

- **Chosen:** make the trait generic over the word width in the same
  sealed way as the core (`Handler<Val, X = Rv32>` with a defaulted
  parameter, or a `Word` associated type on the existing trait). A
  defaulted generic keeps every existing `Handler` impl source-compatible.
  The hash ecall's 8-word payload (`a1..a7,x12..x18` on RV32) becomes
  four 64-bit words on RV64 — the `cirrus-ert-rt` guest side and
  `DefaultHandler` change together; the wire format (32 bytes in, 32
  bytes out) is preserved.
- The `Machine` continues to reset `x0`/`sp` per `reset_fixed_registers`,
  widened.

#### A.4 Guest RT + selftest

- `cirrus-ert-rt`: extend the `hash` asm block to `riscv64` (a1 + four
  inout regs), keep `exit_with!` as-is. `is_ert()` unchanged.
- New `cirrus-ert64-selftest` crate mirroring `cirrus-ert-selftest`
  (same SHA-256 fixture, `riscv64gc-unknown-none-elf`, QEMU `virt`,
  SiFive test finisher). The coroutine bare_metal runner
  (`crates/cirrus-coroutine/tests/bare_metal.rs:101`) already shows the
  exact QEMU invocation pattern for riscv64gc.
- Host unit tests: extend `cirrus-ert/src/tests.rs` patterns — concrete
  RV64 programs exercising every new handler, plus differential checks of
  W-form sign-extension against a native RV64 reference (or a
  hand-computed vector when no runner is available).

#### A.5 Explicit non-goals for Phase A

- No `MULW`/division (M-extension high halves are in; `MULW` is a
  trivial add if a workload needs it — left as a follow-up checkbox).
- No compressed instructions (existing policy), no floating point, no
  atomics, no CSR/fence, no misaligned accesses.
- No symbolic control flow — that is Phase C's entire point.

### Phase B — Call interception (feature `call-hooks`, `alloc`-gated parts)

**Goal:** let the host observe and replace behavior at call boundaries
without perturbing the interpreter core. Two tiers:

#### B.1 Tier 1 — synchronous hooks (no `alloc`)

Add an optional hook to the per-arch handler extension traits
(`RvHandler` — today the reserved no-op tunnel, precisely the seam this
was reserved for; `ArmHandler` analogue later, out of scope for the first
cut):

```rust
/// Observation/replacement at call and return boundaries.
/// Default: no interception (identical behavior and cost to today).
fn call_hook(&mut self, _event: CallEvent<'_, W>) -> Result<CallAction, Self::Error> {
    Ok(CallAction::Proceed)
}
```

- `CallEvent::{Call { target_pc: u64, link: Reg, caller_pc }, Return { target_pc, from_pc }}`,
  carrying `&mut` register-file views like `ecall` does.
- `CallAction::{Proceed, Skip /* execute body symbolically anyway? no —
  replace */, Divert(u64)}`. Initial semantics: `Proceed` (observe only),
  `Divert(pc)` (treat the call as a jump to a host-chosen concrete
  target — e.g. an ecall stub or an instrumented wrapper), and
  `ReturnNow` (write caller-supplied result registers and pop `rstack`
  without executing the callee — the interception that lets a host model
  e.g. `memcpy`/crypto as native calls).
- Fired from `jump_and_link`/`jump_and_link_register` in `handlers.rs`
  (and `return_from_call`), **only when the hook is implemented** — the
  default method body returns `Proceed`, and the handler implementor opts
  in, so there is no cost or behavior change for existing users (matches
  `early_exit_loop_options`' design).
- Concrete-target-only to start: a symbolic call target already fails
  closed today (`Unexpected`); the hook is consulted *before* that error
  so a host can resolve a known-indirect target itself (precursor for
  Phase C's bounded indirect dispatch).

#### B.2 Tier 2 — recording-friendly interception registry (`alloc`, feature `call-hooks`)

Some interceptions need registries (maps from target PC → replacement
behavior, per-call-site state). Gate those:

```toml
[features]
call-hooks = ["dep:..."]   # pulls in `alloc`-using registry helpers
```

- Provide `CallRegistry` (B-tree map `pc → Box<dyn FnMut ...>`-style, but
  concrete: a trait object-free enum of canned replacements — hash,
  memcpy, memset, plus a closure slot) under the feature. Deliberately
  enum-shaped, not trait-object-shaped, to stay recorder-friendly.
- Document the `prepared-recording` precedent in the manifest comment:
  "Keeps allocator-requiring call-interception support out of bare-metal
  ERT users unless they opt in."

**Interplay with the recorder:** when the active context is
`PreparedRecorder`, a `ReturnNow`/`Divert` interception must emit
*circuit* ops (writes into register slots), which the recorder records as
usual — interceptions are just host-driven instruction replacements. A
native-only fast path (concrete result written via `write_constant`) is
allowed and stays invisible to the circuit.

### Phase C — Looped-circuit emulator (`cirrus-ert-loop`, `alloc`-free by default; feature `precompute` opts into caching)

This is the centerpiece. Deliver it as a **new crate**
`crates/ert/cirrus-ert-loop` so the single-pass facades stay untouched. It
*shares instruction semantics* by reusing `cirrus-ert-core`'s word circuits
and (via `pub(crate)` → `pub`-but-`#[doc(hidden)]` promotion, or a shared
internal module) the per-instruction data-path handlers from `cirrus-ert`.

**Default mode is `no_std` + `alloc`-free**: paths are explored **live**,
decoding straight from `RawMemory` exactly like the single-pass
`Machine::run` loop, and symbolic operations are emitted into whatever
`Context` the caller drives — no program-wide pre-computation, no
allocation, no pre-built region table. The difference from single-pass
execution is confined to control flow: when execution meets a branch whose
condition is symbolic, the emulator does not hard-error and does not
unroll — it **closes the current step region**, makes the branch outcome a
select on the **virtual IP**, and transfers control by looping.

#### C.1 Live exploration (default, `alloc`-free)

State is the single-pass `Machine` state (registers, consts, `offs`,
`sp`/`rstack`) plus:

- a **virtual IP**: 64 symbolic wires carrying the PC across step
  boundaries; initialized to the entry PC; while the concrete `pc` still
  equals the virtual IP's known value the machine runs inline exactly as
  today (fallthrough is literal fallthrough — zero dispatch gates);
- a **fixed-capacity region table** — a caller-sized borrowed slice
  `&mut [Option<RegionEntry>]`, same pattern as the `early-exit-loops`
  feature's `loop_sites: [Option<RecognizedSite>; 8]` cache. One entry
  per *branch-boundary PC encountered at run time*:
  `{ base_pc: u64, next_ip: [W; 64] /* baked const or select result */ }`;
- a **done flag** wire.

The run loop:

1. **Inline phase.** Decode-and-execute through the ordinary handlers
   (identical gates, identical `offs`/const bookkeeping) for as long as
   control flow resolves concretely — unconditional jumps, concrete
   conditional branches, conventional calls/returns through the concrete
   `rstack`. This is today's interpreter verbatim; straight-line and
   concretely-directed code never pays for the looped machinery.
2. **Symbolic control-flow point.** On a conditional branch with a
   symbolic condition (or a bounded indirect `jalr`, see C.2.3, or
   reaching a region boundary):
   - emit `next_ip = select_word(cond, taken_target, fallthrough_target)`
     (one `select_word` — the *only* dispatch cost a conditional branch
     ever pays);
   - record the region boundary `{ base_pc, next_ip }` in the fixed table
     (re-encountering the same `base_pc` — e.g. a loop header on every
     iteration — reuses the entry; the *gate emission* for the body
     repeats per executed iteration, exactly as a recorded loop body
     would);
   - set `virtual_ip ← next_ip`, mark the concrete `pc` unknown, and
     **return control to the caller's step loop** (or, in the fused
     executor, continue directly at step 3).
3. **Dispatch.** Compare `virtual_ip` against each `base_pc` in the
   region table (`is_block`-style AND-of-bits decode — the
   `dispatch_accumulator.rs` formulas, evaluated live, not pre-baked).
   - Hit: resume the inline phase at that PC with the region's
     continuation.
   - Miss with a *known-concrete* IP (first visit to a call target or
     branch destination): start a fresh region inline at that PC.
   - Miss with a *symbolic-only* IP that no entry matches: fail closed
     (`Unexpected`), unless it falls under a declared bounded-indirect
     set.
4. **Termination.** The exit ecall (or `ert_func`'s return landing) sets
   `done`; the host loop checks `done`/budget each step, first-done
   latches the return window — the `run_step_loop` contract from
   `../site/crates/proofs/src/vole_storage.rs`.

Backward branches (loop latches) are just cases 2/3: a loop with a
secret trip count emits its body once per *executed* iteration into the
context (gates stream like today — a recorder sees one body per
iteration; a streaming garbler streams per iteration), while the
*program representation* needed to drive it stays O(regions seen),
bounded by the fixed table. Table exhaustion (more distinct boundary
PCs than the caller sized for) fails closed with a diagnostic — the
caller re-runs with a bigger slice, exactly the `loop_sites` discipline.

The executor API mirrors `execute_initialized`'s host loop and works
with any `ContextWithRvOps` context — plaintext, GC, recorder:

```rust
let mut emu = LoopedMachine::new(mem, entry, regions_buf, rstack, /* …same as ert_func */)?;
loop {
    match emu.step(&mut runtime)? {   // emits into the caller's Context
        Step::Continue => {}
        Step::Done => break,
    }
    budget.tick()?;                    // host-enforced; done/budget contract
}
```

#### C.2 What the step circuit is

(Shape contract — identical in both modes; live mode materializes it
incrementally, precompute mode materializes it wholesale.)

1. **Step regions.** Boundaries are exactly: every symbolic conditional
   branch target and fallthrough successor, every symbolic-conditional
   backward-branch (loop) header, every declared indirect target, the
   entry PC, and every return landing not modeled by the concrete rstack.
   Straight-line fallthrough runs are flattened into one region — a
   region body contains **zero control flow decisions** except at its
   terminator.
2. **Region bodies** are emitted with the *same* handler functions the
   interpreter uses — the Boolean gates are identical, only the control
   shell differs. Within a region: no fallthrough branches, no dispatch;
   `sp`-offs bookkeeping identical to `Machine`.
3. **Region terminators** (all specialization happens here):
   - Unconditional concrete jump/fallthrough → **baked**: the region's
     next-IP output is the constant target (zero gates).
   - Conditional branch on a *symbolic* condition → the terminator emits
     `select_word(cond, taken_target, fallthrough_target)` over the IP
     words — the only place dispatch cost is paid.
   - Backward branch (loop latch) → identical treatment; loop-carried
     state is registers + storage, exactly like movfuscate.
   - Bounded indirect (`jalr` with a declared target set of size *k*) →
     a *k*-leaf constant-mux tree over the concrete-target comparison
     (comparator per leaf vs. the register's concrete-or-symbolic value);
     undeclared targets → fail closed (live mode: at encounter;
     precompute mode: at compile time).
   - Calls/returns → see C.4.
4. **Virtual IP & state**: the carried state is the register file
   (32×64 wires), the virtual IP (64 wires), a done flag; caller-visible
   symbolic storage stays external through `ContextWithStorage`
   (unchanged from single-pass ERT — native VOLE storage banks keep
   working; the big advantage over the WASM path's `2^24` dense-image
   trim).
5. **Step circuit assembly** (what one iteration is, viewed as a
   circuit): outputs = `[done] ++ regs' ++ ip' ++ return-window`. For
   each region *r* touched this iteration, `active_r =
   is_block(virtual_ip, base_pc(r))` selects that region's computed
   continuation; regions whose next-IP is constant contribute zero gates
   to `ip'`; conditional terminators contribute one `select_word`.
6. **Termination**: `done` is set when `ip'` equals the entry's return
   landing (the `ert_func` ABI exit) or when the exit ecall fires.
   Budget-checked by the host loop.

#### C.2bis Optional precomputation (feature `precompute`, pulls `alloc`)

Everything the live mode discovers lazily can instead be computed **once,
up front**, when `alloc` is available — this stays valid and is the
recommended host-side mode for recorded/replayed artifacts:

- `LoopedProgram::compile(mem, entry, &CompileOptions { indirect_targets,
  max_regions, .. })` performs the bounded forward disassembly
  (concrete-jump chasing, region partition — the C.1 walker run to
  fixpoint) and returns an owned region table.
- Execution then uses the **identical** step/dispatch path as live mode,
  seeded with the precomputed table; the table is complete, so dispatch
  never hits discovery misses, and `PreparedRecorder` sees a stable,
  iteration-invariant region sequence per loop body.
- Equivalence gate: precomputed execution ≡ live execution on every
  fixture (same gates, same order) — the precompute pass is a cache,
  never a semantic change. This mirrors how `early_exit.rs` scans
  statically at run time vs. how a compiler pass would know the same
  facts ahead of time; both are supported, semantics pinned to the live
  interpreter.

`*_prepared` variants record through `PreparedRecorder` exactly like
`ert_func_prepared` does today: run the emulator against the recorder
context and `finish` the artifact. In live mode the recorded program is
the executed trace (one body per iteration, as today); in precompute
mode the per-iteration region sequence is stable, which is what makes a
`PreparedLoop`-shaped artifact ("one body template, N invocations with a
table") reachable — wiring precomputed regions → `PreparedLoop` instead
of a flat per-iteration stream is an explicit design option to evaluate
during implementation, and is the only piece that *requires* the
`precompute` feature.

#### C.3 How this differs from both existing paths

| | single-pass ERT | WASM/LLVM movfuscate path | **ERT looped** |
|---|---|---|---|
| circuit size | O(execution) | O(program), generic block | O(execution) emitted, **O(regions) driver** |
| fallthrough | sequential emission | dispatch per block | **baked, zero gates** |
| branch cost | hard error (concrete-only) | full accumulator | `select_word` on IP only |
| indirect | rejected | n/a (WASM CF) | bounded constant-mux |
| symbolic memory | virtual stack (native storage) | dense `2^n` image | virtual stack (native storage) |
| trip counts | concrete only | secret, budget-looped | secret, budget-looped |
| allocation | none | required | **none by default** (`precompute` opt-in) |

#### C.4 Calls and returns in the looped emulator

Two modes, selected per call site:

1. **Inline static calls** (default for direct `jal`): while the call
   graph stays concrete, calls and returns keep using the existing
   concrete `rstack` machinery verbatim — the callee's straight-line
   code simply runs in the inline phase. Depth is bounded by the
   caller-supplied `rstack` slice, exactly as today; recursion past that
   bound fails closed, unchanged. Zero new circuit cost.
2. **Intercepted calls** (Phase B hooks): `ReturnNow` lets the host
   replace the callee with native/emitted behavior per iteration;
   `Divert` remaps the target statically.

Returns of inlined calls stay on the concrete rstack. Secret-dependent
*function pointers* route through the bounded indirect-target mechanism
(C.2.3) and never through the return path — the rstack discipline is
preserved by construction.

#### C.5 What stays fail-closed

- Symbolic `sp`-relative forms and symbolic non-stack addresses: rejected
  exactly as today (the virtual stack is the only symbolic memory).
- Undeclared indirect targets, dispatch misses on symbolic-only IPs,
  region-table exhaustion (caller undersized the borrowed slice),
  reaching the end of the decoded image, `done` never set within the
  host's step budget, unbalanced stack at exit — all `Unexpected`.
- Compressed instructions, misalignment, unsupported opcodes — unchanged.

#### C.6 Arm facade

Out of scope for the first cut (Thumb's IT/APSR state makes region
boundaries subtler). The design leaves room: region/terminator machinery
lives in the new crate over the *shared* core traits, not in
`cirrus-ert` internals.

## 3. Implementation checklist

### Phase A — RV64 (crate-local, no `alloc`) — **DONE**

- [x] `cirrus-ert-core`: width-generic `constant_word`/`partial_bitwise_word`/
      `invert_word`/`fixed_shift`/`runtime_shift` (new; `rv32_runtime_shift`
      kept as a 32-bit alias)/`compare_word`/`bitwise_word`; `RawMemory`
      gained the u64-address `read64` sibling.
- [x] `cirrus-ert-core`: `Handler` gained a defaulted `BITS` const parameter
      with `u64`/`i64` concrete metadata; `EarlyExitLoopOptions` unchanged.
      (No sealed `WordWidth` trait was needed — const generics sufficed.)
- [x] `cirrus-ert`: BITS-generic `Machine`/`handlers`; `ert64_func`/
      `ert64_emit` (+`ert64_func_prepared`/`ert64_emit_prepared`); W-form
      sign-extension helpers; 64-bit load/store paths; 6-stage shifter;
      128-bit product (split into two half-width accumulators to stay
      const-generic on stable). RV64 `LUI`/`AUIPC` sign-extension handled.
- [x] `cirrus-ert`: `DefaultHandler` hash ecall 8×32→4×64 payload on RV64;
      RV64 exit accepts both `0xffff_ffff` and `u64::MAX` encodings.
- [x] `cirrus-ert-rt`: `hash` riscv64 asm (4×64-bit regs); `exit_with!`
      confirmed width-generic.
- [x] `cirrus-ert64-selftest` crate + `riscv64gc` QEMU gate
      (`tests/bare_metal64.rs`), incl. ELF64 parsing.
- [x] **Compressed-instruction support** (replacing the no-`alloc` custom
      target workaround): two-stage fetch in `Machine::decode`, length-aware
      pc stepping, early-exit walker length awareness. rv-asm bumped to the
      sibling checkout via the parent `.cargo/config.toml` patch (needed the
      RV64C `C.ADDW`/`C.SUBW` and `C.ANDI` fixes; stale `rvdc::` doctests
      fixed upstream).
- [x] Tests: per-instruction RV64 unit vectors (tests64.rs); W-form
      sign-extension differentials; `Lw`-sext/`Lwu`/`Ld` matrix; symbolic
      shift ≥32; `MULH*` 128-bit vs native; llvm-mc-verified compressed
      encodings; mixed compressed/normal control-flow test.
- [x] Latent bugs fixed along the way (repo policy): symbolic `SUB` inverted
      the wrong operand (computed `src2 - src1`); `JAL` never wrote its link
      register; RV32/64 host harnesses missed v0-mangled fixture sections in
      the linker script (latent, unrelated to RV64).
- [x] Docs: crate docs + `frontend-choice.md` + README RV64/compressed rows.
- Note: the `cirrus-armv8m-ert` QEMU `bare_metal` integration test SIGSEGVs
  at HEAD already (pre-existing harness issue, unrelated to this phase; its
  25 unit tests pass).

### Phase B — call interception — **DONE**

- [x] `cirrus-ert`: `CallEvent`/`CallAction` types; defaulted `call_hook`
      on `RvHandler` (RV32+RV64), fired from `jump_and_link`/
      `jump_and_link_register`/`return_from_call`; the unresolved-indirect
      event is consulted before the historical fail-closed error. The
      blanket `RvHandler` impl was replaced with explicit impls
      (`RvDefaultHandler` delegates, `DefaultHandler` keeps the default) —
      custom handlers now write a trivial `impl RvHandler for X {}`.
- [x] Feature `call-hooks` (`alloc`) on `cirrus-ert`: `hooks::CallRegistry`
      + `CallReplacement::{NoOp, ReturnConstants}` + `apply_replacement`,
      manifest comment per the `prepared-recording` precedent. (Storage-
      touching canned replacements like `memcpy`/`memset` were descoped:
      the hook receives register views only; that would need a storage
      handle in the hook signature — noted as a follow-up.)
- [x] Tests (`tests_hooks.rs`): observer counts calls/returns with exact
      PCs; `ReturnNow` replaces a callee (callee marker untouched, results
      hold); `Divert` to a stub; `Divert` on return overrides the landing;
      unresolved `JALR` fails closed by default and resolves when hooked;
      registry replaces registered targets only; the recording path
      records hook-emitted gates (64 XOR gates in the trace).
- [x] Arm side: no hook firing; the shared `Handler` trait is unchanged,
      and the arm facade's own handler machinery is untouched.
- [x] Docs: crate docs + `frontend-choice.md`.

### Phase C — `cirrus-ert-loop` (`no_std`, `alloc`-free; feature `precompute`)

- [x] New `cirrus-ert-loop` crate: `#![no_std]`, **no `alloc` in the
      default build**; depends on `cirrus-ert`, `cirrus-ert-core`,
      `cirrus-core`, and `rv-asm`. The `precompute` feature alone enables
      `alloc`, with the manifest comment documenting the bare-metal cost
      boundary.
- [x] Live explorer: `LoopedMachine` drives the ordinary handlers from
      `RawMemory` until a symbolic control-flow point, then carries virtual
      IP + done wires and a caller-sized fixed-capacity live-candidate table.
      Candidate overflow, undeclared indirect targets, divergent concrete
      stack state, and unsupported behaviour fail closed.
- [x] Region bodies reuse the promoted `#[doc(hidden)]` ERT `Machine` and
      handler seam, retaining identical register-constant and stack-offset
      bookkeeping.
- [x] Terminators emit `select_word` for symbolic conditionals/backedges and
      a declared-target constant mux for symbolic `JALR`; ordinary
      calls/returns retain the concrete rstack and the Phase-B hook tunnel.
- [x] Dispatch folds each live candidate through its virtual-IP equality
      predicate. Register and storage updates are predicated, and done is
      accumulated as a wire; the exposed state is compatible with the
      `[done] ++ state` step-loop contract.
- [x] `step`, `run`, `done_wire`, result ABI helpers, and deterministic
      budget truncation provide the host executor interface for any ERT
      Boolean context.
- [x] Feature `precompute` (`alloc`): `LoopedProgram::compile()` records a
      bounded forward-disassembly boundary map; attaching it to a machine
      consistency-checks the live walker, so it remains a cache rather than
      an alternative semantics.
- [] `*_prepared`: record via `PreparedRecorder` (both modes); with
      `precompute`, evaluate wiring stable region sequences →
      `PreparedLoop` (document the decision; additive only).
- [~] Equivalence gates: concrete call/loop control-flow equivalence,
      RV32 entry coverage, secret-trip-count completion where the
      single-pass path fails closed, bounded indirect dispatch, predicated
      divergent stack writes, deterministic budget truncation, and recorder
      replay are covered in `cirrus-ert-loop/src/tests.rs`; the RV64 SHA-256
      ELF gate runs the real guest image on the host. Live ≡ precompute is
      covered for the loop fixture. Expanding this to every existing ERT
      fixture and adding `PreparedRecorder`-specific entry points remains.
- [ ] Bare-metal: a default-feature cross build of `cirrus-ert-loop` proves
      the crate itself remains no-alloc; wiring its executor into the RV64
      selftest image and QEMU gate remains a later integration milestone.
- [~] Docs: crate docs, `frontend-choice.md`, and the root README describe
      the live looped executor and its caller-owned candidate/budget
      requirements. A measurement table and region-table sizing benchmark
      remain to be added.

## 4. Risks / open questions

1. **State width.** Carried state = 32×64 register wires + 64 IP + done ≈
   2.1 k wires — trivially small vs. the movfuscate path's dense storage
   images, and the symbolic stack stays external (native VOLE storage).
   Risk: **register-file liveness**. Region-entry liveness could shrink
   the carried set per region (only live-in/live-out regs need folding),
   a clear follow-up optimization (`compact_slots`-style, cf.
   `PreparedProgram::compact_slots`); the plan ships the full file first
   for correctness, with liveness as a documented fast-follow.
2. **Region blowup.** Pathological images (dense branch tables) could
   make region count ~ instruction count, degenerating toward the generic
   movfuscate shape. Mitigation: the region table is caller-sized
   (live mode: borrowed `&mut [Option<RegionEntry>]`; precompute mode:
   `CompileOptions::max_regions`), and exhaustion fails closed with a
   diagnostic; measure on real guests.
3. **Indirect-branch soundness.** The bounded-target mux must be
   exhaustive: every declared set must cover all targets the image can
   produce (caller attests; the emulator re-checks per hit and
   fail-closes on a miss, same as an undeclared concrete target today —
   in live mode at encounter, in precompute mode additionally at compile
   time).
4. **Hook/recorder interaction.** A `ReturnNow` hook that writes native
   constants is invisible to the circuit by design — document that this
   is *only* sound when the replacement is genuinely a public function of
   public inputs; secret-dependent native interceptions must emit gates.
   (Same discipline as the `hash` ecall today, which writes concrete
   registers.)
5. **`Handler` widening churn.** The defaulted generic is source-compatible
   but touches every impl in-tree (arm facade, selftests). Sized as ~1
   day of mechanical edits; alternative (new `Handler64` trait family)
   was rejected — it would fork the ecall contract.
6. **rv-asm drift.** Pinned `7fd8458` vs. local `dc485ea`; ERT RV64 needs
   nothing past the pin, but do not bump casually — the two extra commits
   touch compressed decode, which the ERT intentionally never exercises.
   No in-repo `[patch]` for rv-utils exists in `../.cargo/config.toml`;
   do not add one (repo rule: parent patch configs win; this one isn't
   patched, so keep it that way).
7. **Naming.** `ert64_*` vs. generic `ert_func::<Rv64>` — decide at PR
   time; this plan defaults to explicit names.
8. **PreparedLoop fit.** `PreparedLoop`'s invocation table is built for
   *statically unrolled* loop nests; the looped emulator's iterations are
   dynamic (secret trip count). Live mode records a flat per-iteration
   trace (status quo); a `PreparedLoop`-shaped artifact needs the stable
   region sequences only `precompute` provides, and possibly a
   "repeat until done" invocation kind. Flagged as the one place a
   `cirrus-recompile-core` change might be needed; keep it additive.

## 5. Out of scope (explicit)

- Compressed instructions, floating point, atomics, CSR, misalignment,
  self-modifying code — existing ERT exclusions, unchanged.
- Thumb/Armv8-M looped emulator (design leaves room; see C.5).
- Symbolic non-stack memory / heap allocation in-circuit (the
  `frontend-choice.md` "heap allocation" limit stands).
- SuperNova-style authenticated dispatch across *heterogeneous* step
  circuits (`iop-statement-binding-corrections.md`'s universal-VM
  relation). The ERT looped circuit is uniform (one step circuit), which
  is the friendlier folding target; binding `S = (regs, ip, storage-root)`
  into a chunk relation is a future `cirrus-volar-vole` project, noted
  here only to keep the state layout (`[done] ++ regs' ++ ip' ++ return`)
  compatible with it.
