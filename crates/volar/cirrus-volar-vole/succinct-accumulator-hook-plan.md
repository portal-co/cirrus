# Plan: pluggable succinct-accumulator hook for the Cirrus VOLE verifier context

## Goal

Allow `cirrus-volar-vole`'s **runtime verifier contexts**—not
`volar-weaver`—to maintain Volar's fixed-size IOP accumulator while a Cirrus
program executes. The accumulator must observe every actual verifier-side AND
operation, including ANDs introduced by `Or`, `Mux`, typed arithmetic, and
storage adapters.

The feature will be an optional, pluggable hook on `VoleVerifierContext` and
its locked/storage-capable counterparts. The first provider will adapt
`volar-iop`'s `IopAccumulator<Gf128>` and `fold_gate`, so the final accumulator
can be finalized by Volar's IOP proof APIs. Ordinary VOLE users retain the
current API, error type, transcript order, and zero additional semantic work.

## Existing seams and constraints

- `VoleVerifierContext::bitand` currently consumes the next `hat` and derives
  `q_and` with `volar_spec::vole::setup::derive_and_q`. This is the one runtime
  location that sees `K_a`, `K_b`, `K_c`, `Delta`, `Vhat`, and the precise gate
  order required by Volar's `and_check_r1cs` relation.
- The locked verifier has the equivalent operation in
  `LockedVoleVerifierContext::bitand_by_ref`. The storage contexts delegate to
  those underlying verifier contexts, so hooking the underlying contexts covers
  their gate operations without duplicating the fold logic.
- Volar's weaver-side `VerifierTraceSink`/`IopSink` is a code-generation
  extension point. It must not be reused or made a Cirrus dependency: this work
  is execution-time context instrumentation.
- Use the Git dependency for `volar-iop`, not `../volar` paths:

  ```toml
  volar-iop = { version = "0.1.0", git = "https://github.com/portal-co/volar.git", optional = true }
  ```

  The implementation must not depend on `volar-verifier-iop-runtime`: that
  crate includes a `std` compile/run harness and is not appropriate for this
  `no_std` backend. Lock the resolved Git revision in `Cargo.lock`. Review the
  workspace's existing `volar-spec` patch separately: it may be useful for
  local development, but it must not silently turn the new `volar-iop`
  dependency into a path dependency.
- `volar-iop::fold::fold_gate` needs a challenge for each observed gate and
  consumes scalar field elements. The hook therefore owns challenge production
  and the Volar-field-to-`Gf128` embedding; the generic VOLE context must not
  assume a particular field or transcript construction.

## Design

### 1. Define a small, runtime-only hook interface

In `crates/volar/cirrus-volar-vole/src/lib.rs` (or a focused new `hook.rs`
module), define a public trait conceptually shaped as:

```rust
pub trait VoleVerifierHook<N, T> {
    type Error: core::error::Error;

    fn on_and(
        &mut self,
        gate_index: usize,
        delta: &Delta<N, T>,
        q_a: &Q<N, T>,
        q_b: &Q<N, T>,
        q_c: &Q<N, T>,
        hat: &Array<T, N>,
    ) -> Result<(), Self::Error>;
}
```

The final spelling may refine bounds and borrowing, but preserve these rules:

- It receives the exact values used to derive this gate's output share.
- It receives `q_c` **after** `derive_and_q`, so providers see the actual
  output propagated through the Cirrus computation.
- `gate_index` starts at zero, advances once per successful transcript pull,
  and is not an externally supplied “expected AND count.”
- The interface does not return or alter a VOLE wire. Hooks are observational
  and cannot change verifier semantics.
- Provide `NoopVoleVerifierHook` with `Infallible` error. It is the default.

Keep finalization deliberately out of the generic callback. A concrete provider
may expose `accumulator()`, `into_accumulator()`, or `finish()` on its own
concrete type; the context only owns gate observation.

### 2. Make the verifier context generic over the hook, without breaking users

Change the verifier shape to carry a defaulted hook type, e.g.
`VoleVerifierContext<N, T, I, H = NoopVoleVerifierHook>`, with a public
constructor or explicit field-based construction that makes hook ownership
clear. Add a `next_gate_index` counter.

Do the same for `LockedVoleVerifierContext`. For locked contexts, put the hook
behind the same interior-mutable synchronization boundary as transcript
consumption, so a `&self` operation consumes exactly one `hat`, updates exactly
one hook state, and advances the index atomically relative to other compatible
circuits. Do not add a separate lock with an independently observable order.

Preserve source compatibility for existing literal initialization as far as
possible. If adding fields makes that impossible, introduce constructors and
make the migration explicit across this workspace, then document the breaking
surface. Prefer a compatibility wrapper/constructor over a broad public-struct
break if Rust's initialization rules allow it.

Model hook failure explicitly:

- Introduce a generic verifier error such as
  `VoleVerifierError<E> { HatExhausted, Hook(E) }`.
- Keep `VoleVerifyError` as the `Infallible` specialization (or an equivalent
  compatibility-preserving public alias) so current non-hook users still
  handle `HatExhausted` exactly as today.
- Storage-context errors must wrap the generic verifier error rather than
  discard the provider failure. Update locked and unlocked storage adapters
  consistently.

In `bitand`, use this sequence:

1. Pull one `hat`; return `HatExhausted` before modifying hook state if none
   exists.
2. Derive `q_c` using the existing `derive_and_q` call.
3. Invoke the hook with references to `a`, `b`, `q_c`, `delta`, and `hat`.
4. Advance the gate index only after a successful hook call and return `q_c`.

This makes retry behavior well-defined: a failed hook does not claim a
successfully processed gate. If a provider can fail after mutating itself, its
contract must either make that mutation recoverable or report itself poisoned;
the built-in provider will be infallible after construction.

### 3. Add the optional Volar IOP provider

Add an `iop-accumulator` Cargo feature to `cirrus-volar-vole` which enables
`dep:volar-iop`. Put the adapter in `src/iop_accumulator.rs`, gated by that
feature, and re-export its public provider types from `lib.rs`.

The provider should own:

- `volar_iop::fold::IopAccumulator<Gf128>`;
- a challenge source/callback invoked once per gate; and
- an explicit lane selection policy.

For its `on_and` implementation, lift `q_a.q[lane]`, `q_b.q[lane]`,
`q_c.q[lane]`, `delta.delta[lane]`, and `hat[lane]`, obtain the challenge for
`gate_index`, then call `volar_iop::fold::fold_gate` and store the resulting
accumulator.

Do not copy the weaver's bare-name API (`IopAccumulator`, `IopLift`,
`iop_fold_gate`) into Cirrus. Instead expose normal Rust types and methods,
with names scoped to `cirrus_volar_vole`.

Define a Cirrus-owned embedding trait for supported VOLE scalar types. Start
with the exact field types that Volar's runtime adapter proves are safe:
`volar_spec::field::Galois` and `Bit`, embedded canonically in `Gf128` at the
tower base level. Do not claim support for `Galois128` or arbitrary `T` merely
because the verifier context is generic; add an implementation only after
showing the required field embedding exists and preserves the AND relation.

Make lane behavior impossible to miss:

- Initially require a caller-selected, in-range lane (or explicitly default to
  lane 0 and name it in the type/constructor).
- Document that lane-0-only folding matches Volar's current IOP runtime
  simplification and is not a proof that all parallel VOLE lanes were folded.
- Add a later multi-lane aggregation design only after a soundness review;
  do not silently combine lanes.

Challenges must come from an explicit caller-provided transcript/deriver, not
from `gate_index` alone, a constant, or a non-cryptographic RNG. The provider
API should make domain separation and binding to the verifier transcript the
caller's responsibility visible in its type/docs.

### 4. Cover all context variants and public documentation

- Wire the hook through `VoleVerifierContext` and
  `LockedVoleVerifierContext` first.
- Verify `VoleVerifierStorageContext` and
  `LockedVoleVerifierStorageContext` propagate the resulting generic errors
  unchanged through their existing `Verification` variants.
- Confirm `TypedContext for VoleVerifierContext` inherits the hook through its
  ordinary Boolean operation path. This is important because typed add/mul/
  comparison operations synthesize many ANDs.
- Update crate-level docs to state that the regular VOLE online step still only
  derives shares; the hook is an optional observation/folding facility and its
  final IOP proof must be finalized and verified separately by the caller.

## Test plan

1. **No-op regression:** existing `program_roundtrip`, `ferret_stack`,
   `ert_roundtrip`, and locked-parallel tests compile and pass with no feature
   enabled. Assert the original transcript consumption and `HatExhausted`
   behavior remain unchanged.
2. **Single gate:** with `iop-accumulator`, use a `U1`/`Galois` honest VOLE AND
   transcript; execute verifier `bitand`; obtain the provider accumulator; and
   check its relaxed R1CS witness satisfies `volar_iop::fold::and_check_r1cs`.
3. **Chain and ordering:** execute several ANDs, supply distinct transcript
   challenges, and compare the provider state with direct sequential
   `fold_gate` calls over the same gate observations. Verify both hook index
   and `hat` consumption order.
4. **Tampering:** modify a middle `hat` or challenge and prove the final
   accumulator fails the expected relaxed-relation/finalization check. This
   tests that the hook uses the observed verifier values rather than stale
   inputs.
5. **Derived gates:** run `bitor`, `mux`, and a typed arithmetic fixture;
   assert the provider observes every synthesized AND and yields the same
   result as a direct trace of those operations.
6. **Error paths:** test exhausted hats before hook invocation, a deliberately
   failing test hook, and storage-wrapper propagation. Assert no successful
   gate index is reported for either failure.
7. **Locked concurrency:** use the existing locked parallel style with a
   counting/test hook to verify one hook event per consumed `hat`, no duplicate
   indices, and a deterministic serialized transcript/hook order.
8. **Feature matrix:** run `cargo test -p cirrus-volar-vole` and
   `cargo test -p cirrus-volar-vole --features iop-accumulator`; ensure the
   default build remains `no_std` and does not pull the runtime harness.

## Acceptance criteria

- Cirrus can execute a VOLE verifier circuit with the IOP provider and obtain
  a fixed-size `volar_iop::IopAccumulator<Gf128>` independent of gate count.
- The accumulator reflects every executed verifier AND in transcript order and
  matches direct Volar `fold_gate` computation.
- The implementation uses `volar-iop` through its Git dependency, with no
  new path dependency on `../volar` and no dependency on
  `volar-verifier-iop-runtime`.
- Existing no-hook verifier, storage, typed, and locked behavior is preserved.
- The API documents field/lane/challenge limits and does not overstate the
  current lane-0 IOP security model.
