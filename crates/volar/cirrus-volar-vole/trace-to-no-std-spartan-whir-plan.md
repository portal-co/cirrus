# Plan: trace-to-proof, VOLE-state relations, and a `no_std` Spartan-WHIR

> **Status:** implementation plan, not a security approval and not a claim that
> the current code already produces sound end-to-end proofs.
>
> **Baseline:** the crate now has a transparent Mode-A trace/RAM auditor, a
> Mode-B unified R1CS relation, KoalaBear lowering, full unified-witness
> materialization, and a std-only conversion adapter to the pinned upstream
> `spartan-whir` Rust types. The conversion adapter is only a type/shape
> boundary; it is not yet a trace-to-proof protocol.

## 0. Goals and current gaps

The next implementation phase has three deliverables:

1. **Trace wrapper / trace-to-proof:** one deterministic pipeline that takes a
   canonical Boolar execution trace and statement, constructs and validates the
   Mode-B relation and witness, runs the proof lifecycle, and returns a backend
   proof plus verifier-facing statement data.
2. **VOLE verifier-state relations:** a second relation frontend whose rows
   constrain the verifier-side VOLE state transitions (`Q`, `Delta`, and AND
   `hat` values), not only raw Boolean gate values. The gate scheduler must be
   generic so the same circuit schedule can constrain either actual values or
   VOLE correlations of actual values.
3. **`no_std` Spartan-WHIR:** a new allocation-based `no_std` implementation of
   the Spartan-WHIR proving/verifying stack, differentially tested against the
   existing pinned `spartan-whir` dependency as an oracle.

The current gaps are explicit:

- The unified R1CS and witness materializer exist, but there is no public
  one-call trace-to-proof lifecycle.
- RAM permutation challenges are currently supplied to the relation builder.
  That is acceptable for a differential oracle, but not for a sound deployed
  proof: Fiat--Shamir challenges must be derived after the relevant witness
  commitment, and the verifier must independently derive the same values.
- The existing Spartan-WHIR adapter is std-only and does not own setup, prove,
  verify, transcript scheduling, proof serialization, or Solidity fixture
  compatibility.
- The current Mode-B relation constrains actual Boolean values and RAM values.
  It does not yet prove anything about verifier-side VOLE shares or hat
  transcripts.
- The upstream `spartan-whir` implementation is std-only. It can be used as a
  test oracle, but it is not a drop-in `no_std` backend.

## 1. Non-negotiable design rules

These rules apply to all three workstreams.

1. **Statement binding comes first.** Every proof statement must bind the
   relation version, canonical circuit ID, field/profile ID, public input
   order, claimed output order, RAM layout version, VOLE lane/field profile
   when applicable, and backend security profile.
2. **No prover-selected Fiat--Shamir challenges.** RAM permutation challenges,
   sumcheck challenges, PCS challenges, query indices, and proof-of-work
   challenges are transcript outputs under a frozen domain-separation schedule.
3. **Static setup must not depend on per-proof RAM challenges.** The current
   challenge-as-constant R1CS is a test oracle. The deployed relation needs
   challenge variables/public slots or another reviewed static-shape design.
4. **Verifier-selected public values stay verifier-selected.** Public inputs
   and claimed outputs bundled with a proof are untrusted copies. Verification
   receives the expected public vector independently and compares it exactly.
5. **The trace wrapper must fail closed.** Unsupported oracle/RNG/action
   statements, malformed storage witnesses, oversized counts, noncanonical
   field elements, duplicate public bindings, inconsistent traces, and unknown
   profile IDs are errors, not defaults.
6. **`no_std` means no `std` dependency, not low memory by magic.** The new
   backend may use `alloc`. Its memory complexity remains linear in the
   circuit/trace for the initial implementation unless a later streaming
   design explicitly changes that.
7. **An oracle is not a security proof.** Matching upstream `spartan-whir`
   fixtures is strong implementation evidence, but it does not by itself prove
   protocol soundness, parameter security, side-channel resistance, or
   production readiness.
8. **No silent compatibility claims.** Byte compatibility with upstream Rust,
   `sol-spartan-whir`, or any Solidity verifier is true only after checked-in
   cross-language fixtures demonstrate it.

## 2. Deliverable 1: trace wrapper / trace-to-proof

### 2.1 Intended public claim

For a supported Boolar circuit `C`, public inputs `x`, claimed outputs `y`,
and optional storage tables, the wrapper should support the claim:

```text
There exists a canonical execution trace T and RAM witness M such that:
  T is an execution of the circuit identified by circuit_id(C);
  T's public input cells equal x;
  every supported Boolean gate holds;
  every storage read/write obeys the versioned RAM relation;
  C's designated output cells equal y; and
  all proof challenges were derived under the versioned transcript schedule.
```

For the VOLE-state profile in Deliverable 2, the same wrapper shape will
instead carry the VOLE verifier-state claim defined there. The wrapper must not
blur those two statements: they prove different things.

### 2.2 Proposed API shape

Keep the relation builder `no_std` and put std-only upstream proving behind a
feature or an application crate. A plausible core API is:

```rust
pub struct TraceToProof<'a> {
    pub circuit: &'a BCircuit,
    pub public_inputs: &'a [bool],
    pub claimed_outputs: &'a [bool],
    pub primary_witness: &'a [bool],
    pub ram_witness: Option<&'a RamWitness>,
}

pub struct TraceProofArtifacts {
    pub relation: ModeBRelation,
    pub public_instance: ModeBPublicInstance,
    pub unified: UnifiedR1cs,
    pub witness: UnifiedR1csWitness,
    pub public_values: Vec<u32>, // canonical KoalaBear values
}

pub enum RamChallengeSchedule {
    /// Valid only when the circuit has no storage accesses.
    NoStorage,
    /// Challenges are public slots derived from the statement and the initial
    /// witness/PCS commitment under the proof transcript.
    WitnessCommitmentTranscript,
}

pub trait TraceProofBackend {
    type Setup;
    type Proof;
    type Error;

    fn setup(&mut self, artifacts: &TraceProofArtifacts)
        -> Result<Self::Setup, Self::Error>;

    fn prove(
        &mut self,
        setup: &Self::Setup,
        artifacts: TraceProofArtifacts,
    ) -> Result<Self::Proof, Self::Error>;
}
```

The exact names can change. The semantic boundary should not: the core
prepares a validated relation/witness/statement; a backend owns setup,
commitment, transcript, proving, and proof encoding.

### 2.3 Pipeline

The trace wrapper should perform these steps in order:

1. **Canonicalize and validate the circuit.** Recompute `circuit_id`, reject
   unsupported statements, check parameter/output bounds, and check storage
   bounds against the selected RAM profile.
2. **Run the transparent reference evaluator.** Use the existing Mode-A /
   Mode-B Boolean and RAM evaluators to reject an invalid trace before any
   proving work. This is a correctness oracle; it does not mean the final
   proof reveals the trace.
3. **Construct the relation.** Build the Mode-B gate rows and static RAM
   sort/scan rows in the canonical layout.
4. **Materialize the RAM witness.** Pack execution/sorted records and scan
   auxiliaries, and validate them against `RamWitness`.
5. **Run the challenge lifecycle.** The initial implementation supports
   storage-free circuits end to end. Storage-bearing circuits must use the
   challenge-slot redesign in Section 2.4 before they can be proved honestly.
6. **Export and lower the unified R1CS.** Produce canonical KoalaBear rows and
   reject oversized or duplicate terms rather than truncating them.
7. **Materialize the complete field witness.** Use the existing deterministic
   R1CS completion and independently evaluate every lowered row.
8. **Split private witness and public values.** Preserve the frozen
   `[private witness | constant one | public values]` matrix convention and
   `claimed_outputs || public_inputs` public-vector convention.
9. **Invoke the backend.** The backend performs circuit-specific setup,
   witness commitment, Fiat--Shamir, Spartan proving, WHIR/PCS proving, and
   proof serialization.
10. **Return verifier-facing artifacts.** Return the circuit/profile ID,
    canonical public values, verifying-key identifier, proof bytes, and enough
    version metadata for the verifier to select the exact statement. Do not
    return a boolean “valid” from the prover as a substitute for independent
    verification.

### 2.4 Critical prerequisite: RAM challenges in a static Spartan shape

The current implementation encodes `gamma` and `eta` as constants in the
unified R1CS. That creates three problems for a real proof:

1. the challenges must be derived after a witness/PCS commitment;
2. changing constants changes the R1CS shape, while Spartan setup and Solidity
   verifying keys are expected to be circuit/shape-specific; and
3. a verifier cannot accept a shape selected by the prover after seeing the
   proof.

The planned correction is a **challenge-slot relation version**:

- Allocate `gamma[0..5]` and `eta[0..5]` as public field variables in the
  static unified layout.
- Introduce compressed-record auxiliary variables for
  `R_i = K_i + eta * value_i`, with explicit R1CS rows for the extension
  multiplication by the public `eta` variables.
- Use `gamma` and `R_i` linearly in the grand-product recurrence. This keeps
  the R1CS shape fixed while the challenge values vary per proof.
- Extend the proof transcript schedule so the prover first emits the relevant
  witness/PCS commitment, both sides derive `gamma || eta` from
  `domain || relation_id || public_statement || commitment`, and the derived
  values populate the challenge public slots.
- The verifier recomputes those slots independently. The proof must be
  rejected if the public vector supplied with the proof differs from the
  verifier-derived vector.

A zero grand-product denominator should be handled by the standard abort /
re-randomize-commitment behavior, with an explicit transcript/policy record.
Do not silently keep the current “reject or sample again with a counter” idea
without a reviewed verifier-checkable protocol: the verifier generally cannot
recompute private-column denominators from a hiding commitment alone.

This redesign should happen before claiming a storage-bearing trace-to-proof.
A storage-free proof can proceed earlier because it has no RAM permutation
challenges.

### 2.5 Implementation milestones

1. **Serializer hardening.** Replace every fallible count serialized with
   `value as u64` by a checked conversion. Add oversized-count tests.
2. **Core wrapper.** Add the `TraceToProof` input/artifact type and implement
   steps 1--4 and 6--8 for storage-free circuits.
3. **No-storage end-to-end proof.** Wire the artifacts to the existing
   std-only upstream adapter and perform upstream setup/prove/verify behind a
   test-only or app-level feature. The core crate remains `no_std`.
4. **Challenge-slot relation v2.** Add public challenge slots, compressed RAM
   record auxiliaries, and witness materialization for them.
5. **Transcript-integrated storage proof.** Integrate the witness commitment
   and challenge derivation schedule. If upstream does not expose the needed
   schedule hook, do not fake it; implement it in the new backend first and
   keep upstream as a component oracle.
6. **Proof package serializer.** Define versioned proof/statement bytes only
   after the transcript schedule is stable.

### 2.6 Tests and exit criteria

Minimum tests:

- a storage-free circuit proves and verifies end to end;
- a storage write/read circuit proves and verifies after challenge slots are
  implemented;
- Mode-A evaluation, Mode-B Boolean evaluation, unified field evaluation, and
  backend verification all agree on the same trace;
- every mutation of a public input, claimed output, gate output, RAM value,
  sorted-table entry, challenge slot, or proof byte rejects;
- public values bundled with a proof cannot replace the verifier's expected
  public values;
- oversized serializer counts reject instead of truncating;
- the same trace produces byte-identical relation and statement artifacts on
  repeated runs.

Exit criterion: one command or one public function takes a supported trace to
a verified proof without exposing caller-selected Fiat--Shamir challenges.

## 3. Deliverable 2: relations over VOLE verifier state

### 3.1 Why this is a separate relation

The current Mode-B relation proves constraints over actual Boolean wire
values. A VOLE verifier-state relation proves a different statement:

```text
A canonical verifier-side trace of VOLE shares and AND hats is internally
consistent with the declared circuit topology, lane profile, input correlation
commitments, output claims, and storage witnesses.
```

This may be the right outer statement when the object being audited is a
previous verifier execution. It is not interchangeable with the raw-value
statement. Constraining only a list of AND correlations would repeat the old
“some computation happened” defect unless the relation also binds the circuit
topology, input/output positions, lane count, and session/statement
commitments.

### 3.2 Statement components

A versioned VOLE-state public statement should contain at least:

```text
relation/profile version
canonical circuit_id
VOLE scalar-field ID and encoding version
lane count and lane order
input commitment or input-share statement
hat transcript commitment
claimed output opening or output-share statement
storage boundary statement, when storage is present
session/transcript binding supplied by the outer protocol
Delta disclosure mode:
  public Delta, or
  a commitment to secret Delta plus a reviewed setup/consistency proof
```

The first implementation profile should use a **public verifier-state
transcript**, matching the current decision that verifier-side trace values may
be public. A secret-`Delta` profile is a later cryptographic composition, not
an accidental consequence of hiding a field element.

### 3.3 Generic builder architecture

Refactor the current gate lowering into a scheduler generic over a value
representation. The same Boolar statement stream should drive both frontends:

```rust
pub trait ConstraintBuilder {
    type Var;

    fn constant(&mut self, value: i64) -> Self::Var;
    fn enforce_mul(&mut self, a: Self::Var, b: Self::Var, out: Self::Var);
    fn enforce_linear_zero(&mut self, terms: &[(i64, Self::Var)]);
    fn enforce_boolean(&mut self, value: Self::Var);
}

pub trait TraceSemantics<B: ConstraintBuilder> {
    fn emit_statement(&mut self, statement: &BIrStmt, builder: &mut B);
}

pub struct ActualBooleanSemantics;
pub struct VoleVerifierSemantics<const LANES: usize>;
```

The production API may differ, but it must preserve these properties:

- statement order and wire numbering come from the canonical circuit;
- the scheduler is shared by actual-value and VOLE-correlation modes;
- neither mode can silently skip storage, external, RNG, or action statements;
- generated variable indices are deterministic and documented; and
- behavior-changing profile choices are serialized in the relation/profile ID.

### 3.4 Actual-value semantics

`ActualBooleanSemantics` is the current Mode-B lowering, moved behind the
generic scheduler without changing its rows:

```text
Booleanity: x * (x - 1) = 0
And:        a * b = c
Xor:        p = a*b; a + b - 2p = c
Or:         p = a*b; a + b - p = c
Not:        1 - a = c
```

This gives an immediate differential target: the old `ModeBRelation` row list
and the generic builder's row list must be identical for every fixture.

### 3.5 VOLE verifier-correlation semantics

For lane `l`, let `Delta[l]` be the verifier offset, `Q_w[l]` the verifier's
share for wire `w`, `V_w[l]` the prover mask/share component, `x_w` the actual
bit lifted to the VOLE field, and `hat_g[l]` the prover-sent value for AND
gate `g`.

The correlation relation should include:

1. **Input/wire setup correlations**

   ```text
   Q_w[l] = V_w[l] + x_w * Delta[l]
   x_w * (x_w - 1) = 0
   ```

   The exact meaning of `V_w` depends on the selected commitment/setup ABI and
   must be versioned.

2. **Constants**

   ```text
   Q_zero[l] = 0
   Q_one[l]  = Delta[l]
   ```

3. **Linear gates**

   ```text
   Xor: Q_c[l] = Q_a[l] + Q_b[l]
   Not: Q_c[l] = Delta[l] - Q_a[l]
   ```

   `Or` is emitted as Xor plus And, or as the equivalent explicitly reviewed
   set of rows.

4. **AND correlation**

   ```text
   Q_a[l] * Q_b[l] + hat_g[l] = Q_c[l] * Delta[l]
   ```

   This is the soundness-oriented multiplication form. The current
   `derive_and_q` equation with `Delta^-1` is a deterministic replay formula;
   by itself it does not independently check a prover-supplied output share.

5. **Output claims**

   Output wires must be connected to public actual bits or public/committed
   VOLE openings using the same setup correlation. A claimed Boolean output
   `y_j` requires the relation to bind `x_out = y_j` and the corresponding
   `Q_out/V_out/Delta` correlation.

6. **Storage witnesses**

   Storage read/write VOLE values need the same correlation treatment and
   must connect to the RAM relation. This remains fail-closed until the exact
   VOLE storage ABI is selected.

7. **Topology and transcript binding**

   Gate indices, operand wire IDs, output wire IDs, lane order, and the hat
   transcript order are part of the relation/profile statement. The proof must
   not accept a permuted or truncated correlation list.

### 3.6 Field representation decision

There is no field embedding from characteristic two into KoalaBear. A VOLE
trace over `GF(2^128)` cannot be mapped into the prime-field R1CS as a native
field element without losing the actual VOLE algebra.

The plan therefore separates two profiles:

1. **Native-proof-field VOLE profile (initial).** Instantiate the VOLE scalar
   field with the same KoalaBear field used by the proof, or another field for
   which a reviewed exact representation in the R1CS exists. This keeps
   `Q`, `V`, `Delta`, and `hat` arithmetic native and small. It is a new VOLE
   field profile and requires a protocol/security review; existing binary VOLE
   parameter/security claims do not automatically transfer.
2. **Existing binary-field VOLE profile (later).** Preserve existing
   `GF(2^k)` traces by bit/byte-decomposing field elements and proving the
   binary-field add/mul/invert operations as explicit Boolean/byte gadgets in
   the prime-field R1CS. This is much larger and needs canonical limb/range
   constraints, but it does not change the original VOLE protocol.

Do not claim support for existing `GF(2^128)` traces until profile 2 exists
and has differential tests against the current verifier.

### 3.7 Implementation milestones

1. **Define trace records.** Add canonical verifier-state records for input
   commitments, linear-gate derivations when needed, AND gate records, output
   openings, and storage witnesses. Include lane count/order and field profile.
2. **Refactor the scheduler.** Move the current Boolean lowering behind the
   generic builder and prove row-for-row compatibility.
3. **Implement actual-value mode.** Keep all existing Mode-B tests passing.
4. **Implement native-field VOLE mode.** Add the correlation rows for a
   KoalaBear VOLE field profile, with a toy deterministic setup source used
   only in tests.
5. **Integrate the verifier hook.** Add a collector implementing
   `VoleVerifierHook` that records canonical AND records and validates gate
   order. The hook remains an observer; it must not alter verifier behavior.
6. **Connect output and storage claims.** Define and test the exact public
   output/opening ABI and the storage correlation ABI.
7. **Optional binary-field profile.** Add bit/byte decomposition gadgets only
   after the native-field profile is stable.

### 3.8 Tests and exit criteria

Minimum tests:

- old and generic actual-value relations are byte-identical for all existing
  fixtures;
- honest native-field VOLE traces satisfy the correlation relation;
- mutations to `Q`, `V`, `Delta`, `hat`, lane order, gate order, input
  correlation, or output claim reject;
- a truncated hat transcript and an extra hat both reject;
- constants, Xor, Not, And, and derived Or are tested on every Boolean input
  combination;
- storage-bearing VOLE traces fail closed until their ABI is implemented;
- the same trace is evaluated by the actual-value relation and, when the
  profile allows, correlated with a corresponding VOLE-state relation.

Exit criterion: a caller can choose one canonical circuit schedule and emit
either the actual-value R1CS or the VOLE verifier-state R1CS, with tests
showing both bind the intended statement rather than an unstructured list of
equations.

## 4. Deliverable 3: new `no_std` Spartan-WHIR implementation

### 4.1 Scope

Create a new crate, tentatively `cirrus-spartan-whir`, with:

```toml
[features]
default = []
std = []          # tests/tools only
oracle-tests = ["std", "dep:spartan-whir"]
```

The core uses `core` + `alloc`, no `std`, no threads, no process RNG global,
no `OnceLock`-based global state, no file I/O, and no tracing facade that
requires `std`. Randomness and time are caller-supplied traits. Any optional
dependency must be audited for `no_std` support before use.

The initial protocol profile should be narrow:

- KoalaBear base field;
- the quintic extension profile required by the current target;
- R1CS over the frozen `[private | one | public]` column convention;
- no-ZK DirectSparse Spartan-WHIR first;
- caller-supplied deterministic RNG for tests and caller-supplied cryptographic
  RNG for production experiments;
- explicit security profile with no default that silently exceeds the reviewed
  parameter basis.

Full ZK, Spark matrix closing, parallelism, GPU support, and Solidity fixture
export are later profiles. They are not prerequisites for the first honest
`no_std` proof.

### 4.2 Module plan

```text
src/
  lib.rs             # no_std crate root, public API, profile/errors
  field.rs           # KoalaBear base-field arithmetic and canonical encoding
  extension.rs       # quintic extension arithmetic and basis encoding
  polynomial.rs      # MLE, eq polynomials, sparse/dense operations
  poseidon.rs        # hash/compression parameters and test-vector hooks
  merkle.rs          # domain-separated commitments and authentication paths
  transcript.rs      # typed Fiat--Shamir transcript and challenge schedule
  r1cs.rs            # shape, witness, sparse matrices, satisfaction checks
  spartan.rs         # sumcheck prover/verifier and Spartan protocol
  whir.rs            # WHIR PCS prover/verifier
  protocol.rs        # setup/prove/verify orchestration and key types
  serialization.rs   # versioned canonical encodings
  error.rs           # exhaustive, non-String core errors
```

Keep upstream-compatible type names only where they do not imply byte or
transcript compatibility before fixtures prove it.

### 4.3 Implementation slices

#### Slice 0: scaffold and policy

- Create the crate and workspace wiring.
- Enforce `#![no_std]`, exhaustive errors, and missing-doc warnings.
- Add a CI/check target for a bare-metal triple such as
  `thumbv7em-none-eabihf` or `riscv32imac-unknown-none-elf`, depending on the
  installed toolchain.
- Define `RngCore`-like caller traits without requiring a system RNG.
- Add deterministic test RNG only under `std`/tests.

#### Slice 1: field and extension arithmetic

- Implement canonical KoalaBear arithmetic, inversion, powers, batch inverse,
  and encoding.
- Implement the quintic extension and all reduction rules.
- Differential-test against the pinned Plonky3/upstream implementation on
  fixed vectors and randomized sequences.
- Test rejection of noncanonical serialized base-field elements.

#### Slice 2: polynomial and R1CS substrate

- Implement dense/sparse vector operations, MLE construction, equality
  polynomials, matrix-vector multiplication, and padding rules.
- Import or mirror the existing `R1csShape`/`R1csWitness` validation logic.
- Compare matrices, MLEs, evaluations, and satisfaction results with upstream.

#### Slice 3: hash, Merkle, and transcript

- Implement the selected Poseidon configuration and domain-separated Merkle
  tree.
- Implement a typed transcript with explicit observe/sample labels.
- Freeze a transcript schedule before implementing the prover.
- Generate byte-level fixtures from upstream and compare digest, challenge,
  query, and proof-of-work sequences. Any mismatch must either be fixed or be
  declared a deliberately different protocol profile ID.

#### Slice 4: Spartan sumcheck and R1CS proof layer

- Implement prover and verifier sumcheck rounds.
- Implement Spartan's R1CS evaluation reduction for DirectSparse mode.
- Use upstream as the round-polynomial, challenge, accept/reject, and error
  oracle.
- Mutation-test every round polynomial, challenge, evaluation, matrix entry,
  public input, and witness element.

#### Slice 5: WHIR PCS

- Implement Reed--Solomon/MLE commitment, folding, Merkle openings, query
  derivation, proof-of-work, and verification.
- Freeze parameter selection and soundness assumption in a typed
  `SecurityProfile`.
- Differential-test commitments, openings, query schedules, and whole PCS
  proofs against upstream where upstream exposes compatible fixtures.
- Reject malformed paths, duplicate/out-of-range queries, wrong domain sizes,
  wrong folding counts, and noncanonical field encodings.

#### Slice 6: setup/prove/verify API

A minimal target API is:

```rust
pub struct NoStdSpartanWhir;

pub struct ProvingKey { /* shape/profile-specific */ }
pub struct VerifyingKey { /* shape/profile-specific */ }
pub struct Proof { /* versioned protocol objects */ }

impl NoStdSpartanWhir {
    pub fn setup(
        shape: &R1csShape,
        profile: SecurityProfile,
    ) -> Result<(ProvingKey, VerifyingKey), Error>;

    pub fn prove<R: CryptoRngSource>(
        pk: &ProvingKey,
        witness: &R1csWitness,
        public_values: &[Field],
        rng: &mut R,
    ) -> Result<Proof, Error>;

    pub fn verify(
        vk: &VerifyingKey,
        expected_public_values: &[Field],
        proof: &Proof,
    ) -> Result<(), Error>;
}
```

The proof lifecycle must expose a controlled way to insert the RAM challenge
schedule from Section 2.4. Do not hide that in a convenience function that
accepts caller-selected challenges.

#### Slice 7: serialization and fixtures

- Define versioned canonical proof bytes.
- Add round-trip tests and trailing-byte/noncanonical-element rejection.
- Generate checked-in fixtures from upstream and from the new implementation.
- Only after those fixtures pass, attempt `sol-spartan-whir` consumption.

#### Slice 8: hardening

- Bounds and allocation limits for adversarial shapes/proofs.
- Deterministic behavior under malformed lengths and duplicate sparse entries.
- Constant-time review for secret-dependent field operations if private
  witnesses are later supported.
- Side-channel documentation: the initial no-ZK profile is transparent and
  not a privacy mechanism.
- External cryptographic review before any production-security statement.

### 4.4 Oracle-test strategy

The existing pinned `spartan-whir` dependency remains the implementation
oracle. Add a std-only test crate or feature that can call both
implementations.

Required oracle layers, from smallest to largest:

1. field addition, subtraction, multiplication, inversion, exponentiation,
   extension reduction, and canonical encodings;
2. Poseidon compression/hash, Merkle roots/paths, transcript challenge byte
   sequences, and query indices;
3. sparse matrix canonicalization, padding, MLEs, matrix-vector products, and
   R1CS satisfaction;
4. each sumcheck round polynomial and verifier decision;
5. WHIR commitment roots, folding intermediate values, openings, and query
   verification;
6. setup-key metadata, whole proof verification, and proof serialization;
7. cross-direction tests: upstream verifies new proofs and the new verifier
   verifies upstream proofs, if the selected profile is intended to be
   transcript-compatible.

Use deterministic RNG and fixed statements for fixtures. Add property tests
for small random R1CS instances. Record any intentional divergence as a new
profile/version, never as an undocumented compatibility claim.

### 4.5 Security gates

Do not advertise the new implementation as production-ready until all gates
pass:

- protocol mapping from the Spartan and WHIR papers to the implemented
  transcript is independently reviewed;
- soundness assumption and parameter calculator are documented and tested;
- Fiat--Shamir domain separation and challenge ordering are frozen;
- Poseidon parameters and implementations have test vectors and review;
- denial-of-service bounds for untrusted proofs are implemented;
- oracle and mutation suites pass;
- external cryptographic review has explicitly covered the selected profile.

The current upstream configuration includes multiple possible security
assumptions. The new implementation must not inherit a default merely because
upstream exposes it. Choose and document the profile deliberately.

## 5. Recommended execution order

The lowest-risk order is:

1. **Harden serializers and wrap the existing artifacts.** This makes the
   current relation/witness pipeline easy to test without changing protocol
   semantics.
2. **Finish no-storage trace-to-proof through the existing std adapter.** This
   proves the wrapper boundary while avoiding the RAM challenge redesign.
3. **Refactor to the generic constraint scheduler.** Keep actual-value rows
   byte-identical and establish the API needed by VOLE-state mode.
4. **Implement challenge-slot RAM v2.** This is required before any honest
   end-to-end storage proof with static setup.
5. **Implement native-field VOLE verifier-state mode.** Start with a public
   verifier-state profile and a native proof-field VOLE lane profile.
6. **Build the `no_std` backend in oracle-tested slices.** Field/hash/R1CS
   parity comes before sumcheck; sumcheck before WHIR; WHIR before end-to-end.
7. **Move storage trace-to-proof to the `no_std` backend.** The upstream
   dependency remains the oracle until the new backend has complete fixtures.
8. **Only then consider Solidity fixtures, full-ZK, Spark, and binary-field
   VOLE trace support.** These are separate profiles and should not block the
   first honest `no_std` no-ZK proof.

## 6. Risk register

| Risk | Severity | Honest assessment | Planned mitigation |
|---|---:|---|---|
| RAM challenges encoded as shape constants | Critical | Current code is a differential oracle, not a sound deployed proof schedule. | Public challenge slots plus transcript-integrated derivation. |
| Verifier cannot recompute private-column denominator failures | High | A naive resampling counter is not verifier-checkable. | Abort/re-randomize commitment or specify a reviewed checkable policy. |
| VOLE field mismatch | Critical | `GF(2^128)` does not embed into KoalaBear. | Native-field profile first; explicit binary-field gadget profile later. |
| Secret `Delta` composition | High | Hiding `Delta` without a setup/commitment proof proves little. | Public-state profile first; secret profile only after cryptographic design. |
| Ideal VOLE setup assumption | High | Correlation rows do not prove the COT/setup was generated honestly. | Bind an explicit setup transcript/commitment or document ideal-setup assumption. |
| Upstream std implementation divergence | Medium | The oracle may have behavior not promised by its public docs. | Pin revision, byte-level fixtures, profile IDs for intentional divergence. |
| WHIR parameter/soundness choice | High | Different assumptions give different security claims. | Typed security profile and external parameter review. |
| `no_std` resource usage | Medium | `no_std` does not reduce proof memory or prover complexity. | Publish allocation model, limits, and benchmarks on target classes. |
| Solidity compatibility | High | Rust proof verification does not imply `sol-spartan-whir` compatibility. | Separate fixture milestone with exact bytes and transcript. |
| Oracle/RNG/action statements | Medium | Current relation deliberately rejects them. | Keep fail-closed until versioned ABIs are designed. |
| Side channels | Medium | The initial trace is public/no-ZK; later private use changes requirements. | Explicit privacy boundary and constant-time review for private profiles. |

## 7. Definition of done

The combined work is done only when all of the following are true:

1. A storage-free Boolar trace can be converted to a proof and independently
   verified through one public trace-to-proof API.
2. A storage-bearing trace uses transcript-derived RAM challenges and a static
   challenge-slot R1CS shape; no caller-selected permutation challenges remain
   in the proving path.
3. The generic scheduler emits both the actual-value relation and the VOLE
   verifier-state relation from the same canonical circuit schedule.
4. The VOLE-state profile explicitly declares its field, lane, disclosure,
   setup, output, and storage assumptions.
5. The new Spartan-WHIR crate builds for a `no_std` target and passes field,
   transcript, R1CS, sumcheck, PCS, and end-to-end differential tests against
   the pinned upstream oracle.
6. Malformed statements, witnesses, challenges, proofs, and public vectors
   reject deterministically.
7. The documentation states the exact security profile and explicitly avoids
   ZK, Solidity, binary-VOLE, and production-security claims until those
   separate profiles are implemented and reviewed.
