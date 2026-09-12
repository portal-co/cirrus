# Corrections: bind the IOP proof to the circuit statement and wire trace

> **Status:** research draft / design review required. Nothing here authorizes an implementation or a production-security claim.
>
> **Scope:** both current Volar IOP paths: (1) weave-time `IopSink` plus `volar-verifier-iop-runtime`, and (2) Cirrus runtime `IopAccumulatorHook`. They feed the same `volar_iop::fold::IopAccumulator`, so they share the same statement-binding defect.
>
> **Decision needed:** do we want (A) a transparent, publicly verifiable *succinct argument* for a circuit evaluation, or (B) a non-succinct authenticated audit trace that opens every used wire? These are different protocols and costs.

## 1. The defect, precisely

The current IOP finalizer proves only that a final compact relaxed-R1CS object `(W, E, u)` satisfies the three-row generic AND relation. It does **not** make the verifier check, as one statement, that:

1. a particular canonical circuit `C` was selected;
2. particular public inputs, output claim, and public/committed state boundaries were selected;
3. the records folded into the accumulator are wire values from one evaluation of `C`; or
4. the claimed output is `C`’s designated output wire(s).

Once individual gate records are discarded, the compact `(W,E,u)` identifies neither wire identities nor adjacency, gate count, circuit, or output wires. A proof of the final relaxed relation therefore establishes “some folded gate relation is satisfiable,” not “this computation produced this claimed output.”

Adding a circuit or output digest to the old finalization message alone is insufficient: an unrelated satisfying accumulator can be paired with metadata unless the proof establishes the linkage.

The existing Volar design document independently records that non-interactive per-gate fold soundness is open. The correction must therefore change the proved statement rather than attach an unaudited root.

## 2. What a Merkle root does—and does not—provide

Let `T` be a canonical, padded serialization of the execution: inputs, witness/private inputs, all wire values (or an equivalent trace table), storage/event records, and output values. Define:

```text
trace_root = MerkleCommit(EncodeTrace(C, public_statement, witness))
```

With an unambiguous encoding and a collision/second-preimage-resistant hash, the root is a computationally binding commitment to `T`. An opening `(index, value, path)` proves that *this value* is in *this position* under that root.

That is necessary but not a circuit proof by itself:

- Opening every wire and checking every gate proves the trace consistency, but sends `Θ(number of wires)` values. Batched multi-openings reduce duplicated paths, not the linear witness values. This is an **audit proof**, not succinct.
- Opening a static circuit-selected subset proves only those constraints; an adversary can violate an unopened gate unless a separate randomized algebraic/IOP check covers all constraints.
- Root commitment after Fiat–Shamir queries is unsound: the prover can tailor a trace to the sampled checks. Commit the complete trace before deriving tests of it.

“Lazy root” is sound only if it means the prover builds or can deterministically recompute the complete committed tree before absorbing its root, then derives requested paths lazily. It cannot mean leaves are selected after the root or queries are known.

## 3. Corrected public statement

Before proof challenges, canonicalize and transcript-bind this `ExecutionStatement`:

```text
protocol_version
hash_suite_id + exact parameter-set identifier
canonical_circuit_id = H(EncodeCanonicalCircuit(C))
trace_layout_id + trace length/padding rule
public-input commitment or canonical public-input bytes
claimed public outputs
initial/final state roots (when applicable)
VOLE field/lane configuration and session/transcript binding
```

The desired claim is:

```text
There exists a trace T committed by trace_root such that:
  T has the declared layout for C;
  T's public-input cells equal the public statement;
  every gate, transition, and storage constraint of C holds on T;
  designated output cells equal claimed_public_outputs; and
  declared state boundaries equal the statement's state roots.
```

`canonical_circuit_id` must include wire numbering, gate order, input/output positions, type/lane layout, constants, and protocol/compiler version—not source text or a compiler name alone. The verifier independently canonicalizes `C`, or retrieves it by an authenticated identifier.

### Privacy boundary

A Merkle root is binding, not hiding. Raw openings disclose values. The current outer IOP is explicitly non-ZK. If trace values include private inputs or VOLE-derived secrets, this needs a ZK-capable proof/commitment layer or a deliberately public trace boundary; this document makes no ZK claim.

## 4. Two delivery modes

### Mode A — complete authenticated audit (simple; not succinct)

1. The prover constructs canonical `T`, builds the tree, publishes `trace_root`, and binds `ExecutionStatement || trace_root` into the transcript.
2. A circuit-specific planner enumerates every wire cell needed for every gate, output, input, and memory transition, deduplicated by leaf index.
3. The prover gives all requested values and a batched Merkle multi-opening.
4. The verifier checks every gate, transition, output, and boundary.

This directly realizes “opening paths are decided by the specific circuit.” It is a valuable correctness oracle and test harness. It is not a succinct proof, because all needed witness values are revealed.

### Mode B — trace-committed IOP/STARK-style argument (required for succinctness)

Use `trace_root` as an oracle commitment and have an actual polynomial/IOP prove the entire trace relation:

1. Fix a canonical trace table: wire/gate rows, opcode/constant metadata, output selectors, and state/memory-transition columns. Group values normally queried together into one leaf/row.
2. Commit every trace-oracle column before query challenges. Typed length-prefix-bind the statement, circuit ID, hash suite, roots, and all oracle roots.
3. Use a reviewed constraint reduction (for example sumcheck/AIR) plus a low-degree proof such as FRI. These random tests, not Merkle paths alone, cover every gate/transition.
4. Fiat–Shamir derives queries only after commitments. Open query rows, neighbor rows, and FRI/codeword positions with batched paths.
5. Check paths, proximity/low-degree proof, sampled local constraints, boundary/output constraints, and transcript schedule.

This is the committed execution-trace pattern used by STARKs. The existing fixed-size accumulator can remain only as an optimization whose relation to the committed trace is itself proven; it cannot remain the sole Phase-1 artifact.

**Safe sequence:** build Mode A first, use it to fix layout/canonicalization/output semantics and test adversarial mutations, then replace exhaustive checks with a reviewed Mode-B IOP. Do not present the replacement as a small patch to the existing finalizer.

## 5. The two handlers must share one binding interface

| Site | Required correction |
|---|---|
| Weaver `IopSink` / `volar-verifier-iop-runtime` | Emit/consume canonical trace rows and the commitment schedule. Finalization takes `ExecutionStatement`, `trace_root`, and trace proof, not only `(W,E,u,mem_acc_in,out)`. |
| Cirrus `IopAccumulatorHook` | Record observed AND values in the same canonical layout, or obtain them from the shared trace provider. Specify a stable `gate_index → row` mapping. Finalize against the same statement/root. |

A selected VOLE lane is a separate gap. A lane-0 trace/fold does not prove all parallel lanes. The statement must declare one-lane semantics, commit/check every required lane, or use a reviewed multi-lane aggregation argument.

## 6. Poseidon2: applicability correction

### Conclusion

**Do not instantiate ordinary Poseidon2 over Volar's binary tower field.** Its original construction/parameter security analysis is for prime fields. Reusing its S-box, matrices, or round counts over `GF(2^128)` is an unreviewed new permutation and provides no inherited Poseidon2 security claim.

The relevant candidate is **Poseidon2b**, a separately designed binary-field variant, not “Poseidon2 on a binary field.” The public Poseidon2b reference repository identifies it as *“Poseidon2b: A Binary Field Version of Poseidon2”* and includes Binius integrations and concrete `GF(2^128)` instances. Its reference `poseidon2b_x7_128_512.rs` uses state width `t=4`, 8 full rounds, 58 partial rounds, and an `x^7` S-box. Those facts are not permission to reuse them blindly: field representation, constants, width/rate, padding, domain tags, security target, and parameter version must match exactly.

The natural target field is Volar’s `Gf128`, which is a binary tower rooted at the same `GF(2^8)` representation. It is *algebraically compatible in characteristic*, but compatibility is not byte-level interoperability with Binius’s `BinaryField128b`; establish an explicit field-isomorphism/encoding relation and test vectors before claiming the Poseidon2b parameters apply.

### Why `x^7` needs its own check

A power S-box `x ↦ x^7` is a permutation on `GF(2^n)` precisely when `gcd(7, 2^n - 1) = 1`. For `n=128`, this holds: `2^128 mod 7 = 2`, so `7` does not divide `2^128 - 1`. This only establishes invertibility—not differential, algebraic, interpolation, invariant-subspace, or implementation security. Use the selected Poseidon2b paper’s complete parameter/security analysis, not this arithmetic condition, as the security basis.

### Do we need Poseidon2b here?

No. Hashing happens *outside* the arithmetic constraints in the proposed transparent Merkle commitment. SHA3-256 already backs `volar-iop` Merkle commitments and is a much lower-risk immediate choice. Poseidon2b is compelling only if we intend to prove/hash Merkle computations **inside a binary-field proof system**, or need shared primitive/circuit efficiency. Replacing SHA3 merely for native Merkle hashing increases review burden without itself improving statement binding or succinctness.

If Poseidon2b is chosen later:

- define a `HashSuite` with a fixed, versioned Poseidon2b parameter-set ID;
- specify leaf/node/empty/padding/domain encodings; never use the raw permutation directly as an unframed variable-length hash;
- domain-separate trace leaves, internal nodes, circuit IDs, public statement, storage roots, and Fiat–Shamir transcript; and
- require official test vectors plus cross-implementation tests before use.

## 7. Security argument sketch and assumptions

For Mode A, acceptance implies the claimed computation **assuming**:

1. canonical encoding/layout maps each circuit role to one unique trace location;
2. Merkle binding holds for the selected domain-separated hash;
3. verifier exhaustively checks all required opened locations and every circuit/state constraint; and
4. public statement, root, and proof are bound to the same session/context.

A false accepted output then supplies either a trace violating a locally checked relation (contradiction) or two different values/traces accepted under the same root (breaks binding). This is deliberately linear-size and transparent.

For Mode B, add the soundness error and assumptions of the exact IOP/AIR/sumcheck/low-degree protocol, Fiat–Shamir in the random-oracle model, Merkle binding, and the correct composition theorem. The combined error must be parameterized and documented. No such theorem or concrete parameters exist in the present Volar IOP code; its current Reed–Solomon spot-checking is not a substitute for a reviewed whole-trace constraint IOP.

## 8. Open questions for joint decision

1. Is Mode A an acceptable initial correction, even though it is not succinct and exposes opened values, to make the statement semantics executable and testable?
2. Which values are public, committed-but-hidden, and permitted to be opened? Is a transparent outer proof actually valid for the intended VOLE deployment?
3. What is the canonical circuit artifact: existing BIR/LIR, a frozen serialized gate list, or another versioned IR? Who authenticates its ID?
4. What trace granularity is desired: one leaf per wire, per gate row, or packed fixed-size blocks? This determines memory, proof paths, and concurrency behavior.
5. Do we require all VOLE lanes, and if so what is the intended multi-lane proof relation?
6. Are storage roots part of the claimed result, and what exact state-transition model must be proved?
7. Do we need in-proof hashing soon enough to justify Poseidon2b, or should SHA3-256 remain the hash suite while the statement-binding protocol is stabilized?
8. If Poseidon2b is desired: which pinned paper/repository revision and exact instance receives a formal parameter/paper-binding review? The public search results currently contain inconsistent ePrint identifiers, so resolve this before citation or implementation.

## 9. Decisions recorded in this review

The following resolves the corresponding questions in Section 8 for the initial prototype:

- **Mode A is accepted initially.** It is the viability/correctness baseline; it is not represented as a succinct proof.
- **Trace values may be public.** They are verifier-side artifacts of a previous interactive proof, invalid in a live context, and need only be internally consistent. This removes the immediate ZK requirement, but does not remove binding, freshness/session binding, or trace-integrity requirements.
- **Circuit artifact:** canonical Boolar IR / Volar IR. The ID is a hash of a frozen, canonical serialization of the fully lowered IR, including the items listed in Section 3.
- **Trace layout:** one Merkle leaf per wire. The layout must also allocate unambiguous leaves for inputs, constants when represented as values, outputs, state roots, and memory-access records. A fixed `WireId -> leaf index` map belongs to the circuit artifact.
- **VOLE lanes:** all lanes are part of the statement. The same externally specified coefficient/challenge is applied across lanes as requested. The proof format must carry the lane count/order and reject lane omission, reordering, or a different coefficient schedule. This supersedes the unsafe lane-0-only prototype behavior for a final statement-binding proof.
- **State:** initial and final storage roots are claimed public results. “Valid state transition” means valid memory access; its exact read/write semantics must be made canonical as described below.

### Mode-A shape after these decisions

For each circuit (or later, each chunk), the prover derives the canonical all-wire vector, computes the one-leaf-per-wire tree before challenges, and publishes the root. The verifier’s static IR planner enumerates every gate’s input/output wire leaves, designated IO leaves, and every required memory-access leaf. A batched multi-opening then supplies all of them. The verifier replays Boolar/Volar IR and rejects unless every gate and every memory access is valid, all lanes have the declared same-coefficient treatment, and the final roots/output leaves match the statement.

This can be streamed at the prover and verifier despite linear total communication: values and authentication nodes need not coexist in memory. It does not eliminate the total wire-value bandwidth.

### Implementation status (initial vertical slice)

`cirrus-volar-vole` now starts this reference layer in `src/trace_audit.rs`: it commits one Boolean leaf per canonical Boolar wire with an indexed, length-bound, domain-separated SHA3-256 Merkle tree; provides every canonical wire opening; checks each path; replays `Zero`, `One`, `And`, `Or`, `Xor`, and `Not`; and binds the first wire leaves and designated output leaves to public input/output claims. It also has the initial full-trace RAM permutation witness: execution-order `StorageRead`/`StorageWrite` (and `pre_init`) records are deterministically derived from the IR plus authenticated wire values, duplicated in an address/time-sorted table, checked for exact permutation equality, and checked so each read observes the prior write at that `(storage, lane, address)` or the zero default. This is transparent and linear-size, not yet a succinct permutation argument. External/action statements remain unsupported until their canonical trace ABI is specified.

## 10. Alternative: IOR/IVC over chunked valid-state transitions

### Short answer

**This is a good longer-term alternative to sending an entire execution transcript, but not a shortcut around defining and proving the trace relation.** More precisely, the idea is an **incrementally verifiable computation (IVC)** construction built from a reviewed folding/accumulation scheme. An **interactive oracle reduction (IOR)** can be an appropriate modular way to build/compile the underlying accumulator, but IOR is not by itself the state machine or the memory proof.

The clean formulation is not “reduce valid input to valid state” informally. Make initialization an ordinary first state transition:

```text
R_step(program_id, S_before, chunk_descriptor, S_after; chunk_witness) = 1
```

where `S` contains at least the authenticated memory root and the execution-control state (for example PC, call/continuation state, chunk number, public output accumulator, lane/VOLE transcript state, and protocol version). Then define a distinguished genesis state `S_0` and an **initialization chunk** that checks the public input and installs the initial memory/input state:

```text
R_step(program_id, S_genesis, Initialize(public_input), S_initial; witness) = 1.
```

Every later chunk proves the same relation from `S_i` to `S_(i+1)`. Folding/accumulation then establishes one accumulated claim that all accepted chunk transitions form a chain beginning at `S_genesis`; the final public instance exposes the claimed final root/output/control state. This makes input validity part of the same transition relation rather than an unaudited side relation.

### What it buys

- The final proof/accumulator and public state can be independent of the number of chunks (subject to the chosen folding scheme and any final compression).
- A prover can process a large computation incrementally, retain only current execution state plus folding state, and discard prior chunks after their contribution is folded.
- Chunks can correspond to batches of Boolar/Volar IR gates, avoiding a materialized all-wire execution transcript as the final proof artifact.
- The initialization transition gives a crisp security boundary: accepted final state is reachable from the uniquely defined public input/genesis state by valid chunk transitions.

It is therefore a better *target architecture* than Mode A if trace storage/communication is the concern.

### What it does not buy

- A folding scheme does not make an arbitrary program relation automatically correct. The chunk circuit/relation must still constrain every Boolar/Volar gate, output, lane calculation, and memory operation in that chunk.
- A Merkle root only authenticates individual memory cells. It does not by itself prove global RAM semantics across time. Each read must prove that it reads the prior value for that address; each write must update exactly that address; and all accesses must be ordered or otherwise subject to a sound memory-consistency argument.
- “One relation” does not necessarily mean one fixed circuit. A fixed Boolar/Volar circuit can use a uniform chunk relation. General dynamic programs need either a universal VM/ISA relation, a predeclared family of step circuits with authenticated dispatch, or a non-uniform IVC scheme. The selected model changes the trusted circuit artifact and performance materially.
- Folding reduces the retained/final transcript; it does not make proving free. The prover must still execute and constrain all chunks. Standard folding systems also generally need a final succinct compression proof for constant/logarithmic third-party verification.
- The present custom Volar accumulator is not a substitute. It folds a narrow AND relation, has no trace/circuit linkage, and cannot simply be relabeled as an arbitrary-program folding scheme.

### Required state and valid-memory-access relation

A proposed state tuple is:

```text
S = (
  protocol_version, program_id, chunk_index, pc_or_continuation,
  memory_root, public_input_digest, public_output_digest,
  vole_lane_count, coefficient_schedule_id, transcript_context
)
```

The exact field encoding must be fixed and all fields must be transcript/public-instance bound. For an access `(address, op, value, old_value)`, a chunk witness requires a Merkle authentication path under `memory_root_before`. A read checks `value = old_value`; a write checks the old leaf and derives `memory_root_after` by replacing exactly that leaf. Multiple accesses must be sequenced, feeding each operation’s new root into the next operation. This direct authenticated-memory approach is simple and works for a prototype, but costs `O(log M)` hash work/witness per access.

For higher throughput, replace per-access path chaining with a reviewed RAM argument (typically a sorted/permuted access-table consistency check plus timestamps, or a lookup/permutation construction). That is a distinct protocol component and must be folded/accumulated along with the chunk relation. Do not claim “valid memory access” from only initial/final roots.

### How all VOLE lanes enter the chunk relation

For every AND record/lane `ell`, the chunk relation must include the required VOLE equation and the common specified coefficient schedule. The relation should use an explicit lane vector or a binding aggregation whose verifier equation covers every indexed lane. If aggregation is used, the challenge is sampled only after the chunk commitment/accumulator state is fixed and is domain-separated by `(program_id, chunk_index, gate_id, lane_count, coefficient_schedule_id)`. An implementation must demonstrate that changing, dropping, or permuting any lane causes rejection.

### Concrete protocol path

1. **Specify the canonical IR and state ABI.** Freeze Boolar/Volar canonical serialization, `WireId` allocation, chunk boundaries, `S` encoding, and initialization semantics. Include program ID, chunk count/bound, and state version in the public instance.
2. **Build the Mode-A chunk oracle.** For one chunk, commit its one-leaf-per-wire trace and authenticate every needed wire/memory path. Replay the chunk and verify the direct state-root chain. This is the executable reference relation.
3. **Choose a reviewed general folding/IVC backend whose algebra matches the target.** Nova is uniform-R1CS; SuperNova addresses a fixed family of circuits; HyperNova generalizes to CCS; Protostar targets general Plonkish-style relations; Nebula adds machinery for memory/machine execution. None is presently a drop-in `GF(2^128)` backend for this project. Many mature deployments use prime fields and curve cycles for recursion/compression, so a native binary-tower implementation requires a separate construction and security/engineering review.
4. **Encode the reference chunk relation into that backend.** This includes Boolar/Volar gates, memory access paths or RAM argument, all lanes, and the initialization transition. Prove equivalence against Mode A with differential tests.
5. **Use a formally specified accumulator/folding transcript.** Bind the incoming/outgoing state instances, chunk identity, program ID, lane/coefficient configuration, and all commitments before Fiat–Shamir challenges. Adopt the exact security model and error bounds of the selected scheme; do not invent fold equations.
6. **Add final compression only after correct incremental folding.** If an externally succinct proof is required, use the selected system’s reviewed compression/recursion mechanism. Verify the final compressed proof against `(S_genesis, S_final, program_id, statement context)`.
7. **Adversarial tests.** Mutate a wire, gate order, chunk boundary, input, output, memory address, old value, write root, lane, coefficient, PC, and program ID; each must reject. Test reordering, deletion, duplication, and splicing of individually valid chunks.

### IOR-specific assessment

IORs are relevant because they model reductions where a verifier transforms an instance into a new oracle-based instance rather than merely accepting/rejecting. Recent work uses IORs as a foundation for hash-based accumulation/folding. This is promising research infrastructure for a transparent/binary-field-friendly route, especially if the eventual target is code-based and hash-based rather than curve-recursive.

But it remains a **research-selection task**, not a ready protocol choice for Cirrus:

- identify an IOR/accumulation construction with a full reduction for the exact constraint and RAM relations, not merely a generic slogan;
- verify its field/code requirements against Volar’s `GF(2^128)` tower and its desired security level;
- supply a BCS/Fiat–Shamir compilation, commitment scheme, query schedule, and concrete soundness bound;
- establish composition of initialization, chunk-step, RAM, lane aggregation, and final verification; and
- implement/test it as a new backend, not by adapting `IopAccumulator`’s seven-element AND witness.

Thus the answer is: **adopt chunked IVC as the architectural direction; retain Mode A as its reference implementation; do not commit yet to a particular IOR paper/backend until the field/recursion and RAM choices are resolved.**

## 11. WHIR assessment: not a native Volar Mode-B backend

### Decision

**Do not lock in or start an implementation of WHIR for the native Volar `GF(2^128)` path.** WHIR is a strong candidate *only* for a field/backend that satisfies its required multiplicative-domain algebra. That is not our current binary-tower field, and treating the paper as a generic drop-in “succinctness layer” would be unsound.

WHIR (ePrint 2024/1586) is an IOP of proximity for constrained Reed–Solomon codes. It can be compiled with BCS/Fiat–Shamir and, paired with the paper's generalized-R1CS Σ-IOP/compiler, can provide the kind of trace-relation argument we need in a compatible setting. Its multi-constraint construction is also conceptually useful for binding circuit, output, storage, and lane constraints. Those positives do **not** remove the following blocker.

### Hard field incompatibility

The paper defines its Reed–Solomon domain as a *multiplicative coset of `F*` whose order is a power of two*, with degree bound `2^m`; its folding pairs `x` and `-x` and maps them through `x^2`. The reference parameter choices use prime fields (a 192-bit smooth prime and Goldilocks), not a characteristic-two field.

Volar's target field has characteristic two:

```text
F = GF(2^128),       |F*| = 2^128 - 1  (odd),       -x = x.
```

Therefore `F*` has no nontrivial power-of-two-order subgroup, and the WHIR pair `{x, -x}` collapses to one point. Its smooth multiplicative domain and fold are unavailable over this field. This is not a missing implementation optimization; it invalidates the construction's stated setup and proof path.

Moving only the polynomial proof to a prime field is also not a small adapter: there is no field embedding preserving the `GF(2^128)` arithmetic into a prime-characteristic field. A prime-field proof can encode Boolean wire values as `0/1`, but the desired all-lane VOLE equations and binary-field coefficient semantics then require a separately specified and proved bit/field-representation relation. No such relation, encoding, range discipline, or composition proof exists here. Until it does, a prime-field WHIR proof would prove a different statement from the Volar/VOLE execution.

### Security/implementation status even on a compatible field

Before choosing WHIR for a future prime-field backend, pin all of the following:

- field, smooth domain, rate, extension field, hash/Merkle suite, BCS transcript, and exact soundness target;
- a **unique-decoding** parameterization or a deliberate, reviewed choice of its mutual-correlated-agreement assumptions. The paper's Johnson/capacity configurations rely on Conjecture 4.12; later work challenges capacity-style claims. Do not use a conjectural capacity setting by default;
- the full Σ-IOP/compiler for the actual Boolar relation, rather than just WHIR's low-degree/proximity component;
- a maintained, reviewed implementation. The authors' Rust implementation is an arkworks academic prototype, while Cirrus is `no_std` and has neither arkworks nor a compatible field backend.

### What remains missing before *any* Mode-B implementation begins

The Mode-A code is a useful relation oracle, but we do not yet have a machine-checkable Mode-B relation/backend boundary. In particular:

1. **Canonical public statement:** no implementation yet hashes/binds a frozen Boolar/Volar IR serialization, protocol version, trace layout, session context, outputs, initial/final state claims, lane count/order, and coefficient schedule.
2. **Trace algebraization:** one Merkle leaf per Boolean wire is an audit layout, not yet the padded field-valued trace/oracle columns, row selectors, Booleanity constraints, and boundary rows needed by an IOP.
3. **All-VOLE-lane relation:** the current optional hook folds one selected lane; the requested all-lane common-coefficient rule has not been encoded or tested as a circuit/IOP relation.
4. **Storage statement:** Mode A has a transparent execution-table/sorted-table check. It does not yet have a polynomial permutation/grand-product (or equivalent) constraint, table layout, zero/default encoding, or initial/final Merkle-root transition relation for Mode B.
5. **Proof system:** no selected compatible low-degree test, sumcheck/constraint compiler, concrete parameters, Fiat–Shamir schedule, or end-to-end soundness composition exists. The existing `volar-iop` Ligero proof finalizes a fixed accumulator and cannot fill this role.
6. **Verifier artifact:** the verifier must independently obtain/authenticate the canonical circuit by `circuit_id`; this distribution/serialization contract is not implemented.

Accordingly, the right next deliverable is a **backend-neutral, executable relation specification** plus differential tests against Mode A—not WHIR code. Once a field-compatible proof backend is selected, that relation can be compiled without changing the claimed computation.

## 12. Prior art and primary sources

- Kilian, *A Note on Efficient Zero-Knowledge Proofs and Arguments* (STOC 1992), the PCP-to-commitment-and-query argument pattern. DOI: <https://doi.org/10.1145/129712.129782>.
- Ben-Sasson et al., *Scalable, transparent, and post-quantum secure computational integrity* (STARK), ePrint 2018/046: <https://eprint.iacr.org/2018/046.pdf>. Its trace commitments and queried-row openings are the closest architecture for Mode B.
- Ben-Sasson et al., *Aurora: Transparent Succinct Arguments for R1CS*, ePrint 2018/828: <https://eprint.iacr.org/2018/828.pdf>. Relevant for a hash-committed IOP for R1CS rather than a bespoke accumulator finalizer.
- Setty, *Nova: Recursive Zero-Knowledge Arguments from Folding Schemes*, ePrint 2021/370: <https://eprint.iacr.org/2021/370>. Foundational uniform-R1CS folding/IVC reference; it does not make dynamic arbitrary programs or authenticated RAM automatic.
- Kothapalli and Setty, *SuperNova: Proving Universal Machine Executions without Universal Circuits*, ePrint 2022/1758: <https://eprint.iacr.org/2022/1758>. Relevant when execution is a predefined family of step circuits rather than one uniform relation.
- Srinath Setty, *HyperNova: Recursive Arguments for Customizable Constraint Systems*, ePrint 2023/573: <https://eprint.iacr.org/2023/573>. CCS/general-constraint-system folding reference.
- Bünz et al., *Protostar: Generic Accumulation and Folding for Special-Sound Protocols*, ePrint 2023/620: <https://eprint.iacr.org/2023/620>. General Plonkish/special-sound-protocol accumulation direction.
- *Nebula: Efficient Read-Once Memory Checking for Zero-Knowledge Proofs*, ePrint 2024/1605: <https://eprint.iacr.org/2024/1605>. Relevant stateful-machine/memory component; verify the exact construction against the required memory semantics before selection.
- *ARC* (hash-based accumulation via interactive oracle reductions), ePrint 2024/1731: <https://eprint.iacr.org/2024/1731>. Relevant IOR/accumulation research direction; not a drop-in Cirrus backend.
- Arnon, Chiesa, Fenzi, and Yogev, *WHIR: Reed–Solomon Proximity Testing with Super-Fast Verification*, ePrint 2024/1586: <https://eprint.iacr.org/2024/1586.pdf>. WHIR is an IOPP for constrained Reed–Solomon codes and supports a BCS-compiled argument/Σ-IOP route to generalized R1CS, but its stated smooth multiplicative `2`-power domain and `x,-x` folding exclude Volar's characteristic-two `GF(2^128)` field. Its authors' reference implementation: <https://github.com/worldfnd/whir> (academic prototype; not a selected Cirrus dependency).
- Grassi et al., *Poseidon2: A New Hash Function for Zero-Knowledge Proof Systems*, ePrint 2023/323: <https://eprint.iacr.org/2023/323.pdf>. Original Poseidon2 source; do not transpose its parameters to binary fields.
- Grassi, Khovratovich, Koschatko, Rechberger, Schofnegger, Schröppel, *Poseidon2b: A Binary Field Version of Poseidon2*, cited by the authors’ reference repo as ePrint 2025/058: <https://eprint.iacr.org/2025/058>. Reference repository (pinned review target still required): <https://github.com/Poseidon-Hash/Poseidon2b>.
- Local evidence: `../volar/docs/prove-the-verifier-iop.md`, `../volar/crates/iop/volar-iop/src/{fold,ligero,merkle,verifier}.rs`, and `../volar/crates/iop/volar-verifier-iop-runtime/src/lib.rs`. These establish the current accumulator-only statement and the project’s own unpinned/open-soundness caveats.
