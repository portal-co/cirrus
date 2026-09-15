# TinyLabels and Volar-port plan for Cirrus

**Status:** research and integration plan, 2026-09-14.

This plan extends, but does not supersede,
[`TINYLABELS_REVIEW.md`](TINYLABELS_REVIEW.md). TinyLabels remains an explicit
server-class experiment. It is **not** the default Cirrus input path, a label
shortening format, or an embedded feature.

The companion Volar plan is
[`../../../mpc/volar/docs/research/tinylabels-cross-repository-plan.md`](../../../mpc/volar/docs/research/tinylabels-cross-repository-plan.md).
The two documents deliberately use the same module/seam vocabulary while
preserving their different execution models.

## Non-negotiable deployment split

| Mode | Default delivery | TinyLabels availability |
| --- | --- | --- |
| Embedded ERT / MCU garbler | Direct selected-label delivery with bounded table streaming | **Unavailable.** Reject it at profile construction. |
| Desktop/server fixed interpreter trace | Direct selected labels, optionally Ferret-backed | Experimental explicit opt-in after resource admission. |
| Offline/server coordinator for high reuse | Direct delivery remains available | Candidate after full frame, sampler, encoding, and review gates. |

The TinyLabels paper compresses selected **input-label transport**. It does
not reduce the 16-byte `Label`/`Garble<U16>` security parameter, table-stream
bytes, evaluator-derived intermediate labels, or opaque durable labels. Its
reference degree-4096 profile uses about 34 MB of raw public parameters and
about 2.55 GB of raw reusable `ct1`, which is categorically unsuitable for the
current 256 KiB RAM / 2 MiB flash deployment profile. [DLL24]

## Architecture

### `InputLabelDelivery` is the real seam

Cirrus needs a deep `InputLabelDelivery` module at the external-label boundary,
not a TinyLabels-specific hook inside a garbler or interpreter. Its interface
is deliberately batch-oriented:

```text
prepare(profile, ordered manifest) -> reusable handle
begin_use(reusable handle, transcript binding) -> use handle
receive_selected(use handle, role-owned choices) -> ordered active labels
metrics() -> public resource counters
```

The ordered manifest binds program/topology identity, selected external-wire
positions, direction/owner, label width, and exact count. The module returns
exactly one active label for each manifest position. It does not decode a
Boolean, derive intermediate labels, choose a table format, or know the ERT
instruction set.

Two adapters sit at that seam:

- **Direct delivery** is the small default adapter. It delivers selected
  labels directly through the normal OT/transport profile.
- **TinyLabels delivery** owns Ring-LWE arithmetic, the offline/reusable
  state, canonical frames, parameter validation, transcript bindings, and
  resource admission. It is host/server-only until a separate embedded
  feasibility proof exists.

The existing `cirrus-garbled-circuit-tinylabels` crate is already the natural
implementation home. Its `LabelBatch` establishes the allocation-free shape;
its `ring_lwe::BatchSelect` establishes typed mathematical stages. Neither is
yet the security protocol.

### Preserve streaming and interpreter locality

ERT and LLVM produce a fixed topology before the input-label path begins.
Tables continue to cross the existing `Pusher`/iterator seam in circuit order.
The delivery module must use a separate ordered frame stream; it must not
reinterpret `GarblingRecord` or `GarbleTable` records.

`MachineHandler` remains the host-operation seam for making runtime external
words explicit. TinyLabels cannot make symbolic control flow valid, cannot
supply missing topology, and cannot transform the present concrete `hash`
callback into a symbolic operation on its own.

## Porting recent Volar work into Cirrus

The goal is behavior parity where the concept fits Cirrus's streaming
interpreter methodology—not source-level duplication of Volar's strict-chain
or durable-storage implementation.

| Volar feature | Cirrus port | Acceptance test |
| --- | --- | --- |
| `deferred_remap_schedule` | Add an allocation-free `OpaqueRebase` helper in `cirrus-volar-garble`. It combines a held garbler false base with a fresh public-zero base by free XOR; the evaluator combines the held active label with its matching zero label. | ERT and `Program` streaming replay: output label opens under the fresh base; no table record is emitted; no Boolean is decoded. |
| `FerretOtChannel` plus public metrics | Extract Cirrus's currently test-local tagged `StackIo` framing from `cirrus-volar-vole/tests/ferret_stack.rs` into a host transport adapter with frame/byte counters. Reuse it for VOLE and future direct selected-label delivery. | TCP test for both roles, refill behavior, malformed tag/length rejection, and counters. `FERRET_REG_TOY` remains test-only. |
| Material-store cache / storage phase | No direct port today. Cirrus has no paired split-key durable-label store or strict-chain phase protocol. Record this as a future outer adapter at the interpreter execution boundary; do not overload TinyLabels to carry durable material. | A later role-local persistence protocol must preserve opaque labels and use explicit load/flush boundaries. |
| Shared-key AES material batching | No direct port today for the same reason. It belongs to a future Cirrus persistence adapter, not the garbling stream or TinyLabels. | Circuit-independent material primitive test before integration. |
| Public material metrics | Generalize the useful part into `StreamingMetrics` at the transport adapter: table records/bytes, label frames/bytes, Ferret frames/bytes, and peak buffered records. | Thumb SHA-256 and one `Program` test assert table ordering plus bounded buffering. |

## Implementation sequence

### C0 — lock defaults and admission policy

1. Add a public profile enum that defaults to direct input delivery and requires
   an explicit `TinyLabelsServer` selection.
2. Make profile construction reject TinyLabels on `target_os = "none"`, ARM,
   and RV32 builds unless a future dedicated embedded profile supplies a
   measured bound.
3. Keep label width fixed at 16 bytes. The profile changes delivery mechanics,
   not `Label<N>` or the global free-XOR relation.

### C1 — finish TinyLabels protocol prerequisites

1. Define a canonical injective `Label<16> <-> [Z_p; 3]` encoding, exact
   inverse, endianness, and malformed-value rejection. Test that selection
   reconstructs raw labels byte-for-byte and does not alter the free-XOR
   relation.
2. Define canonical stage frames for `pp`, reusable `ct1`, per-use `ct2`,
   selection/key material, and completion/error. Every frame carries a version,
   parameter fingerprint, stage, exact count, direction, session ID, manifest
   digest, and authenticated length.
3. Add bounded streaming encoders/decoders rather than allocating a reference
   profile's state. Raw SEAL NTT dumps remain forbidden as a wire format.
4. Integrate a reviewed CSPRNG and exact clipped discrete-Gaussian sampler;
   state and test the decryption-failure bound. Keep `ZeroNoise` test-only.
5. Build a host-only semantic interoperability runner for the pinned author
   artifact. Follow Construction 1 and sample the LEnc `r` vector; do not
   reproduce the artifact's apparent unsampled-`r` behavior merely for
   byte compatibility.

### C2 — streaming interpreter integration

1. Complete the planned `MachineHandler` seam to expose a canonical external
   input manifest for both RV32 and Thumb ABI adapters.
2. Make direct delivery the first `InputLabelDelivery` adapter and run it
   through the exact ERT Thumb SHA-256 replay.
3. Add TinyLabels as a second host-only adapter, first under deterministic
   toy/semantic parameters and then a separately gated reference-profile
   harness.
4. Use `cirrus-coroutine` to pair a bounded table `Pusher` with frame readers.
   Test that a slow evaluator applies backpressure and that neither side
   materializes the entire table or TinyLabels ciphertext stream.
5. Measure direct versus TinyLabels using the same manifest: setup/reusable
   bytes, per-use bytes, online rounds, CPU, peak heap, table bytes, and reuse
   break-even. The large reference profile must use a disk/server streaming
   harness, not normal CI.

### C3 — port the relevant transport/rebase behavior

1. Land `OpaqueRebase` with ERT and `cirrus-recompile-rt` tests before exposing
   it to any persistence design.
2. Move the Ferret framing implementation out of the VOLE test into one
   transport module; retain the test as an integration consumer rather than a
   duplicate implementation.
3. Add `StreamingMetrics` to that transport module and wire it into the
   existing Thumb measurement record.
4. Only then investigate an explicit role-local durable-material module. Its
   design must be reviewed independently of input delivery and must not imply
   any TinyLabels dependency.

## Security gates

Before a TinyLabels delivery adapter can process protected labels:

- independent cryptographic review of field encoding, parameter derivation,
  Ring-LWE sampler, failure probability, and transcript design;
- exact input-choice ownership and malicious-party analysis;
- authenticated framing, anti-replay/counter rules, and strict manifest
  binding;
- bidirectional interoperability and malformed-frame test corpus;
- server resource benchmark plus embedded admission/rejection tests; and
- explicit product opt-in with direct delivery as the fallback.

Authenticated garbling, cut-and-choose, or evaluator auditing may be pursued
for active-security goals, but none is a generic justification for shortening
labels or enabling TinyLabels in an embedded profile.

## Sources

- **[DLL24]** Marian Dietz, Hanjun Li, and Huijia Lin, *TinyLabels: How to
  Compress Garbled Circuit Input Labels, Efficiently*, IACR ePrint 2024/2048.
  https://eprint.iacr.org/2024/2048
- Existing implementation review and author-artifact audit:
  [`cirrus-garbled-circuit-tinylabels/RESEARCH.md`](cirrus-garbled-circuit-tinylabels/RESEARCH.md).
