# TinyLabels (ePrint 2024/2048) review log

Status: the paper/backend decision record is complete. No TinyLabels,
row-reduced, or other experimental garbling implementation has been added yet.

Source reviewed: *TinyLabels: How to Compress Garbled Circuit Input Labels,
Efficiently*, Marian Dietz, Hanjun Li, and Huijia Lin, ePrint 2024/2048
(`/Users/g/Downloads/2024-2048.pdf`), especially pp. 1, 6--9, 13, and
34--36.

## Decision status

The decisions needed to begin the paper-relevant backend work are resolved:
the current target has approximately one million reuses, 256 KiB RAM, 2 MiB
flash, 128-bit security, a microcontroller garbler, and a server evaluator.
They are sufficient to begin the baseline evaluator and first-row-fixed
reduction work, and to scope a separate TinyLabels feasibility study.

The end-to-end deployment protocol and symbolic host-call semantics remain
design work. They are deliberately not treated as settled requirements for
every open-source integrator, or as blockers for the evaluator completeness
work below.

## Working deployment profile (still in design)

This is the repository's current focus, not a restriction on other
integrators:

- The microcontroller is the **garbler**. It streams the fixed-workload
  garbling to a server evaluator, which combines it with FHE operations; the
  garbled circuit supplies the FHE re-encryption work.
- Repeated evaluations are expected. Reusing a fixed preprocessed garbling and
  sending only the selected labels when inputs arrive is consequently a useful
  optimisation target.
- The security target is 128 bits. The evaluator/server may be malicious and
  large enough to attempt to recover internal wire values, including FHE
  plaintexts.
- `hash` is a symbolic runtime operation in the target design, not an
  off-circuit concrete result.
- The garbler and server are mutually distrustful: the garbler requires
  privacy against the server, while the server wants integrity of the claimed
  hardware-mediated result. Transport authentication is standard TLS; a
  deviant garbler's output-bit integrity is a separate, potentially
  in-circuit, concern rather than the first backend milestone.
- The design point expects roughly one million uses of reusable garbling
  material. The online garbler continuously receives and decrypts ciphertexts
  and derives circuit input/output labels while the server performs the FHE
  work. The circuit itself performs the encryption needed for that exchange.
- Offline state is capped at 256 KiB of RAM and 2 MiB of flash. The RAM cap
  applies while garbling and while the continuous online label path runs.

## What the paper is, and is not

TinyLabels is a Ring-LWE batch-selection protocol for communicating selected
garbled-circuit **input labels**. It is not a row-reduction technique and not
a replacement for a standard garbling construction. Its concrete evaluation
uses a degree-4096 ring, 109-bit modulus, NTT operations, and SEAL on a
2.10 GHz Xeon; those numbers are useful as a server baseline, not evidence
that the construction fits or performs well on a Cortex-M33 (pp. 34--36).

The paper's preprocessing-garbling construction goes further by garbling a
universal circuit before the particular function is known (p. 13). That is a
different proposition from this repository's fixed interpreter workload. The
paper notes that the universal circuit is at least `O(log |f|)` larger than the
target circuit (p. 73), so it should not be an assumed optimisation here.

## Mapping it to the ERT workload

The intended fixed-workload path is viable, with these exact boundaries:

| Item | Fixed-workload preprocessing | Runtime/input phase |
| --- | --- | --- |
| ERT program image, static constants, and concrete control flow | Run the interpreter once to generate the fixed circuit topology and stream its tables. | Not recomputed. |
| True external input bits | Allocate their two labels and any TinyLabels offline material. | Select/deliver the labels after the input is known. |
| Ordinary symbolic intermediates | Become wires in the preprocessed circuit. | The evaluator derives their labels from gate tables; they are not separately delivered. |
| Current `hash` ECALL result | The present callback returns bytes during interpretation, which ERT materializes as constants. | Target: lower it through a symbolic machine handler. This requires a new host-call contract; it is not deferred by input-label compression alone. |
| Branches, flags, and non-stack addresses | Must be concrete before topology generation, as required by ERT. | A symbolic decision remains `Unexpected`; TinyLabels does not change that. |

Thus, the device does need to process the exact workload once when the circuit
topology is fixed, as proposed. This is **function-dependent preprocessing**,
not the paper's function-independent universal-circuit preprocessing. Inputs
can arrive later. A `hash` evaluation becomes runtime only after deliberately
replacing the current host callback with a circuit-defined runtime operation;
it is not deferred by input-label compression alone.

The targeted symbolic hash mixes host-only random data into the computation,
with SHA-256 available as a fallback layer. Its symbolic output words are
additional evaluator inputs. The eventual handler contract must explicitly
identify that randomness and those output words, so the evaluator can consume
the right labels without learning their semantic values.

## Required machine-handler seam

Before a garbling backend can lower `hash` symbolically, the private
interpreter `Machine` should hold a `dyn MachineHandler`. The handler is the
deep module at the host-operation seam: it owns the Boolean context and the
auxiliary system-call adapters, while the decoder and generic instruction
handlers remain concerned only with machine state and canonical operations.

The concrete handler should be generic over the Boolean context and its
auxiliary operations. It should implement the object-safe `MachineHandler`
interface used by `Machine`, rather than making the decoder know a particular
garbling scheme. The interface needs only symbolic word operations and a
typed host-call request/result carrying symbolic words plus their optional
concrete metadata. In particular, a symbolic hash result must return eight
words whose metadata is normally unknown; it must not be coerced through the
current `[u8; 32]` callback.

The existing public `ert_emit` and `ert_func` helpers should remain useful by
constructing a legacy callback adapter internally. New handler-aware helpers,
or a builder that accepts a handler, can expose symbolic host calls without
forcing a policy on other integrators. RV32 and Thumb must use the same host
call request representation, with their existing ABI decoders acting as small
adapters at that seam.

## Evaluator completeness seam

Every garbler implementation needs a paired host evaluator before it is used
for cost claims. This is a completeness check, not a network protocol or an
authentication layer: given selected input labels and ordered garbling records,
the evaluator must derive the expected output labels and decode the known test
result.

The evaluator should pull an `Iterator` of ordered table/hint records. One
ordered record type is preferable to separately synchronized table and hint
iterators: ordering, end-of-stream, and malformed-record errors then belong to
one small interface. A streaming transport adapter can later supply that
iterator from TLS or from bounded buffers without changing evaluator logic.

The baseline evaluator becomes the common test shape. The first row-reduced
backend gets its own evaluator that implements the same role but its own table
format: the first row is fixed and omitted from the stream, while the remaining
rows are pulled on demand. Any later reduction remains a different
implementation, with a new format/version and the same evaluator-completeness
tests.

## Relevance to the measured bottleneck

For the locked SHA-256 workload, the stable streaming baseline emits 10.5 MB
of Thumb tables and 27.5 MB of RV32 tables at 16-byte labels. TinyLabels
compresses input-label delivery, not the table stream. It may therefore matter
for repeated fixed circuits, late-arriving inputs, or an input-label bottleneck,
but it does not by itself reduce the principal first-run table traffic.

The paper reports an amortized offline cost of about six bits per transferred
128-bit value in its selected parameter setting, but also reports roughly
29 MB public parameters, 2.2 GB reusable ciphertext before amortization, and
56 KB for the input-dependent key over batches of about 699,050 messages
(pp. 34--36). Those figures rule out treating the paper construction as a
drop-in microcontroller primitive without a separate parameter, code-size,
RAM, and energy study.

## Package boundary and implementation order

All garbling implementations now live under `crates/garbled-circuit/`. The
stable `cirrus-garbled-circuit` crate remains a four-row streaming baseline.
Its synchronous `Pusher` is the transport seam and remains integrator-owned.

1. Introduce and test the private `MachineHandler` seam with a legacy concrete
   callback adapter and a symbolic-hash test adapter. Preserve the present
   public execution helpers while adding an opt-in handler-aware entry point.
2. Add paired host evaluators for the existing baseline and each subsequent
   garbler. Test them by pulling an ordered `Iterator` of tables and hints,
   first for primitive gates and then for the locked Thumb workload. Keep this
   as a completeness check rather than a transport or authentication protocol.
3. Design and validate a first-row-fixed row-reduced backend as its own crate,
   for example `cirrus-garbled-circuit-row-reduced`. It needs its own table
   format, garbler/evaluator algorithm, primitive truth-table vectors, and
   streaming protocol version. It must not silently reinterpret the baseline's
   four rows as a three-row scheme.
4. Only after that interface is explicit, prototype TinyLabels in a separate
   crate such as `cirrus-garbled-circuit-tinylabels`. It must not depend on
   baseline wire-format internals, although both crates can use the Boolean
   context seam and the same ERT workload measurements. Its design must respect
   the 256 KiB RAM, 2 MiB flash, and approximately one-million-use targets.
5. Compare each backend on the exact Thumb SHA-256 workload first, with a
   streaming sink, bounded transport buffer, an evaluator/interoperability
   test, and separately measured device RAM, flash, time, energy, table bytes,
   input-label bytes, and preprocessing reuse count. RV32 remains a regression
   comparison rather than the preferred target.

## Deployment and host-call design work (non-blocking)

- Define the integrity claim and threat model precisely: which incorrect
  outputs the server must detect, what a malicious server may observe, and what
  (if any) in-circuit authentication proves the garbler's claimed hardware
  execution. TLS only authenticates the transport peer.
- Specify the symbolic hash operation: algorithm/parameters, word ordering,
  output decoding, host-only randomness lifecycle, and whether the SHA-256
  fallback is fixed in the circuit or chosen by an allowed concrete selector.
- Choose the concrete 128-bit label, hash, and row-reduction construction only
  after an interoperable garbler/evaluator protocol is specified. The existing
  crate remains a construction/measurement baseline, rather than that protocol.

These open design questions must be resolved before making deployment security
claims or fixing a TinyLabels wire format. They do not block the paired
evaluator, first-row-fixed reduction, or the initial feasibility measurements.
