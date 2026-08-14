# TinyLabels (ePrint 2024/2048) review log

Status: design review only. No TinyLabels, row-reduced, or other experimental
garbling implementation has been added.

Source reviewed: *TinyLabels: How to Compress Garbled Circuit Input Labels,
Efficiently*, Marian Dietz, Hanjun Li, and Huijia Lin, ePrint 2024/2048
(`/Users/g/Downloads/2024-2048.pdf`), especially pp. 1, 6--9, 13, and
34--36.

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
| Current `hash` ECALL result | The present callback returns bytes during interpretation, which ERT materializes as constants. | It cannot become a runtime symbolic value without a new host-call lowering/circuit contract. |
| Branches, flags, and non-stack addresses | Must be concrete before topology generation, as required by ERT. | A symbolic decision remains `Unexpected`; TinyLabels does not change that. |

Thus, the device does need to process the exact workload once when the circuit
topology is fixed, as proposed. This is **function-dependent preprocessing**,
not the paper's function-independent universal-circuit preprocessing. Inputs
can arrive later. A `hash` evaluation can be runtime only after deliberately
replacing the current host callback with a circuit-defined runtime operation;
it is not deferred by input-label compression alone.

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

1. Design and validate a row-reduced backend as its own crate, for example
   `cirrus-garbled-circuit-row-reduced`. It needs its own table format,
   garbler/evaluator algorithm, security argument, primitive truth-table
   vectors, and streaming protocol version. It must not silently reinterpret
   the baseline's four rows as a three-row scheme.
2. Only after that interface is explicit, prototype TinyLabels in a separate
   crate such as `cirrus-garbled-circuit-tinylabels`. It must not depend on
   baseline wire-format internals, although both crates can use the Boolean
   context seam and the same ERT workload measurements.
3. Compare each backend on the exact Thumb SHA-256 workload first, with a
   streaming sink, bounded transport buffer, an evaluator/interoperability
   test, and separately measured device RAM, flash, time, energy, table bytes,
   input-label bytes, and preprocessing reuse count. RV32 remains a regression
   comparison rather than the preferred target.

## Decisions requested before implementation

- Is the first deployment model one garbler and one evaluator, and which party
  is the microcontroller? The paper's roles, offline state, and security
  assumptions depend on that answer.
- Does the protocol need repeated evaluations with reusable preprocessing, or
  only one evaluation? The paper's largest reusable component only amortizes
  over repeated use.
- Should `hash` stay an off-circuit concrete host result, or become a defined
  symbolic circuit operation? The latter changes the interpreter/host-call
  interface and the workload topology.
- What security target and label size are required? The existing crate is a
  construction/measurement baseline, not yet a complete interoperable
  garbler/evaluator protocol, so an experimental backend must specify these
  before performance claims are meaningful.

Until those choices are recorded, no paper-derived implementation should be
started. This preserves the stable backend while making the research path and
its assumptions reviewable.
