# Garbled-circuit implementations

This package family keeps each garbling wire format and its implementation
behind a separate crate. The stable baseline is `cirrus-garbled-circuit`.
Experimental row-reduced and paper-derived backends belong beside it, rather
than changing the baseline's table representation or its streaming interface.
The current [TinyLabels review log](TINYLABELS_REVIEW.md) records why the
ePrint 2024/2048 protocol is an input-label-compression experiment rather than
a replacement garbling scheme.

`cirrus-garbled-circuit-row-reduced` is the first experimental backend. It
derives and omits the first AND-table row, then streams the other three. It is
a completeness and cost baseline, not an authenticated-garbling protocol.
Its locked Thumb SHA-256 replay and memory/traffic result are recorded in
[`THUMB_SHA256_MEASUREMENT.md`](THUMB_SHA256_MEASUREMENT.md).

`cirrus-garbled-circuit-tinylabels` keeps the allocation-free local label
selector and a separate experimental implementation of the ePrint 2024/2048
Ring-LWE batch-select arithmetic. The latter has typed `setup`/`enc1`/`enc2`/
`keygen`/`dec` stages, a caller-supplied CSPRNG and noise-sampler seam, and
small-profile correctness tests. It is not a security protocol: canonical
label encoding, framed streaming messages, a reviewed discrete Gaussian, and
independent parameter/security validation remain necessary. Its cited
construction and an audit of the authors' reference are recorded in
[`cirrus-garbled-circuit-tinylabels/RESEARCH.md`](cirrus-garbled-circuit-tinylabels/RESEARCH.md).

All implementations use the Boolean-context seam consumed by the ERT facades.
Their tests must exercise the same primitive truth tables and locked RV32/Thumb
SHA-256 workloads. Each garbler also needs a paired host evaluator that pulls
its ordered table/hint records through an `Iterator`; this is a completeness
check, not a transport or authentication protocol. Transport and coroutine
scheduling remain an integrator responsibility.

Garbler contexts use their crate's `Label` handle, which records a wire by its
logical-zero label. Construct constants and symbolic inputs from that handle;
the paired evaluator instead receives the raw selected labels. This preserves
free-XOR polarity through symbolic inversion and constants without exposing a
per-gate protocol in the ERT API.
