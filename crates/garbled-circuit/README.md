# Garbled-circuit implementations

This package family keeps each garbling wire format and its implementation
behind a separate crate. The stable baseline is `cirrus-garbled-circuit`.
Experimental row-reduced and paper-derived backends belong beside it, rather
than changing the baseline's table representation or its streaming interface.
The current [TinyLabels review log](TINYLABELS_REVIEW.md) records why the
ePrint 2024/2048 protocol is an input-label-compression experiment rather than
a replacement garbling scheme.

All implementations use the Boolean-context seam consumed by the ERT facades.
Their tests must exercise the same primitive truth tables and locked RV32/Thumb
SHA-256 workloads, while treating transport and coroutine scheduling as an
integrator responsibility.
