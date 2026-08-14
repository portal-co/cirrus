# Thumb SHA-256 replay measurement

This is the first evaluator-completeness and traffic measurement for the
locked Thumb SHA-256 compression workload. It runs the public
`cirrus_armv8m_ert::ert_func` seam twice: first with the first-row-fixed
garbler and then with its pull-based evaluator. The evaluator receives only
the selected input labels and an ordered `Iterator` of the emitted records.

The fixture builds `cirrus-armv8m-ert-selftest` in release mode for
`thumbv8m.main-none-eabi`, maps its linked `.text.ert_workload` and
`.rodata.ert_workload` at `0x2000_0000`, and uses the same one-block `abc`
SHA-256 test vector as the bare-metal gate. The native and replayed result is
the symbolic, non-concrete word `0xba78_16bf`.

Run it with:

```sh
cargo test -p cirrus-garbled-circuit-row-reduced --test thumb_sha256
```

## Deterministic table traffic

The test locks the workload at 124,160 non-free gates and uses 16-byte labels.
(The ERT interpreters constant-fold arithmetic/bitwise ops whose operands are
both already known, materializing the result from the `zero`/`one` wires
instead of emitting gates; this dropped the count from an earlier
164,288-gate baseline.)

| Format | Rows per gate | Bytes per gate | Total table bytes | Change from four rows |
| --- | ---: | ---: | ---: | ---: |
| Stable four-row baseline | 4 | 64 | 7,946,240 (7.58 MiB) | — |
| First-row-fixed experiment | 3 | 48 | 5,959,680 (5.68 MiB) | -1,986,560 bytes (-25%) |

The test intentionally records all three-row tables in a host `Vec` so that a
fresh evaluator can replay them. That 5.68 MiB allocation is **not** a target
RAM requirement. A deployment must stream each 48-byte record to its server
sink, with only an integrator-defined transport frame and backpressure buffer.

For this fixed interface, 16 symbolic input words need 512 selected labels:
8,192 bytes at 16 bytes per label. One 32-bit output word needs 512 bytes of
selected output labels for decoding. Retaining both labels for these input
wires takes 16 KiB before any protocol-specific preprocessing material.

At one million reuses, the one-time three-row table stream amortizes to about
6.0 bytes per evaluation, excluding persistence, framing, input labels, and
protocol messages. This makes reusable garbling materially attractive only if
the server persists the table stream and the device can retain or regenerate
the required input/output-label material.

## Current symbolic-memory envelope

The replay provisions 2,048 symbolic-stack labels and locks the current
workload's high-water mark at 1,408 labels. With a 16-byte `Label`, that is:

| Item | Provisioned bytes | Observed/touched bytes |
| --- | ---: | ---: |
| Symbolic stack | 32,768 | 22,528 |
| 16 core-register words | 8,192 | 8,192 |
| 512-entry static return stack | 2,048 | 8 (two entries) |

The simple provisioned subtotal is 43,008 bytes, before code, decoder state,
input-label inventory, hash state, transport buffering, allocator/runtime
overhead, and any future cryptographic preprocessing. It is therefore
plausible to fit the current *interpreter working set* under a 256 KiB RAM
budget after right-sizing the stack, but this is not yet a device-RAM or
energy measurement. In particular, the existing generic host test's table
collector must not be copied into an embedded deployment.

An uncached host test run observed roughly 0.67 seconds garbling and 0.14
seconds evaluating. Those timings are a regression signal only: they are not
a Cortex-M33 benchmark and must not be extrapolated to target hardware.

## Verdict and next work

Thumb remains the preferred ERT target: it uses the same fixed workload as
RV32 while producing far fewer non-free gates. First-row fixing is a useful,
low-risk 25% traffic reduction and now has a full public-ERT replay test.
It does not remove the need for streaming.

The next practical measurements are:

1. Run a bounded streaming sink on Cortex-M33-class hardware and measure
   peak RAM, flash, time, and energy; retain the 2,048-label stack budget as a
   regression fixture until a smaller safe budget is demonstrated.
2. Separate reusable offline material from online selected-label delivery and
   measure the latter over repeated evaluations. The device/server deployment
   still needs the symbolic `hash` machine-handler design described in the
   [TinyLabels review](TINYLABELS_REVIEW.md).
3. Keep the four-row baseline and this row-reduced format interoperable only
   through their own evaluator tests. A TinyLabels implementation belongs in a
   separate crate: it targets selected input-label compression, not this
   5.68 MiB table stream.
