# Mode-B backend-neutral relation, version 1 (draft implementation contract)

This document defines the relation that a future succinct backend proves. It is
**not a proof format** and it does not select a field, a commitment scheme, an
IOPP, a Fiat–Shamir transcript, or a RAM argument. The same relation has a
transparent Boolean evaluator (implemented in `mode_b_relation.rs`) and is
intended to be lowered to a prime-field R1CS/Spartan instance later.

The authoritative source for the initial executable gate subset is
`ModeBRelation::from_boolar`; this document explains its stable boundaries.
It emits Booleanity and supported Boolean-gate rows today, and carries storage
layout metadata. It intentionally does **not** pretend that metadata is a RAM
proof: the concrete RAM rows/argument remain a separately versioned lowering.

## Goals and non-goals

The public claim is:

```text
There is a Boolean wire assignment W and, if storage is used, RAM tables M,
such that the canonical Boolar circuit identified by circuit_id maps the
public inputs to the claimed public outputs, satisfies every supported gate,
and satisfies the stated RAM access relation.
```

`circuit_id` binds the canonical structural circuit representation, *not*
provenance metadata. `W` is private to a future succinct proof. The current
Mode-A audit may reveal it; that is a reference implementation choice, not a
property of this relation.

This draft emits arithmetic rows for `Zero`, `One`, `And`, `Or`, `Xor`, and
`Not`, and recognizes `StorageRead`/`StorageWrite` to describe their RAM
layout. The latter await the concrete RAM lowering described below. All oracle,
RNG, and action variants fail closed. Their
ABI and their connection to the public statement must be specified before a
new relation version supports them.

## Canonical wire and public-instance layout

Wire `i` denotes:

```text
0 <= i < params                 input wire i
params + s                      result of statement s
```

All primary wires are Boolean. A public instance contains, in this exact
logical order:

```text
relation_version (= 1)
circuit_id (32 octets; statement binding outside the field)
params, statement_count, wire_count, output_count
public input bits in wire order
claimed output bits in circuit.outputs order
memory-argument presence and layout version
```

The eventual proof/backend serializer must length-prefix every variable-length
component and domain-separate this object from the witness commitment and
proof transcript. It must also bind the exact relation/backend parameter
version. A digest cannot replace the verifier independently obtaining the
canonical circuit for `circuit_id`.

## Arithmetic lowering over a non-binary field

For a target field `K` with `char(K) != 2`, each Boolean wire is represented
by `w in K` and constrained with:

```text
w * (w - 1) = 0.
```

This is essential: replacing Boolar XOR with field addition without Booleanity
would prove a different relation. For every Boolean gate, introduce a private
product helper `p` when needed and use only coefficients in `{ -2, -1, 0, 1 }`:

```text
Zero:  0 = c
One:   1 = c
And:   a * b = c
Xor:   p = a*b;  a + b - 2*p = c
Or:    p = a*b;  a + b - p = c
Not:   1 - a = c
```

Each equality is emitted as an R1CS row by multiplying its left linear form
by the constant one. `p` is an auxiliary witness variable; it is not a Boolar
wire and never appears in the public wire numbering. The relation descriptor
retains signed small coefficients rather than prematurely encoding a
particular field element, so lowering maps `-1` and `-2` according to the
target modulus.

This makes the computation portion directly suitable for a Spartan-style
prime-field R1CS frontend. It does **not** embed `GF(2^128)` VOLE arithmetic
into that field. A future all-lane VOLE relation requires an explicit
bit/limb representation plus a separately reviewed composition argument.

## Storage relation

A storage access record is logically:

```text
(storage_id, lane_id, address_bits_little_endian, time, kind, value)
```

`pre_init` records are writes first; statement accesses follow statement
order. The relation contains two equal-length tables:

1. `execution`: circuit execution order;
2. `address_sorted`: the same records sorted lexicographically by
   `(storage_id, lane_id, address_bits, time, kind, value)`.

The relation requires exact permutation equality and scans the sorted table.
For each cell, a read equals the preceding write value; its first read equals
zero. A write updates the preceding value. This is the same semantics as
Mode A, expressed as a backend-neutral relation family.

The executable reference now exposes this as `RamWitness`: it derives the
execution and address-sorted tables from a Boolar witness and rejects a
missing table, a non-derived execution table, a non-canonical permutation, or
an invalid read. This is intentionally still linear-size and is the oracle for
the field/RAM lowering.

## Prime-field RAM-permutation lowering v1 (specification)

This subsection is the implemented **relation-level** lowering target for the
companion KoalaBear/Spartan-WHIR exporter. The executable exporter emits the
rows below in a static unified shape; Fiat--Shamir challenge values are public
slots rather than R1CS constants.

### Preconditions and fixed bounds

The initial, versioned target is the exact extension field used by the current
high-security `sol-spartan-whir` target:

```text
F = KoalaBear[X] / (X^5 + X^2 - 1)
p = 2^31 - 2^24 + 1 = 2,130,706,433
format = "koalabear-ext5-ram-v1"
```

`PrimeFieldRamConfig::default()` fixes `S=16`, `L=16`, `A=32`, and `T=32`.
Thus it accepts all 32-bit addresses, storage/lane IDs below `2^16`, and at
most `2^32` canonical accesses. It deliberately does **not** claim that the
base KoalaBear field can pack that tuple: it cannot. The quintic extension has
about 155 bits and carries record encodings as five base-field coefficients.
Reject a circuit whose storage-id, lane-id, address width, or access count
exceeds these bounds. Every bit below has `b(b-1)=0`; integers are represented
by little-endian bit vectors, not host integers. The `kind` bit encodes
`read=0`, `write=1`.

Split the 97-bit key prefix across extension coefficients, with every packed
coefficient below `2^30 < p` (a 32-bit coefficient would overflow KoalaBear):

```text
K0 = storage[0..16] + 2^16 lane[0..14]         # 30 bits
K1 = lane[14..16] + 2^2 address[0..28]         # 30 bits
K2 = address[28..32] + 2^4 time[0..26]         # 30 bits
K3 = time[26..32] + 2^6 kind                   # 7 bits
K4 = 0
K(r) = (K0, K1, K2, K3, K4) in F
```

The ranges are little-endian half-open bit ranges. This is injective in the
extension's canonical polynomial basis and avoids both base-field overflow and
field-element reduction. The remaining extension-coordinate capacity is **not**
implicitly repurposed. `value` remains a Boolean column and is compressed with
an independently sampled extension challenge. The field polynomial, modulus,
bounds, coefficient order, bit order, and format string are circuit-ID/exporter
ABI.

### Permutation rows

Let `E_i` be execution-table records and `Q_i` sorted-table records for
`0 <= i < n`. The prover commits to every bit/limb column before challenges.
After both commitments, Fiat--Shamir derives `gamma` and an independent `eta`
from a domain-separated transcript including relation bytes, public-instance
bytes, both commitments, `n`, and the configured bounds. In the R1CS, all ten
base-field coefficients of `gamma` and `eta` are public variables in a fixed
slot order: outputs, inputs, `gamma[0..5]`, then `eta[0..5]`. This keeps the
setup/verifying-key shape static while the verifier recomputes and supplies
the challenge values for each proof. Define a complete-record compression:

```text
R(r) = K(r) + eta * value,

where `eta` is sampled as a full non-base-field `F` element.
Z_0 = 1
Z_(i+1) * (gamma + R(Q_i)) = Z_i * (gamma + R(E_i))
Z_n = 1.
```

Each extension-coordinate equality is materialized as base-field R1CS rows:
the exporter emits explicit `eta_coordinate * value` products, compressed
`gamma + K + eta * value` variables, all 25 coordinate products for each
extension multiplication, and reductions using `X^5 = 1-X^2`.
`PrimeRamR1cs` freezes the static bit/key/sort/scan layout;
`PrimeRamR1cs::permutation_rows` emits public challenge slots, compressed
record variables, `Z` variables, and these extension multiplication/equality
rows without accepting challenge values.

Every sorted-table factor `gamma + R(Q_i)` also receives an extension-inverse
witness and a product-equals-one constraint. A zero denominator therefore
makes the relation unsatisfiable instead of allowing a malicious `Z`
assignment to route around the missing inverse. An honest prover aborts and
the enclosing transcript must re-randomize/recommit according to the reviewed
schedule; no prover-selected resampling counter is part of this relation.
The recurrence, nonzero-denominator rows, plus endpoints proves multiset
equality except with the usual random-compression soundness error. It does
**not** prove that `Q` is sorted, so ordering and RAM rows remain mandatory.

### Sorted-table and latest-value rows

For adjacent sorted records, `PrimeRamR1cs` emits Boolean equality flags for
storage, lane, and each address bit, their AND-prefix, and a first-differing
bit selector. `same_cell_i` equals the final prefix. When it is zero, the
selectors sum to one and constrain that first difference to be `0 -> 1`; this
proves strict lexicographic ordering of the cell tuple without a comparison
operation over unconstrained field values. The concrete exporter uses the
declared most-significant-first comparison order; address bits are stored
little-endian but compared in reverse bit order. Same-cell ordering is then by
the time column, which is bound by the permutation to unique canonical
execution times. A duplicate complete record is therefore impossible.

Let `same_i` mean `Q_i` and `Q_(i-1)` address the same cell, `read_i=1-kind_i`,
and let `V_i` be the running latest value. Define the four Boolean, one-hot
selectors:

```text
FR=(1-same_i)*read_i; LR=same_i*read_i;
FW=(1-same_i)*kind_i; LW=same_i*kind_i.
```

The following gated quadratic rows completely define the scan (with the
missing predecessor at `i=0` treated as `same_0=0` and `V_-1=0`):

```text
FR * value(Q_i)              = 0       # untouched-cell read returns zero
LR * (value(Q_i)-V_(i-1))    = 0       # later read returns latest value
(FR+LR) * (V_i-value(Q_i))   = 0       # a read preserves that value
FW * (V_i-value(Q_i))        = 0       # first write establishes value
LW * (V_i-value(Q_i))        = 0       # later write replaces value.
```

The selector definitions, their Booleanity, and `FR+LR+FW+LW=1` are also
constrained. Every conditional is therefore a finite set of explicit R1CS
rows rather than an ambiguous natural-language branch. Pre-initialization
writes are simply earliest execution records and participate in the same
permutation and scan.

A future prime-field lowering must select and version this exact RAM argument
(or replace the whole subsection with another reviewed argument). No
implementation may pack an unbounded little-endian address or arbitrary `u32`
identifier into KoalaBear and call it injective.

## Planned Spartan-WHIR adapter

`sol-spartan-whir` verifies Spartan-WHIR proofs over the prime KoalaBear field
(`p = 2^31 - 2^24 + 1`) and exposes a `SpartanInstance` consisting of public
field inputs and a witness commitment. The adapter boundary is deliberately
outside this crate:

```text
ModeBRelation + ModeBPublicInstance + witness
  -> versioned R1CS matrices and ordered public field elements
  -> companion Spartan-WHIR prover/exporter
  -> byte-compatible proof consumed by sol-spartan-whir
```

The adapter must freeze matrix-variable ordering (constant, public inputs,
primary witness wires, gate helpers, RAM auxiliaries), constraint ordering,
field encoding, circuit-id binding, and public-input order. The Rust relation
now provides deterministic `ModeBRelation::canonical_bytes()` and
`ModeBPublicInstance::canonical_bytes()` as the version-1 pre-field serializer:
they use domain tags, LE fixed-width scalars, and `u64` length prefixes. They
serialize an R1CS descriptor and public bit statement, **not** a proof and not
a KoalaBear field encoding. The exporter must reject overlarge counts rather
than truncating them and must publish field-element encoding vectors. The `circuit_id`
bytes should be absorbed in the Spartan/WHIR statement transcript (or exposed
as canonical field limbs with a documented injective encoding); it must not be
silently omitted just because a Solidity verifier currently receives only
field elements and a witness commitment.

The Solidity repository is a verifier for proofs generated by its companion
Rust Spartan-WHIR implementation, not a generic Boolar frontend. Therefore
this relation is deliberately upstream of that exporter and no claim of
proof-format compatibility is made until cross-language fixtures verify the
exact matrices, statement limbs, transcript, and proof bytes.

## Required tests before selecting a backend

- Differentially evaluate the same supported circuit/witness with Mode A and
  this relation evaluator.
- Mutate every input/output, gate operand, result, helper, memory address,
  time, value, table entry, and circuit ID; the appropriate verifier must
  reject.
- For the prime-field adapter, test every relation row over the selected
  field, including `Xor`/`Or` at all four Boolean inputs.
- Add golden matrices and statement encodings shared with the companion
  Spartan-WHIR exporter before attempting Solidity proof verification.
