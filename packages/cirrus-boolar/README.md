# @portal-solutions/cirrus-boolar

An eager TypeScript executor for circuit-fused `volar-ir v1` Boolar text.
It mirrors Cirrus's Boolean-context seam: callers supply operations for wire
creation, AND/OR/XOR, and one-bit storage reads and writes. The executor keeps
wires opaque, so the same circuit can run over booleans, garbled labels, or a
symbolic host.

```ts
import { booleanContext, execute, fuse, parseBoolarText } from "@portal-solutions/cirrus-boolar";

const circuit = fuse(parseBoolarText(source));
const outputs = execute(circuit, inputBits, { context: booleanContext });
```

It accepts direct `oracle_bit`, `rng_bit`, and `action_store_bit` statements,
plus the Boolean and storage operations. Legacy handle/projection external
operations deliberately throw: they need a separate aggregate-call contract.
Only one-block `jmp return` circuits are executable, matching Cirrus's fused
`BCircuit` trait implementation.

`pre_init` remains available in the programmatic IR. The upstream v1 text
format does not serialize it, so this package refuses to silently omit it when
emitting text.
