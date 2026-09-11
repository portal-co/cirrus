# @portal-solutions/cirrus-boolar-volar-spec

Concrete-Boolean external-operation bridge for
`@portal-solutions/cirrus-boolar`. Register a codec and a function from
`@portal-solutions/volar-spec-ts` for each named Boolar operation.

```ts
import { createVolarSpecRegistry, unsignedBits } from "@portal-solutions/cirrus-boolar-volar-spec";
import { example_operation } from "@portal-solutions/volar-spec-ts";

const u32 = unsignedBits(32);
const externals = createVolarSpecRegistry([
  { kind: "oracle", name: "example", input: u32, output: u32, invoke: example_operation },
]);
```

Codecs are explicit because Boolar transports only flattened bits and an
operation name; it does not carry a TypeScript function signature. The bridge
invokes direct action bits independently, as the Boolar IR specifies.
