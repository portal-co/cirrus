import assert from "node:assert/strict";

import {
  BoolarError,
  booleanContext,
  emitBoolarText,
  execute,
  fuse,
  parseBoolarText,
} from "../dist/index.js";
import {
  booleanBit,
  createVolarSpecRegistry,
} from "../../cirrus-boolar-volar-spec/dist/index.js";

const bank = (value) => ({ storage: 0, lane: 0, addressBits: 1, value });

const storageProgram = parseBoolarText(`
volar-ir v1
boolar:
begin_block 0
params 2
v2 = storage_read storage=0 lane=0 addr=[v0]
v3 = xor v2 v1
v4 = storage_write storage=0 lane=0 src=v3 addr=[v0]
jmp return args=[v3, v4]
end_block
`);

assert.equal(emitBoolarText(storageProgram), `volar-ir v1
boolar:
begin_block 0
params 2
v2 = storage_read storage=0 lane=0 addr=[v0]
v3 = xor v2 v1
v4 = storage_write storage=0 lane=0 src=v3 addr=[v0]
jmp return args=[v3, v4]
end_block
`);

const initializedStorage = [false, false];
const initializedCircuit = fuse({
  ...storageProgram,
  preInit: [{ storage: 0, lane: 0, address: [false], data: [true] }],
});
assert.deepEqual(
  execute(initializedCircuit, [false, false], { context: booleanContext, banks: [bank(initializedStorage)] }),
  [true, false],
);
assert.equal(initializedStorage[0], true, "storage writes use LSB-first addresses and return dummy zero");
assert.throws(
  () => emitBoolarText({ ...storageProgram, preInit: [{ storage: 0, lane: 0, address: [], data: [true] }] }),
  (error) => error instanceof BoolarError && error.code === "unsupported_statement",
);

const externalProgram = fuse(parseBoolarText(`
volar-ir v1
boolar:
begin_block 0
params 2
v2 = oracle_bit "invert" args=[v0] bit=0 occurrence=4
v3 = rng_bit "nonce" bit=0 occurrence=5
v4 = action_store_bit "write" guard=v1 args=[v2] fallback=v0 storage=0 lane=0 addr=[v0] bit=0 occurrence=6
jmp return args=[v2, v3, v4]
end_block
`));

let actionInvocations = 0;
const registry = createVolarSpecRegistry([
  {
    kind: "oracle",
    name: "invert",
    input: booleanBit(),
    output: booleanBit(),
    // This has the same shape as a function imported from generated.ts.
    invoke: (value) => !value,
  },
  {
    kind: "rng",
    name: "nonce",
    source: { nextBit: (bit, occurrence) => bit === 0 && occurrence === 5n },
  },
  {
    kind: "action",
    name: "write",
    input: booleanBit(),
    output: booleanBit(),
    invoke: (value) => { actionInvocations++; return value; },
  },
]);

const actionStorage = [false, false];
assert.deepEqual(
  execute(externalProgram, [false, true], { context: booleanContext, banks: [bank(actionStorage)], externals: registry }),
  [true, true, false],
);
assert.equal(actionStorage[0], true);
assert.equal(actionInvocations, 1);

const fallbackStorage = [true, false];
execute(externalProgram, [false, false], { context: booleanContext, banks: [bank(fallbackStorage)], externals: registry });
assert.equal(fallbackStorage[0], false, "an unguarded action stores the Boolar fallback");
assert.equal(actionInvocations, 1, "an unguarded action is not invoked");

const legacyProgram = fuse(parseBoolarText(`
volar-ir v1
boolar:
begin_block 0
params 0
v0 = oracle_call "old" args=[] num_bits=1
jmp return args=[v0]
end_block
`));
assert.throws(
  () => execute(legacyProgram, [], { context: booleanContext }),
  (error) => error instanceof BoolarError && error.code === "unsupported_statement",
);

console.log("Boolar interpreter semantics passed");
